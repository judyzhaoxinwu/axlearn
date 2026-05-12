# Copyright © 2026 Apple Inc.

"""AXLearn SpmdTrainer orchestrating generation steps on twin models natively."""

import re
from typing import Dict, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from absl import logging
from jax.sharding import PartitionSpec

from axlearn.common import utils
from axlearn.common.config import REQUIRED, InstantiableConfig, Required, config_class
from axlearn.common.module import Module, child_context
from axlearn.common.module import functional as F
from axlearn.common.module import new_output_collection
from axlearn.common.trainer import SpmdTrainer, TrainerState
from axlearn.common.update_transformation import ForwardOutputs
from axlearn.common.utils import NestedTensor, Tensor, with_sharding_constraint


class GrpoSpmdTrainer(SpmdTrainer):
    """Extended SpmdTrainer subclass evaluating steps tasks concurrently."""

    @config_class
    class Config(SpmdTrainer.Config):
        """Configures GrpoSpmdTrainer."""

        num_generations: int = 4
        vocab: Required[InstantiableConfig] = REQUIRED  # SentencePiece vocabulary configuration!

    def __init__(self, cfg: Config, *, parent: Module):
        super().__init__(cfg, parent=parent)
        # Instantiate the configured vocabulary directly at startup!
        self._vocab = cfg.vocab.instantiate()

    def _train_step(
        self,
        state: TrainerState,
        input_batch: dict,
    ) -> Tuple[TrainerState, Dict]:

        def train_cast(in_tree):
            per_param_train_dtype = self._per_param_train_dtype(in_tree)
            return utils.cast_floats_per_param(in_tree, per_param_train_dtype)

        input_batch = train_cast(input_batch)
        input_batch = self.input.dispatch_global_batch(input_batch)

        new_prng_key, param_noise_key, forward_key, learner_key = jax.random.split(
            state.prng_key, 4
        )

        def _forward(*, inputs: NestedTensor, model_params: NestedTensor) -> ForwardOutputs:
            params = train_cast(model_params)

            actor_params = params["actor"]
            ref_params = params["reference"]

            # 1. Extract the true question prompt prefix from input_ids!
            # Autoregressive input_ids carries both [BOS] + Question + Ground-truth answer concatenated.
            # The target_labels masks out the prompt prefix tokens using negative numbers (< 0, e.g., -1).
            input_ids = inputs["input_batch"]["input_ids"]
            target_labels = inputs["input_batch"]["target_labels"]
            pad_id = inputs.get("pad_id", 0)

            # Retrieve the model's exact configured padding token ID directly from the Actor's decoder config!
            # This keeps the code perfectly dynamic, clean, and completely free of hardcoded magic numbers.
            model_pad_id = self.model.actor.decoder.config.pad_token_id

            # Query target_labels < 0 to identify prompt tokens, but strictly exclude seqio's trailing
            # pad tokens (where input_ids is 0)! This is because AXLearn's map_targets_out_of_class maps
            # trailing pad labels from 0 to -1, which would otherwise corrupt our prefill boundaries.
            prompt_mask = (target_labels < 0) & (input_ids != 0)
            clean_prefix = jnp.where(prompt_mask, input_ids, model_pad_id)

            # 2. Identify the dynamic actual question prompt length for each batch sequence!
            # We repeat this along the batch dimension by the number of generations
            # to perfectly align with full_sequences shape!
            num_generations = self.learner.config.num_generations
            flat_prompt_len = jnp.repeat(jnp.sum(prompt_mask, axis=-1), num_generations, axis=0)

            # 1. Generate trajectories via active Actor model
            model_output_collection = new_output_collection()

            with child_context(
                "model/actor/decoder",  # Enter the decoder child context namespace!
                module=self.model.actor.decoder,
                state=actor_params["decoder"],  # Pass decoder sub-weights cleanly!
                prng_key=inputs["forward_key"],
                output_collection=model_output_collection,
            ):
                # Call sample_decode directly on the Decoder, passing the 512-length clean_prefix!
                # The decoder will automatically, dynamically infer the correct starting step (time_step)
                # using infer_initial_time_step since the prefix is padded with 128004!
                sample_outputs = self.model.actor.decoder.sample_decode(
                    input_batch={"prefix": clean_prefix},
                    max_sequence_length=512,  # Forces full generation length!
                    num_decodes=num_generations,
                )

            # Extract full sequences (already includes prefix!) and reshape to 2D
            generated_sequences = sample_outputs.sequences  # Shape [batch, num_decodes, total_len]
            batch_size, _, total_len = generated_sequences.shape
            full_sequences = generated_sequences.reshape(
                -1, total_len
            )  # Shape [batch * num_decodes, total_len]

            # Extract target sequence labels and repeat them along the batch dimension
            flat_targets = jnp.repeat(
                inputs["input_batch"]["target_labels"], num_generations, axis=0
            )

            # Create a token position index tensor matching full_sequences total length
            token_indices = jnp.arange(total_len)[None, :]

            # Dynamically zero out only the prompt prefix positions, keeping ALL active generated tokens
            # all the way to the EOS or end of sequence index, completely independent of the ground-truth answer length!
            # We shift flat_prompt_len by +1 because seqio's make_autoregressive_inputs intentionally leaves
            # the very last prompt token unmasked (as a positive target label) to predict the first answer token.
            flat_completions = jnp.where(
                token_indices >= (flat_prompt_len[:, None] + 1), full_sequences, 0
            )

            # 2. Partition the concatenated sequences along the FSDP axis!
            # Dimension 0 (batch * generations = 16) is partitioned 16-ways across FSDP chips,
            # while Dimension 1 (sequence length 1025) remains unpartitioned (None).
            batch_sharding = PartitionSpec("fsdp", None)
            sharded_sequences = with_sharding_constraint(full_sequences, batch_sharding)

            eval_input = {
                "input_ids": sharded_sequences,
                "positions": jnp.arange(total_len)[None, :],
                "target_labels": sharded_sequences,
            }

            # 2. Likelihood Calculations: Execute forward operations across models
            # Process Actor results
            with child_context(
                "model/actor",
                module=self.model.actor,
                state=actor_params,
                prng_key=inputs["forward_key"],
                output_collection=model_output_collection,
            ):
                actor_results = self.model.actor.predict(eval_input)

            # Process Reference results
            ref_output_collection = new_output_collection()

            # Wrap Reference parameters in stop_gradient to indicate they are completely non-differentiable!
            # This prevents JAX from allocating any intermediate activations for the Reference model during backprop!
            frozen_ref_params = jax.tree.map(jax.lax.stop_gradient, ref_params)

            with child_context(
                "model/reference",
                module=self.model.reference,
                state=frozen_ref_params,  # Passes the completely frozen parameters!
                prng_key=inputs["forward_key"],
                output_collection=ref_output_collection,
            ):
                reference_results = self.model.reference.predict(eval_input)

            # 3. Convert network output logits to sequence probabilities
            def extract_sequence_logps(logits: Tensor, token_ids: Tensor) -> Tensor:
                log_probs = jax.nn.log_softmax(logits, axis=-1)
                target_ids = token_ids[:, 1:]

                gathered = jnp.take_along_axis(
                    log_probs[:, :-1, :], jnp.expand_dims(target_ids, axis=-1), axis=-1
                ).squeeze(-1)
                return gathered

            actor_seq_logps = extract_sequence_logps(actor_results["logits"], full_sequences)
            ref_seq_logps = extract_sequence_logps(reference_results["logits"], full_sequences)

            # Create target sequence extraction token masks
            token_positions = jnp.arange(total_len - 1)[None, :]
            completion_mask = (token_positions >= (flat_prompt_len[:, None] - 1)) & (
                full_sequences[:, 1:] != pad_id
            )

            # 4. JAX-Safe GSM8K Reward Evaluation via Host CPU Callbacks
            # Since JIT cannot compile string operations, we execute parsing on
            # Host CPU via jax.pure_callback!
            def _parse_gsm8k_number(text: str) -> str:
                # Extracts the final numeric answer usually located after '####'
                # or at the end of text.
                text = text.split("####")[-1].strip() if "####" in text else text
                match = re.findall(r"[-+]?\d*\.\d+|\d+", text)
                return match[-1] if match else ""

            def score_gsm8k_rollouts_python(completions_np, targets_np):

                # Executed strictly on Host CPU using regular Python string operations!
                # Retrieve vocabulary directly from the trainer's instantiated vocab!
                vocab = self._vocab

                # Only print diagnostic logs for the first 2 samples of the batch at logging steps!
                n = self.config.log_every_n_steps or 100
                should_log = (self.step % n == 0) or (0 <= self.step <= 5)

                rewards_list = []
                for i, (comp_ids, gt_ids) in enumerate(zip(completions_np, targets_np)):
                    # Cleanly filter out both data-loader pad ID (0) and model pad ID (128004)
                    # from the lists before passing to sentencepiece vocabulary decoder!
                    comp_list = comp_ids.tolist()
                    gt_list = gt_ids.tolist()

                    # Strictly filter out seqio pad (0) and model pad (128004)
                    clean_comp_ids = [tid for tid in comp_list if tid not in (0, 128004)]
                    clean_gt_ids = [tid for tid in gt_list if tid not in (0, 128004)]

                    # Truncate the lists immediately at the very first EOS (128001) or EOT (128009) token!
                    # This perfectly, cleanly cuts off any garbage repetition loops or token degradation.
                    for stop_id in (128001, 128009):
                        if stop_id in clean_comp_ids:
                            clean_comp_ids = clean_comp_ids[: clean_comp_ids.index(stop_id)]
                        if stop_id in clean_gt_ids:
                            clean_gt_ids = clean_gt_ids[: clean_gt_ids.index(stop_id)]

                    comp_str = vocab.decode(clean_comp_ids)
                    gt_str = vocab.decode(clean_gt_ids)

                    # Parse the final answers. If the model outputs standard LaTeX boxed notation (\boxed{...}),
                    # we isolate the content inside the box directly to completely bypass any trailing repetition clutter!
                    def _extract_boxed_or_full(text: str) -> str:
                        if "\\boxed{" in text:
                            try:
                                return text.split("\\boxed{")[-1].split("}")[0].strip()
                            except Exception:
                                pass
                        return text

                    extracted_gen = _parse_gsm8k_number(_extract_boxed_or_full(comp_str))
                    extracted_gt = _parse_gsm8k_number(_extract_boxed_or_full(gt_str))

                    reward = 1.0 if extracted_gen == extracted_gt and extracted_gen != "" else 0.0
                    rewards_list.append(reward)

                    if should_log and i < 2:
                        logging.info("[eshenlog] Step %d | Sample %d", self.step, i + 1)
                        # Rich systems diagnostics showing the raw layout of incoming TPU tensors!
                        logging.info(
                            "[eshenlog] Raw comp_ids slice (first 30): %s", str(comp_list[:30])
                        )
                        logging.info(
                            "[eshenlog] Raw comp_ids unique values: %s",
                            str(sorted(list(set(comp_list)))),
                        )
                        logging.info(
                            "[eshenlog] Clean comp_ids count: %d | Raw count: %d",
                            len(clean_comp_ids),
                            len(comp_list),
                        )

                        # Eliminates interior newlines to satisfy GKE multi-line print specifications
                        logging.info(
                            "[eshenlog] Generated: %s", comp_str.replace("\n", " ").strip()
                        )
                        logging.info(
                            "[eshenlog] Ground Truth: %s", gt_str.replace("\n", " ").strip()
                        )
                        logging.info(
                            "[eshenlog] Extracted Gen: '%s' | GT: '%s' | Reward: %.1f",
                            extracted_gen,
                            extracted_gt,
                            reward,
                        )
                        logging.info("[eshenlog] %s", "=" * 60)

                return np.array(rewards_list, dtype=np.float32)

            # Repeat and clean ground-truth target labels to align with siblings batch dimensions,
            # replacing negative ignore mask values (< 0) with the clean pad_id [0] before string decoding!
            flat_targets = jnp.repeat(
                inputs["input_batch"]["target_labels"], num_generations, axis=0
            )
            clean_targets = jnp.where(flat_targets >= 0, flat_targets, pad_id)

            # Invoke Python string parser on Host CPU dynamically from the compiled TPU graph!
            raw_rewards = jax.pure_callback(
                score_gsm8k_rollouts_python,
                jax.ShapeDtypeStruct((flat_completions.shape[0],), jnp.float32),
                flat_completions,
                clean_targets,  # Passes the cleaned, perfect ground-truth target token IDs!
            )

            # Wrap in stop_gradient to indicate rewards are non-differentiable (zero gradients)
            real_rewards = jax.lax.stop_gradient(raw_rewards)

            # Freeze the actor's current log probabilities to act as the old policy baseline.
            # This ensures the PPO importance ratio is differentiable ONLY with respect to the new actor_logps!
            # Without this, JAX would differentiate through both numerator and denominator, mathematically zeroing out the gradients!
            old_seq_logps = jax.lax.stop_gradient(actor_seq_logps)

            # Determine surrogate gradients objectives
            learner_outputs = self.learner.grpo_loss(
                actor_logps=actor_seq_logps,
                old_logps=old_seq_logps,
                ref_logps=ref_seq_logps,
                advantages=self.learner.compute_advantages(real_rewards),
                completion_mask=completion_mask,
            )

            loss, loss_metrics = learner_outputs

            # Inject active and highly meaningful metrics directly into the auxiliary logger!
            # 'mean_reward' shows absolute GSM8K math accuracy (percentage of correct rollouts).
            # 'std_advantage' shows the strength/variance of your relative learning signal!
            loss_metrics["mean_reward"] = jnp.mean(real_rewards)
            loss_metrics["std_advantage"] = jnp.std(self.learner.compute_advantages(real_rewards))

            return ForwardOutputs(
                loss=loss, aux=loss_metrics, output_collection=model_output_collection
            )

        opt_params = self._opt_params(state.model)

        fwd_bwd_outputs, learner_output_collection = F(
            self.learner,
            method="forward_and_backward",
            state=state.learner,
            is_training=True,
            prng_key=learner_key,
            inputs=dict(
                fn=_forward,
                opt_params=opt_params,
                inputs=dict(
                    input_batch=input_batch,
                    forward_key=forward_key,
                    param_noise_key=param_noise_key,
                    num_generations=self.config.num_generations,
                ),
            ),
        )

        forward_outputs: ForwardOutputs = fwd_bwd_outputs.forward_outputs
        updated_model_params = fwd_bwd_outputs.backward_outputs.updated_params

        updated_state = TrainerState(
            prng_key=new_prng_key,
            model=updated_model_params,
            learner=learner_output_collection.state_updates,
        )

        summaries = dict(
            model=forward_outputs.output_collection.summaries,
            learner=learner_output_collection.summaries,
        )

        return updated_state, dict(
            summaries=summaries,
            loss=forward_outputs.loss,
            aux=forward_outputs.aux,
        )
