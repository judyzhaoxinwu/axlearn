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

        # Dynamically build dual meshes by splitting global devices 50/50
        devices = self._mesh.devices.flatten()
        num_devices = len(devices)
        num_trainer_devices = int(num_devices * 0.5)
        trainer_devices = devices[:num_trainer_devices]
        sampler_devices = devices[num_trainer_devices:]

        # Shape trainer mesh as FSDP-friendly 6D mesh shape
        trainer_shape = [1, 1, 1, len(trainer_devices), 1, 1]
        # Shape sampler mesh as Data-parallel-friendly 6D mesh shape
        sampler_shape = [1, len(sampler_devices), 1, 1, 1, 1]

        self.trainer_mesh = jax.sharding.Mesh(
            np.array(trainer_devices).reshape(trainer_shape), self._mesh.axis_names
        )
        self.rollout_mesh = jax.sharding.Mesh(
            np.array(sampler_devices).reshape(sampler_shape), self._mesh.axis_names
        )

        logging.info("GrpoSpmdTrainer initialized with disaggregated architecture:")
        logging.info("  Global Mesh: %s", self._mesh)
        logging.info("  Trainer Mesh: %s", self.trainer_mesh)
        logging.info("  Rollout Mesh: %s", self.rollout_mesh)

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
            # Extract sampler parameters, fallback to actor if disaggregated configuration is not set
            sampler_params = params.get("sampler", actor_params)

            # 1. Extract the true question prompt prefix from input_ids!
            input_ids = inputs["input_batch"]["input_ids"]
            target_labels = inputs["input_batch"]["target_labels"]
            pad_id = inputs.get("pad_id", 0)

            # Retrieve the model's exact configured padding token ID directly from the Actor's decoder config!
            model_pad_id = self.model.actor.decoder.config.pad_token_id

            # Query target_labels < 0 to identify prompt tokens
            prompt_mask = (target_labels < 0) & (input_ids != 0)
            clean_prefix = jnp.where(prompt_mask, input_ids, model_pad_id)

            # Identify prompt length and repeat by generations
            num_generations = self.learner.config.num_generations
            flat_prompt_len = jnp.repeat(jnp.sum(prompt_mask, axis=-1), num_generations, axis=0)

            # 1. Generate trajectories via sampler on rollout_mesh
            model_output_collection = new_output_collection()

            # Select correct sampler decoder module dynamically
            sampler_decoder = (
                self.model.sampler.decoder
                if hasattr(self.model, "sampler")
                else self.model.actor.decoder
            )

            with child_context(
                (
                    "model/sampler/decoder"
                    if hasattr(self.model, "sampler")
                    else "model/actor/decoder"
                ),
                module=sampler_decoder,
                state=sampler_params["decoder"],
                prng_key=inputs["forward_key"],
                output_collection=model_output_collection,
            ):
                with self.rollout_mesh:
                    sample_outputs = sampler_decoder.sample_decode(
                        input_batch={"prefix": clean_prefix},
                        max_sequence_length=512,
                        num_decodes=num_generations,
                    )

            # Extract full sequences
            generated_sequences = sample_outputs.sequences  # Shape [batch, num_decodes, total_len]
            batch_size, _, total_len = generated_sequences.shape
            full_sequences = generated_sequences.reshape(-1, total_len)

            # Extract targets repeated
            flat_targets = jnp.repeat(
                inputs["input_batch"]["target_labels"], num_generations, axis=0
            )

            token_indices = jnp.arange(total_len)[None, :]

            # Zero out prompt prefix
            flat_completions = jnp.where(
                token_indices >= (flat_prompt_len[:, None] + 1), full_sequences, 0
            )

            # 2. Prepare inputs for scoring
            eval_input = {
                "input_ids": full_sequences,
                "positions": jnp.arange(total_len)[None, :],
                "target_labels": full_sequences,
            }

            # 3. Likelihood Calculations under trainer_mesh context
            with self.trainer_mesh:
                # Apply sharding constraint under trainer_mesh context to trigger compiler-generated resharding
                eval_input["input_ids"] = with_sharding_constraint(
                    eval_input["input_ids"], PartitionSpec("fsdp", None)
                )
                eval_input["target_labels"] = eval_input["input_ids"]
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
                frozen_ref_params = jax.tree.map(jax.lax.stop_gradient, ref_params)

                with child_context(
                    "model/reference",
                    module=self.model.reference,
                    state=frozen_ref_params,
                    prng_key=inputs["forward_key"],
                    output_collection=ref_output_collection,
                ):
                    reference_results = self.model.reference.predict(eval_input)

                # Convert network output logits to sequence probabilities
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

                # GSM8K Reward Evaluation via Host CPU Callbacks
                def _parse_gsm8k_number(text: str) -> str:
                    text = text.split("####")[-1].strip() if "####" in text else text
                    match = re.findall(r"[-+]?\d*\.\d+|\d+", text)
                    return match[-1] if match else ""

                def score_gsm8k_rollouts_python(completions_np, targets_np):
                    vocab = self._vocab
                    n = self.config.log_every_n_steps or 100
                    should_log = (self.step % n == 0) or (0 <= self.step <= 5)

                    rewards_list = []
                    for i, (comp_ids, gt_ids) in enumerate(zip(completions_np, targets_np)):
                        comp_list = comp_ids.tolist()
                        gt_list = gt_ids.tolist()

                        clean_comp_ids = [tid for tid in comp_list if tid not in (0, 128004)]
                        clean_gt_ids = [tid for tid in gt_list if tid not in (0, 128004)]

                        for stop_id in (128001, 128009):
                            if stop_id in clean_comp_ids:
                                clean_comp_ids = clean_comp_ids[: clean_comp_ids.index(stop_id)]
                            if stop_id in clean_gt_ids:
                                clean_gt_ids = clean_gt_ids[: clean_gt_ids.index(stop_id)]

                        comp_str = vocab.decode(clean_comp_ids)
                        gt_str = vocab.decode(clean_gt_ids)

                        def _extract_boxed_or_full(text: str) -> str:
                            if "\\boxed{" in text:
                                try:
                                    return text.split("\\boxed{")[-1].split("}")[0].strip()
                                except Exception:
                                    pass
                            return text

                        extracted_gen = _parse_gsm8k_number(_extract_boxed_or_full(comp_str))
                        extracted_gt = _parse_gsm8k_number(_extract_boxed_or_full(gt_str))

                        reward = (
                            1.0 if extracted_gen == extracted_gt and extracted_gen != "" else 0.0
                        )
                        rewards_list.append(reward)

                        if should_log and i < 2:
                            logging.info("[eshenlog] Step %d | Sample %d", self.step, i + 1)
                            logging.info(
                                "[eshenlog] Raw comp_ids slice (first 30): %s", str(comp_list[:30])
                            )
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

                flat_targets = jnp.repeat(
                    inputs["input_batch"]["target_labels"], num_generations, axis=0
                )
                clean_targets = jnp.where(flat_targets >= 0, flat_targets, pad_id)

                raw_rewards = jax.pure_callback(
                    score_gsm8k_rollouts_python,
                    jax.ShapeDtypeStruct((flat_completions.shape[0],), jnp.float32),
                    flat_completions,
                    clean_targets,
                )

                real_rewards = jax.lax.stop_gradient(raw_rewards)
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
                loss_metrics["mean_reward"] = jnp.mean(real_rewards)
                loss_metrics["std_advantage"] = jnp.std(
                    self.learner.compute_advantages(real_rewards)
                )

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

        # 4. Disaggregated Weight Sync: Reshard updated actor weights to the sampler parameters on rollout_mesh
        if "sampler" in updated_model_params:
            sampler_sharding = self.trainer_state_partition_specs.model["sampler"]
            synchronized_sampler_params = jax.tree.map(
                lambda actor_param, sharding: jax.device_put(actor_param, sharding),
                updated_model_params["actor"],
                sampler_sharding,
            )
            # Update the model state PyTree
            updated_model_params = {
                **updated_model_params,
                "sampler": synchronized_sampler_params,
            }

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
