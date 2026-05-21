# Copyright © 2026 Apple Inc.

"""AXLearn SpmdTrainer orchestrating generation steps on twin models natively."""

import os
import re
from typing import Dict, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from absl import logging
from jax.experimental.pjit import pjit
from jax.sharding import PartitionSpec

from axlearn.common import file_system as fs
from axlearn.common import measurement, utils
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
        reward_type: str = "dummy"  # One of "gsm8k", "dummy", "exact_match"

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
        logging.info("  Reward Type: %s", cfg.reward_type)

    def _score_gsm8k_rollouts_python(self, completions_np, targets_np) -> np.ndarray:
        def _parse_gsm8k_number(text: str) -> str:
            text = text.split("####")[-1].strip() if "####" in text else text
            match = re.findall(r"[-+]?\d*\.\d+|\d+", text)
            return match[-1] if match else ""

        def _extract_boxed_or_full(text: str) -> str:
            if "\\boxed{" in text:
                try:
                    return text.split("\\boxed{")[-1].split("}")[0].strip()
                except Exception:
                    pass
            return text

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

            extracted_gen = _parse_gsm8k_number(_extract_boxed_or_full(comp_str))
            extracted_gt = _parse_gsm8k_number(_extract_boxed_or_full(gt_str))

            reward = 1.0 if extracted_gen == extracted_gt and extracted_gen != "" else 0.0
            rewards_list.append(reward)

            if should_log and i < 2:
                logging.info("[eshenlog] Step %d | Sample %d", self.step, i + 1)
                logging.info("[eshenlog] Raw comp_ids slice (first 30): %s", str(comp_list[:30]))
                logging.info("[eshenlog] Generated: %s", comp_str.replace("\n", " ").strip())
                logging.info("[eshenlog] Ground Truth: %s", gt_str.replace("\n", " ").strip())
                logging.info(
                    "[eshenlog] Extracted Gen: '%s' | GT: '%s' | Reward: %.1f",
                    extracted_gen,
                    extracted_gt,
                    reward,
                )
                logging.info("[eshenlog] %s", "=" * 60)

        return np.array(rewards_list, dtype=np.float32)

    def _prepare_training(self, prng_key: Tensor) -> bool:
        self._maybe_record_event(measurement.Event.START_TRAINING_PREPARATION)
        cfg = self.config

        # Restore latest checkpoint
        self.restore_checkpoint(restore_step=None)

        if self.step is None:
            self.init(prng_key)
            self._step = 0
            self.save_checkpoint(self._run_eval())

        model_analysis = self._log_trainer_state_stats()

        # Log trainer state tree
        if not self.step and jax.process_index() == 0:
            with fs.open(os.path.join(cfg.dir, "trainer_state_tree.txt"), "w") as f:
                f.write(str(jax.tree_util.tree_structure(self._trainer_state)))

            with fs.open(os.path.join(cfg.dir, "model_analysis.txt"), "w") as f:
                f.write(model_analysis)

        # Log config
        self.summary_writer.log_config(cfg, step=self.step)

        if self.step >= cfg.max_step:
            self._step_log("Already reached max_step=%s. Stopping", cfg.max_step)
            return False

        # Compile rollout and update steps
        self._jit_rollout_step = pjit(
            self._rollout_step,
            in_shardings=(
                self._trainer_state_partition_specs.model,
                None,
                self._train_step_input_partition_specs(),
            ),
            out_shardings=(None, None, None, None),
        )

        self._jit_update_step = pjit(
            self._update_step,
            in_shardings=(
                self._trainer_state_partition_specs,
                self._train_step_input_partition_specs(),
                None,
                None,
                None,
                None,
                None,
                None,
            ),
            out_shardings=(
                self._trainer_state_partition_specs,
                dict(
                    summaries=None,
                    loss=None,
                    aux=None,
                ),
            ),
            donate_argnums=(0,),
        )

        self._maybe_record_event(measurement.Event.END_TRAINING_PREPARATION)
        return True

    def _run_step(
        self, input_batch: NestedTensor, *, force_run_evals: Optional[set[str]] = None
    ) -> NestedTensor:
        logging.log_first_n(logging.INFO, "global_input_batch=%s", 3, utils.shapes(input_batch))

        with jax.profiler.StepTraceAnnotation("train", step_num=self.step):
            # 1. Split key on host CPU
            keys = jax.random.split(self.trainer_state.prng_key, 5)
            new_prng_key = keys[0]
            param_noise_key = keys[1]
            rollout_key = keys[2]
            forward_key = keys[3]
            learner_key = keys[4]

            # 2. Execute Rollout on TPU
            full_seqs, flat_completions, flat_prompt_len, clean_targets = self._jit_rollout_step(
                self.trainer_state.model, rollout_key, input_batch
            )

            # 3. Bring arrays to host CPU memory
            flat_completions_np = np.asarray(flat_completions)
            logging.info(
                "[eshenlog] Total number of generated answers in this batch: %d",
                flat_completions_np.shape[0],
            )
            clean_targets_np = np.asarray(clean_targets)

            # 4. Evaluate rewards in host Python
            if self.config.reward_type == "gsm8k":
                raw_rewards = self._score_gsm8k_rollouts_python(
                    flat_completions_np, clean_targets_np
                )
            elif self.config.reward_type == "dummy":
                raw_rewards = np.ones((flat_completions_np.shape[0],), dtype=np.float32)
            elif self.config.reward_type == "exact_match":
                len_diff = flat_completions_np.shape[1] - clean_targets_np.shape[1]
                if len_diff > 0:
                    padded_clean_targets = np.pad(
                        clean_targets_np,
                        ((0, 0), (0, len_diff)),
                        mode="constant",
                        constant_values=0,
                    )
                    padded_completions = flat_completions_np
                elif len_diff < 0:
                    padded_completions = np.pad(
                        flat_completions_np,
                        ((0, 0), (0, -len_diff)),
                        mode="constant",
                        constant_values=0,
                    )
                    padded_clean_targets = clean_targets_np
                else:
                    padded_clean_targets = clean_targets_np
                    padded_completions = flat_completions_np
                raw_rewards = np.all(padded_completions == padded_clean_targets, axis=-1).astype(
                    np.float32
                )
            else:
                raise ValueError(f"Unsupported reward_type: {self.config.reward_type}")

            # 5. Calculate normalized advantages in host NumPy
            num_generations = self.config.num_generations
            batch_size = raw_rewards.shape[0] // num_generations
            grouped = raw_rewards.reshape(batch_size, num_generations)
            means = np.mean(grouped, axis=-1, keepdims=True)
            stds = np.std(grouped, axis=-1, ddof=1, keepdims=True)
            normalized = (grouped - means) / (stds + 1e-4)
            advantages_np = normalized.flatten()

            # 6. Convert to JAX device array
            advantages_jax = jnp.array(advantages_np)

            # 7. Execute JIT-compiled Update on TPU
            self._trainer_state, outputs = self._jit_update_step(
                self.trainer_state,
                input_batch,
                full_seqs,
                flat_prompt_len,
                advantages_jax,
                param_noise_key,
                forward_key,
                learner_key,
            )

            # Append mean_reward to metrics
            outputs["aux"]["mean_reward"] = jnp.mean(jnp.array(raw_rewards))

            # Log weight synchronization verification
            n = self._config.log_every_n_steps or 100
            should_log = (self.step % n == 0) or (0 <= self.step <= 5)
            if should_log and "sampler" in self._trainer_state.model:
                actor_params = self._trainer_state.model["actor"]
                sampler_params = self._trainer_state.model["sampler"]
                flat_actor, _ = jax.tree_util.tree_flatten(actor_params)
                flat_sampler, _ = jax.tree_util.tree_flatten(sampler_params)
                if flat_actor:
                    actor_norm = np.asarray(jnp.sqrt(jnp.sum(jnp.square(flat_actor[0]))))
                    sampler_norm = np.asarray(jnp.sqrt(jnp.sum(jnp.square(flat_sampler[0]))))
                    logging.info(
                        "[eshenlog] Disaggregated Weight Sync Verification | Actor Norm: %.6f | Sampler Norm: %.6f",
                        actor_norm,
                        sampler_norm,
                    )

            # Explicitly advance the state PRNG key
            self._trainer_state = TrainerState(
                prng_key=new_prng_key,
                model=self._trainer_state.model,
                learner=self._trainer_state.learner,
            )

        n = self._config.log_every_n_steps or 100
        if self.step % n == 0 or 0 <= self.step <= 5:
            self._step_log(
                "loss=%s aux=%s",
                outputs["loss"],
                jax.tree.map(lambda x: x.item() if x.ndim == 0 else f"T{x.shape}", outputs["aux"]),
            )

        self.summary_writer(self.step, {"loss": outputs["loss"], **outputs["summaries"]})
        evaler_summaries = self._run_eval(
            train_summaries=outputs["summaries"], force_runs=force_run_evals
        )
        self.save_checkpoint(evaler_summaries=evaler_summaries)

        return_dict = {"loss": outputs["loss"], "aux": outputs["aux"]}
        if force_run_evals:
            return_dict["evaler_summaries"] = evaler_summaries

        return return_dict

    def _rollout_step(
        self,
        model_params: NestedTensor,
        rollout_key: Tensor,
        input_batch: dict,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Runs rollout generation on the rollout mesh."""

        def train_cast(in_tree):
            per_param_train_dtype = self._per_param_train_dtype(in_tree)
            return utils.cast_floats_per_param(in_tree, per_param_train_dtype)

        input_batch = train_cast(input_batch)
        input_batch = self.input.dispatch_global_batch(input_batch)

        params = train_cast(model_params)
        actor_params = params["actor"]
        sampler_params = params.get("sampler", actor_params)

        input_ids = input_batch["input_ids"]
        target_labels = input_batch["target_labels"]

        model_pad_id = self.model.actor.decoder.config.pad_token_id

        prompt_mask = (target_labels < 0) & (input_ids != 0)
        clean_prefix = jnp.where(prompt_mask, input_ids, model_pad_id)

        num_generations = self.config.num_generations
        flat_prompt_len = jnp.repeat(jnp.sum(prompt_mask, axis=-1), num_generations, axis=0)

        sampler_decoder = (
            self.model.sampler.decoder
            if hasattr(self.model, "sampler")
            else self.model.actor.decoder
        )

        with self.rollout_mesh:
            sample_outputs, _ = F(
                sampler_decoder,
                method="sample_decode",
                state=sampler_params["decoder"],
                prng_key=rollout_key,
                is_training=False,
                inputs=dict(
                    input_batch={"prefix": clean_prefix},
                    max_sequence_length=512,
                    num_decodes=num_generations,
                ),
            )

        generated_sequences = sample_outputs.sequences
        batch_size, _, total_len = generated_sequences.shape
        full_sequences = generated_sequences.reshape(-1, total_len)

        token_indices = jnp.arange(total_len)[None, :]
        flat_completions = jnp.where(
            token_indices >= (flat_prompt_len[:, None] + 1), full_sequences, 0
        )

        flat_targets = jnp.repeat(input_batch["target_labels"], num_generations, axis=0)
        pad_id = 0
        clean_targets = jnp.where(flat_targets >= 0, flat_targets, pad_id)

        return full_sequences, flat_completions, flat_prompt_len, clean_targets

    def _update_step(
        self,
        state: TrainerState,
        input_batch: dict,
        full_sequences: Tensor,
        flat_prompt_len: Tensor,
        advantages: Tensor,
        param_noise_key: Tensor,
        forward_key: Tensor,
        learner_key: Tensor,
    ) -> Tuple[TrainerState, Dict]:
        """Applies forward pass and updates model parameters on the trainer mesh."""

        def train_cast(in_tree):
            per_param_train_dtype = self._per_param_train_dtype(in_tree)
            return utils.cast_floats_per_param(in_tree, per_param_train_dtype)

        input_batch = train_cast(input_batch)
        input_batch = self.input.dispatch_global_batch(input_batch)

        def _forward(*, inputs: NestedTensor, model_params: NestedTensor) -> ForwardOutputs:
            params = train_cast(model_params)
            actor_params = params["actor"]
            ref_params = params["reference"]

            with self.trainer_mesh:
                eval_input = {
                    "input_ids": with_sharding_constraint(
                        inputs["full_sequences"], PartitionSpec("fsdp", None)
                    ),
                    "positions": jnp.arange(inputs["full_sequences"].shape[1])[None, :],
                    "target_labels": with_sharding_constraint(
                        inputs["full_sequences"], PartitionSpec("fsdp", None)
                    ),
                }

                model_output_collection = new_output_collection()
                with child_context(
                    "model/actor",
                    module=self.model.actor,
                    state=actor_params,
                    prng_key=inputs["forward_key"],
                    output_collection=model_output_collection,
                ):
                    actor_results = self.model.actor.predict(eval_input)

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

                def extract_sequence_logps(logits: Tensor, token_ids: Tensor) -> Tensor:
                    log_probs = jax.nn.log_softmax(logits, axis=-1)
                    target_ids = token_ids[:, 1:]
                    gathered = jnp.take_along_axis(
                        log_probs[:, :-1, :], jnp.expand_dims(target_ids, axis=-1), axis=-1
                    ).squeeze(-1)
                    return gathered

                actor_seq_logps = extract_sequence_logps(
                    actor_results["logits"], inputs["full_sequences"]
                )
                ref_seq_logps = extract_sequence_logps(
                    reference_results["logits"], inputs["full_sequences"]
                )

                pad_id = 0
                token_positions = jnp.arange(inputs["full_sequences"].shape[1] - 1)[None, :]
                completion_mask = (token_positions >= (inputs["flat_prompt_len"][:, None] - 1)) & (
                    inputs["full_sequences"][:, 1:] != pad_id
                )

                old_seq_logps = jax.lax.stop_gradient(actor_seq_logps)

                learner_outputs = self.learner.grpo_loss(
                    actor_logps=actor_seq_logps,
                    old_logps=old_seq_logps,
                    ref_logps=ref_seq_logps,
                    advantages=inputs["advantages"],
                    completion_mask=completion_mask,
                )

                loss, loss_metrics = learner_outputs
                loss_metrics["std_advantage"] = jnp.std(inputs["advantages"])

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
                    full_sequences=full_sequences,
                    flat_prompt_len=flat_prompt_len,
                    advantages=advantages,
                    forward_key=forward_key,
                    param_noise_key=param_noise_key,
                ),
            ),
        )

        forward_outputs: ForwardOutputs = fwd_bwd_outputs.forward_outputs
        updated_model_params = fwd_bwd_outputs.backward_outputs.updated_params

        # Disaggregated Weight Sync: Reshard updated actor weights to the sampler parameters on rollout_mesh
        if "sampler" in updated_model_params:
            sampler_sharding = self.trainer_state_partition_specs.model["sampler"]
            synchronized_sampler_params = jax.tree.map(
                lambda actor_param, sharding: jax.device_put(actor_param, sharding),
                updated_model_params["actor"],
                sampler_sharding,
            )
            updated_model_params = {
                **updated_model_params,
                "sampler": synchronized_sampler_params,
            }

        updated_state = TrainerState(
            prng_key=state.prng_key,
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
