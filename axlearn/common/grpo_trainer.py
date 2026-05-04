# Copyright © 2026 Apple Inc.

"""AXLearn SpmdTrainer orchestrating generation steps on twin models natively."""

from typing import Dict, Tuple

import jax
import jax.numpy as jnp

from axlearn.common import utils
from axlearn.common.config import config_class
from axlearn.common.module import child_context
from axlearn.common.module import functional as F
from axlearn.common.trainer import SpmdTrainer, TrainerState
from axlearn.common.update_transformation import ForwardOutputs
from axlearn.common.utils import NestedTensor, Tensor


class GrpoSpmdTrainer(SpmdTrainer):
    """Extended SpmdTrainer subclass evaluating steps tasks concurrently."""

    @config_class
    class Config(SpmdTrainer.Config):
        """Configures GrpoSpmdTrainer."""

        num_generations: int = 4

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

            prompt_prefix = inputs["input_batch"]["prefix"]
            num_generations = inputs.get("num_generations", 4)
            pad_id = inputs.get("pad_id", 0)

            # 1. Generate trajectories via active Actor model
            model_output_collection = F.new_output_collection()
            with child_context(
                "model/actor",
                module=self.model.actor,
                state=actor_params,
                prng_key=inputs["forward_key"],
                output_collection=model_output_collection,
            ):
                sample_outputs = self.model.actor.sample_decode(
                    input_batch={"prefix": prompt_prefix},
                    num_decodes=num_generations,
                )

            generated_sequences = sample_outputs.sequences

            # Reshape matrices across group space size
            _, _, gen_len = generated_sequences.shape
            flat_completions = generated_sequences.reshape(-1, gen_len)

            prompt_len = prompt_prefix.shape[1]
            flat_prompts = jnp.repeat(prompt_prefix, num_generations, axis=0)

            full_sequences = jnp.concatenate([flat_prompts, flat_completions], axis=-1)
            total_len = full_sequences.shape[-1]

            eval_input = {
                "input_ids": full_sequences,
                "positions": jnp.arange(total_len)[None, :],
                "target_labels": full_sequences,
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
            ref_output_collection = F.new_output_collection()
            with child_context(
                "model/reference",
                module=self.model.reference,
                state=ref_params,
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
            completion_mask = (token_positions >= (prompt_len - 1)) & (
                full_sequences[:, 1:] != pad_id
            )

            # Run mock evaluation scoring strategy
            mock_rewards = jnp.ones((flat_completions.shape[0],))

            # Determine surrogate gradients objectives
            learner_outputs = self.learner.grpo_loss(
                actor_logps=actor_seq_logps,
                old_logps=actor_seq_logps,
                ref_logps=ref_seq_logps,
                advantages=self.learner.compute_advantages(mock_rewards),
                completion_mask=completion_mask,
            )

            loss, loss_metrics = learner_outputs
            return ForwardOutputs(
                loss=loss, aux=loss_metrics, output_collection=model_output_collection
            )

        opt_params = self._opt_params(state.model)

        fwd_bwd_outputs, learner_output_collection = F.functional_call(
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
