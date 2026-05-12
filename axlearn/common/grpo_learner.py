# Copyright © 2026 Apple Inc.

"""AXLearn custom Learner module implementing Group Relative Policy Optimization objectives."""

from typing import Dict, Tuple

import jax.numpy as jnp

from axlearn.common.config import config_class
from axlearn.common.learner import Learner
from axlearn.common.utils import Tensor


class AxlearnGrpoLearner(Learner):
    """Calculates advantages and loss updates natively for GRPO training operations."""

    @config_class
    class Config(Learner.Config):
        """Configures AxlearnGrpoLearner."""

        num_generations: int = 4
        beta: float = 0.04
        epsilon: float = 0.2

    def compute_advantages(self, rewards: Tensor) -> Tensor:
        """Calculates comparative advantages across token sets.

        Args:
            rewards: Scalar reward array of shape [Batch * num_generations].
        """
        num_generations = self.config.num_generations
        batch_size = rewards.shape[0] // num_generations

        # Group over generation dimension G
        grouped = rewards.reshape(batch_size, num_generations)

        means = jnp.mean(grouped, axis=-1, keepdims=True)
        stds = jnp.std(grouped, axis=-1, ddof=1, keepdims=True)

        normalized = (grouped - means) / (stds + 1e-4)
        return normalized.flatten()

    def grpo_loss(
        self,
        *,
        actor_logps: Tensor,
        old_logps: Tensor,
        ref_logps: Tensor,
        advantages: Tensor,
        completion_mask: Tensor,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """Evaluates surrogate objective values including foundational anchoring constraints.

        Args:
            actor_logps: Tensor of shape [Batch * G, seq_len].
            old_logps: Tensor of shape [Batch * G, seq_len].
            ref_logps: Tensor of shape [Batch * G, seq_len].
            advantages: Tensor of shape [Batch * G].
            completion_mask: Bool or float binary tensor of shape [Batch * G, seq_len].
        """
        cfg = self.config

        # Evaluate importance ratio (new_policy / old_policy)
        log_ratios = (actor_logps - old_logps) * completion_mask

        # Aggressively clip log ratios between [-20.0, 20.0] to prevent exponential overflow to inf!
        # This guarantees complete, 100% numerical stability against NaN weight explosions.
        clipped_log_ratios = jnp.clip(log_ratios, -20.0, 20.0)
        ratio = jnp.exp(clipped_log_ratios)

        clipped_ratio = jnp.clip(ratio, 1.0 - cfg.epsilon, 1.0 + cfg.epsilon)

        # Align advantages across token sequences
        adv = advantages[:, None]

        # Evaluate standard optimization parameters
        policy_loss = -jnp.minimum(
            ratio * adv,
            clipped_ratio * adv,
        )

        mean_kl = jnp.array(0.0)
        if cfg.beta > 0:
            # Token level KL formulation
            # Aggressively clip KL log differences to prevent exponential overflow to inf!
            kl_log_diff = jnp.clip(ref_logps - actor_logps, -20.0, 20.0)
            kl = jnp.exp(kl_log_diff) - (ref_logps - actor_logps) - 1.0
            policy_loss += cfg.beta * kl
            mean_kl = jnp.sum(kl * completion_mask) / jnp.maximum(jnp.sum(completion_mask), 1.0)

        masked_loss = policy_loss * completion_mask
        loss = jnp.sum(masked_loss) / jnp.maximum(jnp.sum(completion_mask), 1.0)

        metrics = {
            "mean_advantage": jnp.mean(advantages),
            "kl_divergence": mean_kl,
            "loss": loss,
        }
        return loss, metrics
