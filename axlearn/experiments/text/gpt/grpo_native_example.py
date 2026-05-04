# Copyright © 2026 Apple Inc.

"""Driver configurations module utilizing tfds_input over local/remote sources."""

from typing import Dict

from axlearn.common.config import InstantiableConfig, TrainerConfigFn, config_for_function
from axlearn.common.grpo_learner import AxlearnGrpoLearner
from axlearn.common.grpo_model import GrpoModel
from axlearn.common.grpo_trainer import GrpoSpmdTrainer
from axlearn.experiments.text.common import vocab
from axlearn.experiments.text.gpt import fuji
from axlearn.experiments.text.gpt.common import tfds_input


def _vocab_cfg(vocab_size: int) -> InstantiableConfig:
    """Constructs vocabulary configuration instance based on size."""
    if vocab_size == 32 * 1024:
        return config_for_function(vocab).set(sentencepiece_model_name="bpe_32k_c4.model")
    if vocab_size == 128 * 1024:
        return config_for_function(vocab).set(sentencepiece_model_name="bpe_128k_c4.model")
    raise ValueError(f"Unsupported vocabulary size: {vocab_size}")


def _train_input_source(*, max_sequence_length: int) -> InstantiableConfig:
    """Instantiates native AXLearn streaming input pipeline configured over gsm8k."""
    return config_for_function(tfds_input).set(
        dataset_name="gsm8k",
        split="train",
        is_training=True,
        vocab_cfg=_vocab_cfg(32 * 1024),
        max_sequence_length=max_sequence_length,
    )


def _eval_input_sources(*, max_sequence_length: int) -> dict[str, InstantiableConfig]:
    """Evaluates streaming input configurations targeting validations."""
    return {
        "validation": config_for_function(tfds_input).set(
            dataset_name="gsm8k",
            split="test",
            is_training=False,
            vocab_cfg=_vocab_cfg(32 * 1024),
            max_sequence_length=max_sequence_length,
        )
    }


def trainer_configs(
    model_size: str,
    version: fuji.Version = fuji.Version.V1,
) -> Dict[str, TrainerConfigFn]:
    """Named configurations mapper."""
    config_map = {}

    def get_config_fn():
        fuji_kwargs = fuji.get_trainer_kwargs(
            model_size, vocab_size=32 * 1024, version=version, flash_attention=True
        )

        fuji_model_cfg = fuji_kwargs["model_kwargs"]
        max_sequence_length = fuji_kwargs["max_sequence_length"]

        grpo_model_cfg = GrpoModel.default_config().set(
            actor=fuji.model_config(**fuji_model_cfg),
            reference=fuji.model_config(**fuji_model_cfg),
        )

        trainer_cfg = GrpoSpmdTrainer.default_config().set(
            model=grpo_model_cfg,
            learner=AxlearnGrpoLearner.default_config().set(
                num_generations=4,
                beta=0.04,
            ),
            mesh_shape=fuji_kwargs["mesh_shape"],
        )

        # Connect with native storage structures Input pipeline
        trainer_cfg.input.source = _train_input_source(max_sequence_length=max_sequence_length)
        return trainer_cfg

    config_key = f"grpo-fuji-{model_size}-v1"
    config_map[config_key] = get_config_fn
    return config_map


def named_trainer_configs() -> Dict[str, TrainerConfigFn]:
    """Configuration registry entry."""
    config_map = {}
    config_map.update(trainer_configs("test"))
    config_map.update(trainer_configs("7B"))
    config_map.update(trainer_configs("70B"))
    return config_map
