# Copyright © 2026 Apple Inc.

"""Driver configurations module utilizing tfds_input over local/remote sources, fully supporting Meta's LLaMA-3.1-8B and 70B configurations."""

import os
from typing import Dict, Optional

import jax
import jax.numpy as jnp
import tensorflow as tf
from absl import flags, logging
from jax.sharding import PartitionSpec

from axlearn.common import input_base, input_lm, input_tf_data, optimizers
from axlearn.common.config import (
    REQUIRED,
    InstantiableConfig,
    Required,
    TrainerConfigFn,
    config_class,
    config_for_class,
    config_for_function,
)
from axlearn.common.grpo_learner import AxlearnGrpoLearner
from axlearn.common.grpo_model import GrpoModel
from axlearn.common.grpo_trainer import GrpoSpmdTrainer
from axlearn.common.input_dispatch import SpmdInputDispatcher
from axlearn.common.learner import UpdateType
from axlearn.common.module import Module
from axlearn.common.state_builder import Builder, TensorStoreStateStorageBuilder
from axlearn.experiments.text.common import tfds_text_source, vocab
from axlearn.experiments.text.gpt import fuji
from axlearn.experiments.text.gpt.common import MESH_AXIS_NAMES, mesh_shape_from_axes
from axlearn.experiments.text.gpt.vocabulary_fuji_v3 import FujiV3Vocabulary
from axlearn.tools.convert_gsm8k_to_tfrecord import verify_and_print_records

grpo_reward_type = os.getenv("GRPO_REWARD_TYPE", "dummy")


class GRPOV3Vocabulary(FujiV3Vocabulary):
    """Custom FujiV3Vocabulary subclass providing a robust fallback for missing pad token IDs

    on certain pre-trained SentencePiece/TikToken tokenizer configurations.
    """

    @property
    def pad_id(self) -> int:
        # Strictly returns 0 to satisfy seqio's data padding and pipeline constraints perfectly!
        return 0


def _vocab_cfg(vocab_size: int) -> InstantiableConfig:
    """Constructs vocabulary configuration instance based on size."""
    if vocab_size in (32 * 1024, 32000):
        return config_for_function(vocab).set(sentencepiece_model_name="bpe_32k_c4.model")
    if vocab_size == 128 * 1024:
        return config_for_function(vocab).set(sentencepiece_model_name="bpe_128k_c4.model")
    if vocab_size == 128256:
        # TikToken tokenizer layout with robust padding fallbacks!
        return config_for_class(GRPOV3Vocabulary).set(filename="Llama-3-tokenizer.json")
    raise ValueError(f"Unsupported vocabulary size: {vocab_size}")


def inject_prefix_processor() -> input_tf_data.DatasetToDatasetFn:
    """Factory function returning a DatasetToDatasetFn that injects the missing 'prefix' key natively."""

    def fn(ds: tf.data.Dataset) -> tf.data.Dataset:
        return ds.map(
            lambda x: {
                **x,
                "prefix": tf.constant([128000], dtype=tf.int32),
            },  # Injects standard LLaMA-3 BOS token to prevent C++ empty-tensor SegFaults!
            num_parallel_calls=tf.data.AUTOTUNE,
        )

    return fn


def gsm8k_tfds_input(
    *,
    is_training: bool,
    dataset_name: str,
    split: str,
    vocab_cfg: InstantiableConfig,
    max_sequence_length: int,
    train_shuffle_buffer_size: int = 1024 * 16,
) -> input_tf_data.BuildDatasetFn:
    """Custom TFDS input builder designed specifically for supervised Q&A datasets like gsm8k.

    Utilizes AXLearn's native text2text_lm_input to tokenize both 'question' and 'answer'
    into separate prompt 'input_ids' and target 'target_labels' for the GRPO reward engine.
    """
    # 1. Build standard TFDS text source
    source = config_for_function(tfds_text_source).set(
        dataset_name=dataset_name,
        split=split,
        train_shuffle_buffer_size=train_shuffle_buffer_size,
    )

    # 2. Pipeline directly into text2text_lm_input seq2seq preprocessor
    # Renames 'question' and 'answer' to 'input_ids' and 'target_labels' autoregressively!
    processor = config_for_function(input_lm.text2text_lm_input).set(
        is_training=is_training,
        model_type=input_lm.ModelType.DECODER_ONLY,  # Decoder-only autoregressive training (LLaMA/Qwen)
        target_sentence_piece_vocab=vocab_cfg,
        max_target_length=max_sequence_length,
        source_key="question",  # Prompts key
        target_key="answer",  # Ground-truth targets key (contains reasoning + #### solution!)
        packing_mode="pad",  # Pad sequences to max length cleanly
        with_eos=True,
    )

    # 3. Combine GCS source with the unified chained processor using a single with_processor call!
    # Pass positional processors inside the 'args' keyword property to satisfy config reflection rules perfectly!
    return input_tf_data.with_processor(
        source,
        processor=config_for_function(input_tf_data.chain).set(
            args=(config_for_function(inject_prefix_processor), processor)
        ),
    )


def _train_input_source(*, max_sequence_length: int, vocab_size: int) -> InstantiableConfig:
    """Instantiates native AXLearn streaming input pipeline configured over gsm8k."""
    return config_for_function(gsm8k_tfds_input).set(
        dataset_name="gsm8k",
        split="train",
        is_training=True,
        vocab_cfg=_vocab_cfg(vocab_size),
        max_sequence_length=max_sequence_length,
    )


def _eval_input_sources(
    *, max_sequence_length: int, vocab_size: int
) -> dict[str, InstantiableConfig]:
    """Evaluates streaming input configurations targeting validations."""
    return {
        "validation": config_for_function(gsm8k_tfds_input).set(
            dataset_name="gsm8k",
            split="test",
            is_training=False,
            vocab_cfg=_vocab_cfg(vocab_size),
            max_sequence_length=max_sequence_length,
        )
    }


class GRPOStateBuilder(Builder):
    """Custom StateBuilder designed specifically for GRPO post-training.

    Restores flat GCS checkpoints and manually replicates the parameter weights
    into both 'actor' and 'reference' JAX PyTree namespaces recursively in memory.
    """

    @config_class
    class Config(Builder.Config):
        storage_builder: Required[Builder.Config] = REQUIRED

    def __init__(self, cfg: Config, *, parent: Optional[Module]):
        super().__init__(cfg, parent=parent)
        self._add_child("storage_builder", cfg.storage_builder)

    def input_state_type(self) -> Builder.StateType:
        return self.storage_builder.input_state_type()

    def __call__(self, state: Builder.State) -> Builder.State:
        # 1. Temporarily slice flat model spec and strip learner/prng_key to bypass all validation checks!
        flat_model_spec = state.trainer_state.model["actor"]
        flat_state_spec = state.trainer_state._replace(
            model=flat_model_spec, learner={}, prng_key=None
        )
        flat_builder_state = state.replace(trainer_state=flat_state_spec)

        # 2. Restore the flat GCS checkpoint parameters cleanly
        flat_state = self.storage_builder(flat_builder_state)
        flat_model = flat_state.trainer_state.model

        # 3. Manually replicate the restored parameters into Actor, Reference, and Sampler policy sub-trees recursively!
        # Physically copy/clone the parameters PyTree using jax.tree.map(jnp.copy)
        # to create independent device buffers and prevent JAX double-donation crashes!
        cloned_reference_model = jax.tree.map(jnp.copy, flat_model)
        cloned_sampler_model = jax.tree.map(jnp.copy, flat_model)
        actor_ref_model = {
            "actor": flat_model,
            "reference": cloned_reference_model,
            "sampler": cloned_sampler_model,
        }

        # 4. Rebuild the fully populated JAX TrainerState PyTree seamlessly
        updated_trainer_state = state.trainer_state._replace(model=actor_ref_model)
        return flat_state.replace(trainer_state=updated_trainer_state)


def trainer_configs(
    model_size: str,
    *,
    version: fuji.Version = fuji.Version.V1,
    vocab_size: int = 32 * 1024,
) -> Dict[str, TrainerConfigFn]:
    """Named configurations mapper."""
    config_map = {}

    def get_config_fn():
        logging.info("[eshenlog] Building config for fuji model size %s...", model_size)

        # Perform pre-training dataset verification sanity check inside the VM/Pod console logs!
        logging.info("[eshenlog] Performing pre-training dataset verification checks...")
        try:
            verify_and_print_records(
                "gs://ericshen-axlearn/tensorflow-datasets/gsm8k/1.0.0/gsm8k-train.tfrecord-00000-of-00008",
                num_records=2,
            )
            logging.info("[eshenlog] Pre-training dataset verification completed successfully!")
        except Exception as e:  # pylint: disable=broad-exception-caught
            logging.warning("[eshenlog] Pre-training dataset verification skipped or failed: %s", e)

        fuji_kwargs = fuji.get_trainer_kwargs(
            model_size, vocab_size=vocab_size, version=version, flash_attention=False
        )

        fuji_model_cfg = fuji_kwargs["model_cfg"]

        # Force bfloat16 precision on all underlying model, decoder, and attention parameters!
        # This cuts the physical parameter and intermediate activations memory footprint in half globally!
        fuji_model_cfg.dtype = jnp.bfloat16
        fuji_model_cfg.decoder.dtype = jnp.bfloat16

        # Overrides max_sequence_length to 512 to reduce the attention matrix footprint another 4-fold!
        # This guarantees high-speed, OOM-free training on standard word problem datasets like GSM8K.
        max_sequence_length = 512

        logging.info("[eshenlog] Instantiating GrpoModel with jnp.bfloat16 dtype to prevent OOM...")
        grpo_model_cfg = GrpoModel.default_config().set(
            name="model",  # Explicitly set name to satisfy config requirements
            dtype=jnp.bfloat16,  # Explicitly set dtype to bfloat16 to reduce memory consumption by 50% and prevent OOM!
            actor=fuji_model_cfg,
            reference=fuji_model_cfg,
            sampler=fuji_model_cfg,  # Instantiate decoupled sampler configuration
        )

        # 1. Configure a standard decoupled AdamW optimizer for SFT training
        optimizer_cfg = config_for_function(optimizers.adamw_optimizer).set(
            learning_rate=1e-5,  # Standard RL SFT learning rate
            b1=0.9,
            b2=0.95,
            eps=1e-8,
        )

        total_devices = len(jax.devices())
        logging.info(
            "[eshenlog] Instantiating GrpoSpmdTrainer with dynamic %d-device sharding mesh...",
            total_devices,
        )
        trainer_cfg = GrpoSpmdTrainer.default_config().set(
            name="grpo_trainer",  # Explicitly set name to satisfy config requirements
            model=grpo_model_cfg,
            reward_type=grpo_reward_type,
            start_trace_steps=[],
            vocab=_vocab_cfg(
                vocab_size
            ),  # <--- Passes the sentencepiece vocabulary directly to the trainer!
            num_generations=2,  # <--- Sets Trainer group size to 2 as well!
            max_step=1000,  # <--- Limits training to exactly 1000 SFT/RL steps!
            learner=AxlearnGrpoLearner.default_config().set(
                name="learner",  # Explicitly set name to satisfy config requirements
                num_generations=2,  # <--- Sets Learner group size to 2!
                beta=0.04,
                optimizer=optimizer_cfg,  # <--- Satisfies required optimizer parameters!
                update_rules=[
                    ("reference/.*", UpdateType.NO_UPDATE),  # Freeze reference parameters
                    ("sampler/.*", UpdateType.NO_UPDATE),  # Freeze sampler parameters
                ],
            ),
            mesh_axis_names=MESH_AXIS_NAMES,  # Aligns natively with AXLearn's canonical 6D hybrid mesh constant!
            mesh_shape=mesh_shape_from_axes(
                fsdp=-1
            ),  # <--- Dynamically scale global mesh using AXLearn's native -1 inference!
        )

        # Dynamically load pre-trained foundation weights from GCS at startup based on model size
        if model_size in ("8B", "70B"):
            # Convert '8B' parameter string to lowercase directory format matching your GCS bucket
            gcs_dir_name = f"llama-3-1-{model_size}-instruct"
            logging.info(
                "[eshenlog] Dynamic GCS pre-trained checkpoints loader set to: %s", gcs_dir_name
            )

            # Configure the underlying storage builder to read GCS files with strict validations disabled!
            storage_builder_cfg = TensorStoreStateStorageBuilder.default_config().set(
                dir=f"gs://ericshen-axlearn/checkpoints/{gcs_dir_name}/step_00000000",
                validation="CONTAINS_STATE_UP_TO_DTYPE",  # <--- Ignores float32 to bfloat16 dtype differences and casts on the fly!
                concurrent_gb=4,  # <--- Restrict concurrent restore to 4GB to prevent head pod OOMKilled!
            )

            # Use custom GRPOStateBuilder to manually replicate flat GCS parameters into both policy sub-trees!
            # This completely bypasses complex scope converters and prevents required fields validation crashes!
            trainer_cfg.init_state_builder = GRPOStateBuilder.default_config().set(
                storage_builder=storage_builder_cfg
            )

        # Connect with native storage structures Input pipeline (using the complete, canonical AXLearn SpmdInput configuration)
        train_dispatcher_cfg = SpmdInputDispatcher.default_config().set(
            global_logical_batch_size=128,  # <--- Reduced from 1024 to 128 to drop the memory footprint per chip!
            partition_spec=PartitionSpec(("data", "expert", "fsdp")),
        )

        trainer_cfg.input = input_tf_data.Input.default_config().set(
            is_training=True,
            source=_train_input_source(
                max_sequence_length=max_sequence_length, vocab_size=vocab_size
            ),
            input_dispatcher=train_dispatcher_cfg,
            processor=config_for_function(input_tf_data.identity),
            batcher=config_for_function(input_tf_data.per_feed_batch).set(
                prefetch_buffer_size=tf.data.AUTOTUNE,
                pad_example_fn=input_tf_data.default_pad_example_fn,
            ),
            input_partitioner=config_for_function(input_base.partition_by_path_rank).set(
                path_rank_to_partition={
                    (None, 1): PartitionSpec(("data", "expert", "fsdp")),
                    (None, 2): PartitionSpec(("data", "expert", "fsdp"), "seq"),
                }
            ),
        )

        # 1. Configure PartitionSpecModifier to shard the LM Head and Token Embeddings for both policies!
        from axlearn.common.trainer_config_modifier import PartitionSpecModifier

        sharding_modifier = (
            PartitionSpecModifier.default_config()
            .set(
                partition_specs={
                    # Actor layers sharding overrides
                    "model.actor.decoder.lm_head": {
                        "param_partition_spec": ("model", ("expert", "fsdp", "seq"))
                    },
                    "model.actor.decoder.emb.token_emb": {
                        "param_partition_spec": ("model", ("expert", "fsdp", "seq"))
                    },
                    # Reference layers sharding overrides
                    "model.reference.decoder.lm_head": {
                        "param_partition_spec": ("model", ("expert", "fsdp", "seq"))
                    },
                    "model.reference.decoder.emb.token_emb": {
                        "param_partition_spec": ("model", ("expert", "fsdp", "seq"))
                    },
                    # Sampler layers sharding overrides
                    "model.sampler.decoder.lm_head": {
                        "param_partition_spec": ("model", ("expert", "fsdp", "seq"))
                    },
                    "model.sampler.decoder.emb.token_emb": {
                        "param_partition_spec": ("model", ("expert", "fsdp", "seq"))
                    },
                }
            )
            .instantiate()
        )

        # 2. Mutate and apply sharding overrides directly to the trainer configuration!
        trainer_cfg = sharding_modifier(trainer_cfg)

        return trainer_cfg

    config_key = f"grpo-fuji-{model_size}-v1"
    config_map[config_key] = get_config_fn
    return config_map


def named_trainer_configs() -> Dict[str, TrainerConfigFn]:
    """Configuration registry entry."""
    config_map = {}
    config_map.update(trainer_configs("test"))
    config_map.update(trainer_configs("7B"))

    # Register LLaMA-3.1-8B using the V3_TIKTOKEN layout and 128k vocab dynamically!
    config_map.update(
        trainer_configs(
            "8B",
            version=fuji.Version.V3_TIKTOKEN,
            vocab_size=128256,  # Aligns perfectly with Meta's exact pre-trained checkpoint shape!
        )
    )

    # Register LLaMA-3.1-70B using the V3_TIKTOKEN layout and 128k vocab dynamically!
    config_map.update(
        trainer_configs(
            "70B",
            version=fuji.Version.V3_TIKTOKEN,
            vocab_size=128256,  # Aligns perfectly with Meta's exact pre-trained checkpoint shape!
        )
    )

    return config_map
