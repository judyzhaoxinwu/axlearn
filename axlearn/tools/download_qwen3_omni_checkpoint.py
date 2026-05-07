# Copyright © 2026 Apple Inc.

"""Utility script serializing Qwen MoE checkpoints into sharded JAX TensorStore arrays cleanly, fully aligned with JAX specs shapes."""

import _thread
import contextvars
import copy
import gc
import os
import select
import threading
import time

import jax
import numpy as np
import tensorflow as tf
import torch
from absl import app, flags, logging
from huggingface_hub import snapshot_download
from transformers import AutoModelForTextToWaveform

from axlearn.common.checkpointer import TensorStoreStateStorage
from axlearn.experiments.text.gpt import qwen

FLAGS = flags.FLAGS
flags.DEFINE_string(
    "model_id",
    "Qwen/Qwen3-Omni-30B-A3B-Instruct",
    "Hugging Face model identifier repo path",
    required=False,
)
flags.DEFINE_string(
    "output_dir",
    None,
    "Destination GCS Bucket path or local folder for JAX checkpoints",
    required=True,
)


def download_hf_model(model_id: str) -> str:
    """Downloads PyTorch model checkpoints from Hugging Face Hub."""
    logging.info("Downloading Hugging Face model: %s", model_id)
    local_dir = snapshot_download(
        repo_id=model_id,
        ignore_patterns=["*.msgpack", "*.h5", "*.ot"],
    )
    logging.info("Model successfully downloaded locally to: %s", local_dir)
    return local_dir


def extract_numpy_weights(hf_model) -> dict:
    """Extracts all PyTorch weights directly into contiguous pre-allocated NumPy arrays fully matching JAX 4D shapes."""
    num_layers = len(hf_model.thinker.model.layers)
    logging.info("Pre-allocating contiguous NumPy arrays for %d layers...", num_layers)

    # Extract dimensions dynamically from JAX config (Vocab size is exactly 151936 for Qwen3)
    vocab_size = qwen.QWEN3_VOCAB_SIZE
    first_layer = hf_model.thinker.model.layers[0]
    hidden_dim = first_layer.input_layernorm.weight.shape[0]
    num_experts = 128

    gate_up_fused = first_layer.mlp.experts.gate_up_proj.detach().cpu().to(torch.float32).numpy()
    intermediate_dim = gate_up_fused.shape[1] // 2

    flat_params = {
        "token_emb": hf_model.thinker.model.embed_tokens.weight.detach()
        .cpu()
        .to(torch.float32)
        .numpy()[:vocab_size, :],  # Slice padded rows back to base 151936
        "lm_head": hf_model.thinker.lm_head.weight.detach()
        .cpu()
        .to(torch.float32)
        .numpy()[:vocab_size, :],  # Slice padded rows back to base 151936
        "output_norm": hf_model.thinker.model.norm.weight.detach().cpu().to(torch.float32).numpy(),
        # Pre-allocated contiguous JAX-aligned arrays
        "input_norm": np.empty((num_layers, hidden_dim), dtype=np.float32),
        "post_norm": np.empty((num_layers, hidden_dim), dtype=np.float32),
        "qkv": np.empty(
            (num_layers, hidden_dim, 40, 128), dtype=np.float32
        ),  # 4D JAX shape [layers, hidden_dim, 40, 128]
        "o": np.empty(
            (num_layers, hidden_dim, 32, 128), dtype=np.float32
        ),  # 4D JAX shape [layers, hidden_dim, 32, 128]
        "scale_query": np.empty((num_layers, 128), dtype=np.float32),  # Attention RMSNorm scale
        "scale_key": np.empty((num_layers, 128), dtype=np.float32),  # Attention RMSNorm scale
        "moe_gate_weight": np.empty((num_layers, hidden_dim, num_experts), dtype=np.float32),
        "moe_wi_0": np.empty(
            (num_layers, num_experts, hidden_dim, intermediate_dim), dtype=np.float32
        ),
        "moe_wi_1": np.empty(
            (num_layers, num_experts, hidden_dim, intermediate_dim), dtype=np.float32
        ),
        "moe_wo": np.empty(
            (num_layers, num_experts, intermediate_dim, hidden_dim), dtype=np.float32
        ),
    }

    # 2. Extract layers parameters sequentially directly into the pre-allocated slots
    for i in range(num_layers):
        layer = hf_model.thinker.model.layers[i]
        logging.info(
            "Extracting layer %d/%d directly into pre-allocated memory...", i + 1, num_layers
        )

        flat_params["input_norm"][i] = (
            layer.input_layernorm.weight.detach().cpu().to(torch.float32).numpy()
        )
        flat_params["post_norm"][i] = (
            layer.post_attention_layernorm.weight.detach().cpu().to(torch.float32).numpy()
        )

        # Attention projections GQA concatenation (Reshaping to 4D [hidden_dim, heads, head_dim] instead of flattening)
        q = layer.self_attn.q_proj.weight.detach().cpu().to(torch.float32).numpy()  # [4096, 2048]
        k = layer.self_attn.k_proj.weight.detach().cpu().to(torch.float32).numpy()  # [512, 2048]
        v = layer.self_attn.v_proj.weight.detach().cpu().to(torch.float32).numpy()  # [512, 2048]

        q = q.reshape(32, 128, hidden_dim)
        k = k.reshape(4, 128, hidden_dim)
        v = v.reshape(4, 128, hidden_dim)

        qkv_layer = np.concatenate([q, k, v], axis=0)  # shape [40, 128, 2048]
        flat_params["qkv"][i] = qkv_layer.transpose(2, 0, 1)  # Transpose to [2048, 40, 128]

        # Extract o_proj.weight and reshape to [hidden_dim, 32, 128]
        o = layer.self_attn.o_proj.weight.detach().cpu().to(torch.float32).numpy()  # [2048, 4096]
        flat_params["o"][i] = o.reshape(hidden_dim, 32, 128)

        # Extract Attention Query/Key RMSNorm scales
        flat_params["scale_query"][i] = (
            layer.self_attn.q_norm.weight.detach().cpu().to(torch.float32).numpy()
        )
        flat_params["scale_key"][i] = (
            layer.self_attn.k_norm.weight.detach().cpu().to(torch.float32).numpy()
        )

        # Sparse MoE parameters extraction
        if hasattr(layer, "mlp"):
            flat_params["moe_gate_weight"][i] = (
                layer.mlp.gate.weight.detach().cpu().to(torch.float32).numpy().T
            )

            gate_up_fused = layer.mlp.experts.gate_up_proj.detach().cpu().to(torch.float32).numpy()
            gate_up = gate_up_fused.reshape(num_experts, 2, intermediate_dim, hidden_dim)

            flat_params["moe_wi_0"][i] = gate_up[:, 0, :, :].transpose(0, 2, 1)
            flat_params["moe_wi_1"][i] = gate_up[:, 1, :, :].transpose(0, 2, 1)

            down_fused = layer.mlp.experts.down_proj.detach().cpu().to(torch.float32).numpy()
            flat_params["moe_wo"][i] = down_fused.transpose(0, 2, 1)

        # Free PyTorch layer memory immediately
        hf_model.thinker.model.layers[i] = None
        gc.collect()

    return flat_params


def populate_jax_state(flat_params: dict) -> dict:
    """Structures the pre-allocated NumPy arrays directly into JAX Arrays using progressive sequential memory collection."""
    logging.info("Structuring JAX parameter tree natively with pre-allocated JAX Arrays...")
    import jax.numpy as jnp

    # Initialize JAX parameter variables sequentially, executing gc.collect() after each major pop
    emb_weight = jnp.array(flat_params.pop("token_emb"))
    gc.collect()

    lm_head_weight = jnp.array(flat_params.pop("lm_head"))
    gc.collect()

    output_norm_scale = jnp.array(flat_params.pop("output_norm"))
    gc.collect()

    input_norm_scale = jnp.array(flat_params.pop("input_norm"))
    gc.collect()

    qkv_weight = jnp.array(flat_params.pop("qkv"))
    gc.collect()

    o_weight = jnp.array(flat_params.pop("o"))
    gc.collect()

    scale_query_scale = jnp.array(flat_params.pop("scale_query"))
    gc.collect()

    scale_key_scale = jnp.array(flat_params.pop("scale_key"))
    gc.collect()

    post_norm_scale = jnp.array(flat_params.pop("post_norm"))
    gc.collect()

    gate_weight = jnp.array(flat_params.pop("moe_gate_weight"))
    gc.collect()

    wi_0_weight = jnp.array(flat_params.pop("moe_wi_0"))
    gc.collect()

    wi_1_weight = jnp.array(flat_params.pop("moe_wi_1"))
    gc.collect()

    wo_weight = jnp.array(flat_params.pop("moe_wo"))
    gc.collect()

    # Reconstruct the exact nested JAX parameters dictionary expected by SpmdTrainer natively
    jax_params = {
        "decoder": {
            "emb": {"token_emb": {"weight": emb_weight}},
            "lm_head": {"weight": lm_head_weight},
            "output_norm": {"scale": output_norm_scale},
            "transformer": {
                "repeat": {
                    "layer": {
                        "self_attention": {
                            "norm": {"scale": input_norm_scale},
                            "attention": {
                                "i_proj": {"i_proj": {"qkv_proj": {"weight": qkv_weight}}},
                                "o_proj": {"weight": o_weight},
                                "scale_query": {"norm": {"scale": scale_query_scale}},
                                "scale_key": {"norm": {"scale": scale_key_scale}},
                            },
                        },
                        "feed_forward": {
                            "norm": {"scale": post_norm_scale},
                            "gate_weight": gate_weight,
                            "wi_0_weight": wi_0_weight,
                            "wi_1_weight": wi_1_weight,
                            "wo_weight": wo_weight,
                        },
                    }
                }
            },
        }
    }
    return jax_params


def monitor_upload_progress(step_dir: str, total_expected_files: int, stop_event: threading.Event):
    """Polls the GCS output folder recursively to count successfully written files."""
    logging.info("Starting GCS upload progress monitor thread...")

    while not stop_event.is_set():
        try:
            gda_dir = os.path.join(step_dir, "gda")
            file_count = 0
            if tf.io.gfile.exists(gda_dir):
                for root, dirs, files in tf.io.gfile.walk(gda_dir):
                    file_count += len(files)

            percentage = (
                (file_count / total_expected_files) * 100 if total_expected_files > 0 else 0
            )
            logging.info(
                "Uploading weights to GCS... %d%% [%d/%d files written]",
                int(percentage),
                file_count,
                total_expected_files,
            )
        except Exception:
            pass
        time.sleep(5)


def convert_and_save(local_pytorch_dir: str, output_dir: str):
    logging.info("Loading Hugging Face PyTorch Qwen Model...")
    hf_model = AutoModelForTextToWaveform.from_pretrained(
        local_pytorch_dir, torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="cpu"
    )

    # --- Step 1: Pre-allocate and Extract contiguous weights ---
    flat_params = extract_numpy_weights(hf_model)

    logging.info("Deleting PyTorch model and clearing garbage collector...")
    del hf_model
    gc.collect()

    # --- Step 2: Structure JAX parameters tree natively ---
    jax_params = populate_jax_state(flat_params)

    logging.info("Purging intermediate flat parameters dictionary...")
    del flat_params
    gc.collect()

    # --- Step 3: Serialize and Save ---
    logging.info("Serializing sharded JAX parameters to GCS path...")

    # Temporarily bypass deepcopy entirely during instantiation to avoid Python 3.12 C-level pickling crashes
    _orig_deepcopy = copy.deepcopy
    copy.deepcopy = lambda x, memo=None: x
    try:
        storage_cfg = TensorStoreStateStorage.default_config().set(max_concurrent_gb=16)
        storage = TensorStoreStateStorage(storage_cfg)
    finally:
        # Restore original deepcopy immediately
        copy.deepcopy = _orig_deepcopy

    # Save the model state natively under step_00000000 folder
    step_dir = os.path.join(output_dir, "step_00000000")
    tf.io.gfile.makedirs(step_dir)

    # Mapped structure expected inside trainer_state
    trainer_state = {"model": jax_params}

    # Start progress monitoring thread in background (Dynamically count exact sharded JAX leaves!)
    total_expected_files = len(jax.tree_util.tree_leaves(jax_params))
    stop_event = threading.Event()
    monitor_thread = threading.Thread(
        target=monitor_upload_progress, args=(step_dir, total_expected_files, stop_event)
    )

    monitor_thread.daemon = True
    monitor_thread.start()

    try:
        storage.save_to_dir(
            step=0,
            state=trainer_state,
            ckpt_dir=step_dir,
        )

        logging.info("Waiting for asynchronous GCS serialization threads to fully commit...")
        storage._manager.wait_until_finished()  # <--- Force main thread to block until all 60GB are written!
    finally:
        stop_event.set()
        monitor_thread.join(timeout=2)

    logging.info("Checkpoints successfully sharded and saved to GCS.")


def main(_):
    devices = np.array(jax.devices()).reshape(-1, 1)
    mesh = jax.sharding.Mesh(
        devices=devices,
        axis_names=("data", "model"),
    )

    with mesh:
        local_pytorch_dir = download_hf_model(FLAGS.model_id)
        convert_and_save(local_pytorch_dir, FLAGS.output_dir)


if __name__ == "__main__":
    app.run(main)
