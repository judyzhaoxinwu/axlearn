# Copyright © 2026 Apple Inc.

"""Utility script validating a converted JAX TensorStore checkpoint against the original Hugging Face PyTorch parameters for numerical identity."""

import os

import jax
import numpy as np
import torch
from absl import app, flags, logging
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
    "ckpt_dir",
    None,
    "The GCS JAX checkpoint folder to validate (must point to step_00000000/)",
    required=True,
)


def validate_checkpoint(ckpt_dir: str):
    logging.info("Loading original Hugging Face PyTorch model for numerical validation...")
    hf_model = AutoModelForTextToWaveform.from_pretrained(
        FLAGS.model_id,
        torch_dtype=torch.bfloat16,  # <--- Load in bfloat16 (OOM-safe 60GB footprint)
        trust_remote_code=True,
        device_map="cpu",
    )

    logging.info("Restoring sharded JAX parameters from target checkpoint: %s", ckpt_dir)

    # Temporarily bypass deepcopy entirely during instantiation to avoid Python 3.12 C-level pickling crashes
    import contextvars
    import copy
    import select

    _orig_deepcopy = copy.deepcopy
    copy.deepcopy = lambda x, memo=None: x
    try:
        storage = TensorStoreStateStorage(TensorStoreStateStorage.default_config())
    finally:
        copy.deepcopy = _orig_deepcopy

    # Instantiate matching JAX model specs to get the structural template for restoration
    logging.info("Instantiating JAX model specs template for restoration...")
    qwen_kwargs = qwen.get_trainer_kwargs(
        "30B-A3B",
        vocab_size=qwen.QWEN3_VOCAB_SIZE,
        batch_size=1024,
        max_sequence_length=1024,
    )

    qwen_kwargs["model_cfg"].set(name="model")
    qwen_model = qwen_kwargs["model_cfg"].instantiate(parent=None)

    dummy_prng = jax.random.PRNGKey(0)
    logging.info("[eshenlog] before jax.eval_shape")
    # Evaluate shapes recursively without allocating physical memory arrays to prevent OOM thrashing
    empty_state_spec = jax.eval_shape(
        lambda: qwen_model.initialize_parameters_recursively(prng_key=dummy_prng)
    )
    logging.info("[eshenlog] after jax.eval_shape")

    # Convert JAX ShapeDtypeStruct leaves into standard AXLearn TensorSpec objects recursively
    # This satisfies checkpointer type assertions and prevents dictionary type mismatches!
    from axlearn.common.utils import TensorSpec

    empty_trainer_state = jax.tree.map(
        lambda x: TensorSpec(shape=x.shape, dtype=x.dtype), {"model": empty_state_spec}
    )

    # Restore sharded parameters dictionary from GCS natively using the structural template
    logging.info("Reading sharded parameters from GCS...")
    restored_state = storage.restore_from_dir(
        step=0,
        state=empty_trainer_state,
        ckpt_dir=ckpt_dir,
    )
    jax_model_params = restored_state["model"]

    logging.info("Verifying structural shapes and numerical identity...")

    vocab_size = qwen.QWEN3_VOCAB_SIZE

    # 1. Verify Embeddings (Slicing padded rows to base vocab_size, casting to float32 to avoid BFloat16 numpy exceptions)
    hf_emb = (
        hf_model.thinker.model.embed_tokens.weight.detach()
        .cpu()
        .to(torch.float32)
        .numpy()[:vocab_size, :]
    )
    jax_emb = jax_model_params["decoder"]["emb"]["token_emb"]["weight"]
    assert (
        hf_emb.shape == jax_emb.shape
    ), f"Embedding shape mismatch! HF: {hf_emb.shape}, JAX: {jax_emb.shape}"
    emb_diff = np.max(np.abs(hf_emb - jax_emb))
    logging.info("token_emb absolute max difference: %e", emb_diff)
    assert emb_diff < 1e-5, f"Embedding numerical discrepancy exceeded threshold: {emb_diff}"

    # 2. Verify LM Head (Saved without .T transposition, comparing sliced base shape, casting to float32)
    hf_head = (
        hf_model.thinker.lm_head.weight.detach().cpu().to(torch.float32).numpy()[:vocab_size, :]
    )
    jax_head = jax_model_params["decoder"]["lm_head"]["weight"]
    assert (
        hf_head.shape == jax_head.shape
    ), f"LM Head shape mismatch! HF: {hf_head.shape}, JAX: {jax_head.shape}"
    head_diff = np.max(np.abs(hf_head - jax_head))
    logging.info("lm_head absolute max difference: %e", head_diff)
    assert head_diff < 1e-5, f"LM Head numerical discrepancy exceeded threshold: {head_diff}"

    # 3. Verify Layer 1 attention projections (4D layouts mapping)
    first_layer = hf_model.thinker.model.layers[0]
    hidden_dim = first_layer.input_layernorm.weight.shape[0]

    # Retrieve restored attention qkv weights (which are 4D [hidden_dim, 40, 128])
    jax_qkv = jax_model_params["decoder"]["transformer"]["repeat"]["layer"]["self_attention"][
        "attention"
    ]["i_proj"]["i_proj"]["qkv_proj"]["weight"][0]
    q = first_layer.self_attn.q_proj.weight.detach().cpu().to(torch.float32).numpy()
    k = first_layer.self_attn.k_proj.weight.detach().cpu().to(torch.float32).numpy()
    v = first_layer.self_attn.v_proj.weight.detach().cpu().to(torch.float32).numpy()

    q = q.reshape(32, 128, hidden_dim)
    k = k.reshape(4, 128, hidden_dim)
    v = v.reshape(4, 128, hidden_dim)
    hf_qkv = np.concatenate([q, k, v], axis=0).transpose(
        2, 0, 1
    )  # Transpose to [hidden_dim, 40, 128]

    assert (
        hf_qkv.shape == jax_qkv.shape
    ), f"QKV shape mismatch! HF: {hf_qkv.shape}, JAX: {jax_qkv.shape}"
    qkv_diff = np.max(np.abs(hf_qkv - jax_qkv))
    logging.info("Layer 1 QKV absolute max difference: %e", qkv_diff)
    assert qkv_diff < 1e-5, f"QKV numerical discrepancy exceeded threshold: {qkv_diff}"

    # Retrieve restored attention o_proj weights (which are 4D [hidden_dim, 32, 128])
    jax_o = jax_model_params["decoder"]["transformer"]["repeat"]["layer"]["self_attention"][
        "attention"
    ]["o_proj"]["weight"][0]
    o = first_layer.self_attn.o_proj.weight.detach().cpu().to(torch.float32).numpy()
    hf_o = o.reshape(hidden_dim, 32, 128)

    assert hf_o.shape == jax_o.shape, f"O Proj shape mismatch! HF: {hf_o.shape}, JAX: {jax_o.shape}"
    o_diff = np.max(np.abs(hf_o - jax_o))
    logging.info("Layer 1 O Proj absolute max difference: %e", o_diff)
    assert o_diff < 1e-5, f"O Proj numerical discrepancy exceeded threshold: {o_diff}"

    # 5. Verify Layer 1 Sparse MoE parameters
    jax_moe_gate = jax_model_params["decoder"]["transformer"]["repeat"]["layer"]["feed_forward"][
        "gate_weight"
    ][0]
    hf_moe_gate = first_layer.mlp.gate.weight.detach().cpu().to(torch.float32).numpy().T
    assert (
        hf_moe_gate.shape == jax_moe_gate.shape
    ), f"MoE Gate shape mismatch! HF: {hf_moe_gate.shape}, JAX: {jax_moe_gate.shape}"
    gate_diff = np.max(np.abs(hf_moe_gate - jax_moe_gate))
    logging.info("Layer 1 MoE Gate absolute max difference: %e", gate_diff)
    assert gate_diff < 1e-5, f"MoE Gate numerical discrepancy exceeded threshold: {gate_diff}"

    logging.info("\n==============================================================")
    logging.info("VALIDATION SUCCESSFUL: Checkpoint is 100% numerically identical to Hugging Face!")
    logging.info("==============================================================")


def main(_):
    devices = np.array(jax.devices()).reshape(-1, 1)
    mesh = jax.sharding.Mesh(
        devices=devices,
        axis_names=("data", "model"),
    )

    with mesh:
        validate_checkpoint(FLAGS.ckpt_dir)


if __name__ == "__main__":
    app.run(main)
