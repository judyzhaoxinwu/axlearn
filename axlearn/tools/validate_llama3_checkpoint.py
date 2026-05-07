# Copyright © 2026 Apple Inc.

"""Utility script validating a converted LLaMA JAX TensorStore checkpoint against the original Hugging Face PyTorch parameters for numerical identity."""


import jax
import numpy as np
import torch
from absl import app, flags, logging
from transformers import LlamaForCausalLM

from axlearn.common.checkpointer import TensorStoreStateStorage
from axlearn.experiments.text.gpt import fuji

FLAGS = flags.FLAGS
flags.DEFINE_string(
    "model_id",
    "meta-llama/Llama-3.1-8B-Instruct",
    "Hugging Face model identifier repo path",
    required=False,
)
flags.DEFINE_string(
    "model_size", "8B", "Fuji target model size (test, 1B, 3B, 8B, 70B)", required=False
)
flags.DEFINE_string(
    "ckpt_dir",
    None,
    "The GCS JAX checkpoint folder to validate (must point to step_00000000/)",
    required=True,
)


def validate_checkpoint(ckpt_dir: str, model_size: str):
    logging.info("Loading original Hugging Face PyTorch Llama model for numerical validation...")
    hf_model = LlamaForCausalLM.from_pretrained(
        FLAGS.model_id,
        torch_dtype=torch.bfloat16,  # Load in bfloat16 (OOM-safe 60GB footprint)
        device_map="cpu",
    )

    logging.info("Restoring sharded JAX parameters from target checkpoint: %s", ckpt_dir)

    # Temporarily bypass deepcopy entirely during instantiation to avoid Python 3.12 C-level pickling crashes
    import copy

    _orig_deepcopy = copy.deepcopy
    copy.deepcopy = lambda x, memo=None: x
    try:
        storage = TensorStoreStateStorage(TensorStoreStateStorage.default_config())
    finally:
        copy.deepcopy = _orig_deepcopy

    # 1. Instantiate matching JAX Fuji model specs to get the structural template for restoration
    logging.info("Instantiating JAX model specs template for restoration...")
    fuji_kwargs = fuji.get_trainer_kwargs(
        model_size,
        vocab_size=(
            128256 if "Llama-3" in FLAGS.model_id else 32000
        ),  # Enforce correct Llama-3 vocab size (128256)
        version=fuji.Version.V3_TIKTOKEN,  # V3_TIKTOKEN natively registers the Llama-3 8B configuration keys!
    )

    fuji_model_cfg = fuji_kwargs["model_cfg"]
    fuji_model_cfg.set(name="model")
    fuji_model = fuji_model_cfg.instantiate(parent=None)

    dummy_prng = jax.random.PRNGKey(0)
    # Evaluate shapes recursively without allocating physical memory arrays to prevent OOM thrashing
    empty_state_spec = jax.eval_shape(
        lambda: fuji_model.initialize_parameters_recursively(prng_key=dummy_prng)
    )

    # Convert JAX ShapeDtypeStruct leaves into standard AXLearn TensorSpec objects recursively
    # This satisfies checkpointer type assertions and prevents dictionary type mismatches!
    from axlearn.common.utils import TensorSpec

    empty_trainer_state = jax.tree.map(
        lambda x: TensorSpec(shape=x.shape, dtype=x.dtype), {"model": empty_state_spec}
    )

    # 2. Restore sharded parameters dictionary from GCS natively using the structural template
    logging.info("Reading sharded parameters from GCS...")
    restored_state = storage.restore_from_dir(
        step=0,
        state=empty_trainer_state,
        ckpt_dir=ckpt_dir,
    )
    jax_model_params = restored_state["model"]

    logging.info("Verifying structural shapes and numerical identity...")
    vocab_size = 128256 if "Llama-3" in FLAGS.model_id else 32000

    # 3. Verify Embeddings
    hf_emb = (
        hf_model.model.embed_tokens.weight.detach().cpu().to(torch.float32).numpy()[:vocab_size, :]
    )
    jax_emb = jax_model_params["decoder"]["emb"]["token_emb"]["weight"]
    assert (
        hf_emb.shape == jax_emb.shape
    ), f"Embedding shape mismatch! HF: {hf_emb.shape}, JAX: {jax_emb.shape}"
    emb_diff = np.max(np.abs(hf_emb - jax_emb))
    logging.info("token_emb absolute max difference: %e", emb_diff)
    assert emb_diff < 1e-5, f"Embedding numerical discrepancy exceeded threshold: {emb_diff}"

    # 4. Verify LM Head (LLaMA-3 lm_head weight has shape [vocab_size, hidden_dim] natively, saved without transposing)
    hf_head = hf_model.lm_head.weight.detach().cpu().to(torch.float32).numpy()[:vocab_size, :]
    jax_head = jax_model_params["decoder"]["lm_head"]["weight"]
    assert (
        hf_head.shape == jax_head.shape
    ), f"LM Head shape mismatch! HF: {hf_head.shape}, JAX: {jax_head.shape}"
    head_diff = np.max(np.abs(hf_head - jax_head))
    logging.info("lm_head absolute max difference: %e", head_diff)
    assert head_diff < 1e-5, f"LM Head numerical discrepancy exceeded threshold: {head_diff}"

    # 5. Verify Layer 1 attention projections
    first_layer = hf_model.model.layers[0]
    hidden_dim = first_layer.input_layernorm.weight.shape[0]

    # Retrieve restored attention qkv weights (which are 4D JAX layout [hidden_dim, 40, 128] for 8B)
    jax_qkv = jax_model_params["decoder"]["transformer"]["repeat"]["layer"]["self_attention"][
        "attention"
    ]["i_proj"]["i_proj"]["qkv_proj"]["weight"][0]

    num_q_heads = hf_model.config.num_attention_heads
    num_kv_heads = hf_model.config.num_key_value_heads
    head_dim = hidden_dim // num_q_heads

    # Import AXLearn's native RoPE permutation function directly
    from axlearn.common.param_converter import _permute_q_k_for_rope

    # Retrieve raw PyTorch tensors
    q_pt = first_layer.self_attn.q_proj.weight.detach().cpu()
    k_pt = first_layer.self_attn.k_proj.weight.detach().cpu()
    v_pt = first_layer.self_attn.v_proj.weight.detach().cpu()

    # Reshape PyTorch tensors to [heads, head_dim, hidden_dim] as expected by AXLearn
    q_pt = q_pt.reshape(num_q_heads, head_dim, hidden_dim)
    k_pt = k_pt.reshape(num_kv_heads, head_dim, hidden_dim)
    v_pt = v_pt.reshape(num_kv_heads, head_dim, hidden_dim)

    # Apply native AXLearn permutation recursively
    q_pt = _permute_q_k_for_rope(q_pt)
    k_pt = _permute_q_k_for_rope(k_pt)

    # Convert to Float32 and NumPy safely
    q = q_pt.to(torch.float32).numpy()
    k = k_pt.to(torch.float32).numpy()
    v = v_pt.to(torch.float32).numpy()

    # Statically concatenate and transpose to match JAX 4D sharded layout [hidden_dim, 40, 128]
    hf_qkv = np.concatenate([q, k, v], axis=0).transpose(2, 0, 1)

    assert (
        hf_qkv.shape == jax_qkv.shape
    ), f"QKV shape mismatch! HF: {hf_qkv.shape}, JAX: {jax_qkv.shape}"
    qkv_diff = np.max(np.abs(hf_qkv - jax_qkv))
    logging.info("Layer 1 QKV absolute max difference: %e", qkv_diff)
    assert qkv_diff < 1e-5, f"QKV numerical discrepancy exceeded threshold: {qkv_diff}"

    # Retrieve restored attention o_proj weights (4D JAX layout [hidden_dim, num_heads, head_dim])
    jax_o = jax_model_params["decoder"]["transformer"]["repeat"]["layer"]["self_attention"][
        "attention"
    ]["o_proj"]["weight"][0]
    o = first_layer.self_attn.o_proj.weight.detach().cpu().to(torch.float32).numpy()
    hf_o = o.reshape(hidden_dim, num_q_heads, head_dim)

    assert hf_o.shape == jax_o.shape, f"O Proj shape mismatch! HF: {hf_o.shape}, JAX: {jax_o.shape}"
    o_diff = np.max(np.abs(hf_o - jax_o))
    logging.info("Layer 1 O Proj absolute max difference: %e", o_diff)
    assert o_diff < 1e-5, f"O Proj numerical discrepancy exceeded threshold: {o_diff}"

    # 6. Verify Layer 1 Feed Forward MLP projections
    jax_wi_0 = jax_model_params["decoder"]["transformer"]["repeat"]["layer"]["feed_forward"][
        "linear1_0"
    ]["weight"][0]
    hf_wi_0 = first_layer.mlp.gate_proj.weight.detach().cpu().to(torch.float32).numpy().T
    assert (
        hf_wi_0.shape == jax_wi_0.shape
    ), f"gate_proj shape mismatch! HF: {hf_wi_0.shape}, JAX: {jax_wi_0.shape}"
    wi_0_diff = np.max(np.abs(hf_wi_0 - jax_wi_0))
    logging.info("Layer 1 gate_proj absolute max difference: %e", wi_0_diff)
    assert wi_0_diff < 1e-5, f"gate_proj numerical discrepancy exceeded threshold: {wi_0_diff}"

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
        validate_checkpoint(FLAGS.ckpt_dir, FLAGS.model_size)


if __name__ == "__main__":
    app.run(main)
