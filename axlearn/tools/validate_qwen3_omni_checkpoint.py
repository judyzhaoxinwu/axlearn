# Copyright © 2026 Apple Inc.

"""Utility script validating a converted JAX TensorStore checkpoint against the original Hugging Face PyTorch parameters for numerical identity."""

import os

import jax
import numpy as np
import torch
from absl import app, flags, logging
from transformers import AutoModel, AutoModelForCausalLM

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


def permute_q_k_for_rope_numpy(vector: np.ndarray) -> np.ndarray:
    """Permutes Q and K vectors from HuggingFace split-half layout to AxLearn interleaved layout.

    Args:
        vector: A numpy array of shape [num_heads, head_dim, hidden_dim].
    Returns:
        A numpy array of the same shape with the head_dim axis interleaved.
    """
    n, h, d = vector.shape
    vector = vector.reshape(n, 2, h // 2, d).transpose(0, 2, 1, 3)
    return vector.reshape(n, h, d)


def validate_checkpoint(ckpt_dir: str):
    logging.info("Loading original Hugging Face PyTorch model for numerical validation...")

    # Try loading with AutoModelForCausalLM first (for text-only models like Qwen3-30B-Instruct-2507),
    # and fall back to AutoModel (for multimodal models like Qwen3-Omni) if unrecognized.
    try:
        hf_model = AutoModelForCausalLM.from_pretrained(
            FLAGS.model_id,
            torch_dtype=torch.bfloat16,  # <--- Load in bfloat16 (OOM-safe 60GB footprint)
            trust_remote_code=True,
            device_map="cpu",
        )
    except ValueError:
        hf_model = AutoModel.from_pretrained(
            FLAGS.model_id,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="cpu",
        )

    # Dynamically resolve the text model block (nested under .thinker for Omni speech models,
    # or at the top level for standard text-only causal language models).
    text_model = hf_model.thinker if hasattr(hf_model, "thinker") else hf_model

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

    # Remap empty_trainer_state from outer to inner to match the new checkpoint layout
    def remap_outer_to_inner(d):
        if hasattr(d, "items"):
            new_dict = {}
            for k, v in d.items():
                new_dict[k] = remap_outer_to_inner(v)
            scale_key = new_dict.pop("scale_key", None)
            scale_query = new_dict.pop("scale_query", None)
            if scale_key is not None or scale_query is not None:
                if "i_proj" not in new_dict:
                    new_dict["i_proj"] = {}
                if scale_key is not None:
                    new_dict["i_proj"]["scale_key"] = scale_key
                if scale_query is not None:
                    new_dict["i_proj"]["scale_query"] = scale_query
            return new_dict
        if isinstance(d, list):
            return [remap_outer_to_inner(x) for x in d]
        if isinstance(d, tuple):
            return tuple(remap_outer_to_inner(x) for x in d)
        return d

    empty_trainer_state = remap_outer_to_inner(empty_trainer_state)

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
        text_model.model.embed_tokens.weight.detach()
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
    hf_head = text_model.lm_head.weight.detach().cpu().to(torch.float32).numpy()[:vocab_size, :]
    jax_head = jax_model_params["decoder"]["lm_head"]["weight"]
    assert (
        hf_head.shape == jax_head.shape
    ), f"LM Head shape mismatch! HF: {hf_head.shape}, JAX: {jax_head.shape}"
    head_diff = np.max(np.abs(hf_head - jax_head))
    logging.info("lm_head absolute max difference: %e", head_diff)
    assert head_diff < 1e-5, f"LM Head numerical discrepancy exceeded threshold: {head_diff}"

    num_layers = len(text_model.model.layers)
    logging.info(
        f"Verifying all {num_layers} layers across QKV, O Proj, Scales, Gate, wi_0, wi_1, and wo..."
    )

    for idx in range(num_layers):
        layer_hf = text_model.model.layers[idx]
        hidden_dim = layer_hf.input_layernorm.weight.shape[0]

        # Retrieve restored attention qkv weights
        jax_qkv = jax_model_params["decoder"]["transformer"]["repeat"]["layer"]["self_attention"][
            "attention"
        ]["i_proj"]["i_proj"]["qkv_proj"]["weight"][idx]
        q = layer_hf.self_attn.q_proj.weight.detach().cpu().to(torch.float32).numpy()
        k = layer_hf.self_attn.k_proj.weight.detach().cpu().to(torch.float32).numpy()
        v = layer_hf.self_attn.v_proj.weight.detach().cpu().to(torch.float32).numpy()

        q = q.reshape(32, 128, hidden_dim)
        k = k.reshape(4, 128, hidden_dim)
        v = v.reshape(4, 128, hidden_dim)

        q_permuted = permute_q_k_for_rope_numpy(q)
        k_permuted = permute_q_k_for_rope_numpy(k)

        hf_qkv = np.concatenate([q_permuted, k_permuted, v], axis=0).transpose(
            2, 0, 1
        )  # Transpose to [hidden_dim, 40, 128]

        assert (
            hf_qkv.shape == jax_qkv.shape
        ), f"Layer {idx} QKV shape mismatch! HF: {hf_qkv.shape}, JAX: {jax_qkv.shape}"
        qkv_diff = np.max(np.abs(hf_qkv - jax_qkv))
        assert (
            qkv_diff < 1e-5
        ), f"Layer {idx} QKV numerical discrepancy exceeded threshold: {qkv_diff}"

        # 4. Verify attention o_proj
        jax_o = jax_model_params["decoder"]["transformer"]["repeat"]["layer"]["self_attention"][
            "attention"
        ]["o_proj"]["weight"][idx]
        o = layer_hf.self_attn.o_proj.weight.detach().cpu().to(torch.float32).numpy()
        hf_o = o.reshape(hidden_dim, 32, 128)

        assert (
            hf_o.shape == jax_o.shape
        ), f"Layer {idx} O Proj shape mismatch! HF: {hf_o.shape}, JAX: {jax_o.shape}"
        o_diff = np.max(np.abs(hf_o - jax_o))
        assert (
            o_diff < 1e-5
        ), f"Layer {idx} O Proj numerical discrepancy exceeded threshold: {o_diff}"

        # 5. Verify QK-Norm Scales (at their inner attention paths!)
        jax_scale_query = jax_model_params["decoder"]["transformer"]["repeat"]["layer"][
            "self_attention"
        ]["attention"]["i_proj"]["scale_query"]["norm"]["scale"][idx]
        jax_scale_key = jax_model_params["decoder"]["transformer"]["repeat"]["layer"][
            "self_attention"
        ]["attention"]["i_proj"]["scale_key"]["norm"]["scale"][idx]

        hf_scale_query = layer_hf.self_attn.q_norm.weight.detach().cpu().to(torch.float32).numpy()
        hf_scale_key = layer_hf.self_attn.k_norm.weight.detach().cpu().to(torch.float32).numpy()

        assert (
            hf_scale_query.shape == jax_scale_query.shape
        ), f"Layer {idx} Scale Query shape mismatch!"
        assert hf_scale_key.shape == jax_scale_key.shape, f"Layer {idx} Scale Key shape mismatch!"

        query_norm_diff = np.max(np.abs(hf_scale_query - jax_scale_query))
        key_norm_diff = np.max(np.abs(hf_scale_key - jax_scale_key))

        assert (
            query_norm_diff < 1e-5
        ), f"Layer {idx} Scale Query discrepancy exceeded threshold: {query_norm_diff}"
        assert (
            key_norm_diff < 1e-5
        ), f"Layer {idx} Scale Key discrepancy exceeded threshold: {key_norm_diff}"

        # 6. Verify Sparse MoE parameters
        jax_moe_gate = jax_model_params["decoder"]["transformer"]["repeat"]["layer"][
            "feed_forward"
        ]["gate_weight"][idx]
        hf_moe_gate = layer_hf.mlp.gate.weight.detach().cpu().to(torch.float32).numpy().T
        assert (
            hf_moe_gate.shape == jax_moe_gate.shape
        ), f"Layer {idx} MoE Gate shape mismatch! HF: {hf_moe_gate.shape}, JAX: {jax_moe_gate.shape}"
        gate_diff = np.max(np.abs(hf_moe_gate - jax_moe_gate))
        assert (
            gate_diff < 1e-5
        ), f"Layer {idx} MoE Gate numerical discrepancy exceeded threshold: {gate_diff}"

        # 7. Verify Sparse MoE Expert Projections (wi_0, wi_1, wo)
        jax_wi_0 = jax_model_params["decoder"]["transformer"]["repeat"]["layer"]["feed_forward"][
            "wi_0_weight"
        ][idx]
        jax_wi_1 = jax_model_params["decoder"]["transformer"]["repeat"]["layer"]["feed_forward"][
            "wi_1_weight"
        ][idx]
        jax_wo = jax_model_params["decoder"]["transformer"]["repeat"]["layer"]["feed_forward"][
            "wo_weight"
        ][idx]

        if hasattr(layer_hf.mlp, "experts") and isinstance(
            layer_hf.mlp.experts, (list, torch.nn.ModuleList)
        ):
            num_experts = len(layer_hf.mlp.experts)
            hf_wi_0 = np.stack(
                [
                    layer_hf.mlp.experts[e]
                    .gate_proj.weight.detach()
                    .cpu()
                    .to(torch.float32)
                    .numpy()
                    .T
                    for e in range(num_experts)
                ],
                axis=0,
            )
            hf_wi_1 = np.stack(
                [
                    layer_hf.mlp.experts[e]
                    .up_proj.weight.detach()
                    .cpu()
                    .to(torch.float32)
                    .numpy()
                    .T
                    for e in range(num_experts)
                ],
                axis=0,
            )
            hf_wo = np.stack(
                [
                    layer_hf.mlp.experts[e]
                    .down_proj.weight.detach()
                    .cpu()
                    .to(torch.float32)
                    .numpy()
                    .T
                    for e in range(num_experts)
                ],
                axis=0,
            )
        else:
            gate_up_fused = (
                layer_hf.mlp.experts.gate_up_proj.weight.detach().cpu().to(torch.float32).numpy()
            )
            num_experts = jax_wi_0.shape[0]
            intermediate_dim = jax_wi_0.shape[2]
            hidden_dim = jax_wi_0.shape[1]
            gate_up = gate_up_fused.reshape(num_experts, 2, intermediate_dim, hidden_dim)
            hf_wi_0 = gate_up[:, 0, :, :].transpose(0, 2, 1)
            hf_wi_1 = gate_up[:, 1, :, :].transpose(0, 2, 1)
            down_fused = (
                layer_hf.mlp.experts.down_proj.weight.detach().cpu().to(torch.float32).numpy()
            )
            hf_wo = down_fused.transpose(0, 2, 1)

        assert (
            hf_wi_0.shape == jax_wi_0.shape
        ), f"Layer {idx} wi_0 shape mismatch! HF: {hf_wi_0.shape}, JAX: {jax_wi_0.shape}"
        assert (
            hf_wi_1.shape == jax_wi_1.shape
        ), f"Layer {idx} wi_1 shape mismatch! HF: {hf_wi_1.shape}, JAX: {jax_wi_1.shape}"
        assert (
            hf_wo.shape == jax_wo.shape
        ), f"Layer {idx} wo shape mismatch! HF: {hf_wo.shape}, JAX: {jax_wo.shape}"

        wi_0_diff = np.max(np.abs(hf_wi_0 - jax_wi_0))
        wi_1_diff = np.max(np.abs(hf_wi_1 - jax_wi_1))
        wo_diff = np.max(np.abs(hf_wo - jax_wo))

        assert (
            wi_0_diff < 1e-5
        ), f"Layer {idx} wi_0 numerical discrepancy exceeded threshold: {wi_0_diff}"
        assert (
            wi_1_diff < 1e-5
        ), f"Layer {idx} wi_1 numerical discrepancy exceeded threshold: {wi_1_diff}"
        assert wo_diff < 1e-5, f"Layer {idx} wo numerical discrepancy exceeded threshold: {wo_diff}"

        logging.info(
            f"Layer {idx:02d} OK | max diffs -> QKV: {qkv_diff:.2e}, O: {o_diff:.2e}, Gate: {gate_diff:.2e}, wi_0: {wi_0_diff:.2e}, wi_1: {wi_1_diff:.2e}, wo: {wo_diff:.2e}"
        )

    logging.info("\n==============================================================")
    logging.info(
        "VALIDATION SUCCESSFUL: Converted JAX checkpoint is 100% mathematically aligned and verified!"
    )
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
