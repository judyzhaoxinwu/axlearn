# Copyright © 2026 Apple Inc.


"""Utility script serializing Qwen MoE checkpoints into sharded JAX TensorStore arrays cleanly, fully aligned with JAX specs shapes."""

import copy
import gc
import json
import os

import jax
import jax.numpy as jnp
import numpy as np
import tensorflow as tf
import tensorstore as ts
from absl import app, flags, logging
from huggingface_hub import snapshot_download
from jax.experimental.array_serialization import serialization
from safetensors.torch import safe_open

from axlearn.common import utils
from axlearn.common.checkpointer import TensorStoreStateStorage, write_index_file
from axlearn.common.utils import TensorSpec
from axlearn.experiments.text.gpt import qwen

FLAGS = flags.FLAGS
flags.DEFINE_string(
    "model_id",
    "Qwen/Qwen3-30B-A3B-Instruct-2507",
    "Hugging Face model identifier repo path",
    required=False,
)
flags.DEFINE_string(
    "output_dir",
    None,
    "Destination GCS Bucket path or local folder for JAX checkpoints",
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


def download_hf_model(model_id: str) -> str:
    """Downloads PyTorch model checkpoints from Hugging Face Hub."""
    logging.info("Downloading Hugging Face model: %s", model_id)
    local_dir = snapshot_download(
        repo_id=model_id,
        ignore_patterns=["*.msgpack", "*.h5", "*.ot"],
    )
    logging.info("Model successfully downloaded locally to: %s", local_dir)
    return local_dir


def convert_and_save(local_pytorch_dir: str, output_dir: str):
    logging.info("Instantiating AXLearn model template for streaming conversion...")
    qwen_kwargs = qwen.get_trainer_kwargs(
        "30B-A3B",
        vocab_size=qwen.QWEN3_VOCAB_SIZE,
        batch_size=1024,
        max_sequence_length=1024,
    )
    qwen_kwargs["model_cfg"].set(name="model")
    qwen_model = qwen_kwargs["model_cfg"].instantiate(parent=None)
    empty_state_spec = jax.eval_shape(
        lambda: qwen_model.initialize_parameters_recursively(prng_key=jax.random.PRNGKey(0))
    )

    # Remap empty_state_spec from outer to inner to match the target serving layout (RoFormerQKVLinear)
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

    empty_state_spec = remap_outer_to_inner(empty_state_spec)

    empty_trainer_state = jax.tree.map(
        lambda x: TensorSpec(shape=x.shape, dtype=x.dtype), {"model": empty_state_spec}
    )

    storage = TensorStoreStateStorage(TensorStoreStateStorage.default_config())
    step_dir = os.path.join(output_dir, "step_00000000")
    tf.io.gfile.makedirs(step_dir)
    spec = storage._get_spec(step=0, state=empty_trainer_state, ckpt_dir=step_dir)

    dirs = sorted(list(set(os.path.dirname(path) for path in spec.storage_paths)))
    logging.info("Creating TensorStore directories on disk...")
    for d in dirs:
        tf.io.gfile.makedirs(d)
    logging.info("Writing checkpointer index file...")
    write_index_file(ckpt_dir=step_dir, index=spec.index)

    ts_specs = {}
    for storage_path, ts_spec in zip(spec.storage_paths, spec.tensorstore_specs):
        rel_path = storage_path.split("gda/model/")[1]
        ts_specs[rel_path] = ts_spec

    param_specs = {}
    for path, value in utils.flatten_items(empty_state_spec, separator="/"):
        param_specs[path] = value

    def open_ts(rel_path: str):
        ts_spec = copy.deepcopy(ts_specs[rel_path])
        spec_obj = param_specs[rel_path]
        if "dtype" not in ts_spec:
            ts_spec["dtype"] = jnp.dtype(spec_obj.dtype).name
        if "metadata" not in ts_spec:
            meta = serialization._get_metadata(spec_obj)
            shape = list(spec_obj.shape)
            if len(shape) == 4 and shape[0] == 48 and shape[1] == 128:
                meta["chunks"] = [1, 1, shape[2], shape[3]]
            elif len(shape) >= 2 and shape[0] == 48:
                meta["chunks"] = [1] + shape[1:]
            ts_spec["metadata"] = meta
        return ts.open(
            ts.Spec(ts_spec),
            create=True,
            open=True,
            context=serialization.TS_CONTEXT,
        ).result()

    logging.info("Reading Hugging Face safetensors index...")
    with open(
        os.path.join(local_pytorch_dir, "model.safetensors.index.json"),
        "r",
        encoding="utf-8",
    ) as f:
        index_json = json.load(f)
    weight_map = index_json["weight_map"]

    prefix = ""
    for k in weight_map:
        if k.startswith("thinker."):
            prefix = "thinker."
            break

    current_sf = {"name": None, "f": None}

    def get_hf(hf_name: str):
        full_name = f"{prefix}{hf_name}"
        if full_name not in weight_map:
            raise KeyError(f"Weight {full_name} not found in HF index weight map")
        sf_name = weight_map[full_name]
        if current_sf["name"] != sf_name:
            if current_sf["f"] is not None:
                current_sf["f"] = None
                gc.collect()
            sf_path = os.path.join(local_pytorch_dir, sf_name)
            current_sf["name"] = sf_name
            current_sf["f"] = safe_open(sf_path, framework="numpy", device="cpu")
        return current_sf["f"].get_tensor(full_name)

    vocab_size = qwen.QWEN3_VOCAB_SIZE

    logging.info("Streaming Embeddings and LM Head to SSD...")
    token_emb = get_hf("model.embed_tokens.weight")[:vocab_size, :].astype(np.float32)
    open_ts("decoder/emb/token_emb/weight").write(token_emb).result()
    del token_emb

    lm_head = get_hf("lm_head.weight")[:vocab_size, :].astype(np.float32)
    open_ts("decoder/lm_head/weight").write(lm_head).result()
    del lm_head

    output_norm = get_hf("model.norm.weight").astype(np.float32)
    open_ts("decoder/output_norm/scale").write(output_norm).result()
    del output_norm
    gc.collect()

    wi_0_shape = param_specs["decoder/transformer/repeat/layer/feed_forward/wi_0_weight"].shape
    num_layers = wi_0_shape[0]
    num_experts = wi_0_shape[1]
    hidden_dim = wi_0_shape[2]
    intermediate_dim = wi_0_shape[3]

    logging.info("Opening multi-layer TensorStore streaming datasets...")
    ts_input_norm = open_ts("decoder/transformer/repeat/layer/self_attention/norm/scale")
    ts_post_norm = open_ts("decoder/transformer/repeat/layer/feed_forward/norm/scale")
    ts_qkv = open_ts(
        "decoder/transformer/repeat/layer/self_attention/attention/i_proj/i_proj/qkv_proj/weight"
    )
    ts_o = open_ts("decoder/transformer/repeat/layer/self_attention/attention/o_proj/weight")

    # Write QK-Norm scales directly to the inner i_proj paths to natively
    # match our mathematically correct model execution structure (RoFormerQKVLinear).
    ts_scale_query = open_ts(
        "decoder/transformer/repeat/layer/self_attention/attention/i_proj/scale_query/norm/scale"
    )
    ts_scale_key = open_ts(
        "decoder/transformer/repeat/layer/self_attention/attention/i_proj/scale_key/norm/scale"
    )

    ts_moe_gate = open_ts("decoder/transformer/repeat/layer/feed_forward/gate_weight")
    ts_wi_0 = open_ts("decoder/transformer/repeat/layer/feed_forward/wi_0_weight")
    ts_wi_1 = open_ts("decoder/transformer/repeat/layer/feed_forward/wi_1_weight")
    ts_wo = open_ts("decoder/transformer/repeat/layer/feed_forward/wo_weight")

    is_module_list = f"{prefix}model.layers.0.mlp.experts.0.gate_proj.weight" in weight_map

    logging.info("Streaming 48 transformer layers one by one directly to SSD...")
    for i in range(num_layers):
        logging.info("Streaming Layer %d/%d to TensorStore...", i + 1, num_layers)

        ts_input_norm[i].write(
            get_hf(f"model.layers.{i}.input_layernorm.weight").astype(np.float32)
        ).result()
        ts_post_norm[i].write(
            get_hf(f"model.layers.{i}.post_attention_layernorm.weight").astype(np.float32)
        ).result()

        q = get_hf(f"model.layers.{i}.self_attn.q_proj.weight").astype(np.float32)
        k = get_hf(f"model.layers.{i}.self_attn.k_proj.weight").astype(np.float32)
        v = get_hf(f"model.layers.{i}.self_attn.v_proj.weight").astype(np.float32)

        q = q.reshape(32, 128, hidden_dim)
        k = k.reshape(4, 128, hidden_dim)
        v = v.reshape(4, 128, hidden_dim)

        # Permute Q and K weights from Hugging Face split-half RoPE format
        # to AxLearn interleaved RoPE format before writing to the sharded TensorStore checkpoint.
        q_permuted = permute_q_k_for_rope_numpy(q)
        k_permuted = permute_q_k_for_rope_numpy(k)

        qkv_layer = np.concatenate([q_permuted, k_permuted, v], axis=0)
        ts_qkv[i].write(qkv_layer.transpose(2, 0, 1)).result()
        del q, k, v, q_permuted, k_permuted, qkv_layer

        o = get_hf(f"model.layers.{i}.self_attn.o_proj.weight").astype(np.float32)
        ts_o[i].write(o.reshape(hidden_dim, 32, 128)).result()
        del o

        ts_scale_query[i].write(
            get_hf(f"model.layers.{i}.self_attn.q_norm.weight").astype(np.float32)
        ).result()
        ts_scale_key[i].write(
            get_hf(f"model.layers.{i}.self_attn.k_norm.weight").astype(np.float32)
        ).result()

        # MoE Experts
        ts_moe_gate[i].write(
            get_hf(f"model.layers.{i}.mlp.gate.weight").astype(np.float32).T
        ).result()

        if is_module_list:
            for e in range(num_experts):
                ts_wi_0[i, e].write(
                    get_hf(f"model.layers.{i}.mlp.experts.{e}.gate_proj.weight")
                    .astype(np.float32)
                    .T
                ).result()
                ts_wi_1[i, e].write(
                    get_hf(f"model.layers.{i}.mlp.experts.{e}.up_proj.weight").astype(np.float32).T
                ).result()
                ts_wo[i, e].write(
                    get_hf(f"model.layers.{i}.mlp.experts.{e}.down_proj.weight")
                    .astype(np.float32)
                    .T
                ).result()
        else:
            gate_up_fused = get_hf(f"model.layers.{i}.mlp.experts.gate_up_proj.weight").astype(
                np.float32
            )
            gate_up = gate_up_fused.reshape(num_experts, 2, intermediate_dim, hidden_dim)

            ts_wi_0[i].write(gate_up[:, 0, :, :].transpose(0, 2, 1)).result()
            ts_wi_1[i].write(gate_up[:, 1, :, :].transpose(0, 2, 1)).result()

            down_fused = get_hf(f"model.layers.{i}.mlp.experts.down_proj.weight").astype(np.float32)
            ts_wo[i].write(down_fused.transpose(0, 2, 1)).result()
            del gate_up_fused, gate_up, down_fused

        gc.collect()

    logging.info("Streaming conversion successfully completed!")
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
