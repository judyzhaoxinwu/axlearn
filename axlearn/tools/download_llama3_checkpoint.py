# Copyright © 2026 Apple Inc.

"""Utility script downloading Hugging Face checkpoints, converting weights to Fuji JAX structures, and saving to GCS."""

import copy
import gc
import os
import threading
import time

import jax
import jax.numpy as jnp
import numpy as np
import tensorflow as tf
import torch
from absl import app, flags, logging
from huggingface_hub import snapshot_download
from transformers import LlamaForCausalLM

# Monkey-patch axlearn's as_tensor recursively to support PyTorch BFloat16 conversion!
from axlearn.common import param_converter
from axlearn.common import utils as axlearn_utils
from axlearn.common.checkpointer import TensorStoreStateStorage
from axlearn.experiments.text.gpt import fuji

_orig_as_tensor = axlearn_utils.as_tensor


def _patched_as_tensor(x):
    if isinstance(x, torch.Tensor) and x.dtype == torch.bfloat16:
        return jnp.asarray(x.detach().cpu().to(torch.float32).numpy())
    return _orig_as_tensor(x)


axlearn_utils.as_tensor = _patched_as_tensor

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "model_id", "meta-llama/Llama-3.2-1B", "Hugging Face model identifier repo path", required=False
)

flags.DEFINE_string(
    "model_size", "1B", "Fuji target model size (test, 1B, 3B, 7B, 70B)", required=False
)
flags.DEFINE_string(
    "output_dir",
    None,
    "Destination GCS Bucket path or local write folder for JAX checkpoint",
    required=True,
)


def download_hf_model(model_id: str) -> str:
    """Downloads PyTorch model checkpoints from Hugging Face Hub."""
    logging.info("Starting download for Hugging Face model: %s", model_id)
    local_dir = snapshot_download(
        repo_id=model_id,
        ignore_patterns=["*.msgpack", "*.h5", "*.ot"],
    )
    logging.info("Model checkpoint successfully downloaded locally to: %s", local_dir)
    return local_dir


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


def convert_and_save(local_pytorch_dir: str, model_size: str, output_dir: str):
    logging.info("Instantiating Hugging Face PyTorch Llama model...")
    # Load weights in bfloat16 to reduce memory footprint from 120GB to 60GB (OOM safety)
    llama_model = LlamaForCausalLM.from_pretrained(
        local_pytorch_dir, torch_dtype=torch.bfloat16, device_map="cpu"
    )

    logging.info("Instantiating matching AXLearn Fuji-%s JAX model specs...", model_size)
    fuji_kwargs = fuji.get_trainer_kwargs(
        model_size,
        vocab_size=(
            128256 if "Llama-3" in FLAGS.model_id else 32000
        ),  # Enforce correct Llama-3 vocab size (128256) instead of 131072
        version=fuji.Version.V3_TIKTOKEN,  # V3_TIKTOKEN natively registers the Llama-3 8B configuration keys!
    )

    fuji_model_cfg = fuji_kwargs["model_cfg"]
    fuji_model_cfg.set(name="model")  # Explicitly set name to satisfy config rules
    fuji_model = fuji_model_cfg.instantiate(parent=None)

    # Initialize empty structural parameters state spec (correct PRNGKey spelling)
    dummy_prng = jax.random.PRNGKey(0)
    empty_state_spec = fuji_model.initialize_parameters_recursively(prng_key=dummy_prng)

    logging.info("Mapping PyTorch parameters into sharded JAX structures...")
    # Map Torch weights natively to JAX parameters dict
    jax_converted_params = param_converter.parameters_from_llama_3(llama_model, empty_state_spec)

    logging.info("Deleting PyTorch model and clearing garbage collector...")
    del llama_model
    gc.collect()

    logging.info("Serializing sharded JAX parameters to GCS output checkpoint: %s", output_dir)

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

    # AXLearn checkpoint structure requires grouping inside 'model' scope
    trainer_state = {"model": jax_converted_params}

    # Start progress monitoring thread in background (Dynamically count exact GCS sharded files!)
    total_expected_files = len(jax.tree_util.tree_leaves(jax_converted_params)) * 2
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
        storage._manager.wait_until_finished()  # <--- Force main thread to block until all 70B weights are written!
    finally:
        stop_event.set()
        monitor_thread.join(timeout=2)

    logging.info("Weight conversion successfully completed and saved to GCS.")


def main(_):
    # Reshape flat device array to match ("data", "model") axis names
    devices = np.array(jax.devices()).reshape(-1, 1)
    mesh = jax.sharding.Mesh(
        devices=devices,
        axis_names=("data", "model"),
    )

    with mesh:
        local_pytorch_dir = download_hf_model(FLAGS.model_id)
        convert_and_save(local_pytorch_dir, FLAGS.model_size, FLAGS.output_dir)


if __name__ == "__main__":
    app.run(main)
