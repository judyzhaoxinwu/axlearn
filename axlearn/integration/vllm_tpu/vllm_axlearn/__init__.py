# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""vLLM-TPU integration."""

import ctypes
import os
import sys

from tpu_inference.logger import init_logger
from tpu_inference.models.common.model_loader import register_model

from .axlearn_model import AxLearnForCausalLM

logger = init_logger(__name__)


def register():
    logger.info("Registering AxLearnForCausalLM model with tpu_inference and vllm.")
    register_model("AxLearnForCausalLM", AxLearnForCausalLM)
    logger.info("Successfully registered AxLearnForCausalLM model.")


# 1. Protobuf symbols isolation monkeypatch
if hasattr(sys, "setdlopenflags") and hasattr(sys, "getdlopenflags"):
    _original_setdlopenflags = sys.setdlopenflags

    def _custom_setdlopenflags(flags):
        rtld_global = getattr(os, "RTLD_GLOBAL", getattr(ctypes, "RTLD_GLOBAL", 0))
        rtld_local = getattr(os, "RTLD_LOCAL", getattr(ctypes, "RTLD_LOCAL", 0))
        if flags & rtld_global:
            # Mask out RTLD_GLOBAL and apply RTLD_LOCAL
            flags = (flags & ~rtld_global) | rtld_local
        return _original_setdlopenflags(flags)

    sys.setdlopenflags = _custom_setdlopenflags

# 2. Dynamic sharding mesh axis injection to support MoE models locally on single device
try:
    import tpu_inference.layers.common.sharding as sharding_mod

    # Add fsdp, expert, and seq logical axes to MESH_AXIS_NAMES_2D
    sharding_mod.MESH_AXIS_NAMES_2D = ("data", "model", "fsdp", "expert", "seq")
    print(
        "[AxLearn Preloader] Successfully patched logical JAX mesh axis names with fsdp, expert, and seq!",
        file=sys.stderr,
    )
except ImportError:
    pass

# 3. Dynamic single-device sharding mesh dimensional and axis_types auto-aligner
try:
    import jax._src.mesh
    import numpy as np

    _original_mesh_new = jax._src.mesh.Mesh.__new__

    def _custom_mesh_new(cls, devices, axis_names, axis_types=None, *args, **kwargs):
        axis_names = list(axis_names)
        # If running locally on a single device (devices.size == 1), and the number of axis names
        # exceeds the dimensions of the devices array, dynamically align dimensions and axis_types!
        if len(axis_names) > devices.ndim and devices.size == 1:
            devices = np.array(devices).reshape((1,) * len(axis_names))
            if axis_types is not None:
                from jax.sharding import AxisType

                axis_types = list(axis_types) + [AxisType.Auto] * (
                    len(axis_names) - len(axis_types)
                )
                axis_types = tuple(axis_types)
        return _original_mesh_new(cls, devices, axis_names, axis_types, *args, **kwargs)

    jax._src.mesh.Mesh.__new__ = _custom_mesh_new
    print(
        "[AxLearn Preloader] Successfully installed JAX Mesh dimension and axis_types auto-aligner!",
        file=sys.stderr,
    )
except Exception as e:
    print(f"[AxLearn Preloader] Warning: Failed to install JAX Mesh reshaper: {e}", file=sys.stderr)
