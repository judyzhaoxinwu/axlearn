# vLLM-TPU AxLearn Plugin Adapter

This directory provides an out-of-tree plugin to integrate **AxLearn** models with **vLLM / TPU Inference** (`tpu_inference`).

By using vLLM's `vllm.general_plugins` entrypoint system, this plugin registers `AxLearnForCausalLM` without requiring any modifications to the upstream `tpu-inference` repository.

## Directory Structure

```
axlearn/integration/vllm_tpu/
├── setup.py          # Plugin package setup script
├── README.md         # Documentation
└── vllm_axlearn/     # Python package module
    ├── __init__.py   # Entrypoint & runtime compatibility patches
    └── axlearn_model.py # AxLearnForCausalLM model wrapper
```

## Installation

To install the plugin into your Python environment:

```bash
pip install -e axlearn/integration/vllm_tpu --no-deps
```

## Dockerfile Usage

When building a vLLM TPU container image:

```dockerfile
# 1. Install AxLearn core
WORKDIR /root/axlearn
RUN uv pip install -e ".[core,tpu]" --find-links https://storage.googleapis.com/axlearn-wheels/wheels.html --system

# 2. Install out-of-tree plugin
RUN uv pip install -e axlearn/integration/vllm_tpu --no-deps --system

# 3. Install generic tpu-inference
WORKDIR /root/tpu-inference
RUN uv pip install -e . --system
```
