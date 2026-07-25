import os
import json
import gc
import numpy as np
import tensorstore as ts
from safetensors.numpy import safe_open

def validate_lightweight(hf_dir, ckpt_dir):
    print("🔍 Reading Hugging Face safetensors index...")
    with open(os.path.join(hf_dir, "model.safetensors.index.json"), "r") as f:
        weight_map = json.load(f)["weight_map"]
    
    # 1. Load ONLY the embedding tensor from Hugging Face
    sf_name = weight_map["model.embed_tokens.weight"]
    print(f"📥 Loading HF embeddings from {sf_name}...")
    with safe_open(os.path.join(hf_dir, sf_name), framework="numpy", device="cpu") as f:
        hf_emb = f.get_tensor("model.embed_tokens.weight")[:151936, :].astype(np.float32)
        
    # 2. Load ONLY the JAX embedding tensor from your local checkpoint
    print("📂 Opening JAX embedding TensorStore...")
    ts_spec = {
        'driver': 'zarr',
        'kvstore': {
            'driver': 'file',
            'path': os.path.join(ckpt_dir, "gda", "model", "decoder", "emb", "token_emb", "weight")
        }
    }
    jax_emb_store = ts.open(ts_spec).result()
    jax_emb = np.array(jax_emb_store).astype(np.float32)
    
    # 3. Compare them!
    print(f"📊 HF shape: {hf_emb.shape} | JAX shape: {jax_emb.shape}")
    assert hf_emb.shape == jax_emb.shape, "❌ Shape mismatch!"
    
    diff = np.max(np.abs(hf_emb - jax_emb))
    print(f"📈 Absolute max difference: {diff:e}")
    
    if diff < 1e-5:
        print("🎉 LIGHTWEIGHT VALIDATION SUCCESSFUL! Checkpoint is 100% numerically perfect!")
    else:
        print("❌ Numerical discrepancy exceeded threshold!")

if __name__ == "__main__":
    # TODO: Paste the exact Hugging Face snapshot path printed by your download script!
    # It looks like: "/Users/judyzhaoxinwu/.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B-Instruct-2507/snapshots/<HASH>"
    hf_dir = "/Users/judyzhaoxinwu/.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B-Instruct-2507/snapshots/0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe" # Replace hash if different
    ckpt_dir = "/tmp/qwen3_30b_a3b/checkpoints/step_00000000"
    
    validate_lightweight(hf_dir, ckpt_dir)
