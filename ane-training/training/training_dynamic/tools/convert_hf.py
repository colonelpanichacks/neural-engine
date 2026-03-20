#!/usr/bin/env python3
"""Convert Hugging Face model weights to ANE training checkpoint format.

Usage:
    python tools/convert_hf.py Qwen/Qwen3-0.6B
    python tools/convert_hf.py Qwen/Qwen3-0.6B --output weights_qwen3.bin
    python tools/convert_hf.py meta-llama/Llama-2-7b-hf --config models/llama2_7b.h

Creates a full checkpoint (weights + zeroed Adam state) that ./train --resume can load.
"""
import argparse
import struct
import numpy as np
from pathlib import Path

def load_hf_weights(model_id):
    """Load weights from HF, return state dict + config."""
    try:
        from safetensors import safe_open
        from huggingface_hub import hf_hub_download, list_repo_files
        import json

        # Download config
        config_path = hf_hub_download(model_id, "config.json")
        with open(config_path) as f:
            config = json.load(f)

        # Find safetensors files
        files = list_repo_files(model_id)
        st_files = [f for f in files if f.endswith(".safetensors")]

        state_dict = {}
        for st_file in st_files:
            path = hf_hub_download(model_id, st_file)
            with safe_open(path, framework="numpy") as f:
                for key in f.keys():
                    state_dict[key] = f.get_tensor(key)

        return state_dict, config

    except ImportError:
        print("Install dependencies: pip install -r tools/requirements.txt")
        raise

def get_model_dims(config):
    """Extract dimensions from HF config."""
    dim = config["hidden_size"]
    hidden = config["intermediate_size"]
    heads = config["num_attention_heads"]
    kv_heads = config.get("num_key_value_heads", heads)
    hd = config.get("head_dim", dim // heads)
    n_layers = config["num_hidden_layers"]
    vocab = config["vocab_size"]
    q_dim = heads * hd
    kv_dim = kv_heads * hd
    return dim, hidden, heads, kv_heads, hd, n_layers, vocab, q_dim, kv_dim

def write_checkpoint(output_path, state_dict, config, seq=256, total_steps=10000):
    """Write weights as a full checkpoint with zeroed Adam states."""
    dim, hidden, heads, kv_heads, hd, n_layers, vocab, q_dim, kv_dim = get_model_dims(config)

    # Weight sizes
    wq_sz = q_dim * dim
    wk_sz = kv_dim * dim
    wv_sz = kv_dim * dim
    wo_sz = dim * q_dim
    w1_sz = hidden * dim
    w2_sz = dim * hidden
    w3_sz = hidden * dim

    print(f"Model: {config.get('model_type', 'unknown')}")
    print(f"  dim={dim} q_dim={q_dim} kv_dim={kv_dim} hd={hd} hidden={hidden}")
    print(f"  heads={heads} kv_heads={kv_heads} layers={n_layers} vocab={vocab}")

    # Weight name mapping (HF -> our layout)
    # Our layout: Wq[Q_DIM, DIM], Wk[KV_DIM, DIM], etc.
    # HF layout: q_proj.weight[Q_DIM, DIM], etc. (same!)
    def get_layer_weight(layer_idx, name):
        key = f"model.layers.{layer_idx}.{name}"
        if key not in state_dict:
            raise KeyError(f"Missing weight: {key}")
        return state_dict[key].astype(np.float32).flatten()

    # Check for QK-norm (Qwen3 has it, most models don't)
    has_qk_norm = f"model.layers.0.self_attn.q_norm.weight" in state_dict
    if has_qk_norm:
        print("  NOTE: Model has QK-norm (q_norm/k_norm). These are not used in")
        print("        our ANE pipeline and will be skipped. Training still works")
        print("        but fine-tuned results may differ slightly from HF reference.")

    # Check for tied embeddings
    if "lm_head.weight" in state_dict:
        embed = state_dict["lm_head.weight"].astype(np.float32).flatten()
        print("  Using lm_head.weight for embeddings")
    else:
        embed = state_dict["model.embed_tokens.weight"].astype(np.float32).flatten()
        print("  Using tied embed_tokens.weight for embeddings")

    rms_final = state_dict["model.norm.weight"].astype(np.float32).flatten()

    with open(output_path, "wb") as f:
        # CkptHdr (must match config.h struct exactly)
        # int magic, version, step, total_steps
        # int n_layers, vocab_size, dim, hidden_dim, n_heads, seq_len
        # float lr, loss
        # double cum_compile, cum_train, cum_wall
        # int cum_steps, cum_batches, adam_t
        # int kv_heads, head_dim, q_dim
        hdr = struct.pack(
            "iiii"       # magic, version, step, total_steps
            "iiiiii"     # n_layers, vocab_size, dim, hidden_dim, n_heads, seq_len
            "ff"         # lr, loss
            "ddd"        # cum_compile, cum_train, cum_wall
            "iii"        # cum_steps, cum_batches, adam_t
            "iii",       # kv_heads, head_dim, q_dim
            0x424C5A54, 4, 0, total_steps,
            n_layers, vocab, dim, hidden, heads, seq,
            3e-4, 0.0,
            0.0, 0.0, 0.0,
            0, 0, 0,
            kv_heads, hd, q_dim
        )
        f.write(hdr)

        # Per-layer weights + zeroed Adam states
        for L in range(n_layers):
            # Weights
            wq = get_layer_weight(L, "self_attn.q_proj.weight")
            wk = get_layer_weight(L, "self_attn.k_proj.weight")
            wv = get_layer_weight(L, "self_attn.v_proj.weight")
            wo = get_layer_weight(L, "self_attn.o_proj.weight")
            w1 = get_layer_weight(L, "mlp.gate_proj.weight")
            w2 = get_layer_weight(L, "mlp.down_proj.weight")
            w3 = get_layer_weight(L, "mlp.up_proj.weight")
            rms_att = get_layer_weight(L, "input_layernorm.weight")
            rms_ffn = get_layer_weight(L, "post_attention_layernorm.weight")

            assert wq.size == wq_sz, f"Layer {L} Wq size mismatch: {wq.size} vs {wq_sz}"
            assert wk.size == wk_sz, f"Layer {L} Wk size mismatch: {wk.size} vs {wk_sz}"

            f.write(wq.tobytes())
            f.write(wk.tobytes())
            f.write(wv.tobytes())
            f.write(wo.tobytes())
            f.write(w1.tobytes())
            f.write(w2.tobytes())
            f.write(w3.tobytes())
            f.write(rms_att.tobytes())
            f.write(rms_ffn.tobytes())

            # Adam states (m and v for each, all zeros)
            for sz in [wq_sz, wk_sz, wv_sz, wo_sz, w1_sz, w2_sz, w3_sz, dim, dim]:
                f.write(np.zeros(sz, dtype=np.float32).tobytes())  # m
                f.write(np.zeros(sz, dtype=np.float32).tobytes())  # v

            if (L + 1) % 7 == 0 or L == n_layers - 1:
                print(f"  Layer {L+1}/{n_layers} written")

        # rms_final + Adam state
        f.write(rms_final.tobytes())
        f.write(np.zeros(dim, dtype=np.float32).tobytes())  # m
        f.write(np.zeros(dim, dtype=np.float32).tobytes())  # v

        # embed + Adam state
        f.write(embed.tobytes())
        f.write(np.zeros(vocab * dim, dtype=np.float32).tobytes())  # m
        f.write(np.zeros(vocab * dim, dtype=np.float32).tobytes())  # v

    size_mb = Path(output_path).stat().st_size / 1e6
    print(f"\nCheckpoint saved: {output_path} ({size_mb:.1f} MB)")
    print(f"Load with: ./train --resume")

def main():
    parser = argparse.ArgumentParser(description="Convert HF model to ANE checkpoint")
    parser.add_argument("model_id", help="HF model ID (e.g. Qwen/Qwen3-0.6B)")
    parser.add_argument("--output", "-o", default=None,
                        help="Output checkpoint path (default: auto from model config)")
    parser.add_argument("--steps", type=int, default=10000,
                        help="Total training steps to set in checkpoint header")
    args = parser.parse_args()

    print(f"Downloading {args.model_id} from Hugging Face...")
    state_dict, config = load_hf_weights(args.model_id)
    print(f"Loaded {len(state_dict)} tensors")

    # Auto-detect output path from model type
    if args.output is None:
        model_type = config.get("model_type", "model")
        args.output = f"ane_{model_type}_dyn_ckpt.bin"

    write_checkpoint(args.output, state_dict, config, total_steps=args.steps)

if __name__ == "__main__":
    main()
