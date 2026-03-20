#!/usr/bin/env python3
"""Export ANE training checkpoint back to Hugging Face safetensors format.

Usage:
    python tools/export_hf.py ane_qwen3_dyn_ckpt.bin --model Qwen/Qwen3-0.6B --output ./my_model
    python tools/export_hf.py ane_qwen3_dyn_ckpt.bin --model Qwen/Qwen3-0.6B --output ./my_model --push user/my-model

The exported model can be loaded with transformers, MLX, or converted to GGUF for llama.cpp.
"""
import argparse
import struct
import json
import shutil
import numpy as np
from pathlib import Path

def read_checkpoint(path):
    """Read our checkpoint format, return header + weight arrays."""
    with open(path, "rb") as f:
        # CkptHdr
        hdr_fmt = "iiii" "iiiiii" "ff" "ddd" "iii" "iii"
        hdr_size = struct.calcsize(hdr_fmt)
        hdr_data = struct.unpack(hdr_fmt, f.read(hdr_size))

        magic, version, step, total_steps = hdr_data[0:4]
        n_layers, vocab, dim, hidden, heads, seq = hdr_data[4:10]
        lr, loss = hdr_data[10:12]
        kv_heads, hd, q_dim = hdr_data[15:18]

        assert magic == 0x424C5A54, f"Bad magic: {hex(magic)}"
        assert version == 4, f"Unsupported version: {version}"

        kv_dim = kv_heads * hd
        wq_sz = q_dim * dim
        wk_sz = kv_dim * dim
        wv_sz = kv_dim * dim
        wo_sz = dim * q_dim
        w1_sz = hidden * dim
        w2_sz = dim * hidden
        w3_sz = hidden * dim

        info = {
            "step": step, "loss": loss, "lr": lr,
            "dim": dim, "hidden": hidden, "heads": heads,
            "kv_heads": kv_heads, "hd": hd, "q_dim": q_dim,
            "kv_dim": kv_dim, "n_layers": n_layers, "vocab": vocab,
        }

        print(f"Checkpoint: step={step} loss={loss:.4f}")
        print(f"  dim={dim} q_dim={q_dim} kv_dim={kv_dim} hidden={hidden}")
        print(f"  heads={heads} kv_heads={kv_heads} layers={n_layers} vocab={vocab}")

        layers = []
        for L in range(n_layers):
            layer = {}
            layer["Wq"] = np.frombuffer(f.read(wq_sz * 4), dtype=np.float32).reshape(q_dim, dim)
            layer["Wk"] = np.frombuffer(f.read(wk_sz * 4), dtype=np.float32).reshape(kv_dim, dim)
            layer["Wv"] = np.frombuffer(f.read(wv_sz * 4), dtype=np.float32).reshape(kv_dim, dim)
            layer["Wo"] = np.frombuffer(f.read(wo_sz * 4), dtype=np.float32).reshape(dim, q_dim)
            layer["W1"] = np.frombuffer(f.read(w1_sz * 4), dtype=np.float32).reshape(hidden, dim)
            layer["W2"] = np.frombuffer(f.read(w2_sz * 4), dtype=np.float32).reshape(dim, hidden)
            layer["W3"] = np.frombuffer(f.read(w3_sz * 4), dtype=np.float32).reshape(hidden, dim)
            layer["rms_att"] = np.frombuffer(f.read(dim * 4), dtype=np.float32).copy()
            layer["rms_ffn"] = np.frombuffer(f.read(dim * 4), dtype=np.float32).copy()

            # Skip Adam states (m and v for each weight)
            for sz in [wq_sz, wk_sz, wv_sz, wo_sz, w1_sz, w2_sz, w3_sz, dim, dim]:
                f.read(sz * 4 * 2)  # m + v

            layers.append(layer)

        rms_final = np.frombuffer(f.read(dim * 4), dtype=np.float32).copy()
        f.read(dim * 4 * 2)  # Adam state for rms_final

        embed = np.frombuffer(f.read(vocab * dim * 4), dtype=np.float32).reshape(vocab, dim)

    return info, layers, rms_final, embed

def export_safetensors(output_dir, info, layers, rms_final, embed, model_id=None):
    """Write HF-compatible safetensors + config."""
    try:
        from safetensors.numpy import save_file
    except ImportError:
        print("Install dependencies: pip install -r tools/requirements.txt")
        raise

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build state dict
    tensors = {}
    tensors["model.embed_tokens.weight"] = embed.copy()
    tensors["model.norm.weight"] = rms_final.copy()

    for L, layer in enumerate(layers):
        prefix = f"model.layers.{L}"
        tensors[f"{prefix}.self_attn.q_proj.weight"] = layer["Wq"].copy()
        tensors[f"{prefix}.self_attn.k_proj.weight"] = layer["Wk"].copy()
        tensors[f"{prefix}.self_attn.v_proj.weight"] = layer["Wv"].copy()
        tensors[f"{prefix}.self_attn.o_proj.weight"] = layer["Wo"].copy()
        tensors[f"{prefix}.mlp.gate_proj.weight"] = layer["W1"].copy()
        tensors[f"{prefix}.mlp.down_proj.weight"] = layer["W2"].copy()
        tensors[f"{prefix}.mlp.up_proj.weight"] = layer["W3"].copy()
        tensors[f"{prefix}.input_layernorm.weight"] = layer["rms_att"].copy()
        tensors[f"{prefix}.post_attention_layernorm.weight"] = layer["rms_ffn"].copy()

    # Save safetensors
    st_path = output_dir / "model.safetensors"
    save_file(tensors, str(st_path))
    print(f"Saved: {st_path} ({st_path.stat().st_size / 1e6:.1f} MB)")

    # Copy config/tokenizer from original model if provided
    if model_id:
        try:
            from huggingface_hub import hf_hub_download, list_repo_files
            files_to_copy = [
                "config.json", "tokenizer.json", "tokenizer_config.json",
                "special_tokens_map.json", "vocab.json", "merges.txt",
                "generation_config.json",
            ]
            repo_files = list_repo_files(model_id)
            for fname in files_to_copy:
                if fname in repo_files:
                    src = hf_hub_download(model_id, fname)
                    shutil.copy(src, output_dir / fname)
            print(f"Copied config + tokenizer from {model_id}")
        except Exception as e:
            print(f"Warning: Could not copy config/tokenizer: {e}")
            write_minimal_config(output_dir, info)
    else:
        write_minimal_config(output_dir, info)

    print(f"\nModel exported to: {output_dir}/")
    print(f"Use with transformers:")
    print(f"  from transformers import AutoModelForCausalLM, AutoTokenizer")
    print(f'  model = AutoModelForCausalLM.from_pretrained("{output_dir}")')
    print(f"\nConvert to GGUF for llama.cpp:")
    print(f"  python llama.cpp/convert_hf_to_gguf.py {output_dir}")

def write_minimal_config(output_dir, info):
    """Write a minimal HF config.json."""
    config = {
        "architectures": ["Qwen3ForCausalLM"],
        "model_type": "qwen3",
        "hidden_size": info["dim"],
        "intermediate_size": info["hidden"],
        "num_attention_heads": info["heads"],
        "num_key_value_heads": info["kv_heads"],
        "head_dim": info["hd"],
        "num_hidden_layers": info["n_layers"],
        "vocab_size": info["vocab"],
        "max_position_embeddings": 32768,
        "rms_norm_eps": 1e-5,
        "tie_word_embeddings": True,
        "torch_dtype": "float32",
    }
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

def main():
    parser = argparse.ArgumentParser(description="Export ANE checkpoint to HF format")
    parser.add_argument("checkpoint", help="Path to ANE checkpoint (.bin)")
    parser.add_argument("--model", "-m", default=None,
                        help="Original HF model ID (copies config + tokenizer)")
    parser.add_argument("--output", "-o", required=True,
                        help="Output directory for HF model")
    parser.add_argument("--push", default=None,
                        help="Push to HF Hub (e.g. username/model-name)")
    args = parser.parse_args()

    info, layers, rms_final, embed = read_checkpoint(args.checkpoint)
    export_safetensors(args.output, info, layers, rms_final, embed, args.model)

    if args.push:
        try:
            from huggingface_hub import HfApi
            api = HfApi()
            api.create_repo(args.push, private=True, exist_ok=True)
            api.upload_folder(folder_path=args.output, repo_id=args.push)
            print(f"\nPushed to: https://huggingface.co/{args.push}")
        except Exception as e:
            print(f"Push failed: {e}")

if __name__ == "__main__":
    main()
