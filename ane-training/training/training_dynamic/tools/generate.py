#!/usr/bin/env python3
"""Generate text from an ANE training checkpoint using a HF tokenizer.

Usage:
    python tools/generate.py ane_qwen3_dyn_ckpt.bin --model Qwen/Qwen3-0.6B
    python tools/generate.py ane_qwen3_dyn_ckpt.bin --model Qwen/Qwen3-0.6B --prompt "Once upon a time"
    python tools/generate.py ane_qwen3_dyn_ckpt.bin --model Qwen/Qwen3-0.6B --tokens 200 --temp 0.8

Pure numpy inference — no torch, no ANE. Slow but useful for verifying training worked.
"""
import argparse
import struct
import sys
import numpy as np
from pathlib import Path


def read_checkpoint(path):
    """Read checkpoint, return header info + weight arrays."""
    with open(path, "rb") as f:
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
        gqa_ratio = heads // kv_heads
        wq_sz = q_dim * dim
        wk_sz = kv_dim * dim
        wv_sz = kv_dim * dim
        wo_sz = dim * q_dim
        w1_sz = hidden * dim
        w2_sz = dim * hidden
        w3_sz = hidden * dim

        info = {
            "step": step, "loss": loss, "dim": dim, "hidden": hidden,
            "heads": heads, "kv_heads": kv_heads, "hd": hd,
            "q_dim": q_dim, "kv_dim": kv_dim, "gqa_ratio": gqa_ratio,
            "n_layers": n_layers, "vocab": vocab,
        }

        print(f"Checkpoint: step={step} loss={loss:.4f}")
        print(f"  dim={dim} q_dim={q_dim} kv_dim={kv_dim} hidden={hidden}")
        print(f"  heads={heads} kv_heads={kv_heads} layers={n_layers} vocab={vocab}")

        layers = []
        for L in range(n_layers):
            layer = {}
            layer["Wq"] = np.frombuffer(f.read(wq_sz * 4), dtype=np.float32).reshape(q_dim, dim).copy()
            layer["Wk"] = np.frombuffer(f.read(wk_sz * 4), dtype=np.float32).reshape(kv_dim, dim).copy()
            layer["Wv"] = np.frombuffer(f.read(wv_sz * 4), dtype=np.float32).reshape(kv_dim, dim).copy()
            layer["Wo"] = np.frombuffer(f.read(wo_sz * 4), dtype=np.float32).reshape(dim, q_dim).copy()
            layer["W1"] = np.frombuffer(f.read(w1_sz * 4), dtype=np.float32).reshape(hidden, dim).copy()
            layer["W2"] = np.frombuffer(f.read(w2_sz * 4), dtype=np.float32).reshape(dim, hidden).copy()
            layer["W3"] = np.frombuffer(f.read(w3_sz * 4), dtype=np.float32).reshape(hidden, dim).copy()
            layer["rms_att"] = np.frombuffer(f.read(dim * 4), dtype=np.float32).copy()
            layer["rms_ffn"] = np.frombuffer(f.read(dim * 4), dtype=np.float32).copy()

            # Skip Adam states
            for sz in [wq_sz, wk_sz, wv_sz, wo_sz, w1_sz, w2_sz, w3_sz, dim, dim]:
                f.read(sz * 4 * 2)

            layers.append(layer)

        rms_final = np.frombuffer(f.read(dim * 4), dtype=np.float32).copy()
        f.read(dim * 4 * 2)  # Adam state

        embed = np.frombuffer(f.read(vocab * dim * 4), dtype=np.float32).reshape(vocab, dim).copy()

    return info, layers, rms_final, embed


def rmsnorm(x, w, eps=1e-5):
    """RMSNorm: x is [dim], w is [dim]."""
    rms = np.sqrt(np.mean(x ** 2) + eps)
    return (x / rms) * w


def softmax(x):
    x = x - np.max(x)
    e = np.exp(x)
    return e / np.sum(e)


def silu(x):
    return x * (1.0 / (1.0 + np.exp(-x)))


def rope(x, pos, hd):
    """Apply RoPE to x[dim] at position pos. dim = n_heads * hd."""
    out = x.copy()
    n_heads = len(x) // hd
    for h in range(n_heads):
        for i in range(hd // 2):
            freq = 1.0 / (10000.0 ** (2.0 * i / hd))
            theta = pos * freq
            cos_t, sin_t = np.cos(theta), np.sin(theta)
            idx = h * hd + 2 * i
            v0, v1 = out[idx], out[idx + 1]
            out[idx] = v0 * cos_t - v1 * sin_t
            out[idx + 1] = v0 * sin_t + v1 * cos_t
    return out


def generate(info, layers, rms_final, embed, tokenizer, prompt_tokens, max_tokens, temperature, top_p):
    """Autoregressive generation with KV cache."""
    dim = info["dim"]
    heads = info["heads"]
    kv_heads = info["kv_heads"]
    hd = info["hd"]
    q_dim = info["q_dim"]
    kv_dim = info["kv_dim"]
    gqa_ratio = info["gqa_ratio"]
    hidden = info["hidden"]
    n_layers = info["n_layers"]
    vocab = info["vocab"]

    # KV cache: [n_layers, kv_heads, max_seq, hd]
    max_seq = len(prompt_tokens) + max_tokens
    k_cache = [np.zeros((kv_heads, max_seq, hd), dtype=np.float32) for _ in range(n_layers)]
    v_cache = [np.zeros((kv_heads, max_seq, hd), dtype=np.float32) for _ in range(n_layers)]

    tokens = list(prompt_tokens)
    generated = []

    for pos in range(len(tokens) + max_tokens - 1):
        # Get current token
        if pos < len(tokens):
            tok = tokens[pos]
        else:
            tok = next_tok

        # Embed
        x = embed[tok].copy()  # [dim]

        # Transformer layers
        for L in range(n_layers):
            layer = layers[L]

            # Self-attention
            xn = rmsnorm(x, layer["rms_att"])

            q = layer["Wq"] @ xn   # [q_dim]
            k = layer["Wk"] @ xn   # [kv_dim]
            v = layer["Wv"] @ xn   # [kv_dim]

            # RoPE
            q = rope(q, pos, hd)
            k = rope(k, pos, hd)

            # Store KV
            for h in range(kv_heads):
                k_cache[L][h, pos] = k[h * hd:(h + 1) * hd]
                v_cache[L][h, pos] = v[h * hd:(h + 1) * hd]

            # Attention (GQA)
            attn_out = np.zeros(q_dim, dtype=np.float32)
            scale = 1.0 / np.sqrt(hd)
            for qh in range(heads):
                kvh = qh // gqa_ratio
                q_head = q[qh * hd:(qh + 1) * hd]

                # scores against all cached keys up to pos
                scores = np.zeros(pos + 1, dtype=np.float32)
                for t in range(pos + 1):
                    scores[t] = np.dot(q_head, k_cache[L][kvh, t]) * scale

                attn_weights = softmax(scores)
                head_out = np.zeros(hd, dtype=np.float32)
                for t in range(pos + 1):
                    head_out += attn_weights[t] * v_cache[L][kvh, t]

                attn_out[qh * hd:(qh + 1) * hd] = head_out

            # Output projection + residual
            x = x + layer["Wo"] @ attn_out

            # FFN (SwiGLU)
            xn = rmsnorm(x, layer["rms_ffn"])
            h1 = layer["W1"] @ xn   # gate
            h3 = layer["W3"] @ xn   # up
            h1 = silu(h1) * h3
            x = x + layer["W2"] @ h1

        # Final norm + logits
        x = rmsnorm(x, rms_final)
        logits = embed @ x  # tied embeddings: [vocab]

        # Only sample after prompt is processed
        if pos >= len(tokens) - 1:
            if temperature < 1e-6:
                next_tok = int(np.argmax(logits))
            else:
                logits = logits / temperature
                probs = softmax(logits)

                # Top-p sampling
                if top_p < 1.0:
                    sorted_idx = np.argsort(-probs)
                    cum = np.cumsum(probs[sorted_idx])
                    cutoff = np.searchsorted(cum, top_p) + 1
                    mask = np.zeros(vocab, dtype=np.float32)
                    mask[sorted_idx[:cutoff]] = probs[sorted_idx[:cutoff]]
                    mask /= mask.sum()
                    probs = mask

                next_tok = int(np.random.choice(vocab, p=probs))

            generated.append(next_tok)

            # Decode and print token
            text = tokenizer.decode([next_tok])
            print(text, end="", flush=True)

            # Stop on EOS
            if next_tok == tokenizer.eos_token_id:
                break

    print()
    return generated


def main():
    parser = argparse.ArgumentParser(description="Generate text from ANE checkpoint")
    parser.add_argument("checkpoint", help="Path to ANE checkpoint (.bin)")
    parser.add_argument("--model", "-m", required=True,
                        help="HF model ID for tokenizer (e.g. Qwen/Qwen3-0.6B)")
    parser.add_argument("--prompt", "-p", default="Once upon a time",
                        help="Text prompt (default: 'Once upon a time')")
    parser.add_argument("--tokens", "-n", type=int, default=100,
                        help="Max tokens to generate (default: 100)")
    parser.add_argument("--temp", "-t", type=float, default=0.8,
                        help="Sampling temperature (default: 0.8, 0=greedy)")
    parser.add_argument("--top-p", type=float, default=0.9,
                        help="Top-p nucleus sampling (default: 0.9)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility")
    args = parser.parse_args()

    if args.seed is not None:
        np.random.seed(args.seed)

    try:
        from transformers import AutoTokenizer
    except ImportError:
        print("Install dependencies: pip install -r tools/requirements.txt")
        sys.exit(1)

    # Load checkpoint
    info, layers, rms_final, embed = read_checkpoint(args.checkpoint)

    # Load tokenizer
    print(f"Loading tokenizer: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    # Tokenize prompt
    prompt_tokens = tokenizer.encode(args.prompt)
    print(f"Prompt ({len(prompt_tokens)} tokens): {args.prompt}")
    print(f"Generating {args.tokens} tokens (temp={args.temp}, top_p={args.top_p})...")
    print("---")

    generated = generate(info, layers, rms_final, embed, tokenizer,
                        prompt_tokens, args.tokens, args.temp, args.top_p)

    print("---")
    print(f"Generated {len(generated)} tokens")


if __name__ == "__main__":
    main()
