#!/usr/bin/env python3
"""Tokenize text files into binary format for ANE training.

Usage:
    python tools/tokenize_data.py data/*.txt --model Qwen/Qwen3-0.6B
    python tools/tokenize_data.py mybook.txt --model Qwen/Qwen3-0.6B --output my_data.bin
    cat corpus.txt | python tools/tokenize_data.py - --model Qwen/Qwen3-0.6B

Output is a flat binary file of uint16 token IDs, ready for ./train --data.
"""
import argparse
import sys
import numpy as np
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Tokenize text for ANE training")
    parser.add_argument("inputs", nargs="+", help="Text files to tokenize (use - for stdin)")
    parser.add_argument("--model", "-m", required=True,
                        help="HF model ID for tokenizer (e.g. Qwen/Qwen3-0.6B)")
    parser.add_argument("--output", "-o", default="train_data.bin",
                        help="Output binary file (default: train_data.bin)")
    parser.add_argument("--max-tokens", type=int, default=None,
                        help="Maximum number of tokens to output")
    args = parser.parse_args()

    try:
        from transformers import AutoTokenizer
    except ImportError:
        print("Install dependencies: pip install -r tools/requirements.txt")
        sys.exit(1)

    print(f"Loading tokenizer: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    print(f"Vocab size: {tokenizer.vocab_size}")

    # Collect all text
    all_text = []
    for inp in args.inputs:
        if inp == "-":
            print("Reading from stdin...")
            all_text.append(sys.stdin.read())
        else:
            path = Path(inp)
            if not path.exists():
                print(f"Warning: {inp} not found, skipping")
                continue
            print(f"Reading {inp} ({path.stat().st_size / 1e6:.1f} MB)")
            all_text.append(path.read_text(encoding="utf-8", errors="ignore"))

    if not all_text:
        print("No input text found")
        sys.exit(1)

    combined = "\n\n".join(all_text)
    print(f"Total text: {len(combined):,} characters")

    # Tokenize
    print("Tokenizing...")
    tokens = tokenizer.encode(combined)
    print(f"Total tokens: {len(tokens):,}")

    if args.max_tokens and len(tokens) > args.max_tokens:
        tokens = tokens[:args.max_tokens]
        print(f"Truncated to {len(tokens):,} tokens")

    # Check token range fits uint16
    max_tok = max(tokens)
    if max_tok > 65535:
        print(f"ERROR: Token ID {max_tok} exceeds uint16 range. This tokenizer is not supported.")
        sys.exit(1)

    # Write binary
    arr = np.array(tokens, dtype=np.uint16)
    arr.tofile(args.output)
    size_mb = Path(args.output).stat().st_size / 1e6
    print(f"\nSaved: {args.output} ({len(tokens):,} tokens, {size_mb:.1f} MB)")
    print(f"Train with: ./train --resume --data {args.output}")

if __name__ == "__main__":
    main()
