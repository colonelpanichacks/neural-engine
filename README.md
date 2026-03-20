# Neural Engine

Training transformers directly on Apple's Neural Engine via reverse-engineered private APIs, with hand-written ARM SME2 assembly kernels for Apple Silicon.

## Overview

This research project explores pushing Apple Silicon's Neural Engine (ANE) beyond its inference-only CoreML limitations. The ANE is a 16-core ML accelerator capable of 18.5 TOPS (fp16) that Apple restricts to inference. This project bypasses that restriction to run full transformer training (forward + backward + optimizer) on ANE hardware.

### Components

**ANE Training Pipeline** (`ane-training/training/training_dynamic/`) — Full transformer training loop on ANE:
- Dynamic weight kernels via MIL (Machine Learning Intermediate Language) — weights update without kernel recompilation
- GQA (Grouped Query Attention) support for modern architectures
- Async CPU/ANE overlap — CPU gradient accumulation runs concurrently with ANE backward pass
- fp16 activation storage — eliminates unnecessary fp16-fp32-fp16 roundtrip conversions between ANE kernels
- Per-layer IOSurface weight staging for zero-copy ANE dispatch
- Two model configs: Stories110M (MHA, 12 layers) and Qwen3-0.6B (GQA 16q/8kv, 28 layers)

**SME2 Assembly Kernels** (`sme2-kernels/`) — 153 hand-written ARM SME2 assembly kernels:
- Full ML operator taxonomy: matmul, attention, conv2d, activations, normalization, losses, optimizers
- Achieves 8.5 TOPS INT8 (2x Apple's BNNS framework)
- Targets `armv9-a+sme2+sve2+sme-lutv2`
- All 172 tests passing on M4 Pro

## Quick Start

```bash
# ANE Training (requires macOS + Apple Silicon)
cd ane-training/training/training_dynamic
make MODEL=qwen3_06b
./train --scratch

# SME2 Kernels (requires M4 or later with SME2)
cd sme2-kernels
mkdir build && cd build
cmake .. && make -j$(sysctl -n hw.logicalcpu)
ctest
```

See [setup-guide.md](setup-guide.md) for detailed build instructions, CLI flags, and known issues.

## Training Pipeline

The end-to-end workflow takes a pretrained model from Hugging Face, fine-tunes it on custom data using ANE, and produces a usable model as output.

### 1. Install Python dependencies

```bash
cd ane-training/training/training_dynamic
pip install -r tools/requirements.txt
```

### 2. Convert a pretrained model

Download weights from Hugging Face and convert to the ANE checkpoint format:

```bash
python tools/convert_hf.py Qwen/Qwen3-0.6B
# Creates: ane_qwen3_dyn_ckpt.bin (~3.6 GB with Adam states)
```

### 3. Prepare training data

Tokenize any text file (books, code, docs) into binary token IDs:

```bash
python tools/tokenize_data.py my_corpus.txt --model Qwen/Qwen3-0.6B
# Creates: train_data.bin

# Multiple files work too
python tools/tokenize_data.py chapter1.txt chapter2.txt chapter3.txt --model Qwen/Qwen3-0.6B

# Or pipe from stdin
cat *.txt | python tools/tokenize_data.py - --model Qwen/Qwen3-0.6B
```

### 4. Build and train

```bash
make MODEL=qwen3_06b
./train --resume --data train_data.bin
# Checkpoint auto-saves periodically to ane_qwen3_dyn_ckpt.bin
```

Training resumes from where it left off if interrupted. Use `--steps N` to set total steps, `--lr F` to adjust learning rate.

### 5. Test the trained model

A quick sanity check generates text directly from the checkpoint (pure numpy, no GPU needed):

```bash
python tools/generate.py ane_qwen3_dyn_ckpt.bin --model Qwen/Qwen3-0.6B --prompt "Once upon a time"
python tools/generate.py ane_qwen3_dyn_ckpt.bin --model Qwen/Qwen3-0.6B --prompt "Hello" --temp 0.6 --tokens 200
```

This is slow (~seconds/token) but confirms the fine-tune produced meaningful results.

### 6. Export for real inference

Convert back to Hugging Face safetensors format:

```bash
python tools/export_hf.py ane_qwen3_dyn_ckpt.bin --model Qwen/Qwen3-0.6B --output ./my_model
```

The exported model works with any HF-compatible tool:

```bash
# Use with transformers
python -c "from transformers import pipeline; print(pipeline('text-generation', './my_model')('Hello'))"

# Convert to GGUF for llama.cpp / Ollama
python llama.cpp/convert_hf_to_gguf.py ./my_model

# Push to Hugging Face Hub
python tools/export_hf.py ane_qwen3_dyn_ckpt.bin --model Qwen/Qwen3-0.6B --output ./my_model --push yourname/my-model
```

## Performance (M4 Pro)

| Metric | Value |
|--------|-------|
| Qwen3-0.6B step time | ~280-330ms |
| ANE peak throughput (fp16) | 18.5 TOPS |
| ANE peak throughput (int8) | 34.9 TOPS |
| SME2 INT8 throughput | 8.5 TOPS (2x BNNS) |
| Current ANE utilization | 5-9% |

### Training Time Estimates (Qwen3-0.6B, M4 Pro)

Each step processes 256 tokens at ~300ms.

| Dataset Size | Steps | Wall Time |
|-------------|-------|-----------|
| 1M tokens | ~4,000 | ~20 min |
| 10M tokens | ~40,000 | ~3.5 hrs |
| 100M tokens | ~390,000 | ~35 hrs |

Fine-tuning from pretrained weights (a book, codebase, or domain corpus) typically needs 1-10M tokens. Pretraining from scratch requires significantly more data and time.

See [benchmarks.md](benchmarks.md) for full measurements and [optimization-roadmap.md](optimization-roadmap.md) for the plan targeting 90%+ utilization.

## Documentation

- [setup-guide.md](setup-guide.md) — Build guide, CLI flags, known issues
- [benchmarks.md](benchmarks.md) — M4 Pro throughput measurements and analysis
- [research-notes.md](research-notes.md) — ANE architecture research notes
- [optimization-roadmap.md](optimization-roadmap.md) — Optimization plan (Phases 1-6)
- [CLAUDE.md](CLAUDE.md) — Technical reference for AI-assisted development

## Acknowledgments

This project builds on prior work by:

**[maderix](https://github.com/maderix)** — [ANE Training](https://github.com/maderix/ANE)
Demonstrated that backpropagation on the Apple Neural Engine is possible by reverse-engineering the private `_ANEClient` and `_ANECompiler` APIs. Built the first working transformer training pipeline running entirely on ANE — forward pass, backward pass, and weight updates — without CoreML, Metal, or GPU.

**[Josh Morgan (joshmorgan1000)](https://github.com/joshmorgan1000)** — [SME2 Kernels](https://github.com/joshmorgan1000/ane)
Wrote 153 hand-tuned ARM SME2 assembly kernels and demonstrated that Apple's Neural Engine hardware capabilities extend to the SME2 matrix tiles accessible from userspace. Achieved 2x throughput over Apple's BNNS framework and documented the relationship between the ANE software stack and the underlying hardware.

Both upstream projects are MIT licensed. Their original LICENSE files are preserved in their respective directories.

## License

MIT — see individual LICENSE files in `ane-training/` and `sme2-kernels/` for upstream terms.
