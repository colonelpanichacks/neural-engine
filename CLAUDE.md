# Neural Engine

Research project for training transformers directly on Apple's Neural Engine (ANE) via reverse-engineered private APIs, plus a collection of hand-written ARM SME2 assembly kernels. Targets Apple Silicon M4 Pro (14-core CPU, 16-core ANE, 48GB RAM).

## Project Structure

```
neural-engine/
  setup-guide.md        # Build guide, CLI flags, known issues
  benchmarks.md         # M4 Pro throughput measurements
  research-notes.md     # ANE architecture research notes
  optimization-roadmap.md # Optimization plan (Phases 1-6)

  ane-training/         # Transformer training on ANE
    bridge/             # C-callable ANE wrapper (libane_bridge.dylib)
    training/
      training_dynamic/ # Active development — dynamic weight pipeline
        train.m         # Main training loop (forward + backward + Adam)
        config.h        # Model-agnostic structs, derived sizes, alloc helpers
        io.h            # IOSurface I/O, NEON fp16 conversion, weight staging
        cpu_ops.h       # RMSNorm, cross-entropy, Adam, embedding, RoPE
        mil_dynamic.h   # MIL code generators for all ANE kernels
        Makefile        # Build: make MODEL=qwen3_06b
        models/
          stories110m.h # 109M params, 12 layers, MHA (Llama2-style)
          qwen3_06b.h   # 596M params, 28 layers, GQA 16q/8kv
        tools/
          convert_hf.py   # HF safetensors -> ANE checkpoint
          tokenize_data.py # Text -> binary uint16 token IDs
          export_hf.py    # ANE checkpoint -> HF safetensors
          generate.py     # Inference/text generation from checkpoint
          requirements.txt # Python deps (torch, transformers, etc.)
      training/         # Legacy static pipeline (not actively used)
      dashboard.py      # TUI training dashboard
      tinystories_data00.bin  # Pretokenized training data (~41MB)

  sme2-kernels/         # 153 ARM SME2 assembly kernels
    include/ane/ane.hpp # Single-header C++ API
    src/ane/asm/        # Assembly source (.s files)
    tests/              # 8 test suites, 172 tests (all passing)
    CMakeLists.txt      # CMake build
```

## Build & Run

### ANE Training (primary)

```bash
cd ane-training/training/training_dynamic
make MODEL=qwen3_06b    # or MODEL=stories110m
./train --scratch        # train from random init
./train --resume         # resume from checkpoint
```

**CLI flags:** `--scratch`, `--resume`, `--steps N`, `--lr F`, `--accum N`, `--clip F`, `--warmup N`

**Dependencies:** Xcode Command Line Tools only. Uses Foundation, IOSurface, Accelerate frameworks. No external dependencies.

### Training Pipeline (Python tools)

```bash
cd ane-training/training/training_dynamic
pip install -r tools/requirements.txt

# 1. Convert a pretrained HF model to ANE checkpoint format
python tools/convert_hf.py Qwen/Qwen3-0.6B

# 2. Tokenize your training data
python tools/tokenize_data.py my_corpus.txt --model Qwen/Qwen3-0.6B --output train_data.bin

# 3. Train on ANE
make MODEL=qwen3_06b
./train --resume --data train_data.bin

# 4. Export trained checkpoint back to HF safetensors
python tools/export_hf.py ane_qwen3_dyn_ckpt.bin --model Qwen/Qwen3-0.6B --output ./my_model

# 5. Generate text (pure numpy inference, slow but verifies training)
python tools/generate.py ane_qwen3_dyn_ckpt.bin --model Qwen/Qwen3-0.6B --prompt "Hello world"

# 6. Or convert to GGUF for fast inference with llama.cpp
python llama.cpp/convert_hf_to_gguf.py ./my_model
```

### SME2 Kernels

```bash
cd sme2-kernels/build
cmake .. && make -j$(sysctl -n hw.logicalcpu)
ctest  # run 172 tests
```

## Architecture

### How ANE Training Works

1. **MIL Generation** (`mil_dynamic.h`): At startup, generates MIL (Machine Learning Intermediate Language) program text for each kernel type. Weights are packed into the spatial dimension of input IOSurfaces, allowing weight updates without kernel recompilation.

2. **Kernel Compilation**: MIL programs are compiled to ANE via private `_ANEInMemoryModel` APIs from `AppleNeuralEngine.framework`. Each kernel type is compiled once and shared across all layers via per-layer IOSurface bindings.

3. **IOSurface Data Flow** (`io.h`): All ANE I/O uses IOSurfaces in `[1, C, 1, S]` layout (fp16). Weight staging functions pack activations and weights into the spatial dimension. Per-layer input surfaces are pre-allocated with weights baked in; output surfaces are shared.

4. **Training Loop** (`train.m`):
   - **Forward**: Per-layer: CPU RMSNorm -> ANE sdpaFwd -> ANE woFwd -> residual -> CPU RMSNorm -> ANE ffnFused -> residual
   - **Backward**: Per-layer: ANE ffnBwdW2t -> ANE ffnBwdFused -> ANE wotBwd -> ANE sdpaBwd1 -> ANE sdpaBwd2 -> ANE qBwd/kvBwd -> CPU RMSNorm backward. dW gradients computed on CPU via cblas_sgemm, overlapped with ANE via dispatch_group.
   - **Adam**: vDSP-vectorized Adam optimizer with gradient accumulation and cosine LR schedule.

5. **Async Overlap**: ANE kernels dispatched via `ane_eval_req_async()` with CPU work (memcpy, fp16 conversion, gradient accumulation) overlapped during ANE execution.

### Key Technical Details

- **GQA (Grouped Query Attention)**: Qwen3-0.6B uses 16 query heads / 8 KV heads (ratio 2). Q_DIM (2048) != DIM (1024). KV heads are tiled for SDPA forward and reduced in backward.
- **fp16 Activation Storage**: Activations flowing between ANE kernels (Q, K, V, h1, h3, silu_out, dsilu) are stored as fp16 to eliminate fp16->fp32->fp16 roundtrip conversions. Only converted to fp32 where CPU computation actually needs it.
- **Dynamic Weights**: Weights packed into IOSurface spatial dimension, sliced inside MIL via `slice_by_size`. Weights can be updated (Adam step + restaging) without recompiling kernels.
- **119 Kernel Compile Limit**: ANE enforces a per-process compile limit. Training uses checkpoint + exec() restart to work around this. Current pipeline compiles 10-11 kernels (well within limit).

### ANE Kernel Inventory (Dynamic Pipeline)

| Kernel | Purpose | Input Channels | Spatial |
|--------|---------|---------------|---------|
| sdpaFwd | QKV projection + SDPA (GQA) | DIM | SEQ + Q_DIM + KV_DIM*2 |
| woFwd | Output projection | Q_DIM | SEQ + DIM |
| ffnFused | SwiGLU FFN (W1,W3,SiLU,W2) | DIM | SEQ + HIDDEN*2 + DIM |
| ffnBwdW2t | dffn @ W2^T -> dsilu | DIM | SEQ + HIDDEN |
| ffnBwdFused | SiLU backward + W1^T/W3^T | HIDDEN | 3*SEQ + 2*DIM |
| ffnBwdW13t | dh1/dh3 @ W1^T/W3^T (legacy) | HIDDEN | 2*SEQ + 2*DIM |
| wotBwd | dy @ Wo^T -> da | DIM | SEQ + Q_DIM |
| sdpaBwd1 | SDPA backward dQ | Q_DIM | 4*Q_DIM*SEQ |
| sdpaBwd2 | SDPA backward dK,dV | SCORE_CH | 2*SCORE_CH*SEQ + 2*Q_DIM*SEQ |
| qBwd | da @ Wq^T -> dx_attn | Q_DIM | SEQ + DIM |
| kvBwd | dk/dv @ Wk^T/Wv^T -> dx_kv | KV_DIM | 2*SEQ + DIM |

### CPU Operations (Not on ANE)

- **RMSNorm forward/backward**: vDSP vectorized. Attempted ANE fusion failed at Qwen3 dimensions (status=0x1d hardware resource limit).
- **RoPE backward**: vDSP vectorized with precomputed cos/sin tables.
- **Cross-entropy loss**: Per-token softmax + gradient with strided column access.
- **dW gradients**: cblas_sgemm dispatched to concurrent queue, overlapped with ANE backward.
- **Embedding lookup/backward**: Scalar strided scatter/gather (cache-hostile at large dims).

## Adding a New Model

Create `models/mymodel.h`:

```c
#pragma once
#define MODEL_NAME "MyModel"
#define DIM 512        // model dimension
#define HIDDEN 1536    // FFN hidden dim (typically 3x or 4x DIM)
#define HEADS 8        // number of query attention heads
#define KV_HEADS 4     // number of KV heads (= HEADS for MHA)
#define HD 64          // head dimension (may differ from DIM/HEADS)
#define GQA_RATIO (HEADS / KV_HEADS)
#define Q_DIM (HEADS * HD)
#define KV_DIM (KV_HEADS * HD)
#define SEQ 256        // sequence length
#define NLAYERS 6      // number of transformer layers
#define VOCAB 32000    // vocabulary size
#define CKPT_PATH "ane_mymodel_ckpt.bin"
#define DEFAULT_DATA_PATH "../tinystories_data00.bin"
```

Then `make MODEL=mymodel`.

## Performance (M4 Pro, Qwen3-0.6B)

| Metric | Value |
|--------|-------|
| Step time (10-step avg) | ~280-330ms |
| ANE forward | ~72ms |
| ANE backward | ~115-130ms |
| IO staging (fwd) | ~12ms |
| IO staging (bwd) | ~35-65ms |
| RMSNorm (fwd+bwd) | ~15ms |
| Cross-entropy | ~12ms |
| FLOPs/step | 676B (fwd+bwd) |

### ANE Hardware Notes

- ANE operates in fp16 only. All IOSurface data is fp16.
- ANE has ~32MB SRAM; efficiency drops ~3.7x when weight footprint exceeds this.
- Peak throughput: 18.5 TOPS (fp16), 34.9 TOPS (int8).
- Current utilization is 5-9% — bottlenecked by synchronous dispatch and CPU fallbacks.
- ANE is the H16G coprocessor, separate from CPU/GPU. Accessed via `AppleNeuralEngine.framework` private APIs.

## Known Limitations

- **119 kernel compile limit** per process (ANE driver limitation). Worked around via exec() restart.
- **RMSNorm on ANE** fails at DIM>=1024 (hardware resource limit 0x1d). Must stay on CPU.
- **Single IOSurface input** per kernel (ANE API constraint). Weights and activations packed into one surface.
- **No int8 training** — ANE int8 is inference-only (W8A8 matmul). Training uses fp16.
- **Causal mask** for SDPA uses large lower-triangular matrix baked as a const weight blob.

## Optimization History

1. **Baseline**: 95ms/step (Stories110M)
2. **Phase 2** (async dispatch, vDSP, fused SiLU+W13t on ANE): 88.3ms
3. **Phase 3** (SiLU backward fused into ANE): 86.4ms
4. **fp16 roundtrip elimination** (Q/K/V/h1/h3/silu_out/dsilu stored as fp16, direct memcpy between ANE kernels): ~280ms/step for Qwen3-0.6B (io_fwd -27%, io_bwd -35%)

## Git Conventions

- `master`: stable releases
- `dev`: active development (work here)
- Never commit directly to master; merge from dev when tested.
