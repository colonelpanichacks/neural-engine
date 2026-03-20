# Apple Neural Engine — Setup & Usage Guide

**Test Hardware:** Mac mini M4 Pro, 14-core (10P+4E), 48GB, 16-core ANE @ 38 TOPS
**OS Required:** macOS 15+

This guide covers cloning, building, and running both ANE research projects on compatible Apple Silicon hardware. Everything runs locally with zero external dependencies beyond Xcode CLI tools.

---

## Prerequisites

```bash
# Xcode command line tools (required for both projects)
xcode-select --install

# CMake (required for joshmorgan1000/ane only)
brew install cmake

# Python packages (optional — maderix dashboard only)
pip install blessed psutil numpy
```

No cloud accounts, no pip packages for core functionality, no Docker.

---

## Project 1: SME2 Assembly Kernels (joshmorgan1000/ane)

153 hand-written ARM SME2 kernels. Direct matrix acceleration, 2x faster than Apple BNNS on INT8.

### Clone & Build

```bash
cd neural-engine
git clone https://github.com/joshmorgan1000/ane.git sme2-kernels
cd sme2-kernels
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(sysctl -n hw.logicalcpu)
```

CMake will auto-detect SME2 support on M4-class hardware. If configure fails, the target CPU does not support SME2.

### Install to Local Prefix

```bash
cmake --install . --prefix ~/.local
```

After install, use in any C++ project:

```cmake
find_package(ane REQUIRED)
target_link_libraries(my_app PRIVATE ane::ane)
```

### Run Tests & Benchmarks

```bash
# Unit tests
ctest

# Full throughput benchmarks (builds temp binaries, runs timing)
bash ../tests/run_full_throughput_tests.sh
```

### Hardware Probes (Discover Supported Instructions)

```bash
bash ../probes/probe_all_registers.sh    # Enumerate accessible registers
bash ../probes/probe_instructions.sh     # Test which instructions work
```

### Using the Kernels

```cpp
#include <ane/ane.hpp>

// Elementwise
float a[1024], b[1024], c[1024];
ane::kernel::add_fp32(a, b, c, 1024);

// Matrix multiply (M=64, N=32, K=128)
float A[64*128], B[128*32], C[64*32];
ane::kernel::matmul_fp32(A, B, C, 64, 32, 128);

// INT8 matmul (2x faster than BNNS)
int8_t Aq[64*128], Bq[128*32];
int32_t Cq[64*32];
ane::kernel::matmul_int8(Aq, Bq, Cq, 64, 32, 128);

// Flash attention
ane::kernel::flash_attn_prefill_causal_fp32(Q, K, V, O, N, d);

// Adam optimizer step
ane::kernel::adam_fp32(params, grads, m, v, lr, beta1, beta2, eps, t, n);
```

### Key Kernel Categories

| Category | Kernels | Notes |
|---|---|---|
| Matmul | `matmul_fp32`, `matmul_int8`, `matmul_bfp16` | Transposed variants: `_nt`, `_tn` |
| Attention | `flash_attn_*`, `sdp_attn_*`, `gqa_attn_*`, `cross_attn_*` | Prefill, decode, causal/noncausal |
| Conv2D | `conv2d_fp32` + fused variants | Forward + backward passes |
| Activations | relu, gelu, silu, sigmoid, mish, elu, selu, etc. | All have backward variants |
| Normalization | layernorm, rms_norm, batchnorm | Fused combos available |
| Losses | mse, cross_entropy, bce, mae | All with backward passes |
| Optimizers | adam, sgd, param_update | Full training support |
| Quantization | fp32<->int8, fp32<->int4, q8_0_matvec | For quantized inference |
| Stochastic | dropout, gaussian_noise | Training regularization |

---

## Project 2: ANE Training Pipeline (maderix/ANE)

Full transformer training on the Neural Engine via reverse-engineered private `_ANEClient` APIs. Pure Objective-C, zero external deps.

### Clone & Download Data

```bash
cd neural-engine
git clone https://github.com/maderix/ANE.git ane-training
cd ane-training/training
bash download_data.sh
```

This downloads pretokenized TinyStories (~993MB archive, extracts ~41MB `tinystories_data00.bin` with ~20M tokens, then cleans up).

### Build & Train Stories110M (109M params)

```bash
cd training_dynamic
make MODEL=stories110m
./train --scratch
```

Training starts immediately — ~91ms/step on an M4 Pro test system.

### Build & Train Qwen3-0.6B (596M params)

```bash
make MODEL=qwen3_06b
./train --scratch
```

Slower at ~412ms/step, 28 layers with grouped-query attention.

### Training CLI Flags

```bash
./train --scratch              # Train from random initialization
./train --resume               # Resume from checkpoint
./train --steps 200            # Custom step count (default: 10000)
./train --lr 1e-4              # Custom learning rate (default: 3e-4)
./train --model ckpt.bin       # Load specific checkpoint
./train --data path/data.bin   # Custom training data
```

### Monitor Training (TUI Dashboard)

```bash
sudo python3 dashboard.py --dynamic
```

Shows live loss curve, power consumption, CPU/memory usage, and an efficiency report on completion (compile vs train time split, sustained TFLOPS).

### Build the Bridge Library (C-callable ANE Wrapper)

```bash
cd bridge
make
```

Produces `libane_bridge.dylib`. This is the most integration-friendly piece — call ANE from Python via ctypes, from C, or from any FFI.

```bash
make test    # Verify bridge works
```

### Run INT8 Benchmarks

```bash
xcrun clang -O2 -fobjc-arc -framework Foundation -framework IOSurface -ldl \
  -o ane_int8_bench ane_int8_bench.m
./ane_int8_bench
```

### Run GPU+ANE Hybrid Inference Demo

```bash
xcrun clang -O2 -fobjc-arc -framework Foundation -framework IOSurface \
  -framework Metal -framework MetalPerformanceShaders -ldl \
  -o gpu_ane gpu_prefill_ane_decode.m
./gpu_ane
```

Expected: Stories110M in ~8.8ms total (6.7ms GPU prefill + 1.9ms ANE decode).

### Adding a Custom Model

Create `training/training_dynamic/models/mymodel.h`:

```c
#pragma once
#define MODEL_NAME    "MyModel"
#define DIM           2048
#define HIDDEN        5504
#define HEADS         32
#define KV_HEADS      8       // Set equal to HEADS for MHA, less for GQA
#define HD            64      // Head dimension (explicit, not derived)
#define SEQ           256
#define NLAYERS       22
#define VOCAB         32000
#define CKPT_PATH     "ane_mymodel_dyn_ckpt.bin"
#define DEFAULT_DATA_PATH "../tinystories_data00.bin"
```

Constraints:
- `HEADS` must be evenly divisible by `KV_HEADS`
- `HD` is explicit (Qwen3 uses HD=128 while DIM/HEADS=64)

Build: `make MODEL=mymodel`

---

## Training Time Estimates (M4 Pro)

| Model | ms/step | 10K steps | 100K steps | Full epoch (TinyStories) |
|---|---|---|---|---|
| Stories110M | ~91 | ~15 min | ~2.5 hrs | ~53 hrs |
| Qwen3-0.6B | ~412 | ~69 min | ~11.5 hrs | ~240 hrs |

ANE utilization is currently 5-9% of peak. These times reflect real-world throughput, not theoretical.

---

## Known Issues & Workarounds

| Issue | What Happens | Fix |
|---|---|---|
| ~119 compile limit | ANE compiler leaks resources, crashes after ~119 compilations | Pipeline auto-restarts via `exec()` with checkpoint save/restore |
| FP16 gradient underflow | Backward pass matmuls lose precision | Auto-applied loss scaling: `256 * NLAYERS` |
| SDPA causal masking | ANE hardware ignores attention masks | Decomposed: Q@K^T on ANE, mask+softmax on CPU, scores@V on ANE |
| Multi-input ANE requests | Error 0x1d when passing multiple inputs | All inputs packed into single IOSurface spatial dimension |
| macOS updates | Private APIs can break any time | Pin macOS version when things work. No fix otherwise. |

---

## Capabilities and Limitations

### Working Today

1. **ANE benchmarking** — Run INT8/FP16 throughput tests on M4 Pro hardware, measure real TOPS numbers
2. **Stories110M training** — Full transformer training on-device, no cloud
3. **Qwen3-0.6B training** — Larger model, slower but functional
4. **Hybrid GPU+ANE inference** — Sub-10ms inference for small models
5. **SME2 kernels** — 2x BNNS throughput for any custom compute pipeline
6. **Bridge library** — Call ANE from Python/C for custom inference

### Near-term (with engineering effort)

1. **Custom small model training** — Define a model header, train on custom data
2. **INT8 quantized training** — 1.88x speedup over FP16, built into the maderix pipeline
3. **SME2 kernel integration** — Accelerate preprocessing/postprocessing for on-device ML
4. **Python wrapper** — Bridge library + ctypes for rapid prototyping

### Not feasible yet

1. **100% ANE utilization** — See section below
2. **Training large models (7B+)** — Memory bandwidth and pipeline limitations
3. **Replacing Ollama/MLX** — Would require forking their entire inference backend
4. **Production deployment** — Private APIs, no stability guarantees

---

## On Getting to 100% ANE Utilization

Current utilization sits at 5-9% of peak. The gap comes from:

**1. CPU-ANE synchronization overhead**
Every time data moves between CPU and ANE, there is a sync stall. The training loop follows the pattern: ANE forward -> CPU RMSNorm -> ANE backward -> CPU optimizer. Each handoff burns cycles.

**2. Causal masking falls back to CPU**
ANE hardware ignores attention masks. The pipeline decomposes SDPA into three steps (ANE -> CPU -> ANE), breaking what should be a single fused kernel.

**3. Memory bandwidth bottleneck**
The ANE can compute faster than data can be fed to it. IOSurface transfers, even zero-copy, have latency.

**4. Compiler overhead**
Even with the dynamic pipeline (single compile at startup), the MIL -> E5 compilation path is not optimized for rapid iteration.

**What would need to change to hit 100%:**

- **Full graph execution** — Eliminate CPU fallbacks entirely. Every op (including RMSNorm, softmax, masking) would need to run on ANE. This requires either Apple adding causal mask support or finding a mathematical workaround.
- **Pipelined execution** — Use `_ANEChainingRequest` (unexplored API) to chain multiple compiled models in a single dispatch, eliminating inter-kernel sync.
- **Hardware counters** — Use `_ANEPerformanceStats` (unexplored API) to identify exactly where cycles are wasted.
- **Multi-process ANE sharing** — Use `_ANEVirtualClient` (unexplored API) to saturate ANE with parallel workloads.

**Realistic target:** 20-30% utilization is achievable with engineering effort (fusing more ops, reducing CPU fallbacks, better overlap). 100% would require Apple cooperation or breakthroughs in the unexplored APIs listed above.

---

## File Structure

```
neural-engine/
  research-notes.md        # Deep technical research notes
  setup-guide.md           # This file — practical setup & usage
  sme2-kernels/            # joshmorgan1000/ane (clone)
    include/ane/ane.hpp     # Single-header C++ API
    src/ane/asm/            # 153 assembly kernel files
    tests/                  # Unit tests + throughput benchmarks
    build/                  # Build output (libane.a)
  ane-training/            # maderix/ANE (clone)
    bridge/                 # C-callable ANE wrapper (libane_bridge.dylib)
    training/               # Training data + static pipeline
      training_dynamic/     # Dynamic pipeline (recommended)
        models/             # Model config headers
        train.m             # Main training loop
    gpu_prefill_ane_decode.m  # Hybrid inference demo
    ane_int8_bench.m        # INT8 benchmark
```
