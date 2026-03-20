# Apple Neural Engine — Reverse Engineering and Direct Access Research

## Overview

Two independent research efforts have exposed Apple's Neural Engine on M4 hardware, bypassing CoreML and demonstrating capabilities Apple never intended to make available — including **training** on inference-only hardware.

This document consolidates findings from:
- **joshmorgan1000/ane** — 153 hand-written ARM SME2 assembly kernels demonstrating that ANE = CPU matrix tiles
- **maderix/ANE** — Full transformer training via reverse-engineered private `_ANEClient` APIs
- **"Inside the M4 Apple Neural Engine"** — Three-part Substack series documenting the reverse engineering process

---

## Two Competing Theories of "What Is the ANE?"

### Theory 1: ANE Is the SME2 Matrix Tiles (joshmorgan1000)

The claim: Apple's "Neural Engine" is not a separate coprocessor — it is the **ARM SME2 `za` matrix tiles** integrated into the CPU, plus Apple's software scheduling layer (BNNS/CoreML).

**Evidence — hardware contention benchmarks (M4 Max):**

| Configuration | Throughput (TOPS) | Notes |
|---|---|---|
| GPU alone (Metal INT8) | 37.4 | Unaffected by CPU load |
| SME alone (4 threads, INT8) | 8.5 | Raw `smopa` performance |
| BNNS INT8 (4 threads) | 4.2 | Apple's framework |
| SME + BNNS concurrent | 5.2 + 1.6 = 6.9 | **Contention proves shared HW** |
| GPU + SME concurrent | 37.4 + 8.5 = 45.9 | Independent — no contention |

When SME and BNNS run simultaneously, SME drops 39% and BNNS drops 62%. The two compete for the same hardware. GPU remains unaffected, confirming it occupies genuinely separate silicon.

Direct SME access achieves **2x the throughput** of Apple's BNNS framework on INT8.

### Theory 2: ANE Is a Dedicated Graph Execution Engine (maderix)

The ANE (codename **H16G**) is a fixed-function accelerator with:
- **16 cores**
- Queue depth of **127 concurrent evaluation requests**
- Independent DVFS (dynamic voltage/frequency scaling)
- Complete power gating (zero consumption when idle)
- Execution of entire neural network graphs atomically, not individual instructions

**Evidence:** Direct IOKit driver access, `_ANEClient` API discovery, E5 binary analysis, power domain profiling showing independent ANE DVFS channels separate from CPU/GPU.

### Reconciliation

These theories may not be mutually exclusive. BNNS could use SME2 tiles for its compute while the dedicated ANE coprocessor (H16G) handles graph-level execution through a separate hardware path. The contention between SME and BNNS demonstrates that BNNS uses SME tiles, but does not prove the ANE coprocessor is absent — the maderix project accesses ANE through a completely different API path (IOKit) than BNNS.

---

## The ANE Software Stack

Discovered by maderix via `dyld_info -objc` introspection and method swizzling:

```
CoreML (public API)
  |
AppleNeuralEngine.framework (private, 40+ undocumented classes)
  |
  +-- _ANEClient          (compile -> load -> evaluate pipeline)
  +-- _ANECompiler         (MIL -> E5 binary compilation)
  +-- _ANEInMemoryModelDescriptor  (in-memory compilation, no disk)
  +-- _ANEModel            (program handle management)
  +-- _ANEIOSurfaceObject  (zero-copy I/O)
  |
ANECompiler.framework
  |
IOKit kernel driver
  |
H16G Hardware (16 cores)
```

### Unexplored APIs (potential future research)
- `_ANEChainingRequest` — chain multiple compiled models in single dispatch
- `_ANESharedEvents` / `_ANESharedSignalEvent` / `_ANESharedWaitEvent` — Metal-style fence/signal primitives for GPU<->ANE sync
- `_ANEPerformanceStats` — hardware performance counters
- `_ANEVirtualClient` — multi-process ANE sharing

---

## MIL: Machine Learning Intermediate Language

The ANE's input format. A typed SSA (Static Single Assignment) representation.

```
program(1.3) {
    func main<ios18>(
        tensor<fp16, [1, 1024, 1, 1024]> x,
        tensor<fp16, [1, 1024, 1, 1024]> w
    ) {
        tensor<fp16, [1, 1024, 1, 1024]> out =
            matmul(transpose_x = false, transpose_y = false,
                   x = x, y = w);
    } -> (out);
}
```

Tensors use **NCDHW + Interleave** layout: `[Batch, Channels, Depth, Height, Width]`.

### E5 Binary Format
- FlatBuffer-structured compiled output
- A 1024x1024 matmul compiles to **2,688 bytes**; a 128x128 to **2,680 bytes**
- Near-identical size regardless of matrix dimensions = ANE contains **parameterized compute primitives**, not hardcoded algorithms
- First compilation: ~20-40ms; cache hits: near-zero latency
- Cache location: `~/Library/Caches/<app>/com.apple.e5rt.e5bundlecache/`

### IOSurface I/O
All data transfer uses IOSurfaces (same mechanism as GPU texture sharing):
- Enables zero-copy GPU<->ANE pipelines
- Native tensor format: `[1, Channels, 1, Spatial]`
- FP16 direct I/O is ~37% faster than FP32

---

## joshmorgan1000/ane — SME2 Assembly Kernels

**153 hand-written ARM assembly kernels** targeting `armv9-a+sme2+sve2+sme-lutv2`.

### Requirements
- Apple M4 (or ARM with SME2)
- CMake 3.19+, C++20 compiler (Apple Clang)
- Node.js 20+ (optional, for dashboard UI)

### Build
```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(sysctl -n hw.logicalcpu)
cmake --install build --prefix ~/.local
```

### API
```cpp
#include <ane/ane.hpp>

// Elementwise
ane::kernel::add_fp32(float* a, float* b, float* c, size_t n);

// Matrix multiply
ane::kernel::matmul_fp32(float* A, float* B, float* C, size_t M, size_t N, size_t K);
ane::kernel::matmul_int8(int8_t* A, int8_t* B, int32_t* C, size_t M, size_t N, size_t K);
```

CMake integration:
```cmake
find_package(ane REQUIRED)
target_link_libraries(my_app PRIVATE ane::ane)
```

### Complete Kernel Catalog (153 kernels)

| Category | Operations |
|---|---|
| **Elementwise** | `add_fp32`, `mul_fp32`, `div_fp32`, `fma_fp32`, `scalar_mul_fp32` |
| **Activations** | `relu_fp32`, `gelu_fp32`, `silu_fp32`, `sigmoid_fp32`, `mish_fp32`, `elu_fp32` |
| **Reductions** | `reduce_sum_fp32`, `reduce_max_fp32`, `dot_fp32`, `sumsqr_fp32`, `argmax_fp32` |
| **Matrix Multiply** | `matmul_fp32`, `matmul_int8`, `matmul_bfp16`, transposed variants |
| **Convolution** | `conv2d_fp32`, fused bias/relu/bn/swish/gelu variants, backward passes |
| **Normalization** | `layernorm_fp32`, `fused_rms_norm_scale_fp32`, `fused_batchnorm_relu_fp32` |
| **Attention** | `flash_attn_*`, `sdp_attn_*`, `gqa_attn_*`, `cross_attn_*`, `rope_fp32` |
| **Losses** | `mse_loss_fp32`, `cross_entropy_loss_fp32`, `bce_loss_fp32`, `mae_loss_fp32` |
| **Optimizers** | `adam_fp32`, `sgd_fp32`, `param_update_fp32` |
| **Quantization** | `quantize_fp32_int8`, `dequantize_int8_fp32`, `q8_0_matvec` |
| **Type Convert** | `fp32_to_bf16`, `bf16_to_fp32`, `fp32_to_int32` |
| **Bitwise** | `and_u32`, `or_u32`, `xor_u32`, `shl_u32`, `shr_u32` |
| **LUT** | `tbl_u8`, `luti4_u8`, `luti2_u8` |
| **Stochastic** | `dropout_fp32`, `gaussian_noise_fp32` |

### Limitations
- FP8 operations unsupported (non-functional)
- Basic `add`/`mul`/`sub` on `za` tile do not work — `fmla` must be used instead
- BF16 and FP32 `fmopa` performance is identical (suggests a unified pipeline)
- Single-threaded kernel design
- **Not production-ready** — undocumented machine code

---

## maderix/ANE — Training on the Neural Engine

Full transformer training via reverse-engineered private APIs. Zero external dependencies — Objective-C + C only, using system frameworks with runtime API resolution via `objc_msgSend`.

### Requirements
- macOS 15+, Apple Silicon (tested on M4)
- No external dependencies

### Build
```bash
# Dynamic training (recommended)
cd training/training_dynamic
make MODEL=stories110m    # 12L MHA, 109M params
make MODEL=qwen3_06b      # 28L GQA, 596M params
./train --scratch         # Random init
./train --resume          # From checkpoint

# INT8 benchmark
xcrun clang -O2 -fobjc-arc -framework Foundation \
  -framework IOSurface -ldl -o ane_int8_bench ane_int8_bench.m

# Bridge library (C-callable ANE wrapper)
cd bridge && make

# Training data
cd training && bash download_data.sh
```

### Trained Models

| Model | Params | Step Time | Layers | Attention | Kernels/Layer |
|---|---|---|---|---|---|
| Stories110M | 109M | 91 ms/step | 12 | MHA (12/12) | 6 |
| Qwen3-0.6B | 596M | 412 ms/step | 28 | GQA (16/8) | 10 |

### Training Pipeline

**MHA Pipeline (Stories110M) — 6 kernels per layer:**
1. `sdpaFwd` — QKV projection + SDPA + output projection
2. `ffnFused` — SwiGLU FFN (W1, W3, SiLU, W2 fused)
3. `ffnBwdW2t` — FFN backward for W2^T gradient
4. `ffnBwdW13t` — FFN backward for W1,W3^T gradients
5. `sdpaBwd1` — Attention backward phase 1
6. `sdpaBwd2` — Attention backward phase 2

**GQA Pipeline (Qwen3-0.6B) — 10 kernels per layer:**
Additional kernels include `woFwd`, `qBwd`, `kvBwd` for grouped-query mechanics.

**CPU-handled operations:** RMSNorm, residual connections (DeepNet alpha scaling), loss computation, dW gradient accumulation (cblas_sgemm), Adam optimizer.

### How Dynamic Training Works

1. **MIL Generation** — Objective-C constructs MIL program text at runtime
2. **In-Memory Compilation** — `_ANEInMemoryModelDescriptor` compiles MIL + weight blobs directly (no disk)
3. **IOSurface I/O** — Tensors passed via shared memory in `[1, C, 1, S]` format (FP16 or FP32)
4. **Dynamic Weights** — Packed into a single spatial input dimension, sliced internally; updates occur without recompilation
5. **Gradient Flow** — Forward taps expose intermediates; backward dx computed on ANE, dW computed on CPU via cblas
6. **INT8 Quantization** — `constexpr_affine_dequantize` for weights, `quantize`/`dequantize` for activation caching in L2 SRAM

### Peak ANE Throughput (M4)

| Config | FP16 | INT8 W8A8 | Speedup |
|---|---|---|---|
| 128x conv, 512ch, 64x64 | 18.6 TOPS, 14.8ms | 35.1 TOPS, 7.8ms | 1.88x |
| 64x conv, 512ch, 64x64 | 18.4 TOPS, 7.5ms | 34.1 TOPS, 4.0ms | 1.85x |

### GPU<->ANE Inference Pipeline (seq=256)

| Model | GPU Prefill | ANE Decode | Total |
|---|---|---|---|
| Stories110M | 6.7ms | 1.9ms | 8.8ms |
| Qwen3-0.6B | 9.7ms | 2.3ms | 12.0ms |

### Key Optimizations
- **Channel-first CPU layout** matches ANE IOSurface `[1,C,1,S]` — no transpose overhead
- **vDSP vectorized RMSNorm** — 10x speedup (6.7ms -> 0.7ms)
- **GCD async cblas overlap** — dW gradient sgemms run parallel to ANE evals
- **Deferred cblas wait** — sync pushed to next step's forward for maximum overlap
- **ANE RMSNorm fusion** — folded into forward kernels as MIL ops
- **Wo^T fusion** — output projection backward merged into SDPA backward
- **Forward taps** — expose intermediates, avoiding CPU recompute

### Limitations & Workarounds

| Issue | Impact | Workaround |
|---|---|---|
| SDPA causal masking | ANE ignores `attn_mask` | Decompose: Q@K^T (ANE) -> mask+softmax (CPU) -> scores@V (ANE) |
| Compiler resource leak | ~119 compiles per process | `exec()` subprocess restart with checkpoint |
| FP16 gradient underflow | Backward matmuls underflow | Global loss scaling: `256 * num_layers` |
| Single-input constraint | Multi-input requests fail (0x1d error) | Pack all inputs into single spatial dim, slice internally |
| Low utilization | ~5-9% of peak ANE capacity | Memory bandwidth bottleneck + CPU-ANE sync overhead |
| Element-wise ops | Not all run on ANE | CPU fallback for unsupported ops |

---

## Prior Art & References

| Project | Coverage |
|---|---|
| [hollance/neural-engine](https://github.com/hollance/neural-engine) | Comprehensive behavioral documentation |
| [mdaiter/ane](https://github.com/mdaiter/ane) | Early Python/ObjC samples, ANECompiler framework |
| [eiln/ane](https://github.com/eiln/ane) | Linux driver (Asahi Linux), kernel-level interface |
| [apple/ml-ane-transformers](https://github.com/apple/ml-ane-transformers) | Official reference: channel-first layout, 1x1 conv patterns |

### Substack Series
- Part 1: Reverse Engineering (API discovery, MIL, E5 format)
- Part 2: Benchmarks (matmul scaling, SRAM cliffs, conv 3x faster than matmul, debunking "38 TOPS")
- Part 3: Training implementation on inference-only hardware

---

## Potential Applications

### Local LLM Inference
- GPU prefill -> ANE decode pipeline achieves Stories110M in 8.8ms, Qwen3-0.6B in 12.0ms
- The bridge library (`maderix/ANE/bridge/`) provides a C-callable wrapper suitable for integration with existing inference engines

### On-Device Fine-Tuning of Small Models
- Stories110M trains at 91ms/step entirely on-device
- No cloud GPU required for small model customization
- Checkpoint/resume support enables long training runs

### Edge ML
- Small classifiers and detectors can be trained locally
- Trained weights are exportable for embedded/edge deployment
- INT8 quantization is built into the pipeline

### Raw Compute via SME2 Kernels
- 153 optimized kernels available as a C++ library
- 2x throughput over BNNS on INT8
- Applicable to any matrix-heavy workload on M4

---

## Hardware Requirements Summary

| Component | joshmorgan1000/ane | maderix/ANE |
|---|---|---|
| **Chip** | M4 (ARM SME2) | Apple Silicon (M4 tested) |
| **OS** | Any (bare metal ASM) | macOS 15+ |
| **Language** | C++20 / ARM ASM | Objective-C / C |
| **Dependencies** | CMake, Clang | None (system frameworks only) |
| **Production Ready** | No | No |

---

## Open Questions

- Exact ANE core microarchitecture and ISA
- How cores are assigned within graph execution
- Dynamic ANE clock frequency values
- Hardware performance counter accessibility
- SRAM topology (banked vs unified vs per-core)
- Whether SME2 contention fully explains BNNS behavior or if the ANE coprocessor operates through a separate path
- Whether `_ANEChainingRequest` can enable multi-model pipelines
- Whether `_ANEPerformanceStats` can expose hardware counters for profiling
