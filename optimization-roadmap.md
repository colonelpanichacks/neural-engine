# ANE Unlocked — Roadmap to Maximum Utilization

**Goal:** Increase ANE utilization from 5-9% to as close to 100% as the hardware allows.
**Hardware:** Mac mini M4 Pro, 14-core CPU, 48GB RAM, 16-core ANE @ 38 TOPS

---

## Current State

Both research repos are cloned and ready to build:
- `sme2-kernels/` — joshmorgan1000/ane (153 ARM SME2 assembly kernels)
- `ane-training/` — maderix/ANE (transformer training via private _ANEClient APIs)

Training works today at 5-9% ANE utilization. The gap is not hardware — it is engineering.

---

## Root Causes of Low Utilization

### 1. Synchronous Dispatch (biggest single bottleneck)
Every `evaluateWithQoS:options:request:error:` call blocks the main thread. The CPU idles while ANE runs, and ANE idles while the CPU runs. No overlap, no pipelining.

### 2. RMSNorm on CPU (112 calls per step for Qwen3-0.6B)
4 RMSNorm ops per layer x 28 layers. Each involves reduction + rsqrt + elementwise scale. The old static pipeline (`stories_mil.h`) already proved this works on ANE using `reduce_sum + pow(x, -0.5)`. The dynamic system moved it to CPU and never moved it back.

### 3. Weight Gradients Serial on CPU (7 cblas_sgemm per layer)
All dW outer products run on a single serial GCD queue via cblas. The forward pass has a `dispatch_group_wait` at the top of every iteration — it cannot start until all dW from the previous backward pass completes.

### 4. Unvectorized Scalar CPU Ops
- **RoPE backward**: triple-nested loop with `cosf/sinf` per element (524K trig ops per call, 2x per layer)
- **SiLU backward**: elementwise chain using vDSP/vvexpf/vvrecf
- **Adam optimizer**: scalar loop over all parameters
- **Weight transpose**: scalar nested loop

### 5. Memory Allocation Churn
Hundreds of malloc/free per step. 5-11 mallocs per layer just for async dispatch copies (~10MB per layer). RMSNorm allocates/frees per call.

---

## Phase 1 — Measure (Week 1)

**Objective:** Establish real baseline numbers on the M4 Pro before changing anything.

### Tasks

- [ ] Build sme2-kernels (`cmake .. && make`)
- [ ] Run SME2 throughput benchmarks (`run_full_throughput_tests.sh`)
- [ ] Run hardware probes (`probe_all_registers.sh`, `probe_instructions.sh`)
- [ ] Build ane-training dynamic pipeline (`make MODEL=stories110m`)
- [ ] Download TinyStories data (`bash download_data.sh`)
- [ ] Run Stories110M training (`./train --scratch --steps 100`)
- [ ] Capture timing breakdown (the printout at step completion)
- [ ] Run INT8 benchmark (`ane_int8_bench.m`)
- [ ] Build and run `test_perf_stats.m` — read `_ANEPerformanceStats` counter values
- [ ] Build and run `test_qos_sweep.m` — measure QoS impact on latency
- [ ] Build and run `test_ane_advanced.m` — enumerate all runtime ANE classes
- [ ] Run SRAM probe (`sram_probe.m`) to find the chip's SRAM cliff point
- [ ] Document all baseline numbers

### Key Questions to Answer
- What are the actual TOPS (FP16 and INT8)?
- Where is the SRAM cliff on the M4 Pro?
- What do the `_ANEPerformanceStats` counters actually report?
- What QoS value gives best latency?
- What ANE classes exist at runtime on macOS 15?

---

## Phase 2 — Eliminate CPU Fallbacks (Week 2-3)

**Objective:** Move all possible ops from CPU to ANE. Target: 20-30% utilization.

### 2A. RMSNorm Back to ANE
The static pipeline proved this works. The MIL pattern:
```
ss = reduce_sum(x * x)           // sum of squares
rrms = pow(ss / dim, -0.5)       // reciprocal RMS
out = x * rrms * weight          // normalize + scale
```
All three MIL ops (`reduce_sum`, `pow`, `mul`) are supported. These should be fused into existing sdpaFwd/ffnFused kernels to eliminate separate dispatches.

**Impact:** Eliminates 112 CPU operations per step.

### 2B. SiLU Backward to ANE
Forward SiLU is already in `ffnFused` MIL (sigmoid + mul). Backward needs:
```
sig = sigmoid(x)
dsilu = sig * (1 + x * (1 - sig))
dx = dout * dsilu
```
All ops are available in MIL. These should be fused into the `ffnBwdW2t` kernel.

**Impact:** Eliminates 28 CPU elementwise chains per step.

### 2C. RoPE Backward to ANE
Forward RoPE is already in `sdpaFwd` MIL using reshape+slice+concat+mul+add with pre-computed cos/sin. Backward is the same math with transposed inputs. A new MIL kernel can be built, or this can be fused into `sdpaBwd2`.

**Impact:** Eliminates 56 unvectorized trig loops per step.

### 2D. GQA Backward Reduce
Forward GQA tile (KV_HEADS->HEADS) is on ANE via concat. Backward reduce (HEADS->KV_HEADS) is CPU memcpy+add. This is a `reduce_sum` with reshape — doable in MIL.

**Impact:** Eliminates 28 CPU data movement ops per step.

---

## Phase 3 — Async Pipeline (Week 3-4)

**Objective:** Overlap CPU and ANE execution. Target: 40-50% utilization.

### 3A. Probe Async Dispatch
- Investigate `_ANEInputBuffersReady` class (exists in framework — suggests an async path)
- Probe the `options:` dictionary on `evaluateWithQoS:` — always passed empty, never explored
- Try `-evaluateAsyncWithQoS:` or similar selectors via `class_copyMethodList`
- Instantiate `_ANEVirtualClient` and probe its methods

### 3B. Double-Buffer IOSurfaces
Create two sets of input/output IOSurfaces. While ANE evaluates on set A, the CPU stages data into set B. Swap on completion. Requires either async dispatch (3A) or a dedicated ANE dispatch thread.

### 3C. Pipeline Forward and Backward
Currently strictly sequential: all forward, then all backward. With async dispatch, layer N forward could overlap with layer N-1 backward. Requires careful dependency tracking.

### 3D. Concurrent GCD Queues for dW
The dW dispatch queue is serial. Making it concurrent allows the 7 sgemms per layer to run independently. The M4 Pro has 10 P-cores to fill.

---

## Phase 4 — SME2 Hybrid Engine (Week 4-6)

**Objective:** Use SME2 for CPU-side compute, running concurrently with ANE at zero contention. Target: 50-70% combined utilization.

### Key Insight
SME2 and ANE run on **completely independent hardware**. Benchmarks prove zero contention at 45.9 combined TOPS (37.4 GPU + 8.5 SME). Both can run simultaneously.

### 4A. dW Gradients on SME2
Replace `cblas_sgemm` with `ane::kernel::matmul_fp32` from the SME2 library. Benefits:
- 2x throughput over BNNS/Accelerate
- Runs concurrently with ANE evaluations (proven zero contention)
- Multi-thread across M dimension for additional scaling

### 4B. Write adam_fp32.s
The header declares `adam_fp32` but no assembly implementation exists. It is a straightforward streaming elementwise kernel:
```
m = beta1 * m + (1 - beta1) * g
v = beta2 * v + (1 - beta2) * g^2
p -= lr_corrected * m / (sqrt(v) + eps)
```
4 arrays, 4 scalars, pure streaming access. ~50 lines of SVE2 assembly using vlx4 loads/stores.

### 4C. Multi-Thread SME2 Matmul
Throughput tests prove 4-thread SME scales linearly (8.5 TOPS). Partition the M dimension across threads. Pre-allocate per-thread workspace buffers (matmul_int8 already supports caller-provided workspace; extend to fp32).

### 4D. Pre-Pack Weight Matrices
`matmul_fp32_pack_b` exists in the API. Pack weight matrices once after the Adam update, skipping Phase 1 preprocessing on every subsequent forward/backward call.

---

## Phase 5 — Kernel Fusion (Week 6-8)

**Objective:** Minimize dispatch overhead by maximizing work per ANE call. Target: 70-90% utilization.

### 5A. Fuse RMSNorm into Forward Kernels
Instead of separate RMSNorm dispatch + sdpaFwd dispatch, bake RMSNorm into the MIL program for sdpaFwd. One dispatch instead of two per attention block.

### 5B. Fuse Softmax + Causal Mask
Current decomposition: write -inf mask, then softmax reads it. Fuse into a single MIL program that applies the mask during softmax computation.

### 5C. Mega-Kernels
The peak benchmark (`inmem_peak.m`) proved that deep chains of 128+ convolutions reach near-peak TFLOPS in a single dispatch. The key insight: ANE utilization scales with compute-per-dispatch. Build "mega-kernels" that execute entire transformer blocks (RMSNorm + Attn + RMSNorm + FFN) in a single MIL program.

### 5D. Eliminate All malloc/free in Hot Path
Pre-allocate ring buffers for all temporary data. Replace per-layer malloc+memcpy for async dispatch with double-buffered pre-allocated blocks.

---

## Phase 6 — Uncharted Territory (Ongoing)

### Unexplored APIs
| API | Status | Potential |
|---|---|---|
| `_ANEPerformanceStats` | EXISTS, instantiates | Hardware counters — reveals exact stall sources |
| `_ANEVirtualClient` | EXISTS, never used | Multi-client concurrent submissions |
| `_ANEChainingRequest` | NOT FOUND at runtime | Chain evals without CPU roundtrip |
| `_ANESharedEvents` | NOT FOUND | GPU<->ANE zero-copy sync |
| `_ANEDeviceInfo` | EXISTS | SRAM size, core count, clock speed |
| `optionsPlist:` param | Always nil | Compilation options, SRAM strategy |
| `options:` on eval | Always empty `@{}` | Async mode? Priority? Profiling? |
| `procedureIndex:` | Only 0 works | Multi-function MIL programs? |

### GPU+ANE Inference Pipeline
The hybrid GPU-prefill + ANE-decode approach (8.8ms Stories110M, 12.0ms Qwen3-0.6B) is already implemented. This could be extended into a full local inference serving pipeline.

### INT8 Training
INT8 gives 1.88x throughput over FP16. The dynamic pipeline already supports `constexpr_affine_dequantize` for INT8 weights. Extending to full INT8 forward+backward would nearly double training throughput.

---

## Success Metrics

| Phase | Target Utilization | Key Indicator |
|---|---|---|
| Baseline | 5-9% | Current state |
| Phase 2 | 20-30% | RMSNorm + SiLU + RoPE on ANE |
| Phase 3 | 40-50% | Async dispatch working |
| Phase 4 | 50-70% | SME2 hybrid, zero CPU stalls |
| Phase 5 | 70-90% | Mega-kernels, full fusion |
| Phase 6 | 90%+ | Unexplored APIs unlocked |

---

## Architecture: The End State

```
┌─────────────────────────────────────────────────────┐
│                    Training Step                     │
├──────────────────────┬──────────────────────────────┤
│   ANE (16 cores)     │   SME2 (CPU matrix tiles)    │
│                      │                              │
│   Forward mega-      │   dW gradients (matmul_fp32) │
│   kernels:           │   Adam optimizer (adam_fp32)  │
│   RMSNorm+Attn+FFN   │   Weight transpose           │
│   fused per block    │   Weight re-staging           │
│                      │                              │
│   Backward mega-     │   All running CONCURRENTLY    │
│   kernels:           │   with ANE at zero contention │
│   Full backward      │                              │
│   including SiLU,    │                              │
│   RoPE, GQA reduce   │                              │
│                      │                              │
│   Async dispatch     │   Multi-threaded (4+ cores)  │
│   Double-buffered IO │   Pre-packed weight matrices  │
└──────────────────────┴──────────────────────────────┘
         Zero CPU fallbacks in the hot path
```
