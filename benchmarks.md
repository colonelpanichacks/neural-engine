# ANE Baseline Benchmarks — M4 Pro

**Date:** 2026-03-20
**Hardware:** Mac mini M4 Pro, 14-core (10P+4E), 48GB unified memory, 16-core ANE
**OS:** macOS 15 (Darwin 25.2.0)

---

## 1. Compute Throughput (TOPS)

### Solo Runs

| Engine | Threads | TOPS | Notes |
|---|---|---|---|
| GPU (Metal INT8) | - | 18.284 | char4 multiply-accumulate |
| SME2 (SMOPA INT8) | 4 | 8.392 | Raw ARM assembly outer product |
| BNNS (Accelerate INT8) | 4 | 4.118 | Apple's framework |
| NEON (FP32 FMLA) | 7 | 0.860 | Standard ARM vector FMA |

### Concurrent Runs (Key Combinations)

| Configuration | GPU | SME | BNNS | NEON | Total |
|---|---|---|---|---|---|
| GPU + SME | 18.498 | 8.464 | - | - | **26.961** |
| GPU + BNNS | 18.509 | - | 3.976 | - | 22.486 |
| SME + BNNS | - | 5.105 | 1.570 | - | 6.674 |
| GPU + SME + NEON | 14.909 | 7.472 | - | 0.708 | 23.090 |
| All four | 15.386 | 4.134 | 1.374 | 0.446 | 21.339 |

### Key Findings

- **GPU + SME run at zero contention**: 26.961 combined TOPS (99.9% of solo sum)
- **SME + BNNS contend heavily**: BNNS drops 62% (4.118 -> 1.570), SME drops 39% (8.392 -> 5.105). This confirms they share hardware.
- **SME is 2.04x faster than BNNS** on INT8 (8.392 vs 4.118)
- **GPU is completely independent** of CPU compute engines

---

## 2. ANE Coprocessor (H16G)

### INT8 vs FP16 Throughput

| Config | Weight (MB) | FP16 TOPS | INT8 TOPS | Speedup |
|---|---|---|---|---|
| 128x conv 512ch 64x64 | 64/32 | 18.51 | 34.90 | 1.89x |
| 64x conv 512ch 64x64 | 32/16 | 18.15 | 34.05 | 1.88x |
| 256x conv 256ch 64x64 | 32/16 | 16.49 | 30.70 | 1.86x |
| 128x conv 256ch 64x64 | 16/8 | 16.02 | 29.54 | 1.84x |
| 128x conv 384ch 64x64 | 36/18 | 18.24 | 33.02 | 1.81x |

**Peak: 34.90 TOPS INT8, 18.51 TOPS FP16.**

### SRAM Topology (Weight Spill Points)

| Channels | Weight (MB) | ms/eval | TFLOPS | GFLOPS/MB | Status |
|---|---|---|---|---|---|
| 512 | 0.5 | 0.092 | 0.37 | 730.3 | Peak efficiency |
| 1536 | 4.5 | 0.114 | 2.66 | 590.1 | Good |
| 3584 | 24.5 | 0.393 | 4.19 | 170.8 | Declining |
| 4096 | 32.0 | 1.053 | 2.04 | 63.8 | **SRAM cliff** |
| 5120 | 50.0 | 0.609 | 5.51 | 110.1 | Recovery (paging?) |
| 8192 | 128.0 | 3.888 | 1.10 | 8.6 | Heavy spilling |

**SRAM cliff at ~32MB weights.** Efficiency drops 3.7x when weights exceed this threshold. Individual kernel weight payloads should be kept under 32MB for peak performance.

---

## 3. Training Pipeline (Stories110M, 109M params)

### Step Timing Breakdown (steady state, steps 10-40)

| Component | ms/step | % of step | Description |
|---|---|---|---|
| ane_fwd | 17.9-18.6 | 19% | ANE forward pass (all layers) |
| ane_bwd | 25.7-26.8 | 27% | ANE backward pass (all layers) |
| io_fwd | 4.0-4.5 | 5% | IOSurface staging (fwd) |
| io_bwd | 8.6-10.2 | 10% | IOSurface staging (bwd) |
| rms | 1.8-1.9 | 2% | RMSNorm forward (CPU) |
| rms_bwd | 3.7-4.0 | 4% | RMSNorm backward (CPU) |
| silu | 6.3-7.3 | 7% | SiLU backward (CPU) |
| cls | 12.4-16.5 | 15% | Classifier matmul + loss (CPU cblas) |
| dw_copy | 2.8-3.4 | 3% | malloc+memcpy for async dispatch |
| cblas_wait | 0.0 | 0% | dW gradient wait (fully overlapped) |

**Total: ~95ms/step** (after warmup step 0 at 183ms)

### Time Distribution

```
ANE compute (fwd+bwd):   44.5ms  (46%)  <-- actual useful work
IO staging:              13.5ms  (14%)  <-- fp32<->fp16 conversion + lock/unlock
CPU fallbacks:           12.0ms  (13%)  <-- RMSNorm + SiLU backward
Classifier:              14.5ms  (15%)  <-- CPU cblas_sgemm
Overhead:                 6.5ms  (7%)   <-- dw_copy, misc
Async dW:                 0.0ms  (0%)   <-- fully overlapped (good)
```

### Compilation

- 10 kernels compiled once at startup: 616ms total
- Zero recompilations during training (dynamic weight approach)

### Training Stats

- Loss: 9.11 -> 8.10 in 50 steps (learning)
- FLOPs/step: 130.5 GFLOP (forward + backward)
- Sustained: ~1.37 TFLOPS effective (130.5 GFLOP / 95ms)
- ANE peak: 18.51 TFLOPS FP16
- **Utilization: 7.4%** (1.37 / 18.51)

---

## 4. API Discovery

### Previously Unknown APIs Found on the Benchmark System

**`_ANERequest` supports `sharedEvents:` parameter:**
```objc
+ requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:sharedEvents:
+ requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:sharedEvents:transactionHandle:
```
The `sharedEvents` property type is `_ANESharedEvents` -- GPU<->ANE synchronization IS in the API.

**`_ANERequest` has `completionHandler` property:**
```objc
@property completionHandler  [T@?,C,V_completionHandler]
- setCompletionHandler:
```
Async dispatch IS possible. The request object supports a completion block.

**`_ANEClient` has async methods:**
```objc
- buffersReadyWithModel:inputBuffers:options:qos:error:
- doBuffersReadyWithModel:inputBuffers:options:qos:error:
- doEnqueueSetsWithModel:outputSet:options:qos:error:
- enqueueSetsWithModel:outputSet:options:qos:error:
- doEvaluateWithModelLegacy:options:request:qos:completionEvent:error:
- doEvaluateWithModel:options:request:qos:completionEvent:error:
- evaluateRealTimeWithModel:options:request:error:
```
`buffersReady` + `enqueueSets` is a two-phase submit pattern. `completionEvent` parameter on evaluate methods enables async notification.

**`_ANEClient` has chaining support:**
```objc
- doPrepareChainingWithModel:options:chainingReq:qos:error:
- prepareChainingWithModel:options:chainingReq:qos:error:
```
Chaining IS accessible through `_ANEClient`, even though `_ANEChainingRequest` class was not found.

**`_ANEClient` has real-time methods:**
```objc
- beginRealTimeTask
- endRealTimeTask
- loadRealTimeModel:options:qos:error:
- unloadRealTimeModel:options:qos:error:
- evaluateRealTimeWithModel:options:request:error:
```
Real-time scheduling for latency-sensitive workloads.

**`_ANEModel` has `queueDepth` property:**
```objc
@property queueDepth  [Tc,N,V_queueDepth]
- setQueueDepth:
```
Queue depth IS configurable per model. Default is likely 1.

**`_ANEDeviceInfo` reports core count:**
```objc
+ numANECores  [I16@0:8]
+ numANEs      [I16@0:8]
+ aneArchitectureType
+ aneBoardType
+ aneSubType
```

**`_ANEPerformanceStats` construction methods:**
```objc
+ statsWithHardwareExecutionNS:
+ statsWithReconstructed:hardwareExecutionNS:aneStatsRawData:
+ statsWithRequestPerformanceBuffer:statsBufferSize:
- hwExecutionTime
- perfCounterData
- performanceCounters
```
Hardware execution time and raw perf counter data ARE accessible.

---

## 5. Optimization Targets (Ranked by Impact)

Based on the timing breakdown:

| Target | Potential Savings | Approach |
|---|---|---|
| **Classifier on ANE** | ~14.5ms (15%) | Move embed matmul to ANE or use SME2 matmul_fp32 |
| **IO staging elimination** | ~13.5ms (14%) | Keep data in FP16 throughout, minimize conversions |
| **SiLU backward on ANE** | ~6.8ms (7%) | Add to ffnBwdW2t MIL kernel (sigmoid+mul available) |
| **RMSNorm on ANE** | ~5.8ms (6%) | Proven in static pipeline (reduce_sum+pow+mul) |
| **Async ANE dispatch** | ~20ms+ (overlap) | Use completionHandler + buffersReady/enqueueSets |
| **Queue depth > 1** | unknown | Set queueDepth on _ANEModel, pipeline submissions |
| **dW on SME2** | ~3ms saved + overlap | Replace cblas_sgemm with 2x-faster SME2 matmul |

**Combined potential: 40-50ms savings per step (42-53% reduction)**, bringing step time from ~95ms to ~45-55ms.

---

## 6. Phase 2 Probe Results (Async & Device Discovery)

### _ANEDeviceInfo

| Property | Value |
|---|---|
| numANECores | 16 |
| numANEs | 1 |
| aneArchitectureType | Probe needed |
| aneBoardType | Probe needed |

### Async Dispatch (completionHandler)

**Confirmed working.** `_ANERequest` supports `setCompletionHandler:` with a void block. The handler fires after ANE eval completes.

- Handler latency: **0.105ms** (time from eval return to handler invocation)
- Pattern: set handler on request before eval, handler fires post-completion
- This enables CPU work to overlap with ANE execution

### Queue Depth

`_ANEModel` `queueDepth` defaults to **127**. This means up to 127 concurrent evaluations can be pipelined to the ANE without blocking. Significant potential for throughput when combined with async dispatch.

### Real-Time Scheduling

`evaluateRealTimeWithModel:options:request:error:` **works** on the test system. Provides priority scheduling for latency-sensitive workloads.

### Eval Latency

| QoS | Latency | Evals/sec |
|---|---|---|
| 21 (default) | 0.185ms | 5,418 |

### API Status Summary

| API | Status | Notes |
|---|---|---|
| completionHandler | **WORKS** | Async dispatch confirmed |
| queueDepth | **WORKS** | Default 127, settable |
| evaluateRealTime | **WORKS** | Real-time scheduling |
| perfStats | BLOCKED | Needs statType protocol conformance |
| prepareChainingWithModel | BLOCKED | Needs NSSecureCoding-conformant request |
| buffersReady/enqueueSets | BLOCKED | Incorrect argument types |
| doEvaluateWithModel:completionEvent: | NOT FOUND | May not exist on macOS 15 |

---

## 7. Pipeline Optimization Results (Phase 2 Implementation)

### Changes Applied

1. **Concurrent dW queue**: `DISPATCH_QUEUE_SERIAL` -> `DISPATCH_QUEUE_CONCURRENT` -- allows parallel gradient sgemms across the M4 Pro's 10 P-cores
2. **Async ANE eval with CPU overlap**: dW copy+dispatch overlapped with `ffnBwdW13t` and `wotBwd` ANE evals via `dispatch_semaphore`
3. **Concurrent qBwd + kvBwd**: Both dispatched simultaneously to ANE (independent kernels, independent IOSurfaces)
4. **dV read + GQA reduce overlapped with sdpaBwd2**: Read dV from sdpaBwd1 while sdpaBwd2 runs on ANE
5. **Vectorized RoPE backward**: Precomputed cos/sin table, vDSP rotations (eliminated 2.4M `cosf`/`sinf` calls per step)
6. **Vectorized Adam optimizer**: Replaced scalar loop with vDSP bulk operations
7. **Eliminated hot-path allocations**: Pre-allocated dx2_scaled, dx_kv, dx_rms1, dx_rms_final; pre-allocated RMSNorm temp buffers (ss, rrms, dot)
8. **Vectorized elementwise backward ops**: dx_attn += dx_kv, dy = dx_rms1 + dx2 via vDSP_vadd

### Results (Stories110M, 50 steps)

| Metric | Baseline | Optimized | Change |
|---|---|---|---|
| Step time (steady state) | 95ms | 88.3ms | **-7.1%** |
| dw_copy | 2.8-3.4ms | 0.6ms | **-79%** |
| rms_bwd | 3.7-4.0ms | 3.4-4.2ms | -8% |
| Compile time | 616ms | 442ms | -28% |
| Loss @ step 50 | 8.10 | 8.10 | Identical |
| Grad norms | 7.12 | 7.12 | Identical |

**Effective throughput: 1.48 TFLOPS** (130.5 GFLOP / 88.3ms), up from 1.37 TFLOPS.

### Remaining CPU Bottlenecks (ranked by impact)

| Bottleneck | Time/step | % of step | Approach |
|---|---|---|---|
| Classifier (cblas_sgemm) | 13-15ms | 15-17% | Move to ANE or SME2 |
| IO staging (fp32<->fp16) | 12-15ms | 14-17% | Keep data in FP16 throughout |
| SiLU backward (vDSP) | 6.0-6.8ms | 7-8% | Fuse into ffnBwdW2t MIL kernel |
| RMSNorm fwd+bwd (vDSP) | 5.2-6.0ms | 6-7% | Move to ANE (proven in static pipeline) |

---

## 8. Phase 3: Fused SiLU Backward + Classifier Investigation

### Changes Applied

1. **Fused SiLU backward + W13t ANE kernel**: Combined CPU SiLU derivative + ffnBwdW13t matmul into single ANE kernel (`ffnBwdFused`). The kernel takes dsilu_raw + h1 + h3 as input, computes SiLU backward (sigmoid, elementwise ops) and both W1t/W3t matmuls on ANE, outputting dx_ffn + dh1 + dh3.
2. **Batched IOSurface locks**: Combined 4 separate lock/unlock pairs for sdpaBwd1/sdpaBwd2 IO into single lock per surface.

### Results (Stories110M, 50 steps)

| Metric | Phase 2 | Phase 3 | Change |
|---|---|---|---|
| Step time (steady state) | 88.3ms | 86.4ms | **-2.1%** |
| SiLU backward (CPU) | 6.0-6.8ms | **0.0ms** | **Eliminated** |
| ANE backward | 25.7-26.8ms | 28-29ms | +10% (more work) |
| Loss @ step 50 | 8.10 | 8.10 | Identical |
| Grad norms | 7.12 | 7.12 | Identical |

**Effective throughput: 1.51 TFLOPS** (130.5 GFLOP / 86.4ms), up from 1.48 TFLOPS.

### Classifier Optimization Attempts (Blocked)

**ANE classifier (FP16)**: Forward pass loses precision -- logits in FP16 all round to ~ln(1/V), loss stuck at 9.1275. Backward works but IO staging overhead (18.9MB for CV=9205 channels) exceeds compute savings. **Verdict: classifier must stay FP32 on CPU.**

**SME2 matmul**: `matmul_fp32_tn` produces incorrect gradients (~1000x too small at step 0). Cause undiagnosed. All 172 unit tests pass, suggesting a layout/convention mismatch in the classifier context. **Verdict: needs further investigation.**

### Updated Timing Breakdown

```
ANE compute (fwd+bwd):   46ms   (53%)  <-- actual useful work
IO staging:              16ms   (19%)  <-- fp32<->fp16 conversion + lock/unlock
Classifier:              14ms   (16%)  <-- CPU cblas_sgemm (FP32 required)
RMSNorm fwd+bwd:          6ms   (7%)  <-- CPU vDSP
Overhead:                  5ms   (6%)  <-- dw_copy, misc
```

### Remaining Bottlenecks

| Bottleneck | Time/step | % | Status |
|---|---|---|---|
| Classifier (cblas_sgemm) | 13-16ms | 16% | **Blocked** -- needs FP32 precision |
| IO staging (fp32<->fp16) | 12-16ms | 19% | Architectural change needed |
| RMSNorm fwd+bwd | 5-6ms | 7% | Must fuse into existing kernels |
| SiLU backward | 0ms | 0% | **Done** -- moved to ANE |

---

## 9. SME2 Unit Test Results

All 172 tests passed, zero failures:

| Suite | Tests | Status |
|---|---|---|
| Activations | 42 | PASS |
| Elementwise | 36 | PASS |
| Losses | 10 | PASS |
| Matmul | 18 | PASS |
| Memory Ops | 24 | PASS |
| Reductions | 30 | PASS |
| Softmax | 12 | PASS |
