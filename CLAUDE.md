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
   - **Forward**: Per-layer: CPU RMSNorm -> ANE sdpaFwd(+Wo) -> residual -> CPU RMSNorm -> ANE ffnFused -> residual
   - **Backward**: Per-layer: ANE ffnBwdFull -> ANE wotBwd -> ANE sdpaBwd1 -> ANE sdpaBwd2 -> ANE qkvBwd -> CPU RMSNorm backward. dW gradients computed on CPU via cblas_sgemm, overlapped with ANE via dispatch_group.
   - **Adam**: vDSP-vectorized Adam optimizer with gradient accumulation and cosine LR schedule.

5. **Async Overlap**: ANE kernels dispatched via `ane_eval_req_async()` with CPU work (memcpy, fp16 conversion, gradient accumulation) overlapped during ANE execution.

### Key Technical Details

- **GQA (Grouped Query Attention)**: Qwen3-0.6B uses 16 query heads / 8 KV heads (ratio 2). Q_DIM (2048) != DIM (1024). KV heads are tiled for SDPA forward and reduced in backward.
- **fp16 Activation Storage**: Activations flowing between ANE kernels (Q, K, V, h1, h3, silu_out, dsilu) are stored as fp16 to eliminate fp16->fp32->fp16 roundtrip conversions. Only converted to fp32 where CPU computation actually needs it.
- **Dynamic Weights**: Weights packed into IOSurface spatial dimension, sliced inside MIL via `slice_by_size`. Weights can be updated (Adam step + restaging) without recompiling kernels.
- **119 Kernel Compile Limit**: ANE enforces a per-process compile limit. Training uses checkpoint + exec() restart to work around this. Current pipeline compiles 7 kernels (well within limit).

### ANE Kernel Inventory (Dynamic Pipeline)

| Kernel | Purpose | Input Channels | Spatial |
|--------|---------|---------------|---------|
| sdpaFwd | QKV projection + SDPA + Wo (fused GQA) | DIM | SEQ + Q_DIM + 2*KV_DIM + Q_DIM |
| ffnFused | SwiGLU FFN (W1,W3,SiLU,W2) | DIM | 2*SEQ + 3*HIDDEN |
| ffnBwdFull | W2^T + SiLU bwd + W1^T/W3^T (fully fused) | HIDDEN | 3*SEQ + 3*DIM |
| wotBwd | dy @ Wo^T -> da | DIM | SEQ + Q_DIM |
| sdpaBwd1 | SDPA backward: dV, probs, dp | 4*Q_DIM | SEQ |
| sdpaBwd2 | SDPA backward: dQ, dK | 2*SCORE_CH + 2*Q_DIM | SEQ |
| qkvBwd | dq@Wq + dk@Wk + dv@Wv -> dx_attn (fused) | Q_DIM | 3*SEQ + 3*DIM |

### CPU Operations (Not on ANE)

- **RMSNorm forward/backward**: vDSP vectorized. RMSNorm on ANE works standalone (`pow(x,-0.5)` instead of `rsqrt`), but cannot fuse with matmul (0x1d at eval time — ANE can't co-schedule reduction + matmul hardware units). CPU (0.09ms) is faster than ANE (0.17ms + IO staging).
- **RoPE backward**: vDSP vectorized with precomputed cos/sin tables.
- **Cross-entropy loss**: `dispatch_apply` parallelized across 256 tokens with per-token scratch buffers.
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

| Metric | SEQ=256 | SEQ=512 |
|--------|---------|---------|
| Step time (50-step avg) | ~210ms | ~412ms |
| Best step | 199.9ms | ~400ms |
| ANE forward | ~68ms | ~139ms |
| ANE backward | ~107ms | ~193ms |
| IO staging (fwd) | ~4ms | ~9ms |
| IO staging (bwd) | ~14ms | ~17ms |
| RMSNorm (fwd+bwd) | ~10ms | ~24ms |
| Cross-entropy | ~4ms | ~8ms |
| RoPE backward | ~1ms | ~2ms |
| FLOPs/step | 676B | 1,353B |
| Tokens/step | 256 | 512 |
| Tokens/sec | ~1,220 | ~1,243 |
| Compute utilization | ~17.4% | ~17.9% |

### ANE Hardware Notes

- ANE operates in fp16 only. All IOSurface data is fp16.
- ANE has ~32MB SRAM; efficiency drops ~3.7x when weight footprint exceeds this.
- Peak throughput: 18.5 TOPS (fp16), 34.9 TOPS (int8).
- Current utilization is ~16-18% of peak compute — bottlenecked by dispatch overhead (~0.28ms/eval, ~55ms total for 196 evals/step) and serial data dependencies between layers.
- ANE is the H16G coprocessor, separate from CPU/GPU. Accessed via `AppleNeuralEngine.framework` private APIs.
- **Channel-offset slicing** in MIL (begin[1]!=0) is significantly slower than spatial-offset slicing (begin[3]!=0). Prefer packing data in the spatial dimension.
- **IOSurface sharing** between kernel output→input works when output IC < shared surface IC (ANE only writes declared output channels, preserving pre-staged data at higher channels).

### ANE API Surface (Reverse-Engineered)

| API Path | Status | Notes |
|----------|--------|-------|
| `_ANEClient.doEvaluateDirectWithModel` | **Production** | 0.28ms/eval, bypasses XPC |
| `_ANEProgramForEvaluation.processRequest` | Works | Same speed, different params |
| `_ANEClient.evaluateRealTimeWithModel` | Works (serial) | 14% slower with async queues |
| `prepareChainingWithModel` | Error 15 | Firmware-gated, not compile-time |
| `doBuffersReady` + `doEnqueueSets` | No-op | Needs chaining prep first |
| `processInputBuffers` / `processOutputSet` | Returns false | Two-phase split blocked |
| `_ANEVirtualClient.completionEvent` | Daemon-only | Lives in aned, not userspace |
| IOKit `H11ANEIn` external methods | Unsupported | Entitlement-blocked (all 32 selectors) |
| `neuralEngineCompilerOptions` | Read-only | `"EnableLowEffortCPAllocation=true"` |
| Performance counters | Non-functional | Never invoked during eval |

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
5. **Kernel fusion round** (sdpaFwd+Wo, ffnBwdFull, qkvBwd): 12→7 kernels, ~253ms/step (io_bwd -52%, step -23%). sdpaBwd1+2 fusion blocked by ANE compiler "cycle path" bug on complex DAGs.
6. **Fused cvt_scatter_f32_f16** + W2^T pre-transpose: eliminated 8 intermediate fp16 buffers (~84MB/step saved), ~258ms/step.
7. **IO staging overlap**: Pre-staged sdpaBwd1/2 inputs during wotBwd execution (3MB overlapped), io_bwd 30.6→26.9ms.
8. **Cross-entropy parallelization**: `dispatch_apply` across 256 tokens, cls 13.4→6.1ms (-54%).
9. **Forward IO overlap**: Deferred backward-only copies (attn_out, Q/K/V, h1/h3/silu) to async queue during ANE execution, io_fwd 10.8→5.8ms (-46%). ~247ms/step.
10. **`doEvaluateDirectWithModel`**: Bypasses XPC overhead for ANE eval, 10% faster per dispatch (0.287ms vs 0.317ms). ~6ms/step saved across 196 evals.
11. **NEON intrinsics**: RoPE backward (vfmaq/vfmsq FMA, eliminated 7 vDSP calls/pair): 5.6→2.5ms (-54%). RMSNorm fwd (NEON FMA loops): 4.9→3.0ms (-35%). RMSNorm bwd: 11.1→9.6ms (-14%). ~235ms/step.
12. **h1/h3 forward pre-staging**: Scatter h1/h3 directly into ffnBwdFull input IOSurface during forward pass (async, overlapped with ANE). Eliminates 7.2ms of backward critical-path strided copies. Also freed 84MB memory (h1_fp16/h3_fp16 buffers). io_bwd ffn_w: 11.1→1.7ms (-85%).
13. **Parallel `dispatch_apply` scatter**: Multi-core `cvt_scatter_f32_f16_par` for all activation writes (>512 channels). qkv_w: 6.1→4.2ms, ffn_w: 1.7→1.3ms, wot_w: 1.6→1.2ms. io_bwd: 18.2→14.5ms.
14. **dV backward pre-staging**: Pre-stage dV into qkvBwd input IOSurface during sdpaBwd2 overlap window. qkv_w: 4.2→2.9ms (-31%). io_bwd: ~14.5→~13ms.
15. **qkvBwd async + dW overlap**: Made qkvBwd eval async, overlap dWq/dWk/dWv malloc+memcpy+dispatch with ANE execution. gap: 5.0→1.7ms (-3.3ms). ~236ms/step.
16. **fp16 RoPE + direct staging**: Eliminated fp16→fp32→fp16 roundtrip for dQ/dK path. RoPE backward in native fp16 NEON (8-wide `vfmaq_f16`), GQA reduce in fp16, scatter fp16 directly to qkvBwd input. fp32 conversion for dW sgemm moved to qkvBwd overlap window. s2r: 4.3→2.1ms (-51%), rope: 2.9→1.2ms (-59%), gqa: 3.0→0ms (folded into s2r). ~230ms/step.
17. **IOSurface sharing (sdpaBwd1→sdpaBwd2)**: Reordered sdpaBwd1 output concat to (probs, dp, dV) and bound sdpaBwd1's output IOSurface as sdpaBwd2's input. probs+dp land directly where sdpaBwd2 expects them — eliminates 4MB s2 memcpy. Only dV read (1MB) + Q re-stage (1MB) needed = 2MB vs 4MB (-50% bandwidth). s2: 3.3→2.1ms (-36%). ~228ms/step.
18. **Contiguous logits layout**: Changed classifier sgemm to output logits as [SEQ, CV] (row-per-token) instead of [CV, SEQ]. Eliminates strided `cblas_scopy` gather/scatter in cross-entropy (9205 elements × stride-256 per token). CE operates directly on contiguous rows. Folded loss_scale into CE gradient (eliminates separate vDSP_vsmul pass over 2.35M elements). cls: 6.0→4.3ms (-28%). ~226ms/step.
19. **IOSurface sharing (wotBwd→sdpaBwd1)**: Reordered sdpaBwd1 input to (da, Q, K, V) and bound wotBwd's output as sdpaBwd1's input. wotBwd writes da to ch[0:Q_DIM], Q/K/V pre-staged at ch[Q_DIM:4*Q_DIM] before dispatch. Eliminates s1 memcpy entirely. s1: 1.3→0.0ms. ~224ms/step.
20. **Q/K/V pre-staging overlap**: Moved GQA tile + Q/K/V staging into sdpaBwd1 input from wotBwd critical path to ffnBwdFull ANE overlap window. Q/K/V staging now fully hidden behind ANE execution. wot_w: 4.5→1.3ms (-71%). ~220ms/step.
21. **fp16 dV path**: Eliminated fp16→fp32→fp16 roundtrip in dV backward flow. Read dV from shared surface as raw fp16 memcpy (no conversion), GQA reduce in fp16, scatter fp16 directly to qkvBwd input. s2: 2.2→1.7ms (-23%). ~217ms/step.
22. **8-wide NEON RMSNorm unrolling**: Unrolled all loops in rmsnorm (fwd+bwd) from 4-wide to 8-wide NEON (2× float32x4_t per iteration). Dual accumulators reduce loop overhead and improve instruction pipelining. rms_bwd: 22→13ms at SEQ=512 (-40%), rms_fwd: 11.4→10.4ms at SEQ=512 (-9%). Combined RMSNorm: 33→24ms at SEQ=512. ~215ms/step (SEQ=256), ~420ms/step (SEQ=512).

23. **fp16 attn_out storage**: Store attention output as fp16 (was fp32). Eliminates fp16→fp32 conversion in deferred forward copy (4MB→2MB per layer). fp32 conversion deferred to dWo capture in overlap window. Reduces deferred copy pressure at SEQ=512.
24. **Parallel Adam weight restaging**: `dispatch_apply` across layers for weight restaging in Adam step (cvt_scatter weight updates parallelized across 28 layers). Adam step: 983→525ms (-47%).
25. **Fused fp16 residual + direct layer_in write**: Eliminated separate `cvt_f16_f32` + `vDSP_vsma` for attention residual — single NEON pass reads fp16 o_out, converts, scales, and adds in-place. Also writes ffnFused output directly to next layer's `layer_in` buffer (eliminates extra memcpy). io_fwd: 5.0→3.8ms (-24%). ~210ms/step.

### ANE Compiler/Hardware Limitations Discovered

- **"Graph has a cycle path"**: ANE's MIL compiler rejects DAGs where tensors fan out to multiple matmuls that reconverge (e.g., Q used in both attention score and dK computation). Not a real cycle — a compiler bug.
- **Status 0x1d (compile)**: Hardware resource limit. Hit when RMSNorm channel dim >= 1024 was attempted as part of a fused kernel.
- **Status 0x1d (eval)**: `reduce_mean` + `matmul` in the same program fails at eval time at ALL sizes. ANE cannot co-schedule reduction and matmul hardware units. Compiles/loads fine, only fails when firmware attempts scheduling.
- **`rsqrt` invalid**: Not a valid MIL op. Use `pow(x, -0.5)` instead. See `research-notes-ane-ops.md` for full operator compatibility table.
- **completionHandler adds overhead**: `setCompletionHandler:` on `_ANERequest` is 40-120% slower than semaphore-based async. Block allocation + dispatch cost.
- **Concurrent submission 1.3x speedup**: Two independent ANE kernels submitted concurrently via separate dispatch queues execute 1.24-1.30x faster than sequential. Limited by data dependencies in transformer layers.
- **`evaluateRealTimeWithModel`**: 4-9% faster in serial benchmarks vs `doEvaluateDirectWithModel`, but 14% *slower* with async dispatch queues. Unusable for training. Uses different ANE scheduling path.
- **ANE chaining API** (`_ANEChainingRequest`): Takes `_ANEBuffer` (IOSurfaceObject + symbolIndex + source), `_ANEIOSurfaceOutputSets` (statsSurface + outputBuffers), `_ANEOutputSetEnqueue` (procedureIndex + signalValue). `doPrepareChainingWithModel` returns error 15 ("Program chaining prepare error") — likely requires loopback buffer configuration or special compile options.
- **IOSurfaceSharedEvent signaling**: ANE does NOT signal `IOSurfaceSharedEvent` via `doEvaluateDirectWithModel` or `evaluateRealTimeWithModel`, regardless of agentMask (0x0-0xFF) or eventType (0-2). Signaling likely only works through the chaining/enqueue path.
- **QoS has negligible effect**: QoS values 0 (realTime), 9 (utility), 21 (default), 25 (background), 33 (userInteractive) all give 0.274-0.303 ms/eval when ANE is sole user.
- **Performance counters not accessible**: `_ANEModel.perfStatsMask` exists (unsigned int, settable) and `_ANEVirtualClient +updatePerformanceStats:performanceStatsLength:perfStatsRawIOSurfaceRef:performanceStatsRawLength:hwExecutionTime:` is the callback, but it's never invoked during eval. The mask is stored locally but never propagated to the driver. `_ANEPerformanceStatsIOSurface` (objectWithIOSurface:statType:) passes `_ANERequest validate` but ANE never writes to the IOSurface. Apple-internal infrastructure, non-functional via available APIs.
- **dW gradient overlap is complete**: `dispatch_group_wait(dw_grp)` at forward layer start always returns immediately (dwait=0.0ms). CPU dW sgemm fully overlaps with ANE execution — no layer-pipelining opportunity.
- **Mixed contiguous+strided writes cause cache thrashing on M4 Pro**: Writing to both a contiguous fp32 buffer and a strided fp16 IOSurface in the same inner loop causes 2-3x regression vs separate passes. Applies to fused RMSNorm+scatter, fused RoPE+scatter, and any dual-destination write pattern. Root cause: L1 cache line eviction from alternating between two incompatible stride patterns.
- **ANE chaining is firmware-gated**: Error 15 is not a compile-time or options problem. All compile option keys (`enableChaining`, `ANEChainingEnabled`, etc.) are blindly passed through without affecting the compiled artifact. The `prepareChainingWithModel` failure occurs at the ANEd/firmware level regardless of queueDepth, compile options, or model configuration. The chaining infrastructure exists in the framework but is gated at the driver/firmware scheduler.
- **IOKit ANE access is entitlement-blocked**: `H11ANEIn` user client opens with type=1 but all 32 external method selectors return `kIOReturnUnsupported`. The driver enforces private entitlements before dispatching any method. `_ANEVirtualClient` (with `completionEvent` dispatch) is daemon-side only — `_ANEClient._virtualClient` is always nil in userspace.
- **Two-phase dispatch (`buffersReady`+`enqueueSets`) is a no-op**: Both `doBuffersReadyWithModel` and `doEnqueueSetsWithModel` return success (ret=0) with `_ANEOutputSetEnqueue` objects, but ANE does not execute the kernel. The two-phase path requires prior `prepareChainingWithModel` success which is firmware-blocked.
- **`_ANEProgramForEvaluation.processRequest` is identical to `doEvaluateDirectWithModel`**: Same speed (0.155ms), same underlying IOKit call. No dispatch shortcut exists at any framework level.
- **Dispatch overhead floor**: ~0.155ms for small kernels, ~0.28ms for training-sized kernels. 196 evals/step × 0.28ms = ~55ms dispatch overhead (26% of step time). This is the dominant bottleneck but unreducible from userspace.

### IOKit / Hardware Registry (M4 Pro)

- **ANE device**: `ane0` at `arm-io@10F00000/ane0@84000000`, compatible `ane,t8020`
- **Driver**: `H11ANEIn` (bundle `com.apple.driver.AppleH16ANEInterface`), firmware loaded
- **HAL**: `AppleT6041ANEHAL` (M4 Pro specific)
- **Load balancer**: `H1xANELoadBalancer` at `/IOResources/ANEDriverRoot` with 9 direct-path clients
- **DART/IOMMU**: `dart-ane0`, `mapper-ane0`, `mapper-ane0-mpm`
- **Device properties**: SubType=7, BoardType=272, MinorVersion=17, MaxPowerState=1
- **65 ANE-related classes** in AppleNeuralEngine.framework

## Git Conventions

- `master`: stable releases
- `dev`: active development (work here)
- Never commit directly to master; merge from dev when tested.
