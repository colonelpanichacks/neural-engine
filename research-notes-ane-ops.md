# ANE MIL Operator Compatibility & Optimization Research

## MIL Operator Availability on ANE (H16G, M4 Pro)

Tested via raw MIL compilation at DIM=16, SEQ=64, fp16.

### Working Operators

| Op | Syntax | Notes |
|---|---|---|
| `mul` | `mul(x=a, y=b)` | Element-wise, broadcast-capable |
| `add` | `add(x=a, y=b)` | Element-wise |
| `sub` | `sub(x=a, y=b)` | Element-wise |
| `real_div` | `real_div(x=a, y=b)` | Element-wise division |
| `sigmoid` | `sigmoid(x=a)` | Used for SiLU = x * sigmoid(x) |
| `relu` | `relu(x=a)` | |
| `tanh` | `tanh(x=a)` | |
| `sqrt` | `sqrt(x=a)` | |
| `exp` | `exp(x=a)` | |
| `exp2` | `exp2(x=a)` | |
| `abs` | `abs(x=a)` | |
| `square` | `square(x=a)` | |
| `floor` | `floor(x=a)` | |
| `ceil` | `ceil(x=a)` | |
| `clip` | `clip(x=a, alpha=fp16(lo), beta=fp16(hi))` | |
| `sign` | `sign(x=a)` | |
| `round` | `round(x=a)` | |
| `sin` | `sin(x=a)` | |
| `cos` | `cos(x=a)` | |
| `erf` | `erf(x=a)` | Enables GELU: x * 0.5 * (1 + erf(x/sqrt(2))) |
| `pow` | `pow(x=a, y=scalar)` | **pow(x, -0.5) = rsqrt!** |
| `matmul` | `matmul(transpose_x=bF, transpose_y=bF, x=a, y=b)` | Core compute |
| `reduce_sum` | `reduce_sum(x=a, axes=ax, keep_dims=kd)` | Works on all axes |
| `reduce_mean` | `reduce_mean(x=a, axes=ax, keep_dims=kd)` | Works on all axes |
| `layer_norm` | `layer_norm(x=a, axes=ax, gamma=g, beta=b, epsilon=eps)` | **Native LayerNorm!** |
| `reshape` | `reshape(shape=s, x=a)` | |
| `transpose` | `transpose(perm=p, x=a)` | |
| `slice_by_size` | `slice_by_size(x=a, begin=b, size=s)` | Dynamic weight slicing |
| `concat` | `concat(values=(a,b), axis=N, interleave=bF)` | |

### Non-Working Operators

| Op | Error | Workaround |
|---|---|---|
| `rsqrt` | InvalidMILProgram | Use `pow(x, -0.5)` |
| `log` | InvalidMILProgram | None found |
| `inverse` | InvalidMILProgram | Use `real_div(1, x)` |
| `softmax` | InvalidMILProgram | Manual: exp(x-max) / sum(exp(x-max)) |
| `instance_norm` | InvalidMILProgram | Manual construction |

## RMSNorm on ANE

### Status: Works standalone, cannot fuse with matmul

RMSNorm compiles and evaluates correctly at DIM=1024, SEQ=256:
```
x * pow(reduce_mean(x^2, axis=channel) + eps, -0.5)
```
- **Performance**: 0.166-0.175 ms/eval (verified correct output)
- **Correctness**: Matches CPU reference to fp16 precision

### Fusion Limitation (0x1d)

Combining `reduce_mean` + `matmul` in the same MIL program fails at
**eval time** with status 0x1d (hardware resource limit). This occurs at
ALL tested sizes (DIM=256-1024, OC=64-2048, SEQ=64-256).

The ANE hardware cannot allocate both reduction and matmul units in the
same program. This is a fundamental H16G architecture constraint, not a
size/memory issue — the program compiles and loads successfully, only
failing when the firmware attempts to schedule it.

### Implication

RMSNorm must remain a separate kernel from matmul. In the training pipeline,
CPU RMSNorm (0.09ms via vDSP) is faster than ANE RMSNorm (0.17ms) when
accounting for IOSurface staging overhead.

## Concurrent Submission Pipeline

### ANE supports concurrent kernel execution (1.24-1.30x speedup)

Submitting two independent ANE kernels concurrently via separate dispatch
queues yields significant speedup over sequential submission:

| Pattern | Time/pair | Speedup |
|---|---|---|
| Sequential (main thread) | 0.617 ms | 1.00x |
| Sequential (async queue) | 0.651 ms | 0.95x |
| Concurrent (2 queues) | 0.495 ms | 1.24x |
| Pipeline (fire both, wait) | 0.476 ms | 1.30x |

### Dispatch overhead: ~0.035 ms/pair (negligible)

### Application limitation

Data dependencies between transformer layer kernels prevent most concurrent
submission. Each kernel's output feeds the next kernel's input. The only
viable pipeline is between independent operations (e.g., dW gradient
sgemm on CPU while ANE runs the next backward kernel — already implemented).

## Eval Method Comparison

| Method | ms/eval | Notes |
|---|---|---|
| `evaluateRealTimeWithModel:` | 0.28 | Fastest — currently used |
| `evaluateWithQoS:` | 0.30 | ~7% slower |
| `evaluateWithQoS:` + completionHandler | 0.42-0.63 | 40-120% SLOWER |

**completionHandler adds overhead** — block allocation + dispatch cost.
The current semaphore-based async pattern is optimal.

## Test Files

- `test_mil_ops.m` — MIL operator compatibility probe
- `test_rmsnorm_ane3.m` — RMSNorm on ANE correctness test
- `test_rmsnorm_fuse2.m` — RMSNorm+matmul fusion size sweep
- `test_rmsnorm_pipeline.m` — RMSNorm/matmul pipelining benchmark
- `bench_eval_methods.m` — Eval method comparison
- `bench_pipeline.m` — Concurrent submission benchmark
