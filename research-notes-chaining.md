# ANE Kernel Chaining Research

## Status: Firmware-Level Blocked

The chaining API object hierarchy is fully reverse-engineered and validate passes,
but `prepareChainingWithModel` fails at the firmware level (error code 15:
"ANEProgramChainingPrepare() Failed"). Models compiled via raw MIL don't contain
symbol routing metadata needed for chaining.

## Verified Object Hierarchy

```
_ANEChainingRequest
  inputs:  [@[_ANEBuffer]]
  outputs: [@[_ANEIOSurfaceOutputSets]]
  lbInputSymbolId: NSArray<NSNumber>
  lbOutputSymbolId: NSArray<NSNumber>
  procedureIndex: NSNumber
  signalEvents: NSArray<_ANESharedSignalEvent>
  transactionHandle: NSNumber
  fwEnqueueDelay: NSNumber
  memoryPoolId: NSNumber

_ANEBuffer
  initWithIOSurfaceObject:symbolIndex:source:
  - ioSurfaceObject: _ANEIOSurfaceObject
  - symbolIndex: NSNumber (0-based symbol index)
  - source: int64 (0=input producer, 1=output producer)

_ANEIOSurfaceOutputSets
  initWithstatsSurRef:outputBuffer:
  - statsSurRef: IOSurfaceRef (for perf stats)
  - outputBuffer: NSArray<_ANEBuffer>
```

## Option Keys (exported from AppleNeuralEngine.framework)

| Symbol | String Value | Purpose |
|--------|-------------|---------|
| kANEFEnableFWToFWSignal | "kANEFEnableFWToFWSignal" | Firmware-to-firmware signaling without CPU |
| kANEFDisableIOFencesUseSharedEventsKey | "kANEFDisableIOFencesUseSharedEventsKey" | Use shared events instead of IOFences |
| kANEFMemoryPoolIDKey | "kANEFMemoryPoolIDKey" | Memory pool for intermediate buffers |
| kANEFSkipPreparePhaseKey | "kANEFSkipPreparePhaseKey" | Skip prepare phase |
| kANEFEnableLateLatchKey | "kANEFEnableLateLatchKey" | Late-latch input binding |

## _ANEModel Properties

- queueDepth: 127 (max concurrent requests)
- programHandle: uint64 (firmware program handle)
- intermediateBufferHandle: uint64 (shared memory handle)
- inputSymbolIndicesForProcedureIndex: → (no indexes) for MIL-compiled models
- outputSymbolIndicesForProcedureIndex: → (no indexes) for MIL-compiled models

## Root Cause of Failure

Models compiled via `_ANEInMemoryModel` (raw MIL text → ANE) don't contain symbol
routing metadata. `inputSymbolIndicesForProcedureIndex:` returns `(no indexes)`.
The compiled ANE program format needs explicit symbol/procedure annotations for
chaining to work. These are likely added by CoreML's pipeline during `.mlpackage`
compilation, not available in the raw MIL path.

## Next Steps to Unblock

1. Try compiling via CoreML (`.mlpackage` → `.mlmodelc` → load) instead of raw MIL
2. Investigate the compiled ANE program binary format for symbol metadata fields
3. Check if `_ANEInMemoryModel` has compile options that enable symbol generation
4. Reverse-engineer how CoreML populates symbol indices during compilation
5. Look at `_ANEProcedureData` and multi-procedure models — chaining may be
   between procedures of the SAME model, not separate models

## Test Code

`chain_probe.m` in `ane-training/training/training_dynamic/` — standalone test
demonstrating the full chaining API usage with correct object types.
