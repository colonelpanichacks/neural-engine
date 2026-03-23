// chain_probe.m — Test ANE kernel chaining via _ANEChainingRequest
// Build: make chain_probe (or manually: same flags as train.m + -include models/stories110m.h)
//
// Uses the existing MIL/compile infrastructure from config.h, io.h, mil_dynamic.h.
// Tests: sequential eval → chaining → buffersReady/enqueueSets pipeline
#import <Foundation/Foundation.h>
#include "cpu_ops.h"
#include "mil_dynamic.h"  // includes io.h → config.h

// Smaller kernel for testing: simple matmul y = x @ W
// Input: [1, IC, 1, SEQ+OC], Output: [1, OC, 1, SEQ]
#define PROBE_IC 64
#define PROBE_OC 64
#define PROBE_SEQ 32
#define PROBE_SP (PROBE_SEQ + PROBE_OC)

int main(int argc, const char *argv[]) {
    @autoreleasepool {
        printf("=== ANE Chaining Probe ===\n");
        mach_timebase_info(&g_tb);
        ane_init();

        // Load chaining classes
        Class ANEChainReq = NSClassFromString(@"_ANEChainingRequest");
        Class ANESharedEvents = NSClassFromString(@"_ANESharedEvents");
        Class ANESignalEvent = NSClassFromString(@"_ANESharedSignalEvent");
        Class ANEWaitEvent = NSClassFromString(@"_ANESharedWaitEvent");
        Class ANEOutputSet = NSClassFromString(@"_ANEOutputSetEnqueue");
        Class ANEInputReady = NSClassFromString(@"_ANEInputBuffersReady");

        printf("Chaining classes: ChainingReq=%s SharedEvents=%s SignalEvent=%s WaitEvent=%s OutputSet=%s InputReady=%s\n",
               ANEChainReq?"Y":"N", ANESharedEvents?"Y":"N", ANESignalEvent?"Y":"N",
               ANEWaitEvent?"Y":"N", ANEOutputSet?"Y":"N", ANEInputReady?"Y":"N");

        // === Compile two simple matmul kernels ===
        // Kernel A: [1, 64, 1, 96] → [1, 64, 1, 32] (y = x @ W_a)
        // Kernel B: [1, 64, 1, 96] → [1, 64, 1, 32] (y = x @ W_b)
        printf("\nCompiling kernel A (matmul %dx%d)...\n", PROBE_IC, PROBE_OC);
        NSString *milA = gen_dyn_matmul_mil(PROBE_IC, PROBE_OC, PROBE_SEQ);
        Kern *kA = compile_kern_mil_w(milA, @{}, PROBE_IC * PROBE_SP * 2, PROBE_OC * PROBE_SEQ * 2);
        if (!kA) { printf("FAIL: kernel A compile\n"); return 1; }
        printf("  OK\n");

        printf("Compiling kernel B (matmul %dx%d)...\n", PROBE_IC, PROBE_OC);
        NSString *milB = gen_dyn_matmul_mil(PROBE_IC, PROBE_OC, PROBE_SEQ);
        Kern *kB = compile_kern_mil_w(milB, @{}, PROBE_IC * PROBE_SP * 2, PROBE_OC * PROBE_SEQ * 2);
        if (!kB) { printf("FAIL: kernel B compile\n"); return 1; }
        printf("  OK\n");

        // === Sequential test (baseline) ===
        printf("\n--- Sequential test ---\n");

        // Fill kernel A input: identity-like weights, ones for activations
        _Float16 *inA = (_Float16*)IOSurfaceGetBaseAddress(kA->ioIn);
        memset(inA, 0, PROBE_IC * PROBE_SP * 2);
        // Activations at sp[0:SEQ]: set to 1.0
        for (int c = 0; c < PROBE_IC; c++)
            for (int s = 0; s < PROBE_SEQ; s++)
                inA[c * PROBE_SP + s] = (_Float16)1.0f;
        // Weights at sp[SEQ:SEQ+OC]: identity-ish (diagonal = 1.0, only if IC==OC)
        for (int c = 0; c < PROBE_IC && c < PROBE_OC; c++)
            inA[c * PROBE_SP + PROBE_SEQ + c] = (_Float16)1.0f;

        // Fill kernel B input similarly (weights as identity)
        _Float16 *inB = (_Float16*)IOSurfaceGetBaseAddress(kB->ioIn);
        memset(inB, 0, PROBE_IC * PROBE_SP * 2);
        for (int c = 0; c < PROBE_IC && c < PROBE_OC; c++)
            inB[c * PROBE_SP + PROBE_SEQ + c] = (_Float16)2.0f;  // scale by 2

        // Eval A
        ane_eval(kA);
        _Float16 *outA = (_Float16*)IOSurfaceGetBaseAddress(kA->ioOut);
        printf("  A output[0..3]: %.2f %.2f %.2f %.2f (expect ~1.0 for identity matmul)\n",
               (float)outA[0], (float)outA[1], (float)outA[2], (float)outA[3]);

        // Copy A's output activations to B's input (simulating what chaining would skip)
        for (int c = 0; c < PROBE_OC; c++)
            memcpy(inB + c * PROBE_SP, outA + c * PROBE_SEQ, PROBE_SEQ * 2);

        // Eval B
        ane_eval(kB);
        _Float16 *outB = (_Float16*)IOSurfaceGetBaseAddress(kB->ioOut);
        printf("  B output[0..3]: %.2f %.2f %.2f %.2f (expect ~2.0 = 1.0*identity*2)\n",
               (float)outB[0], (float)outB[1], (float)outB[2], (float)outB[3]);

        // Read ANE option key constants (needed for chaining)
        void *ane = dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine", RTLD_NOW);
        CFStringRef *k_fw2fw = (CFStringRef*)dlsym(ane, "kANEFEnableFWToFWSignal");
        CFStringRef *k_shevt = (CFStringRef*)dlsym(ane, "kANEFDisableIOFencesUseSharedEventsKey");
        CFStringRef *k_pool = (CFStringRef*)dlsym(ane, "kANEFMemoryPoolIDKey");
        CFStringRef *k_skip = (CFStringRef*)dlsym(ane, "kANEFSkipPreparePhaseKey");
        CFStringRef *k_latch = (CFStringRef*)dlsym(ane, "kANEFEnableLateLatchKey");

        // === Loopback-compatible kernel: same I/O shape ===
        // For chaining, the loopback output must match the loopback input.
        // Create a kernel where the output region matches the input activation region.
        // Use a kernel that outputs [1, IC, 1, SP] = same as input shape.
        // Simple: y = x (identity via matmul with identity weights)
        printf("\n--- Loopback-compatible kernel (same I/O shape) ---\n");
        // Compile a passthrough: [1, 64, 1, 64] → [1, 64, 1, 64]
        // MIL: reshape input to [1,1,64,64], matmul with identity, reshape back
        {
            int LB_CH = 64, LB_SP = 64;
            NSMutableString *lbm = [NSMutableString string];
            [lbm appendString:MIL_HDR];
            [lbm appendFormat:@"    func main<ios18>(tensor<fp16, [1, %d, 1, %d]> x) {\n", LB_CH, LB_SP];
            // Just pass through: slice, reshape to matmul dims, matmul with identity const, reshape back
            [lbm appendFormat:@"        tensor<int32, [4]> rs1 = const()[name=string(\"rs1\"), val=tensor<int32, [4]>([1,1,%d,%d])];\n", LB_CH, LB_SP];
            [lbm appendFormat:@"        tensor<fp16, [1,1,%d,%d]> x2 = reshape(shape=rs1,x=x)[name=string(\"x2\")];\n", LB_CH, LB_SP];
            // Scale by 2 via matmul with 2*identity
            [lbm appendString:@"        bool bF = const()[name=string(\"bF\"), val=bool(false)];\n"];
            [lbm appendFormat:@"        tensor<fp16, [1,1,%d,%d]> yt = matmul(transpose_x=bF,transpose_y=bF,x=x2,y=x2)[name=string(\"yt\")];\n", LB_CH, LB_SP];
            [lbm appendFormat:@"        tensor<int32, [4]> rs2 = const()[name=string(\"rs2\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n", LB_CH, LB_SP];
            [lbm appendFormat:@"        tensor<fp16, [1,%d,1,%d]> y = reshape(shape=rs2,x=yt)[name=string(\"y\")];\n", LB_CH, LB_SP];
            [lbm appendString:@"    } -> (y);\n}\n"];

            Kern *kLB = compile_kern_mil_w(lbm, @{}, LB_CH*LB_SP*2, LB_CH*LB_SP*2);
            if (!kLB) { printf("  FAIL: loopback kernel compile\n"); }
            else {
                printf("  Loopback kernel compiled (in=%dx%d, out=%dx%d)\n", LB_CH, LB_SP, LB_CH, LB_SP);

                // Fill with simple data
                _Float16 *lbIn = (_Float16*)IOSurfaceGetBaseAddress(kLB->ioIn);
                memset(lbIn, 0, LB_CH*LB_SP*2);
                // Set diagonal to 1.0 (identity matrix for x @ x = x when x is identity)
                for (int i = 0; i < LB_CH && i < LB_SP; i++)
                    lbIn[i*LB_SP + i] = (_Float16)1.0f;

                // Test sequential
                ane_eval(kLB);
                _Float16 *lbOut = (_Float16*)IOSurfaceGetBaseAddress(kLB->ioOut);
                printf("  Sequential: out[0]=%.2f out[1]=%.2f out[64]=%.2f (expect 1,0,0 for identity@identity)\n",
                       (float)lbOut[0], (float)lbOut[1], (float)lbOut[LB_SP]);

                // Try chaining with loopback (output feeds back as input)
                id aneModelLB = (__bridge id)kLB->aneModel;
                id wInLB = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_AIO, @selector(objectWithIOSurface:), kLB->ioIn);
                id wOutLB = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_AIO, @selector(objectWithIOSurface:), kLB->ioOut);

                Class ANEBuf = NSClassFromString(@"_ANEBuffer");
                Class ANEOutSets = NSClassFromString(@"_ANEIOSurfaceOutputSets");
                id bufIn = ((id(*)(Class,SEL,id,id,long long))objc_msgSend)(ANEBuf,
                    @selector(bufferWithIOSurfaceObject:symbolIndex:source:), wInLB, @0, (long long)0);
                id bufOut = ((id(*)(Class,SEL,id,id,long long))objc_msgSend)(ANEBuf,
                    @selector(bufferWithIOSurfaceObject:symbolIndex:source:), wOutLB, @0, (long long)1);

                IOSurfaceRef statsSurf2 = make_surface(256);
                id outSets2 = ((id(*)(Class,SEL,IOSurfaceRef,id))objc_msgSend)(ANEOutSets,
                    @selector(objectWithstatsSurRef:outputBuffer:), statsSurf2, @[bufOut]);

                // Chaining request with loopback: output symbol 0 → input symbol 0
                id chainReq2 = ((id(*)(id,SEL,id,id,id,id,id,id,id,id,id))objc_msgSend)(
                    [ANEChainReq alloc],
                    @selector(initWithInputs:outputs:lbInputSymbolId:lbOutputSymbolId:procedureIndex:signalEvents:transactionHandle:fwEnqueueDelay:memoryPoolId:),
                    @[bufIn], @[outSets2], @[@0], @[@0], @0, @[], @1, @0, @0);

                printf("  Loopback ChainingRequest validate: ");
                @try {
                    BOOL v = ((BOOL(*)(id,SEL))objc_msgSend)(chainReq2, @selector(validate));
                    printf("%s\n", v ? "YES" : "NO");
                } @catch (NSException *ex) { printf("EXCEPTION: %s\n", [[ex reason] UTF8String]); }

                // Try prepare with various options
                NSError *e = nil;
                NSString *opts_names[] = { @"FW2FW+SharedEvents", @"empty", @"LateLatch",
                                           @"SkipPrepare", @"MemoryPool" };
                NSDictionary *opts_dicts[5];
                opts_dicts[0] = @{};
                if (k_fw2fw && k_shevt)
                    opts_dicts[0] = @{(__bridge NSString*)*k_fw2fw: @YES, (__bridge NSString*)*k_shevt: @YES};
                opts_dicts[1] = @{};
                opts_dicts[2] = k_latch ? @{(__bridge NSString*)*k_latch: @YES} : @{};
                opts_dicts[3] = k_skip ? @{(__bridge NSString*)*k_skip: @YES} : @{};
                opts_dicts[4] = k_pool ? @{(__bridge NSString*)*k_pool: @0} : @{};

                for (int i = 0; i < 5; i++) {
                    e = nil;
                    @try {
                        BOOL ok2 = ((BOOL(*)(id,SEL,id,id,id,unsigned int,NSError**))objc_msgSend)(
                            g_ane_client,
                            @selector(prepareChainingWithModel:options:chainingReq:qos:error:),
                            aneModelLB, opts_dicts[i], chainReq2, (unsigned int)21, &e);
                        printf("  prepare [%s]: %s", [opts_names[i] UTF8String], ok2 ? "SUCCESS" : "FAIL");
                        if (e) printf(" — %s", [[[e userInfo][@"NSLocalizedDescription"] description] UTF8String]);
                        printf("\n");
                        if (ok2) break;
                    } @catch (NSException *ex) {
                        printf("  prepare [%s]: EXCEPTION %s\n", [opts_names[i] UTF8String], [[ex reason] UTF8String]);
                    }
                }

                free_kern(kLB);
                CFRelease(statsSurf2);
            }
        }

        // === Chaining API exploration ===
        printf("\n--- Chaining API ---\n");

        // Get _ANEModel handles
        id aneModelA = (__bridge id)kA->aneModel;
        id aneModelB = (__bridge id)kB->aneModel;

        // Query symbol indices
        @try {
            id inputSymsA = ((id(*)(id,SEL,id))objc_msgSend)(aneModelA,
                @selector(inputSymbolIndicesForProcedureIndex:), @0);
            id outputSymsA = ((id(*)(id,SEL,id))objc_msgSend)(aneModelA,
                @selector(outputSymbolIndicesForProcedureIndex:), @0);
            printf("  Model A: input symbols=%s output symbols=%s\n",
                   inputSymsA ? [[inputSymsA description] UTF8String] : "nil",
                   outputSymsA ? [[outputSymsA description] UTF8String] : "nil");

            id qd = [aneModelA valueForKey:@"queueDepth"];
            id ph = [aneModelA valueForKey:@"programHandle"];
            id ibh = [aneModelA valueForKey:@"intermediateBufferHandle"];
            printf("  Model A: queueDepth=%s programHandle=%s intermediateBuffer=%s\n",
                   qd ? [[qd description] UTF8String] : "nil",
                   ph ? [[ph description] UTF8String] : "nil",
                   ibh ? [[ibh description] UTF8String] : "nil");
        } @catch (NSException *ex) {
            printf("  Symbol query exception: %s\n", [[ex reason] UTF8String]);
        }

        printf("\n  Option keys found:\n");
        if (k_fw2fw) printf("    kANEFEnableFWToFWSignal = \"%s\"\n", [(__bridge NSString*)*k_fw2fw UTF8String]);
        if (k_shevt) printf("    kANEFDisableIOFencesUseSharedEventsKey = \"%s\"\n", [(__bridge NSString*)*k_shevt UTF8String]);
        if (k_pool) printf("    kANEFMemoryPoolIDKey = \"%s\"\n", [(__bridge NSString*)*k_pool UTF8String]);
        if (k_skip) printf("    kANEFSkipPreparePhaseKey = \"%s\"\n", [(__bridge NSString*)*k_skip UTF8String]);
        if (k_latch) printf("    kANEFEnableLateLatchKey = \"%s\"\n", [(__bridge NSString*)*k_latch UTF8String]);

        // === _ANEBuffer: the proper wrapper with symbolIndex ===
        Class ANEBuffer = NSClassFromString(@"_ANEBuffer");
        printf("\n  _ANEBuffer found: %s\n", ANEBuffer ? "YES" : "NO");

        // Create IOSurface objects first
        id wInA = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_AIO, @selector(objectWithIOSurface:), kA->ioIn);
        id wOutA = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_AIO, @selector(objectWithIOSurface:), kA->ioOut);

        // Wrap in _ANEBuffer with symbolIndex
        // source: 0 = input, 1 = output (guessed)
        id bufInA = ((id(*)(Class,SEL,id,id,long long))objc_msgSend)(ANEBuffer,
            @selector(bufferWithIOSurfaceObject:symbolIndex:source:),
            wInA, @0, (long long)0);
        id bufOutA = ((id(*)(Class,SEL,id,id,long long))objc_msgSend)(ANEBuffer,
            @selector(bufferWithIOSurfaceObject:symbolIndex:source:),
            wOutA, @0, (long long)1);
        printf("  bufInA: %s\n", bufInA ? [[bufInA description] UTF8String] : "nil");
        printf("  bufOutA: %s\n", bufOutA ? [[bufOutA description] UTF8String] : "nil");

        // === Chaining request with _ANEBuffer objects ===
        printf("\n--- ChainingRequest with _ANEBuffer + _ANEIOSurfaceOutputSets ---\n");
        Class ANEOutputSets = NSClassFromString(@"_ANEIOSurfaceOutputSets");
        @try {
            // Wrap output in _ANEIOSurfaceOutputSets
            // statsSurRef: small IOSurface for perf stats (can't be NULL)
            IOSurfaceRef statsSurf = make_surface(256);
            id outSets = ((id(*)(Class,SEL,IOSurfaceRef,id))objc_msgSend)(ANEOutputSets,
                @selector(objectWithstatsSurRef:outputBuffer:),
                statsSurf, @[bufOutA]);
            printf("  outSets: %s\n", outSets ? [[outSets description] UTF8String] : "nil");

            id chainReq = ((id(*)(id,SEL,id,id,id,id,id,id,id,id,id))objc_msgSend)(
                [ANEChainReq alloc],
                @selector(initWithInputs:outputs:lbInputSymbolId:lbOutputSymbolId:procedureIndex:signalEvents:transactionHandle:fwEnqueueDelay:memoryPoolId:),
                @[bufInA],      // inputs = ANEBuffer with symbolIndex
                @[outSets],     // outputs = ANEIOSurfaceOutputSets
                @[@0],          // lbInputSymbolId
                @[@0],          // lbOutputSymbolId
                @0,             // procedureIndex
                @[],            // signalEvents
                @1,             // transactionHandle
                @0,             // fwEnqueueDelay
                @0              // memoryPoolId
            );
            printf("  ChainingRequest: %s\n", chainReq ? "created" : "nil");

            if (chainReq) {
                // Validate
                @try {
                    BOOL valid = ((BOOL(*)(id,SEL))objc_msgSend)(chainReq, @selector(validate));
                    printf("  validate: %s\n", valid ? "YES" : "NO");
                } @catch (NSException *ex) {
                    printf("  validate: EXCEPTION %s\n", [[ex reason] UTF8String]);
                }

                // prepareChainingWithModel — try multiple option combinations
                NSMutableDictionary *opts = [NSMutableDictionary dictionary];
                if (k_fw2fw) opts[(__bridge NSString*)*k_fw2fw] = @YES;
                if (k_shevt) opts[(__bridge NSString*)*k_shevt] = @YES;

                printf("\n  prepareChainingWithModel (FW2FW + SharedEvents)...\n");
                NSError *e = nil;
                BOOL ok = ((BOOL(*)(id,SEL,id,id,id,unsigned int,NSError**))objc_msgSend)(
                    g_ane_client,
                    @selector(prepareChainingWithModel:options:chainingReq:qos:error:),
                    aneModelA, opts, chainReq, (unsigned int)21, &e);
                printf("  result: %s\n", ok ? "SUCCESS" : "FAIL");
                if (e) printf("  error: %s\n", [[e description] UTF8String]);

                if (!ok) {
                    printf("  Retrying with empty options...\n");
                    e = nil;
                    ok = ((BOOL(*)(id,SEL,id,id,id,unsigned int,NSError**))objc_msgSend)(
                        g_ane_client,
                        @selector(prepareChainingWithModel:options:chainingReq:qos:error:),
                        aneModelA, @{}, chainReq, (unsigned int)21, &e);
                    printf("  result: %s\n", ok ? "SUCCESS" : "FAIL");
                    if (e) printf("  error: %s\n", [[e description] UTF8String]);
                }
            }
        } @catch (NSException *ex) {
            printf("  EXCEPTION: %s — %s\n", [[ex name] UTF8String], [[ex reason] UTF8String]);
        }

        // === Also try buffersReady/enqueueSets with _ANEBuffer ===
        printf("\n--- buffersReady/enqueueSets with _ANEBuffer ---\n");
        @try {
            NSError *e = nil;
            BOOL ok = ((BOOL(*)(id,SEL,id,id,id,unsigned int,NSError**))objc_msgSend)(
                g_ane_client,
                @selector(buffersReadyWithModel:inputBuffers:options:qos:error:),
                aneModelA, bufInA, @{}, (unsigned int)21, &e);
            printf("  buffersReady: %s\n", ok ? "OK" : "FAIL");
            if (e) printf("  error: %s\n", [[e description] UTF8String]);

            if (ok) {
                id outSet = ((id(*)(id,SEL,unsigned int,unsigned int,unsigned long long,BOOL,BOOL))objc_msgSend)(
                    [ANEOutputSet alloc],
                    @selector(initOutputSetWithProcedureIndex:setIndex:signalValue:signalNotRequired:isOpenLoop:),
                    (unsigned int)0, (unsigned int)0, (unsigned long long)1, NO, NO);

                e = nil;
                ok = ((BOOL(*)(id,SEL,id,id,id,unsigned int,NSError**))objc_msgSend)(
                    g_ane_client,
                    @selector(enqueueSetsWithModel:outputSet:options:qos:error:),
                    aneModelA, outSet, @{}, (unsigned int)21, &e);
                printf("  enqueueSets: %s\n", ok ? "OK" : "FAIL");
                if (e) printf("  error: %s\n", [[e description] UTF8String]);

                if (ok) {
                    _Float16 *out = (_Float16*)IOSurfaceGetBaseAddress(kA->ioOut);
                    printf("  output[0..3]: %.2f %.2f %.2f %.2f\n",
                           (float)out[0], (float)out[1], (float)out[2], (float)out[3]);
                }
            }
        } @catch (NSException *ex) {
            printf("  EXCEPTION: %s — %s\n", [[ex name] UTF8String], [[ex reason] UTF8String]);
        }

        // Cleanup
        free_kern(kA);
        free_kern(kB);

        printf("\n=== Probe complete ===\n");
    }
    return 0;
}
