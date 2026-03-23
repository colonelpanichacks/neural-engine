// probe_perfstats2.m — Get ANE hardware execution time via _ANEPerformanceStats
// Uses factory method statsWithHardwareExecutionNS: and perfStatsMask
// Also tests doEvaluateDirectWithModel for potential XPC bypass speedup
#import <Foundation/Foundation.h>
#import <IOSurface/IOSurface.h>
#import <objc/runtime.h>
#import <objc/message.h>
#import <dlfcn.h>
#import <mach/mach_time.h>

static mach_timebase_info_data_t g_tb;
static double ms_t(uint64_t t) { return (double)t * g_tb.numer / g_tb.denom / 1e6; }
static Class g_D, g_I, g_AR, g_AIO;
static id g_client;

static void ane_init(void) {
    dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine", RTLD_NOW);
    g_D  = NSClassFromString(@"_ANEInMemoryModelDescriptor");
    g_I  = NSClassFromString(@"_ANEInMemoryModel");
    g_AR = NSClassFromString(@"_ANERequest");
    g_AIO= NSClassFromString(@"_ANEIOSurfaceObject");
    g_client = ((id(*)(Class,SEL))objc_msgSend)(NSClassFromString(@"_ANEClient"), @selector(sharedConnection));
}

static IOSurfaceRef make_surf(size_t b) {
    return IOSurfaceCreate((__bridge CFDictionaryRef)@{
        (id)kIOSurfaceWidth:@(b),(id)kIOSurfaceHeight:@1,
        (id)kIOSurfaceBytesPerElement:@1,(id)kIOSurfaceBytesPerRow:@(b),
        (id)kIOSurfaceAllocSize:@(b),(id)kIOSurfacePixelFormat:@0});
}

// Realistic matmul: 1024→2048 @ seq=256 (same as Qwen3 sdpa forward scale)
static NSData *gen_matmul(int IC, int OC, int SEQ) {
    int SP = SEQ + OC;
    NSMutableString *m = [NSMutableString string];
    [m appendString:@"program(1.3)\n[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"},{\"coremlc-version\", \"3505.4.1\"},{\"coremltools-component-milinternal\", \"\"},{\"coremltools-version\", \"9.0\"}})]\n{\n"];
    [m appendFormat:@"    func main<ios18>(tensor<fp16, [1, %d, 1, %d]> x) {\n", IC, SP];
    [m appendString:@"        tensor<int32, [4]> ba = const()[name=string(\"ba\"), val=tensor<int32, [4]>([0,0,0,0])];\n"];
    [m appendFormat:@"        tensor<int32, [4]> sa = const()[name=string(\"sa\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n", IC, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> act = slice_by_size(x=x,begin=ba,size=sa)[name=string(\"act\")];\n", IC, SEQ];
    [m appendFormat:@"        tensor<int32, [4]> bw = const()[name=string(\"bw\"), val=tensor<int32, [4]>([0,0,0,%d])];\n", SEQ];
    [m appendFormat:@"        tensor<int32, [4]> sw = const()[name=string(\"sw\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n", IC, OC];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> wt = slice_by_size(x=x,begin=bw,size=sw)[name=string(\"wt\")];\n", IC, OC];
    [m appendFormat:@"        tensor<int32, [4]> ra = const()[name=string(\"ra\"), val=tensor<int32, [4]>([1,1,%d,%d])];\n", IC, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> a2 = reshape(shape=ra,x=act)[name=string(\"a2\")];\n", IC, SEQ];
    [m appendString:@"        tensor<int32, [4]> pm = const()[name=string(\"pm\"), val=tensor<int32, [4]>([0,1,3,2])];\n"];
    [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> a3 = transpose(perm=pm,x=a2)[name=string(\"a3\")];\n", SEQ, IC];
    [m appendFormat:@"        tensor<int32, [4]> rw = const()[name=string(\"rw\"), val=tensor<int32, [4]>([1,1,%d,%d])];\n", IC, OC];
    [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> W = reshape(shape=rw,x=wt)[name=string(\"W\")];\n", IC, OC];
    [m appendString:@"        bool bF = const()[name=string(\"bF\"), val=bool(false)];\n"];
    [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> yh = matmul(transpose_x=bF,transpose_y=bF,x=a3,y=W)[name=string(\"yh\")];\n", SEQ, OC];
    [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> yt = transpose(perm=pm,x=yh)[name=string(\"yt\")];\n", OC, SEQ];
    [m appendFormat:@"        tensor<int32, [4]> ro = const()[name=string(\"ro\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n", OC, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> y = reshape(shape=ro,x=yt)[name=string(\"y\")];\n", OC, SEQ];
    [m appendString:@"    } -> (y);\n}\n"];
    return [m dataUsingEncoding:NSUTF8StringEncoding];
}

int main(int argc, char *argv[]) {
    @autoreleasepool {
        setbuf(stdout, NULL);
        mach_timebase_info(&g_tb);
        ane_init();

        Class perfClass = NSClassFromString(@"_ANEPerformanceStats");

        int IC=1024, OC=2048, SEQ=256;
        printf("=== ANE PerfStats + Direct Eval Probe ===\n");
        printf("Kernel: %dx%d matmul, seq=%d\n\n", IC, OC, SEQ);

        NSData *mil = gen_matmul(IC, OC, SEQ);
        id desc = ((id(*)(Class,SEL,id,id,id))objc_msgSend)(g_D,@selector(modelWithMILText:weights:optionsPlist:),mil,@{},nil);
        id mdl = ((id(*)(Class,SEL,id))objc_msgSend)(g_I,@selector(inMemoryModelWithDescriptor:),desc);
        id hx = ((id(*)(id,SEL))objc_msgSend)(mdl,@selector(hexStringIdentifier));
        NSString *td = [NSTemporaryDirectory() stringByAppendingPathComponent:hx];
        [[NSFileManager defaultManager] createDirectoryAtPath:[td stringByAppendingPathComponent:@"weights"] withIntermediateDirectories:YES attributes:nil error:nil];
        [mil writeToFile:[td stringByAppendingPathComponent:@"model.mil"] atomically:YES];

        NSError *e = nil;
        ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(mdl,@selector(compileWithQoS:options:error:),21,@{},&e);
        ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(mdl,@selector(loadWithQoS:options:error:),21,@{},&e);
        printf("Compile+Load OK\n");

        size_t in_bytes = (size_t)IC*(SEQ+OC)*2;
        size_t out_bytes = (size_t)OC*SEQ*2;
        IOSurfaceRef ioIn = make_surf(in_bytes);
        IOSurfaceRef ioOut = make_surf(out_bytes);
        _Float16 *inp = (_Float16*)IOSurfaceGetBaseAddress(ioIn);
        for (size_t i=0; i<in_bytes/2; i++) inp[i]=(_Float16)(0.01f*(i%100));

        id wI=((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_AIO,@selector(objectWithIOSurface:),ioIn);
        id wO=((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_AIO,@selector(objectWithIOSurface:),ioOut);

        id aneModel = ((id(*)(id,SEL))objc_msgSend)(mdl,@selector(model));

        // ===== 1. Try perfStatsMask =====
        printf("\n--- perfStatsMask ---\n");
        unsigned int currentMask = ((unsigned int(*)(id,SEL))objc_msgSend)(aneModel,@selector(perfStatsMask));
        printf("  current mask: 0x%x\n", currentMask);

        // Try setting mask to enable hw execution time
        // ANE stats mask values from _ANEDeviceInfo driverMaskForANEFMask:
        for (unsigned int mask = 1; mask <= 0x10; mask <<= 1) {
            ((void(*)(id,SEL,unsigned int))objc_msgSend)(aneModel,@selector(setPerfStatsMask:),mask);
            unsigned int readback = ((unsigned int(*)(id,SEL))objc_msgSend)(aneModel,@selector(perfStatsMask));
            printf("  set 0x%x, readback: 0x%x\n", mask, readback);
        }
        // Set to 0xF (enable all)
        ((void(*)(id,SEL,unsigned int))objc_msgSend)(aneModel,@selector(setPerfStatsMask:),0xF);

        // ===== 2. Create perfStats via factory =====
        printf("\n--- _ANEPerformanceStats factory methods ---\n");
        if (perfClass) {
            // Try statsWithHardwareExecutionNS:
            @try {
                id ps1 = ((id(*)(Class,SEL,uint64_t))objc_msgSend)(perfClass,@selector(statsWithHardwareExecutionNS:),(uint64_t)0);
                printf("  statsWithHardwareExecutionNS(0): %s\n", ps1 ? "non-nil" : "nil");
                if (ps1) {
                    uint64_t hwt = ((uint64_t(*)(id,SEL))objc_msgSend)(ps1,@selector(hwExecutionTime));
                    printf("    hwExecutionTime: %llu ns\n", hwt);
                }
            } @catch (NSException *ex) {
                printf("  statsWithHardwareExecutionNS: EXCEPTION %s\n", [[ex reason] UTF8String]);
            }

            // Try passing perfStats to request, eval, then read back
            printf("\n--- perfStats in request + eval ---\n");
            @try {
                id ps = ((id(*)(Class,SEL,uint64_t))objc_msgSend)(perfClass,@selector(statsWithHardwareExecutionNS:),(uint64_t)0);
                if (ps) {
                    // perfStats param expects an array (request has perfStatsArray property)
                    id req = ((id(*)(Class,SEL,id,id,id,id,id,id,id))objc_msgSend)(g_AR,
                        @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:),
                        @[wI],@[@0],@[wO],@[@0],nil,@[ps],@0);

                    // Warmup
                    for (int i=0;i<10;i++) {
                        ((BOOL(*)(id,SEL,id,id,id,NSError**))objc_msgSend)(g_client,
                            @selector(evaluateRealTimeWithModel:options:request:error:),aneModel,@{},req,&e);
                    }

                    // Read hwExecutionTime after warmup
                    uint64_t hwt = ((uint64_t(*)(id,SEL))objc_msgSend)(ps,@selector(hwExecutionTime));
                    printf("  hwExecutionTime after 10 warmup evals: %llu ns (%.3f ms)\n", hwt, hwt/1e6);

                    // Now do 100 evals and measure wall time vs hw time
                    uint64_t t0 = mach_absolute_time();
                    for (int i=0;i<100;i++) {
                        ((BOOL(*)(id,SEL,id,id,id,NSError**))objc_msgSend)(g_client,
                            @selector(evaluateRealTimeWithModel:options:request:error:),aneModel,@{},req,&e);
                    }
                    double wall_ms = ms_t(mach_absolute_time()-t0);
                    uint64_t hwt2 = ((uint64_t(*)(id,SEL))objc_msgSend)(ps,@selector(hwExecutionTime));
                    printf("  After 100 more evals:\n");
                    printf("    wall time: %.1f ms (%.3f ms/eval)\n", wall_ms, wall_ms/100);
                    printf("    hwExecutionTime: %llu ns (%.3f ms)\n", hwt2, hwt2/1e6);
                    printf("    hw time per eval: %.3f ms\n", (hwt2-hwt)/100.0/1e6);
                    printf("    dispatch overhead: %.3f ms/eval\n", wall_ms/100 - (hwt2-hwt)/100.0/1e6);

                    // Check perfCounterData
                    id pcd = ((id(*)(id,SEL))objc_msgSend)(ps,@selector(perfCounterData));
                    printf("    perfCounterData: %s (%lu bytes)\n",
                           pcd ? "non-nil" : "nil",
                           pcd ? (unsigned long)[(NSData*)pcd length] : 0);
                    id psr = ((id(*)(id,SEL))objc_msgSend)(ps,@selector(pStatsRawData));
                    printf("    pStatsRawData: %s (%lu bytes)\n",
                           psr ? "non-nil" : "nil",
                           psr ? (unsigned long)[(NSData*)psr length] : 0);

                    // Try performanceCounters
                    @try {
                        id pc = ((id(*)(id,SEL))objc_msgSend)(ps,@selector(performanceCounters));
                        printf("    performanceCounters: %s\n", pc ? [[pc description] UTF8String] : "nil");
                    } @catch (NSException *ex) {
                        printf("    performanceCounters: EXCEPTION\n");
                    }
                }
            } @catch (NSException *ex) {
                printf("  EXCEPTION: %s\n", [[ex reason] UTF8String]);
            }

            // Try statsWithReconstructed:
            printf("\n--- statsWithReconstructed ---\n");
            @try {
                id ps2 = ((id(*)(Class,SEL,id,uint64_t,id))objc_msgSend)(perfClass,
                    @selector(statsWithReconstructed:hardwareExecutionNS:aneStatsRawData:),
                    nil, (uint64_t)0, nil);
                printf("  statsWithReconstructed(nil,0,nil): %s\n", ps2 ? "non-nil" : "nil");
            } @catch (NSException *ex) {
                printf("  EXCEPTION: %s\n", [[ex reason] UTF8String]);
            }

            // driverMaskForANEFMask: crashes (not actually a class method on this arch)
            // Skipped
        }

        // ===== 3. doEvaluateDirectWithModel =====
        printf("\n--- doEvaluateDirectWithModel benchmark ---\n");
        {
            id req = ((id(*)(Class,SEL,id,id,id,id,id,id,id))objc_msgSend)(g_AR,
                @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:),
                @[wI],@[@0],@[wO],@[@0],nil,nil,@0);

            // Warmup with normal path
            for (int i=0;i<20;i++) {
                ((BOOL(*)(id,SEL,id,id,id,NSError**))objc_msgSend)(g_client,
                    @selector(evaluateRealTimeWithModel:options:request:error:),aneModel,@{},req,&e);
            }

            // Benchmark: evaluateRealTimeWithModel (current)
            int N = 500;
            uint64_t t0 = mach_absolute_time();
            for (int i=0;i<N;i++) {
                ((BOOL(*)(id,SEL,id,id,id,NSError**))objc_msgSend)(g_client,
                    @selector(evaluateRealTimeWithModel:options:request:error:),aneModel,@{},req,&e);
            }
            double rt_ms = ms_t(mach_absolute_time()-t0);
            printf("  evaluateRealTime:  %.1f ms total, %.3f ms/eval\n", rt_ms, rt_ms/N);

            // Benchmark: doEvaluateDirectWithModel (XPC bypass?)
            @try {
                // Test single call first
                e = nil;
                BOOL ok = ((BOOL(*)(id,SEL,id,id,id,unsigned int,NSError**))objc_msgSend)(g_client,
                    @selector(doEvaluateDirectWithModel:options:request:qos:error:),
                    aneModel,@{},req,21,&e);
                if (ok) {
                    t0 = mach_absolute_time();
                    for (int i=0;i<N;i++) {
                        ((BOOL(*)(id,SEL,id,id,id,unsigned int,NSError**))objc_msgSend)(g_client,
                            @selector(doEvaluateDirectWithModel:options:request:qos:error:),
                            aneModel,@{},req,21,&e);
                    }
                    double de_ms = ms_t(mach_absolute_time()-t0);
                    printf("  doEvaluateDirect:  %.1f ms total, %.3f ms/eval\n", de_ms, de_ms/N);
                    printf("  Speedup: %.2fx\n", rt_ms/de_ms);
                } else {
                    printf("  doEvaluateDirect: FAIL (%s)\n", e ? [[e description] UTF8String] : "unknown");
                }
            } @catch (NSException *ex) {
                printf("  doEvaluateDirect: EXCEPTION %s\n", [[ex description] UTF8String]);
            }

            // Benchmark: evaluateWithQoS (model method, not client)
            t0 = mach_absolute_time();
            for (int i=0;i<N;i++) {
                ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(mdl,
                    @selector(evaluateWithQoS:options:request:error:),21,@{},req,&e);
            }
            double eq_ms = ms_t(mach_absolute_time()-t0);
            printf("  evaluateWithQoS:   %.1f ms total, %.3f ms/eval\n", eq_ms, eq_ms/N);

            // Benchmark: loadRealTimeModel vs regular load difference
            printf("\n--- loadRealTimeModel ---\n");
            @try {
                e = nil;
                BOOL ok = ((BOOL(*)(id,SEL,unsigned int,NSError**))objc_msgSend)(mdl,
                    @selector(unloadWithQoS:error:),21,&e);
                printf("  unload: %s\n", ok ? "OK" : "FAIL");

                // Reload with real-time path
                e = nil;
                ok = ((BOOL(*)(id,SEL,id,id,unsigned int,NSError**))objc_msgSend)(
                    g_client,@selector(loadRealTimeModel:options:qos:error:),
                    aneModel,@{},21,&e);
                printf("  loadRealTimeModel: %s\n", ok ? "OK" : "FAIL");
                if (!ok && e) printf("  error: %s\n", [[e description] UTF8String]);

                if (ok) {
                    // Re-benchmark after real-time load
                    for (int i=0;i<20;i++) {
                        ((BOOL(*)(id,SEL,id,id,id,NSError**))objc_msgSend)(g_client,
                            @selector(evaluateRealTimeWithModel:options:request:error:),aneModel,@{},req,&e);
                    }
                    t0 = mach_absolute_time();
                    for (int i=0;i<N;i++) {
                        ((BOOL(*)(id,SEL,id,id,id,NSError**))objc_msgSend)(g_client,
                            @selector(evaluateRealTimeWithModel:options:request:error:),aneModel,@{},req,&e);
                    }
                    double rtl_ms = ms_t(mach_absolute_time()-t0);
                    printf("  after loadRealTime: %.3f ms/eval (was %.3f)\n", rtl_ms/N, rt_ms/N);
                }
            } @catch (NSException *ex) {
                printf("  EXCEPTION: %s\n", [[ex description] UTF8String]);
            }
        }

        // ===== 4. queueDepth effect =====
        printf("\n--- queueDepth effect ---\n");
        {
            // Reload fresh
            e = nil;
            ((BOOL(*)(id,SEL,unsigned int,NSError**))objc_msgSend)(mdl,@selector(unloadWithQoS:error:),21,&e);
            ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(mdl,@selector(loadWithQoS:options:error:),21,@{},&e);
            aneModel = ((id(*)(id,SEL))objc_msgSend)(mdl,@selector(model));

            id req = ((id(*)(Class,SEL,id,id,id,id,id,id,id))objc_msgSend)(g_AR,
                @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:),
                @[wI],@[@0],@[wO],@[@0],nil,nil,@0);

            int N = 500;
            int depths[] = {1, 2, 4, 8, 16, 127};
            for (int d = 0; d < 6; d++) {
                ((void(*)(id,SEL,char))objc_msgSend)(aneModel,@selector(setQueueDepth:),(char)depths[d]);
                // Warmup
                for (int i=0;i<20;i++) {
                    ((BOOL(*)(id,SEL,id,id,id,NSError**))objc_msgSend)(g_client,
                        @selector(evaluateRealTimeWithModel:options:request:error:),aneModel,@{},req,&e);
                }
                uint64_t t0 = mach_absolute_time();
                for (int i=0;i<N;i++) {
                    ((BOOL(*)(id,SEL,id,id,id,NSError**))objc_msgSend)(g_client,
                        @selector(evaluateRealTimeWithModel:options:request:error:),aneModel,@{},req,&e);
                }
                double ms = ms_t(mach_absolute_time()-t0);
                printf("  queueDepth=%3d: %.3f ms/eval\n", depths[d], ms/N);
            }
        }

        CFRelease(ioIn); CFRelease(ioOut);
        printf("\n=== Done ===\n");
    }
    return 0;
}
