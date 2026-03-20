// probe_async_and_device.m — Probe undiscovered ANE APIs
// Uses the working ane_runtime.h compile path with dynamic matmul MIL
// SAFE: read-only probes, no system modifications
#import <Foundation/Foundation.h>
#import <IOSurface/IOSurface.h>
#import <objc/runtime.h>
#import <objc/message.h>
#import <dlfcn.h>
#import <mach/mach_time.h>

static mach_timebase_info_data_t g_tb;
static double ms_t(uint64_t t) { return (double)t * g_tb.numer / g_tb.denom / 1e6; }

static Class g_ANEDesc, g_ANEInMem, g_ANEReq, g_ANEIO;

static void ane_init(void) {
    dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine", RTLD_NOW);
    g_ANEDesc  = NSClassFromString(@"_ANEInMemoryModelDescriptor");
    g_ANEInMem = NSClassFromString(@"_ANEInMemoryModel");
    g_ANEReq   = NSClassFromString(@"_ANERequest");
    g_ANEIO    = NSClassFromString(@"_ANEIOSurfaceObject");
}

// Generate a simple dynamic matmul MIL: y = x_act @ x_weight
// Input: [1, IC, 1, SEQ+OC] fp16 (activations + weight packed in spatial dim)
// Output: [1, OC, 1, SEQ] fp16
static NSData *gen_matmul_mil(int ic, int oc, int seq) {
    NSMutableString *m = [NSMutableString string];
    [m appendString:@"program(1.3)\n"
        "[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}, "
        "{\"coremlc-version\", \"3505.4.1\"}, {\"coremltools-component-milinternal\", \"\"}, "
        "{\"coremltools-version\", \"9.0\"}})]\n{\n"];
    int sp = seq + oc;
    [m appendFormat:@"    func main<ios18>(tensor<fp16, [1, %d, 1, %d]> x) {\n", ic, sp];

    // Slice activations [1,IC,1,SEQ]
    [m appendString:@"        tensor<int32, [4]> ba = const()[name=string(\"ba\"), val=tensor<int32, [4]>([0,0,0,0])];\n"];
    [m appendFormat:@"        tensor<int32, [4]> sa = const()[name=string(\"sa\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n", ic, seq];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> act = slice_by_size(x=x,begin=ba,size=sa)[name=string(\"act\")];\n", ic, seq];

    // Slice weight [1,IC,1,OC]
    [m appendFormat:@"        tensor<int32, [4]> bw = const()[name=string(\"bw\"), val=tensor<int32, [4]>([0,0,0,%d])];\n", seq];
    [m appendFormat:@"        tensor<int32, [4]> sw = const()[name=string(\"sw\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n", ic, oc];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> wt = slice_by_size(x=x,begin=bw,size=sw)[name=string(\"wt\")];\n", ic, oc];

    // Reshape + transpose for matmul
    [m appendFormat:@"        tensor<int32, [4]> ra = const()[name=string(\"ra\"), val=tensor<int32, [4]>([1,1,%d,%d])];\n", ic, seq];
    [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> a2 = reshape(shape=ra,x=act)[name=string(\"a2\")];\n", ic, seq];
    [m appendString:@"        tensor<int32, [4]> pm = const()[name=string(\"pm\"), val=tensor<int32, [4]>([0,1,3,2])];\n"];
    [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> a3 = transpose(perm=pm,x=a2)[name=string(\"a3\")];\n", seq, ic];
    [m appendFormat:@"        tensor<int32, [4]> rw = const()[name=string(\"rw\"), val=tensor<int32, [4]>([1,1,%d,%d])];\n", ic, oc];
    [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> W = reshape(shape=rw,x=wt)[name=string(\"W\")];\n", ic, oc];

    // Matmul
    [m appendString:@"        bool bF = const()[name=string(\"bF\"), val=bool(false)];\n"];
    [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> yh = matmul(transpose_x=bF,transpose_y=bF,x=a3,y=W)[name=string(\"yh\")];\n", seq, oc];

    // Transpose + reshape back to [1,OC,1,SEQ]
    [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> yt = transpose(perm=pm,x=yh)[name=string(\"yt\")];\n", oc, seq];
    [m appendFormat:@"        tensor<int32, [4]> ro = const()[name=string(\"ro\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n", oc, seq];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> y = reshape(shape=ro,x=yt)[name=string(\"y\")];\n", oc, seq];

    [m appendString:@"    } -> (y);\n}\n"];
    return [m dataUsingEncoding:NSUTF8StringEncoding];
}

static IOSurfaceRef make_surf(size_t bytes) {
    return IOSurfaceCreate((__bridge CFDictionaryRef)@{
        (id)kIOSurfaceWidth: @(bytes),
        (id)kIOSurfaceHeight: @1,
        (id)kIOSurfaceBytesPerElement: @1,
        (id)kIOSurfaceBytesPerRow: @(bytes),
        (id)kIOSurfaceAllocSize: @(bytes),
        (id)kIOSurfacePixelFormat: @0
    });
}

int main(int argc, char *argv[]) {
    @autoreleasepool {
        mach_timebase_info(&g_tb);
        ane_init();

        Class ANEDevInfo = NSClassFromString(@"_ANEDeviceInfo");
        Class ANEPerfStats = NSClassFromString(@"_ANEPerformanceStats");
        Class ANEClient = NSClassFromString(@"_ANEClient");

        // ============================
        // 1. Device Info
        // ============================
        printf("=== _ANEDeviceInfo ===\n");
        if (ANEDevInfo) {
            BOOL hasANE = ((BOOL(*)(Class,SEL))objc_msgSend)(ANEDevInfo, @selector(hasANE));
            printf("  hasANE: %s\n", hasANE ? "YES" : "NO");
            unsigned int numCores = ((unsigned int(*)(Class,SEL))objc_msgSend)(ANEDevInfo, @selector(numANECores));
            printf("  numANECores: %u\n", numCores);
            unsigned int numANEs = ((unsigned int(*)(Class,SEL))objc_msgSend)(ANEDevInfo, @selector(numANEs));
            printf("  numANEs: %u\n", numANEs);
            id archType = ((id(*)(Class,SEL))objc_msgSend)(ANEDevInfo, @selector(aneArchitectureType));
            printf("  aneArchitectureType: %s\n", archType ? [[archType description] UTF8String] : "nil");
            long long boardType = ((long long(*)(Class,SEL))objc_msgSend)(ANEDevInfo, @selector(aneBoardType));
            printf("  aneBoardType: %lld\n", boardType);
            id subType = ((id(*)(Class,SEL))objc_msgSend)(ANEDevInfo, @selector(aneSubType));
            printf("  aneSubType: %s\n", subType ? [[subType description] UTF8String] : "nil");
            id subTypeVariant = ((id(*)(Class,SEL))objc_msgSend)(ANEDevInfo, @selector(aneSubTypeVariant));
            printf("  aneSubTypeVariant: %s\n", subTypeVariant ? [[subTypeVariant description] UTF8String] : "nil");
            id productName = ((id(*)(Class,SEL))objc_msgSend)(ANEDevInfo, @selector(productName));
            printf("  productName: %s\n", productName ? [[productName description] UTF8String] : "nil");
        }

        // ============================
        // 2. Compile a test kernel using working MIL format
        // ============================
        printf("\n=== Compile Test Kernel (256x256 matmul, seq=64) ===\n");
        int IC=256, OC=256, SEQ=64;
        size_t in_bytes = IC * (SEQ+OC) * 2;  // fp16
        size_t out_bytes = OC * SEQ * 2;

        NSData *mil = gen_matmul_mil(IC, OC, SEQ);

        id desc = ((id(*)(Class,SEL,id,id,id))objc_msgSend)(
            g_ANEDesc, @selector(modelWithMILText:weights:optionsPlist:), mil, @{}, nil);
        printf("  descriptor: %s\n", desc ? "OK" : "nil");

        id mdl = ((id(*)(Class,SEL,id))objc_msgSend)(
            g_ANEInMem, @selector(inMemoryModelWithDescriptor:), desc);
        printf("  model: %s\n", mdl ? "OK" : "nil");

        id hx = ((id(*)(id,SEL))objc_msgSend)(mdl, @selector(hexStringIdentifier));
        NSString *td = [NSTemporaryDirectory() stringByAppendingPathComponent:hx];
        NSFileManager *fm = [NSFileManager defaultManager];
        [fm createDirectoryAtPath:[td stringByAppendingPathComponent:@"weights"]
            withIntermediateDirectories:YES attributes:nil error:nil];
        [mil writeToFile:[td stringByAppendingPathComponent:@"model.mil"] atomically:YES];

        NSError *e = nil;
        BOOL ok;
        ok = ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(
            mdl, @selector(compileWithQoS:options:error:), 21, @{}, &e);
        printf("  compile: %s%s\n", ok ? "OK" : "FAIL", e ? [[NSString stringWithFormat:@" (%@)", e] UTF8String] : "");
        if (!ok) { printf("ABORT: compile failed\n"); return 1; }
        e = nil;

        ok = ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(
            mdl, @selector(loadWithQoS:options:error:), 21, @{}, &e);
        printf("  load: %s%s\n", ok ? "OK" : "FAIL", e ? [[NSString stringWithFormat:@" (%@)", e] UTF8String] : "");
        if (!ok) { printf("ABORT: load failed\n"); return 1; }

        // ============================
        // 3. queueDepth
        // ============================
        printf("\n=== queueDepth ===\n");
        char qd = ((char(*)(id,SEL))objc_msgSend)(mdl, @selector(queueDepth));
        printf("  default queueDepth: %d\n", (int)qd);
        ((void(*)(id,SEL,char))objc_msgSend)(mdl, @selector(setQueueDepth:), 4);
        qd = ((char(*)(id,SEL))objc_msgSend)(mdl, @selector(queueDepth));
        printf("  after setQueueDepth(4): %d\n", (int)qd);
        ((void(*)(id,SEL,char))objc_msgSend)(mdl, @selector(setQueueDepth:), 1);

        // ============================
        // 4. Create IOSurfaces and request, test eval
        // ============================
        IOSurfaceRef ioIn = make_surf(in_bytes);
        IOSurfaceRef ioOut = make_surf(out_bytes);

        // Fill input with some data
        IOSurfaceLock(ioIn, 0, NULL);
        _Float16 *inp = (void*)IOSurfaceGetBaseAddress(ioIn);
        for (size_t i = 0; i < in_bytes/2; i++) inp[i] = (_Float16)(0.01f * (i % 100));
        IOSurfaceUnlock(ioIn, 0, NULL);

        NSArray *ins = @[((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_ANEIO, @selector(objectWithIOSurface:), ioIn)];
        NSArray *outs = @[((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_ANEIO, @selector(objectWithIOSurface:), ioOut)];

        id req = ((id(*)(Class,SEL,id,id,id,id,id,id,id))objc_msgSend)(
            g_ANEReq, @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:),
            ins, @[@0], outs, @[@0], nil, nil, @0);

        printf("\n=== Basic Eval ===\n");
        e = nil;
        ok = ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
            mdl, @selector(evaluateWithQoS:options:request:error:), 21, @{}, req, &e);
        printf("  result: %s%s\n", ok ? "OK" : "FAIL", e ? [[NSString stringWithFormat:@" (%@)", e] UTF8String] : "");

        if (!ok) {
            printf("ABORT: basic eval failed, cannot continue probes\n");
            goto cleanup;
        }

        // ============================
        // 5. perfStats — skip for now, needs statType protocol investigation
        // ============================
        printf("\n=== Performance Stats ===\n");
        printf("  _ANEPerformanceStats needs statType protocol — future work\n");

        // ============================
        // 6. completionHandler on request
        // ============================
        {
            printf("\n=== completionHandler on _ANERequest ===\n");
            id completionReq = ((id(*)(Class,SEL,id,id,id,id,id,id,id))objc_msgSend)(
                g_ANEReq, @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:),
                ins, @[@0], outs, @[@0], nil, nil, @0);

            __block BOOL handlerCalled = NO;
            __block uint64_t handlerTime = 0;
            void (^handler)(void) = ^{
                handlerCalled = YES;
                handlerTime = mach_absolute_time();
            };

            @try {
                ((void(*)(id,SEL,id))objc_msgSend)(completionReq, @selector(setCompletionHandler:), handler);
                printf("  setCompletionHandler: set OK\n");

                uint64_t t0 = mach_absolute_time();
                e = nil;
                ok = ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
                    mdl, @selector(evaluateWithQoS:options:request:error:), 21, @{}, completionReq, &e);
                uint64_t t1 = mach_absolute_time();
                printf("  eval: %s (%.3f ms)\n", ok ? "OK" : "FAIL", ms_t(t1-t0));

                // Give time for async callback
                [[NSRunLoop currentRunLoop] runUntilDate:[NSDate dateWithTimeIntervalSinceNow:0.1]];
                [NSThread sleepForTimeInterval:0.1];

                printf("  completionHandler called: %s\n", handlerCalled ? "YES" : "NO");
                if (handlerCalled) {
                    printf("  handler fired %.3f ms after eval start\n", ms_t(handlerTime - t0));
                }
            } @catch (NSException *ex) {
                printf("  EXCEPTION: %s\n", [[ex description] UTF8String]);
            }
        }

        // ============================
        // 7. _ANEClient methods
        // ============================
        {
            printf("\n=== _ANEClient async methods ===\n");
            id client = ((id(*)(Class,SEL))objc_msgSend)(ANEClient, @selector(sharedConnection));
            printf("  sharedConnection: %s\n", client ? "obtained" : "nil");

            if (client) {
                BOOL isVirt = ((BOOL(*)(id,SEL))objc_msgSend)(client, @selector(isVirtualClient));
                printf("  isVirtualClient: %s\n", isVirt ? "YES" : "NO");

                id aneModel = ((id(*)(id,SEL))objc_msgSend)(mdl, @selector(model));
                printf("  underlying _ANEModel: %s\n", aneModel ? "obtained" : "nil");

                if (aneModel) {
                    uint64_t ph = ((uint64_t(*)(id,SEL))objc_msgSend)(aneModel, @selector(programHandle));
                    printf("  programHandle: %llu\n", ph);
                    uint64_t ih = ((uint64_t(*)(id,SEL))objc_msgSend)(aneModel, @selector(intermediateBufferHandle));
                    printf("  intermediateBufferHandle: %llu\n", ih);
                    char mQD = ((char(*)(id,SEL))objc_msgSend)(aneModel, @selector(queueDepth));
                    printf("  _ANEModel.queueDepth: %d\n", (int)mQD);

                    // Try evaluateRealTime
                    printf("\n  --- evaluateRealTime ---\n");
                    @try {
                        e = nil;
                        ok = ((BOOL(*)(id,SEL,id,id,id,NSError**))objc_msgSend)(
                            client, @selector(evaluateRealTimeWithModel:options:request:error:),
                            aneModel, @{}, req, &e);
                        printf("  result: %s\n", ok ? "OK" : "FAIL");
                        if (!ok && e) printf("  error: %s\n", [[e description] UTF8String]);
                    } @catch (NSException *ex) {
                        printf("  EXCEPTION: %s\n", [[ex description] UTF8String]);
                    }

                    // Try chaining — known to crash (XPC serialization issue)
                    printf("\n  --- prepareChainingWithModel ---\n");
                    printf("  SKIPPED: _ANERequest doesn't conform to NSSecureCoding for XPC\n");
                    printf("  Needs a different request type (possibly _ANEChainingRequest)\n");

                    // Try buffersReady
                    printf("\n  --- buffersReady ---\n");
                    @try {
                        e = nil;
                        ok = ((BOOL(*)(id,SEL,id,id,id,unsigned int,NSError**))objc_msgSend)(
                            client, @selector(buffersReadyWithModel:inputBuffers:options:qos:error:),
                            aneModel, ins, @{}, 21, &e);
                        printf("  result: %s\n", ok ? "OK" : "FAIL");
                        if (!ok && e) printf("  error: %s\n", [[e description] UTF8String]);
                    } @catch (NSException *ex) {
                        printf("  EXCEPTION: %s\n", [[ex description] UTF8String]);
                    }

                    // Try enqueueSets
                    printf("\n  --- enqueueSets ---\n");
                    @try {
                        e = nil;
                        ok = ((BOOL(*)(id,SEL,id,id,id,unsigned int,NSError**))objc_msgSend)(
                            client, @selector(enqueueSetsWithModel:outputSet:options:qos:error:),
                            aneModel, outs, @{}, 21, &e);
                        printf("  result: %s\n", ok ? "OK" : "FAIL");
                        if (!ok && e) printf("  error: %s\n", [[e description] UTF8String]);
                    } @catch (NSException *ex) {
                        printf("  EXCEPTION: %s\n", [[ex description] UTF8String]);
                    }

                    // Try doEvaluateWithModel with completionEvent
                    printf("\n  --- doEvaluateWithModel (completionEvent=nil) ---\n");
                    @try {
                        e = nil;
                        ok = ((BOOL(*)(id,SEL,id,id,id,unsigned int,id,NSError**))objc_msgSend)(
                            client, @selector(doEvaluateWithModel:options:request:qos:completionEvent:error:),
                            aneModel, @{}, req, 21, nil, &e);
                        printf("  result: %s\n", ok ? "OK" : "FAIL");
                        if (!ok && e) printf("  error: %s\n", [[e description] UTF8String]);
                    } @catch (NSException *ex) {
                        printf("  EXCEPTION: %s\n", [[ex description] UTF8String]);
                    }
                }
            }
        }

        // ============================
        // 8. Eval latency benchmark
        // ============================
        {
            printf("\n=== Eval Latency (1000 iterations) ===\n");
            // Warmup
            for (int i = 0; i < 10; i++) {
                ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
                    mdl, @selector(evaluateWithQoS:options:request:error:), 21, @{}, req, &e);
            }

            uint64_t t0 = mach_absolute_time();
            for (int i = 0; i < 1000; i++) {
                ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
                    mdl, @selector(evaluateWithQoS:options:request:error:), 21, @{}, req, &e);
            }
            uint64_t t1 = mach_absolute_time();
            double total_ms = ms_t(t1-t0);
            printf("  QoS=21: %.1f ms total, %.3f ms/eval (%.0f evals/sec)\n",
                   total_ms, total_ms/1000, 1000000.0/total_ms);

            // QoS 0
            t0 = mach_absolute_time();
            for (int i = 0; i < 1000; i++) {
                ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
                    mdl, @selector(evaluateWithQoS:options:request:error:), 0, @{}, req, &e);
            }
            t1 = mach_absolute_time();
            total_ms = ms_t(t1-t0);
            printf("  QoS=0:  %.1f ms total, %.3f ms/eval (%.0f evals/sec)\n",
                   total_ms, total_ms/1000, 1000000.0/total_ms);

            // QoS 33
            t0 = mach_absolute_time();
            for (int i = 0; i < 1000; i++) {
                ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
                    mdl, @selector(evaluateWithQoS:options:request:error:), 33, @{}, req, &e);
            }
            t1 = mach_absolute_time();
            total_ms = ms_t(t1-t0);
            printf("  QoS=33: %.1f ms total, %.3f ms/eval (%.0f evals/sec)\n",
                   total_ms, total_ms/1000, 1000000.0/total_ms);
        }

cleanup:
        // Safe cleanup - handle potential nil model
        @try {
            e = nil;
            ((BOOL(*)(id,SEL,unsigned int,NSError**))objc_msgSend)(
                mdl, @selector(unloadWithQoS:error:), 21, &e);
        } @catch (NSException *ex) {
            printf("  unload exception (non-fatal): %s\n", [[ex description] UTF8String]);
        }
        if (ioIn) CFRelease(ioIn);
        if (ioOut) CFRelease(ioOut);
        [fm removeItemAtPath:td error:nil];

        printf("\n=== Done ===\n");
    }
    return 0;
}
