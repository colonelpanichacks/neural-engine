// probe_chaining4.m — Comprehensive ANE chaining probe
// Combines learnings from probe_chaining{1,2,3} and chain_probe
// Focus: get prepareChainingWithModel to return SUCCESS
#import <Foundation/Foundation.h>
#import <IOSurface/IOSurface.h>
#import <objc/runtime.h>
#import <objc/message.h>
#import <dlfcn.h>
#import <mach/mach_time.h>

static mach_timebase_info_data_t g_tb;
static double ms_t(uint64_t t) { return (double)t * g_tb.numer / g_tb.denom / 1e6; }
static Class g_AIO, g_AR;
static id g_client = nil;

static IOSurfaceRef make_surf(size_t bytes) {
    return IOSurfaceCreate((__bridge CFDictionaryRef)@{
        (id)kIOSurfaceWidth:@(bytes), (id)kIOSurfaceHeight:@1,
        (id)kIOSurfaceBytesPerElement:@1, (id)kIOSurfaceBytesPerRow:@(bytes),
        (id)kIOSurfaceAllocSize:@(bytes), (id)kIOSurfacePixelFormat:@0});
}

// Simple matmul kernel: y = x @ W
static NSData *gen_mil(int ic, int oc, int seq) {
    NSMutableString *m = [NSMutableString string];
    [m appendString:@"program(1.3)\n[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}, "
        "{\"coremlc-version\", \"3505.4.1\"}, {\"coremltools-component-milinternal\", \"\"}, "
        "{\"coremltools-version\", \"9.0\"}})]{\n"];
    int sp = seq + oc;
    [m appendFormat:@"    func main<ios18>(tensor<fp16, [1, %d, 1, %d]> x) {\n", ic, sp];
    [m appendFormat:@"        tensor<int32, [4]> ba = const()[name=string(\"ba\"), val=tensor<int32, [4]>([0,0,0,0])];\n"];
    [m appendFormat:@"        tensor<int32, [4]> sa = const()[name=string(\"sa\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n", ic, seq];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> act = slice_by_size(x=x,begin=ba,size=sa)[name=string(\"act\")];\n", ic, seq];
    [m appendFormat:@"        tensor<int32, [4]> bw = const()[name=string(\"bw\"), val=tensor<int32, [4]>([0,0,0,%d])];\n", seq];
    [m appendFormat:@"        tensor<int32, [4]> sw = const()[name=string(\"sw\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n", ic, oc];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> wt = slice_by_size(x=x,begin=bw,size=sw)[name=string(\"wt\")];\n", ic, oc];
    [m appendFormat:@"        tensor<int32, [4]> ra = const()[name=string(\"ra\"), val=tensor<int32, [4]>([1,1,%d,%d])];\n", ic, seq];
    [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> a2 = reshape(shape=ra,x=act)[name=string(\"a2\")];\n", ic, seq];
    [m appendString:@"        tensor<int32, [4]> pm = const()[name=string(\"pm\"), val=tensor<int32, [4]>([0,1,3,2])];\n"];
    [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> a3 = transpose(perm=pm,x=a2)[name=string(\"a3\")];\n", seq, ic];
    [m appendFormat:@"        tensor<int32, [4]> rw = const()[name=string(\"rw\"), val=tensor<int32, [4]>([1,1,%d,%d])];\n", ic, oc];
    [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> W = reshape(shape=rw,x=wt)[name=string(\"W\")];\n", ic, oc];
    [m appendString:@"        bool bF = const()[name=string(\"bF\"), val=bool(false)];\n"];
    [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> yh = matmul(transpose_x=bF,transpose_y=bF,x=a3,y=W)[name=string(\"yh\")];\n", seq, oc];
    [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> yt = transpose(perm=pm,x=yh)[name=string(\"yt\")];\n", oc, seq];
    [m appendFormat:@"        tensor<int32, [4]> ro = const()[name=string(\"ro\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n", oc, seq];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> y = reshape(shape=ro,x=yt)[name=string(\"y\")];\n", oc, seq];
    [m appendString:@"    } -> (y);\n}\n"];
    return [m dataUsingEncoding:NSUTF8StringEncoding];
}

typedef struct { id model, aneModel; IOSurfaceRef ioIn, ioOut; id request; } Kern;

static Kern compile(int ic, int oc, int seq) {
    Kern k = {0};
    Class D = NSClassFromString(@"_ANEInMemoryModelDescriptor");
    Class I = NSClassFromString(@"_ANEInMemoryModel");
    NSData *mil = gen_mil(ic, oc, seq);
    size_t inB = ic*(seq+oc)*2, outB = oc*seq*2;
    id desc = ((id(*)(Class,SEL,id,id,id))objc_msgSend)(D, @selector(modelWithMILText:weights:optionsPlist:), mil, @{}, nil);
    k.model = ((id(*)(Class,SEL,id))objc_msgSend)(I, @selector(inMemoryModelWithDescriptor:), desc);
    id hx = ((id(*)(id,SEL))objc_msgSend)(k.model, @selector(hexStringIdentifier));
    NSString *td = [NSTemporaryDirectory() stringByAppendingPathComponent:hx];
    [[NSFileManager defaultManager] createDirectoryAtPath:[td stringByAppendingPathComponent:@"weights"]
        withIntermediateDirectories:YES attributes:nil error:nil];
    [mil writeToFile:[td stringByAppendingPathComponent:@"model.mil"] atomically:YES];
    // Set queueDepth BEFORE compile — may affect compilation for chaining support
    ((void(*)(id,SEL,char))objc_msgSend)(k.model, @selector(setQueueDepth:), (char)4);
    NSError *e = nil;
    ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(k.model, @selector(compileWithQoS:options:error:), 21, @{}, &e);
    ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(k.model, @selector(loadWithQoS:options:error:), 21, @{}, &e);
    k.aneModel = ((id(*)(id,SEL))objc_msgSend)(k.model, @selector(model));
    k.ioIn = make_surf(inB); k.ioOut = make_surf(outB);
    _Float16 *p = (void*)IOSurfaceGetBaseAddress(k.ioIn);
    for (size_t i = 0; i < inB/2; i++) p[i] = (_Float16)(0.01f*(i%100));
    id wI = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_AIO, @selector(objectWithIOSurface:), k.ioIn);
    id wO = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_AIO, @selector(objectWithIOSurface:), k.ioOut);
    k.request = ((id(*)(Class,SEL,id,id,id,id,id,id,id))objc_msgSend)(g_AR,
        @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:),
        @[wI], @[@0], @[wO], @[@0], nil, nil, @0);
    return k;
}

int main(int argc, char *argv[]) {
    @autoreleasepool {
        setbuf(stdout, NULL);
        mach_timebase_info(&g_tb);
        dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine", RTLD_NOW);
        g_AIO = NSClassFromString(@"_ANEIOSurfaceObject");
        g_AR  = NSClassFromString(@"_ANERequest");
        g_client = ((id(*)(Class,SEL))objc_msgSend)(NSClassFromString(@"_ANEClient"), @selector(sharedConnection));

        // Load all ANE option keys
        void *ane = dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine", RTLD_NOW);
        CFStringRef *k_fw2fw = (CFStringRef*)dlsym(ane, "kANEFEnableFWToFWSignal");
        CFStringRef *k_shevt = (CFStringRef*)dlsym(ane, "kANEFDisableIOFencesUseSharedEventsKey");
        CFStringRef *k_pool  = (CFStringRef*)dlsym(ane, "kANEFMemoryPoolIDKey");
        CFStringRef *k_skip  = (CFStringRef*)dlsym(ane, "kANEFSkipPreparePhaseKey");
        CFStringRef *k_latch = (CFStringRef*)dlsym(ane, "kANEFEnableLateLatchKey");

        printf("=== ANE Chaining Probe v4 ===\n");
        printf("Option keys: fw2fw=%s shevt=%s pool=%s skip=%s latch=%s\n",
               k_fw2fw?"Y":"N", k_shevt?"Y":"N", k_pool?"Y":"N", k_skip?"Y":"N", k_latch?"Y":"N");
        if (k_fw2fw) printf("  fw2fw = \"%s\"\n", [(__bridge NSString*)*k_fw2fw UTF8String]);
        if (k_shevt) printf("  shevt = \"%s\"\n", [(__bridge NSString*)*k_shevt UTF8String]);
        if (k_pool)  printf("  pool  = \"%s\"\n", [(__bridge NSString*)*k_pool  UTF8String]);
        if (k_skip)  printf("  skip  = \"%s\"\n", [(__bridge NSString*)*k_skip  UTF8String]);
        if (k_latch) printf("  latch = \"%s\"\n", [(__bridge NSString*)*k_latch UTF8String]);

        // Set queueDepth > 0 before compile (might enable chaining support)
    Kern kA = compile(64, 64, 32);
    // Try setting queueDepth on both _ANEInMemoryModel and _ANEModel
    @try {
        ((void(*)(id,SEL,char))objc_msgSend)(kA.model, @selector(setQueueDepth:), (char)2);
        char qd1 = ((char(*)(id,SEL))objc_msgSend)(kA.model, @selector(queueDepth));
        printf("  InMemoryModel queueDepth set to: %d\n", qd1);
    } @catch (NSException *ex) { printf("  setQueueDepth on InMemoryModel: %s\n", [[ex reason] UTF8String]); }
    @try {
        char qd0 = ((char(*)(id,SEL))objc_msgSend)(kA.aneModel, @selector(queueDepth));
        printf("  ANEModel queueDepth (before): %d\n", qd0);
        // queueDepth is on _ANEModel, try setting it there too
        ((void(*)(id,SEL,char))objc_msgSend)(kA.aneModel, @selector(setQueueDepth:), (char)2);
        qd0 = ((char(*)(id,SEL))objc_msgSend)(kA.aneModel, @selector(queueDepth));
        printf("  ANEModel queueDepth (after): %d\n", qd0);
    } @catch (NSException *ex) { printf("  queueDepth on ANEModel: %s\n", [[ex reason] UTF8String]); }
    printf("\nKernel compiled: [64,96]→[64,32]\n");

        Class ANEBuf     = NSClassFromString(@"_ANEBuffer");
        Class ANEOutSets  = NSClassFromString(@"_ANEIOSurfaceOutputSets");
        Class ANEChainReq = NSClassFromString(@"_ANEChainingRequest");
        Class ANEOutSetEnq = NSClassFromString(@"_ANEOutputSetEnqueue");
        Class ANEInputReady = NSClassFromString(@"_ANEInputBuffersReady");

        // Wrap IOSurfaces in _ANEBuffer
        id wIn  = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_AIO, @selector(objectWithIOSurface:), kA.ioIn);
        id wOut = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_AIO, @selector(objectWithIOSurface:), kA.ioOut);
        IOSurfaceRef statsSurf = make_surf(4096);
        id wStats = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_AIO, @selector(objectWithIOSurface:), statsSurf);

        id bufIn  = ((id(*)(Class,SEL,id,id,long long))objc_msgSend)(ANEBuf,
            @selector(bufferWithIOSurfaceObject:symbolIndex:source:), wIn, @0, (long long)0);
        id bufOut = ((id(*)(Class,SEL,id,id,long long))objc_msgSend)(ANEBuf,
            @selector(bufferWithIOSurfaceObject:symbolIndex:source:), wOut, @0, (long long)1);
        printf("bufIn:  %s\n", [[bufIn description] UTF8String]);
        printf("bufOut: %s\n", [[bufOut description] UTF8String]);

        // Wrap output in _ANEIOSurfaceOutputSets (this was the missing step in probe3)
        id outSets = ((id(*)(Class,SEL,IOSurfaceRef,id))objc_msgSend)(ANEOutSets,
            @selector(objectWithstatsSurRef:outputBuffer:), statsSurf, @[bufOut]);
        printf("outSets: %s\n", [[outSets description] UTF8String]);

        // === Test 1: Chaining with proper _ANEIOSurfaceOutputSets ===
        printf("\n=== Test 1: prepareChainingWithModel (proper OutputSets) ===\n");

        // Try different loopback configurations
        int lb_configs[][2] = { {-1,-1}, {0,0}, {-1,0}, {0,-1} };
        const char *lb_names[] = { "no-loopback", "lb(0,0)", "lb(-1,0)", "lb(0,-1)" };

        for (int cfg = 0; cfg < 4; cfg++) {
            @try {
                id chainReq = ((id(*)(id,SEL,id,id,id,id,id,id,id,id,id))objc_msgSend)(
                    [ANEChainReq alloc],
                    @selector(initWithInputs:outputs:lbInputSymbolId:lbOutputSymbolId:procedureIndex:signalEvents:transactionHandle:fwEnqueueDelay:memoryPoolId:),
                    @[bufIn],       // inputs: array of _ANEBuffer
                    @[outSets],     // outputs: array of _ANEIOSurfaceOutputSets
                    @[@(lb_configs[cfg][0])],  // lbInputSymbolId (array)
                    @[@(lb_configs[cfg][1])],  // lbOutputSymbolId (array)
                    @0,             // procedureIndex
                    @[],            // signalEvents
                    @1,             // transactionHandle
                    @0,             // fwEnqueueDelay
                    @0              // memoryPoolId
                );

                // Try with different option combinations
                NSDictionary *opt_sets[] = {
                    @{},
                    k_fw2fw && k_shevt ? @{(__bridge NSString*)*k_fw2fw:@YES, (__bridge NSString*)*k_shevt:@YES} : @{},
                    k_latch ? @{(__bridge NSString*)*k_latch:@YES} : @{},
                    k_skip ? @{(__bridge NSString*)*k_skip:@YES} : @{},
                };
                const char *opt_names[] = { "empty", "fw2fw+shevt", "latch", "skip" };

                for (int o = 0; o < 4; o++) {
                    NSError *e = nil;
                    @try {
                        // Use doPrepareChainingWithModel (direct, bypasses XPC)
                        BOOL ok = ((BOOL(*)(id,SEL,id,id,id,unsigned int,NSError**))objc_msgSend)(
                            g_client,
                            @selector(doPrepareChainingWithModel:options:chainingReq:qos:error:),
                            kA.aneModel, opt_sets[o], chainReq, (unsigned int)21, &e);
                        printf("  [%s][%s]: %s", lb_names[cfg], opt_names[o], ok?"SUCCESS":"FAIL");
                        if (e) printf(" err=%ld", (long)[e code]);
                        printf("\n");
                        if (ok) {
                            printf("  *** CHAINING PREPARE SUCCEEDED! ***\n");
                            goto done_chaining;
                        }
                    } @catch (NSException *ex) {
                        printf("  [%s][%s]: EXCEPTION %s\n", lb_names[cfg], opt_names[o], [[ex reason] UTF8String]);
                    }
                }
            } @catch (NSException *ex) {
                printf("  [%s]: init EXCEPTION %s\n", lb_names[cfg], [[ex reason] UTF8String]);
            }
        }
        done_chaining:

        // === Test 2: buffersReady + enqueueSets (two-phase dispatch) ===
        printf("\n=== Test 2: doBuffersReady + doEnqueueSets (direct, bypass XPC) ===\n");
        @try {
            id ibr = ((id(*)(id,SEL,unsigned int,unsigned int,unsigned long long,unsigned int))objc_msgSend)(
                [ANEInputReady alloc],
                @selector(initInputsProcedureIndex:inputBufferInfoIndex:inputFreeValue:executionDelay:),
                (unsigned int)0, (unsigned int)0, (unsigned long long)0, (unsigned int)0);
            printf("  InputBuffersReady: %s\n", ibr ? "OK" : "nil");

            NSError *e = nil;
            // Use doBuffersReadyWithModel (direct variant)
            printf("  ibr class: %s\n", class_getName([ibr class]));
            printf("  ibr desc: %s\n", [[ibr description] UTF8String]);
            BOOL ok = ((BOOL(*)(id,SEL,id,id,id,unsigned int,NSError**))objc_msgSend)(
                g_client, @selector(doBuffersReadyWithModel:inputBuffers:options:qos:error:),
                kA.aneModel, ibr, @{}, 21, &e);
            printf("  doBuffersReady: %s", ok?"OK":"FAIL");
            if (e) printf(" err=%ld: %s", (long)[e code], [[e description] UTF8String]);
            else printf(" (no error object)");
            printf("\n");

            if (ok) {
                id ose = ((id(*)(id,SEL,unsigned int,unsigned int,unsigned long long,BOOL,BOOL))objc_msgSend)(
                    [ANEOutSetEnq alloc],
                    @selector(initOutputSetWithProcedureIndex:setIndex:signalValue:signalNotRequired:isOpenLoop:),
                    (unsigned int)0, (unsigned int)0, (unsigned long long)1, NO, NO);
                printf("  OutputSetEnqueue: %s\n", ose ? [[ose description] UTF8String] : "nil");
                e = nil;
                // Use doEnqueueSetsWithModel (direct variant), pass ose directly (not array)
                BOOL ok2 = ((BOOL(*)(id,SEL,id,id,id,unsigned int,NSError**))objc_msgSend)(
                    g_client, @selector(doEnqueueSetsWithModel:outputSet:options:qos:error:),
                    kA.aneModel, ose, @{}, 21, &e);
                printf("  doEnqueueSets: %s", ok2?"OK":"FAIL");
                if (e) printf(" err=%ld: %s", (long)[e code], [[e localizedDescription] UTF8String]);
                printf("\n");

                if (ok2) {
                    // Verify output
                    _Float16 *out = (_Float16*)IOSurfaceGetBaseAddress(kA.ioOut);
                    printf("  output[0..3]: %.4f %.4f %.4f %.4f\n",
                           (float)out[0], (float)out[1], (float)out[2], (float)out[3]);

                    // Benchmark two-phase dispatch
                    printf("\n  Benchmarking two-phase dispatch...\n");
                    int N = 500;
                    // Warmup
                    for (int i = 0; i < 20; i++) {
                        ((BOOL(*)(id,SEL,id,id,id,unsigned int,NSError**))objc_msgSend)(
                            g_client, @selector(buffersReadyWithModel:inputBuffers:options:qos:error:),
                            kA.aneModel, ibr, @{}, 21, &e);
                        ((BOOL(*)(id,SEL,id,id,id,unsigned int,NSError**))objc_msgSend)(
                            g_client, @selector(enqueueSetsWithModel:outputSet:options:qos:error:),
                            kA.aneModel, ose, @{}, 21, &e);
                    }
                    uint64_t t0 = mach_absolute_time();
                    for (int i = 0; i < N; i++) {
                        ((BOOL(*)(id,SEL,id,id,id,unsigned int,NSError**))objc_msgSend)(
                            g_client, @selector(buffersReadyWithModel:inputBuffers:options:qos:error:),
                            kA.aneModel, ibr, @{}, 21, &e);
                        ((BOOL(*)(id,SEL,id,id,id,unsigned int,NSError**))objc_msgSend)(
                            g_client, @selector(enqueueSetsWithModel:outputSet:options:qos:error:),
                            kA.aneModel, ose, @{}, 21, &e);
                    }
                    double two_phase_ms = ms_t(mach_absolute_time() - t0);

                    // Baseline: doEvaluateDirectWithModel
                    for (int i = 0; i < 20; i++) {
                        ((BOOL(*)(id,SEL,id,id,id,NSError**))objc_msgSend)(
                            g_client, @selector(doEvaluateDirectWithModel:options:request:error:),
                            kA.aneModel, @{}, kA.request, &e);
                    }
                    t0 = mach_absolute_time();
                    for (int i = 0; i < N; i++) {
                        ((BOOL(*)(id,SEL,id,id,id,NSError**))objc_msgSend)(
                            g_client, @selector(doEvaluateDirectWithModel:options:request:error:),
                            kA.aneModel, @{}, kA.request, &e);
                    }
                    double baseline_ms = ms_t(mach_absolute_time() - t0);

                    printf("  Two-phase: %.1fms (%.3f ms/eval)\n", two_phase_ms, two_phase_ms/N);
                    printf("  Baseline:  %.1fms (%.3f ms/eval)\n", baseline_ms, baseline_ms/N);
                    printf("  Speedup: %.2fx\n", baseline_ms/two_phase_ms);
                }
            }
        } @catch (NSException *ex) {
            printf("  EXCEPTION: %s — %s\n", [[ex name] UTF8String], [[ex reason] UTF8String]);
        }

        printf("\nDone.\n");
        CFRelease(statsSurf);
    }
    return 0;
}
