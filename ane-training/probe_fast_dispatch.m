// probe_fast_dispatch.m — Find fastest ANE dispatch path
// Test all discovered eval methods on realistic kernel sizes
// Focus: minimizing per-dispatch overhead
#import <Foundation/Foundation.h>
#import <IOSurface/IOSurface.h>
#import <objc/runtime.h>
#import <objc/message.h>
#import <dlfcn.h>
#import <mach/mach_time.h>

static mach_timebase_info_data_t g_tb;
static double ms_t(uint64_t t) { return (double)t * g_tb.numer / g_tb.denom / 1e6; }
static Class g_D, g_I, g_AR, g_AIO;
static id g_client = nil;

static void ane_init(void) {
    dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine", RTLD_NOW);
    g_D  = NSClassFromString(@"_ANEInMemoryModelDescriptor");
    g_I  = NSClassFromString(@"_ANEInMemoryModel");
    g_AR = NSClassFromString(@"_ANERequest");
    g_AIO= NSClassFromString(@"_ANEIOSurfaceObject");
    g_client = ((id(*)(Class,SEL))objc_msgSend)(NSClassFromString(@"_ANEClient"), @selector(sharedConnection));
}

static IOSurfaceRef make_surf(size_t bytes) {
    return IOSurfaceCreate((__bridge CFDictionaryRef)@{
        (id)kIOSurfaceWidth:@(bytes), (id)kIOSurfaceHeight:@1,
        (id)kIOSurfaceBytesPerElement:@1, (id)kIOSurfaceBytesPerRow:@(bytes),
        (id)kIOSurfaceAllocSize:@(bytes), (id)kIOSurfacePixelFormat:@0});
}

static NSData *gen_matmul_mil(int ic, int oc, int seq) {
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

static Kern compile_kern(int ic, int oc, int seq) {
    Kern k = {0};
    NSData *mil = gen_matmul_mil(ic, oc, seq);
    size_t in_b = ic * (seq + oc) * 2, out_b = oc * seq * 2;
    id desc = ((id(*)(Class,SEL,id,id,id))objc_msgSend)(g_D, @selector(modelWithMILText:weights:optionsPlist:), mil, @{}, nil);
    k.model = ((id(*)(Class,SEL,id))objc_msgSend)(g_I, @selector(inMemoryModelWithDescriptor:), desc);
    id hx = ((id(*)(id,SEL))objc_msgSend)(k.model, @selector(hexStringIdentifier));
    NSString *td = [NSTemporaryDirectory() stringByAppendingPathComponent:hx];
    [[NSFileManager defaultManager] createDirectoryAtPath:[td stringByAppendingPathComponent:@"weights"] withIntermediateDirectories:YES attributes:nil error:nil];
    [mil writeToFile:[td stringByAppendingPathComponent:@"model.mil"] atomically:YES];
    NSError *e = nil;
    ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(k.model, @selector(compileWithQoS:options:error:), 21, @{}, &e);
    ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(k.model, @selector(loadWithQoS:options:error:), 21, @{}, &e);
    k.aneModel = ((id(*)(id,SEL))objc_msgSend)(k.model, @selector(model));
    k.ioIn = make_surf(in_b); k.ioOut = make_surf(out_b);
    _Float16 *p = (void*)IOSurfaceGetBaseAddress(k.ioIn);
    for (size_t i = 0; i < in_b/2; i++) p[i] = (_Float16)(0.01f * (i % 100));
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
        ane_init();

        printf("=== Fast Dispatch Benchmark ===\n\n");

        // Test on realistic Qwen3 kernel sizes
        typedef struct { int ic, oc, seq; const char *name; } Spec;
        Spec specs[] = {
            {1024, 2048, 256, "sdpaFwd-like"},
            {2048, 1024, 256, "woFwd-like"},
            {1024, 3072, 256, "ffnW1-like"},
        };

        for (int s = 0; s < 3; s++) {
            Spec *sp = &specs[s];
            printf("--- %s (%d→%d, seq=%d) ---\n", sp->name, sp->ic, sp->oc, sp->seq);
            Kern k = compile_kern(sp->ic, sp->oc, sp->seq);
            NSError *e = nil;
            int N = 500;

            // Warmup
            for (int i = 0; i < 50; i++)
                ((BOOL(*)(id,SEL,id,id,id,NSError**))objc_msgSend)(
                    g_client, @selector(evaluateRealTimeWithModel:options:request:error:),
                    k.aneModel, @{}, k.request, &e);

            // 1. evaluateWithQoS (original)
            uint64_t t0 = mach_absolute_time();
            for (int i = 0; i < N; i++)
                ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
                    k.model, @selector(evaluateWithQoS:options:request:error:), 21, @{}, k.request, &e);
            double qos_ms = ms_t(mach_absolute_time() - t0);

            // 2. evaluateRealTimeWithModel (current)
            t0 = mach_absolute_time();
            for (int i = 0; i < N; i++)
                ((BOOL(*)(id,SEL,id,id,id,NSError**))objc_msgSend)(
                    g_client, @selector(evaluateRealTimeWithModel:options:request:error:),
                    k.aneModel, @{}, k.request, &e);
            double rt_ms = ms_t(mach_absolute_time() - t0);

            // 3. doEvaluateDirectWithModel
            t0 = mach_absolute_time();
            for (int i = 0; i < N; i++)
                ((BOOL(*)(id,SEL,id,id,id,unsigned int,NSError**))objc_msgSend)(
                    g_client, @selector(doEvaluateDirectWithModel:options:request:qos:error:),
                    k.aneModel, @{}, k.request, 21, &e);
            double direct_ms = ms_t(mach_absolute_time() - t0);

            // 4. evaluateWithModel via client (bypasses _ANEInMemoryModel wrapper)
            t0 = mach_absolute_time();
            for (int i = 0; i < N; i++)
                ((BOOL(*)(id,SEL,id,id,id,unsigned int,NSError**))objc_msgSend)(
                    g_client, @selector(evaluateWithModel:options:request:qos:error:),
                    k.aneModel, @{}, k.request, 21, &e);
            double ewm_ms = ms_t(mach_absolute_time() - t0);

            // 5. evaluateRealTime with nil options (skip dict creation)
            t0 = mach_absolute_time();
            for (int i = 0; i < N; i++)
                ((BOOL(*)(id,SEL,id,id,id,NSError**))objc_msgSend)(
                    g_client, @selector(evaluateRealTimeWithModel:options:request:error:),
                    k.aneModel, nil, k.request, &e);
            double rt_nil_ms = ms_t(mach_absolute_time() - t0);

            // 6. Alternating between two pre-created requests (simulate per-layer requests)
            IOSurfaceRef ioIn2 = make_surf(sp->ic * (sp->seq + sp->oc) * 2);
            IOSurfaceRef ioOut2 = make_surf(sp->oc * sp->seq * 2);
            id wI2 = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_AIO, @selector(objectWithIOSurface:), ioIn2);
            id wO2 = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_AIO, @selector(objectWithIOSurface:), ioOut2);
            id req2 = ((id(*)(Class,SEL,id,id,id,id,id,id,id))objc_msgSend)(g_AR,
                @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:),
                @[wI2], @[@0], @[wO2], @[@0], nil, nil, @0);
            _Float16 *p2 = (void*)IOSurfaceGetBaseAddress(ioIn2);
            for (size_t i = 0; i < (size_t)(sp->ic * (sp->seq + sp->oc)); i++) p2[i] = (_Float16)(0.01f * (i % 100));

            t0 = mach_absolute_time();
            for (int i = 0; i < N; i++) {
                id req = (i & 1) ? req2 : k.request;
                ((BOOL(*)(id,SEL,id,id,id,NSError**))objc_msgSend)(
                    g_client, @selector(evaluateRealTimeWithModel:options:request:error:),
                    k.aneModel, @{}, req, &e);
            }
            double alt_ms = ms_t(mach_absolute_time() - t0);
            CFRelease(ioIn2); CFRelease(ioOut2);

            // 7. Simulated training layer pipeline: 28 dispatches with same model, different requests
            // This mimics what the training loop does — pre-allocated per-layer requests
            #define NLAYERS_TEST 28
            IOSurfaceRef layerIn[NLAYERS_TEST], layerOut[NLAYERS_TEST];
            id layerReq[NLAYERS_TEST];
            for (int L = 0; L < NLAYERS_TEST; L++) {
                layerIn[L] = make_surf(sp->ic * (sp->seq + sp->oc) * 2);
                layerOut[L] = make_surf(sp->oc * sp->seq * 2);
                _Float16 *pl = (void*)IOSurfaceGetBaseAddress(layerIn[L]);
                for (size_t i = 0; i < (size_t)(sp->ic * (sp->seq + sp->oc)); i++) pl[i] = (_Float16)(0.01f * (i % 100));
                id wIL = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_AIO, @selector(objectWithIOSurface:), layerIn[L]);
                id wOL = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_AIO, @selector(objectWithIOSurface:), layerOut[L]);
                layerReq[L] = ((id(*)(Class,SEL,id,id,id,id,id,id,id))objc_msgSend)(g_AR,
                    @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:),
                    @[wIL], @[@0], @[wOL], @[@0], nil, nil, @0);
            }

            int reps = N / NLAYERS_TEST;
            // Sequential (current approach)
            t0 = mach_absolute_time();
            for (int r = 0; r < reps; r++)
                for (int L = 0; L < NLAYERS_TEST; L++)
                    ((BOOL(*)(id,SEL,id,id,id,NSError**))objc_msgSend)(
                        g_client, @selector(evaluateRealTimeWithModel:options:request:error:),
                        k.aneModel, @{}, layerReq[L], &e);
            double layer_seq_ms = ms_t(mach_absolute_time() - t0);

            // Burst: fire all layer dispatches async then wait
            dispatch_queue_attr_t attr = dispatch_queue_attr_make_with_qos_class(
                DISPATCH_QUEUE_SERIAL, QOS_CLASS_USER_INTERACTIVE, 0);
            dispatch_queue_t q = dispatch_queue_create("burst", attr);
            t0 = mach_absolute_time();
            for (int r = 0; r < reps; r++) {
                dispatch_semaphore_t done = dispatch_semaphore_create(0);
                __block int completed = 0;
                int total = NLAYERS_TEST;
                for (int L = 0; L < NLAYERS_TEST; L++) {
                    id aneM = k.aneModel;
                    id lreq = layerReq[L];
                    id cli = g_client;
                    dispatch_async(q, ^{
                        NSError *err = nil;
                        ((BOOL(*)(id,SEL,id,id,id,NSError**))objc_msgSend)(
                            cli, @selector(evaluateRealTimeWithModel:options:request:error:),
                            aneM, @{}, lreq, &err);
                        if (__sync_add_and_fetch(&completed, 1) == total)
                            dispatch_semaphore_signal(done);
                    });
                }
                dispatch_semaphore_wait(done, DISPATCH_TIME_FOREVER);
            }
            double layer_burst_ms = ms_t(mach_absolute_time() - t0);

            for (int L = 0; L < NLAYERS_TEST; L++) {
                CFRelease(layerIn[L]); CFRelease(layerOut[L]);
            }

            printf("  evaluateWithQoS:        %5.3f ms/eval\n", qos_ms/N);
            printf("  evaluateRealTime:       %5.3f ms/eval\n", rt_ms/N);
            printf("  doEvaluateDirect:       %5.3f ms/eval\n", direct_ms/N);
            printf("  evaluateWithModel:      %5.3f ms/eval\n", ewm_ms/N);
            printf("  realTime (nil opts):    %5.3f ms/eval\n", rt_nil_ms/N);
            printf("  alternating requests:   %5.3f ms/eval\n", alt_ms/N);
            printf("  28-layer sequential:    %5.3f ms/eval (%5.1f ms total)\n",
                   layer_seq_ms/(reps*NLAYERS_TEST), layer_seq_ms/reps);
            printf("  28-layer burst:         %5.3f ms/eval (%5.1f ms total)\n",
                   layer_burst_ms/(reps*NLAYERS_TEST), layer_burst_ms/reps);
            printf("  burst speedup: %.2fx\n\n", layer_seq_ms / layer_burst_ms);

            ((BOOL(*)(id,SEL,unsigned int,NSError**))objc_msgSend)(k.model, @selector(unloadWithQoS:error:), 21, &e);
            CFRelease(k.ioIn); CFRelease(k.ioOut);
        }

        printf("=== Done ===\n");
    }
    return 0;
}
