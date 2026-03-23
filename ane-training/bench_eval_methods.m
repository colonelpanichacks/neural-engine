// bench_eval_methods.m — Compare evaluateWithQoS vs evaluateRealTimeWithModel
// Tests with realistic Qwen3-sized kernels (DIM=1024, SEQ=256)
#import <Foundation/Foundation.h>
#import <IOSurface/IOSurface.h>
#import <objc/runtime.h>
#import <objc/message.h>
#import <dlfcn.h>
#import <mach/mach_time.h>

static mach_timebase_info_data_t g_tb;
static double ms_t(uint64_t t) { return (double)t * g_tb.numer / g_tb.denom / 1e6; }

static Class g_D, g_I, g_AR, g_AIO;

static void ane_init(void) {
    dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine", RTLD_NOW);
    g_D  = NSClassFromString(@"_ANEInMemoryModelDescriptor");
    g_I  = NSClassFromString(@"_ANEInMemoryModel");
    g_AR = NSClassFromString(@"_ANERequest");
    g_AIO= NSClassFromString(@"_ANEIOSurfaceObject");
}

static IOSurfaceRef make_surf(size_t bytes) {
    return IOSurfaceCreate((__bridge CFDictionaryRef)@{
        (id)kIOSurfaceWidth:@(bytes), (id)kIOSurfaceHeight:@1,
        (id)kIOSurfaceBytesPerElement:@1, (id)kIOSurfaceBytesPerRow:@(bytes),
        (id)kIOSurfaceAllocSize:@(bytes), (id)kIOSurfacePixelFormat:@0});
}

// Generate dynamic matmul MIL: y = x_act @ x_weight
static NSData *gen_matmul_mil(int ic, int oc, int seq) {
    NSMutableString *m = [NSMutableString string];
    [m appendString:@"program(1.3)\n"
        "[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}, "
        "{\"coremlc-version\", \"3505.4.1\"}, {\"coremltools-component-milinternal\", \"\"}, "
        "{\"coremltools-version\", \"9.0\"}})]\n{\n"];
    int sp = seq + oc;
    [m appendFormat:@"    func main<ios18>(tensor<fp16, [1, %d, 1, %d]> x) {\n", ic, sp];
    [m appendString:@"        tensor<int32, [4]> ba = const()[name=string(\"ba\"), val=tensor<int32, [4]>([0,0,0,0])];\n"];
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

typedef struct {
    id model;
    id aneModel;
    id client;
    IOSurfaceRef ioIn, ioOut;
    id request;
} TestKernel;

static TestKernel compile_kernel(int ic, int oc, int seq) {
    TestKernel tk = {0};
    NSData *mil = gen_matmul_mil(ic, oc, seq);
    size_t in_bytes = ic * (seq + oc) * 2;
    size_t out_bytes = oc * seq * 2;

    id desc = ((id(*)(Class,SEL,id,id,id))objc_msgSend)(g_D, @selector(modelWithMILText:weights:optionsPlist:), mil, @{}, nil);
    tk.model = ((id(*)(Class,SEL,id))objc_msgSend)(g_I, @selector(inMemoryModelWithDescriptor:), desc);
    id hx = ((id(*)(id,SEL))objc_msgSend)(tk.model, @selector(hexStringIdentifier));
    NSString *td = [NSTemporaryDirectory() stringByAppendingPathComponent:hx];
    [[NSFileManager defaultManager] createDirectoryAtPath:[td stringByAppendingPathComponent:@"weights"]
        withIntermediateDirectories:YES attributes:nil error:nil];
    [mil writeToFile:[td stringByAppendingPathComponent:@"model.mil"] atomically:YES];

    NSError *e = nil;
    ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(tk.model, @selector(compileWithQoS:options:error:), 21, @{}, &e);
    ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(tk.model, @selector(loadWithQoS:options:error:), 21, @{}, &e);

    tk.ioIn = make_surf(in_bytes);
    tk.ioOut = make_surf(out_bytes);

    // Fill with test data
    IOSurfaceLock(tk.ioIn, 0, NULL);
    _Float16 *inp = (void*)IOSurfaceGetBaseAddress(tk.ioIn);
    for (size_t i = 0; i < in_bytes/2; i++) inp[i] = (_Float16)(0.01f * (i % 100));
    IOSurfaceUnlock(tk.ioIn, 0, NULL);

    id wI = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_AIO, @selector(objectWithIOSurface:), tk.ioIn);
    id wO = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_AIO, @selector(objectWithIOSurface:), tk.ioOut);
    tk.request = ((id(*)(Class,SEL,id,id,id,id,id,id,id))objc_msgSend)(g_AR,
        @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:),
        @[wI], @[@0], @[wO], @[@0], nil, nil, @0);

    tk.aneModel = ((id(*)(id,SEL))objc_msgSend)(tk.model, @selector(model));
    tk.client = ((id(*)(Class,SEL))objc_msgSend)(NSClassFromString(@"_ANEClient"), @selector(sharedConnection));

    return tk;
}

int main(int argc, char *argv[]) {
    @autoreleasepool {
        setbuf(stdout, NULL);
        mach_timebase_info(&g_tb);
        ane_init();

        printf("=== ANE Eval Method Benchmark ===\n\n");

        // Test sizes matching Qwen3 kernels
        typedef struct { int ic, oc, seq; const char *name; } KernSpec;
        KernSpec specs[] = {
            {1024, 2048, 256, "sdpaFwd-like (1024->2048, seq=256)"},
            {2048, 1024, 256, "woFwd-like (2048->1024, seq=256)"},
            {1024, 3072, 256, "ffnW1-like (1024->3072, seq=256)"},
            {3072, 1024, 256, "ffnW2-like (3072->1024, seq=256)"},
            {256, 256, 64,   "small (256->256, seq=64)"},
        };
        int n_specs = sizeof(specs)/sizeof(specs[0]);

        for (int s = 0; s < n_specs; s++) {
            KernSpec *sp = &specs[s];
            printf("--- %s ---\n", sp->name);
            TestKernel tk = compile_kernel(sp->ic, sp->oc, sp->seq);

            int N = 200;
            NSError *e = nil;

            // Warmup
            for (int i = 0; i < 20; i++) {
                ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
                    tk.model, @selector(evaluateWithQoS:options:request:error:), 21, @{}, tk.request, &e);
            }

            // Method 1: evaluateWithQoS (standard)
            uint64_t t0 = mach_absolute_time();
            for (int i = 0; i < N; i++) {
                ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
                    tk.model, @selector(evaluateWithQoS:options:request:error:), 21, @{}, tk.request, &e);
            }
            double std_ms = ms_t(mach_absolute_time() - t0);

            // Method 2: evaluateRealTimeWithModel (via _ANEClient)
            t0 = mach_absolute_time();
            for (int i = 0; i < N; i++) {
                ((BOOL(*)(id,SEL,id,id,id,NSError**))objc_msgSend)(
                    tk.client, @selector(evaluateRealTimeWithModel:options:request:error:),
                    tk.aneModel, @{}, tk.request, &e);
            }
            double rt_ms = ms_t(mach_absolute_time() - t0);

            // Method 3: evaluateWithQoS + completionHandler (async fire-and-forget)
            __block int completed = 0;
            dispatch_semaphore_t done = dispatch_semaphore_create(0);
            t0 = mach_absolute_time();
            for (int i = 0; i < N; i++) {
                id req2 = ((id(*)(Class,SEL,id,id,id,id,id,id,id))objc_msgSend)(g_AR,
                    @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:),
                    @[((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_AIO, @selector(objectWithIOSurface:), tk.ioIn)],
                    @[@0],
                    @[((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_AIO, @selector(objectWithIOSurface:), tk.ioOut)],
                    @[@0], nil, nil, @0);
                void (^handler)(void) = ^{
                    completed++;
                    if (completed == N) dispatch_semaphore_signal(done);
                };
                ((void(*)(id,SEL,id))objc_msgSend)(req2, @selector(setCompletionHandler:), handler);
                ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
                    tk.model, @selector(evaluateWithQoS:options:request:error:), 21, @{}, req2, &e);
            }
            dispatch_semaphore_wait(done, DISPATCH_TIME_FOREVER);
            double async_ms = ms_t(mach_absolute_time() - t0);

            printf("  evaluateWithQoS:       %6.1f ms (%5.3f ms/eval)\n", std_ms, std_ms/N);
            printf("  evaluateRealTime:      %6.1f ms (%5.3f ms/eval)\n", rt_ms, rt_ms/N);
            printf("  evalWithQoS+callback:  %6.1f ms (%5.3f ms/eval)\n", async_ms, async_ms/N);
            printf("  realtime speedup: %.1fx\n\n", std_ms / rt_ms);

            // Cleanup
            ((BOOL(*)(id,SEL,unsigned int,NSError**))objc_msgSend)(tk.model, @selector(unloadWithQoS:error:), 21, &e);
            CFRelease(tk.ioIn); CFRelease(tk.ioOut);
        }

        printf("=== Done ===\n");
    }
    return 0;
}
