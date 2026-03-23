// bench_pipeline.m — Test if ANE benefits from pipelined concurrent submissions
// Compares: sequential eval vs concurrent submission with queueDepth > 1
#import <Foundation/Foundation.h>
#import <IOSurface/IOSurface.h>
#import <objc/runtime.h>
#import <objc/message.h>
#import <dlfcn.h>
#import <mach/mach_time.h>
#import <dispatch/dispatch.h>

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

static IOSurfaceRef make_surf(size_t bytes) {
    return IOSurfaceCreate((__bridge CFDictionaryRef)@{
        (id)kIOSurfaceWidth:@(bytes), (id)kIOSurfaceHeight:@1,
        (id)kIOSurfaceBytesPerElement:@1, (id)kIOSurfaceBytesPerRow:@(bytes),
        (id)kIOSurfaceAllocSize:@(bytes), (id)kIOSurfacePixelFormat:@0});
}

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
    IOSurfaceRef ioIn, ioOut;
    id request;
} Kern;

static Kern compile_kern(int ic, int oc, int seq) {
    Kern k = {0};
    NSData *mil = gen_matmul_mil(ic, oc, seq);
    size_t in_bytes = ic * (seq + oc) * 2;
    size_t out_bytes = oc * seq * 2;
    id desc = ((id(*)(Class,SEL,id,id,id))objc_msgSend)(g_D, @selector(modelWithMILText:weights:optionsPlist:), mil, @{}, nil);
    k.model = ((id(*)(Class,SEL,id))objc_msgSend)(g_I, @selector(inMemoryModelWithDescriptor:), desc);
    id hx = ((id(*)(id,SEL))objc_msgSend)(k.model, @selector(hexStringIdentifier));
    NSString *td = [NSTemporaryDirectory() stringByAppendingPathComponent:hx];
    [[NSFileManager defaultManager] createDirectoryAtPath:[td stringByAppendingPathComponent:@"weights"]
        withIntermediateDirectories:YES attributes:nil error:nil];
    [mil writeToFile:[td stringByAppendingPathComponent:@"model.mil"] atomically:YES];
    NSError *e = nil;
    ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(k.model, @selector(compileWithQoS:options:error:), 21, @{}, &e);
    ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(k.model, @selector(loadWithQoS:options:error:), 21, @{}, &e);
    k.ioIn = make_surf(in_bytes);
    k.ioOut = make_surf(out_bytes);
    IOSurfaceLock(k.ioIn, 0, NULL);
    _Float16 *inp = (void*)IOSurfaceGetBaseAddress(k.ioIn);
    for (size_t i = 0; i < in_bytes/2; i++) inp[i] = (_Float16)(0.01f * (i % 100));
    IOSurfaceUnlock(k.ioIn, 0, NULL);
    id wI = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_AIO, @selector(objectWithIOSurface:), k.ioIn);
    id wO = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_AIO, @selector(objectWithIOSurface:), k.ioOut);
    k.request = ((id(*)(Class,SEL,id,id,id,id,id,id,id))objc_msgSend)(g_AR,
        @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:),
        @[wI], @[@0], @[wO], @[@0], nil, nil, @0);
    k.aneModel = ((id(*)(id,SEL))objc_msgSend)(k.model, @selector(model));
    return k;
}

static void sync_eval(Kern *k) {
    NSError *e = nil;
    ((BOOL(*)(id,SEL,id,id,id,NSError**))objc_msgSend)(
        g_client, @selector(evaluateRealTimeWithModel:options:request:error:),
        k->aneModel, @{}, k->request, &e);
}

int main(int argc, char *argv[]) {
    @autoreleasepool {
        setbuf(stdout, NULL);
        mach_timebase_info(&g_tb);
        ane_init();

        printf("=== ANE Pipeline Benchmark ===\n\n");

        // Compile 2 different kernels (simulating forward: sdpaFwd + ffnFused)
        printf("Compiling kernels...\n");
        Kern kA = compile_kern(1024, 2048, 256);  // sdpaFwd-like
        Kern kB = compile_kern(1024, 3072, 256);  // ffnFused-like

        // Also compile with separate IOSurfaces for pipelining
        Kern kA2 = compile_kern(1024, 2048, 256);
        Kern kB2 = compile_kern(1024, 3072, 256);

        int N = 100;

        // Warmup
        for (int i = 0; i < 20; i++) { sync_eval(&kA); sync_eval(&kB); }

        // Test 1: Sequential A→B→A→B (current training pattern)
        uint64_t t0 = mach_absolute_time();
        for (int i = 0; i < N; i++) {
            sync_eval(&kA);
            sync_eval(&kB);
        }
        double seq_ms = ms_t(mach_absolute_time() - t0);

        // Test 2: Concurrent submission via dispatch_async (2 queues)
        dispatch_queue_t q1 = dispatch_queue_create("ane_q1",
            dispatch_queue_attr_make_with_qos_class(DISPATCH_QUEUE_SERIAL, QOS_CLASS_USER_INTERACTIVE, 0));
        dispatch_queue_t q2 = dispatch_queue_create("ane_q2",
            dispatch_queue_attr_make_with_qos_class(DISPATCH_QUEUE_SERIAL, QOS_CLASS_USER_INTERACTIVE, 0));

        // Warmup concurrent
        for (int i = 0; i < 20; i++) {
            dispatch_semaphore_t s1 = dispatch_semaphore_create(0);
            dispatch_semaphore_t s2 = dispatch_semaphore_create(0);
            dispatch_async(q1, ^{ sync_eval(&kA2); dispatch_semaphore_signal(s1); });
            dispatch_async(q2, ^{ sync_eval(&kB2); dispatch_semaphore_signal(s2); });
            dispatch_semaphore_wait(s1, DISPATCH_TIME_FOREVER);
            dispatch_semaphore_wait(s2, DISPATCH_TIME_FOREVER);
        }

        t0 = mach_absolute_time();
        for (int i = 0; i < N; i++) {
            dispatch_semaphore_t s1 = dispatch_semaphore_create(0);
            dispatch_semaphore_t s2 = dispatch_semaphore_create(0);
            dispatch_async(q1, ^{ sync_eval(&kA2); dispatch_semaphore_signal(s1); });
            dispatch_async(q2, ^{ sync_eval(&kB2); dispatch_semaphore_signal(s2); });
            dispatch_semaphore_wait(s1, DISPATCH_TIME_FOREVER);
            dispatch_semaphore_wait(s2, DISPATCH_TIME_FOREVER);
        }
        double conc_ms = ms_t(mach_absolute_time() - t0);

        // Test 3: Fire-and-forget burst (submit N evals, wait at end)
        dispatch_queue_t burst_q = dispatch_queue_create("ane_burst",
            dispatch_queue_attr_make_with_qos_class(DISPATCH_QUEUE_SERIAL, QOS_CLASS_USER_INTERACTIVE, 0));

        // Warmup
        for (int i = 0; i < 20; i++) sync_eval(&kA);

        t0 = mach_absolute_time();
        dispatch_semaphore_t burst_done = dispatch_semaphore_create(0);
        for (int i = 0; i < 2*N; i++) {
            Kern *k = (i % 2 == 0) ? &kA : &kB;
            if (i == 2*N - 1) {
                dispatch_async(burst_q, ^{
                    sync_eval(k);
                    dispatch_semaphore_signal(burst_done);
                });
            } else {
                dispatch_async(burst_q, ^{ sync_eval(k); });
            }
        }
        dispatch_semaphore_wait(burst_done, DISPATCH_TIME_FOREVER);
        double burst_ms = ms_t(mach_absolute_time() - t0);

        // Test 4: Direct sequential on main thread (no dispatch overhead)
        t0 = mach_absolute_time();
        for (int i = 0; i < N; i++) {
            sync_eval(&kA);
            sync_eval(&kB);
        }
        double direct_ms = ms_t(mach_absolute_time() - t0);

        // Test 5: Pipelined — submit kA, immediately submit kB, then wait both
        // Uses separate IOSurfaces to avoid data hazards
        dispatch_queue_t pq1 = dispatch_queue_create("pipe1",
            dispatch_queue_attr_make_with_qos_class(DISPATCH_QUEUE_SERIAL, QOS_CLASS_USER_INTERACTIVE, 0));
        dispatch_queue_t pq2 = dispatch_queue_create("pipe2",
            dispatch_queue_attr_make_with_qos_class(DISPATCH_QUEUE_SERIAL, QOS_CLASS_USER_INTERACTIVE, 0));

        // Warmup
        for (int i = 0; i < 20; i++) {
            dispatch_semaphore_t s1 = dispatch_semaphore_create(0);
            dispatch_async(pq1, ^{ sync_eval(&kA); dispatch_semaphore_signal(s1); });
            dispatch_semaphore_wait(s1, DISPATCH_TIME_FOREVER);
        }

        t0 = mach_absolute_time();
        for (int i = 0; i < N; i++) {
            // Submit both concurrently
            dispatch_semaphore_t s1 = dispatch_semaphore_create(0);
            dispatch_semaphore_t s2 = dispatch_semaphore_create(0);
            dispatch_async(pq1, ^{ sync_eval(&kA); dispatch_semaphore_signal(s1); });
            // Don't wait for kA — immediately submit kB
            dispatch_async(pq2, ^{ sync_eval(&kB); dispatch_semaphore_signal(s2); });
            // Now wait for both
            dispatch_semaphore_wait(s1, DISPATCH_TIME_FOREVER);
            dispatch_semaphore_wait(s2, DISPATCH_TIME_FOREVER);
        }
        double pipe_ms = ms_t(mach_absolute_time() - t0);

        printf("\nResults (%d iterations of kA+kB each):\n", N);
        printf("  Sequential (main thread):  %6.1f ms (%5.3f ms/pair)\n", direct_ms, direct_ms/N);
        printf("  Sequential (async queue):  %6.1f ms (%5.3f ms/pair)\n", seq_ms, seq_ms/N);
        printf("  Concurrent (2 queues):     %6.1f ms (%5.3f ms/pair)\n", conc_ms, conc_ms/N);
        printf("  Burst (submit all, wait):  %6.1f ms (%5.3f ms/pair)\n", burst_ms, burst_ms/N);
        printf("  Pipeline (submit both):    %6.1f ms (%5.3f ms/pair)\n", pipe_ms, pipe_ms/N);
        printf("\n  Concurrent speedup vs seq: %.2fx\n", direct_ms / conc_ms);
        printf("  Pipeline speedup vs seq:   %.2fx\n", direct_ms / pipe_ms);
        printf("  Dispatch overhead:         %.3f ms/pair\n", (seq_ms - direct_ms) / N);

        printf("\n=== Done ===\n");
    }
    return 0;
}
