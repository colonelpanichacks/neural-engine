// probe_pipeline.m — Test double-buffered pipelined ANE dispatch
// Goal: overlap IO staging for kernel N+1 with ANE execution of kernel N
// Also probe: multi-request batching, _ANEClient methods
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

typedef struct {
    id model, aneModel;
    IOSurfaceRef ioIn[2], ioOut[2];  // Double-buffered
    id request[2];                    // Requests per buffer
} PipeKern;

static PipeKern compile_pipe(int ic, int oc, int seq) {
    PipeKern pk = {0};
    NSData *mil = gen_matmul_mil(ic, oc, seq);
    size_t in_bytes = ic * (seq + oc) * 2;
    size_t out_bytes = oc * seq * 2;

    id desc = ((id(*)(Class,SEL,id,id,id))objc_msgSend)(g_D, @selector(modelWithMILText:weights:optionsPlist:), mil, @{}, nil);
    pk.model = ((id(*)(Class,SEL,id))objc_msgSend)(g_I, @selector(inMemoryModelWithDescriptor:), desc);
    id hx = ((id(*)(id,SEL))objc_msgSend)(pk.model, @selector(hexStringIdentifier));
    NSString *td = [NSTemporaryDirectory() stringByAppendingPathComponent:hx];
    [[NSFileManager defaultManager] createDirectoryAtPath:[td stringByAppendingPathComponent:@"weights"]
        withIntermediateDirectories:YES attributes:nil error:nil];
    [mil writeToFile:[td stringByAppendingPathComponent:@"model.mil"] atomically:YES];

    NSError *e = nil;
    ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(pk.model, @selector(compileWithQoS:options:error:), 21, @{}, &e);
    ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(pk.model, @selector(loadWithQoS:options:error:), 21, @{}, &e);
    pk.aneModel = ((id(*)(id,SEL))objc_msgSend)(pk.model, @selector(model));

    for (int b = 0; b < 2; b++) {
        pk.ioIn[b] = make_surf(in_bytes);
        pk.ioOut[b] = make_surf(out_bytes);
        // Fill with test data
        _Float16 *inp = (void*)IOSurfaceGetBaseAddress(pk.ioIn[b]);
        for (size_t i = 0; i < in_bytes/2; i++) inp[i] = (_Float16)(0.01f * (i % 100));
        id wI = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_AIO, @selector(objectWithIOSurface:), pk.ioIn[b]);
        id wO = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_AIO, @selector(objectWithIOSurface:), pk.ioOut[b]);
        pk.request[b] = ((id(*)(Class,SEL,id,id,id,id,id,id,id))objc_msgSend)(g_AR,
            @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:),
            @[wI], @[@0], @[wO], @[@0], nil, nil, @0);
    }
    return pk;
}

// Dump all _ANEClient methods
static void dump_client_methods(void) {
    Class cls = NSClassFromString(@"_ANEClient");
    printf("=== _ANEClient instance methods ===\n");
    unsigned int count = 0;
    Method *methods = class_copyMethodList(cls, &count);
    for (unsigned int i = 0; i < count; i++) {
        SEL sel = method_getName(methods[i]);
        printf("  %s\n", sel_getName(sel));
    }
    free(methods);
    printf("Total: %u methods\n\n", count);
}

// Dump _ANERequest methods
static void dump_request_methods(void) {
    Class cls = NSClassFromString(@"_ANERequest");
    printf("=== _ANERequest class & instance methods ===\n");
    unsigned int count = 0;
    Method *methods = class_copyMethodList(cls, &count);
    for (unsigned int i = 0; i < count; i++) {
        SEL sel = method_getName(methods[i]);
        printf("  - %s\n", sel_getName(sel));
    }
    free(methods);

    Method *cmethods = class_copyMethodList(object_getClass(cls), &count);
    for (unsigned int i = 0; i < count; i++) {
        SEL sel = method_getName(cmethods[i]);
        printf("  + %s\n", sel_getName(cmethods[i]));
    }
    free(cmethods);
    printf("\n");
}

// Dump _ANEInMemoryModel methods
static void dump_model_methods(void) {
    Class cls = NSClassFromString(@"_ANEInMemoryModel");
    printf("=== _ANEInMemoryModel instance methods ===\n");
    unsigned int count = 0;
    Method *methods = class_copyMethodList(cls, &count);
    for (unsigned int i = 0; i < count; i++) {
        printf("  %s\n", sel_getName(method_getName(methods[i])));
    }
    free(methods);
    printf("Total: %u methods\n\n", count);
}

int main(int argc, char *argv[]) {
    @autoreleasepool {
        setbuf(stdout, NULL);
        mach_timebase_info(&g_tb);
        ane_init();

        printf("=== ANE Pipeline & API Probe ===\n\n");

        // Dump all relevant class methods
        dump_client_methods();
        dump_request_methods();
        dump_model_methods();

        // Also dump _ANEModel (the inner model from .model property)
        Class aneModelCls = NSClassFromString(@"_ANEModel");
        if (aneModelCls) {
            printf("=== _ANEModel instance methods ===\n");
            unsigned int count = 0;
            Method *methods = class_copyMethodList(aneModelCls, &count);
            for (unsigned int i = 0; i < count; i++)
                printf("  %s\n", sel_getName(method_getName(methods[i])));
            free(methods);
            printf("Total: %u methods\n\n", count);
        }

        // Compile a Qwen3-sized kernel for pipeline testing
        int ic = 1024, oc = 2048, seq = 256;
        printf("Compiling pipeline kernel (%d->%d, seq=%d)...\n", ic, oc, seq);
        PipeKern pk = compile_pipe(ic, oc, seq);
        NSError *e = nil;
        int N = 200;

        // Warmup
        for (int i = 0; i < 20; i++) {
            ((BOOL(*)(id,SEL,id,id,id,NSError**))objc_msgSend)(
                g_client, @selector(evaluateRealTimeWithModel:options:request:error:),
                pk.aneModel, @{}, pk.request[0], &e);
        }

        // Test 1: Sequential (baseline)
        uint64_t t0 = mach_absolute_time();
        for (int i = 0; i < N; i++) {
            ((BOOL(*)(id,SEL,id,id,id,NSError**))objc_msgSend)(
                g_client, @selector(evaluateRealTimeWithModel:options:request:error:),
                pk.aneModel, @{}, pk.request[0], &e);
        }
        double seq_ms = ms_t(mach_absolute_time() - t0);

        // Test 2: Double-buffered pipelined dispatch
        // Alternate between buffer 0 and buffer 1 using async dispatch
        dispatch_queue_attr_t attr = dispatch_queue_attr_make_with_qos_class(
            DISPATCH_QUEUE_SERIAL, QOS_CLASS_USER_INTERACTIVE, 0);
        dispatch_queue_t ane_q = dispatch_queue_create("ane_pipe", attr);

        t0 = mach_absolute_time();
        __block dispatch_semaphore_t prev_sem = NULL;
        for (int i = 0; i < N; i++) {
            int buf = i & 1;
            dispatch_semaphore_t sem = dispatch_semaphore_create(0);
            id aneModel = pk.aneModel;
            id req = pk.request[buf];
            id client = g_client;

            dispatch_async(ane_q, ^{
                NSError *err = nil;
                ((BOOL(*)(id,SEL,id,id,id,NSError**))objc_msgSend)(
                    client, @selector(evaluateRealTimeWithModel:options:request:error:),
                    aneModel, @{}, req, &err);
                dispatch_semaphore_signal(sem);
            });

            // Simulate IO staging work on CPU while ANE runs (memset the OTHER buffer)
            int other = 1 - buf;
            _Float16 *p = (_Float16*)IOSurfaceGetBaseAddress(pk.ioIn[other]);
            memset(p, 0, ic * (seq + oc) * 2);

            if (prev_sem) dispatch_semaphore_wait(prev_sem, DISPATCH_TIME_FOREVER);
            prev_sem = sem;
        }
        if (prev_sem) dispatch_semaphore_wait(prev_sem, DISPATCH_TIME_FOREVER);
        double pipe_ms = ms_t(mach_absolute_time() - t0);

        // Test 3: Fire N evals as fast as possible (queue depth test)
        dispatch_semaphore_t all_done = dispatch_semaphore_create(0);
        __block int completed = 0;
        t0 = mach_absolute_time();
        for (int i = 0; i < N; i++) {
            int buf = i & 1;
            id aneModel = pk.aneModel;
            id req = pk.request[buf];
            id client = g_client;
            dispatch_async(ane_q, ^{
                NSError *err = nil;
                ((BOOL(*)(id,SEL,id,id,id,NSError**))objc_msgSend)(
                    client, @selector(evaluateRealTimeWithModel:options:request:error:),
                    aneModel, @{}, req, &err);
                if (__sync_add_and_fetch(&completed, 1) == N)
                    dispatch_semaphore_signal(all_done);
            });
        }
        dispatch_semaphore_wait(all_done, DISPATCH_TIME_FOREVER);
        double burst_ms = ms_t(mach_absolute_time() - t0);

        // Test 4: Two DIFFERENT kernels alternating (simulates sdpaFwd→woFwd pipeline)
        int ic2 = 2048, oc2 = 1024;
        printf("Compiling second kernel (%d->%d, seq=%d)...\n", ic2, oc2, seq);
        PipeKern pk2 = compile_pipe(ic2, oc2, seq);
        for (int i = 0; i < 20; i++) {
            ((BOOL(*)(id,SEL,id,id,id,NSError**))objc_msgSend)(
                g_client, @selector(evaluateRealTimeWithModel:options:request:error:),
                pk2.aneModel, @{}, pk2.request[0], &e);
        }

        // Baseline: sequential A then B
        t0 = mach_absolute_time();
        for (int i = 0; i < N; i++) {
            ((BOOL(*)(id,SEL,id,id,id,NSError**))objc_msgSend)(
                g_client, @selector(evaluateRealTimeWithModel:options:request:error:),
                pk.aneModel, @{}, pk.request[0], &e);
            ((BOOL(*)(id,SEL,id,id,id,NSError**))objc_msgSend)(
                g_client, @selector(evaluateRealTimeWithModel:options:request:error:),
                pk2.aneModel, @{}, pk2.request[0], &e);
        }
        double ab_seq_ms = ms_t(mach_absolute_time() - t0);

        // Pipelined: dispatch A async, stage B's input, wait A, dispatch B async, stage A's input, wait B
        t0 = mach_absolute_time();
        for (int i = 0; i < N; i++) {
            // Dispatch A
            dispatch_semaphore_t sem_a = dispatch_semaphore_create(0);
            id aneModelA = pk.aneModel;
            id reqA = pk.request[0];
            id client = g_client;
            dispatch_async(ane_q, ^{
                NSError *err = nil;
                ((BOOL(*)(id,SEL,id,id,id,NSError**))objc_msgSend)(
                    client, @selector(evaluateRealTimeWithModel:options:request:error:),
                    aneModelA, @{}, reqA, &err);
                dispatch_semaphore_signal(sem_a);
            });
            // Overlap: "stage" B's input
            memset(IOSurfaceGetBaseAddress(pk2.ioIn[0]), 0, ic2 * (seq + oc2) * 2);
            dispatch_semaphore_wait(sem_a, DISPATCH_TIME_FOREVER);

            // Dispatch B
            dispatch_semaphore_t sem_b = dispatch_semaphore_create(0);
            id aneModelB = pk2.aneModel;
            id reqB = pk2.request[0];
            dispatch_async(ane_q, ^{
                NSError *err = nil;
                ((BOOL(*)(id,SEL,id,id,id,NSError**))objc_msgSend)(
                    client, @selector(evaluateRealTimeWithModel:options:request:error:),
                    aneModelB, @{}, reqB, &err);
                dispatch_semaphore_signal(sem_b);
            });
            // Overlap: "stage" A's input
            memset(IOSurfaceGetBaseAddress(pk.ioIn[0]), 0, ic * (seq + oc) * 2);
            dispatch_semaphore_wait(sem_b, DISPATCH_TIME_FOREVER);
        }
        double ab_pipe_ms = ms_t(mach_absolute_time() - t0);

        printf("\n=== Results (N=%d) ===\n", N);
        printf("Sequential (same kernel):     %6.1f ms (%5.3f ms/eval)\n", seq_ms, seq_ms/N);
        printf("Double-buffered pipeline:     %6.1f ms (%5.3f ms/eval)\n", pipe_ms, pipe_ms/N);
        printf("Burst fire (queue all):       %6.1f ms (%5.3f ms/eval)\n", burst_ms, burst_ms/N);
        printf("A→B sequential:               %6.1f ms (%5.3f ms/pair)\n", ab_seq_ms, ab_seq_ms/N);
        printf("A→B pipelined (overlap IO):   %6.1f ms (%5.3f ms/pair)\n", ab_pipe_ms, ab_pipe_ms/N);
        printf("Pipeline speedup (same):  %.2fx\n", seq_ms / pipe_ms);
        printf("Pipeline speedup (A→B):   %.2fx\n", ab_seq_ms / ab_pipe_ms);

        // Cleanup
        ((BOOL(*)(id,SEL,unsigned int,NSError**))objc_msgSend)(pk.model, @selector(unloadWithQoS:error:), 21, &e);
        ((BOOL(*)(id,SEL,unsigned int,NSError**))objc_msgSend)(pk2.model, @selector(unloadWithQoS:error:), 21, &e);
        for (int b = 0; b < 2; b++) {
            CFRelease(pk.ioIn[b]); CFRelease(pk.ioOut[b]);
            CFRelease(pk2.ioIn[b]); CFRelease(pk2.ioOut[b]);
        }

        printf("\n=== Done ===\n");
    }
    return 0;
}
