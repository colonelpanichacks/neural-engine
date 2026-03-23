// probe_pfe_bench.m — Benchmark doEvaluateDirectWithModel vs PFE processRequest
// Compares _ANEClient dispatch (production path) with _ANEProgramForEvaluation dispatch
// Uses a 1024x1024 matmul kernel, 1000 iterations each
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

// Generate MIL for dynamic matmul: y = x @ W
// Input: [1, IC, 1, SEQ+OC] (activation in sp[0:SEQ], weight in sp[SEQ:SEQ+OC])
// Output: [1, OC, 1, SEQ]
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

typedef struct {
    id model;       // _ANEInMemoryModel
    id aneModel;    // _ANEModel (from [model model])
    IOSurfaceRef ioIn, ioOut;
    id request;     // _ANERequest
} Kern;

static Kern compile_kernel(int ic, int oc, int seq) {
    Kern k = {0};
    Class D = NSClassFromString(@"_ANEInMemoryModelDescriptor");
    Class I = NSClassFromString(@"_ANEInMemoryModel");
    NSData *mil = gen_mil(ic, oc, seq);
    size_t inB = ic * (seq + oc) * 2;  // fp16
    size_t outB = oc * seq * 2;

    id desc = ((id(*)(Class,SEL,id,id,id))objc_msgSend)(
        D, @selector(modelWithMILText:weights:optionsPlist:), mil, @{}, nil);
    if (!desc) { printf("ERROR: modelWithMILText failed\n"); return k; }

    k.model = ((id(*)(Class,SEL,id))objc_msgSend)(
        I, @selector(inMemoryModelWithDescriptor:), desc);
    if (!k.model) { printf("ERROR: inMemoryModelWithDescriptor failed\n"); return k; }

    // Pre-populate temp dir
    id hx = ((id(*)(id,SEL))objc_msgSend)(k.model, @selector(hexStringIdentifier));
    NSString *td = [NSTemporaryDirectory() stringByAppendingPathComponent:hx];
    [[NSFileManager defaultManager] createDirectoryAtPath:[td stringByAppendingPathComponent:@"weights"]
        withIntermediateDirectories:YES attributes:nil error:nil];
    [mil writeToFile:[td stringByAppendingPathComponent:@"model.mil"] atomically:YES];

    // Compile + load
    NSError *e = nil;
    BOOL ok = ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(
        k.model, @selector(compileWithQoS:options:error:), 21, @{}, &e);
    if (!ok) { printf("ERROR: compile failed: %s\n", e ? [[e description] UTF8String] : "unknown"); return k; }

    e = nil;
    ok = ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(
        k.model, @selector(loadWithQoS:options:error:), 21, @{}, &e);
    if (!ok) { printf("ERROR: load failed: %s\n", e ? [[e description] UTF8String] : "unknown"); return k; }

    // Get underlying _ANEModel
    k.aneModel = ((id(*)(id,SEL))objc_msgSend)(k.model, @selector(model));
    if (!k.aneModel) { printf("ERROR: [model model] returned nil\n"); return k; }

    // Create IOSurfaces
    k.ioIn = make_surf(inB);
    k.ioOut = make_surf(outB);

    // Fill input with test data
    IOSurfaceLock(k.ioIn, 0, NULL);
    _Float16 *p = (_Float16*)IOSurfaceGetBaseAddress(k.ioIn);
    for (size_t i = 0; i < inB / 2; i++) p[i] = (_Float16)(0.01f * (i % 100));
    IOSurfaceUnlock(k.ioIn, 0, NULL);

    // Build request
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
        g_client = ((id(*)(Class,SEL))objc_msgSend)(
            NSClassFromString(@"_ANEClient"), @selector(sharedConnection));

        if (!g_client) { printf("ERROR: Failed to get _ANEClient\n"); return 1; }

        printf("=== ANE Dispatch Benchmark: doEvaluateDirect vs PFE processRequest ===\n");
        printf("Kernel: 1024x1024 matmul (SEQ=256)\n\n");

        // ============================================================
        // Step 1: Compile 1024x1024 matmul kernel
        // ============================================================
        int IC = 1024, OC = 1024, SEQ = 256;
        printf("Compiling %dx%d matmul (SEQ=%d)...\n", IC, OC, SEQ);
        Kern k = compile_kernel(IC, OC, SEQ);
        if (!k.aneModel) { printf("FATAL: Kernel compilation failed\n"); return 1; }
        printf("  Compiled and loaded OK\n");
        printf("  _ANEModel class: %s\n", class_getName([k.aneModel class]));

        // Verify baseline eval works
        {
            NSError *e = nil;
            BOOL ok = ((BOOL(*)(id,SEL,id,id,id,unsigned int,NSError**))objc_msgSend)(
                g_client, @selector(doEvaluateDirectWithModel:options:request:qos:error:),
                k.aneModel, @{}, k.request, (unsigned int)21, &e);
            _Float16 *out = (_Float16*)IOSurfaceGetBaseAddress(k.ioOut);
            printf("  Baseline eval: %s, output[0..3]: %.4f %.4f %.4f %.4f\n",
                   ok ? "OK" : "FAIL", (float)out[0], (float)out[1], (float)out[2], (float)out[3]);
            if (!ok) { printf("FATAL: Baseline eval failed\n"); return 1; }
        }

        // ============================================================
        // Step 2: Get _ANEProgramForEvaluation
        // ============================================================
        id pfe = nil;
        @try {
            pfe = ((id(*)(id,SEL))objc_msgSend)(k.aneModel, @selector(program));
        } @catch (NSException *ex) {
            printf("  [aneModel program] threw: %s\n", [[ex reason] UTF8String]);
        }
        if (!pfe) {
            printf("FATAL: Could not get _ANEProgramForEvaluation from [aneModel program]\n");
            return 1;
        }
        printf("  PFE class: %s\n", class_getName([pfe class]));

        // Get string_id for processRequest
        uint64_t stringId = 0;
        @try {
            stringId = ((uint64_t(*)(id,SEL))objc_msgSend)(k.aneModel, @selector(string_id));
        } @catch (NSException *ex) {
            printf("  string_id threw (using 0): %s\n", [[ex reason] UTF8String]);
        }
        printf("  string_id: %llu\n", stringId);

        // Verify PFE processRequest works
        {
            NSError *err = nil;
            unsigned int retVal = 0;
            BOOL ok = ((BOOL(*)(id,SEL,id,id,unsigned int,uint64_t,uint64_t,id,unsigned int*,NSError**))objc_msgSend)(
                pfe,
                @selector(processRequest:model:qos:qIndex:modelStringID:options:returnValue:error:),
                k.request, k.aneModel, (unsigned int)21, (uint64_t)0,
                stringId, @{}, &retVal, &err);
            printf("  PFE processRequest verify: %s (retVal=%u)\n", ok ? "OK" : "FAIL", retVal);
            if (!ok) {
                printf("FATAL: PFE processRequest failed: %s\n", err ? [[err description] UTF8String] : "unknown");
                return 1;
            }
        }

        int N = 1000;
        int WARMUP = 50;
        printf("\nBenchmark: %d iterations, %d warmup\n\n", N, WARMUP);

        // ============================================================
        // Benchmark 1: doEvaluateDirectWithModel via _ANEClient
        // ============================================================
        printf("--- Method 1: _ANEClient doEvaluateDirectWithModel ---\n");
        {
            // Warmup
            for (int i = 0; i < WARMUP; i++) {
                NSError *e = nil;
                ((BOOL(*)(id,SEL,id,id,id,unsigned int,NSError**))objc_msgSend)(
                    g_client, @selector(doEvaluateDirectWithModel:options:request:qos:error:),
                    k.aneModel, @{}, k.request, (unsigned int)21, &e);
            }

            // Timed run
            uint64_t t0 = mach_absolute_time();
            for (int i = 0; i < N; i++) {
                NSError *e = nil;
                ((BOOL(*)(id,SEL,id,id,id,unsigned int,NSError**))objc_msgSend)(
                    g_client, @selector(doEvaluateDirectWithModel:options:request:qos:error:),
                    k.aneModel, @{}, k.request, (unsigned int)21, &e);
            }
            uint64_t t1 = mach_absolute_time();
            double total = ms_t(t1 - t0);
            printf("  Total: %.3f ms\n", total);
            printf("  Per-eval: %.4f ms\n", total / N);
        }

        // ============================================================
        // Benchmark 2: PFE processRequest
        // ============================================================
        printf("\n--- Method 2: _ANEProgramForEvaluation processRequest ---\n");
        {
            // Warmup
            for (int i = 0; i < WARMUP; i++) {
                NSError *err = nil;
                unsigned int retVal = 0;
                ((BOOL(*)(id,SEL,id,id,unsigned int,uint64_t,uint64_t,id,unsigned int*,NSError**))objc_msgSend)(
                    pfe,
                    @selector(processRequest:model:qos:qIndex:modelStringID:options:returnValue:error:),
                    k.request, k.aneModel, (unsigned int)21, (uint64_t)0,
                    stringId, @{}, &retVal, &err);
            }

            // Timed run
            uint64_t t0 = mach_absolute_time();
            for (int i = 0; i < N; i++) {
                NSError *err = nil;
                unsigned int retVal = 0;
                ((BOOL(*)(id,SEL,id,id,unsigned int,uint64_t,uint64_t,id,unsigned int*,NSError**))objc_msgSend)(
                    pfe,
                    @selector(processRequest:model:qos:qIndex:modelStringID:options:returnValue:error:),
                    k.request, k.aneModel, (unsigned int)21, (uint64_t)0,
                    stringId, @{}, &retVal, &err);
            }
            uint64_t t1 = mach_absolute_time();
            double total = ms_t(t1 - t0);
            printf("  Total: %.3f ms\n", total);
            printf("  Per-eval: %.4f ms\n", total / N);
        }

        // ============================================================
        // Summary
        // ============================================================
        printf("\n=== Summary ===\n");
        {
            // Re-run both once more for a clean comparison
            // doEvaluateDirect
            for (int i = 0; i < WARMUP; i++) {
                NSError *e = nil;
                ((BOOL(*)(id,SEL,id,id,id,unsigned int,NSError**))objc_msgSend)(
                    g_client, @selector(doEvaluateDirectWithModel:options:request:qos:error:),
                    k.aneModel, @{}, k.request, (unsigned int)21, &e);
            }
            uint64_t t0 = mach_absolute_time();
            for (int i = 0; i < N; i++) {
                NSError *e = nil;
                ((BOOL(*)(id,SEL,id,id,id,unsigned int,NSError**))objc_msgSend)(
                    g_client, @selector(doEvaluateDirectWithModel:options:request:qos:error:),
                    k.aneModel, @{}, k.request, (unsigned int)21, &e);
            }
            uint64_t t1 = mach_absolute_time();
            double direct_ms = ms_t(t1 - t0);

            // PFE processRequest
            for (int i = 0; i < WARMUP; i++) {
                NSError *err = nil;
                unsigned int retVal = 0;
                ((BOOL(*)(id,SEL,id,id,unsigned int,uint64_t,uint64_t,id,unsigned int*,NSError**))objc_msgSend)(
                    pfe,
                    @selector(processRequest:model:qos:qIndex:modelStringID:options:returnValue:error:),
                    k.request, k.aneModel, (unsigned int)21, (uint64_t)0,
                    stringId, @{}, &retVal, &err);
            }
            uint64_t t2 = mach_absolute_time();
            for (int i = 0; i < N; i++) {
                NSError *err = nil;
                unsigned int retVal = 0;
                ((BOOL(*)(id,SEL,id,id,unsigned int,uint64_t,uint64_t,id,unsigned int*,NSError**))objc_msgSend)(
                    pfe,
                    @selector(processRequest:model:qos:qIndex:modelStringID:options:returnValue:error:),
                    k.request, k.aneModel, (unsigned int)21, (uint64_t)0,
                    stringId, @{}, &retVal, &err);
            }
            uint64_t t3 = mach_absolute_time();
            double pfe_ms = ms_t(t3 - t2);

            printf("  doEvaluateDirect: %.3f ms total, %.4f ms/eval\n", direct_ms, direct_ms / N);
            printf("  PFE processReq:   %.3f ms total, %.4f ms/eval\n", pfe_ms, pfe_ms / N);
            double diff_pct = (pfe_ms - direct_ms) / direct_ms * 100.0;
            printf("  Delta: %+.1f%% (PFE vs Direct)\n", diff_pct);
        }

        // Cleanup
        {
            NSError *e = nil;
            ((BOOL(*)(id,SEL,unsigned int,NSError**))objc_msgSend)(
                k.model, @selector(unloadWithQoS:error:), 21, &e);
        }
        CFRelease(k.ioIn);
        CFRelease(k.ioOut);

        printf("\n=== Done ===\n");
    }
    return 0;
}
