// probe_seq512.m — Test ANE kernel compilation/eval at SEQ=512 (double current SEQ=256)
//
// Tests three kernel types:
//   1. Simple matmul: [1, 1024, 1, 512+1024] -> [1, 1024, 1, 512] (wotBwd-like)
//   2. SDPA-like: Q@K^T -> softmax -> probs@V with HEADS=16, HD=128, SEQ=512
//   3. Large channel: [1, 8192, 1, 512] input (sdpaBwd1-like)
//
// Build:
//   xcrun clang -O2 -framework Foundation -framework IOSurface -framework Accelerate \
//     -isysroot $(xcrun --show-sdk-path) -fobjc-arc -o probe_seq512 probe_seq512.m

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
    g_D   = NSClassFromString(@"_ANEInMemoryModelDescriptor");
    g_I   = NSClassFromString(@"_ANEInMemoryModel");
    g_AR  = NSClassFromString(@"_ANERequest");
    g_AIO = NSClassFromString(@"_ANEIOSurfaceObject");
    g_client = ((id(*)(Class,SEL))objc_msgSend)(NSClassFromString(@"_ANEClient"), @selector(sharedConnection));
}

static IOSurfaceRef make_surf(size_t bytes) {
    return IOSurfaceCreate((__bridge CFDictionaryRef)@{
        (id)kIOSurfaceWidth:@(bytes), (id)kIOSurfaceHeight:@1,
        (id)kIOSurfaceBytesPerElement:@1, (id)kIOSurfaceBytesPerRow:@(bytes),
        (id)kIOSurfaceAllocSize:@(bytes), (id)kIOSurfacePixelFormat:@0});
}

#pragma mark — MIL Generators

// Test 1: Simple matmul via dynamic weights (wotBwd-like)
// Input: [1, IC, 1, SEQ+OC] where first SEQ cols are activation, last OC cols are weight
// Output: [1, OC, 1, SEQ]
static NSData *gen_matmul_mil(int ic, int oc, int seq) {
    NSMutableString *m = [NSMutableString string];
    [m appendString:@"program(1.3)\n"
        "[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}, "
        "{\"coremlc-version\", \"3505.4.1\"}, {\"coremltools-component-milinternal\", \"\"}, "
        "{\"coremltools-version\", \"9.0\"}})]\n{\n"];
    int sp = seq + oc;
    [m appendFormat:@"    func main<ios18>(tensor<fp16, [1, %d, 1, %d]> x) {\n", ic, sp];
    // Slice activation [1, IC, 1, SEQ]
    [m appendString:@"        tensor<int32, [4]> ba = const()[name=string(\"ba\"), val=tensor<int32, [4]>([0,0,0,0])];\n"];
    [m appendFormat:@"        tensor<int32, [4]> sa = const()[name=string(\"sa\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n", ic, seq];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> act = slice_by_size(x=x,begin=ba,size=sa)[name=string(\"act\")];\n", ic, seq];
    // Slice weight [1, IC, 1, OC]
    [m appendFormat:@"        tensor<int32, [4]> bw = const()[name=string(\"bw\"), val=tensor<int32, [4]>([0,0,0,%d])];\n", seq];
    [m appendFormat:@"        tensor<int32, [4]> sw = const()[name=string(\"sw\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n", ic, oc];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> wt = slice_by_size(x=x,begin=bw,size=sw)[name=string(\"wt\")];\n", ic, oc];
    // Reshape for matmul: act -> [1,1,IC,SEQ] -> transpose -> [1,1,SEQ,IC]
    [m appendFormat:@"        tensor<int32, [4]> ra = const()[name=string(\"ra\"), val=tensor<int32, [4]>([1,1,%d,%d])];\n", ic, seq];
    [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> a2 = reshape(shape=ra,x=act)[name=string(\"a2\")];\n", ic, seq];
    [m appendString:@"        tensor<int32, [4]> pm = const()[name=string(\"pm\"), val=tensor<int32, [4]>([0,1,3,2])];\n"];
    [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> a3 = transpose(perm=pm,x=a2)[name=string(\"a3\")];\n", seq, ic];
    // Reshape weight: wt -> [1,1,IC,OC]
    [m appendFormat:@"        tensor<int32, [4]> rw = const()[name=string(\"rw\"), val=tensor<int32, [4]>([1,1,%d,%d])];\n", ic, oc];
    [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> W = reshape(shape=rw,x=wt)[name=string(\"W\")];\n", ic, oc];
    // Matmul: [1,1,SEQ,IC] @ [1,1,IC,OC] -> [1,1,SEQ,OC]
    [m appendString:@"        bool bF = const()[name=string(\"bF\"), val=bool(false)];\n"];
    [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> yh = matmul(transpose_x=bF,transpose_y=bF,x=a3,y=W)[name=string(\"yh\")];\n", seq, oc];
    // Transpose back and reshape to [1, OC, 1, SEQ]
    [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> yt = transpose(perm=pm,x=yh)[name=string(\"yt\")];\n", oc, seq];
    [m appendFormat:@"        tensor<int32, [4]> ro = const()[name=string(\"ro\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n", oc, seq];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> y = reshape(shape=ro,x=yt)[name=string(\"y\")];\n", oc, seq];
    [m appendString:@"    } -> (y);\n}\n"];
    return [m dataUsingEncoding:NSUTF8StringEncoding];
}

// Test 2: SDPA-like kernel
// Q@K^T -> softmax -> probs@V
// Input packs Q, K, V into channel dim: [1, HEADS*(HD+HD+HD), 1, SEQ] = [1, HEADS*3*HD, 1, SEQ]
// For HEADS=16, HD=128: [1, 6144, 1, 512]
// Output: [1, HEADS*HD, 1, SEQ] = [1, 2048, 1, 512]
static NSData *gen_sdpa_mil(int heads, int hd, int seq) {
    NSMutableString *m = [NSMutableString string];
    int qkv_ch = heads * 3 * hd;  // input channels: Q+K+V packed
    int out_ch = heads * hd;       // output channels
    [m appendString:@"program(1.3)\n"
        "[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}, "
        "{\"coremlc-version\", \"3505.4.1\"}, {\"coremltools-component-milinternal\", \"\"}, "
        "{\"coremltools-version\", \"9.0\"}})]\n{\n"];
    [m appendFormat:@"    func main<ios18>(tensor<fp16, [1, %d, 1, %d]> x) {\n", qkv_ch, seq];

    // Slice Q: channels [0, HEADS*HD), shape [1, HEADS*HD, 1, SEQ]
    [m appendString:@"        tensor<int32, [4]> bq = const()[name=string(\"bq\"), val=tensor<int32, [4]>([0,0,0,0])];\n"];
    [m appendFormat:@"        tensor<int32, [4]> sq = const()[name=string(\"sq\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n", out_ch, seq];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> Q_flat = slice_by_size(x=x,begin=bq,size=sq)[name=string(\"Q_flat\")];\n", out_ch, seq];

    // Slice K: channels [HEADS*HD, 2*HEADS*HD)
    [m appendFormat:@"        tensor<int32, [4]> bk = const()[name=string(\"bk\"), val=tensor<int32, [4]>([0,%d,0,0])];\n", out_ch];
    [m appendFormat:@"        tensor<int32, [4]> sk = const()[name=string(\"sk\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n", out_ch, seq];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> K_flat = slice_by_size(x=x,begin=bk,size=sk)[name=string(\"K_flat\")];\n", out_ch, seq];

    // Slice V: channels [2*HEADS*HD, 3*HEADS*HD)
    [m appendFormat:@"        tensor<int32, [4]> bv = const()[name=string(\"bv\"), val=tensor<int32, [4]>([0,%d,0,0])];\n", 2*out_ch];
    [m appendFormat:@"        tensor<int32, [4]> sv = const()[name=string(\"sv\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n", out_ch, seq];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> V_flat = slice_by_size(x=x,begin=bv,size=sv)[name=string(\"V_flat\")];\n", out_ch, seq];

    // Reshape Q -> [HEADS, 1, SEQ, HD] for batched matmul
    [m appendFormat:@"        tensor<int32, [4]> rq = const()[name=string(\"rq\"), val=tensor<int32, [4]>([%d,1,%d,%d])];\n", heads, hd, seq];
    [m appendFormat:@"        tensor<fp16, [%d,1,%d,%d]> Q3 = reshape(shape=rq,x=Q_flat)[name=string(\"Q3\")];\n", heads, hd, seq];
    [m appendString:@"        tensor<int32, [4]> pm = const()[name=string(\"pm\"), val=tensor<int32, [4]>([0,1,3,2])];\n"];
    [m appendFormat:@"        tensor<fp16, [%d,1,%d,%d]> Q = transpose(perm=pm,x=Q3)[name=string(\"Q\")];\n", heads, seq, hd];

    // Reshape K -> [HEADS, 1, HD, SEQ] (transposed for Q@K^T)
    [m appendFormat:@"        tensor<int32, [4]> rk = const()[name=string(\"rk\"), val=tensor<int32, [4]>([%d,1,%d,%d])];\n", heads, hd, seq];
    [m appendFormat:@"        tensor<fp16, [%d,1,%d,%d]> K = reshape(shape=rk,x=K_flat)[name=string(\"K\")];\n", heads, hd, seq];
    // K stays as [HEADS, 1, HD, SEQ] — this is K^T already

    // Reshape V -> [HEADS, 1, SEQ, HD]
    [m appendFormat:@"        tensor<int32, [4]> rv = const()[name=string(\"rv\"), val=tensor<int32, [4]>([%d,1,%d,%d])];\n", heads, hd, seq];
    [m appendFormat:@"        tensor<fp16, [%d,1,%d,%d]> V3 = reshape(shape=rv,x=V_flat)[name=string(\"V3\")];\n", heads, hd, seq];
    [m appendFormat:@"        tensor<fp16, [%d,1,%d,%d]> V = transpose(perm=pm,x=V3)[name=string(\"V\")];\n", heads, seq, hd];

    // Q @ K^T: [HEADS, 1, SEQ, HD] @ [HEADS, 1, HD, SEQ] -> [HEADS, 1, SEQ, SEQ]
    [m appendString:@"        bool bF = const()[name=string(\"bF\"), val=bool(false)];\n"];
    [m appendFormat:@"        tensor<fp16, [%d,1,%d,%d]> scores = matmul(transpose_x=bF,transpose_y=bF,x=Q,y=K)[name=string(\"scores\")];\n", heads, seq, seq];

    // Scale by 1/sqrt(HD) — use mul by const
    [m appendFormat:@"        fp16 sc = const()[name=string(\"sc\"), val=fp16(%.8f)];\n", 1.0 / sqrt((double)hd)];
    [m appendFormat:@"        tensor<fp16, [%d,1,%d,%d]> scaled = mul(x=scores, y=sc)[name=string(\"scaled\")];\n", heads, seq, seq];

    // Softmax along last axis (key dim)
    [m appendString:@"        tensor<int32, [1]> ax = const()[name=string(\"ax\"), val=tensor<int32, [1]>([-1])];\n"];
    [m appendFormat:@"        tensor<fp16, [%d,1,%d,%d]> probs = softmax(x=scaled, axis=ax)[name=string(\"probs\")];\n", heads, seq, seq];

    // probs @ V: [HEADS, 1, SEQ, SEQ] @ [HEADS, 1, SEQ, HD] -> [HEADS, 1, SEQ, HD]
    [m appendFormat:@"        tensor<fp16, [%d,1,%d,%d]> attn = matmul(transpose_x=bF,transpose_y=bF,x=probs,y=V)[name=string(\"attn\")];\n", heads, seq, hd];

    // Transpose back and reshape to [1, HEADS*HD, 1, SEQ]
    [m appendFormat:@"        tensor<fp16, [%d,1,%d,%d]> att = transpose(perm=pm,x=attn)[name=string(\"att\")];\n", heads, hd, seq];
    [m appendFormat:@"        tensor<int32, [4]> ro = const()[name=string(\"ro\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n", out_ch, seq];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> y = reshape(shape=ro,x=att)[name=string(\"y\")];\n", out_ch, seq];
    [m appendString:@"    } -> (y);\n}\n"];
    return [m dataUsingEncoding:NSUTF8StringEncoding];
}

// Test 3: Large channel matmul (sdpaBwd1-like)
// Input: [1, CH, 1, SEQ+OC] -> matmul -> [1, OC, 1, SEQ]
static NSData *gen_large_ch_mil(int ic, int oc, int seq) {
    // Reuse the same matmul pattern but with large channel count
    return gen_matmul_mil(ic, oc, seq);
}

#pragma mark — Test Runner

typedef struct {
    const char *name;
    NSData *mil;
    size_t in_bytes;
    size_t out_bytes;
    double gflops;
} TestSpec;

static void run_test(TestSpec *spec) {
    printf("\n--- %s ---\n", spec->name);

    @autoreleasepool {
        NSError *e = nil;
        NSData *mil = [spec->mil copy];

        // Create descriptor
        id desc = ((id(*)(Class,SEL,id,id,id))objc_msgSend)(g_D,
            @selector(modelWithMILText:weights:optionsPlist:), mil, @{}, nil);
        if (!desc) { printf("  Descriptor:  FAIL (nil)\n"); return; }
        printf("  Descriptor:  OK\n");

        // Create model
        id model = ((id(*)(Class,SEL,id))objc_msgSend)(g_I,
            @selector(inMemoryModelWithDescriptor:), desc);
        if (!model) { printf("  Model:       FAIL (nil)\n"); return; }
        printf("  Model:       OK\n");

        // Write MIL to temp dir (required for compile)
        id hexId = ((id(*)(id,SEL))objc_msgSend)(model, @selector(hexStringIdentifier));
        NSString *tmpDir = [NSTemporaryDirectory() stringByAppendingPathComponent:hexId];
        NSFileManager *fm = [NSFileManager defaultManager];
        [fm createDirectoryAtPath:[tmpDir stringByAppendingPathComponent:@"weights"]
            withIntermediateDirectories:YES attributes:nil error:nil];
        [mil writeToFile:[tmpDir stringByAppendingPathComponent:@"model.mil"] atomically:YES];

        // Compile
        BOOL ok = ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(
            model, @selector(compileWithQoS:options:error:), 21, @{}, &e);
        printf("  Compile:     %s", ok ? "OK" : "FAIL");
        if (!ok && e) printf(" (%s)", [[e localizedDescription] UTF8String]);
        printf("\n");
        if (!ok) { [fm removeItemAtPath:tmpDir error:nil]; return; }

        // Load
        e = nil;
        ok = ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(
            model, @selector(loadWithQoS:options:error:), 21, @{}, &e);
        printf("  Load:        %s", ok ? "OK" : "FAIL");
        if (!ok && e) printf(" (%s)", [[e localizedDescription] UTF8String]);
        printf("\n");
        if (!ok) { [fm removeItemAtPath:tmpDir error:nil]; return; }

        // Create IOSurfaces
        IOSurfaceRef ioIn  = make_surf(spec->in_bytes);
        IOSurfaceRef ioOut = make_surf(spec->out_bytes);
        if (!ioIn || !ioOut) {
            printf("  IOSurface:   FAIL\n");
            ((BOOL(*)(id,SEL,unsigned int,NSError**))objc_msgSend)(model, @selector(unloadWithQoS:error:), 21, &e);
            if (ioIn) CFRelease(ioIn);
            if (ioOut) CFRelease(ioOut);
            [fm removeItemAtPath:tmpDir error:nil];
            return;
        }

        // Fill input with small values
        IOSurfaceLock(ioIn, 0, NULL);
        _Float16 *inp = (void*)IOSurfaceGetBaseAddress(ioIn);
        for (size_t i = 0; i < spec->in_bytes / 2; i++)
            inp[i] = (_Float16)(0.01f * (i % 100));
        IOSurfaceUnlock(ioIn, 0, NULL);

        id wIn  = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_AIO, @selector(objectWithIOSurface:), ioIn);
        id wOut = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_AIO, @selector(objectWithIOSurface:), ioOut);
        id req = ((id(*)(Class,SEL,id,id,id,id,id,id,id))objc_msgSend)(g_AR,
            @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:),
            @[wIn], @[@0], @[wOut], @[@0], nil, nil, @0);

        // Evaluate (first test if it works)
        e = nil;
        id aneModel = ((id(*)(id,SEL))objc_msgSend)(model, @selector(model));
        ok = ((BOOL(*)(id,SEL,id,id,id,unsigned int,NSError**))objc_msgSend)(
            g_client, @selector(doEvaluateDirectWithModel:options:request:qos:error:),
            aneModel, @{}, req, 21, &e);
        printf("  Evaluate:    %s", ok ? "OK" : "FAIL");
        if (!ok && e) printf(" (%s)", [[e localizedDescription] UTF8String]);
        printf("\n");
        if (!ok) {
            ((BOOL(*)(id,SEL,unsigned int,NSError**))objc_msgSend)(model, @selector(unloadWithQoS:error:), 21, &e);
            CFRelease(ioIn); CFRelease(ioOut);
            [fm removeItemAtPath:tmpDir error:nil];
            return;
        }

        // Benchmark: warmup
        for (int i = 0; i < 10; i++) {
            ((BOOL(*)(id,SEL,id,id,id,unsigned int,NSError**))objc_msgSend)(
                g_client, @selector(doEvaluateDirectWithModel:options:request:qos:error:),
                aneModel, @{}, req, 21, &e);
        }

        // Benchmark: timed
        int iters = 100;
        uint64_t t0 = mach_absolute_time();
        for (int i = 0; i < iters; i++) {
            ((BOOL(*)(id,SEL,id,id,id,unsigned int,NSError**))objc_msgSend)(
                g_client, @selector(doEvaluateDirectWithModel:options:request:qos:error:),
                aneModel, @{}, req, 21, &e);
        }
        double total_ms = ms_t(mach_absolute_time() - t0);
        double per_eval = total_ms / iters;
        printf("  Benchmark:   %.3f ms/eval (100 evals, total %.1f ms)\n", per_eval, total_ms);
        printf("  Input:       %.2f MB   Output: %.2f MB\n",
               spec->in_bytes / 1048576.0, spec->out_bytes / 1048576.0);
        if (spec->gflops > 0) {
            double tflops = spec->gflops / per_eval;
            printf("  Throughput:  %.2f GFLOP -> %.2f TFLOPS\n", spec->gflops, tflops);
        }

        ((BOOL(*)(id,SEL,unsigned int,NSError**))objc_msgSend)(model, @selector(unloadWithQoS:error:), 21, &e);
        CFRelease(ioIn); CFRelease(ioOut);
        [fm removeItemAtPath:tmpDir error:nil];
    }
}

int main(int argc, char *argv[]) {
    @autoreleasepool {
        setbuf(stdout, NULL);
        mach_timebase_info(&g_tb);
        ane_init();

        printf("=== ANE SEQ=512 Probe ===\n");
        printf("Testing whether ANE can compile/load/eval kernels at SEQ=512\n");
        printf("(current training uses SEQ=256)\n");

        // Test 1: Simple matmul (wotBwd-like)
        // wotBwd at SEQ=256: [1, DIM, 1, SEQ+Q_DIM] = [1, 1024, 1, 256+2048]
        // At SEQ=512: [1, 1024, 1, 512+1024] -> [1, 1024, 1, 512]
        {
            int ic = 1024, oc = 1024, seq = 512;
            NSData *mil = gen_matmul_mil(ic, oc, seq);
            size_t in_bytes  = (size_t)ic * (seq + oc) * 2;
            size_t out_bytes = (size_t)oc * seq * 2;
            double gf = 2.0 * ic * oc * seq / 1e9;
            TestSpec spec = {
                .name = "Test 1: Simple matmul 1024x1024 @ SEQ=512 (wotBwd-like)",
                .mil = mil, .in_bytes = in_bytes, .out_bytes = out_bytes, .gflops = gf
            };
            run_test(&spec);
        }

        // Test 2: SDPA-like kernel
        // HEADS=16, HD=128, SEQ=512
        // Q@K^T: [16, SEQ, HD] @ [16, HD, SEQ] -> [16, SEQ, SEQ]
        // probs@V: [16, SEQ, SEQ] @ [16, SEQ, HD] -> [16, SEQ, HD]
        // SCORE_CH = 16 * 512 = 8192
        // Input: [1, 16*3*128, 1, 512] = [1, 6144, 1, 512]
        // Output: [1, 16*128, 1, 512] = [1, 2048, 1, 512]
        {
            int heads = 16, hd = 128, seq = 512;
            NSData *mil = gen_sdpa_mil(heads, hd, seq);
            int qkv_ch = heads * 3 * hd;
            int out_ch = heads * hd;
            size_t in_bytes  = (size_t)qkv_ch * seq * 2;
            size_t out_bytes = (size_t)out_ch * seq * 2;
            // Q@K^T: 16 * 2 * 512 * 128 * 512 + probs@V: 16 * 2 * 512 * 512 * 128
            double gf = 2.0 * (double)heads * seq * hd * seq / 1e9   // Q@K^T
                       + 2.0 * (double)heads * seq * seq * hd / 1e9;  // probs@V
            TestSpec spec = {
                .name = "Test 2: SDPA HEADS=16 HD=128 SEQ=512 (Q@K^T + softmax + probs@V)",
                .mil = mil, .in_bytes = in_bytes, .out_bytes = out_bytes, .gflops = gf
            };
            run_test(&spec);
        }

        // Test 3: Large channel matmul (sdpaBwd1-like)
        // At SEQ=512, sdpaBwd1 has 4*Q_DIM = 4*2048 = 8192 input channels
        // [1, 8192, 1, 512+512] -> [1, 512, 1, 512]
        // (simplified: large IC matmul)
        {
            int ic = 8192, oc = 512, seq = 512;
            NSData *mil = gen_matmul_mil(ic, oc, seq);
            size_t in_bytes  = (size_t)ic * (seq + oc) * 2;
            size_t out_bytes = (size_t)oc * seq * 2;
            double gf = 2.0 * ic * oc * seq / 1e9;
            TestSpec spec = {
                .name = "Test 3: Large channel 8192->512 @ SEQ=512 (sdpaBwd1-like)",
                .mil = mil, .in_bytes = in_bytes, .out_bytes = out_bytes, .gflops = gf
            };
            run_test(&spec);
        }

        // Test 2b: Simpler SDPA — just Q@K^T matmul (no softmax, no V)
        // To isolate whether the issue is the DAG complexity or the tensor sizes
        printf("\n--- SDPA Decomposition Tests ---\n");

        // Test 2b: Q@K^T only — [HEADS, 1, SEQ, HD] @ [HEADS, 1, HD, SEQ] -> [HEADS, 1, SEQ, SEQ]
        {
            int heads = 16, hd = 128, seq = 512;
            int in_ch = heads * 2 * hd;  // Q+K packed
            int out_ch = heads * seq;     // score matrix: HEADS*SEQ = SCORE_CH
            NSMutableString *m = [NSMutableString string];
            [m appendString:@"program(1.3)\n"
                "[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}, "
                "{\"coremlc-version\", \"3505.4.1\"}, {\"coremltools-component-milinternal\", \"\"}, "
                "{\"coremltools-version\", \"9.0\"}})]\n{\n"];
            [m appendFormat:@"    func main<ios18>(tensor<fp16, [1, %d, 1, %d]> x) {\n", in_ch, seq];
            // Slice Q
            [m appendString:@"        tensor<int32, [4]> bq = const()[name=string(\"bq\"), val=tensor<int32, [4]>([0,0,0,0])];\n"];
            [m appendFormat:@"        tensor<int32, [4]> sq = const()[name=string(\"sq\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n", heads*hd, seq];
            [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> Q_flat = slice_by_size(x=x,begin=bq,size=sq)[name=string(\"Q_flat\")];\n", heads*hd, seq];
            // Slice K
            [m appendFormat:@"        tensor<int32, [4]> bk = const()[name=string(\"bk\"), val=tensor<int32, [4]>([0,%d,0,0])];\n", heads*hd];
            [m appendFormat:@"        tensor<int32, [4]> sk = const()[name=string(\"sk\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n", heads*hd, seq];
            [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> K_flat = slice_by_size(x=x,begin=bk,size=sk)[name=string(\"K_flat\")];\n", heads*hd, seq];
            // Reshape Q -> [HEADS, 1, HD, SEQ] -> transpose -> [HEADS, 1, SEQ, HD]
            [m appendFormat:@"        tensor<int32, [4]> rq = const()[name=string(\"rq\"), val=tensor<int32, [4]>([%d,1,%d,%d])];\n", heads, hd, seq];
            [m appendFormat:@"        tensor<fp16, [%d,1,%d,%d]> Q3 = reshape(shape=rq,x=Q_flat)[name=string(\"Q3\")];\n", heads, hd, seq];
            [m appendString:@"        tensor<int32, [4]> pm = const()[name=string(\"pm\"), val=tensor<int32, [4]>([0,1,3,2])];\n"];
            [m appendFormat:@"        tensor<fp16, [%d,1,%d,%d]> Q = transpose(perm=pm,x=Q3)[name=string(\"Q\")];\n", heads, seq, hd];
            // Reshape K -> [HEADS, 1, HD, SEQ] (already K^T)
            [m appendFormat:@"        tensor<int32, [4]> rk = const()[name=string(\"rk\"), val=tensor<int32, [4]>([%d,1,%d,%d])];\n", heads, hd, seq];
            [m appendFormat:@"        tensor<fp16, [%d,1,%d,%d]> K = reshape(shape=rk,x=K_flat)[name=string(\"K\")];\n", heads, hd, seq];
            // Matmul Q@K^T
            [m appendString:@"        bool bF = const()[name=string(\"bF\"), val=bool(false)];\n"];
            [m appendFormat:@"        tensor<fp16, [%d,1,%d,%d]> scores = matmul(transpose_x=bF,transpose_y=bF,x=Q,y=K)[name=string(\"scores\")];\n", heads, seq, seq];
            // Reshape to [1, HEADS*SEQ, 1, SEQ]
            [m appendFormat:@"        tensor<int32, [4]> ro = const()[name=string(\"ro\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n", heads*seq, seq];
            [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> y = reshape(shape=ro,x=scores)[name=string(\"y\")];\n", heads*seq, seq];
            [m appendString:@"    } -> (y);\n}\n"];
            NSData *mil = [m dataUsingEncoding:NSUTF8StringEncoding];
            size_t in_bytes  = (size_t)in_ch * seq * 2;
            size_t out_bytes = (size_t)out_ch * seq * 2;
            double gf = 2.0 * heads * seq * (double)hd * seq / 1e9;
            TestSpec spec = {
                .name = "Test 2b: Q@K^T only, HEADS=16 HD=128 SEQ=512",
                .mil = mil, .in_bytes = in_bytes, .out_bytes = out_bytes, .gflops = gf
            };
            run_test(&spec);
        }

        // Test 2c: Same Q@K^T at smaller scale — HEADS=4, HD=64, SEQ=512
        {
            int heads = 4, hd = 64, seq = 512;
            int in_ch = heads * 2 * hd;
            int out_ch = heads * seq;
            NSMutableString *m = [NSMutableString string];
            [m appendString:@"program(1.3)\n"
                "[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}, "
                "{\"coremlc-version\", \"3505.4.1\"}, {\"coremltools-component-milinternal\", \"\"}, "
                "{\"coremltools-version\", \"9.0\"}})]\n{\n"];
            [m appendFormat:@"    func main<ios18>(tensor<fp16, [1, %d, 1, %d]> x) {\n", in_ch, seq];
            [m appendString:@"        tensor<int32, [4]> bq = const()[name=string(\"bq\"), val=tensor<int32, [4]>([0,0,0,0])];\n"];
            [m appendFormat:@"        tensor<int32, [4]> sq = const()[name=string(\"sq\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n", heads*hd, seq];
            [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> Q_flat = slice_by_size(x=x,begin=bq,size=sq)[name=string(\"Q_flat\")];\n", heads*hd, seq];
            [m appendFormat:@"        tensor<int32, [4]> bk = const()[name=string(\"bk\"), val=tensor<int32, [4]>([0,%d,0,0])];\n", heads*hd];
            [m appendFormat:@"        tensor<int32, [4]> sk = const()[name=string(\"sk\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n", heads*hd, seq];
            [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> K_flat = slice_by_size(x=x,begin=bk,size=sk)[name=string(\"K_flat\")];\n", heads*hd, seq];
            [m appendFormat:@"        tensor<int32, [4]> rq = const()[name=string(\"rq\"), val=tensor<int32, [4]>([%d,1,%d,%d])];\n", heads, hd, seq];
            [m appendFormat:@"        tensor<fp16, [%d,1,%d,%d]> Q3 = reshape(shape=rq,x=Q_flat)[name=string(\"Q3\")];\n", heads, hd, seq];
            [m appendString:@"        tensor<int32, [4]> pm = const()[name=string(\"pm\"), val=tensor<int32, [4]>([0,1,3,2])];\n"];
            [m appendFormat:@"        tensor<fp16, [%d,1,%d,%d]> Q = transpose(perm=pm,x=Q3)[name=string(\"Q\")];\n", heads, seq, hd];
            [m appendFormat:@"        tensor<int32, [4]> rk = const()[name=string(\"rk\"), val=tensor<int32, [4]>([%d,1,%d,%d])];\n", heads, hd, seq];
            [m appendFormat:@"        tensor<fp16, [%d,1,%d,%d]> K = reshape(shape=rk,x=K_flat)[name=string(\"K\")];\n", heads, hd, seq];
            [m appendString:@"        bool bF = const()[name=string(\"bF\"), val=bool(false)];\n"];
            [m appendFormat:@"        tensor<fp16, [%d,1,%d,%d]> scores = matmul(transpose_x=bF,transpose_y=bF,x=Q,y=K)[name=string(\"scores\")];\n", heads, seq, seq];
            [m appendFormat:@"        tensor<int32, [4]> ro = const()[name=string(\"ro\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n", heads*seq, seq];
            [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> y = reshape(shape=ro,x=scores)[name=string(\"y\")];\n", heads*seq, seq];
            [m appendString:@"    } -> (y);\n}\n"];
            NSData *mil = [m dataUsingEncoding:NSUTF8StringEncoding];
            size_t in_bytes  = (size_t)in_ch * seq * 2;
            size_t out_bytes = (size_t)out_ch * seq * 2;
            double gf = 2.0 * heads * seq * (double)hd * seq / 1e9;
            TestSpec spec = {
                .name = "Test 2c: Q@K^T only, HEADS=4 HD=64 SEQ=512",
                .mil = mil, .in_bytes = in_bytes, .out_bytes = out_bytes, .gflops = gf
            };
            run_test(&spec);
        }

        // Test 2d: Full SDPA at smaller scale — HEADS=4, HD=64, SEQ=512
        {
            int heads = 4, hd = 64, seq = 512;
            NSData *mil = gen_sdpa_mil(heads, hd, seq);
            int qkv_ch = heads * 3 * hd;
            int out_ch = heads * hd;
            size_t in_bytes  = (size_t)qkv_ch * seq * 2;
            size_t out_bytes = (size_t)out_ch * seq * 2;
            double gf = 2.0 * (double)heads * seq * hd * seq / 1e9
                       + 2.0 * (double)heads * seq * seq * hd / 1e9;
            TestSpec spec = {
                .name = "Test 2d: Full SDPA HEADS=4 HD=64 SEQ=512",
                .mil = mil, .in_bytes = in_bytes, .out_bytes = out_bytes, .gflops = gf
            };
            run_test(&spec);
        }

        // Bonus: Compare with SEQ=256 baselines
        printf("\n--- SEQ=256 Baselines for comparison ---\n");

        // Baseline 1: wotBwd at SEQ=256
        {
            int ic = 1024, oc = 1024, seq = 256;
            NSData *mil = gen_matmul_mil(ic, oc, seq);
            size_t in_bytes  = (size_t)ic * (seq + oc) * 2;
            size_t out_bytes = (size_t)oc * seq * 2;
            double gf = 2.0 * ic * oc * seq / 1e9;
            TestSpec spec = {
                .name = "Baseline: matmul 1024x1024 @ SEQ=256",
                .mil = mil, .in_bytes = in_bytes, .out_bytes = out_bytes, .gflops = gf
            };
            run_test(&spec);
        }

        // Baseline 2: SDPA at SEQ=256
        {
            int heads = 16, hd = 128, seq = 256;
            NSData *mil = gen_sdpa_mil(heads, hd, seq);
            int qkv_ch = heads * 3 * hd;
            int out_ch = heads * hd;
            size_t in_bytes  = (size_t)qkv_ch * seq * 2;
            size_t out_bytes = (size_t)out_ch * seq * 2;
            double gf = 2.0 * (double)heads * seq * hd * seq / 1e9
                       + 2.0 * (double)heads * seq * seq * hd / 1e9;
            TestSpec spec = {
                .name = "Baseline: SDPA HEADS=16 HD=128 SEQ=256",
                .mil = mil, .in_bytes = in_bytes, .out_bytes = out_bytes, .gflops = gf
            };
            run_test(&spec);
        }

        printf("\n=== Done ===\n");
    }
    return 0;
}
