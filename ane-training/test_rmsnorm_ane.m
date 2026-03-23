// test_rmsnorm_ane.m — Test RMSNorm on ANE at DIM=1024 via channel decomposition
// Strategy: split 1024 channels into groups, reduce per-group, combine
#import <Foundation/Foundation.h>
#import <IOSurface/IOSurface.h>
#import <objc/runtime.h>
#import <objc/message.h>
#import <dlfcn.h>
#import <mach/mach_time.h>
#import <Accelerate/Accelerate.h>

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

// Try compiling a MIL program that does RMSNorm-like reduction at DIM=1024
// Approach 1: Direct reduce_mean (known to fail at 1024)
// Approach 2: Reshape [1,1024,1,S] -> [1,4,256,S], reduce inner dim, combine
// Approach 3: Use matmul-based reduction (x^2 @ ones_vector / dim)

static BOOL try_compile(NSData *mil, const char *name) {
    id desc = ((id(*)(Class,SEL,id,id,id))objc_msgSend)(g_D,
        @selector(modelWithMILText:weights:optionsPlist:), mil, @{}, nil);
    id model = ((id(*)(Class,SEL,id))objc_msgSend)(g_I,
        @selector(inMemoryModelWithDescriptor:), desc);
    id hx = ((id(*)(id,SEL))objc_msgSend)(model, @selector(hexStringIdentifier));
    NSString *td = [NSTemporaryDirectory() stringByAppendingPathComponent:hx];
    [[NSFileManager defaultManager] createDirectoryAtPath:[td stringByAppendingPathComponent:@"weights"]
        withIntermediateDirectories:YES attributes:nil error:nil];
    [mil writeToFile:[td stringByAppendingPathComponent:@"model.mil"] atomically:YES];

    NSError *e = nil;
    BOOL ok = ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(
        model, @selector(compileWithQoS:options:error:), 21, @{}, &e);
    if (!ok) {
        printf("  [%s] Compile FAILED: %s\n", name, e ? [[e description] UTF8String] : "unknown");
        return NO;
    }
    ok = ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(
        model, @selector(loadWithQoS:options:error:), 21, @{}, &e);
    if (!ok) {
        printf("  [%s] Load FAILED: %s\n", name, e ? [[e description] UTF8String] : "unknown");
        return NO;
    }
    printf("  [%s] Compile+Load OK!\n", name);

    // Quick eval test
    int DIM = 1024, SEQ = 256;
    size_t in_bytes = DIM * SEQ * 2;
    size_t out_bytes = DIM * SEQ * 2;
    IOSurfaceRef ioIn = make_surf(in_bytes);
    IOSurfaceRef ioOut = make_surf(out_bytes);

    _Float16 *inp = (_Float16*)IOSurfaceGetBaseAddress(ioIn);
    for (int i = 0; i < DIM*SEQ; i++) inp[i] = (_Float16)(0.01f * (i % 50 + 1));

    id wI = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_AIO, @selector(objectWithIOSurface:), ioIn);
    id wO = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_AIO, @selector(objectWithIOSurface:), ioOut);
    id req = ((id(*)(Class,SEL,id,id,id,id,id,id,id))objc_msgSend)(g_AR,
        @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:),
        @[wI], @[@0], @[wO], @[@0], nil, nil, @0);
    id aneModel = ((id(*)(id,SEL))objc_msgSend)(model, @selector(model));

    e = nil;
    ok = ((BOOL(*)(id,SEL,id,id,id,NSError**))objc_msgSend)(
        g_client, @selector(evaluateRealTimeWithModel:options:request:error:),
        aneModel, @{}, req, &e);
    if (!ok) {
        printf("  [%s] Eval FAILED: %s\n", name, e ? [[e description] UTF8String] : "unknown");
        CFRelease(ioIn); CFRelease(ioOut);
        return NO;
    }

    // Benchmark
    uint64_t t0 = mach_absolute_time();
    int N = 200;
    for (int i = 0; i < N; i++) {
        ((BOOL(*)(id,SEL,id,id,id,NSError**))objc_msgSend)(
            g_client, @selector(evaluateRealTimeWithModel:options:request:error:),
            aneModel, @{}, req, &e);
    }
    double total_ms = ms_t(mach_absolute_time() - t0);
    printf("  [%s] Eval: %.3f ms/eval (%d iters)\n", name, total_ms/N, N);

    // Verify output
    _Float16 *out = (_Float16*)IOSurfaceGetBaseAddress(ioOut);
    printf("  [%s] Output[0..3]: %f %f %f %f\n", name,
        (float)out[0], (float)out[1], (float)out[2], (float)out[3]);

    CFRelease(ioIn); CFRelease(ioOut);
    return YES;
}

int main(int argc, char *argv[]) {
    @autoreleasepool {
        setbuf(stdout, NULL);
        mach_timebase_info(&g_tb);
        ane_init();

        int DIM = 1024, SEQ = 256;

        printf("=== RMSNorm on ANE (DIM=%d, SEQ=%d) ===\n\n", DIM, SEQ);

        // Approach 1: Direct reduce_mean on 1024 channels
        // x^2, reduce_mean across channels, rsqrt, multiply
        {
            printf("--- Approach 1: Direct reduce_mean (DIM=%d) ---\n", DIM);
            NSMutableString *m = [NSMutableString string];
            [m appendString:@"program(1.3)\n"
                "[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}, "
                "{\"coremlc-version\", \"3505.4.1\"}, {\"coremltools-component-milinternal\", \"\"}, "
                "{\"coremltools-version\", \"9.0\"}})]\n{\n"];
            [m appendFormat:@"    func main<ios18>(tensor<fp16, [1, %d, 1, %d]> x) {\n", DIM, SEQ];
            // x^2
            [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> x2 = mul(x=x, y=x)[name=string(\"x2\")];\n", DIM, SEQ];
            // reduce_mean across channel dim
            [m appendString:@"        tensor<int32, [1]> ax = const()[name=string(\"ax\"), val=tensor<int32, [1]>([1])];\n"];
            [m appendString:@"        bool kd = const()[name=string(\"kd\"), val=bool(true)];\n"];
            [m appendFormat:@"        tensor<fp16, [1,1,1,%d]> ms = reduce_mean(x=x2, axes=ax, keep_dims=kd)[name=string(\"ms\")];\n", SEQ];
            // add eps
            [m appendFormat:@"        tensor<fp16, [1,1,1,%d]> eps = const()[name=string(\"eps\"), val=tensor<fp16, [1,1,1,%d]>(", SEQ, SEQ];
            for (int i = 0; i < SEQ; i++) [m appendFormat:@"%s1e-5", i>0?",":""];
            [m appendString:@")];\n"];
            [m appendFormat:@"        tensor<fp16, [1,1,1,%d]> mse = add(x=ms, y=eps)[name=string(\"mse\")];\n", SEQ];
            // rsqrt
            [m appendFormat:@"        tensor<fp16, [1,1,1,%d]> rrms = rsqrt(x=mse)[name=string(\"rrms\")];\n", SEQ];
            // scale
            [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> y = mul(x=x, y=rrms)[name=string(\"y\")];\n", DIM, SEQ];
            [m appendString:@"    } -> (y);\n}\n"];
            try_compile([m dataUsingEncoding:NSUTF8StringEncoding], "direct_reduce");
        }

        // Approach 2: Matmul-based reduction (x^2 matmul with ones vector / dim)
        // This uses the ANE's matmul cores for the reduction
        {
            printf("\n--- Approach 2: Matmul reduction (x^2 @ ones / DIM) ---\n");
            NSMutableString *m = [NSMutableString string];
            [m appendString:@"program(1.3)\n"
                "[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}, "
                "{\"coremlc-version\", \"3505.4.1\"}, {\"coremltools-component-milinternal\", \"\"}, "
                "{\"coremltools-version\", \"9.0\"}})]\n{\n"];
            // Input includes x and ones vector (packed in spatial)
            // Layout: [1, DIM, 1, SEQ] for x
            // We'll use a const ones vector for the reduction
            int sp = SEQ;
            [m appendFormat:@"    func main<ios18>(tensor<fp16, [1, %d, 1, %d]> x) {\n", DIM, sp];

            // x^2
            [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> x2 = mul(x=x, y=x)[name=string(\"x2\")];\n", DIM, SEQ];

            // Reshape x2 to [1,1,DIM,SEQ] for matmul
            [m appendFormat:@"        tensor<int32, [4]> rsh = const()[name=string(\"rsh\"), val=tensor<int32, [4]>([1,1,%d,%d])];\n", DIM, SEQ];
            [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> x2r = reshape(shape=rsh, x=x2)[name=string(\"x2r\")];\n", DIM, SEQ];

            // Transpose: [1,1,SEQ,DIM]
            [m appendString:@"        tensor<int32, [4]> pm = const()[name=string(\"pm\"), val=tensor<int32, [4]>([0,1,3,2])];\n"];
            [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> x2t = transpose(perm=pm, x=x2r)[name=string(\"x2t\")];\n", SEQ, DIM];

            // Ones vector [1,1,DIM,1] as const
            [m appendFormat:@"        tensor<fp16, [1,1,%d,1]> ones = const()[name=string(\"ones\"), val=tensor<fp16, [1,1,%d,1]>(", DIM, DIM];
            for (int i = 0; i < DIM; i++) [m appendFormat:@"%s1.0", i>0?",":""];
            [m appendString:@")];\n"];

            // matmul: [1,1,SEQ,DIM] @ [1,1,DIM,1] -> [1,1,SEQ,1]
            [m appendString:@"        bool bF = const()[name=string(\"bF\"), val=bool(false)];\n"];
            [m appendFormat:@"        tensor<fp16, [1,1,%d,1]> ss = matmul(transpose_x=bF, transpose_y=bF, x=x2t, y=ones)[name=string(\"ss\")];\n", SEQ];

            // Divide by DIM
            float inv_dim = 1.0f / DIM;
            [m appendFormat:@"        tensor<fp16, [1,1,%d,1]> sc = const()[name=string(\"sc\"), val=tensor<fp16, [1,1,%d,1]>(", SEQ, SEQ];
            for (int i = 0; i < SEQ; i++) [m appendFormat:@"%s%f", i>0?",":"", inv_dim];
            [m appendString:@")];\n"];
            [m appendFormat:@"        tensor<fp16, [1,1,%d,1]> ms = mul(x=ss, y=sc)[name=string(\"ms\")];\n", SEQ];

            // add eps
            [m appendFormat:@"        tensor<fp16, [1,1,%d,1]> eps = const()[name=string(\"eps\"), val=tensor<fp16, [1,1,%d,1]>(", SEQ, SEQ];
            for (int i = 0; i < SEQ; i++) [m appendFormat:@"%s1e-5", i>0?",":""];
            [m appendString:@")];\n"];
            [m appendFormat:@"        tensor<fp16, [1,1,%d,1]> mse = add(x=ms, y=eps)[name=string(\"mse\")];\n", SEQ];

            // rsqrt
            [m appendFormat:@"        tensor<fp16, [1,1,%d,1]> rrms = rsqrt(x=mse)[name=string(\"rrms\")];\n", SEQ];

            // Transpose back: [1,1,1,SEQ]
            [m appendFormat:@"        tensor<fp16, [1,1,1,%d]> rr = transpose(perm=pm, x=rrms)[name=string(\"rr\")];\n", SEQ];

            // Broadcast multiply with x: [1,DIM,1,SEQ] * [1,1,1,SEQ]
            [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> y = mul(x=x, y=rr)[name=string(\"y\")];\n", DIM, SEQ];

            [m appendString:@"    } -> (y);\n}\n"];
            try_compile([m dataUsingEncoding:NSUTF8StringEncoding], "matmul_reduce");
        }

        // Approach 3: Reshape to [1, 4, 256, SEQ], reduce per-group, combine
        {
            printf("\n--- Approach 3: Grouped reduction (4 groups of 256) ---\n");
            int GROUPS = 4, GSIZE = DIM / GROUPS;
            NSMutableString *m = [NSMutableString string];
            [m appendString:@"program(1.3)\n"
                "[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}, "
                "{\"coremlc-version\", \"3505.4.1\"}, {\"coremltools-component-milinternal\", \"\"}, "
                "{\"coremltools-version\", \"9.0\"}})]\n{\n"];
            [m appendFormat:@"    func main<ios18>(tensor<fp16, [1, %d, 1, %d]> x) {\n", DIM, SEQ];

            // x^2
            [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> x2 = mul(x=x, y=x)[name=string(\"x2\")];\n", DIM, SEQ];

            // Reshape to [1, GROUPS, GSIZE, SEQ]
            [m appendFormat:@"        tensor<int32, [4]> rsh = const()[name=string(\"rsh\"), val=tensor<int32, [4]>([1,%d,%d,%d])];\n", GROUPS, GSIZE, SEQ];
            [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> x2g = reshape(shape=rsh, x=x2)[name=string(\"x2g\")];\n", GROUPS, GSIZE, SEQ];

            // reduce_sum across height dim (axis 2): [1, GROUPS, 1, SEQ]
            [m appendString:@"        tensor<int32, [1]> ax2 = const()[name=string(\"ax2\"), val=tensor<int32, [1]>([2])];\n"];
            [m appendString:@"        bool kd = const()[name=string(\"kd\"), val=bool(true)];\n"];
            [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> ps = reduce_sum(x=x2g, axes=ax2, keep_dims=kd)[name=string(\"ps\")];\n", GROUPS, SEQ];

            // reduce_sum across channel dim (axis 1): [1, 1, 1, SEQ]
            [m appendString:@"        tensor<int32, [1]> ax1 = const()[name=string(\"ax1\"), val=tensor<int32, [1]>([1])];\n"];
            [m appendFormat:@"        tensor<fp16, [1,1,1,%d]> ts = reduce_sum(x=ps, axes=ax1, keep_dims=kd)[name=string(\"ts\")];\n", SEQ];

            // Divide by DIM, add eps, rsqrt
            float inv_dim = 1.0f / DIM;
            [m appendFormat:@"        tensor<fp16, [1,1,1,%d]> sc = const()[name=string(\"sc\"), val=tensor<fp16, [1,1,1,%d]>(", SEQ, SEQ];
            for (int i = 0; i < SEQ; i++) [m appendFormat:@"%s%f", i>0?",":"", inv_dim];
            [m appendString:@")];\n"];
            [m appendFormat:@"        tensor<fp16, [1,1,1,%d]> ms = mul(x=ts, y=sc)[name=string(\"ms\")];\n", SEQ];
            [m appendFormat:@"        tensor<fp16, [1,1,1,%d]> eps = const()[name=string(\"eps\"), val=tensor<fp16, [1,1,1,%d]>(", SEQ, SEQ];
            for (int i = 0; i < SEQ; i++) [m appendFormat:@"%s1e-5", i>0?",":""];
            [m appendString:@")];\n"];
            [m appendFormat:@"        tensor<fp16, [1,1,1,%d]> mse = add(x=ms, y=eps)[name=string(\"mse\")];\n", SEQ];
            [m appendFormat:@"        tensor<fp16, [1,1,1,%d]> rrms = rsqrt(x=mse)[name=string(\"rrms\")];\n", SEQ];

            // Broadcast multiply with x: [1,DIM,1,SEQ] * [1,1,1,SEQ]
            [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> y = mul(x=x, y=rrms)[name=string(\"y\")];\n", DIM, SEQ];

            [m appendString:@"    } -> (y);\n}\n"];
            try_compile([m dataUsingEncoding:NSUTF8StringEncoding], "grouped_reduce");
        }

        // Approach 4: Direct reduce on smaller dim to find the threshold
        for (int test_dim = 256; test_dim <= 2048; test_dim *= 2) {
            printf("\n--- Approach 4: Direct reduce_mean DIM=%d ---\n", test_dim);
            NSMutableString *m = [NSMutableString string];
            [m appendString:@"program(1.3)\n"
                "[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}, "
                "{\"coremlc-version\", \"3505.4.1\"}, {\"coremltools-component-milinternal\", \"\"}, "
                "{\"coremltools-version\", \"9.0\"}})]\n{\n"];
            [m appendFormat:@"    func main<ios18>(tensor<fp16, [1, %d, 1, %d]> x) {\n", test_dim, SEQ];
            [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> x2 = mul(x=x, y=x)[name=string(\"x2\")];\n", test_dim, SEQ];
            [m appendString:@"        tensor<int32, [1]> ax = const()[name=string(\"ax\"), val=tensor<int32, [1]>([1])];\n"];
            [m appendString:@"        bool kd = const()[name=string(\"kd\"), val=bool(true)];\n"];
            [m appendFormat:@"        tensor<fp16, [1,1,1,%d]> ms = reduce_mean(x=x2, axes=ax, keep_dims=kd)[name=string(\"ms\")];\n", SEQ];
            [m appendFormat:@"        tensor<fp16, [1,1,1,%d]> rrms = rsqrt(x=ms)[name=string(\"rrms\")];\n", SEQ];
            [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> y = mul(x=x, y=rrms)[name=string(\"y\")];\n", test_dim, SEQ];
            [m appendString:@"    } -> (y);\n}\n"];
            try_compile([m dataUsingEncoding:NSUTF8StringEncoding],
                [[NSString stringWithFormat:@"reduce_%d", test_dim] UTF8String]);
        }

        printf("\n=== Done ===\n");
    }
    return 0;
}
