// test_rmsnorm_ane2.m — Test RMSNorm on ANE using reduce_sum on spatial axis
// Strategy: transpose DIM to spatial, reduce_sum, rsqrt, broadcast back
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

typedef struct {
    id model, aneModel;
    IOSurfaceRef ioIn, ioOut;
    id request;
} Kern;

static Kern try_compile_eval(NSData *mil, size_t in_bytes, size_t out_bytes, const char *name) {
    Kern k = {0};
    id desc = ((id(*)(Class,SEL,id,id,id))objc_msgSend)(g_D,
        @selector(modelWithMILText:weights:optionsPlist:), mil, @{}, nil);
    k.model = ((id(*)(Class,SEL,id))objc_msgSend)(g_I,
        @selector(inMemoryModelWithDescriptor:), desc);
    id hx = ((id(*)(id,SEL))objc_msgSend)(k.model, @selector(hexStringIdentifier));
    NSString *td = [NSTemporaryDirectory() stringByAppendingPathComponent:hx];
    [[NSFileManager defaultManager] createDirectoryAtPath:[td stringByAppendingPathComponent:@"weights"]
        withIntermediateDirectories:YES attributes:nil error:nil];
    [mil writeToFile:[td stringByAppendingPathComponent:@"model.mil"] atomically:YES];

    NSError *e = nil;
    BOOL ok = ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(
        k.model, @selector(compileWithQoS:options:error:), 21, @{}, &e);
    if (!ok) {
        printf("  [%s] Compile FAILED: %s\n", name,
            e ? [[[e userInfo][@"NSUnderlyingError"] description] UTF8String] : "unknown");
        return k;
    }
    ok = ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(
        k.model, @selector(loadWithQoS:options:error:), 21, @{}, &e);
    if (!ok) {
        printf("  [%s] Load FAILED (status likely 0x1d): %s\n", name,
            e ? [[e description] UTF8String] : "unknown");
        return k;
    }

    k.ioIn = make_surf(in_bytes);
    k.ioOut = make_surf(out_bytes);
    _Float16 *inp = (_Float16*)IOSurfaceGetBaseAddress(k.ioIn);
    for (size_t i = 0; i < in_bytes/2; i++) inp[i] = (_Float16)(0.01f * ((i % 50) + 1));

    id wI = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_AIO, @selector(objectWithIOSurface:), k.ioIn);
    id wO = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_AIO, @selector(objectWithIOSurface:), k.ioOut);
    k.request = ((id(*)(Class,SEL,id,id,id,id,id,id,id))objc_msgSend)(g_AR,
        @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:),
        @[wI], @[@0], @[wO], @[@0], nil, nil, @0);
    k.aneModel = ((id(*)(id,SEL))objc_msgSend)(k.model, @selector(model));

    ok = ((BOOL(*)(id,SEL,id,id,id,NSError**))objc_msgSend)(
        g_client, @selector(evaluateRealTimeWithModel:options:request:error:),
        k.aneModel, @{}, k.request, &e);
    if (!ok) {
        printf("  [%s] Eval FAILED: %s\n", name, e ? [[e description] UTF8String] : "unknown");
        return k;
    }

    // Benchmark
    int N = 200;
    uint64_t t0 = mach_absolute_time();
    for (int i = 0; i < N; i++) {
        ((BOOL(*)(id,SEL,id,id,id,NSError**))objc_msgSend)(
            g_client, @selector(evaluateRealTimeWithModel:options:request:error:),
            k.aneModel, @{}, k.request, &e);
    }
    printf("  [%s] OK! %.3f ms/eval (%d iters)\n", name, ms_t(mach_absolute_time() - t0)/N, N);

    _Float16 *out = (_Float16*)IOSurfaceGetBaseAddress(k.ioOut);
    printf("  [%s] out[0..3]: %f %f %f %f\n", name,
        (float)out[0], (float)out[1], (float)out[2], (float)out[3]);
    return k;
}

int main(int argc, char *argv[]) {
    @autoreleasepool {
        setbuf(stdout, NULL);
        mach_timebase_info(&g_tb);
        ane_init();

        int DIM = 1024, SEQ = 256;
        printf("=== RMSNorm ANE Test v2 (DIM=%d, SEQ=%d) ===\n\n", DIM, SEQ);

        // Test 1: Simple reduce_sum on axis -1 (known to work from sdpaBwd2)
        {
            printf("--- Test 1: reduce_sum axis=-1, small (4 channels, 256 spatial) ---\n");
            int C = 4, S = 256;
            NSMutableString *m = [NSMutableString string];
            [m appendString:@"program(1.3)\n"
                "[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}, "
                "{\"coremlc-version\", \"3505.4.1\"}, {\"coremltools-component-milinternal\", \"\"}, "
                "{\"coremltools-version\", \"9.0\"}})]\n{\n"];
            [m appendFormat:@"    func main<ios18>(tensor<fp16, [1, %d, 1, %d]> x) {\n", C, S];
            [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> x2 = mul(x=x, y=x)[name=string(\"x2\")];\n", C, S];
            [m appendString:@"        tensor<int32, [1]> rax = const()[name=string(\"rax\"), val=tensor<int32, [1]>([-1])];\n"];
            [m appendString:@"        bool kd = const()[name=string(\"kd\"), val=bool(true)];\n"];
            [m appendFormat:@"        tensor<fp16, [1,%d,1,1]> ss = reduce_sum(x=x2, axes=rax, keep_dims=kd)[name=string(\"ss\")];\n", C];
            [m appendFormat:@"        tensor<fp16, [1,%d,1,1]> rrms = rsqrt(x=ss)[name=string(\"rrms\")];\n", C];
            [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> y = mul(x=x, y=rrms)[name=string(\"y\")];\n", C, S];
            [m appendString:@"    } -> (y);\n}\n"];
            try_compile_eval([m dataUsingEncoding:NSUTF8StringEncoding], C*S*2, C*S*2, "reduce_small");
        }

        // Test 2: Transpose DIM to spatial, reduce_sum, transpose back
        // [1, DIM, 1, SEQ] -> reshape [1,1,DIM,SEQ] -> transpose [1,1,SEQ,DIM]
        // reduce_sum axis=-1 -> [1,1,SEQ,1] -> rsqrt -> broadcast
        {
            printf("\n--- Test 2: RMSNorm via spatial reduction (DIM=%d) ---\n", DIM);
            NSMutableString *m = [NSMutableString string];
            [m appendString:@"program(1.3)\n"
                "[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}, "
                "{\"coremlc-version\", \"3505.4.1\"}, {\"coremltools-component-milinternal\", \"\"}, "
                "{\"coremltools-version\", \"9.0\"}})]\n{\n"];
            [m appendFormat:@"    func main<ios18>(tensor<fp16, [1, %d, 1, %d]> x) {\n", DIM, SEQ];

            // x^2: [1, DIM, 1, SEQ]
            [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> x2 = mul(x=x, y=x)[name=string(\"x2\")];\n", DIM, SEQ];

            // Reshape: [1, DIM, 1, SEQ] -> [1, 1, DIM, SEQ]
            [m appendFormat:@"        tensor<int32, [4]> r1 = const()[name=string(\"r1\"), val=tensor<int32, [4]>([1,1,%d,%d])];\n", DIM, SEQ];
            [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> x2r = reshape(shape=r1, x=x2)[name=string(\"x2r\")];\n", DIM, SEQ];

            // Transpose: [1, 1, DIM, SEQ] -> [1, 1, SEQ, DIM]
            [m appendString:@"        tensor<int32, [4]> pm = const()[name=string(\"pm\"), val=tensor<int32, [4]>([0,1,3,2])];\n"];
            [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> x2t = transpose(perm=pm, x=x2r)[name=string(\"x2t\")];\n", SEQ, DIM];

            // reduce_sum axis=-1: [1, 1, SEQ, DIM] -> [1, 1, SEQ, 1]
            [m appendString:@"        tensor<int32, [1]> rax = const()[name=string(\"rax\"), val=tensor<int32, [1]>([-1])];\n"];
            [m appendString:@"        bool kd = const()[name=string(\"kd\"), val=bool(true)];\n"];
            [m appendFormat:@"        tensor<fp16, [1,1,%d,1]> ss = reduce_sum(x=x2t, axes=rax, keep_dims=kd)[name=string(\"ss\")];\n", SEQ];

            // Divide by DIM: multiply by 1/DIM
            float inv_dim = 1.0f / DIM;
            [m appendFormat:@"        fp16 invd = const()[name=string(\"invd\"), val=fp16(%f)];\n", inv_dim];
            [m appendFormat:@"        tensor<fp16, [1,1,%d,1]> ms = mul(x=ss, y=invd)[name=string(\"ms\")];\n", SEQ];

            // Add eps
            [m appendFormat:@"        fp16 epsv = const()[name=string(\"epsv\"), val=fp16(1e-5)];\n"];
            [m appendFormat:@"        tensor<fp16, [1,1,%d,1]> mse = add(x=ms, y=epsv)[name=string(\"mse\")];\n", SEQ];

            // rsqrt
            [m appendFormat:@"        tensor<fp16, [1,1,%d,1]> rrms = rsqrt(x=mse)[name=string(\"rrms\")];\n", SEQ];

            // Transpose back: [1, 1, SEQ, 1] -> [1, 1, 1, SEQ]
            [m appendFormat:@"        tensor<fp16, [1,1,1,%d]> rr = transpose(perm=pm, x=rrms)[name=string(\"rr\")];\n", SEQ];

            // Reshape to [1, 1, 1, SEQ] for broadcast
            // Broadcast multiply: [1, DIM, 1, SEQ] * [1, 1, 1, SEQ]
            [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> y = mul(x=x, y=rr)[name=string(\"y\")];\n", DIM, SEQ];

            [m appendString:@"    } -> (y);\n}\n"];
            try_compile_eval([m dataUsingEncoding:NSUTF8StringEncoding], DIM*SEQ*2, DIM*SEQ*2, "spatial_reduce");
        }

        // Test 3: Use matmul for reduction (dot product with ones)
        // [1, 1, SEQ, DIM] @ [1, 1, DIM, 1] -> [1, 1, SEQ, 1]
        {
            printf("\n--- Test 3: RMSNorm via matmul reduction (x^2 @ ones) ---\n");
            NSMutableString *m = [NSMutableString string];
            [m appendString:@"program(1.3)\n"
                "[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}, "
                "{\"coremlc-version\", \"3505.4.1\"}, {\"coremltools-component-milinternal\", \"\"}, "
                "{\"coremltools-version\", \"9.0\"}})]\n{\n"];
            // Input: x [1, DIM, 1, SEQ] + ones vector (packed in spatial as const weight)
            int sp = SEQ;
            [m appendFormat:@"    func main<ios18>(tensor<fp16, [1, %d, 1, %d]> x) {\n", DIM, sp];

            // x^2
            [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> x2 = mul(x=x, y=x)[name=string(\"x2\")];\n", DIM, SEQ];

            // Reshape: [1, DIM, 1, SEQ] -> [1, 1, DIM, SEQ]
            [m appendFormat:@"        tensor<int32, [4]> r1 = const()[name=string(\"r1\"), val=tensor<int32, [4]>([1,1,%d,%d])];\n", DIM, SEQ];
            [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> x2r = reshape(shape=r1, x=x2)[name=string(\"x2r\")];\n", DIM, SEQ];

            // Transpose: [1, 1, DIM, SEQ] -> [1, 1, SEQ, DIM]
            [m appendString:@"        tensor<int32, [4]> pm = const()[name=string(\"pm\"), val=tensor<int32, [4]>([0,1,3,2])];\n"];
            [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> x2t = transpose(perm=pm, x=x2r)[name=string(\"x2t\")];\n", SEQ, DIM];

            // Ones vector as const: [1, 1, DIM, 1]
            // Use 1/DIM to combine sum and mean
            float inv_dim = 1.0f / DIM;
            [m appendFormat:@"        tensor<fp16, [1,1,%d,1]> ones = const()[name=string(\"ones\"), val=tensor<fp16, [1,1,%d,1]>(", DIM, DIM];
            for (int i = 0; i < DIM; i++) [m appendFormat:@"%s%f", i>0?",":"", inv_dim];
            [m appendString:@")];\n"];

            // matmul: [1,1,SEQ,DIM] @ [1,1,DIM,1] -> [1,1,SEQ,1]
            [m appendString:@"        bool bF = const()[name=string(\"bF\"), val=bool(false)];\n"];
            [m appendFormat:@"        tensor<fp16, [1,1,%d,1]> ms = matmul(transpose_x=bF, transpose_y=bF, x=x2t, y=ones)[name=string(\"ms\")];\n", SEQ];

            // Add eps + rsqrt
            [m appendFormat:@"        fp16 epsv = const()[name=string(\"epsv\"), val=fp16(1e-5)];\n"];
            [m appendFormat:@"        tensor<fp16, [1,1,%d,1]> mse = add(x=ms, y=epsv)[name=string(\"mse\")];\n", SEQ];
            [m appendFormat:@"        tensor<fp16, [1,1,%d,1]> rrms = rsqrt(x=mse)[name=string(\"rrms\")];\n", SEQ];

            // Transpose back + broadcast multiply
            [m appendFormat:@"        tensor<fp16, [1,1,1,%d]> rr = transpose(perm=pm, x=rrms)[name=string(\"rr\")];\n", SEQ];
            [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> y = mul(x=x, y=rr)[name=string(\"y\")];\n", DIM, SEQ];

            [m appendString:@"    } -> (y);\n}\n"];
            try_compile_eval([m dataUsingEncoding:NSUTF8StringEncoding], DIM*sp*2, DIM*SEQ*2, "matmul_reduce");
        }

        // Test 4: reduce_sum on height axis (axis 2) instead of width
        {
            printf("\n--- Test 4: reduce_sum on height axis (axis=2) ---\n");
            int C = 1, H = DIM, W = SEQ;
            NSMutableString *m = [NSMutableString string];
            [m appendString:@"program(1.3)\n"
                "[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}, "
                "{\"coremlc-version\", \"3505.4.1\"}, {\"coremltools-component-milinternal\", \"\"}, "
                "{\"coremltools-version\", \"9.0\"}})]\n{\n"];
            [m appendFormat:@"    func main<ios18>(tensor<fp16, [1, %d, %d, %d]> x) {\n", C, H, W];
            [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> x2 = mul(x=x, y=x)[name=string(\"x2\")];\n", C, H, W];
            // reduce_sum on axis 2 (height=DIM)
            [m appendString:@"        tensor<int32, [1]> rax = const()[name=string(\"rax\"), val=tensor<int32, [1]>([2])];\n"];
            [m appendString:@"        bool kd = const()[name=string(\"kd\"), val=bool(true)];\n"];
            [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> ss = reduce_sum(x=x2, axes=rax, keep_dims=kd)[name=string(\"ss\")];\n", C, W];
            [m appendFormat:@"        fp16 invd = const()[name=string(\"invd\"), val=fp16(%f)];\n", 1.0f/DIM];
            [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> ms = mul(x=ss, y=invd)[name=string(\"ms\")];\n", C, W];
            [m appendFormat:@"        fp16 epsv = const()[name=string(\"epsv\"), val=fp16(1e-5)];\n"];
            [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> mse = add(x=ms, y=epsv)[name=string(\"mse\")];\n", C, W];
            [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> rrms = rsqrt(x=mse)[name=string(\"rrms\")];\n", C, W];
            // Broadcast: [1,1,1,SEQ] * [1,1,DIM,SEQ]... need reshape back
            // Actually this layout won't work for broadcast back. Skip for now.
            [m appendString:@"    } -> (rrms);\n}\n"];
            try_compile_eval([m dataUsingEncoding:NSUTF8StringEncoding], C*H*W*2, C*W*2, "height_reduce");
        }

        // Test 5: Channel-dim reduction with very small channel count
        {
            printf("\n--- Test 5: reduce_sum on channel axis (axis=1), C=16 ---\n");
            int C = 16, S = 256;
            NSMutableString *m = [NSMutableString string];
            [m appendString:@"program(1.3)\n"
                "[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}, "
                "{\"coremlc-version\", \"3505.4.1\"}, {\"coremltools-component-milinternal\", \"\"}, "
                "{\"coremltools-version\", \"9.0\"}})]\n{\n"];
            [m appendFormat:@"    func main<ios18>(tensor<fp16, [1, %d, 1, %d]> x) {\n", C, S];
            [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> x2 = mul(x=x, y=x)[name=string(\"x2\")];\n", C, S];
            [m appendString:@"        tensor<int32, [1]> rax = const()[name=string(\"rax\"), val=tensor<int32, [1]>([1])];\n"];
            [m appendString:@"        bool kd = const()[name=string(\"kd\"), val=bool(true)];\n"];
            [m appendFormat:@"        tensor<fp16, [1,1,1,%d]> ss = reduce_sum(x=x2, axes=rax, keep_dims=kd)[name=string(\"ss\")];\n", S];
            [m appendString:@"    } -> (ss);\n}\n"];
            try_compile_eval([m dataUsingEncoding:NSUTF8StringEncoding], C*S*2, S*2, "chan_reduce_16");
        }

        printf("\n=== Done ===\n");
    }
    return 0;
}
