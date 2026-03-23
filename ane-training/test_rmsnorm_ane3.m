// test_rmsnorm_ane3.m — RMSNorm on ANE using valid ops (pow, reduce_mean)
// Also test layer_norm at DIM=1024
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

static BOOL compile_and_eval(NSData *mil, size_t in_bytes, size_t out_bytes,
                              const char *name, _Float16 **out_ptr) {
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
        printf("  [%s] COMPILE FAIL: %s\n", name,
            e ? [[[e userInfo][@"NSUnderlyingError"] description] UTF8String] : "unknown");
        return NO;
    }
    ok = ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(
        model, @selector(loadWithQoS:options:error:), 21, @{}, &e);
    if (!ok) {
        printf("  [%s] LOAD FAIL: %s\n", name, e ? [[e description] UTF8String] : "unknown");
        return NO;
    }

    IOSurfaceRef ioIn = make_surf(in_bytes);
    IOSurfaceRef ioOut = make_surf(out_bytes);

    _Float16 *inp = (_Float16*)IOSurfaceGetBaseAddress(ioIn);
    // Use known values for verification
    int DIM = 1024, SEQ = 256;
    for (int c = 0; c < DIM; c++)
        for (int s = 0; s < SEQ; s++)
            inp[c * SEQ + s] = (_Float16)(0.1f * ((c + s) % 10 + 1));

    id wI = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_AIO, @selector(objectWithIOSurface:), ioIn);
    id wO = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_AIO, @selector(objectWithIOSurface:), ioOut);
    id req = ((id(*)(Class,SEL,id,id,id,id,id,id,id))objc_msgSend)(g_AR,
        @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:),
        @[wI], @[@0], @[wO], @[@0], nil, nil, @0);
    id aneModel = ((id(*)(id,SEL))objc_msgSend)(model, @selector(model));

    ok = ((BOOL(*)(id,SEL,id,id,id,NSError**))objc_msgSend)(
        g_client, @selector(evaluateRealTimeWithModel:options:request:error:),
        aneModel, @{}, req, &e);
    if (!ok) {
        printf("  [%s] EVAL FAIL: %s\n", name, e ? [[e description] UTF8String] : "unknown");
        CFRelease(ioIn); CFRelease(ioOut);
        return NO;
    }

    // Benchmark
    int N = 200;
    uint64_t t0 = mach_absolute_time();
    for (int i = 0; i < N; i++) {
        ((BOOL(*)(id,SEL,id,id,id,NSError**))objc_msgSend)(
            g_client, @selector(evaluateRealTimeWithModel:options:request:error:),
            aneModel, @{}, req, &e);
    }
    double total_ms = ms_t(mach_absolute_time() - t0);
    printf("  [%s] OK! %.3f ms/eval\n", name, total_ms/N);

    _Float16 *out = (_Float16*)IOSurfaceGetBaseAddress(ioOut);
    printf("  [%s] out[0..3]: %f %f %f %f\n", name,
        (float)out[0], (float)out[1], (float)out[2], (float)out[3]);
    if (out_ptr) *out_ptr = out;

    return YES;
}

// Generate RMSNorm MIL: x * pow(reduce_mean(x^2, axis=1) + eps, -0.5)
static NSData *gen_rmsnorm_mil(int DIM, int SEQ) {
    NSMutableString *m = [NSMutableString string];
    [m appendString:@"program(1.3)\n"
        "[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}, "
        "{\"coremlc-version\", \"3505.4.1\"}, {\"coremltools-component-milinternal\", \"\"}, "
        "{\"coremltools-version\", \"9.0\"}})]\n{\n"];
    [m appendFormat:@"    func main<ios18>(tensor<fp16, [1, %d, 1, %d]> x) {\n", DIM, SEQ];

    // x^2
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> x2 = mul(x=x, y=x)[name=string(\"x2\")];\n", DIM, SEQ];

    // reduce_mean across channels (axis=1)
    [m appendString:@"        tensor<int32, [1]> ax = const()[name=string(\"ax\"), val=tensor<int32, [1]>([1])];\n"];
    [m appendString:@"        bool kd = const()[name=string(\"kd\"), val=bool(true)];\n"];
    [m appendFormat:@"        tensor<fp16, [1,1,1,%d]> ms = reduce_mean(x=x2, axes=ax, keep_dims=kd)[name=string(\"ms\")];\n", SEQ];

    // add eps
    [m appendFormat:@"        fp16 epsv = const()[name=string(\"epsv\"), val=fp16(1e-5)];\n"];
    [m appendFormat:@"        tensor<fp16, [1,1,1,%d]> mse = add(x=ms, y=epsv)[name=string(\"mse\")];\n", SEQ];

    // pow(mse, -0.5) = rsqrt
    [m appendFormat:@"        fp16 nhalf = const()[name=string(\"nh\"), val=fp16(-0.5)];\n"];
    [m appendFormat:@"        tensor<fp16, [1,1,1,%d]> rrms = pow(x=mse, y=nhalf)[name=string(\"rrms\")];\n", SEQ];

    // x * rrms (broadcast: [1,DIM,1,SEQ] * [1,1,1,SEQ])
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> y = mul(x=x, y=rrms)[name=string(\"y\")];\n", DIM, SEQ];

    [m appendString:@"    } -> (y);\n}\n"];
    return [m dataUsingEncoding:NSUTF8StringEncoding];
}

// Generate RMSNorm with weights: x * pow(reduce_mean(x^2, axis=1) + eps, -0.5) * w
// Weight w is packed into the spatial dimension of input
static NSData *gen_rmsnorm_weighted_mil(int DIM, int SEQ) {
    int SP = SEQ + DIM;  // activations + weights packed in spatial
    NSMutableString *m = [NSMutableString string];
    [m appendString:@"program(1.3)\n"
        "[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}, "
        "{\"coremlc-version\", \"3505.4.1\"}, {\"coremltools-component-milinternal\", \"\"}, "
        "{\"coremltools-version\", \"9.0\"}})]\n{\n"];
    [m appendFormat:@"    func main<ios18>(tensor<fp16, [1, %d, 1, %d]> inp) {\n", DIM, SP];

    // Slice activations: [1, DIM, 1, SEQ]
    [m appendString:@"        tensor<int32, [4]> ba = const()[name=string(\"ba\"), val=tensor<int32, [4]>([0,0,0,0])];\n"];
    [m appendFormat:@"        tensor<int32, [4]> sa = const()[name=string(\"sa\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n", DIM, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> x = slice_by_size(x=inp,begin=ba,size=sa)[name=string(\"x\")];\n", DIM, SEQ];

    // Slice weight: [1, DIM, 1, 1] — one weight per channel, broadcast across SEQ
    // Actually weights are [DIM] — one per channel. Pack as [1, DIM, 1, 1]
    // But we pack weights as DIM values in spatial: [1, 1, 1, DIM]
    // Need to reshape to [1, DIM, 1, 1] for channel-wise multiply
    [m appendFormat:@"        tensor<int32, [4]> bw = const()[name=string(\"bw\"), val=tensor<int32, [4]>([0,0,0,%d])];\n", SEQ];
    [m appendFormat:@"        tensor<int32, [4]> sw = const()[name=string(\"sw\"), val=tensor<int32, [4]>([1,%d,1,1])];\n", DIM];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,1]> w = slice_by_size(x=inp,begin=bw,size=sw)[name=string(\"w\")];\n", DIM];

    // x^2 + reduce_mean + eps + pow(-0.5) + multiply
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> x2 = mul(x=x, y=x)[name=string(\"x2\")];\n", DIM, SEQ];
    [m appendString:@"        tensor<int32, [1]> ax = const()[name=string(\"ax\"), val=tensor<int32, [1]>([1])];\n"];
    [m appendString:@"        bool kd = const()[name=string(\"kd\"), val=bool(true)];\n"];
    [m appendFormat:@"        tensor<fp16, [1,1,1,%d]> ms = reduce_mean(x=x2, axes=ax, keep_dims=kd)[name=string(\"ms\")];\n", SEQ];
    [m appendFormat:@"        fp16 epsv = const()[name=string(\"epsv\"), val=fp16(1e-5)];\n"];
    [m appendFormat:@"        tensor<fp16, [1,1,1,%d]> mse = add(x=ms, y=epsv)[name=string(\"mse\")];\n", SEQ];
    [m appendFormat:@"        fp16 nhalf = const()[name=string(\"nh\"), val=fp16(-0.5)];\n"];
    [m appendFormat:@"        tensor<fp16, [1,1,1,%d]> rrms = pow(x=mse, y=nhalf)[name=string(\"rrms\")];\n", SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> xn = mul(x=x, y=rrms)[name=string(\"xn\")];\n", DIM, SEQ];
    // Apply weight: [1,DIM,1,SEQ] * [1,DIM,1,1] → broadcast
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> y = mul(x=xn, y=w)[name=string(\"y\")];\n", DIM, SEQ];

    [m appendString:@"    } -> (y);\n}\n"];
    return [m dataUsingEncoding:NSUTF8StringEncoding];
}

int main(int argc, char *argv[]) {
    @autoreleasepool {
        setbuf(stdout, NULL);
        mach_timebase_info(&g_tb);
        ane_init();

        printf("=== RMSNorm on ANE v3 ===\n\n");

        int DIM = 1024, SEQ = 256;

        // Test 1: RMSNorm at DIM=1024 (no weights)
        {
            printf("--- RMSNorm DIM=%d SEQ=%d (no weights) ---\n", DIM, SEQ);
            NSData *mil = gen_rmsnorm_mil(DIM, SEQ);
            compile_and_eval(mil, DIM*SEQ*2, DIM*SEQ*2, "rmsnorm_1024", NULL);
        }

        // Test 2: RMSNorm at different channel dims
        for (int d = 128; d <= 4096; d *= 2) {
            printf("\n--- RMSNorm DIM=%d SEQ=%d ---\n", d, SEQ);
            NSData *mil = gen_rmsnorm_mil(d, SEQ);
            char name[64]; snprintf(name, sizeof(name), "rmsnorm_%d", d);
            compile_and_eval(mil, d*SEQ*2, d*SEQ*2, name, NULL);
        }

        // Test 3: RMSNorm with weights at DIM=1024
        {
            printf("\n--- RMSNorm+weights DIM=%d SEQ=%d ---\n", DIM, SEQ);
            NSData *mil = gen_rmsnorm_weighted_mil(DIM, SEQ);
            int SP = SEQ + DIM;
            compile_and_eval(mil, DIM*SP*2, DIM*SEQ*2, "rmsnorm_wt_1024", NULL);
        }

        // Test 4: layer_norm at DIM=1024 on channel axis
        {
            printf("\n--- layer_norm DIM=%d SEQ=%d (axis=1) ---\n", DIM, SEQ);
            NSMutableString *m = [NSMutableString string];
            [m appendString:@"program(1.3)\n"
                "[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}, "
                "{\"coremlc-version\", \"3505.4.1\"}, {\"coremltools-component-milinternal\", \"\"}, "
                "{\"coremltools-version\", \"9.0\"}})]\n{\n"];
            [m appendFormat:@"    func main<ios18>(tensor<fp16, [1, %d, 1, %d]> x) {\n", DIM, SEQ];

            // Generate gamma (all ones) and beta (all zeros) for DIM channels
            [m appendFormat:@"        tensor<fp16, [%d]> gam = const()[name=string(\"g\"), val=tensor<fp16, [%d]>(", DIM, DIM];
            for (int i = 0; i < DIM; i++) [m appendFormat:@"%s1", i>0?",":""];
            [m appendString:@")];\n"];
            [m appendFormat:@"        tensor<fp16, [%d]> bet = const()[name=string(\"b\"), val=tensor<fp16, [%d]>(", DIM, DIM];
            for (int i = 0; i < DIM; i++) [m appendFormat:@"%s0", i>0?",":""];
            [m appendString:@")];\n"];
            [m appendFormat:@"        fp16 eps = const()[name=string(\"eps\"), val=fp16(1e-5)];\n"];
            [m appendString:@"        tensor<int32, [1]> nax = const()[name=string(\"nax\"), val=tensor<int32, [1]>([1])];\n"];
            [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> y = layer_norm(x=x, axes=nax, gamma=gam, beta=bet, epsilon=eps)[name=string(\"y\")];\n", DIM, SEQ];
            [m appendString:@"    } -> (y);\n}\n"];

            compile_and_eval([m dataUsingEncoding:NSUTF8StringEncoding], DIM*SEQ*2, DIM*SEQ*2, "layer_norm_1024", NULL);
        }

        // Test 5: Verify RMSNorm correctness vs CPU
        {
            printf("\n--- RMSNorm correctness verification ---\n");
            // CPU reference
            float *x_f32 = (float*)malloc(DIM*SEQ*4);
            float *y_cpu = (float*)malloc(DIM*SEQ*4);
            for (int c = 0; c < DIM; c++)
                for (int s = 0; s < SEQ; s++)
                    x_f32[c*SEQ+s] = 0.1f * ((c+s) % 10 + 1);

            // Compute RMSNorm per token (across channels)
            for (int s = 0; s < SEQ; s++) {
                float ss = 0;
                for (int c = 0; c < DIM; c++) ss += x_f32[c*SEQ+s] * x_f32[c*SEQ+s];
                float rrms = 1.0f / sqrtf(ss / DIM + 1e-5f);
                for (int c = 0; c < DIM; c++) y_cpu[c*SEQ+s] = x_f32[c*SEQ+s] * rrms;
            }
            printf("  CPU out[0..3]: %f %f %f %f\n",
                y_cpu[0], y_cpu[1], y_cpu[2], y_cpu[3]);
            free(x_f32); free(y_cpu);
        }

        printf("\n=== Done ===\n");
    }
    return 0;
}
