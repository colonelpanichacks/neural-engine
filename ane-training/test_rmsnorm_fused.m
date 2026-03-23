// test_rmsnorm_fused.m — Test RMSNorm fused with matmul on ANE
// Goal: RMSNorm(x, w) → x_norm @ W → y (single kernel)
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

int main(int argc, char *argv[]) {
    @autoreleasepool {
        setbuf(stdout, NULL);
        mach_timebase_info(&g_tb);
        ane_init();

        int DIM = 1024, OC = 2048, SEQ = 256;
        printf("=== RMSNorm + Matmul Fused Kernel Test ===\n");
        printf("DIM=%d OC=%d SEQ=%d\n\n", DIM, OC, SEQ);

        // Build fused kernel: RMSNorm(x, rms_w) → x_norm, then x_norm @ W → y
        // Input layout [1, DIM, 1, SP]:
        //   [0..SEQ)        = x activations
        //   [SEQ..SEQ+1)    = RMSNorm weights (DIM values at spatial SEQ, one per channel)
        //   [SEQ+1..SEQ+1+OC) = matmul weights W[DIM, OC]
        int SP = SEQ + 1 + OC;  // activations + rmsnorm weights (1 per ch) + matmul weights

        NSMutableString *m = [NSMutableString string];
        [m appendString:@"program(1.3)\n"
            "[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}, "
            "{\"coremlc-version\", \"3505.4.1\"}, {\"coremltools-component-milinternal\", \"\"}, "
            "{\"coremltools-version\", \"9.0\"}})]\n{\n"];
        [m appendFormat:@"    func main<ios18>(tensor<fp16, [1, %d, 1, %d]> inp) {\n", DIM, SP];

        // Slice x: [1, DIM, 1, SEQ]
        [m appendString:@"        tensor<int32, [4]> bx = const()[name=string(\"bx\"), val=tensor<int32, [4]>([0,0,0,0])];\n"];
        [m appendFormat:@"        tensor<int32, [4]> sx = const()[name=string(\"sx\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n", DIM, SEQ];
        [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> x = slice_by_size(x=inp,begin=bx,size=sx)[name=string(\"x\")];\n", DIM, SEQ];

        // Slice RMSNorm weights: [1, DIM, 1, 1]
        [m appendFormat:@"        tensor<int32, [4]> bw = const()[name=string(\"bw\"), val=tensor<int32, [4]>([0,0,0,%d])];\n", SEQ];
        [m appendFormat:@"        tensor<int32, [4]> sw = const()[name=string(\"sw\"), val=tensor<int32, [4]>([1,%d,1,1])];\n", DIM];
        [m appendFormat:@"        tensor<fp16, [1,%d,1,1]> rw = slice_by_size(x=inp,begin=bw,size=sw)[name=string(\"rw\")];\n", DIM];

        // === RMSNorm ===
        // x^2
        [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> x2 = mul(x=x, y=x)[name=string(\"x2\")];\n", DIM, SEQ];
        // reduce_mean across channels
        [m appendString:@"        tensor<int32, [1]> ax = const()[name=string(\"ax\"), val=tensor<int32, [1]>([1])];\n"];
        [m appendString:@"        bool kd = const()[name=string(\"kd\"), val=bool(true)];\n"];
        [m appendFormat:@"        tensor<fp16, [1,1,1,%d]> ms = reduce_mean(x=x2, axes=ax, keep_dims=kd)[name=string(\"ms\")];\n", SEQ];
        // + eps
        [m appendFormat:@"        fp16 epsv = const()[name=string(\"epsv\"), val=fp16(1e-5)];\n"];
        [m appendFormat:@"        tensor<fp16, [1,1,1,%d]> mse = add(x=ms, y=epsv)[name=string(\"mse\")];\n", SEQ];
        // pow(-0.5) = rsqrt
        [m appendFormat:@"        fp16 nhalf = const()[name=string(\"nh\"), val=fp16(-0.5)];\n"];
        [m appendFormat:@"        tensor<fp16, [1,1,1,%d]> rrms = pow(x=mse, y=nhalf)[name=string(\"rrms\")];\n", SEQ];
        // x * rrms * rms_weight
        [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> xn = mul(x=x, y=rrms)[name=string(\"xn\")];\n", DIM, SEQ];
        [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> xnw = mul(x=xn, y=rw)[name=string(\"xnw\")];\n", DIM, SEQ];

        // === Matmul: xnw @ W ===
        // Slice W: [1, DIM, 1, OC]
        [m appendFormat:@"        tensor<int32, [4]> bm = const()[name=string(\"bm\"), val=tensor<int32, [4]>([0,0,0,%d])];\n", SEQ + 1];
        [m appendFormat:@"        tensor<int32, [4]> sm = const()[name=string(\"sm\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n", DIM, OC];
        [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> W = slice_by_size(x=inp,begin=bm,size=sm)[name=string(\"W\")];\n", DIM, OC];

        // Reshape for matmul
        [m appendFormat:@"        tensor<int32, [4]> ra = const()[name=string(\"ra\"), val=tensor<int32, [4]>([1,1,%d,%d])];\n", DIM, SEQ];
        [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> a2 = reshape(shape=ra,x=xnw)[name=string(\"a2\")];\n", DIM, SEQ];
        [m appendString:@"        tensor<int32, [4]> pm = const()[name=string(\"pm\"), val=tensor<int32, [4]>([0,1,3,2])];\n"];
        [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> a3 = transpose(perm=pm,x=a2)[name=string(\"a3\")];\n", SEQ, DIM];
        [m appendFormat:@"        tensor<int32, [4]> rw2 = const()[name=string(\"rw2\"), val=tensor<int32, [4]>([1,1,%d,%d])];\n", DIM, OC];
        [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> Wr = reshape(shape=rw2,x=W)[name=string(\"Wr\")];\n", DIM, OC];
        [m appendString:@"        bool bF = const()[name=string(\"bF\"), val=bool(false)];\n"];
        // [1,1,SEQ,DIM] @ [1,1,DIM,OC] -> [1,1,SEQ,OC]
        [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> yh = matmul(transpose_x=bF,transpose_y=bF,x=a3,y=Wr)[name=string(\"yh\")];\n", SEQ, OC];
        [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> yt = transpose(perm=pm,x=yh)[name=string(\"yt\")];\n", OC, SEQ];
        [m appendFormat:@"        tensor<int32, [4]> ro = const()[name=string(\"ro\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n", OC, SEQ];
        [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> y = reshape(shape=ro,x=yt)[name=string(\"y\")];\n", OC, SEQ];

        [m appendString:@"    } -> (y);\n}\n"];

        NSData *mil = [m dataUsingEncoding:NSUTF8StringEncoding];

        // Compile
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
            printf("COMPILE FAIL: %s\n", e ? [[[e userInfo][@"NSUnderlyingError"] description] UTF8String] : "unknown");
            return 1;
        }
        ok = ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(
            model, @selector(loadWithQoS:options:error:), 21, @{}, &e);
        if (!ok) {
            printf("LOAD FAIL: %s\n", e ? [[e description] UTF8String] : "unknown");
            return 1;
        }
        printf("Compile+Load OK!\n");

        size_t in_bytes = (size_t)DIM * SP * 2;
        size_t out_bytes = (size_t)OC * SEQ * 2;
        IOSurfaceRef ioIn = make_surf(in_bytes);
        IOSurfaceRef ioOut = make_surf(out_bytes);

        // Fill input
        _Float16 *inp = (_Float16*)IOSurfaceGetBaseAddress(ioIn);
        for (int c = 0; c < DIM; c++) {
            // Activations
            for (int s = 0; s < SEQ; s++)
                inp[c*SP + s] = (_Float16)(0.01f * ((c+s) % 20 + 1));
            // RMSNorm weight (all 1.0 for now)
            inp[c*SP + SEQ] = (_Float16)1.0f;
            // Matmul weights (identity-ish)
            for (int o = 0; o < OC; o++)
                inp[c*SP + SEQ + 1 + o] = (_Float16)(c == o ? 1.0f : 0.0f);
        }

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
            printf("EVAL FAIL: %s\n", e ? [[e description] UTF8String] : "unknown");
            return 1;
        }

        _Float16 *out = (_Float16*)IOSurfaceGetBaseAddress(ioOut);
        printf("Eval OK! out[0..3]: %f %f %f %f\n",
            (float)out[0], (float)out[1], (float)out[2], (float)out[3]);

        // Benchmark
        int N = 200;
        // Warmup
        for (int i = 0; i < 20; i++) {
            ((BOOL(*)(id,SEL,id,id,id,NSError**))objc_msgSend)(
                g_client, @selector(evaluateRealTimeWithModel:options:request:error:),
                aneModel, @{}, req, &e);
        }
        uint64_t t0 = mach_absolute_time();
        for (int i = 0; i < N; i++) {
            ((BOOL(*)(id,SEL,id,id,id,NSError**))objc_msgSend)(
                g_client, @selector(evaluateRealTimeWithModel:options:request:error:),
                aneModel, @{}, req, &e);
        }
        double total_ms = ms_t(mach_absolute_time() - t0);
        printf("Benchmark: %.3f ms/eval (%d iters)\n", total_ms/N, N);

        // Compare with separate RMSNorm + matmul (would be 2 separate evals)
        printf("\nFor reference: standalone matmul at same size = ~0.28ms/eval\n");
        printf("Standalone RMSNorm = ~0.17ms/eval\n");
        printf("Sum = ~0.45ms. Fused saves dispatch overhead.\n");

        CFRelease(ioIn); CFRelease(ioOut);
        printf("\n=== Done ===\n");
    }
    return 0;
}
