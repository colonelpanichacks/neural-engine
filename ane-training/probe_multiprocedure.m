// probe_multiprocedure.m — Test multi-procedure MIL programs
// If ANE supports multiple funcs in one program, we can reduce compiled model count
// and potentially use different procedureIndex values to select which to execute.
//
// Build:
//   xcrun clang -O2 -framework Foundation -framework IOSurface \
//     -isysroot /Library/Developer/CommandLineTools/SDKs/MacOSX.sdk \
//     -fobjc-arc -o probe_multiprocedure probe_multiprocedure.m

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
static NSMutableArray *g_keepalive;

static void ane_init(void) {
    dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine", RTLD_NOW);
    g_D  = NSClassFromString(@"_ANEInMemoryModelDescriptor");
    g_I  = NSClassFromString(@"_ANEInMemoryModel");
    g_AR = NSClassFromString(@"_ANERequest");
    g_AIO= NSClassFromString(@"_ANEIOSurfaceObject");
    g_client = ((id(*)(Class,SEL))objc_msgSend)(NSClassFromString(@"_ANEClient"), @selector(sharedConnection));
    g_keepalive = [NSMutableArray array];
}

static IOSurfaceRef make_surf(size_t b) {
    return IOSurfaceCreate((__bridge CFDictionaryRef)@{
        (id)kIOSurfaceWidth:@(b),(id)kIOSurfaceHeight:@1,
        (id)kIOSurfaceBytesPerElement:@1,(id)kIOSurfaceBytesPerRow:@(b),
        (id)kIOSurfaceAllocSize:@(b),(id)kIOSurfacePixelFormat:@0});
}

// Generate MIL with TWO functions — matmul with different output dims
// func0: IC x OC1 matmul
// func1: IC x OC2 matmul
static NSData *gen_dual_matmul(int IC, int OC1, int OC2, int SEQ) {
    NSMutableString *m = [NSMutableString string];
    [m appendString:@"program(1.3)\n[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"},{\"coremlc-version\", \"3505.4.1\"},{\"coremltools-component-milinternal\", \"\"},{\"coremltools-version\", \"9.0\"}})]\n{\n"];

    // Function 0: IC x OC1
    {
        int SP = SEQ + OC1;
        [m appendFormat:@"    func main<ios18>(tensor<fp16, [1, %d, 1, %d]> x) {\n", IC, SP];
        [m appendString:@"        tensor<int32, [4]> ba = const()[name=string(\"ba\"), val=tensor<int32, [4]>([0,0,0,0])];\n"];
        [m appendFormat:@"        tensor<int32, [4]> sa = const()[name=string(\"sa\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n", IC, SEQ];
        [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> act = slice_by_size(x=x,begin=ba,size=sa)[name=string(\"act\")];\n", IC, SEQ];
        [m appendFormat:@"        tensor<int32, [4]> bw = const()[name=string(\"bw\"), val=tensor<int32, [4]>([0,0,0,%d])];\n", SEQ];
        [m appendFormat:@"        tensor<int32, [4]> sw = const()[name=string(\"sw\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n", IC, OC1];
        [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> wt = slice_by_size(x=x,begin=bw,size=sw)[name=string(\"wt\")];\n", IC, OC1];
        [m appendFormat:@"        tensor<int32, [4]> ra = const()[name=string(\"ra\"), val=tensor<int32, [4]>([1,1,%d,%d])];\n", IC, SEQ];
        [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> a2 = reshape(shape=ra,x=act)[name=string(\"a2\")];\n", IC, SEQ];
        [m appendString:@"        tensor<int32, [4]> pm = const()[name=string(\"pm\"), val=tensor<int32, [4]>([0,1,3,2])];\n"];
        [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> a3 = transpose(perm=pm,x=a2)[name=string(\"a3\")];\n", SEQ, IC];
        [m appendFormat:@"        tensor<int32, [4]> rw = const()[name=string(\"rw\"), val=tensor<int32, [4]>([1,1,%d,%d])];\n", IC, OC1];
        [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> W = reshape(shape=rw,x=wt)[name=string(\"W\")];\n", IC, OC1];
        [m appendString:@"        bool bF = const()[name=string(\"bF\"), val=bool(false)];\n"];
        [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> yh = matmul(transpose_x=bF,transpose_y=bF,x=a3,y=W)[name=string(\"yh\")];\n", SEQ, OC1];
        [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> yt = transpose(perm=pm,x=yh)[name=string(\"yt\")];\n", OC1, SEQ];
        [m appendFormat:@"        tensor<int32, [4]> ro = const()[name=string(\"ro\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n", OC1, SEQ];
        [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> y = reshape(shape=ro,x=yt)[name=string(\"y\")];\n", OC1, SEQ];
        [m appendString:@"    } -> (y);\n\n"];
    }

    // Function 1: IC x OC2 (different name)
    {
        int SP = SEQ + OC2;
        [m appendFormat:@"    func func1<ios18>(tensor<fp16, [1, %d, 1, %d]> x) {\n", IC, SP];
        [m appendString:@"        tensor<int32, [4]> ba = const()[name=string(\"ba1\"), val=tensor<int32, [4]>([0,0,0,0])];\n"];
        [m appendFormat:@"        tensor<int32, [4]> sa = const()[name=string(\"sa1\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n", IC, SEQ];
        [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> act = slice_by_size(x=x,begin=ba,size=sa)[name=string(\"act1\")];\n", IC, SEQ];
        [m appendFormat:@"        tensor<int32, [4]> bw = const()[name=string(\"bw1\"), val=tensor<int32, [4]>([0,0,0,%d])];\n", SEQ];
        [m appendFormat:@"        tensor<int32, [4]> sw = const()[name=string(\"sw1\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n", IC, OC2];
        [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> wt = slice_by_size(x=x,begin=bw,size=sw)[name=string(\"wt1\")];\n", IC, OC2];
        [m appendFormat:@"        tensor<int32, [4]> ra = const()[name=string(\"ra1\"), val=tensor<int32, [4]>([1,1,%d,%d])];\n", IC, SEQ];
        [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> a2 = reshape(shape=ra,x=act)[name=string(\"a21\")];\n", IC, SEQ];
        [m appendString:@"        tensor<int32, [4]> pm = const()[name=string(\"pm1\"), val=tensor<int32, [4]>([0,1,3,2])];\n"];
        [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> a3 = transpose(perm=pm,x=a2)[name=string(\"a31\")];\n", SEQ, IC];
        [m appendFormat:@"        tensor<int32, [4]> rw = const()[name=string(\"rw1\"), val=tensor<int32, [4]>([1,1,%d,%d])];\n", IC, OC2];
        [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> W = reshape(shape=rw,x=wt)[name=string(\"W1\")];\n", IC, OC2];
        [m appendString:@"        bool bF = const()[name=string(\"bF1\"), val=bool(false)];\n"];
        [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> yh = matmul(transpose_x=bF,transpose_y=bF,x=a3,y=W)[name=string(\"yh1\")];\n", SEQ, OC2];
        [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> yt = transpose(perm=pm,x=yh)[name=string(\"yt1\")];\n", OC2, SEQ];
        [m appendFormat:@"        tensor<int32, [4]> ro = const()[name=string(\"ro1\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n", OC2, SEQ];
        [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> y = reshape(shape=ro,x=yt)[name=string(\"y1\")];\n", OC2, SEQ];
        [m appendString:@"    } -> (y);\n"];
    }

    [m appendString:@"}\n"];
    return [m dataUsingEncoding:NSUTF8StringEncoding];
}

int main(int argc, char *argv[]) {
    @autoreleasepool {
        setbuf(stdout, NULL);
        mach_timebase_info(&g_tb);
        ane_init();

        printf("=== Multi-Procedure MIL Test ===\n\n");

        int IC = 256, OC1 = 512, OC2 = 1024, SEQ = 64;

        // Test 1: Compile a dual-function MIL program
        printf("--- Test 1: Dual-function MIL (OC1=%d, OC2=%d) ---\n", OC1, OC2);
        {
            NSData *mil = gen_dual_matmul(IC, OC1, OC2, SEQ);
            printf("  MIL size: %lu bytes\n", (unsigned long)[mil length]);

            // Print first few lines of MIL for verification
            NSString *milStr = [[NSString alloc] initWithData:mil encoding:NSUTF8StringEncoding];
            NSArray *lines = [milStr componentsSeparatedByString:@"\n"];
            printf("  MIL preview:\n");
            for (int i = 0; i < MIN(5, (int)[lines count]); i++) {
                printf("    %s\n", [lines[i] UTF8String]);
            }
            printf("    ...\n");

            id desc = ((id(*)(Class,SEL,id,id,id))objc_msgSend)(g_D,
                @selector(modelWithMILText:weights:optionsPlist:), mil, @{}, nil);
            if (!desc) { printf("  ERROR: desc is nil\n"); goto test2; }

            id mdl = ((id(*)(Class,SEL,id))objc_msgSend)(g_I,
                @selector(inMemoryModelWithDescriptor:), desc);
            [g_keepalive addObject:mdl];

            id hx = ((id(*)(id,SEL))objc_msgSend)(mdl, @selector(hexStringIdentifier));
            NSString *td = [NSTemporaryDirectory() stringByAppendingPathComponent:hx];
            [[NSFileManager defaultManager] createDirectoryAtPath:[td stringByAppendingPathComponent:@"weights"]
                withIntermediateDirectories:YES attributes:nil error:nil];
            [mil writeToFile:[td stringByAppendingPathComponent:@"model.mil"] atomically:YES];

            NSError *e = nil;
            BOOL ok = ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(mdl,
                @selector(compileWithQoS:options:error:), 21, @{}, &e);
            printf("  compile: %s %s\n", ok?"OK":"FAIL", e?[[e description] UTF8String]:"");
            if (!ok) goto test2;

            ok = ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(mdl,
                @selector(loadWithQoS:options:error:), 21, @{}, &e);
            printf("  load: %s %s\n", ok?"OK":"FAIL", e?[[e description] UTF8String]:"");
            if (!ok) goto test2;

            id aneModel = ((id(*)(id,SEL))objc_msgSend)(mdl, @selector(model));
            [g_keepalive addObject:aneModel];

            // Try evaluating procedure 0 (main)
            {
                size_t in0_bytes = (size_t)IC * (SEQ + OC1) * 2;
                size_t out0_bytes = (size_t)OC1 * SEQ * 2;
                IOSurfaceRef ioIn = make_surf(in0_bytes);
                IOSurfaceRef ioOut = make_surf(out0_bytes);
                id wI = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_AIO, @selector(objectWithIOSurface:), ioIn);
                id wO = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_AIO, @selector(objectWithIOSurface:), ioOut);
                [g_keepalive addObject:wI]; [g_keepalive addObject:wO];

                id req = ((id(*)(Class,SEL,id,id,id,id,id,id))objc_msgSend)(g_AR,
                    @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:procedureIndex:),
                    @[wI], @[@0], @[wO], @[@0], nil, @0);
                [g_keepalive addObject:req];

                e = nil;
                @try {
                    ok = ((BOOL(*)(id,SEL,id,id,id,unsigned int,NSError**))objc_msgSend)(g_client,
                        @selector(doEvaluateDirectWithModel:options:request:qos:error:),
                        aneModel, @{}, req, 21, &e);
                    printf("  eval procedure 0: %s %s\n", ok?"OK":"FAIL", e?[[e description] UTF8String]:"");
                } @catch (NSException *ex) {
                    printf("  eval procedure 0: EXCEPTION %s\n", [[ex description] UTF8String]);
                }

                // Benchmark
                if (ok) {
                    int N = 200;
                    for (int i = 0; i < 50; i++)
                        ((BOOL(*)(id,SEL,id,id,id,unsigned int,NSError**))objc_msgSend)(g_client,
                            @selector(doEvaluateDirectWithModel:options:request:qos:error:),
                            aneModel, @{}, req, 21, &e);
                    uint64_t t0 = mach_absolute_time();
                    for (int i = 0; i < N; i++)
                        ((BOOL(*)(id,SEL,id,id,id,unsigned int,NSError**))objc_msgSend)(g_client,
                            @selector(doEvaluateDirectWithModel:options:request:qos:error:),
                            aneModel, @{}, req, 21, &e);
                    printf("  procedure 0: %.3f ms/eval\n", ms_t(mach_absolute_time()-t0)/N);
                }

                CFRelease(ioIn);
                CFRelease(ioOut);
            }

            // Try evaluating procedure 1 (func1)
            {
                size_t in1_bytes = (size_t)IC * (SEQ + OC2) * 2;
                size_t out1_bytes = (size_t)OC2 * SEQ * 2;
                IOSurfaceRef ioIn = make_surf(in1_bytes);
                IOSurfaceRef ioOut = make_surf(out1_bytes);
                id wI = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_AIO, @selector(objectWithIOSurface:), ioIn);
                id wO = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_AIO, @selector(objectWithIOSurface:), ioOut);
                [g_keepalive addObject:wI]; [g_keepalive addObject:wO];

                id req = ((id(*)(Class,SEL,id,id,id,id,id,id))objc_msgSend)(g_AR,
                    @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:procedureIndex:),
                    @[wI], @[@0], @[wO], @[@0], nil, @1);
                [g_keepalive addObject:req];

                e = nil;
                @try {
                    ok = ((BOOL(*)(id,SEL,id,id,id,unsigned int,NSError**))objc_msgSend)(g_client,
                        @selector(doEvaluateDirectWithModel:options:request:qos:error:),
                        aneModel, @{}, req, 21, &e);
                    printf("  eval procedure 1: %s %s\n", ok?"OK":"FAIL", e?[[e description] UTF8String]:"");
                } @catch (NSException *ex) {
                    printf("  eval procedure 1: EXCEPTION %s\n", [[ex description] UTF8String]);
                }

                // Benchmark
                if (ok) {
                    int N = 200;
                    for (int i = 0; i < 50; i++)
                        ((BOOL(*)(id,SEL,id,id,id,unsigned int,NSError**))objc_msgSend)(g_client,
                            @selector(doEvaluateDirectWithModel:options:request:qos:error:),
                            aneModel, @{}, req, 21, &e);
                    uint64_t t0 = mach_absolute_time();
                    for (int i = 0; i < N; i++)
                        ((BOOL(*)(id,SEL,id,id,id,unsigned int,NSError**))objc_msgSend)(g_client,
                            @selector(doEvaluateDirectWithModel:options:request:qos:error:),
                            aneModel, @{}, req, 21, &e);
                    printf("  procedure 1: %.3f ms/eval\n", ms_t(mach_absolute_time()-t0)/N);
                }

                CFRelease(ioIn);
                CFRelease(ioOut);
            }

            // Try procedure 2 (should fail — only 2 funcs)
            {
                size_t in_bytes = (size_t)IC * (SEQ + OC1) * 2;
                size_t out_bytes = (size_t)OC1 * SEQ * 2;
                IOSurfaceRef ioIn = make_surf(in_bytes);
                IOSurfaceRef ioOut = make_surf(out_bytes);
                id wI = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_AIO, @selector(objectWithIOSurface:), ioIn);
                id wO = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_AIO, @selector(objectWithIOSurface:), ioOut);

                id req = ((id(*)(Class,SEL,id,id,id,id,id,id))objc_msgSend)(g_AR,
                    @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:procedureIndex:),
                    @[wI], @[@0], @[wO], @[@0], nil, @2);

                e = nil;
                @try {
                    ok = ((BOOL(*)(id,SEL,id,id,id,unsigned int,NSError**))objc_msgSend)(g_client,
                        @selector(doEvaluateDirectWithModel:options:request:qos:error:),
                        aneModel, @{}, req, 21, &e);
                    printf("  eval procedure 2 (should fail): %s %s\n", ok?"OK":"FAIL", e?[[e description] UTF8String]:"");
                } @catch (NSException *ex) {
                    printf("  eval procedure 2: EXCEPTION %s\n", [[ex description] UTF8String]);
                }

                CFRelease(ioIn);
                CFRelease(ioOut);
            }
        }

        test2:
        // Test 2: Compare single vs dual — does multi-procedure cost more per compile?
        printf("\n--- Test 2: Single-function compile for comparison ---\n");
        {
            // Single matmul IC x OC1
            NSMutableString *m = [NSMutableString string];
            int SP = SEQ + OC1;
            [m appendString:@"program(1.3)\n[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"},{\"coremlc-version\", \"3505.4.1\"},{\"coremltools-component-milinternal\", \"\"},{\"coremltools-version\", \"9.0\"}})]\n{\n"];
            [m appendFormat:@"    func main<ios18>(tensor<fp16, [1, %d, 1, %d]> x) {\n", IC, SP];
            [m appendString:@"        tensor<int32, [4]> ba = const()[name=string(\"ba\"), val=tensor<int32, [4]>([0,0,0,0])];\n"];
            [m appendFormat:@"        tensor<int32, [4]> sa = const()[name=string(\"sa\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n", IC, SEQ];
            [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> act = slice_by_size(x=x,begin=ba,size=sa)[name=string(\"act\")];\n", IC, SEQ];
            [m appendFormat:@"        tensor<int32, [4]> bw = const()[name=string(\"bw\"), val=tensor<int32, [4]>([0,0,0,%d])];\n", SEQ];
            [m appendFormat:@"        tensor<int32, [4]> sw = const()[name=string(\"sw\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n", IC, OC1];
            [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> wt = slice_by_size(x=x,begin=bw,size=sw)[name=string(\"wt\")];\n", IC, OC1];
            [m appendFormat:@"        tensor<int32, [4]> ra = const()[name=string(\"ra\"), val=tensor<int32, [4]>([1,1,%d,%d])];\n", IC, SEQ];
            [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> a2 = reshape(shape=ra,x=act)[name=string(\"a2\")];\n", IC, SEQ];
            [m appendString:@"        tensor<int32, [4]> pm = const()[name=string(\"pm\"), val=tensor<int32, [4]>([0,1,3,2])];\n"];
            [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> a3 = transpose(perm=pm,x=a2)[name=string(\"a3\")];\n", SEQ, IC];
            [m appendFormat:@"        tensor<int32, [4]> rw = const()[name=string(\"rw\"), val=tensor<int32, [4]>([1,1,%d,%d])];\n", IC, OC1];
            [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> W = reshape(shape=rw,x=wt)[name=string(\"W\")];\n", IC, OC1];
            [m appendString:@"        bool bF = const()[name=string(\"bF\"), val=bool(false)];\n"];
            [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> yh = matmul(transpose_x=bF,transpose_y=bF,x=a3,y=W)[name=string(\"yh\")];\n", SEQ, OC1];
            [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> yt = transpose(perm=pm,x=yh)[name=string(\"yt\")];\n", OC1, SEQ];
            [m appendFormat:@"        tensor<int32, [4]> ro = const()[name=string(\"ro\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n", OC1, SEQ];
            [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> y = reshape(shape=ro,x=yt)[name=string(\"y\")];\n", OC1, SEQ];
            [m appendString:@"    } -> (y);\n}\n"];
            NSData *mil = [m dataUsingEncoding:NSUTF8StringEncoding];

            id desc = ((id(*)(Class,SEL,id,id,id))objc_msgSend)(g_D,
                @selector(modelWithMILText:weights:optionsPlist:), mil, @{}, nil);
            id mdl = ((id(*)(Class,SEL,id))objc_msgSend)(g_I,
                @selector(inMemoryModelWithDescriptor:), desc);
            [g_keepalive addObject:mdl];
            id hx = ((id(*)(id,SEL))objc_msgSend)(mdl, @selector(hexStringIdentifier));
            NSString *td = [NSTemporaryDirectory() stringByAppendingPathComponent:hx];
            [[NSFileManager defaultManager] createDirectoryAtPath:[td stringByAppendingPathComponent:@"weights"]
                withIntermediateDirectories:YES attributes:nil error:nil];
            [mil writeToFile:[td stringByAppendingPathComponent:@"model.mil"] atomically:YES];

            NSError *e = nil;
            BOOL ok = ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(mdl,
                @selector(compileWithQoS:options:error:), 21, @{}, &e);
            printf("  single compile: %s\n", ok?"OK":"FAIL");
            ok = ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(mdl,
                @selector(loadWithQoS:options:error:), 21, @{}, &e);
            printf("  single load: %s\n", ok?"OK":"FAIL");
            if (!ok) goto done;

            id aneModel = ((id(*)(id,SEL))objc_msgSend)(mdl, @selector(model));
            [g_keepalive addObject:aneModel];

            size_t in_bytes = (size_t)IC * (SEQ + OC1) * 2;
            size_t out_bytes = (size_t)OC1 * SEQ * 2;
            IOSurfaceRef ioIn = make_surf(in_bytes);
            IOSurfaceRef ioOut = make_surf(out_bytes);
            id wI = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_AIO, @selector(objectWithIOSurface:), ioIn);
            id wO = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_AIO, @selector(objectWithIOSurface:), ioOut);
            [g_keepalive addObject:wI]; [g_keepalive addObject:wO];

            id req = ((id(*)(Class,SEL,id,id,id,id,id,id))objc_msgSend)(g_AR,
                @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:procedureIndex:),
                @[wI], @[@0], @[wO], @[@0], nil, @0);

            int N = 200;
            for (int i = 0; i < 50; i++)
                ((BOOL(*)(id,SEL,id,id,id,unsigned int,NSError**))objc_msgSend)(g_client,
                    @selector(doEvaluateDirectWithModel:options:request:qos:error:),
                    aneModel, @{}, req, 21, &e);
            uint64_t t0 = mach_absolute_time();
            for (int i = 0; i < N; i++)
                ((BOOL(*)(id,SEL,id,id,id,unsigned int,NSError**))objc_msgSend)(g_client,
                    @selector(doEvaluateDirectWithModel:options:request:qos:error:),
                    aneModel, @{}, req, 21, &e);
            printf("  single procedure: %.3f ms/eval\n", ms_t(mach_absolute_time()-t0)/N);

            CFRelease(ioIn);
            CFRelease(ioOut);
        }

        done:
        printf("\n=== Done ===\n");
    }
    return 0;
}
