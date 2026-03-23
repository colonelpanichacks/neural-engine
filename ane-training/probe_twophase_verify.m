// Verify two-phase dispatch actually executes the kernel
#import <Foundation/Foundation.h>
#import <IOSurface/IOSurface.h>
#import <objc/runtime.h>
#import <objc/message.h>
#import <dlfcn.h>
#import <mach/mach_time.h>

static mach_timebase_info_data_t g_tb;
static double ms_t(uint64_t t) { return (double)t * g_tb.numer / g_tb.denom / 1e6; }
static Class g_AIO, g_AR;
static id g_client;

static IOSurfaceRef make_surf(size_t bytes) {
    return IOSurfaceCreate((__bridge CFDictionaryRef)@{
        (id)kIOSurfaceWidth:@(bytes), (id)kIOSurfaceHeight:@1,
        (id)kIOSurfaceBytesPerElement:@1, (id)kIOSurfaceBytesPerRow:@(bytes),
        (id)kIOSurfaceAllocSize:@(bytes), (id)kIOSurfacePixelFormat:@0});
}

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

int main(int argc, char *argv[]) {
    @autoreleasepool {
        setbuf(stdout, NULL);
        mach_timebase_info(&g_tb);
        dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine", RTLD_NOW);
        g_AIO = NSClassFromString(@"_ANEIOSurfaceObject");
        g_AR  = NSClassFromString(@"_ANERequest");
        g_client = ((id(*)(Class,SEL))objc_msgSend)(NSClassFromString(@"_ANEClient"), @selector(sharedConnection));
        Class D = NSClassFromString(@"_ANEInMemoryModelDescriptor");
        Class I = NSClassFromString(@"_ANEInMemoryModel");
        Class IBR = NSClassFromString(@"_ANEInputBuffersReady");
        Class OSE = NSClassFromString(@"_ANEOutputSetEnqueue");

        printf("=== Two-Phase Dispatch Verification ===\n");

        // Compile 64x64 matmul
        int ic=64, oc=64, seq=32;
        NSData *mil = gen_mil(ic, oc, seq);
        size_t inB = ic*(seq+oc)*2, outB = oc*seq*2;
        id desc = ((id(*)(Class,SEL,id,id,id))objc_msgSend)(D, @selector(modelWithMILText:weights:optionsPlist:), mil, @{}, nil);
        id model = ((id(*)(Class,SEL,id))objc_msgSend)(I, @selector(inMemoryModelWithDescriptor:), desc);
        id hx = ((id(*)(id,SEL))objc_msgSend)(model, @selector(hexStringIdentifier));
        NSString *td = [NSTemporaryDirectory() stringByAppendingPathComponent:hx];
        [[NSFileManager defaultManager] createDirectoryAtPath:[td stringByAppendingPathComponent:@"weights"]
            withIntermediateDirectories:YES attributes:nil error:nil];
        [mil writeToFile:[td stringByAppendingPathComponent:@"model.mil"] atomically:YES];
        NSError *e = nil;
        ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(model, @selector(compileWithQoS:options:error:), 21, @{}, &e);
        ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(model, @selector(loadWithQoS:options:error:), 21, @{}, &e);
        id aneModel = ((id(*)(id,SEL))objc_msgSend)(model, @selector(model));
        IOSurfaceRef ioIn = make_surf(inB), ioOut = make_surf(outB);
        // Fill input
        _Float16 *inp = (void*)IOSurfaceGetBaseAddress(ioIn);
        for (size_t i = 0; i < inB/2; i++) inp[i] = (_Float16)(0.01f*(i%100));
        id wI = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_AIO, @selector(objectWithIOSurface:), ioIn);
        id wO = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_AIO, @selector(objectWithIOSurface:), ioOut);
        id req = ((id(*)(Class,SEL,id,id,id,id,id,id,id))objc_msgSend)(g_AR,
            @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:),
            @[wI], @[@0], @[wO], @[@0], nil, nil, @0);
        printf("Compiled OK\n");

        // Step 1: Normal eval — get reference output
        e = nil;
        ((BOOL(*)(id,SEL,id,id,id,unsigned int,NSError**))objc_msgSend)(
            g_client, @selector(doEvaluateDirectWithModel:options:request:qos:error:),
            aneModel, @{}, req, 21, &e);
        _Float16 *out = (_Float16*)IOSurfaceGetBaseAddress(ioOut);
        printf("Normal eval output[0..3]: %.4f %.4f %.4f %.4f\n",
               (float)out[0], (float)out[1], (float)out[2], (float)out[3]);
        float ref0 = (float)out[0], ref1 = (float)out[1], ref2 = (float)out[2], ref3 = (float)out[3];

        // Step 2: Zero the output surface
        memset(IOSurfaceGetBaseAddress(ioOut), 0, outB);
        printf("Zeroed output: %.4f %.4f %.4f %.4f\n",
               (float)out[0], (float)out[1], (float)out[2], (float)out[3]);

        // Step 3: Two-phase dispatch
        id ibr = ((id(*)(id,SEL,unsigned int,unsigned int,unsigned long long,unsigned int))objc_msgSend)(
            [IBR alloc],
            @selector(initInputsProcedureIndex:inputBufferInfoIndex:inputFreeValue:executionDelay:),
            (unsigned int)0, (unsigned int)0, (unsigned long long)0, (unsigned int)0);
        id ose = ((id(*)(id,SEL,unsigned int,unsigned int,unsigned long long,BOOL,BOOL))objc_msgSend)(
            [OSE alloc],
            @selector(initOutputSetWithProcedureIndex:setIndex:signalValue:signalNotRequired:isOpenLoop:),
            (unsigned int)0, (unsigned int)0, (unsigned long long)1, NO, NO);

        // Phase 1: buffersReady
        e = nil;
        long ret1 = ((long(*)(id,SEL,id,id,id,unsigned int,NSError**))objc_msgSend)(
            g_client, @selector(doBuffersReadyWithModel:inputBuffers:options:qos:error:),
            aneModel, ibr, @{}, 21, &e);
        printf("doBuffersReady: ret=%ld err=%s\n", ret1, e ? [[e description] UTF8String] : "none");

        // Phase 2: enqueueSets
        e = nil;
        long ret2 = ((long(*)(id,SEL,id,id,id,unsigned int,NSError**))objc_msgSend)(
            g_client, @selector(doEnqueueSetsWithModel:outputSet:options:qos:error:),
            aneModel, ose, @{}, 21, &e);
        printf("doEnqueueSets: ret=%ld err=%s\n", ret2, e ? [[e description] UTF8String] : "none");

        // Give ANE time to execute
        usleep(5000); // 5ms

        // Step 4: Check output
        printf("Two-phase output[0..3]: %.4f %.4f %.4f %.4f\n",
               (float)out[0], (float)out[1], (float)out[2], (float)out[3]);

        BOOL match = (fabsf((float)out[0] - ref0) < 0.01f &&
                      fabsf((float)out[1] - ref1) < 0.01f);
        printf("Output matches reference: %s\n", match ? "YES — TWO-PHASE DISPATCH WORKS!" : "NO — output is stale/zero");

        if (!match) {
            // Try with longer wait
            usleep(50000); // 50ms
            printf("After 50ms wait: %.4f %.4f %.4f %.4f\n",
                   (float)out[0], (float)out[1], (float)out[2], (float)out[3]);
            match = (fabsf((float)out[0] - ref0) < 0.01f);
            printf("Match now: %s\n", match ? "YES" : "NO");
        }

        // Benchmark if it works
        if (match) {
            printf("\n=== BENCHMARK: Two-phase vs normal dispatch ===\n");
            int N = 500;
            // Warmup
            for (int i = 0; i < 20; i++) {
                ((long(*)(id,SEL,id,id,id,unsigned int,NSError**))objc_msgSend)(
                    g_client, @selector(doBuffersReadyWithModel:inputBuffers:options:qos:error:),
                    aneModel, ibr, @{}, 21, &e);
                ((long(*)(id,SEL,id,id,id,unsigned int,NSError**))objc_msgSend)(
                    g_client, @selector(doEnqueueSetsWithModel:outputSet:options:qos:error:),
                    aneModel, ose, @{}, 21, &e);
            }
            uint64_t t0 = mach_absolute_time();
            for (int i = 0; i < N; i++) {
                ((long(*)(id,SEL,id,id,id,unsigned int,NSError**))objc_msgSend)(
                    g_client, @selector(doBuffersReadyWithModel:inputBuffers:options:qos:error:),
                    aneModel, ibr, @{}, 21, &e);
                ((long(*)(id,SEL,id,id,id,unsigned int,NSError**))objc_msgSend)(
                    g_client, @selector(doEnqueueSetsWithModel:outputSet:options:qos:error:),
                    aneModel, ose, @{}, 21, &e);
            }
            double tp_ms = ms_t(mach_absolute_time() - t0);

            // Baseline
            for (int i = 0; i < 20; i++) {
                ((BOOL(*)(id,SEL,id,id,id,unsigned int,NSError**))objc_msgSend)(
                    g_client, @selector(doEvaluateDirectWithModel:options:request:qos:error:),
                    aneModel, @{}, req, 21, &e);
            }
            t0 = mach_absolute_time();
            for (int i = 0; i < N; i++) {
                ((BOOL(*)(id,SEL,id,id,id,unsigned int,NSError**))objc_msgSend)(
                    g_client, @selector(doEvaluateDirectWithModel:options:request:qos:error:),
                    aneModel, @{}, req, 21, &e);
            }
            double base_ms = ms_t(mach_absolute_time() - t0);

            printf("Two-phase: %.1fms (%.3f ms/eval)\n", tp_ms, tp_ms/N);
            printf("Baseline:  %.1fms (%.3f ms/eval)\n", base_ms, base_ms/N);
            printf("Speedup: %.2fx\n", base_ms/tp_ms);
        }

        printf("\nDone.\n");
        CFRelease(ioIn); CFRelease(ioOut);
    }
    return 0;
}
