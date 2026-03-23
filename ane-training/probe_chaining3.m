// probe_chaining3.m — Use _ANEBuffer with symbolIndex for chaining
// Also explore _ANEProgramForEvaluation (direct local eval bypassing XPC)
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
    [[NSFileManager defaultManager] createDirectoryAtPath:[td stringByAppendingPathComponent:@"weights"] withIntermediateDirectories:YES attributes:nil error:nil];
    [mil writeToFile:[td stringByAppendingPathComponent:@"model.mil"] atomically:YES];
    NSError *e = nil;
    ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(k.model, @selector(compileWithQoS:options:error:), 21, @{}, &e);
    ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(k.model, @selector(loadWithQoS:options:error:), 21, @{}, &e);
    k.aneModel = ((id(*)(id,SEL))objc_msgSend)(k.model, @selector(model));
    k.ioIn = make_surf(in_bytes);
    k.ioOut = make_surf(out_bytes);
    _Float16 *p = (void*)IOSurfaceGetBaseAddress(k.ioIn);
    for (size_t i = 0; i < in_bytes/2; i++) p[i] = (_Float16)(0.01f * (i % 100));
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
        ane_init();

        printf("=== ANE Chaining with _ANEBuffer ===\n\n");

        Kern kA = compile_kern(256, 256, 64);
        NSError *e = nil;
        int N = 500;

        // === Baseline eval ===
        for (int i = 0; i < 20; i++) {
            ((BOOL(*)(id,SEL,id,id,id,NSError**))objc_msgSend)(
                g_client, @selector(evaluateRealTimeWithModel:options:request:error:),
                kA.aneModel, @{}, kA.request, &e);
        }
        uint64_t t0 = mach_absolute_time();
        for (int i = 0; i < N; i++) {
            ((BOOL(*)(id,SEL,id,id,id,NSError**))objc_msgSend)(
                g_client, @selector(evaluateRealTimeWithModel:options:request:error:),
                kA.aneModel, @{}, kA.request, &e);
        }
        double baseline_ms = ms_t(mach_absolute_time() - t0);
        printf("Baseline: %.1f ms (%5.3f ms/eval)\n\n", baseline_ms, baseline_ms/N);

        // === Try _ANEBuffer wrapping ===
        Class BufCls = NSClassFromString(@"_ANEBuffer");
        id ioIn = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_AIO, @selector(objectWithIOSurface:), kA.ioIn);
        id ioOut = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_AIO, @selector(objectWithIOSurface:), kA.ioOut);

        printf("=== _ANEBuffer creation ===\n");
        id bufIn = nil, bufOut = nil;
        @try {
            bufIn = ((id(*)(id,SEL,id,id,id))objc_msgSend)(
                [BufCls alloc], @selector(initWithIOSurfaceObject:symbolIndex:source:),
                ioIn, @0, @"input");
            printf("  bufIn: %s\n", bufIn ? [[bufIn description] UTF8String] : "nil");
        } @catch (NSException *ex) {
            printf("  bufIn threw: %s\n", [[ex description] UTF8String]);
        }
        @try {
            bufOut = ((id(*)(id,SEL,id,id,id))objc_msgSend)(
                [BufCls alloc], @selector(initWithIOSurfaceObject:symbolIndex:source:),
                ioOut, @0, @"output");
            printf("  bufOut: %s\n", bufOut ? [[bufOut description] UTF8String] : "nil");
        } @catch (NSException *ex) {
            printf("  bufOut threw: %s\n", [[ex description] UTF8String]);
        }

        // === Try chaining with _ANEBuffer ===
        if (bufIn && bufOut) {
            printf("\n=== Chaining with _ANEBuffer ===\n");
            Class ChainingReqCls = NSClassFromString(@"_ANEChainingRequest");

            // Try chaining with _ANEBuffer directly
            @try {
                id chainReq = ((id(*)(id,SEL,id,id,id,id,id,id,id,NSUInteger,NSUInteger))objc_msgSend)(
                    [ChainingReqCls alloc],
                    @selector(initWithInputs:outputs:lbInputSymbolId:lbOutputSymbolId:procedureIndex:signalEvents:transactionHandle:fwEnqueueDelay:memoryPoolId:),
                    bufIn,       // input: _ANEBuffer
                    @[bufOut],   // outputs: array of _ANEBuffer
                    @(-1),       // lbInputSymbolId (-1 = no loopback)
                    @(-1),       // lbOutputSymbolId
                    @0,          // procedureIndex
                    nil,         // signalEvents
                    nil,         // transactionHandle
                    (NSUInteger)0, // fwEnqueueDelay
                    (NSUInteger)0  // memoryPoolId
                );
                printf("  chainReq: %s\n", chainReq ? "OK" : "nil");
                if (chainReq) {
                    printf("    desc: %s\n", [[chainReq description] UTF8String]);

                    BOOL ok = ((BOOL(*)(id,SEL,id,id,id,unsigned int,NSError**))objc_msgSend)(
                        g_client, @selector(prepareChainingWithModel:options:chainingReq:qos:error:),
                        kA.aneModel, @{}, chainReq, 21, &e);
                    printf("  prepareChaining: %s\n", ok ? "OK" : "FAIL");
                    if (!ok && e) printf("    Error: %s\n", [[e description] UTF8String]);
                    e = nil;

                    // If it worked, try evaluating
                    if (ok) {
                        printf("  Post-chain eval...\n");
                        BOOL ok2 = ((BOOL(*)(id,SEL,id,id,id,NSError**))objc_msgSend)(
                            g_client, @selector(evaluateRealTimeWithModel:options:request:error:),
                            kA.aneModel, @{}, kA.request, &e);
                        printf("  post-chain eval: %s\n", ok2 ? "OK" : "FAIL");
                    }
                }
            } @catch (NSException *ex) {
                printf("  chainReq threw: %s\n", [[ex description] UTF8String]);
            }
        }

        // === Explore _ANEProgramForEvaluation (local eval bypass XPC) ===
        printf("\n=== _ANEProgramForEvaluation (local eval) ===\n");
        Class PFECls = NSClassFromString(@"_ANEProgramForEvaluation");
        Class DCCls = NSClassFromString(@"_ANEDeviceController");

        // Try to create a device controller
        id deviceCtrl = nil;
        @try {
            // initWithProgramHandle:priviledged:
            // We need a programHandle — try from aneModel
            id progHandle = nil;
            @try {
                progHandle = ((id(*)(id,SEL))objc_msgSend)(kA.aneModel, @selector(programHandle));
            } @catch (NSException *ex) {}

            if (progHandle) {
                printf("  programHandle: %s (%s)\n", [[progHandle description] UTF8String],
                       class_getName([progHandle class]));

                deviceCtrl = ((id(*)(id,SEL,id,BOOL))objc_msgSend)(
                    [DCCls alloc], @selector(initWithProgramHandle:priviledged:),
                    progHandle, YES);
                printf("  deviceCtrl: %s\n", deviceCtrl ? "OK" : "nil");
            } else {
                printf("  programHandle: nil\n");
            }
        } @catch (NSException *ex) {
            printf("  threw: %s\n", [[ex description] UTF8String]);
        }

        // === Explore _ANEInputBuffersReady with proper init ===
        printf("\n=== _ANEInputBuffersReady proper init ===\n");
        Class IBRCls = NSClassFromString(@"_ANEInputBuffersReady");
        @try {
            // initInputsProcedureIndex:inputBufferInfoIndex:inputFreeValue:executionDelay:
            id ibr = ((id(*)(id,SEL,id,id,id,id))objc_msgSend)(
                [IBRCls alloc],
                @selector(initInputsProcedureIndex:inputBufferInfoIndex:inputFreeValue:executionDelay:),
                @0,  // procedureIndex
                @0,  // inputBufferInfoIndex
                @0,  // inputFreeValue
                @0   // executionDelay
            );
            printf("  IBR: %s\n", ibr ? "OK" : "nil");
            if (ibr) printf("    desc: %s\n", [[ibr description] UTF8String]);

            // Try passing it to buffersReady
            if (ibr) {
                BOOL ok = ((BOOL(*)(id,SEL,id,id,id,unsigned int,NSError**))objc_msgSend)(
                    g_client, @selector(buffersReadyWithModel:inputBuffers:options:qos:error:),
                    kA.aneModel, ibr, @{}, 21, &e);
                printf("  buffersReady(IBR): %s\n", ok ? "OK" : "FAIL");
                if (!ok && e) printf("    Error: %s\n", [[e description] UTF8String]);
                e = nil;
            }
        } @catch (NSException *ex) {
            printf("  threw: %s\n", [[ex description] UTF8String]);
        }

        // === _ANEOutputSetEnqueue proper init ===
        printf("\n=== _ANEOutputSetEnqueue proper init ===\n");
        Class OSECls = NSClassFromString(@"_ANEOutputSetEnqueue");
        @try {
            // initOutputSetWithProcedureIndex:setIndex:signalValue:signalNotRequired:isOpenLoop:
            id ose = ((id(*)(id,SEL,id,id,id,BOOL,BOOL))objc_msgSend)(
                [OSECls alloc],
                @selector(initOutputSetWithProcedureIndex:setIndex:signalValue:signalNotRequired:isOpenLoop:),
                @0,    // procedureIndex
                @0,    // setIndex
                @1,    // signalValue
                NO,    // signalNotRequired
                NO     // isOpenLoop
            );
            printf("  OSE: %s\n", ose ? "OK" : "nil");
            if (ose) printf("    desc: %s\n", [[ose description] UTF8String]);

            // Try passing to enqueueSets
            if (ose) {
                BOOL ok = ((BOOL(*)(id,SEL,id,id,id,unsigned int,NSError**))objc_msgSend)(
                    g_client, @selector(enqueueSetsWithModel:outputSet:options:qos:error:),
                    kA.aneModel, ose, @{}, 21, &e);
                printf("  enqueueSets(OSE): %s\n", ok ? "OK" : "FAIL");
                if (!ok && e) printf("    Error: %s\n", [[e description] UTF8String]);
                e = nil;
            }
        } @catch (NSException *ex) {
            printf("  threw: %s\n", [[ex description] UTF8String]);
        }

        // === Try evaluateWithModel (the evaluateWithQoS but via client) ===
        printf("\n=== evaluateWithModel ===\n");
        @try {
            t0 = mach_absolute_time();
            for (int i = 0; i < N; i++) {
                ((BOOL(*)(id,SEL,id,id,id,unsigned int,NSError**))objc_msgSend)(
                    g_client, @selector(evaluateWithModel:options:request:qos:error:),
                    kA.aneModel, @{}, kA.request, 21, &e);
            }
            double ewm_ms = ms_t(mach_absolute_time() - t0);
            printf("  evaluateWithModel: %.1f ms (%5.3f ms/eval) speedup=%.2fx\n",
                   ewm_ms, ewm_ms/N, baseline_ms/ewm_ms);
        } @catch (NSException *ex) {
            printf("  threw: %s\n", [[ex description] UTF8String]);
        }

        // === doEvaluateDirectWithModel ===
        printf("\n=== doEvaluateDirectWithModel ===\n");
        @try {
            t0 = mach_absolute_time();
            for (int i = 0; i < N; i++) {
                ((BOOL(*)(id,SEL,id,id,id,unsigned int,NSError**))objc_msgSend)(
                    g_client, @selector(doEvaluateDirectWithModel:options:request:qos:error:),
                    kA.aneModel, @{}, kA.request, 21, &e);
            }
            double ded_ms = ms_t(mach_absolute_time() - t0);
            printf("  doEvaluateDirect: %.1f ms (%5.3f ms/eval) speedup=%.2fx\n",
                   ded_ms, ded_ms/N, baseline_ms/ded_ms);
        } @catch (NSException *ex) {
            printf("  threw: %s\n", [[ex description] UTF8String]);
        }

        // Cleanup
        ((BOOL(*)(id,SEL,unsigned int,NSError**))objc_msgSend)(kA.model, @selector(unloadWithQoS:error:), 21, &e);
        CFRelease(kA.ioIn); CFRelease(kA.ioOut);

        printf("\n=== Done ===\n");
    }
    return 0;
}
