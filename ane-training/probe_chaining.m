// probe_chaining.m — Test ANE chaining API for pipelined kernel execution
// Goal: reduce dispatch overhead by chaining multiple evals into a firmware-managed pipeline
#import <Foundation/Foundation.h>
#import <IOSurface/IOSurface.h>
#import <objc/runtime.h>
#import <objc/message.h>
#import <dlfcn.h>
#import <mach/mach_time.h>

static mach_timebase_info_data_t g_tb;
static double ms_t(uint64_t t) { return (double)t * g_tb.numer / g_tb.denom / 1e6; }

static id g_client;
static Class g_D, g_I, g_AR, g_AIO;

static void ane_init(void) {
    dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine", RTLD_NOW);
    g_D  = NSClassFromString(@"_ANEInMemoryModelDescriptor");
    g_I  = NSClassFromString(@"_ANEInMemoryModel");
    g_AR = NSClassFromString(@"_ANERequest");
    g_AIO= NSClassFromString(@"_ANEIOSurfaceObject");
    g_client = ((id(*)(Class,SEL))objc_msgSend)(NSClassFromString(@"_ANEClient"), @selector(sharedConnection));
}

static IOSurfaceRef make_surf(size_t b) {
    return IOSurfaceCreate((__bridge CFDictionaryRef)@{
        (id)kIOSurfaceWidth:@(b),(id)kIOSurfaceHeight:@1,
        (id)kIOSurfaceBytesPerElement:@1,(id)kIOSurfaceBytesPerRow:@(b),
        (id)kIOSurfaceAllocSize:@(b),(id)kIOSurfacePixelFormat:@0});
}

// Matmul kernel: [1, IC, 1, SEQ+OC] -> [1, OC, 1, SEQ]
static NSData *gen_matmul(int IC, int OC, int SEQ) {
    int SP = SEQ + OC;
    NSMutableString *m = [NSMutableString string];
    [m appendString:@"program(1.3)\n[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"},{\"coremlc-version\", \"3505.4.1\"},{\"coremltools-component-milinternal\", \"\"},{\"coremltools-version\", \"9.0\"}})]\n{\n"];
    [m appendFormat:@"    func main<ios18>(tensor<fp16, [1, %d, 1, %d]> x) {\n", IC, SP];
    [m appendString:@"        tensor<int32, [4]> ba = const()[name=string(\"ba\"), val=tensor<int32, [4]>([0,0,0,0])];\n"];
    [m appendFormat:@"        tensor<int32, [4]> sa = const()[name=string(\"sa\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n", IC, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> act = slice_by_size(x=x,begin=ba,size=sa)[name=string(\"act\")];\n", IC, SEQ];
    [m appendFormat:@"        tensor<int32, [4]> bw = const()[name=string(\"bw\"), val=tensor<int32, [4]>([0,0,0,%d])];\n", SEQ];
    [m appendFormat:@"        tensor<int32, [4]> sw = const()[name=string(\"sw\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n", IC, OC];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> wt = slice_by_size(x=x,begin=bw,size=sw)[name=string(\"wt\")];\n", IC, OC];
    [m appendFormat:@"        tensor<int32, [4]> ra = const()[name=string(\"ra\"), val=tensor<int32, [4]>([1,1,%d,%d])];\n", IC, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> a2 = reshape(shape=ra,x=act)[name=string(\"a2\")];\n", IC, SEQ];
    [m appendString:@"        tensor<int32, [4]> pm = const()[name=string(\"pm\"), val=tensor<int32, [4]>([0,1,3,2])];\n"];
    [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> a3 = transpose(perm=pm,x=a2)[name=string(\"a3\")];\n", SEQ, IC];
    [m appendFormat:@"        tensor<int32, [4]> rw = const()[name=string(\"rw\"), val=tensor<int32, [4]>([1,1,%d,%d])];\n", IC, OC];
    [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> W = reshape(shape=rw,x=wt)[name=string(\"W\")];\n", IC, OC];
    [m appendString:@"        bool bF = const()[name=string(\"bF\"), val=bool(false)];\n"];
    [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> yh = matmul(transpose_x=bF,transpose_y=bF,x=a3,y=W)[name=string(\"yh\")];\n", SEQ, OC];
    [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> yt = transpose(perm=pm,x=yh)[name=string(\"yt\")];\n", OC, SEQ];
    [m appendFormat:@"        tensor<int32, [4]> ro = const()[name=string(\"ro\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n", OC, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> y = reshape(shape=ro,x=yt)[name=string(\"y\")];\n", OC, SEQ];
    [m appendString:@"    } -> (y);\n}\n"];
    return [m dataUsingEncoding:NSUTF8StringEncoding];
}

static void dump_class_methods(Class cls, const char *label) {
    printf("\n--- %s class methods ---\n", label);
    unsigned int count;
    Method *methods = class_copyMethodList(object_getClass(cls), &count);
    for (unsigned int i = 0; i < count; i++) {
        printf("  + %s  %s\n",
               sel_getName(method_getName(methods[i])),
               method_getTypeEncoding(methods[i]));
    }
    free(methods);
}

static void dump_instance_methods(Class cls, const char *label) {
    printf("\n--- %s instance methods ---\n", label);
    unsigned int count;
    Method *methods = class_copyMethodList(cls, &count);
    for (unsigned int i = 0; i < count; i++) {
        printf("  - %s  %s\n",
               sel_getName(method_getName(methods[i])),
               method_getTypeEncoding(methods[i]));
    }
    free(methods);
}

static void dump_properties(Class cls, const char *label) {
    printf("\n--- %s properties ---\n", label);
    unsigned int count;
    objc_property_t *props = class_copyPropertyList(cls, &count);
    for (unsigned int i = 0; i < count; i++) {
        printf("  %s  [%s]\n",
               property_getName(props[i]),
               property_getAttributes(props[i]));
    }
    free(props);
}

static id compile_and_load(NSData *mil) {
    id desc = ((id(*)(Class,SEL,id,id,id))objc_msgSend)(g_D,@selector(modelWithMILText:weights:optionsPlist:),mil,@{},nil);
    id mdl = ((id(*)(Class,SEL,id))objc_msgSend)(g_I,@selector(inMemoryModelWithDescriptor:),desc);
    id hx = ((id(*)(id,SEL))objc_msgSend)(mdl,@selector(hexStringIdentifier));
    NSString *td = [NSTemporaryDirectory() stringByAppendingPathComponent:hx];
    [[NSFileManager defaultManager] createDirectoryAtPath:[td stringByAppendingPathComponent:@"weights"] withIntermediateDirectories:YES attributes:nil error:nil];
    [mil writeToFile:[td stringByAppendingPathComponent:@"model.mil"] atomically:YES];
    NSError *e = nil;
    BOOL ok = ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(mdl,@selector(compileWithQoS:options:error:),21,@{},&e);
    if (!ok) { printf("  Compile FAIL: %s\n", [[e description] UTF8String]); return nil; }
    ok = ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(mdl,@selector(loadWithQoS:options:error:),21,@{},&e);
    if (!ok) { printf("  Load FAIL: %s\n", [[e description] UTF8String]); return nil; }
    return mdl;
}

int main(int argc, char *argv[]) {
    @autoreleasepool {
        setbuf(stdout, NULL);
        mach_timebase_info(&g_tb);
        ane_init();

        printf("=== ANE Chaining & IOSurfaceSharedEvent Probe ===\n\n");

        // ===== 1. Introspect chaining classes =====
        Class chainReqClass = NSClassFromString(@"_ANEChainingRequest");
        Class outSetClass = NSClassFromString(@"_ANEOutputSetEnqueue");
        Class sharedEvtsClass = NSClassFromString(@"_ANESharedEvents");
        Class sigEventClass = NSClassFromString(@"_ANESharedSignalEvent");
        Class waitEventClass = NSClassFromString(@"_ANESharedWaitEvent");
        Class ioSharedEventClass = NSClassFromString(@"IOSurfaceSharedEvent");

        if (chainReqClass) {
            dump_class_methods(chainReqClass, "_ANEChainingRequest");
            dump_instance_methods(chainReqClass, "_ANEChainingRequest");
            dump_properties(chainReqClass, "_ANEChainingRequest");
        } else {
            printf("_ANEChainingRequest: NOT FOUND\n");
        }

        if (outSetClass) {
            dump_class_methods(outSetClass, "_ANEOutputSetEnqueue");
            dump_instance_methods(outSetClass, "_ANEOutputSetEnqueue");
            dump_properties(outSetClass, "_ANEOutputSetEnqueue");
        } else {
            printf("_ANEOutputSetEnqueue: NOT FOUND\n");
        }

        if (sharedEvtsClass) {
            dump_class_methods(sharedEvtsClass, "_ANESharedEvents");
            dump_instance_methods(sharedEvtsClass, "_ANESharedEvents");
            dump_properties(sharedEvtsClass, "_ANESharedEvents");
        }
        if (sigEventClass) {
            dump_class_methods(sigEventClass, "_ANESharedSignalEvent");
            dump_instance_methods(sigEventClass, "_ANESharedSignalEvent");
            dump_properties(sigEventClass, "_ANESharedSignalEvent");
        }
        if (waitEventClass) {
            dump_class_methods(waitEventClass, "_ANESharedWaitEvent");
            dump_instance_methods(waitEventClass, "_ANESharedWaitEvent");
            dump_properties(waitEventClass, "_ANESharedWaitEvent");
        }
        if (ioSharedEventClass) {
            dump_instance_methods(ioSharedEventClass, "IOSurfaceSharedEvent");
            dump_properties(ioSharedEventClass, "IOSurfaceSharedEvent");
        }

        // ===== 2. QoS mapper values =====
        Class qosMapper = NSClassFromString(@"_ANEQoSMapper");
        if (qosMapper) {
            printf("\n--- _ANEQoSMapper ---\n");
            @try {
                unsigned int rtQoS = ((unsigned int(*)(Class,SEL))objc_msgSend)(qosMapper, @selector(aneRealTimeTaskQoS));
                unsigned int uiQoS = ((unsigned int(*)(Class,SEL))objc_msgSend)(qosMapper, @selector(aneUserInteractiveTaskQoS));
                unsigned int defQoS = ((unsigned int(*)(Class,SEL))objc_msgSend)(qosMapper, @selector(aneDefaultTaskQoS));
                printf("  realTime=%u userInteractive=%u default=%u\n", rtQoS, uiQoS, defQoS);
                int rtPri = ((int(*)(Class,SEL))objc_msgSend)(qosMapper, @selector(realTimeProgramPriority));
                uint64_t rtQIdx = ((uint64_t(*)(Class,SEL))objc_msgSend)(qosMapper, @selector(realTimeQueueIndex));
                printf("  realTimePriority=%d realTimeQueueIndex=%llu\n", rtPri, rtQIdx);
                for (unsigned int q = 0; q <= 33; q++) {
                    @try {
                        uint64_t qi = ((uint64_t(*)(Class,SEL,unsigned int))objc_msgSend)(qosMapper, @selector(queueIndexForQoS:), q);
                        int pri = ((int(*)(Class,SEL,unsigned int))objc_msgSend)(qosMapper, @selector(programPriorityForQoS:), q);
                        printf("  QoS %2u -> queue=%llu priority=%d\n", q, qi, pri);
                    } @catch (NSException *ex) {}
                }
            } @catch (NSException *ex) {
                printf("  Exception: %s\n", [[ex reason] UTF8String]);
            }
        }

        // ===== 3. Compile and baseline benchmark =====
        printf("\n--- Compile test kernel ---\n");
        int IC=256, OC=256, SEQ=64;
        id mdlA = compile_and_load(gen_matmul(IC, OC, SEQ));
        if (!mdlA) return 1;
        printf("  OK (matmul %dx%d, seq=%d)\n", IC, OC, SEQ);

        id aneModelA = ((id(*)(id,SEL))objc_msgSend)(mdlA, @selector(model));
        size_t in_bytes = (size_t)IC*(SEQ+OC)*2;
        size_t out_bytes = (size_t)OC*SEQ*2;
        IOSurfaceRef ioIn = make_surf(in_bytes);
        IOSurfaceRef ioOut = make_surf(out_bytes);
        _Float16 *inp = (_Float16*)IOSurfaceGetBaseAddress(ioIn);
        for (size_t i=0; i<in_bytes/2; i++) inp[i]=(_Float16)(0.01f*(i%100));

        id wI=((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_AIO,@selector(objectWithIOSurface:),ioIn);
        id wO=((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_AIO,@selector(objectWithIOSurface:),ioOut);
        NSError *e = nil;

        id reqA = ((id(*)(Class,SEL,id,id,id,id,id,id))objc_msgSend)(g_AR,
            @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:procedureIndex:),
            @[wI],@[@0],@[wO],@[@0],nil,@0);

        printf("\n--- Sequential eval baseline ---\n");
        for (int i=0;i<50;i++)
            ((BOOL(*)(id,SEL,id,id,id,unsigned int,NSError**))objc_msgSend)(g_client,
                @selector(doEvaluateDirectWithModel:options:request:qos:error:),aneModelA,@{},reqA,21,&e);
        uint64_t t0 = mach_absolute_time();
        for (int i=0;i<200;i++)
            ((BOOL(*)(id,SEL,id,id,id,unsigned int,NSError**))objc_msgSend)(g_client,
                @selector(doEvaluateDirectWithModel:options:request:qos:error:),aneModelA,@{},reqA,21,&e);
        double seq_ms = ms_t(mach_absolute_time()-t0);
        printf("  200 evals: %.1f ms (%.3f ms/eval)\n", seq_ms, seq_ms/200);

        // ===== 4. IOSurfaceSharedEvent =====
        printf("\n--- IOSurfaceSharedEvent ---\n");
        if (ioSharedEventClass) {
            @try {
                id shEvt = ((id(*)(id,SEL,uint64_t))objc_msgSend)([ioSharedEventClass alloc], @selector(initWithOptions:), (uint64_t)0);
                if (!shEvt) shEvt = [[ioSharedEventClass alloc] init];
                if (shEvt) {
                    printf("  Created: %s\n", [[shEvt description] UTF8String]);
                    uint64_t sv = ((uint64_t(*)(id,SEL))objc_msgSend)(shEvt, @selector(signaledValue));
                    printf("  signaledValue: %llu\n", sv);
                    ((void(*)(id,SEL,uint64_t))objc_msgSend)(shEvt, @selector(setSignaledValue:), (uint64_t)42);
                    sv = ((uint64_t(*)(id,SEL))objc_msgSend)(shEvt, @selector(signaledValue));
                    printf("  After set(42): %llu\n", sv);
                    BOOL w1 = ((BOOL(*)(id,SEL,uint64_t,uint64_t))objc_msgSend)(shEvt,
                        @selector(waitUntilSignaledValue:timeoutMS:), (uint64_t)42, (uint64_t)10);
                    printf("  wait(42,10ms): %d\n", w1);
                    BOOL w2 = ((BOOL(*)(id,SEL,uint64_t,uint64_t))objc_msgSend)(shEvt,
                        @selector(waitUntilSignaledValue:timeoutMS:), (uint64_t)100, (uint64_t)1);
                    printf("  wait(100,1ms): %d (expect 0/timeout)\n", w2);

                    // Benchmark signal+wait overhead
                    t0 = mach_absolute_time();
                    for (int i=0;i<10000;i++) {
                        ((void(*)(id,SEL,uint64_t))objc_msgSend)(shEvt, @selector(setSignaledValue:), (uint64_t)(i+1));
                    }
                    double sig_ms = ms_t(mach_absolute_time()-t0);
                    printf("  10000 signals: %.2f ms (%.3f us/signal)\n", sig_ms, sig_ms*1000/10000);

                    t0 = mach_absolute_time();
                    for (int i=0;i<10000;i++) {
                        ((BOOL(*)(id,SEL,uint64_t,uint64_t))objc_msgSend)(shEvt,
                            @selector(waitUntilSignaledValue:timeoutMS:), (uint64_t)10000, (uint64_t)0);
                    }
                    double wait_ms = ms_t(mach_absolute_time()-t0);
                    printf("  10000 waits: %.2f ms (%.3f us/wait)\n", wait_ms, wait_ms*1000/10000);
                } else {
                    printf("  Failed to create\n");
                }
            } @catch (NSException *ex) {
                printf("  Exception: %s\n", [[ex reason] UTF8String]);
            }
        }

        // ===== 5. ANE shared events construction =====
        printf("\n--- ANE Shared Events ---\n");
        if (sigEventClass) {
            @try {
                id se = [[sigEventClass alloc] init];
                printf("  _ANESharedSignalEvent alloc/init: %s\n", se ? [[se description] UTF8String] : "nil");
            } @catch (NSException *ex) {
                printf("  Exception: %s\n", [[ex reason] UTF8String]);
            }
        }
        if (sharedEvtsClass) {
            @try {
                id se = [[sharedEvtsClass alloc] init];
                printf("  _ANESharedEvents alloc/init: %s\n", se ? [[se description] UTF8String] : "nil");
            } @catch (NSException *ex) {
                printf("  Exception: %s\n", [[ex reason] UTF8String]);
            }
        }

        // ===== 6. Chaining API =====
        printf("\n--- Chaining API ---\n");
        if (chainReqClass) {
            @try {
                id cr = [[chainReqClass alloc] init];
                printf("  alloc/init: %s\n", cr ? "non-nil" : "nil");
                if (cr) {
                    // Set properties
                    @try { ((void(*)(id,SEL,id))objc_msgSend)(cr, @selector(setInputBuffer:), @[wI]); printf("    setInputBuffer: OK\n"); }
                    @catch (NSException *ex) { printf("    setInputBuffer: %s\n", [[ex reason] UTF8String]); }

                    @try { ((void(*)(id,SEL,id))objc_msgSend)(cr, @selector(setProcedureIndex:), @0); printf("    setProcedureIndex: OK\n"); }
                    @catch (NSException *ex) { printf("    setProcedureIndex: %s\n", [[ex reason] UTF8String]); }

                    @try { ((void(*)(id,SEL,id))objc_msgSend)(cr, @selector(setOutputSets:), @[@[wO]]); printf("    setOutputSets: OK\n"); }
                    @catch (NSException *ex) { printf("    setOutputSets: %s\n", [[ex reason] UTF8String]); }

                    // Try prepareChainingWithModel
                    printf("\n  prepareChainingWithModel...\n");
                    e = nil;
                    BOOL ok = ((BOOL(*)(id,SEL,id,id,id,unsigned int,NSError**))objc_msgSend)(g_client,
                        @selector(prepareChainingWithModel:options:chainingReq:qos:error:),
                        aneModelA, @{}, cr, 21, &e);
                    printf("    result: %d\n", ok);
                    if (!ok && e) printf("    error: %s\n", [[e description] UTF8String]);

                    // Also try doPrepareChainingWithModel (direct path)
                    printf("\n  doPrepareChainingWithModel...\n");
                    e = nil;
                    ok = ((BOOL(*)(id,SEL,id,id,id,unsigned int,NSError**))objc_msgSend)(g_client,
                        @selector(doPrepareChainingWithModel:options:chainingReq:qos:error:),
                        aneModelA, @{}, cr, 21, &e);
                    printf("    result: %d\n", ok);
                    if (!ok && e) printf("    error: %s\n", [[e description] UTF8String]);
                }
            } @catch (NSException *ex) {
                printf("  Exception: %s\n", [[ex reason] UTF8String]);
            }
        }

        // ===== 7. Enqueue path =====
        printf("\n--- Enqueue path ---\n");
        @try {
            e = nil;
            BOOL ok = ((BOOL(*)(id,SEL,id,id,id,unsigned int,NSError**))objc_msgSend)(g_client,
                @selector(buffersReadyWithModel:inputBuffers:options:qos:error:),
                aneModelA, @[wI], @{}, 21, &e);
            printf("  buffersReady: %d", ok);
            if (!ok && e) printf(" error: %s", [[e description] UTF8String]);
            printf("\n");
        } @catch (NSException *ex) {
            printf("  buffersReady exception: %s\n", [[ex reason] UTF8String]);
        }

        if (outSetClass) {
            @try {
                id outSet = [[outSetClass alloc] init];
                if (outSet) {
                    @try { ((void(*)(id,SEL,id))objc_msgSend)(outSet, @selector(setOutputArray:), @[wO]); } @catch (NSException *ex) {}
                    @try { ((void(*)(id,SEL,id))objc_msgSend)(outSet, @selector(setOutputIndexArray:), @[@0]); } @catch (NSException *ex) {}

                    e = nil;
                    BOOL ok = ((BOOL(*)(id,SEL,id,id,id,unsigned int,NSError**))objc_msgSend)(g_client,
                        @selector(enqueueSetsWithModel:outputSet:options:qos:error:),
                        aneModelA, outSet, @{}, 21, &e);
                    printf("  enqueueSets: %d", ok);
                    if (!ok && e) printf(" error: %s", [[e description] UTF8String]);
                    printf("\n");
                }
            } @catch (NSException *ex) {
                printf("  enqueueSets exception: %s\n", [[ex reason] UTF8String]);
            }
        }

        // ===== 8. mapIOSurfaces =====
        printf("\n--- mapIOSurfaces ---\n");
        @try {
            e = nil;
            BOOL ok = ((BOOL(*)(id,SEL,id,id,BOOL,NSError**))objc_msgSend)(g_client,
                @selector(mapIOSurfacesWithModel:request:cacheInference:error:),
                aneModelA, reqA, YES, &e);
            printf("  mapIOSurfaces(cache=YES): %d\n", ok);
            if (!ok && e) printf("    error: %s\n", [[e description] UTF8String]);
            if (ok) {
                for (int i=0;i<50;i++)
                    ((BOOL(*)(id,SEL,id,id,id,unsigned int,NSError**))objc_msgSend)(g_client,
                        @selector(doEvaluateDirectWithModel:options:request:qos:error:),aneModelA,@{},reqA,21,&e);
                t0 = mach_absolute_time();
                for (int i=0;i<200;i++)
                    ((BOOL(*)(id,SEL,id,id,id,unsigned int,NSError**))objc_msgSend)(g_client,
                        @selector(doEvaluateDirectWithModel:options:request:qos:error:),aneModelA,@{},reqA,21,&e);
                double cm = ms_t(mach_absolute_time()-t0);
                printf("  200 evals after map: %.1f ms (%.3f ms/eval) [baseline %.3f]\n", cm, cm/200, seq_ms/200);
            }
        } @catch (NSException *ex) {
            printf("  Exception: %s\n", [[ex reason] UTF8String]);
        }

        // Cleanup
        ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(mdlA,@selector(unloadWithQoS:options:error:),21,@{},&e);
        CFRelease(ioIn); CFRelease(ioOut);
        printf("\n=== Done ===\n");
    }
    return 0;
}
