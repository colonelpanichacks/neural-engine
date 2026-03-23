// probe_shared_event.m — Test ANE hardware signaling and enqueue paths
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

// Keep alive array to prevent ARC from freeing objects ANE might reference
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

int main(int argc, char *argv[]) {
    @autoreleasepool {
        setbuf(stdout, NULL);
        mach_timebase_info(&g_tb);
        ane_init();

        printf("=== ANE Shared Event Probe v3 ===\n\n");

        int IC=256, OC=512, SEQ=64;
        NSData *mil = gen_matmul(IC, OC, SEQ);
        id desc = ((id(*)(Class,SEL,id,id,id))objc_msgSend)(g_D,@selector(modelWithMILText:weights:optionsPlist:),mil,@{},nil);
        id mdl = ((id(*)(Class,SEL,id))objc_msgSend)(g_I,@selector(inMemoryModelWithDescriptor:),desc);
        id hx = ((id(*)(id,SEL))objc_msgSend)(mdl,@selector(hexStringIdentifier));
        NSString *td = [NSTemporaryDirectory() stringByAppendingPathComponent:hx];
        [[NSFileManager defaultManager] createDirectoryAtPath:[td stringByAppendingPathComponent:@"weights"] withIntermediateDirectories:YES attributes:nil error:nil];
        [mil writeToFile:[td stringByAppendingPathComponent:@"model.mil"] atomically:YES];
        NSError *e = nil;
        ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(mdl,@selector(compileWithQoS:options:error:),21,@{},&e);
        ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(mdl,@selector(loadWithQoS:options:error:),21,@{},&e);
        printf("Compiled+loaded OK\n");

        id aneModel = ((id(*)(id,SEL))objc_msgSend)(mdl, @selector(model));
        size_t in_bytes = (size_t)IC*(SEQ+OC)*2;
        size_t out_bytes = (size_t)OC*SEQ*2;
        IOSurfaceRef ioIn = make_surf(in_bytes);
        IOSurfaceRef ioOut = make_surf(out_bytes);

        id wI=((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_AIO,@selector(objectWithIOSurface:),ioIn);
        id wO=((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_AIO,@selector(objectWithIOSurface:),ioOut);
        [g_keepalive addObject:wI];
        [g_keepalive addObject:wO];

        Class ioSEClass = NSClassFromString(@"IOSurfaceSharedEvent");
        Class sigEvtClass = NSClassFromString(@"_ANESharedSignalEvent");
        Class sharedEvtsClass = NSClassFromString(@"_ANESharedEvents");
        Class outSetClass = NSClassFromString(@"_ANEOutputSetEnqueue");

        // Warm up ANE
        id reqStd = ((id(*)(Class,SEL,id,id,id,id,id,id))objc_msgSend)(g_AR,
            @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:procedureIndex:),
            @[wI],@[@0],@[wO],@[@0],nil,@0);
        for (int i=0; i<10; i++)
            ((BOOL(*)(id,SEL,id,id,id,unsigned int,NSError**))objc_msgSend)(g_client,
                @selector(doEvaluateDirectWithModel:options:request:qos:error:),
                aneModel,@{},reqStd,21,&e);
        printf("Warmup done\n");

        // ===== 1. Test doEnqueueSetsWithModel =====
        printf("\n--- doEnqueueSetsWithModel ---\n");
        if (outSetClass) {
            @try {
                // Create output set
                id outSet = ((id(*)(Class,SEL,unsigned int,unsigned int,uint64_t,BOOL,BOOL))objc_msgSend)(outSetClass,
                    @selector(outputSetWithProcedureIndex:setIndex:signalValue:signalNotRequired:isOpenLoop:),
                    (unsigned int)0, (unsigned int)0, (uint64_t)1, NO, NO);
                [g_keepalive addObject:outSet];

                // Dump OutputSet properties
                unsigned int count = 0;
                objc_property_t *props = class_copyPropertyList([outSet class], &count);
                for (unsigned int i = 0; i < count; i++) {
                    const char *name = property_getName(props[i]);
                    @try {
                        id val = [outSet valueForKey:[NSString stringWithUTF8String:name]];
                        printf("  OutputSet.%s = %s\n", name, val ? [[val description] UTF8String] : "nil");
                    } @catch (id ex) { printf("  OutputSet.%s = <err>\n", name); }
                }
                free(props);

                // Try doEnqueueSetsWithModel
                e = nil;
                BOOL ok = ((BOOL(*)(id,SEL,id,id,id,unsigned int,NSError**))objc_msgSend)(g_client,
                    @selector(doEnqueueSetsWithModel:outputSet:options:qos:error:),
                    aneModel, outSet, @{}, 21, &e);
                printf("  doEnqueueSets: %d\n", ok);
                if (!ok && e) printf("    error: %s\n", [[e description] UTF8String]);

                // If it worked, try to benchmark it
                if (ok) {
                    uint64_t t0 = mach_absolute_time();
                    for (int i = 0; i < 100; i++) {
                        ((BOOL(*)(id,SEL,id,id,id,unsigned int,NSError**))objc_msgSend)(g_client,
                            @selector(doEnqueueSetsWithModel:outputSet:options:qos:error:),
                            aneModel, outSet, @{}, 21, &e);
                    }
                    double ms = ms_t(mach_absolute_time() - t0);
                    printf("  100 enqueues: %.1f ms (%.3f ms/enqueue)\n", ms, ms/100);
                }
            } @catch (NSException *ex) {
                printf("  Exception: %s\n", [[ex reason] UTF8String]);
            }
        }

        // ===== 2. Test chaining: prepare + enqueue =====
        printf("\n--- Chaining: prepare + enqueue ---\n");
        {
            Class chainReqClass = NSClassFromString(@"_ANEChainingRequest");
            if (chainReqClass && outSetClass) {
                @try {
                    // Create shared event for signaling
                    id shEvt = ((id(*)(id,SEL,uint64_t))objc_msgSend)([ioSEClass alloc], @selector(initWithOptions:), (uint64_t)0);
                    [g_keepalive addObject:shEvt];

                    id sigEvt = ((id(*)(Class,SEL,uint64_t,unsigned int,int64_t,id))objc_msgSend)(
                        sigEvtClass,
                        @selector(signalEventWithValue:symbolIndex:eventType:sharedEvent:),
                        (uint64_t)1, (unsigned int)0, (int64_t)0, shEvt);
                    [g_keepalive addObject:sigEvt];

                    id outSet = ((id(*)(Class,SEL,unsigned int,unsigned int,uint64_t,BOOL,BOOL))objc_msgSend)(outSetClass,
                        @selector(outputSetWithProcedureIndex:setIndex:signalValue:signalNotRequired:isOpenLoop:),
                        (unsigned int)0, (unsigned int)0, (uint64_t)1, NO, NO);
                    [g_keepalive addObject:outSet];

                    // Create chaining request with signal event
                    id cr = ((id(*)(Class,SEL,id,id,id,id,id,id,id,id,id))objc_msgSend)(chainReqClass,
                        @selector(chainingRequestWithInputs:outputSets:lbInputSymbolId:lbOutputSymbolId:procedureIndex:signalEvents:transactionHandle:fwEnqueueDelay:memoryPoolId:),
                        @[wI], @[outSet], @[], @[], @0, @[sigEvt], nil, @0, @0);
                    [g_keepalive addObject:cr];

                    if (cr) {
                        // Dump ChainingRequest
                        unsigned int count = 0;
                        objc_property_t *props = class_copyPropertyList([cr class], &count);
                        for (unsigned int i = 0; i < count; i++) {
                            const char *name = property_getName(props[i]);
                            @try {
                                id val = [cr valueForKey:[NSString stringWithUTF8String:name]];
                                NSString *desc = val ? [val description] : @"nil";
                                if ([desc length] > 120) desc = [[desc substringToIndex:120] stringByAppendingString:@"..."];
                                printf("  CR.%s = %s\n", name, [desc UTF8String]);
                            } @catch (id ex) { printf("  CR.%s = <err>\n", name); }
                        }
                        free(props);

                        // Prepare
                        e = nil;
                        BOOL ok = ((BOOL(*)(id,SEL,id,id,id,unsigned int,NSError**))objc_msgSend)(g_client,
                            @selector(doPrepareChainingWithModel:options:chainingReq:qos:error:),
                            aneModel, @{}, cr, 21, &e);
                        printf("  prepareChaining: %d\n", ok);
                        if (!ok && e) printf("    error: %s\n", [[e description] UTF8String]);

                        if (ok) {
                            // Enqueue
                            SEL enqSel = @selector(doEnqueueSetsWithModel:outputSet:options:qos:error:);
                            e = nil;
                            BOOL ok2 = ((BOOL(*)(id,SEL,id,id,id,unsigned int,NSError**))objc_msgSend)(g_client,
                                enqSel, aneModel, outSet, @{}, 21, &e);
                            printf("  enqueueSets: %d\n", ok2);
                            if (!ok2 && e) printf("    error: %s\n", [[e description] UTF8String]);

                            uint64_t sv = ((uint64_t(*)(id,SEL))objc_msgSend)(shEvt, @selector(signaledValue));
                            printf("  signaled=%llu%s\n", sv, (sv > 0) ? " *** HW SIGNAL ***" : "");
                        }
                    }
                } @catch (NSException *ex) {
                    printf("  Exception: %s\n", [[ex reason] UTF8String]);
                }
            }
        }

        // ===== 3. Test evaluateRealTimeWithModel =====
        printf("\n--- evaluateRealTimeWithModel ---\n");
        {
            @try {
                e = nil;
                uint64_t t0 = mach_absolute_time();
                BOOL ok = ((BOOL(*)(id,SEL,id,id,id,NSError**))objc_msgSend)(g_client,
                    @selector(evaluateRealTimeWithModel:options:request:error:),
                    aneModel,@{},reqStd,&e);
                double dt = ms_t(mach_absolute_time() - t0);
                printf("  eval: %d time=%.3fms\n", ok, dt);
                if (!ok && e) printf("    error: %s\n", [[e description] UTF8String]);

                if (ok) {
                    // Benchmark
                    for (int i=0; i<50; i++)
                        ((BOOL(*)(id,SEL,id,id,id,NSError**))objc_msgSend)(g_client,
                            @selector(evaluateRealTimeWithModel:options:request:error:),
                            aneModel,@{},reqStd,&e);
                    t0 = mach_absolute_time();
                    for (int i=0; i<200; i++)
                        ((BOOL(*)(id,SEL,id,id,id,NSError**))objc_msgSend)(g_client,
                            @selector(evaluateRealTimeWithModel:options:request:error:),
                            aneModel,@{},reqStd,&e);
                    double ms = ms_t(mach_absolute_time() - t0);
                    printf("  200 evals: %.1f ms (%.3f ms/eval)\n", ms, ms/200);

                    // Compare with doEvaluateDirectWithModel
                    for (int i=0; i<50; i++)
                        ((BOOL(*)(id,SEL,id,id,id,unsigned int,NSError**))objc_msgSend)(g_client,
                            @selector(doEvaluateDirectWithModel:options:request:qos:error:),
                            aneModel,@{},reqStd,21,&e);
                    t0 = mach_absolute_time();
                    for (int i=0; i<200; i++)
                        ((BOOL(*)(id,SEL,id,id,id,unsigned int,NSError**))objc_msgSend)(g_client,
                            @selector(doEvaluateDirectWithModel:options:request:qos:error:),
                            aneModel,@{},reqStd,21,&e);
                    ms = ms_t(mach_absolute_time() - t0);
                    printf("  200 doEvalDirect: %.1f ms (%.3f ms/eval)\n", ms, ms/200);
                }
            } @catch (NSException *ex) {
                printf("  Exception: %s\n", [[ex reason] UTF8String]);
            }
        }

        // ===== 4. Dedicated agentMask test (single iteration, no crash risk) =====
        printf("\n--- agentMask signal test ---\n");
        {
            id shEvt = ((id(*)(id,SEL,uint64_t))objc_msgSend)([ioSEClass alloc], @selector(initWithOptions:), (uint64_t)0);
            [g_keepalive addObject:shEvt];

            id sigEvt = ((id(*)(Class,SEL,uint64_t,unsigned int,int64_t,id))objc_msgSend)(
                sigEvtClass,
                @selector(signalEventWithValue:symbolIndex:eventType:sharedEvent:),
                (uint64_t)1, (unsigned int)0, (int64_t)0, shEvt);
            [g_keepalive addObject:sigEvt];

            // Set agentMask=0xFF via setter
            ((void(*)(id,SEL,uint64_t))objc_msgSend)(sigEvt, @selector(setAgentMask:), (uint64_t)0xFF);
            uint64_t readback = [[sigEvt valueForKey:@"agentMask"] unsignedLongLongValue];
            printf("  agentMask set to 0xFF, readback=0x%llx\n", readback);

            id shEvts = ((id(*)(Class,SEL,id,id))objc_msgSend)(sharedEvtsClass,
                @selector(sharedEventsWithSignalEvents:waitEvents:), @[sigEvt], @[]);
            [g_keepalive addObject:shEvts];

            id reqSE = ((id(*)(Class,SEL,id,id,id,id,id,id,id,id))objc_msgSend)(g_AR,
                @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:sharedEvents:),
                @[wI],@[@0],@[wO],@[@0],nil,nil,@0,shEvts);
            [g_keepalive addObject:reqSE];

            ((void(*)(id,SEL,uint64_t))objc_msgSend)(shEvt, @selector(setSignaledValue:), (uint64_t)0);
            e = nil;
            BOOL ok = ((BOOL(*)(id,SEL,id,id,id,unsigned int,NSError**))objc_msgSend)(g_client,
                @selector(doEvaluateDirectWithModel:options:request:qos:error:),
                aneModel,@{},reqSE,21,&e);
            uint64_t sv = ((uint64_t(*)(id,SEL))objc_msgSend)(shEvt, @selector(signaledValue));
            printf("  doEvalDirect+mask=0xFF: eval=%d signaled=%llu%s\n",
                   ok, sv, (sv > 0) ? " *** HW SIGNAL ***" : "");

            // Also try with evaluateRealTimeWithModel
            ((void(*)(id,SEL,uint64_t))objc_msgSend)(shEvt, @selector(setSignaledValue:), (uint64_t)0);
            e = nil;
            ok = ((BOOL(*)(id,SEL,id,id,id,NSError**))objc_msgSend)(g_client,
                @selector(evaluateRealTimeWithModel:options:request:error:),
                aneModel,@{},reqSE,&e);
            sv = ((uint64_t(*)(id,SEL))objc_msgSend)(shEvt, @selector(signaledValue));
            printf("  evalRealTime+mask=0xFF: eval=%d signaled=%llu%s\n",
                   ok, sv, (sv > 0) ? " *** HW SIGNAL ***" : "");
            if (!ok && e) printf("    error: %s\n", [[e description] UTF8String]);
        }

        printf("\n=== Done ===\n");
    }
    return 0;
}
