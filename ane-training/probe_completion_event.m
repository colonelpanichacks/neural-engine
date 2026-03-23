// probe_completion_event.m — Explore completionEvent parameter on ANE eval paths
// From device controller probe, _ANEVirtualClient has:
//   doEvaluateWithModel:options:request:qos:completionEvent:error:
//   Type: B60@0:8@16@24@32I40@44^@52
//   => BOOL return, model(@), options(@), request(@), qos(I), completionEvent(@), error(^@)
//
// _ANEVirtualClient is daemon-side only (nil in userspace). But _ANEDaemonConnection
// or _ANEClient may have equivalent methods we can reach.
#import <Foundation/Foundation.h>
#import <IOSurface/IOSurface.h>
#import <Accelerate/Accelerate.h>
#import <objc/runtime.h>
#import <objc/message.h>
#import <dlfcn.h>
#import <mach/mach_time.h>

static mach_timebase_info_data_t g_tb;
static double ms_t(uint64_t t) { return (double)t * g_tb.numer / g_tb.denom / 1e6; }
static double us_t(uint64_t t) { return (double)t * g_tb.numer / g_tb.denom / 1e3; }

static id g_client;
static Class g_D, g_I, g_AR, g_AIO;
static NSMutableArray *g_keepalive;

static void ane_init(void) {
    dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine", RTLD_NOW);
    g_D   = NSClassFromString(@"_ANEInMemoryModelDescriptor");
    g_I   = NSClassFromString(@"_ANEInMemoryModel");
    g_AR  = NSClassFromString(@"_ANERequest");
    g_AIO = NSClassFromString(@"_ANEIOSurfaceObject");
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

static void dump_methods(Class cls, const char *label) {
    unsigned int count = 0;
    Method *methods = class_copyMethodList(cls, &count);
    printf("  %s instance methods (%u):\n", label, count);
    for (unsigned int i = 0; i < count; i++) {
        const char *sel = sel_getName(method_getName(methods[i]));
        const char *enc = method_getTypeEncoding(methods[i]);
        printf("    %-85s  %s\n", sel, enc ? enc : "(nil)");
    }
    free(methods);

    unsigned int ccount = 0;
    Method *cmethods = class_copyMethodList(object_getClass(cls), &ccount);
    if (ccount > 0) {
        printf("  %s class methods (%u):\n", label, ccount);
        for (unsigned int i = 0; i < ccount; i++) {
            const char *sel = sel_getName(method_getName(cmethods[i]));
            const char *enc = method_getTypeEncoding(cmethods[i]);
            printf("    %-85s  %s\n", sel, enc ? enc : "(nil)");
        }
    }
    free(cmethods);
}

static void dump_ivars(Class cls, const char *label) {
    unsigned int count = 0;
    Ivar *ivars = class_copyIvarList(cls, &count);
    printf("  %s ivars (%u):\n", label, count);
    for (unsigned int i = 0; i < count; i++) {
        const char *n = ivar_getName(ivars[i]);
        const char *t = ivar_getTypeEncoding(ivars[i]);
        ptrdiff_t off = ivar_getOffset(ivars[i]);
        printf("    offset=%-4td  %-40s  %s\n", off, n ? n : "(anon)", t ? t : "(nil)");
    }
    free(ivars);
}

int main(int argc, char *argv[]) {
    @autoreleasepool {
        setbuf(stdout, NULL);
        mach_timebase_info(&g_tb);
        ane_init();

        printf("=== ANE CompletionEvent Probe ===\n\n");

        // ================================================================
        // Step 1: Compile 64x64 matmul
        // ================================================================
        printf("--- Step 1: Compile 64x64 matmul ---\n");
        int IC=64, OC=64, SEQ=64;
        NSData *mil = gen_matmul(IC, OC, SEQ);
        id desc = ((id(*)(Class,SEL,id,id,id))objc_msgSend)(g_D,
            @selector(modelWithMILText:weights:optionsPlist:), mil, @{}, nil);
        id mdl = ((id(*)(Class,SEL,id))objc_msgSend)(g_I,
            @selector(inMemoryModelWithDescriptor:), desc);
        id hx = ((id(*)(id,SEL))objc_msgSend)(mdl, @selector(hexStringIdentifier));
        NSString *td = [NSTemporaryDirectory() stringByAppendingPathComponent:hx];
        [[NSFileManager defaultManager] createDirectoryAtPath:[td stringByAppendingPathComponent:@"weights"]
            withIntermediateDirectories:YES attributes:nil error:nil];
        [mil writeToFile:[td stringByAppendingPathComponent:@"model.mil"] atomically:YES];
        NSError *e = nil;
        ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(mdl,
            @selector(compileWithQoS:options:error:), 21, @{}, &e);
        if (e) { printf("  Compile error: %s\n", [[e description] UTF8String]); return 1; }
        ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(mdl,
            @selector(loadWithQoS:options:error:), 21, @{}, &e);
        if (e) { printf("  Load error: %s\n", [[e description] UTF8String]); return 1; }
        printf("  Compiled + loaded OK\n");

        id aneModel = ((id(*)(id,SEL))objc_msgSend)(mdl, @selector(model));
        size_t in_bytes = (size_t)IC * (SEQ + OC) * 2;
        size_t out_bytes = (size_t)OC * SEQ * 2;
        IOSurfaceRef ioIn = make_surf(in_bytes);
        IOSurfaceRef ioOut = make_surf(out_bytes);

        _Float16 *inp = (_Float16 *)IOSurfaceGetBaseAddress(ioIn);
        for (size_t i = 0; i < in_bytes / 2; i++) inp[i] = (_Float16)(0.01f * (i % 100));

        id wI = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_AIO, @selector(objectWithIOSurface:), ioIn);
        id wO = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_AIO, @selector(objectWithIOSurface:), ioOut);
        [g_keepalive addObject:wI];
        [g_keepalive addObject:wO];

        id req = ((id(*)(Class,SEL,id,id,id,id,id,id))objc_msgSend)(g_AR,
            @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:procedureIndex:),
            @[wI], @[@0], @[wO], @[@0], nil, @0);
        [g_keepalive addObject:req];

        // Warmup
        for (int i = 0; i < 10; i++) {
            e = nil;
            ((BOOL(*)(id,SEL,id,id,id,unsigned int,NSError**))objc_msgSend)(g_client,
                @selector(doEvaluateDirectWithModel:options:request:qos:error:),
                aneModel, @{}, req, 21, &e);
        }
        _Float16 *refOut = (_Float16 *)IOSurfaceGetBaseAddress(ioOut);
        size_t nout = out_bytes / 2;
        float refSum = 0;
        for (size_t i = 0; i < nout; i++) refSum += (float)refOut[i];
        printf("  Warmup done. Ref checksum: %.4f\n", refSum);

        // ================================================================
        // Step 2: Dump _ANEVirtualClient methods (class exists even if no instance)
        // ================================================================
        printf("\n--- Step 2: _ANEVirtualClient class introspection ---\n");
        Class vcClass = NSClassFromString(@"_ANEVirtualClient");
        if (vcClass) {
            // Just show eval/event methods to keep output focused
            printf("  Eval and event-related methods:\n");
            unsigned int mc = 0;
            Method *mths = class_copyMethodList(vcClass, &mc);
            for (unsigned int i = 0; i < mc; i++) {
                const char *s = sel_getName(method_getName(mths[i]));
                if (strcasestr(s, "eval") || strcasestr(s, "event") ||
                    strcasestr(s, "completion") || strcasestr(s, "async") ||
                    strcasestr(s, "signal") || strcasestr(s, "fence") ||
                    strcasestr(s, "load") || strcasestr(s, "enqueue")) {
                    printf("    -%s\n      %s\n", s, method_getTypeEncoding(mths[i]) ?: "");
                }
            }
            free(mths);
            dump_ivars(vcClass, "_ANEVirtualClient");
        }

        // ================================================================
        // Step 3: Search for completionEvent-related classes
        // ================================================================
        printf("\n--- Step 3: Search for event/completion classes ---\n");
        const char *eventClassNames[] = {
            "_ANECompletionEvent", "_ANESharedEvent", "_ANESharedSignalEvent",
            "_ANEEvent", "_ANECompletionHandler", "_ANEEvaluationEvent",
            "_ANECompletionSignal", "_ANENotificationEvent",
            "_ANEFenceEvent", "_ANECompletionFence",
            "ANECompletionEvent", "ANESharedEvent",
            NULL
        };
        for (int i = 0; eventClassNames[i]; i++) {
            Class c = NSClassFromString([NSString stringWithUTF8String:eventClassNames[i]]);
            if (c) {
                printf("  FOUND: %s\n", eventClassNames[i]);
                dump_methods(c, eventClassNames[i]);
                dump_ivars(c, eventClassNames[i]);
            }
        }

        // Broader scan
        printf("\n  Scanning all classes for ANE*Event/ANE*Completion/ANE*Signal/ANE*Fence:\n");
        unsigned int numClasses = 0;
        Class *allClasses = objc_copyClassList(&numClasses);
        for (unsigned int i = 0; i < numClasses; i++) {
            const char *cn = class_getName(allClasses[i]);
            if (strstr(cn, "ANE") && (strstr(cn, "Event") || strstr(cn, "Completion") ||
                                       strstr(cn, "Signal") || strstr(cn, "Fence"))) {
                printf("    %s\n", cn);
            }
        }
        free(allClasses);

        // ================================================================
        // Step 4: Global search — any class with completionEvent in a selector
        // ================================================================
        printf("\n--- Step 4: Global search for 'completionEvent' in any selector ---\n");
        {
            unsigned int nc2 = 0;
            Class *cls2 = objc_copyClassList(&nc2);
            for (unsigned int c = 0; c < nc2; c++) {
                const char *cn = class_getName(cls2[c]);
                // Instance methods
                unsigned int mc = 0;
                Method *mths = class_copyMethodList(cls2[c], &mc);
                for (unsigned int m2 = 0; m2 < mc; m2++) {
                    const char *s = sel_getName(method_getName(mths[m2]));
                    if (strcasestr(s, "completionEvent")) {
                        printf("  [%s] -%s\n    %s\n", cn, s, method_getTypeEncoding(mths[m2]) ?: "");
                    }
                }
                free(mths);
                // Class methods
                unsigned int cmc = 0;
                Method *cmths = class_copyMethodList(object_getClass(cls2[c]), &cmc);
                for (unsigned int m2 = 0; m2 < cmc; m2++) {
                    const char *s = sel_getName(method_getName(cmths[m2]));
                    if (strcasestr(s, "completionEvent")) {
                        printf("  [%s] +%s\n    %s\n", cn, s, method_getTypeEncoding(cmths[m2]) ?: "");
                    }
                }
                free(cmths);
            }
            free(cls2);
        }

        // ================================================================
        // Step 5: Dump _ANEDaemonConnection (we have _fastConn)
        // ================================================================
        printf("\n--- Step 5: _ANEDaemonConnection introspection ---\n");
        id fastConn = nil;
        {
            Ivar ivar = class_getInstanceVariable([g_client class], "_fastConn");
            if (ivar) fastConn = object_getIvar(g_client, ivar);
        }
        if (fastConn) {
            printf("  _fastConn = %s <%p>\n", class_getName([fastConn class]), fastConn);
            // Show eval/event methods
            printf("  Eval and event-related methods:\n");
            unsigned int mc = 0;
            Method *mths = class_copyMethodList([fastConn class], &mc);
            for (unsigned int i = 0; i < mc; i++) {
                const char *s = sel_getName(method_getName(mths[i]));
                if (strcasestr(s, "eval") || strcasestr(s, "event") ||
                    strcasestr(s, "completion") || strcasestr(s, "async") ||
                    strcasestr(s, "signal") || strcasestr(s, "fence") ||
                    strcasestr(s, "enqueue") || strcasestr(s, "direct")) {
                    printf("    -%s\n      %s\n", s, method_getTypeEncoding(mths[i]) ?: "");
                }
            }
            free(mths);
            dump_ivars([fastConn class], "_ANEDaemonConnection");
        }

        // ================================================================
        // Step 6: Try completionEvent on _ANEClient
        // ================================================================
        printf("\n--- Step 6: _ANEClient completionEvent methods ---\n");
        {
            SEL doEvalCompSel = @selector(doEvaluateWithModel:options:request:qos:completionEvent:error:);
            printf("  _ANEClient responds to doEvaluateWithModel:...:completionEvent:error: ? %s\n",
                   [g_client respondsToSelector:doEvalCompSel] ? "YES" : "NO");

            // Check all _ANEClient eval methods
            printf("  All _ANEClient eval methods:\n");
            unsigned int mc = 0;
            Method *mths = class_copyMethodList([g_client class], &mc);
            for (unsigned int i = 0; i < mc; i++) {
                const char *s = sel_getName(method_getName(mths[i]));
                if (strcasestr(s, "eval") || strcasestr(s, "completionEvent")) {
                    printf("    -%s\n      %s\n", s, method_getTypeEncoding(mths[i]) ?: "");
                }
            }
            free(mths);
        }

        // ================================================================
        // Step 7: Try doEvaluateWithModel on _ANEDaemonConnection
        // ================================================================
        printf("\n--- Step 7: Try eval methods on _ANEDaemonConnection (_fastConn) ---\n");
        if (fastConn) {
            // Check if _ANEDaemonConnection has completionEvent
            SEL doEvalCompSel = @selector(doEvaluateWithModel:options:request:qos:completionEvent:error:);
            SEL doEvalSel = @selector(doEvaluateWithModel:options:request:qos:error:);
            SEL evalDirectSel = @selector(doEvaluateDirectWithModel:options:request:qos:error:);

            printf("  responds to doEvaluateWithModel:...:completionEvent:error: ? %s\n",
                   [fastConn respondsToSelector:doEvalCompSel] ? "YES" : "NO");
            printf("  responds to doEvaluateWithModel:...:error: ? %s\n",
                   [fastConn respondsToSelector:doEvalSel] ? "YES" : "NO");
            printf("  responds to doEvaluateDirectWithModel:...:error: ? %s\n",
                   [fastConn respondsToSelector:evalDirectSel] ? "YES" : "NO");

            // Try doEvaluateWithModel (without completionEvent)
            if ([fastConn respondsToSelector:doEvalSel]) {
                memset(IOSurfaceGetBaseAddress(ioOut), 0, out_bytes);
                e = nil;
                @try {
                    uint64_t t0 = mach_absolute_time();
                    BOOL ok = ((BOOL(*)(id,SEL,id,id,id,unsigned int,NSError**))objc_msgSend)(
                        fastConn, doEvalSel, aneModel, @{}, req, (unsigned int)21, &e);
                    double dt = us_t(mach_absolute_time() - t0);
                    printf("  doEvaluateWithModel (no completionEvent): %s (%.1f us)\n", ok ? "OK" : "FAIL", dt);
                    if (!ok && e) printf("    error: %s\n", [[e description] UTF8String]);
                    if (ok) {
                        float sum = 0;
                        _Float16 *out = (_Float16 *)IOSurfaceGetBaseAddress(ioOut);
                        for (size_t i = 0; i < nout; i++) sum += (float)out[i];
                        printf("    output checksum: %.4f (match ref: %s)\n", sum,
                               fabsf(sum - refSum) < 0.01f ? "YES" : "NO");
                    }
                } @catch (NSException *ex) {
                    printf("  Exception: %s: %s\n", [[ex name] UTF8String], [[ex reason] UTF8String]);
                }
            }

            // Try with completionEvent
            if ([fastConn respondsToSelector:doEvalCompSel]) {
                printf("\n  Testing completionEvent variants on _ANEDaemonConnection:\n");

                // completionEvent = nil
                {
                    memset(IOSurfaceGetBaseAddress(ioOut), 0, out_bytes);
                    e = nil;
                    @try {
                        uint64_t t0 = mach_absolute_time();
                        BOOL ok = ((BOOL(*)(id,SEL,id,id,id,unsigned int,id,NSError**))objc_msgSend)(
                            fastConn, doEvalCompSel, aneModel, @{}, req, (unsigned int)21, nil, &e);
                        double dt = us_t(mach_absolute_time() - t0);
                        printf("    completionEvent=nil: %s (%.1f us)\n", ok ? "OK" : "FAIL", dt);
                        if (!ok && e) printf("      error: %s\n", [[e description] UTF8String]);
                        if (ok) {
                            float sum = 0;
                            _Float16 *out = (_Float16 *)IOSurfaceGetBaseAddress(ioOut);
                            for (size_t i = 0; i < nout; i++) sum += (float)out[i];
                            printf("      checksum: %.4f (match: %s)\n", sum,
                                   fabsf(sum - refSum) < 0.01f ? "YES" : "NO");
                        }
                    } @catch (NSException *ex) {
                        printf("    Exception: %s\n", [[ex reason] UTF8String]);
                    }
                }

                // completionEvent = dispatch_semaphore
                {
                    dispatch_semaphore_t sema = dispatch_semaphore_create(0);
                    memset(IOSurfaceGetBaseAddress(ioOut), 0, out_bytes);
                    e = nil;
                    @try {
                        uint64_t t0 = mach_absolute_time();
                        BOOL ok = ((BOOL(*)(id,SEL,id,id,id,unsigned int,id,NSError**))objc_msgSend)(
                            fastConn, doEvalCompSel, aneModel, @{}, req, (unsigned int)21, (id)sema, &e);
                        double dtCall = us_t(mach_absolute_time() - t0);
                        printf("    completionEvent=semaphore: %s (%.1f us)\n", ok ? "OK" : "FAIL", dtCall);
                        if (!ok && e) printf("      error: %s\n", [[e description] UTF8String]);
                        if (ok) {
                            long waited = dispatch_semaphore_wait(sema, dispatch_time(DISPATCH_TIME_NOW, 100*NSEC_PER_MSEC));
                            printf("      semaphore: %s\n", waited == 0 ? "SIGNALED" : "TIMEOUT");
                            float sum = 0;
                            _Float16 *out = (_Float16 *)IOSurfaceGetBaseAddress(ioOut);
                            for (size_t i = 0; i < nout; i++) sum += (float)out[i];
                            printf("      checksum: %.4f (match: %s)\n", sum,
                                   fabsf(sum - refSum) < 0.01f ? "YES" : "NO");
                        }
                    } @catch (NSException *ex) {
                        printf("    Exception: %s\n", [[ex reason] UTF8String]);
                    }
                }

                // completionEvent = IOSurfaceSharedEvent
                {
                    Class ioSEClass = NSClassFromString(@"IOSurfaceSharedEvent");
                    if (ioSEClass) {
                        id shEvt = ((id(*)(id,SEL,uint64_t))objc_msgSend)([ioSEClass alloc],
                            @selector(initWithOptions:), (uint64_t)0);
                        [g_keepalive addObject:shEvt];
                        memset(IOSurfaceGetBaseAddress(ioOut), 0, out_bytes);
                        e = nil;
                        @try {
                            uint64_t t0 = mach_absolute_time();
                            BOOL ok = ((BOOL(*)(id,SEL,id,id,id,unsigned int,id,NSError**))objc_msgSend)(
                                fastConn, doEvalCompSel, aneModel, @{}, req, (unsigned int)21, shEvt, &e);
                            double dt = us_t(mach_absolute_time() - t0);
                            printf("    completionEvent=IOSurfaceSharedEvent: %s (%.1f us)\n", ok ? "OK" : "FAIL", dt);
                            if (!ok && e) printf("      error: %s\n", [[e description] UTF8String]);
                            if (ok) {
                                uint64_t sv = ((uint64_t(*)(id,SEL))objc_msgSend)(shEvt, @selector(signaledValue));
                                printf("      signaledValue: %llu\n", sv);
                                float sum = 0;
                                _Float16 *out = (_Float16 *)IOSurfaceGetBaseAddress(ioOut);
                                for (size_t i = 0; i < nout; i++) sum += (float)out[i];
                                printf("      checksum: %.4f (match: %s)\n", sum,
                                       fabsf(sum - refSum) < 0.01f ? "YES" : "NO");
                            }
                        } @catch (NSException *ex) {
                            printf("    Exception: %s\n", [[ex reason] UTF8String]);
                        }
                    }
                }

                // completionEvent = _ANESharedSignalEvent
                {
                    Class sigEvtClass = NSClassFromString(@"_ANESharedSignalEvent");
                    Class ioSEClass = NSClassFromString(@"IOSurfaceSharedEvent");
                    if (sigEvtClass && ioSEClass) {
                        id shEvt = ((id(*)(id,SEL,uint64_t))objc_msgSend)([ioSEClass alloc],
                            @selector(initWithOptions:), (uint64_t)0);
                        [g_keepalive addObject:shEvt];
                        id sigEvt = ((id(*)(Class,SEL,uint64_t,unsigned int,int64_t,id))objc_msgSend)(
                            sigEvtClass,
                            @selector(signalEventWithValue:symbolIndex:eventType:sharedEvent:),
                            (uint64_t)1, (unsigned int)0, (int64_t)0, shEvt);
                        [g_keepalive addObject:sigEvt];
                        memset(IOSurfaceGetBaseAddress(ioOut), 0, out_bytes);
                        e = nil;
                        @try {
                            uint64_t t0 = mach_absolute_time();
                            BOOL ok = ((BOOL(*)(id,SEL,id,id,id,unsigned int,id,NSError**))objc_msgSend)(
                                fastConn, doEvalCompSel, aneModel, @{}, req, (unsigned int)21, sigEvt, &e);
                            double dt = us_t(mach_absolute_time() - t0);
                            printf("    completionEvent=_ANESharedSignalEvent: %s (%.1f us)\n", ok ? "OK" : "FAIL", dt);
                            if (!ok && e) printf("      error: %s\n", [[e description] UTF8String]);
                            if (ok) {
                                uint64_t sv = ((uint64_t(*)(id,SEL))objc_msgSend)(shEvt, @selector(signaledValue));
                                printf("      signaledValue: %llu\n", sv);
                                float sum = 0;
                                _Float16 *out = (_Float16 *)IOSurfaceGetBaseAddress(ioOut);
                                for (size_t i = 0; i < nout; i++) sum += (float)out[i];
                                printf("      checksum: %.4f (match: %s)\n", sum,
                                       fabsf(sum - refSum) < 0.01f ? "YES" : "NO");
                            }
                        } @catch (NSException *ex) {
                            printf("    Exception: %s\n", [[ex reason] UTF8String]);
                        }
                    }
                }

                // completionEvent = NSObject (probe type expectation)
                {
                    id obj = [[NSObject alloc] init];
                    memset(IOSurfaceGetBaseAddress(ioOut), 0, out_bytes);
                    e = nil;
                    @try {
                        uint64_t t0 = mach_absolute_time();
                        BOOL ok = ((BOOL(*)(id,SEL,id,id,id,unsigned int,id,NSError**))objc_msgSend)(
                            fastConn, doEvalCompSel, aneModel, @{}, req, (unsigned int)21, obj, &e);
                        double dt = us_t(mach_absolute_time() - t0);
                        printf("    completionEvent=NSObject: %s (%.1f us)\n", ok ? "OK" : "FAIL", dt);
                        if (!ok && e) printf("      error: %s\n", [[e description] UTF8String]);
                        if (ok) {
                            float sum = 0;
                            _Float16 *out = (_Float16 *)IOSurfaceGetBaseAddress(ioOut);
                            for (size_t i = 0; i < nout; i++) sum += (float)out[i];
                            printf("      checksum: %.4f (match: %s)\n", sum,
                                   fabsf(sum - refSum) < 0.01f ? "YES" : "NO");
                        }
                    } @catch (NSException *ex) {
                        printf("    Exception: %s\n", [[ex reason] UTF8String]);
                    }
                }
            }
        }

        // ================================================================
        // Step 8: Benchmark comparison — all available eval paths
        // ================================================================
        printf("\n--- Step 8: Benchmark comparison (500 iterations each) ---\n");
        {
            // Baseline: doEvaluateDirectWithModel via _ANEClient
            for (int i = 0; i < 50; i++) {
                e = nil;
                ((BOOL(*)(id,SEL,id,id,id,unsigned int,NSError**))objc_msgSend)(g_client,
                    @selector(doEvaluateDirectWithModel:options:request:qos:error:),
                    aneModel, @{}, req, 21, &e);
            }
            uint64_t t0 = mach_absolute_time();
            for (int i = 0; i < 500; i++) {
                e = nil;
                ((BOOL(*)(id,SEL,id,id,id,unsigned int,NSError**))objc_msgSend)(g_client,
                    @selector(doEvaluateDirectWithModel:options:request:qos:error:),
                    aneModel, @{}, req, 21, &e);
            }
            double ms1 = ms_t(mach_absolute_time() - t0);
            printf("  _ANEClient doEvaluateDirectWithModel:     %.1f ms (%.3f ms/eval, %.1f us/eval)\n",
                   ms1, ms1/500, ms1*1000/500);

            // _ANEDaemonConnection paths
            if (fastConn) {
                SEL doEvalSel = @selector(doEvaluateWithModel:options:request:qos:error:);
                if ([fastConn respondsToSelector:doEvalSel]) {
                    for (int i = 0; i < 50; i++) {
                        e = nil;
                        ((BOOL(*)(id,SEL,id,id,id,unsigned int,NSError**))objc_msgSend)(
                            fastConn, doEvalSel, aneModel, @{}, req, (unsigned int)21, &e);
                    }
                    t0 = mach_absolute_time();
                    for (int i = 0; i < 500; i++) {
                        e = nil;
                        ((BOOL(*)(id,SEL,id,id,id,unsigned int,NSError**))objc_msgSend)(
                            fastConn, doEvalSel, aneModel, @{}, req, (unsigned int)21, &e);
                    }
                    double ms2 = ms_t(mach_absolute_time() - t0);
                    printf("  _ANEDaemonConn doEvaluateWithModel:       %.1f ms (%.3f ms/eval, %.1f us/eval)\n",
                           ms2, ms2/500, ms2*1000/500);
                }

                SEL doEvalCompSel = @selector(doEvaluateWithModel:options:request:qos:completionEvent:error:);
                if ([fastConn respondsToSelector:doEvalCompSel]) {
                    // completionEvent = nil
                    for (int i = 0; i < 50; i++) {
                        e = nil;
                        ((BOOL(*)(id,SEL,id,id,id,unsigned int,id,NSError**))objc_msgSend)(
                            fastConn, doEvalCompSel, aneModel, @{}, req, (unsigned int)21, nil, &e);
                    }
                    t0 = mach_absolute_time();
                    for (int i = 0; i < 500; i++) {
                        e = nil;
                        ((BOOL(*)(id,SEL,id,id,id,unsigned int,id,NSError**))objc_msgSend)(
                            fastConn, doEvalCompSel, aneModel, @{}, req, (unsigned int)21, nil, &e);
                    }
                    double ms3 = ms_t(mach_absolute_time() - t0);
                    printf("  _ANEDaemonConn doEval+completionEvent=nil: %.1f ms (%.3f ms/eval, %.1f us/eval)\n",
                           ms3, ms3/500, ms3*1000/500);
                }

                SEL evalDirectSel = @selector(doEvaluateDirectWithModel:options:request:qos:error:);
                if ([fastConn respondsToSelector:evalDirectSel]) {
                    for (int i = 0; i < 50; i++) {
                        e = nil;
                        ((BOOL(*)(id,SEL,id,id,id,unsigned int,NSError**))objc_msgSend)(
                            fastConn, evalDirectSel, aneModel, @{}, req, (unsigned int)21, &e);
                    }
                    t0 = mach_absolute_time();
                    for (int i = 0; i < 500; i++) {
                        e = nil;
                        ((BOOL(*)(id,SEL,id,id,id,unsigned int,NSError**))objc_msgSend)(
                            fastConn, evalDirectSel, aneModel, @{}, req, (unsigned int)21, &e);
                    }
                    double ms4 = ms_t(mach_absolute_time() - t0);
                    printf("  _ANEDaemonConn doEvaluateDirectWithModel:  %.1f ms (%.3f ms/eval, %.1f us/eval)\n",
                           ms4, ms4/500, ms4*1000/500);
                }
            }
        }

        // ================================================================
        // Step 9: Async fire-and-forget test (if completionEvent worked)
        // ================================================================
        printf("\n--- Step 9: Async fire-and-forget test ---\n");
        if (fastConn) {
            SEL doEvalCompSel = @selector(doEvaluateWithModel:options:request:qos:completionEvent:error:);
            if ([fastConn respondsToSelector:doEvalCompSel]) {
                // Separate output surface
                IOSurfaceRef ioOut2 = make_surf(out_bytes);
                id wO2 = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_AIO, @selector(objectWithIOSurface:), ioOut2);
                [g_keepalive addObject:wO2];
                id req2 = ((id(*)(Class,SEL,id,id,id,id,id,id))objc_msgSend)(g_AR,
                    @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:procedureIndex:),
                    @[wI], @[@0], @[wO2], @[@0], nil, @0);
                [g_keepalive addObject:req2];

                memset(IOSurfaceGetBaseAddress(ioOut2), 0, out_bytes);
                e = nil;
                @try {
                    uint64_t t0 = mach_absolute_time();
                    BOOL ok = ((BOOL(*)(id,SEL,id,id,id,unsigned int,id,NSError**))objc_msgSend)(
                        fastConn, doEvalCompSel, aneModel, @{}, req2, (unsigned int)21, nil, &e);
                    double dtCall = us_t(mach_absolute_time() - t0);

                    float sum = 0;
                    _Float16 *out2 = (_Float16 *)IOSurfaceGetBaseAddress(ioOut2);
                    for (size_t i = 0; i < nout; i++) sum += (float)out2[i];

                    printf("  call returned in %.1f us, ok=%d\n", dtCall, ok);
                    printf("  immediate checksum: %.4f (match: %s)\n", sum,
                           fabsf(sum - refSum) < 0.01f ? "YES" : "NO");

                    if (ok && fabsf(sum) < 0.001f) {
                        printf("  *** Output is zero — eval may be truly async! Waiting 10ms...\n");
                        usleep(10000);
                        sum = 0;
                        for (size_t i = 0; i < nout; i++) sum += (float)out2[i];
                        printf("  After 10ms wait: checksum=%.4f (match=%s)\n", sum,
                               fabsf(sum - refSum) < 0.01f ? "YES" : "NO");
                    }
                } @catch (NSException *ex) {
                    printf("  Exception: %s\n", [[ex reason] UTF8String]);
                }
                CFRelease(ioOut2);
            } else {
                printf("  Skipped — _ANEDaemonConnection has no completionEvent method\n");
            }
        }

        // ================================================================
        // Step 10: Check _ANEClient eval methods with completionEvent
        // ================================================================
        printf("\n--- Step 10: _ANEClient completionEvent ---\n");
        {
            SEL doEvalCompSel = @selector(doEvaluateWithModel:options:request:qos:completionEvent:error:);
            if ([g_client respondsToSelector:doEvalCompSel]) {
                printf("  _ANEClient HAS doEvaluateWithModel:...:completionEvent:!\n");
                memset(IOSurfaceGetBaseAddress(ioOut), 0, out_bytes);
                e = nil;
                @try {
                    uint64_t t0 = mach_absolute_time();
                    BOOL ok = ((BOOL(*)(id,SEL,id,id,id,unsigned int,id,NSError**))objc_msgSend)(
                        g_client, doEvalCompSel, aneModel, @{}, req, (unsigned int)21, nil, &e);
                    double dt = us_t(mach_absolute_time() - t0);
                    printf("  completionEvent=nil: %s (%.1f us)\n", ok ? "OK" : "FAIL", dt);
                    if (!ok && e) printf("    error: %s\n", [[e description] UTF8String]);
                    if (ok) {
                        float sum = 0;
                        _Float16 *out = (_Float16 *)IOSurfaceGetBaseAddress(ioOut);
                        for (size_t i = 0; i < nout; i++) sum += (float)out[i];
                        printf("    checksum: %.4f (match: %s)\n", sum,
                               fabsf(sum - refSum) < 0.01f ? "YES" : "NO");
                    }
                } @catch (NSException *ex) {
                    printf("  Exception: %s\n", [[ex reason] UTF8String]);
                }
            } else {
                printf("  _ANEClient does NOT have doEvaluateWithModel:...:completionEvent:\n");
            }
        }

        printf("\n=== Done ===\n");
    }
    return 0;
}
