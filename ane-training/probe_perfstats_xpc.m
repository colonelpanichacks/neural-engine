// probe_perfstats_xpc.m — Try XPC eval path for perf stats, check perfStatsArray,
// and probe options dict keys that might enable stats collection
//
// Build:
//   xcrun clang -O2 -framework Foundation -framework IOSurface \
//     -isysroot /Library/Developer/CommandLineTools/SDKs/MacOSX.sdk \
//     -fobjc-arc -o probe_perfstats_xpc probe_perfstats_xpc.m

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

static void dump_obj_ivars(id obj) {
    Class cls = object_getClass(obj);
    while (cls) {
        unsigned int count = 0;
        Ivar *ivars = class_copyIvarList(cls, &count);
        for (unsigned int i = 0; i < count; i++) {
            const char *name = ivar_getName(ivars[i]);
            const char *type = ivar_getTypeEncoding(ivars[i]);
            printf("    ivar: %s (%s)\n", name, type ? type : "?");
        }
        free(ivars);
        cls = class_getSuperclass(cls);
        if (cls == [NSObject class]) break;
    }
}

int main(int argc, char *argv[]) {
    @autoreleasepool {
        setbuf(stdout, NULL);
        mach_timebase_info(&g_tb);
        ane_init();

        printf("=== ANE PerfStats — XPC Path & Options Probe ===\n");

        // Compile kernel
        int IC = 1024, OC = 2048, SEQ = 256;
        NSData *mil = gen_matmul(IC, OC, SEQ);
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
        ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(mdl,
            @selector(compileWithQoS:options:error:), 21, @{}, &e);
        ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(mdl,
            @selector(loadWithQoS:options:error:), 21, @{}, &e);
        id aneModel = ((id(*)(id,SEL))objc_msgSend)(mdl, @selector(model));
        [g_keepalive addObject:aneModel];

        size_t in_bytes = (size_t)IC * (SEQ + OC) * 2;
        size_t out_bytes = (size_t)OC * SEQ * 2;
        IOSurfaceRef ioIn = make_surf(in_bytes);
        IOSurfaceRef ioOut = make_surf(out_bytes);
        id wI = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_AIO, @selector(objectWithIOSurface:), ioIn);
        id wO = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_AIO, @selector(objectWithIOSurface:), ioOut);
        [g_keepalive addObject:wI];
        [g_keepalive addObject:wO];

        printf("Kernel compiled and loaded OK\n\n");

        // 1. Dump _ANERequest ivars to understand internal structure
        printf("=== _ANERequest ivars ===\n");
        {
            id req = ((id(*)(Class,SEL,id,id,id,id,id,id))objc_msgSend)(g_AR,
                @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:procedureIndex:),
                @[wI], @[@0], @[wO], @[@0], nil, @0);
            dump_obj_ivars(req);
        }

        // 2. Dump _ANEClient methods related to eval/perf
        printf("\n=== _ANEClient eval-related methods ===\n");
        {
            Class clientClass = NSClassFromString(@"_ANEClient");
            unsigned int mcount = 0;
            Method *methods = class_copyMethodList(clientClass, &mcount);
            for (unsigned int i = 0; i < mcount; i++) {
                const char *mn = sel_getName(method_getName(methods[i]));
                if (strstr(mn, "eval") || strstr(mn, "Eval") ||
                    strstr(mn, "perf") || strstr(mn, "Perf") ||
                    strstr(mn, "stat") || strstr(mn, "Stat") ||
                    strstr(mn, "option") || strstr(mn, "Option") ||
                    strstr(mn, "enqueue") || strstr(mn, "Enqueue")) {
                    printf("  -%s [%s]\n", mn, method_getTypeEncoding(methods[i]));
                }
            }
            free(methods);
        }

        // 3. Check _ANEModel methods — maybe stats are set at model level
        printf("\n=== _ANEModel methods with stats/perf/option ===\n");
        {
            Class modelClass = NSClassFromString(@"_ANEModel");
            if (modelClass) {
                unsigned int mcount = 0;
                Method *methods = class_copyMethodList(modelClass, &mcount);
                for (unsigned int i = 0; i < mcount; i++) {
                    const char *mn = sel_getName(method_getName(methods[i]));
                    if (strstr(mn, "perf") || strstr(mn, "Perf") ||
                        strstr(mn, "stat") || strstr(mn, "Stat") ||
                        strstr(mn, "option") || strstr(mn, "Option") ||
                        strstr(mn, "queue") || strstr(mn, "Queue") ||
                        strstr(mn, "debug") || strstr(mn, "Debug") ||
                        strstr(mn, "profil") || strstr(mn, "Profil")) {
                        printf("  -%s [%s]\n", mn, method_getTypeEncoding(methods[i]));
                    }
                }
                free(methods);
            } else {
                printf("  _ANEModel class not found\n");
            }
        }

        // 4. Try XPC eval and check perfStats/perfStatsArray after
        printf("\n=== Test 1: XPC eval — check perfStats/perfStatsArray after ===\n");
        {
            id req = ((id(*)(Class,SEL,id,id,id,id,id,id))objc_msgSend)(g_AR,
                @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:procedureIndex:),
                @[wI], @[@0], @[wO], @[@0], nil, @0);
            [g_keepalive addObject:req];

            e = nil;
            BOOL ok = ((BOOL(*)(id,SEL,id,id,id,unsigned int,NSError**))objc_msgSend)(g_client,
                @selector(evaluateWithModel:options:request:qos:error:),
                aneModel, @{}, req, 21, &e);
            printf("  eval (XPC): %s\n", ok?"OK":"FAIL");

            @try {
                id ps = ((id(*)(id,SEL))objc_msgSend)(req, @selector(perfStats));
                printf("  perfStats: %s\n", ps ? [[ps description] UTF8String] : "(nil)");
            } @catch (NSException *ex) {
                printf("  perfStats: EXCEPTION %s\n", [[ex description] UTF8String]);
            }

            @try {
                id psa = ((id(*)(id,SEL))objc_msgSend)(req, @selector(perfStatsArray));
                printf("  perfStatsArray: %s\n", psa ? [[psa description] UTF8String] : "(nil)");
            } @catch (NSException *ex) {
                printf("  perfStatsArray: EXCEPTION %s\n", [[ex description] UTF8String]);
            }
        }

        // 5. Try with _ANEPerformanceStatsIOSurface set, then XPC eval
        printf("\n=== Test 2: XPC eval with perfStats IOSurface set ===\n");
        {
            Class perfStatsIOClass = NSClassFromString(@"_ANEPerformanceStatsIOSurface");
            IOSurfaceRef statsSurf = make_surf(65536);
            IOSurfaceLock(statsSurf, 0, NULL);
            memset(IOSurfaceGetBaseAddress(statsSurf), 0, 65536);
            IOSurfaceUnlock(statsSurf, 0, NULL);

            id perfStats = ((id(*)(Class,SEL,IOSurfaceRef,int))objc_msgSend)(
                perfStatsIOClass, @selector(objectWithIOSurface:statType:), statsSurf, 0);
            [g_keepalive addObject:perfStats];

            id req = ((id(*)(Class,SEL,id,id,id,id,id,id))objc_msgSend)(g_AR,
                @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:procedureIndex:),
                @[wI], @[@0], @[wO], @[@0], nil, @0);
            [g_keepalive addObject:req];
            ((void(*)(id,SEL,id))objc_msgSend)(req, @selector(setPerfStats:), perfStats);

            e = nil;
            BOOL ok = ((BOOL(*)(id,SEL,id,id,id,unsigned int,NSError**))objc_msgSend)(g_client,
                @selector(evaluateWithModel:options:request:qos:error:),
                aneModel, @{}, req, 21, &e);
            printf("  eval (XPC+perfStats): %s\n", ok?"OK":"FAIL");

            // Check surface
            IOSurfaceLock(statsSurf, kIOSurfaceLockReadOnly, NULL);
            const uint8_t *data = (const uint8_t *)IOSurfaceGetBaseAddress(statsSurf);
            int nonzero = 0;
            for (size_t i = 0; i < 65536; i++) if (data[i]) nonzero++;
            printf("  Stats surface non-zero bytes: %d\n", nonzero);
            IOSurfaceUnlock(statsSurf, kIOSurfaceLockReadOnly, NULL);

            // Check request perfStats after eval
            @try {
                id ps = ((id(*)(id,SEL))objc_msgSend)(req, @selector(perfStats));
                printf("  perfStats after: %s\n", ps ? [[ps description] UTF8String] : "(nil)");
                if (ps && ps != perfStats) {
                    // Driver might have replaced it with a populated _ANEPerformanceStats
                    printf("  perfStats CLASS: %s\n", class_getName(object_getClass(ps)));
                    if ([ps respondsToSelector:@selector(hwExecutionTime)]) {
                        uint64_t hwt = ((uint64_t(*)(id,SEL))objc_msgSend)(ps, @selector(hwExecutionTime));
                        printf("  hwExecutionTime: %llu ns\n", hwt);
                    }
                    if ([ps respondsToSelector:@selector(performanceCounters)]) {
                        id pc = ((id(*)(id,SEL))objc_msgSend)(ps, @selector(performanceCounters));
                        printf("  performanceCounters: %s\n", pc ? [[pc description] UTF8String] : "(nil)");
                    }
                }
            } @catch (NSException *ex) {
                printf("  perfStats: EXCEPTION %s\n", [[ex description] UTF8String]);
            }

            @try {
                id psa = ((id(*)(id,SEL))objc_msgSend)(req, @selector(perfStatsArray));
                if (psa) {
                    printf("  perfStatsArray: %s\n", [[psa description] UTF8String]);
                    NSArray *arr = (NSArray *)psa;
                    for (id item in arr) {
                        printf("    item class: %s\n", class_getName(object_getClass(item)));
                        if ([item respondsToSelector:@selector(hwExecutionTime)]) {
                            uint64_t hwt = ((uint64_t(*)(id,SEL))objc_msgSend)(item, @selector(hwExecutionTime));
                            printf("    hwExecutionTime: %llu ns\n", hwt);
                        }
                    }
                } else {
                    printf("  perfStatsArray: (nil)\n");
                }
            } @catch (NSException *ex) {
                printf("  perfStatsArray: EXCEPTION %s\n", [[ex description] UTF8String]);
            }

            CFRelease(statsSurf);
        }

        // 6. Try options dicts that might enable stats
        printf("\n=== Test 3: Eval with various options dicts ===\n");
        {
            NSDictionary *optionSets[] = {
                @{@"EnablePerfStats": @YES},
                @{@"PerfStats": @YES},
                @{@"PerformanceStats": @YES},
                @{@"CollectStats": @YES},
                @{@"ProfileMode": @YES},
                @{@"Debug": @YES},
                @{@"ANEPerfStats": @YES},
                @{@"enablePerfStats": @YES},
                @{@"perfStats": @YES},
                @{@"debug": @YES},
            };
            const char *optionNames[] = {
                "EnablePerfStats", "PerfStats", "PerformanceStats", "CollectStats",
                "ProfileMode", "Debug", "ANEPerfStats", "enablePerfStats",
                "perfStats", "debug",
            };
            int nOpts = sizeof(optionSets) / sizeof(optionSets[0]);

            for (int oi = 0; oi < nOpts; oi++) {
                id req = ((id(*)(Class,SEL,id,id,id,id,id,id))objc_msgSend)(g_AR,
                    @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:procedureIndex:),
                    @[wI], @[@0], @[wO], @[@0], nil, @0);
                [g_keepalive addObject:req];

                e = nil;
                @try {
                    BOOL ok = ((BOOL(*)(id,SEL,id,id,id,unsigned int,NSError**))objc_msgSend)(g_client,
                        @selector(doEvaluateDirectWithModel:options:request:qos:error:),
                        aneModel, optionSets[oi], req, 21, &e);

                    id ps = ((id(*)(id,SEL))objc_msgSend)(req, @selector(perfStats));
                    id psa = ((id(*)(id,SEL))objc_msgSend)(req, @selector(perfStatsArray));
                    printf("  {%s: YES}: eval=%s perfStats=%s perfStatsArray=%s\n",
                           optionNames[oi],
                           ok?"OK":"FAIL",
                           ps?"SET":"nil",
                           psa?"SET":"nil");
                    if (ps) {
                        printf("    perfStats: %s\n", [[ps description] UTF8String]);
                    }
                    if (psa) {
                        printf("    perfStatsArray: %s\n", [[psa description] UTF8String]);
                    }
                } @catch (NSException *ex) {
                    printf("  {%s: YES}: EXCEPTION: %s\n", optionNames[oi], [[ex description] UTF8String]);
                }
            }
        }

        // 7. Check if _ANEModel has any stats-related properties/ivars
        printf("\n=== _ANEModel ivars ===\n");
        {
            Class modelClass = NSClassFromString(@"_ANEModel");
            if (modelClass) {
                unsigned int count = 0;
                Ivar *ivars = class_copyIvarList(modelClass, &count);
                for (unsigned int i = 0; i < count; i++) {
                    const char *name = ivar_getName(ivars[i]);
                    printf("  %s (%s)\n", name, ivar_getTypeEncoding(ivars[i]) ?: "?");
                }
                free(ivars);
            }
        }

        // 8. Check _ANEInMemoryModel for stats/profiling methods
        printf("\n=== _ANEInMemoryModel all methods ===\n");
        {
            unsigned int mcount = 0;
            Method *methods = class_copyMethodList(g_I, &mcount);
            for (unsigned int i = 0; i < mcount; i++) {
                printf("  -%s\n", sel_getName(method_getName(methods[i])));
            }
            free(methods);
            methods = class_copyMethodList(object_getClass(g_I), &mcount);
            for (unsigned int i = 0; i < mcount; i++) {
                printf("  +%s\n", sel_getName(method_getName(methods[i])));
            }
            free(methods);
        }

        printf("\n=== Done ===\n");
    }
    return 0;
}
