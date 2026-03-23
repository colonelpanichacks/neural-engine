// probe_perfstats_io.m — Extract ANE hardware perf counters via _ANEPerformanceStatsIOSurface
//
// Key insight: _ANERequest validate calls [perfStats statType], which only
// _ANEPerformanceStatsIOSurface implements (not bare _ANEPerformanceStats).
//
// Build:
//   xcrun clang -O2 -framework Foundation -framework IOSurface \
//     -isysroot /Library/Developer/CommandLineTools/SDKs/MacOSX.sdk \
//     -fobjc-arc -o probe_perfstats_io probe_perfstats_io.m
//
// Run:
//   ./probe_perfstats_io

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

static void dump_hex(const uint8_t *data, size_t len, size_t max_bytes) {
    size_t n = len < max_bytes ? len : max_bytes;
    for (size_t i = 0; i < n; i++) {
        if (i % 32 == 0) printf("    %04zx: ", i);
        printf("%02x ", data[i]);
        if (i % 32 == 31 || i == n-1) printf("\n");
    }
    if (len > max_bytes) printf("    ... (%zu more bytes)\n", len - max_bytes);
}

static void dump_uint64(const uint8_t *data, size_t len, size_t max_vals) {
    const uint64_t *u64 = (const uint64_t *)data;
    size_t count = len / 8;
    size_t n = count < max_vals ? count : max_vals;
    for (size_t i = 0; i < n; i++) {
        if (i % 4 == 0) printf("    [%3zu]: ", i);
        printf("%16llx ", u64[i]);
        if (i % 4 == 3 || i == n-1) printf("\n");
    }
}

// Enumerate all methods on _ANEPerformanceStatsIOSurface
static void dump_class_methods(const char *className) {
    Class cls = NSClassFromString([NSString stringWithUTF8String:className]);
    if (!cls) { printf("  Class %s not found\n", className); return; }

    printf("\n=== %s instance methods ===\n", className);
    unsigned int mcount = 0;
    Method *methods = class_copyMethodList(cls, &mcount);
    for (unsigned int i = 0; i < mcount; i++) {
        printf("  -%s  [%s]\n", sel_getName(method_getName(methods[i])),
               method_getTypeEncoding(methods[i]));
    }
    free(methods);

    printf("\n=== %s class methods ===\n", className);
    methods = class_copyMethodList(object_getClass(cls), &mcount);
    for (unsigned int i = 0; i < mcount; i++) {
        printf("  +%s  [%s]\n", sel_getName(method_getName(methods[i])),
               method_getTypeEncoding(methods[i]));
    }
    free(methods);

    // Also check superclass
    Class super = class_getSuperclass(cls);
    if (super) printf("\n  Superclass: %s\n", class_getName(super));
}

// Enumerate _ANEPerformanceStats methods too for comparison
static void dump_perf_classes(void) {
    dump_class_methods("_ANEPerformanceStatsIOSurface");
    dump_class_methods("_ANEPerformanceStats");
}

int main(int argc, char *argv[]) {
    @autoreleasepool {
        setbuf(stdout, NULL);
        mach_timebase_info(&g_tb);
        ane_init();

        printf("=== ANE PerfStats via _ANEPerformanceStatsIOSurface ===\n");

        // Phase 1: Dump all methods on both perf stats classes
        dump_perf_classes();

        // Phase 2: Compile a training-scale kernel
        printf("\n=== Compiling kernel (1024x2048, seq=256) ===\n");
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
        BOOL ok = ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(mdl,
            @selector(compileWithQoS:options:error:), 21, @{}, &e);
        printf("  compile: %s %s\n", ok?"OK":"FAIL", e?[[e description] UTF8String]:"");

        ok = ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(mdl,
            @selector(loadWithQoS:options:error:), 21, @{}, &e);
        printf("  load: %s %s\n", ok?"OK":"FAIL", e?[[e description] UTF8String]:"");

        id aneModel = ((id(*)(id,SEL))objc_msgSend)(mdl, @selector(model));
        [g_keepalive addObject:aneModel];

        // Create IO surfaces
        size_t in_bytes = (size_t)IC * (SEQ + OC) * 2;
        size_t out_bytes = (size_t)OC * SEQ * 2;
        IOSurfaceRef ioIn = make_surf(in_bytes);
        IOSurfaceRef ioOut = make_surf(out_bytes);

        id wI = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_AIO,
            @selector(objectWithIOSurface:), ioIn);
        id wO = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_AIO,
            @selector(objectWithIOSurface:), ioOut);
        [g_keepalive addObject:wI];
        [g_keepalive addObject:wO];

        // Phase 3: Try different statType values with _ANEPerformanceStatsIOSurface
        Class perfStatsIOClass = NSClassFromString(@"_ANEPerformanceStatsIOSurface");
        if (!perfStatsIOClass) {
            printf("ERROR: _ANEPerformanceStatsIOSurface class not found!\n");
            return 1;
        }

        // Try statType values 0, 1, 2, 3
        for (int statType = 0; statType <= 3; statType++) {
            printf("\n=== statType = %d ===\n", statType);

            // Create a stats IOSurface (try various sizes)
            // ANE perf data could be various sizes — try 4KB, 16KB, 64KB
            size_t stats_sizes[] = {4096, 16384, 65536};
            for (int si = 0; si < 3; si++) {
                size_t stats_sz = stats_sizes[si];
                IOSurfaceRef statsSurf = make_surf(stats_sz);

                // Zero the stats surface
                IOSurfaceLock(statsSurf, 0, NULL);
                memset(IOSurfaceGetBaseAddress(statsSurf), 0, stats_sz);
                IOSurfaceUnlock(statsSurf, 0, NULL);

                // Create _ANEPerformanceStatsIOSurface with objectWithIOSurface:statType:
                id perfStats = ((id(*)(Class,SEL,IOSurfaceRef,int))objc_msgSend)(
                    perfStatsIOClass,
                    @selector(objectWithIOSurface:statType:),
                    statsSurf, statType);

                if (!perfStats) {
                    printf("  stats_sz=%zu: objectWithIOSurface returned nil\n", stats_sz);
                    CFRelease(statsSurf);
                    continue;
                }
                [g_keepalive addObject:perfStats];

                // Verify statType
                int gotType = ((int(*)(id,SEL))objc_msgSend)(perfStats, @selector(statType));
                printf("  stats_sz=%zu: created OK, statType=%d\n", stats_sz, gotType);

                // Create request with perfStats
                // _ANERequest has setPerfStats: or similar
                id req = ((id(*)(Class,SEL,id,id,id,id,id,id))objc_msgSend)(g_AR,
                    @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:procedureIndex:),
                    @[wI], @[@0], @[wO], @[@0], nil, @0);
                [g_keepalive addObject:req];

                // Set perfStats on request
                @try {
                    ((void(*)(id,SEL,id))objc_msgSend)(req, @selector(setPerfStats:), perfStats);
                    printf("  setPerfStats: OK\n");
                } @catch (NSException *ex) {
                    printf("  setPerfStats: EXCEPTION: %s\n", [[ex description] UTF8String]);
                    CFRelease(statsSurf);
                    continue;
                }

                // Warmup without stats
                id reqNoStats = ((id(*)(Class,SEL,id,id,id,id,id,id))objc_msgSend)(g_AR,
                    @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:procedureIndex:),
                    @[wI], @[@0], @[wO], @[@0], nil, @0);
                [g_keepalive addObject:reqNoStats];
                for (int i = 0; i < 10; i++) {
                    ((BOOL(*)(id,SEL,id,id,id,unsigned int,NSError**))objc_msgSend)(g_client,
                        @selector(doEvaluateDirectWithModel:options:request:qos:error:),
                        aneModel, @{}, reqNoStats, 21, &e);
                }

                // Run eval with perfStats request
                printf("  Running eval with perfStats...\n");
                e = nil;
                @try {
                    uint64_t t0 = mach_absolute_time();
                    ok = ((BOOL(*)(id,SEL,id,id,id,unsigned int,NSError**))objc_msgSend)(g_client,
                        @selector(doEvaluateDirectWithModel:options:request:qos:error:),
                        aneModel, @{}, req, 21, &e);
                    double dt = ms_t(mach_absolute_time() - t0);
                    printf("  eval: %s (%.3f ms) %s\n", ok?"OK":"FAIL", dt,
                           e?[[e description] UTF8String]:"");
                } @catch (NSException *ex) {
                    printf("  eval EXCEPTION: %s\n", [[ex description] UTF8String]);
                    CFRelease(statsSurf);
                    continue;
                }

                if (!ok) {
                    CFRelease(statsSurf);
                    continue;
                }

                // Read the stats surface
                IOSurfaceLock(statsSurf, kIOSurfaceLockReadOnly, NULL);
                const uint8_t *statsData = (const uint8_t *)IOSurfaceGetBaseAddress(statsSurf);

                // Check if any data was written
                int nonzero = 0;
                for (size_t i = 0; i < stats_sz; i++) {
                    if (statsData[i] != 0) { nonzero++; }
                }
                printf("  Non-zero bytes in stats surface: %d / %zu\n", nonzero, stats_sz);

                if (nonzero > 0) {
                    printf("  === RAW STATS DATA (hex) ===\n");
                    dump_hex(statsData, stats_sz, 512);
                    printf("  === AS uint64 ===\n");
                    dump_uint64(statsData, stats_sz, 64);

                    // Try to interpret common perf counter patterns
                    // ANE might store: cycle counts, DMA bytes, stall counts, etc.
                    const uint64_t *u64 = (const uint64_t *)statsData;
                    size_t u64_count = stats_sz / 8;
                    printf("  === Non-zero uint64 values ===\n");
                    for (size_t i = 0; i < u64_count; i++) {
                        if (u64[i] != 0) {
                            printf("    [%3zu] = %llu (0x%llx)\n", i, u64[i], u64[i]);
                        }
                    }

                    // Run multiple evals and compare — look for counters that increase
                    printf("\n  === Running 10 more evals, checking counter deltas ===\n");

                    // Save current values
                    uint64_t *baseline = (uint64_t *)malloc(stats_sz);
                    memcpy(baseline, statsData, stats_sz);

                    IOSurfaceUnlock(statsSurf, kIOSurfaceLockReadOnly, NULL);

                    for (int run = 0; run < 10; run++) {
                        e = nil;
                        ((BOOL(*)(id,SEL,id,id,id,unsigned int,NSError**))objc_msgSend)(g_client,
                            @selector(doEvaluateDirectWithModel:options:request:qos:error:),
                            aneModel, @{}, req, 21, &e);
                    }

                    IOSurfaceLock(statsSurf, kIOSurfaceLockReadOnly, NULL);
                    statsData = (const uint8_t *)IOSurfaceGetBaseAddress(statsSurf);
                    const uint64_t *u64_after = (const uint64_t *)statsData;

                    printf("  Counters that changed after 10 evals:\n");
                    for (size_t i = 0; i < u64_count; i++) {
                        if (u64_after[i] != baseline[i]) {
                            printf("    [%3zu]: %llu -> %llu (delta=%lld)\n",
                                   i, baseline[i], u64_after[i],
                                   (int64_t)(u64_after[i] - baseline[i]));
                        }
                    }
                    free(baseline);
                } else {
                    printf("  (no data written to stats surface)\n");
                }

                IOSurfaceUnlock(statsSurf, kIOSurfaceLockReadOnly, NULL);

                // Also try reading perfStats properties after eval
                printf("\n  === Reading perfStats properties after eval ===\n");
                @try {
                    // Check for common ANE perf stat accessors
                    SEL selectors[] = {
                        @selector(hwExecutionTime),
                        @selector(performanceCounters),
                        @selector(perfCounterData),
                        @selector(executionTime),
                        @selector(ioSurface),
                        @selector(description),
                    };
                    const char *selNames[] = {
                        "hwExecutionTime", "performanceCounters", "perfCounterData",
                        "executionTime", "ioSurface", "description",
                    };
                    for (int si = 0; si < 6; si++) {
                        if ([perfStats respondsToSelector:selectors[si]]) {
                            id val = ((id(*)(id,SEL))objc_msgSend)(perfStats, selectors[si]);
                            printf("    %s: %s\n", selNames[si],
                                   val ? [[NSString stringWithFormat:@"%@", val] UTF8String] : "(nil)");
                        } else {
                            printf("    %s: (not implemented)\n", selNames[si]);
                        }
                    }
                } @catch (NSException *ex) {
                    printf("    EXCEPTION reading props: %s\n", [[ex description] UTF8String]);
                }

                CFRelease(statsSurf);

                // If we got data, no need to try other sizes for this statType
                if (nonzero > 0) break;
            }
        }

        // Phase 4: Also check _ANERequest for any perf-related properties we can read after eval
        printf("\n=== _ANERequest perf properties after eval ===\n");
        {
            id req = ((id(*)(Class,SEL,id,id,id,id,id,id))objc_msgSend)(g_AR,
                @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:procedureIndex:),
                @[wI], @[@0], @[wO], @[@0], nil, @0);
            [g_keepalive addObject:req];

            e = nil;
            ((BOOL(*)(id,SEL,id,id,id,unsigned int,NSError**))objc_msgSend)(g_client,
                @selector(doEvaluateDirectWithModel:options:request:qos:error:),
                aneModel, @{}, req, 21, &e);

            // Enumerate all _ANERequest properties
            unsigned int pcount = 0;
            objc_property_t *props = class_copyPropertyList(g_AR, &pcount);
            printf("  _ANERequest properties (%d total):\n", pcount);
            for (unsigned int i = 0; i < pcount; i++) {
                const char *pn = property_getName(props[i]);
                printf("    %s\n", pn);
            }
            free(props);

            // Try reading perfStats from request after eval
            @try {
                id ps = ((id(*)(id,SEL))objc_msgSend)(req, @selector(perfStats));
                printf("  perfStats after eval (no setPerfStats): %s\n",
                       ps ? [[NSString stringWithFormat:@"%@", ps] UTF8String] : "(nil)");
            } @catch (NSException *ex) {
                printf("  perfStats: EXCEPTION: %s\n", [[ex description] UTF8String]);
            }
        }

        printf("\n=== Done ===\n");
    }
    return 0;
}
