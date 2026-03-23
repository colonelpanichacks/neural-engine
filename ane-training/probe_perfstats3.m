// probe_perfstats3.m — Deep dive into ANE performance counter data
// Reverse-engineers what perf data ANE actually provides via _ANEPerformanceStats.
//
// Build:
//   xcrun clang -O2 -framework Foundation -framework IOSurface \
//     -isysroot /Library/Developer/CommandLineTools/SDKs/MacOSX.sdk \
//     -fobjc-arc -o probe_perfstats3 probe_perfstats3.m
//
// Run:
//   ./probe_perfstats3

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

// Training-scale matmul: 1024x2048, seq=256
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

// Dump raw bytes as hex
static void dump_hex(const uint8_t *data, size_t len, size_t max_bytes) {
    size_t n = len < max_bytes ? len : max_bytes;
    for (size_t i = 0; i < n; i++) {
        if (i % 32 == 0) printf("    %04zx: ", i);
        printf("%02x ", data[i]);
        if (i % 32 == 31 || i == n-1) printf("\n");
    }
}

// Dump as uint32 array
static void dump_uint32(const uint8_t *data, size_t len, size_t max_u32s) {
    const uint32_t *u32 = (const uint32_t *)data;
    size_t count = len / 4;
    size_t n = count < max_u32s ? count : max_u32s;
    for (size_t i = 0; i < n; i++) {
        if (i % 8 == 0) printf("    [%3zu]: ", i);
        printf("%10u ", u32[i]);
        if (i % 8 == 7 || i == n-1) printf("\n");
    }
}

// Dump as uint64 array
static void dump_uint64(const uint8_t *data, size_t len, size_t max_u64s) {
    const uint64_t *u64 = (const uint64_t *)data;
    size_t count = len / 8;
    size_t n = count < max_u64s ? count : max_u64s;
    for (size_t i = 0; i < n; i++) {
        if (i % 4 == 0) printf("    [%3zu]: ", i);
        printf("%18llu ", u64[i]);
        if (i % 4 == 3 || i == n-1) printf("\n");
    }
}

// Introspect all properties of a class
static void dump_class_properties(Class cls) {
    unsigned int propCount = 0;
    objc_property_t *props = class_copyPropertyList(cls, &propCount);
    printf("  Properties (%u):\n", propCount);
    for (unsigned int i = 0; i < propCount; i++) {
        printf("    %s (%s)\n", property_getName(props[i]), property_getAttributes(props[i]));
    }
    free(props);
}

// Introspect all methods of a class
static void dump_class_methods(Class cls, BOOL instanceMethods) {
    unsigned int methodCount = 0;
    Method *methods = class_copyMethodList(instanceMethods ? cls : object_getClass((id)cls), &methodCount);
    printf("  %s methods (%u):\n", instanceMethods ? "Instance" : "Class", methodCount);
    for (unsigned int i = 0; i < methodCount; i++) {
        printf("    %s\n", sel_getName(method_getName(methods[i])));
    }
    free(methods);
}

// Read all accessible properties from a perfStats object
static void read_perfstats(id ps, const char *label) {
    printf("\n  [%s] Reading all perfStats properties:\n", label);

    // hwExecutionTime
    @try {
        uint64_t hwt = ((uint64_t(*)(id,SEL))objc_msgSend)(ps, @selector(hwExecutionTime));
        printf("    hwExecutionTime: %llu ns (%.3f ms)\n", hwt, hwt / 1e6);
    } @catch (NSException *ex) {
        printf("    hwExecutionTime: EXCEPTION %s\n", [[ex reason] UTF8String]);
    }

    // perfCounterData
    @try {
        id pcd = ((id(*)(id,SEL))objc_msgSend)(ps, @selector(perfCounterData));
        if (pcd) {
            NSData *data = (NSData *)pcd;
            size_t len = [data length];
            printf("    perfCounterData: %zu bytes\n", len);
            if (len > 0) {
                const uint8_t *bytes = (const uint8_t *)[data bytes];
                printf("      Raw hex (first 256 bytes):\n");
                dump_hex(bytes, len, 256);
                printf("      As uint32 (first 64 values):\n");
                dump_uint32(bytes, len, 64);
                printf("      As uint64 (first 32 values):\n");
                dump_uint64(bytes, len, 32);
            }
        } else {
            printf("    perfCounterData: nil\n");
        }
    } @catch (NSException *ex) {
        printf("    perfCounterData: EXCEPTION %s\n", [[ex reason] UTF8String]);
    }

    // performanceCounters
    @try {
        id pc = ((id(*)(id,SEL))objc_msgSend)(ps, @selector(performanceCounters));
        if (pc) {
            printf("    performanceCounters: %s\n", [[pc description] UTF8String]);
            // If it's a dictionary, dump keys
            if ([pc isKindOfClass:[NSDictionary class]]) {
                NSDictionary *dict = (NSDictionary *)pc;
                for (id key in dict) {
                    printf("      %s => %s\n", [[key description] UTF8String], [[dict[key] description] UTF8String]);
                }
            }
            // If it's an array, dump elements
            if ([pc isKindOfClass:[NSArray class]]) {
                NSArray *arr = (NSArray *)pc;
                for (NSUInteger i = 0; i < [arr count]; i++) {
                    printf("      [%lu] %s\n", (unsigned long)i, [[arr[i] description] UTF8String]);
                }
            }
        } else {
            printf("    performanceCounters: nil\n");
        }
    } @catch (NSException *ex) {
        printf("    performanceCounters: EXCEPTION %s\n", [[ex reason] UTF8String]);
    }

    // pStatsRawData
    @try {
        id psr = ((id(*)(id,SEL))objc_msgSend)(ps, @selector(pStatsRawData));
        if (psr) {
            NSData *data = (NSData *)psr;
            size_t len = [data length];
            printf("    pStatsRawData: %zu bytes\n", len);
            if (len > 0) {
                const uint8_t *bytes = (const uint8_t *)[data bytes];
                printf("      Raw hex (first 256 bytes):\n");
                dump_hex(bytes, len, 256);
                printf("      As uint32 (first 64 values):\n");
                dump_uint32(bytes, len, 64);
            }
        } else {
            printf("    pStatsRawData: nil\n");
        }
    } @catch (NSException *ex) {
        printf("    pStatsRawData: EXCEPTION %s\n", [[ex reason] UTF8String]);
    }

    // Try additional selectors that might exist
    SEL extraSels[] = {
        @selector(description),
        @selector(statsDescription),
        @selector(hardwareExecutionNS),
        @selector(executionTime),
        @selector(programExecutionTime),
        @selector(totalTime),
        @selector(hwTime),
        @selector(counters),
        @selector(rawData),
        @selector(statsData),
        @selector(perfData),
    };
    const char *extraNames[] = {
        "description", "statsDescription", "hardwareExecutionNS",
        "executionTime", "programExecutionTime", "totalTime", "hwTime",
        "counters", "rawData", "statsData", "perfData"
    };
    printf("    --- extra selector probing ---\n");
    for (int i = 0; i < (int)(sizeof(extraSels)/sizeof(extraSels[0])); i++) {
        if ([ps respondsToSelector:extraSels[i]]) {
            @try {
                id val = ((id(*)(id,SEL))objc_msgSend)(ps, extraSels[i]);
                printf("    %s: responds=YES, value=%s\n", extraNames[i],
                       val ? [[val description] UTF8String] : "nil");
            } @catch (NSException *ex) {
                printf("    %s: responds=YES, EXCEPTION\n", extraNames[i]);
            }
        }
    }
}

int main(int argc, char *argv[]) {
    @autoreleasepool {
        setbuf(stdout, NULL);
        mach_timebase_info(&g_tb);
        ane_init();

        Class perfClass = NSClassFromString(@"_ANEPerformanceStats");

        int IC = 1024, OC = 2048, SEQ = 256;
        printf("============================================================\n");
        printf("  ANE Performance Stats Deep Probe (probe_perfstats3)\n");
        printf("============================================================\n");
        printf("Kernel: %dx%d matmul, seq=%d\n", IC, OC, SEQ);
        printf("Input: %d x %d = %zu bytes\n", IC, SEQ + OC, (size_t)IC * (SEQ + OC) * 2);
        printf("Output: %d x %d = %zu bytes\n\n", OC, SEQ, (size_t)OC * SEQ * 2);

        // ===== 1. Introspect _ANEPerformanceStats class =====
        printf("=== 1. _ANEPerformanceStats Class Introspection ===\n");
        if (!perfClass) {
            printf("  _ANEPerformanceStats class NOT FOUND. Aborting.\n");
            return 1;
        }
        dump_class_properties(perfClass);
        dump_class_methods(perfClass, YES);
        dump_class_methods(perfClass, NO);

        // Also introspect _ANERequest for perfStats-related properties
        printf("\n=== _ANERequest perfStats-related properties ===\n");
        {
            unsigned int propCount = 0;
            objc_property_t *props = class_copyPropertyList(g_AR, &propCount);
            for (unsigned int i = 0; i < propCount; i++) {
                const char *name = property_getName(props[i]);
                if (strcasestr(name, "perf") || strcasestr(name, "stat") || strcasestr(name, "counter")) {
                    printf("  %s (%s)\n", name, property_getAttributes(props[i]));
                }
            }
            free(props);
        }

        // ===== 2. Compile and load model =====
        printf("\n=== 2. Compile + Load Training-Scale Matmul ===\n");
        NSData *mil = gen_matmul(IC, OC, SEQ);
        id desc = ((id(*)(Class,SEL,id,id,id))objc_msgSend)(g_D, @selector(modelWithMILText:weights:optionsPlist:), mil, @{}, nil);
        id mdl = ((id(*)(Class,SEL,id))objc_msgSend)(g_I, @selector(inMemoryModelWithDescriptor:), desc);
        id hx = ((id(*)(id,SEL))objc_msgSend)(mdl, @selector(hexStringIdentifier));
        NSString *td = [NSTemporaryDirectory() stringByAppendingPathComponent:hx];
        [[NSFileManager defaultManager] createDirectoryAtPath:[td stringByAppendingPathComponent:@"weights"]
                                  withIntermediateDirectories:YES attributes:nil error:nil];
        [mil writeToFile:[td stringByAppendingPathComponent:@"model.mil"] atomically:YES];

        NSError *e = nil;
        ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(mdl, @selector(compileWithQoS:options:error:), 21, @{}, &e);
        if (e) { printf("  Compile ERROR: %s\n", [[e description] UTF8String]); return 1; }
        ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(mdl, @selector(loadWithQoS:options:error:), 21, @{}, &e);
        if (e) { printf("  Load ERROR: %s\n", [[e description] UTF8String]); return 1; }
        printf("  Compile+Load OK\n");

        id aneModel = ((id(*)(id,SEL))objc_msgSend)(mdl, @selector(model));

        // Set perfStatsMask to 0xF (enable all counters)
        ((void(*)(id,SEL,unsigned int))objc_msgSend)(aneModel, @selector(setPerfStatsMask:), 0xF);
        unsigned int mask = ((unsigned int(*)(id,SEL))objc_msgSend)(aneModel, @selector(perfStatsMask));
        printf("  perfStatsMask set to 0xF, readback: 0x%x\n", mask);

        // Try higher mask values too
        for (unsigned int testMask = 0x10; testMask <= 0x100; testMask <<= 1) {
            ((void(*)(id,SEL,unsigned int))objc_msgSend)(aneModel, @selector(setPerfStatsMask:), testMask);
            unsigned int rb = ((unsigned int(*)(id,SEL))objc_msgSend)(aneModel, @selector(perfStatsMask));
            if (rb != 0) printf("  perfStatsMask 0x%x -> readback 0x%x\n", testMask, rb);
        }
        // Reset to 0xFF (try maximum)
        ((void(*)(id,SEL,unsigned int))objc_msgSend)(aneModel, @selector(setPerfStatsMask:), 0xFF);
        mask = ((unsigned int(*)(id,SEL))objc_msgSend)(aneModel, @selector(perfStatsMask));
        printf("  perfStatsMask set to 0xFF, readback: 0x%x\n", mask);

        // Create IOSurfaces
        size_t in_bytes = (size_t)IC * (SEQ + OC) * 2;
        size_t out_bytes = (size_t)OC * SEQ * 2;
        IOSurfaceRef ioIn = make_surf(in_bytes);
        IOSurfaceRef ioOut = make_surf(out_bytes);
        _Float16 *inp = (_Float16 *)IOSurfaceGetBaseAddress(ioIn);
        for (size_t i = 0; i < in_bytes / 2; i++) inp[i] = (_Float16)(0.01f * (i % 100));

        id wI = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_AIO, @selector(objectWithIOSurface:), ioIn);
        id wO = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_AIO, @selector(objectWithIOSurface:), ioOut);

        // ===== 3. Create perfStats and eval with it =====
        printf("\n=== 3. perfStats with _ANEPerformanceStats Object ===\n");
        {
            id ps = ((id(*)(Class,SEL,uint64_t))objc_msgSend)(perfClass, @selector(statsWithHardwareExecutionNS:), (uint64_t)0);
            printf("  Created _ANEPerformanceStats: %s\n", ps ? "OK" : "FAILED");
            if (!ps) goto test4;

            // Read initial state
            read_perfstats(ps, "before any eval");

            // Create request with perfStats
            id req = ((id(*)(Class,SEL,id,id,id,id,id,id,id))objc_msgSend)(g_AR,
                @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:),
                @[wI], @[@0], @[wO], @[@0], nil, @[ps], @0);

            // Single warmup eval
            e = nil;
            ((BOOL(*)(id,SEL,id,id,id,NSError**))objc_msgSend)(g_client,
                @selector(evaluateRealTimeWithModel:options:request:error:), aneModel, @{}, req, &e);
            if (e) printf("  Warmup eval ERROR: %s\n", [[e description] UTF8String]);

            read_perfstats(ps, "after 1 warmup eval");

            // ===== Run 10 evals, reading counters between each =====
            printf("\n=== 3b. Counter Changes Across 10 Evals ===\n");

            // Storage for tracking counter data across evals
            NSData *prevPerfCounterData = nil;
            NSData *prevRawData = nil;
            uint64_t prevHwt = 0;

            @try {
                prevHwt = ((uint64_t(*)(id,SEL))objc_msgSend)(ps, @selector(hwExecutionTime));
            } @catch (NSException *ex) {}
            @try {
                id pcd = ((id(*)(id,SEL))objc_msgSend)(ps, @selector(perfCounterData));
                if (pcd) prevPerfCounterData = [pcd copy];
            } @catch (NSException *ex) {}
            @try {
                id psr = ((id(*)(id,SEL))objc_msgSend)(ps, @selector(pStatsRawData));
                if (psr) prevRawData = [psr copy];
            } @catch (NSException *ex) {}

            for (int eval_i = 0; eval_i < 10; eval_i++) {
                e = nil;
                uint64_t t0 = mach_absolute_time();
                ((BOOL(*)(id,SEL,id,id,id,NSError**))objc_msgSend)(g_client,
                    @selector(evaluateRealTimeWithModel:options:request:error:), aneModel, @{}, req, &e);
                double wall = ms_t(mach_absolute_time() - t0);

                uint64_t hwt = 0;
                @try {
                    hwt = ((uint64_t(*)(id,SEL))objc_msgSend)(ps, @selector(hwExecutionTime));
                } @catch (NSException *ex) {}

                printf("  eval %2d: wall=%.3fms  hwExecTime=%llu ns (%.3fms)  delta=%llu ns (%.3fms)\n",
                       eval_i, wall, hwt, hwt / 1e6, hwt - prevHwt, (hwt - prevHwt) / 1e6);

                // Check perfCounterData changes
                @try {
                    id pcd = ((id(*)(id,SEL))objc_msgSend)(ps, @selector(perfCounterData));
                    if (pcd && [(NSData *)pcd length] > 0) {
                        NSData *data = (NSData *)pcd;
                        if (prevPerfCounterData && [prevPerfCounterData length] == [data length]) {
                            const uint32_t *cur = (const uint32_t *)[data bytes];
                            const uint32_t *prev = (const uint32_t *)[prevPerfCounterData bytes];
                            size_t count = [data length] / 4;
                            size_t n = count < 64 ? count : 64;
                            BOOL anyChange = NO;
                            for (size_t j = 0; j < n; j++) {
                                if (cur[j] != prev[j]) {
                                    if (!anyChange) { printf("    perfCounterData changes (uint32):\n"); anyChange = YES; }
                                    printf("      [%2zu]: %10u -> %10u (delta: %d)\n",
                                           j, prev[j], cur[j], (int32_t)(cur[j] - prev[j]));
                                }
                            }
                            if (!anyChange && eval_i == 0) printf("    perfCounterData: no changes from previous\n");
                        } else if (eval_i == 0) {
                            printf("    perfCounterData: %zu bytes (first time with data)\n", [data length]);
                        }
                        prevPerfCounterData = [data copy];
                    }
                } @catch (NSException *ex) {}

                // Check pStatsRawData changes
                @try {
                    id psr = ((id(*)(id,SEL))objc_msgSend)(ps, @selector(pStatsRawData));
                    if (psr && [(NSData *)psr length] > 0) {
                        NSData *data = (NSData *)psr;
                        if (prevRawData && [prevRawData length] == [data length]) {
                            const uint32_t *cur = (const uint32_t *)[data bytes];
                            const uint32_t *prev = (const uint32_t *)[prevRawData bytes];
                            size_t count = [data length] / 4;
                            size_t n = count < 64 ? count : 64;
                            BOOL anyChange = NO;
                            for (size_t j = 0; j < n; j++) {
                                if (cur[j] != prev[j]) {
                                    if (!anyChange) { printf("    pStatsRawData changes (uint32):\n"); anyChange = YES; }
                                    printf("      [%2zu]: %10u -> %10u (delta: %d)\n",
                                           j, prev[j], cur[j], (int32_t)(cur[j] - prev[j]));
                                }
                            }
                            if (!anyChange && eval_i == 0) printf("    pStatsRawData: no changes from previous\n");
                        }
                        prevRawData = [data copy];
                    }
                } @catch (NSException *ex) {}

                prevHwt = hwt;
            }

            // Full dump after all 10 evals
            read_perfstats(ps, "after 10 evals total (+ warmup)");
        }

        // ===== 4. Test: fresh perfStats per eval (not accumulated) =====
        test4:
        printf("\n=== 4. Fresh PerfStats Per Eval (Non-Accumulated) ===\n");
        for (int i = 0; i < 3; i++) {
            id ps_fresh = ((id(*)(Class,SEL,uint64_t))objc_msgSend)(perfClass, @selector(statsWithHardwareExecutionNS:), (uint64_t)0);
            if (!ps_fresh) { printf("  Failed to create fresh perfStats\n"); break; }

            id req_fresh = ((id(*)(Class,SEL,id,id,id,id,id,id,id))objc_msgSend)(g_AR,
                @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:),
                @[wI], @[@0], @[wO], @[@0], nil, @[ps_fresh], @0);

            e = nil;
            ((BOOL(*)(id,SEL,id,id,id,NSError**))objc_msgSend)(g_client,
                @selector(evaluateRealTimeWithModel:options:request:error:), aneModel, @{}, req_fresh, &e);

            uint64_t hwt = 0;
            @try {
                hwt = ((uint64_t(*)(id,SEL))objc_msgSend)(ps_fresh, @selector(hwExecutionTime));
            } @catch (NSException *ex) {}
            printf("  fresh eval %d: hwExecutionTime=%llu ns (%.3fms)\n", i, hwt, hwt / 1e6);

            @try {
                id pcd = ((id(*)(id,SEL))objc_msgSend)(ps_fresh, @selector(perfCounterData));
                if (pcd) printf("    perfCounterData: %zu bytes\n", [(NSData *)pcd length]);
                else printf("    perfCounterData: nil\n");
            } @catch (NSException *ex) {
                printf("    perfCounterData: EXCEPTION\n");
            }
        }

        // ===== 5. Test: NSMutableData as perfStats =====
        printf("\n=== 5. NSMutableData as perfStats (instead of _ANEPerformanceStats) ===\n");
        {
            // Try with a plain NSMutableData object
            for (int sz = 0; sz < 3; sz++) {
                size_t dataSize = (sz == 0) ? 64 : (sz == 1) ? 256 : 4096;
                NSMutableData *fakeStats = [NSMutableData dataWithLength:dataSize];
                printf("  Testing NSMutableData(%zu bytes) as perfStats:\n", dataSize);

                @try {
                    id req_fake = ((id(*)(Class,SEL,id,id,id,id,id,id,id))objc_msgSend)(g_AR,
                        @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:),
                        @[wI], @[@0], @[wO], @[@0], nil, @[fakeStats], @0);

                    e = nil;
                    BOOL ok = ((BOOL(*)(id,SEL,id,id,id,NSError**))objc_msgSend)(g_client,
                        @selector(evaluateRealTimeWithModel:options:request:error:), aneModel, @{}, req_fake, &e);
                    printf("    eval result: %s\n", ok ? "OK" : "FAIL");
                    if (e) printf("    error: %s\n", [[e description] UTF8String]);

                    if (ok) {
                        // Check if the NSMutableData was written to
                        const uint8_t *bytes = (const uint8_t *)[fakeStats bytes];
                        BOOL anyNonZero = NO;
                        for (size_t j = 0; j < dataSize; j++) {
                            if (bytes[j] != 0) { anyNonZero = YES; break; }
                        }
                        printf("    data modified: %s\n", anyNonZero ? "YES" : "NO (all zeros)");
                        if (anyNonZero) {
                            printf("    hex dump (first 128 bytes):\n");
                            dump_hex(bytes, dataSize, 128);
                        }
                    }
                } @catch (NSException *ex) {
                    printf("    EXCEPTION: %s\n", [[ex reason] UTF8String]);
                }
            }
        }

        // ===== 6. Test: nil perfStats array vs empty array =====
        printf("\n=== 6. nil vs Empty Array vs Array of NSNull ===\n");
        {
            struct { id val; const char *name; } tests[] = {
                { nil,              "nil" },
                { @[],              "@[]" },
                { @[[NSNull null]], "@[NSNull]" },
            };
            for (int t = 0; t < 3; t++) {
                @try {
                    id req_t = ((id(*)(Class,SEL,id,id,id,id,id,id,id))objc_msgSend)(g_AR,
                        @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:),
                        @[wI], @[@0], @[wO], @[@0], nil, tests[t].val, @0);

                    e = nil;
                    BOOL ok = ((BOOL(*)(id,SEL,id,id,id,NSError**))objc_msgSend)(g_client,
                        @selector(evaluateRealTimeWithModel:options:request:error:), aneModel, @{}, req_t, &e);
                    printf("  perfStats=%s: eval %s", tests[t].name, ok ? "OK" : "FAIL");
                    if (e) printf(" (%s)", [[e description] UTF8String]);
                    printf("\n");
                } @catch (NSException *ex) {
                    printf("  perfStats=%s: EXCEPTION %s\n", tests[t].name, [[ex reason] UTF8String]);
                }
            }
        }

        // ===== 7. Test: IOSurface as stats buffer =====
        printf("\n=== 7. IOSurface as Stats Buffer ===\n");
        {
            // Some ANE APIs use a separate IOSurface for stats output.
            // Try creating one and passing it via _ANEIOSurfaceObject in perfStats array.
            for (int sz = 0; sz < 3; sz++) {
                size_t statsSize = (sz == 0) ? 256 : (sz == 1) ? 4096 : 16384;
                IOSurfaceRef statsSurf = make_surf(statsSize);
                memset(IOSurfaceGetBaseAddress(statsSurf), 0, statsSize);

                id statsObj = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_AIO, @selector(objectWithIOSurface:), statsSurf);
                printf("  Testing _ANEIOSurfaceObject(%zu bytes) as perfStats:\n", statsSize);

                @try {
                    id req_s = ((id(*)(Class,SEL,id,id,id,id,id,id,id))objc_msgSend)(g_AR,
                        @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:),
                        @[wI], @[@0], @[wO], @[@0], nil, @[statsObj], @0);

                    e = nil;
                    BOOL ok = ((BOOL(*)(id,SEL,id,id,id,NSError**))objc_msgSend)(g_client,
                        @selector(evaluateRealTimeWithModel:options:request:error:), aneModel, @{}, req_s, &e);
                    printf("    eval result: %s\n", ok ? "OK" : "FAIL");
                    if (e) printf("    error: %s\n", [[e description] UTF8String]);

                    if (ok) {
                        const uint8_t *bytes = (const uint8_t *)IOSurfaceGetBaseAddress(statsSurf);
                        BOOL anyNonZero = NO;
                        for (size_t j = 0; j < statsSize; j++) {
                            if (bytes[j] != 0) { anyNonZero = YES; break; }
                        }
                        printf("    IOSurface modified: %s\n", anyNonZero ? "YES" : "NO (all zeros)");
                        if (anyNonZero) {
                            printf("    hex dump (first 256 bytes):\n");
                            dump_hex(bytes, statsSize, 256);
                            printf("    as uint32 (first 64):\n");
                            dump_uint32(bytes, statsSize, 64);
                            printf("    as uint64 (first 32):\n");
                            dump_uint64(bytes, statsSize, 32);
                        }
                    }
                } @catch (NSException *ex) {
                    printf("    EXCEPTION: %s\n", [[ex reason] UTF8String]);
                }
                CFRelease(statsSurf);
            }
        }

        // ===== 8. Test: doEvaluateDirectWithModel + perfStats =====
        printf("\n=== 8. doEvaluateDirectWithModel + perfStats ===\n");
        {
            id ps_direct = ((id(*)(Class,SEL,uint64_t))objc_msgSend)(perfClass, @selector(statsWithHardwareExecutionNS:), (uint64_t)0);
            if (ps_direct) {
                id req_d = ((id(*)(Class,SEL,id,id,id,id,id,id,id))objc_msgSend)(g_AR,
                    @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:),
                    @[wI], @[@0], @[wO], @[@0], nil, @[ps_direct], @0);

                @try {
                    e = nil;
                    BOOL ok = ((BOOL(*)(id,SEL,id,id,id,unsigned int,NSError**))objc_msgSend)(g_client,
                        @selector(doEvaluateDirectWithModel:options:request:qos:error:),
                        aneModel, @{}, req_d, 21, &e);
                    printf("  doEvaluateDirect eval: %s\n", ok ? "OK" : "FAIL");
                    if (e) printf("  error: %s\n", [[e description] UTF8String]);

                    if (ok) {
                        // Do 10 evals and check
                        for (int i = 0; i < 10; i++) {
                            ((BOOL(*)(id,SEL,id,id,id,unsigned int,NSError**))objc_msgSend)(g_client,
                                @selector(doEvaluateDirectWithModel:options:request:qos:error:),
                                aneModel, @{}, req_d, 21, &e);
                        }
                        read_perfstats(ps_direct, "doEvaluateDirect after 10 evals");
                    }
                } @catch (NSException *ex) {
                    printf("  EXCEPTION: %s\n", [[ex reason] UTF8String]);
                }
            }
        }

        // ===== 9. Varying perfStatsMask and checking counter output =====
        printf("\n=== 9. perfStatsMask Sweep: Which Mask Values Produce Data? ===\n");
        {
            unsigned int maskValues[] = { 0x0, 0x1, 0x2, 0x3, 0x4, 0x7, 0x8, 0xF, 0x10, 0x1F, 0x3F, 0x7F, 0xFF, 0xFFFF, 0xFFFFFFFF };
            for (int mi = 0; mi < (int)(sizeof(maskValues)/sizeof(maskValues[0])); mi++) {
                unsigned int mv = maskValues[mi];
                ((void(*)(id,SEL,unsigned int))objc_msgSend)(aneModel, @selector(setPerfStatsMask:), mv);
                unsigned int rb = ((unsigned int(*)(id,SEL))objc_msgSend)(aneModel, @selector(perfStatsMask));

                id ps_m = ((id(*)(Class,SEL,uint64_t))objc_msgSend)(perfClass, @selector(statsWithHardwareExecutionNS:), (uint64_t)0);
                id req_m = ((id(*)(Class,SEL,id,id,id,id,id,id,id))objc_msgSend)(g_AR,
                    @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:),
                    @[wI], @[@0], @[wO], @[@0], nil, @[ps_m], @0);

                e = nil;
                ((BOOL(*)(id,SEL,id,id,id,NSError**))objc_msgSend)(g_client,
                    @selector(evaluateRealTimeWithModel:options:request:error:), aneModel, @{}, req_m, &e);

                uint64_t hwt = 0;
                size_t pcdLen = 0;
                size_t psrLen = 0;
                @try { hwt = ((uint64_t(*)(id,SEL))objc_msgSend)(ps_m, @selector(hwExecutionTime)); } @catch (NSException *ex) {}
                @try {
                    id pcd = ((id(*)(id,SEL))objc_msgSend)(ps_m, @selector(perfCounterData));
                    if (pcd) pcdLen = [(NSData *)pcd length];
                } @catch (NSException *ex) {}
                @try {
                    id psr = ((id(*)(id,SEL))objc_msgSend)(ps_m, @selector(pStatsRawData));
                    if (psr) psrLen = [(NSData *)psr length];
                } @catch (NSException *ex) {}

                printf("  mask=0x%08x (rb=0x%08x): hwt=%llu ns  pcd=%zu bytes  raw=%zu bytes\n",
                       mv, rb, hwt, pcdLen, psrLen);
            }
        }

        CFRelease(ioIn);
        CFRelease(ioOut);
        printf("\n============================================================\n");
        printf("  Done.\n");
        printf("============================================================\n");
    }
    return 0;
}
