// probe_perfmask.m — Enable ANE perf counters via _ANEModel.perfStatsMask
//
// Build:
//   xcrun clang -O2 -framework Foundation -framework IOSurface \
//     -isysroot /Library/Developer/CommandLineTools/SDKs/MacOSX.sdk \
//     -fobjc-arc -o probe_perfmask probe_perfmask.m

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

int main(int argc, char *argv[]) {
    @autoreleasepool {
        setbuf(stdout, NULL);
        mach_timebase_info(&g_tb);
        ane_init();

        printf("=== ANE PerfStats via perfStatsMask ===\n");

        // Check driverMaskForANEFMask
        Class perfStatsClass = NSClassFromString(@"_ANEPerformanceStats");
        printf("\ndriverMaskForANEFMask values:\n");
        for (unsigned int anefMask = 0; anefMask <= 16; anefMask++) {
            unsigned int drvMask = ((unsigned int(*)(Class,SEL,unsigned int))objc_msgSend)(
                perfStatsClass, @selector(driverMaskForANEFMask:), anefMask);
            if (drvMask != 0 || anefMask <= 4)
                printf("  ANEFMask=0x%x -> driverMask=0x%x\n", anefMask, drvMask);
        }
        // Try powers of 2
        for (int bit = 0; bit < 32; bit++) {
            unsigned int anefMask = 1u << bit;
            unsigned int drvMask = ((unsigned int(*)(Class,SEL,unsigned int))objc_msgSend)(
                perfStatsClass, @selector(driverMaskForANEFMask:), anefMask);
            if (drvMask != 0)
                printf("  ANEFMask=0x%x -> driverMask=0x%x\n", anefMask, drvMask);
        }
        // Try all-bits
        {
            unsigned int drvMask = ((unsigned int(*)(Class,SEL,unsigned int))objc_msgSend)(
                perfStatsClass, @selector(driverMaskForANEFMask:), 0xFFFFFFFF);
            printf("  ANEFMask=0xffffffff -> driverMask=0x%x\n", drvMask);
        }

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

        // Read default perfStatsMask
        unsigned int defaultMask = ((unsigned int(*)(id,SEL))objc_msgSend)(aneModel, @selector(perfStatsMask));
        printf("\nDefault perfStatsMask on _ANEModel: 0x%x\n", defaultMask);

        // Also check on _ANEInMemoryModel
        unsigned int immMask = ((unsigned int(*)(id,SEL))objc_msgSend)(mdl, @selector(perfStatsMask));
        printf("Default perfStatsMask on _ANEInMemoryModel: 0x%x\n", immMask);

        size_t in_bytes = (size_t)IC * (SEQ + OC) * 2;
        size_t out_bytes = (size_t)OC * SEQ * 2;
        IOSurfaceRef ioIn = make_surf(in_bytes);
        IOSurfaceRef ioOut = make_surf(out_bytes);
        id wI = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_AIO, @selector(objectWithIOSurface:), ioIn);
        id wO = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_AIO, @selector(objectWithIOSurface:), ioOut);
        [g_keepalive addObject:wI];
        [g_keepalive addObject:wO];

        // Try setting perfStatsMask and evaluating
        unsigned int masks[] = {0x1, 0x3, 0x7, 0xF, 0xFF, 0xFFFF, 0xFFFFFFFF};
        int nMasks = sizeof(masks)/sizeof(masks[0]);

        Class perfStatsIOClass = NSClassFromString(@"_ANEPerformanceStatsIOSurface");

        for (int mi = 0; mi < nMasks; mi++) {
            unsigned int mask = masks[mi];
            printf("\n=== perfStatsMask = 0x%x ===\n", mask);

            // Set mask on model
            @try {
                ((void(*)(id,SEL,unsigned int))objc_msgSend)(aneModel, @selector(setPerfStatsMask:), mask);
                unsigned int readback = ((unsigned int(*)(id,SEL))objc_msgSend)(aneModel, @selector(perfStatsMask));
                printf("  Set OK, readback=0x%x\n", readback);
            } @catch (NSException *ex) {
                printf("  EXCEPTION setting mask: %s\n", [[ex description] UTF8String]);
                continue;
            }

            // Also set on _ANEInMemoryModel
            ((void(*)(id,SEL,unsigned int))objc_msgSend)(mdl, @selector(setPerfStatsMask:), mask);

            // Create stats IOSurface
            size_t stats_sz = 65536;
            IOSurfaceRef statsSurf = make_surf(stats_sz);
            IOSurfaceLock(statsSurf, 0, NULL);
            memset(IOSurfaceGetBaseAddress(statsSurf), 0, stats_sz);
            IOSurfaceUnlock(statsSurf, 0, NULL);

            // Try both statType 0 and 1
            for (int st = 0; st <= 1; st++) {
                printf("  --- statType=%d ---\n", st);

                id perfStats = ((id(*)(Class,SEL,IOSurfaceRef,int))objc_msgSend)(
                    perfStatsIOClass, @selector(objectWithIOSurface:statType:), statsSurf, st);
                [g_keepalive addObject:perfStats];

                id req = ((id(*)(Class,SEL,id,id,id,id,id,id))objc_msgSend)(g_AR,
                    @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:procedureIndex:),
                    @[wI], @[@0], @[wO], @[@0], nil, @0);
                [g_keepalive addObject:req];
                ((void(*)(id,SEL,id))objc_msgSend)(req, @selector(setPerfStats:), perfStats);

                e = nil;
                @try {
                    // Use XPC path
                    BOOL ok = ((BOOL(*)(id,SEL,id,id,id,unsigned int,NSError**))objc_msgSend)(g_client,
                        @selector(evaluateWithModel:options:request:qos:error:),
                        aneModel, @{}, req, 21, &e);
                    printf("  eval (XPC): %s %s\n", ok?"OK":"FAIL", e?[[e description] UTF8String]:"");

                    if (!ok) {
                        // Try direct path
                        e = nil;
                        ok = ((BOOL(*)(id,SEL,id,id,id,unsigned int,NSError**))objc_msgSend)(g_client,
                            @selector(doEvaluateDirectWithModel:options:request:qos:error:),
                            aneModel, @{}, req, 21, &e);
                        printf("  eval (direct): %s %s\n", ok?"OK":"FAIL", e?[[e description] UTF8String]:"");
                    }
                } @catch (NSException *ex) {
                    printf("  eval EXCEPTION: %s\n", [[ex description] UTF8String]);
                    continue;
                }

                // Check stats surface
                IOSurfaceLock(statsSurf, kIOSurfaceLockReadOnly, NULL);
                const uint8_t *data = (const uint8_t *)IOSurfaceGetBaseAddress(statsSurf);
                int nonzero = 0;
                for (size_t i = 0; i < stats_sz; i++) if (data[i]) nonzero++;
                printf("  Stats surface non-zero: %d bytes\n", nonzero);

                if (nonzero > 0) {
                    printf("  === RAW STATS (first 256 bytes) ===\n");
                    dump_hex(data, stats_sz, 256);

                    // Show non-zero uint64 values
                    const uint64_t *u64 = (const uint64_t *)data;
                    printf("  === Non-zero uint64 values ===\n");
                    for (size_t i = 0; i < stats_sz/8; i++) {
                        if (u64[i]) printf("    [%3zu] = %llu (0x%llx)\n", i, u64[i], u64[i]);
                    }
                }
                IOSurfaceUnlock(statsSurf, kIOSurfaceLockReadOnly, NULL);

                // Check request properties after eval
                @try {
                    id ps = ((id(*)(id,SEL))objc_msgSend)(req, @selector(perfStats));
                    if (ps && [ps respondsToSelector:@selector(hwExecutionTime)]) {
                        uint64_t hwt = ((uint64_t(*)(id,SEL))objc_msgSend)(ps, @selector(hwExecutionTime));
                        printf("  hwExecutionTime: %llu ns\n", hwt);
                    }
                    id psa = ((id(*)(id,SEL))objc_msgSend)(req, @selector(perfStatsArray));
                    if (psa) {
                        NSArray *arr = (NSArray *)psa;
                        printf("  perfStatsArray: %lu items\n", (unsigned long)[arr count]);
                        for (id item in arr) {
                            printf("    class=%s\n", class_getName(object_getClass(item)));
                            if ([item respondsToSelector:@selector(hwExecutionTime)]) {
                                uint64_t hwt = ((uint64_t(*)(id,SEL))objc_msgSend)(item, @selector(hwExecutionTime));
                                printf("    hwExecutionTime: %llu ns\n", hwt);
                            }
                            if ([item respondsToSelector:@selector(perfCounterData)]) {
                                id pcd = ((id(*)(id,SEL))objc_msgSend)(item, @selector(perfCounterData));
                                printf("    perfCounterData: %s\n", pcd ? [[pcd description] UTF8String] : "(nil)");
                            }
                            if ([item respondsToSelector:@selector(performanceCounters)]) {
                                id pc = ((id(*)(id,SEL))objc_msgSend)(item, @selector(performanceCounters));
                                printf("    performanceCounters: %s\n", pc ? [[pc description] UTF8String] : "(nil)");
                            }
                        }
                    }
                } @catch (NSException *ex) {
                    printf("  read perf EXCEPTION: %s\n", [[ex description] UTF8String]);
                }

                // Zero stats surface for next iteration
                IOSurfaceLock(statsSurf, 0, NULL);
                memset(IOSurfaceGetBaseAddress(statsSurf), 0, stats_sz);
                IOSurfaceUnlock(statsSurf, 0, NULL);
            }

            CFRelease(statsSurf);
        }

        // Reset mask
        ((void(*)(id,SEL,unsigned int))objc_msgSend)(aneModel, @selector(setPerfStatsMask:), 0);

        printf("\n=== Done ===\n");
    }
    return 0;
}
