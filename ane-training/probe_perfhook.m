// probe_perfhook.m — Swizzle _ANEVirtualClient.updatePerformanceStats to intercept
// driver perf data, and try request factory methods with perfStats parameter
//
// Build:
//   xcrun clang -O2 -framework Foundation -framework IOSurface \
//     -isysroot /Library/Developer/CommandLineTools/SDKs/MacOSX.sdk \
//     -fobjc-arc -o probe_perfhook probe_perfhook.m

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

// Original IMP storage
static IMP g_origUpdatePerfStats = NULL;

// Swizzled updatePerformanceStats
static void hooked_updatePerformanceStats(id self, SEL _cmd,
    void *perfStats, unsigned int perfStatsLen,
    IOSurfaceRef rawSurf, unsigned int rawLen,
    uint64_t hwExecTime) {
    printf("  *** HOOKED updatePerformanceStats ***\n");
    printf("    perfStats=%p len=%u\n", perfStats, perfStatsLen);
    printf("    rawSurf=%p rawLen=%u\n", (void*)rawSurf, rawLen);
    printf("    hwExecutionTime=%llu ns (%.3f ms)\n", hwExecTime, hwExecTime/1e6);

    if (perfStats && perfStatsLen > 0) {
        const uint8_t *data = (const uint8_t *)perfStats;
        printf("    perfStats hex (first 128 bytes):\n");
        size_t n = perfStatsLen < 128 ? perfStatsLen : 128;
        for (size_t i = 0; i < n; i++) {
            if (i % 32 == 0) printf("      %04zx: ", i);
            printf("%02x ", data[i]);
            if (i % 32 == 31 || i == n-1) printf("\n");
        }
    }

    if (rawSurf) {
        size_t sz = IOSurfaceGetAllocSize(rawSurf);
        printf("    rawSurf size: %zu bytes\n", sz);
        IOSurfaceLock(rawSurf, kIOSurfaceLockReadOnly, NULL);
        const uint8_t *data = (const uint8_t *)IOSurfaceGetBaseAddress(rawSurf);
        int nonzero = 0;
        for (size_t i = 0; i < sz && i < 65536; i++) if (data[i]) nonzero++;
        printf("    rawSurf non-zero: %d bytes\n", nonzero);
        if (nonzero > 0) {
            printf("    rawSurf hex (first 128 bytes):\n");
            size_t n = sz < 128 ? sz : 128;
            for (size_t i = 0; i < n; i++) {
                if (i % 32 == 0) printf("      %04zx: ", i);
                printf("%02x ", data[i]);
                if (i % 32 == 31 || i == n-1) printf("\n");
            }
        }
        IOSurfaceUnlock(rawSurf, kIOSurfaceLockReadOnly, NULL);
    }

    // Call original
    if (g_origUpdatePerfStats) {
        ((void(*)(id,SEL,void*,unsigned int,IOSurfaceRef,unsigned int,uint64_t))g_origUpdatePerfStats)(
            self, _cmd, perfStats, perfStatsLen, rawSurf, rawLen, hwExecTime);
    }
}

int main(int argc, char *argv[]) {
    @autoreleasepool {
        setbuf(stdout, NULL);
        mach_timebase_info(&g_tb);
        ane_init();

        printf("=== ANE PerfStats Hook ===\n\n");

        // First, dump all _ANEVirtualClient methods
        Class vcClass = NSClassFromString(@"_ANEVirtualClient");
        printf("_ANEVirtualClient methods:\n");
        {
            unsigned int mcount = 0;
            Method *methods = class_copyMethodList(vcClass, &mcount);
            for (unsigned int i = 0; i < mcount; i++) {
                printf("  -%s [%s]\n", sel_getName(method_getName(methods[i])),
                       method_getTypeEncoding(methods[i]));
            }
            free(methods);
            methods = class_copyMethodList(object_getClass(vcClass), &mcount);
            for (unsigned int i = 0; i < mcount; i++) {
                printf("  +%s [%s]\n", sel_getName(method_getName(methods[i])),
                       method_getTypeEncoding(methods[i]));
            }
            free(methods);
        }

        // Swizzle updatePerformanceStats — it's a CLASS method
        {
            Class metaClass = object_getClass(vcClass);
            SEL sel = @selector(updatePerformanceStats:performanceStatsLength:perfStatsRawIOSurfaceRef:performanceStatsRawLength:hwExecutionTime:);
            Method m = class_getClassMethod(vcClass, sel);
            if (m) {
                g_origUpdatePerfStats = method_getImplementation(m);
                method_setImplementation(m, (IMP)hooked_updatePerformanceStats);
                printf("\nSwizzled updatePerformanceStats OK\n");
            } else {
                printf("\nupdatePerformanceStats method not found!\n");
            }
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

        size_t in_bytes = (size_t)IC * (SEQ + OC) * 2;
        size_t out_bytes = (size_t)OC * SEQ * 2;
        IOSurfaceRef ioIn = make_surf(in_bytes);
        IOSurfaceRef ioOut = make_surf(out_bytes);
        id wI = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_AIO, @selector(objectWithIOSurface:), ioIn);
        id wO = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_AIO, @selector(objectWithIOSurface:), ioOut);
        [g_keepalive addObject:wI];
        [g_keepalive addObject:wO];

        // Test A: eval without perfStatsMask
        printf("\n--- Eval without perfStatsMask (mask=0) ---\n");
        {
            id req = ((id(*)(Class,SEL,id,id,id,id,id,id))objc_msgSend)(g_AR,
                @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:procedureIndex:),
                @[wI], @[@0], @[wO], @[@0], nil, @0);
            [g_keepalive addObject:req];
            e = nil;

            // XPC path
            BOOL ok = ((BOOL(*)(id,SEL,id,id,id,unsigned int,NSError**))objc_msgSend)(g_client,
                @selector(evaluateWithModel:options:request:qos:error:),
                aneModel, @{}, req, 21, &e);
            printf("  XPC eval: %s\n", ok?"OK":"FAIL");
        }

        // Test B: eval with perfStatsMask set
        printf("\n--- Eval with perfStatsMask=0xF ---\n");
        {
            ((void(*)(id,SEL,unsigned int))objc_msgSend)(aneModel, @selector(setPerfStatsMask:), 0xF);
            ((void(*)(id,SEL,unsigned int))objc_msgSend)(mdl, @selector(setPerfStatsMask:), 0xF);

            id req = ((id(*)(Class,SEL,id,id,id,id,id,id))objc_msgSend)(g_AR,
                @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:procedureIndex:),
                @[wI], @[@0], @[wO], @[@0], nil, @0);
            [g_keepalive addObject:req];
            e = nil;

            // XPC path
            BOOL ok = ((BOOL(*)(id,SEL,id,id,id,unsigned int,NSError**))objc_msgSend)(g_client,
                @selector(evaluateWithModel:options:request:qos:error:),
                aneModel, @{}, req, 21, &e);
            printf("  XPC eval: %s\n", ok?"OK":"FAIL");
        }

        // Test C: eval with perfStats on request
        printf("\n--- Eval with perfStats on request (via factory) ---\n");
        {
            Class perfStatsIOClass = NSClassFromString(@"_ANEPerformanceStatsIOSurface");
            IOSurfaceRef statsSurf = make_surf(65536);
            id perfStats = ((id(*)(Class,SEL,IOSurfaceRef,int))objc_msgSend)(
                perfStatsIOClass, @selector(objectWithIOSurface:statType:), statsSurf, 0);
            [g_keepalive addObject:perfStats];

            // Use factory with perfStats parameter
            id req = ((id(*)(Class,SEL,id,id,id,id,id,id,id))objc_msgSend)(g_AR,
                @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:),
                @[wI], @[@0], @[wO], @[@0], nil, perfStats, @0);
            [g_keepalive addObject:req];
            e = nil;

            BOOL ok = ((BOOL(*)(id,SEL,id,id,id,unsigned int,NSError**))objc_msgSend)(g_client,
                @selector(evaluateWithModel:options:request:qos:error:),
                aneModel, @{}, req, 21, &e);
            printf("  XPC eval: %s\n", ok?"OK":"FAIL");

            // Check request after
            @try {
                id ps = ((id(*)(id,SEL))objc_msgSend)(req, @selector(perfStats));
                printf("  perfStats after: class=%s\n",
                       ps ? class_getName(object_getClass(ps)) : "nil");
                if (ps && [ps respondsToSelector:@selector(hwExecutionTime)]) {
                    uint64_t hwt = ((uint64_t(*)(id,SEL))objc_msgSend)(ps, @selector(hwExecutionTime));
                    printf("  hwExecutionTime: %llu\n", hwt);
                }
            } @catch (NSException *ex) {
                printf("  EXCEPTION: %s\n", [[ex description] UTF8String]);
            }

            CFRelease(statsSurf);
        }

        // Test D: direct eval with mask
        printf("\n--- Direct eval with perfStatsMask=0xF ---\n");
        {
            id req = ((id(*)(Class,SEL,id,id,id,id,id,id))objc_msgSend)(g_AR,
                @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:procedureIndex:),
                @[wI], @[@0], @[wO], @[@0], nil, @0);
            [g_keepalive addObject:req];
            e = nil;

            BOOL ok = ((BOOL(*)(id,SEL,id,id,id,unsigned int,NSError**))objc_msgSend)(g_client,
                @selector(doEvaluateDirectWithModel:options:request:qos:error:),
                aneModel, @{}, req, 21, &e);
            printf("  direct eval: %s\n", ok?"OK":"FAIL");
        }

        printf("\n(If no HOOKED messages appeared, updatePerformanceStats was never called)\n");
        printf("\n=== Done ===\n");
    }
    return 0;
}
