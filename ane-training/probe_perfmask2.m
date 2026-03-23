// probe_perfmask2.m — Try perfStatsMask with model reload, check request properties
// Also try setting mask before load, and using completionHandler eval path
//
// Build:
//   xcrun clang -O2 -framework Foundation -framework IOSurface \
//     -isysroot /Library/Developer/CommandLineTools/SDKs/MacOSX.sdk \
//     -fobjc-arc -o probe_perfmask2 probe_perfmask2.m

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

int main(int argc, char *argv[]) {
    @autoreleasepool {
        setbuf(stdout, NULL);
        mach_timebase_info(&g_tb);
        ane_init();

        printf("=== ANE PerfStats — perfStatsMask before load ===\n\n");

        int IC = 1024, OC = 2048, SEQ = 256;
        NSData *mil = gen_matmul(IC, OC, SEQ);

        size_t in_bytes = (size_t)IC * (SEQ + OC) * 2;
        size_t out_bytes = (size_t)OC * SEQ * 2;
        IOSurfaceRef ioIn = make_surf(in_bytes);
        IOSurfaceRef ioOut = make_surf(out_bytes);
        id wI = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_AIO, @selector(objectWithIOSurface:), ioIn);
        id wO = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_AIO, @selector(objectWithIOSurface:), ioOut);
        [g_keepalive addObject:wI];
        [g_keepalive addObject:wO];

        // Test 1: Set perfStatsMask BEFORE load
        printf("--- Test 1: Set perfStatsMask=0xF on InMemoryModel before compile/load ---\n");
        {
            id desc = ((id(*)(Class,SEL,id,id,id))objc_msgSend)(g_D,
                @selector(modelWithMILText:weights:optionsPlist:), mil, @{}, nil);
            id mdl = ((id(*)(Class,SEL,id))objc_msgSend)(g_I,
                @selector(inMemoryModelWithDescriptor:), desc);
            [g_keepalive addObject:mdl];

            // Set mask before compile
            ((void(*)(id,SEL,unsigned int))objc_msgSend)(mdl, @selector(setPerfStatsMask:), 0xF);
            printf("  Set perfStatsMask=0xF on InMemoryModel before compile\n");

            id hx = ((id(*)(id,SEL))objc_msgSend)(mdl, @selector(hexStringIdentifier));
            NSString *td = [NSTemporaryDirectory() stringByAppendingPathComponent:hx];
            [[NSFileManager defaultManager] createDirectoryAtPath:[td stringByAppendingPathComponent:@"weights"]
                withIntermediateDirectories:YES attributes:nil error:nil];
            [mil writeToFile:[td stringByAppendingPathComponent:@"model.mil"] atomically:YES];

            NSError *e = nil;
            BOOL ok = ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(mdl,
                @selector(compileWithQoS:options:error:), 21, @{}, &e);
            printf("  compile: %s\n", ok?"OK":"FAIL");

            ok = ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(mdl,
                @selector(loadWithQoS:options:error:), 21, @{}, &e);
            printf("  load: %s\n", ok?"OK":"FAIL");

            id aneModel = ((id(*)(id,SEL))objc_msgSend)(mdl, @selector(model));
            [g_keepalive addObject:aneModel];

            // Check what mask the _ANEModel got
            unsigned int gotMask = ((unsigned int(*)(id,SEL))objc_msgSend)(aneModel, @selector(perfStatsMask));
            printf("  _ANEModel perfStatsMask after load: 0x%x\n", gotMask);

            // Eval without perfStats on request — check if request gets populated
            id req = ((id(*)(Class,SEL,id,id,id,id,id,id))objc_msgSend)(g_AR,
                @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:procedureIndex:),
                @[wI], @[@0], @[wO], @[@0], nil, @0);
            [g_keepalive addObject:req];

            e = nil;
            ok = ((BOOL(*)(id,SEL,id,id,id,unsigned int,NSError**))objc_msgSend)(g_client,
                @selector(evaluateWithModel:options:request:qos:error:),
                aneModel, @{}, req, 21, &e);
            printf("  eval (XPC, no perfStats on req): %s\n", ok?"OK":"FAIL");

            @try {
                id ps = ((id(*)(id,SEL))objc_msgSend)(req, @selector(perfStats));
                printf("  perfStats: %s\n", ps ?
                    [NSString stringWithFormat:@"class=%s desc=%@",
                        class_getName(object_getClass(ps)), ps].UTF8String : "(nil)");
                if (ps && [ps respondsToSelector:@selector(hwExecutionTime)]) {
                    uint64_t hwt = ((uint64_t(*)(id,SEL))objc_msgSend)(ps, @selector(hwExecutionTime));
                    printf("  hwExecutionTime: %llu ns\n", hwt);
                }
            } @catch (NSException *ex) {
                printf("  EXCEPTION: %s\n", [[ex description] UTF8String]);
            }

            @try {
                id psa = ((id(*)(id,SEL))objc_msgSend)(req, @selector(perfStatsArray));
                printf("  perfStatsArray: %s\n", psa ?
                    [NSString stringWithFormat:@"count=%lu %@",
                        (unsigned long)[(NSArray*)psa count], psa].UTF8String : "(nil)");
            } @catch (NSException *ex) {
                printf("  EXCEPTION: %s\n", [[ex description] UTF8String]);
            }

            // Also try doEvaluateDirectWithModel
            id req2 = ((id(*)(Class,SEL,id,id,id,id,id,id))objc_msgSend)(g_AR,
                @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:procedureIndex:),
                @[wI], @[@0], @[wO], @[@0], nil, @0);
            [g_keepalive addObject:req2];

            e = nil;
            ok = ((BOOL(*)(id,SEL,id,id,id,unsigned int,NSError**))objc_msgSend)(g_client,
                @selector(doEvaluateDirectWithModel:options:request:qos:error:),
                aneModel, @{}, req2, 21, &e);
            printf("  eval (direct, no perfStats on req): %s\n", ok?"OK":"FAIL");

            @try {
                id ps = ((id(*)(id,SEL))objc_msgSend)(req2, @selector(perfStats));
                printf("  perfStats: %s\n", ps ? "SET" : "(nil)");
                id psa = ((id(*)(id,SEL))objc_msgSend)(req2, @selector(perfStatsArray));
                printf("  perfStatsArray: %s\n", psa ? "SET" : "(nil)");
            } @catch (NSException *ex) {
                printf("  EXCEPTION: %s\n", [[ex description] UTF8String]);
            }
        }

        // Test 2: Use completionHandler eval path
        printf("\n--- Test 2: Async eval with completionHandler —--\n");
        {
            id desc = ((id(*)(Class,SEL,id,id,id))objc_msgSend)(g_D,
                @selector(modelWithMILText:weights:optionsPlist:), mil, @{}, nil);
            id mdl = ((id(*)(Class,SEL,id))objc_msgSend)(g_I,
                @selector(inMemoryModelWithDescriptor:), desc);
            [g_keepalive addObject:mdl];
            ((void(*)(id,SEL,unsigned int))objc_msgSend)(mdl, @selector(setPerfStatsMask:), 0xF);

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

            // Use _ANEInMemoryModel's own evaluateWithQoS:options:request:error:
            id req = ((id(*)(Class,SEL,id,id,id,id,id,id))objc_msgSend)(g_AR,
                @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:procedureIndex:),
                @[wI], @[@0], @[wO], @[@0], nil, @0);
            [g_keepalive addObject:req];

            // Try InMemoryModel's own eval
            e = nil;
            @try {
                BOOL ok = ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(mdl,
                    @selector(evaluateWithQoS:options:request:error:), 21, @{}, req, &e);
                printf("  InMemoryModel eval: %s %s\n", ok?"OK":"FAIL", e?[[e description] UTF8String]:"");

                id ps = ((id(*)(id,SEL))objc_msgSend)(req, @selector(perfStats));
                printf("  perfStats: %s\n", ps ? "SET":"(nil)");
                id psa = ((id(*)(id,SEL))objc_msgSend)(req, @selector(perfStatsArray));
                printf("  perfStatsArray: %s\n", psa ? "SET":"(nil)");
                if (ps && [ps respondsToSelector:@selector(hwExecutionTime)]) {
                    uint64_t hwt = ((uint64_t(*)(id,SEL))objc_msgSend)(ps, @selector(hwExecutionTime));
                    printf("  hwExecutionTime: %llu ns\n", hwt);
                }
            } @catch (NSException *ex) {
                printf("  EXCEPTION: %s\n", [[ex description] UTF8String]);
            }
        }

        // Test 3: Try _ANEProgramForEvaluation — might have perf stat methods
        printf("\n--- Test 3: _ANEProgramForEvaluation methods ---\n");
        {
            Class progClass = NSClassFromString(@"_ANEProgramForEvaluation");
            if (progClass) {
                unsigned int mcount = 0;
                Method *methods = class_copyMethodList(progClass, &mcount);
                for (unsigned int i = 0; i < mcount; i++) {
                    const char *mn = sel_getName(method_getName(methods[i]));
                    printf("  -%s [%s]\n", mn, method_getTypeEncoding(methods[i]));
                }
                free(methods);
            } else {
                printf("  class not found\n");
            }
        }

        // Test 4: _ANEProgramIOSurfacesMapper — might map stats surfaces
        printf("\n--- Test 4: _ANEProgramIOSurfacesMapper methods ---\n");
        {
            Class mapperClass = NSClassFromString(@"_ANEProgramIOSurfacesMapper");
            if (mapperClass) {
                unsigned int mcount = 0;
                Method *methods = class_copyMethodList(mapperClass, &mcount);
                for (unsigned int i = 0; i < mcount; i++) {
                    const char *mn = sel_getName(method_getName(methods[i]));
                    if (strstr(mn, "perf") || strstr(mn, "Perf") ||
                        strstr(mn, "stat") || strstr(mn, "Stat") ||
                        strstr(mn, "map") || strstr(mn, "Map")) {
                        printf("  -%s [%s]\n", mn, method_getTypeEncoding(methods[i]));
                    }
                }
                free(methods);
            }
        }

        // Test 5: Search for any class that creates _ANEPerformanceStats
        printf("\n--- Test 5: Classes that reference 'RequestPerformance' ---\n");
        {
            unsigned int classCount = 0;
            Class *classes = objc_copyClassList(&classCount);
            for (unsigned int i = 0; i < classCount; i++) {
                const char *cn = class_getName(classes[i]);
                if (!strstr(cn, "ANE")) continue;
                unsigned int mcount = 0;
                Method *methods = class_copyMethodList(classes[i], &mcount);
                for (unsigned int j = 0; j < mcount; j++) {
                    const char *mn = sel_getName(method_getName(methods[j]));
                    if (strstr(mn, "RequestPerformance") || strstr(mn, "requestPerformance") ||
                        strstr(mn, "perfStats") || strstr(mn, "PerfStats") ||
                        strstr(mn, "statsBuffer")) {
                        printf("  %s -%s\n", cn, mn);
                    }
                }
                free(methods);
                // Class methods
                methods = class_copyMethodList(object_getClass(classes[i]), &mcount);
                for (unsigned int j = 0; j < mcount; j++) {
                    const char *mn = sel_getName(method_getName(methods[j]));
                    if (strstr(mn, "RequestPerformance") || strstr(mn, "requestPerformance") ||
                        strstr(mn, "perfStats") || strstr(mn, "PerfStats") ||
                        strstr(mn, "statsBuffer")) {
                        printf("  %s +%s\n", cn, mn);
                    }
                }
                free(methods);
            }
            free(classes);
        }

        printf("\n=== Done ===\n");
    }
    return 0;
}
