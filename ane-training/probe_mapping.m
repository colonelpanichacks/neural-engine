// probe_mapping.m — Figure out correct IOSurface format for mapIOSurfaces
// Also probe chaining with _ANERequest, and the split buffersReady/enqueueSets path
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

// Make IOSurface matching [1, C, 1, S] tensor layout (fp16)
static IOSurfaceRef make_tensor_surf(int C, int S) {
    size_t bpe = 2;  // fp16
    size_t row = S * bpe;
    size_t total = C * row;
    return IOSurfaceCreate((__bridge CFDictionaryRef)@{
        (id)kIOSurfaceWidth:@(S),
        (id)kIOSurfaceHeight:@(C),
        (id)kIOSurfaceBytesPerElement:@(bpe),
        (id)kIOSurfaceBytesPerRow:@(row),
        (id)kIOSurfaceAllocSize:@(total),
        (id)kIOSurfacePixelFormat:@(0x4F323136)  // kCVPixelFormatType_OneComponent16Half = 'O216'
    });
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

// Dump _ANEIOSurfaceObject methods
static void dump_io_methods(void) {
    printf("=== _ANEIOSurfaceObject methods ===\n");
    unsigned int count = 0;
    Method *methods = class_copyMethodList(g_AIO, &count);
    for (unsigned int i = 0; i < count; i++)
        printf("  %s\n", sel_getName(method_getName(methods[i])));
    free(methods);
    // Class methods
    Method *cmethods = class_copyMethodList(object_getClass(g_AIO), &count);
    for (unsigned int i = 0; i < count; i++)
        printf("  + %s\n", sel_getName(method_getName(cmethods[i])));
    free(cmethods);
    printf("\n");
}

// Dump _ANEProgramProcedureInfo if it exists
static void dump_procedure_info(void) {
    Class cls = NSClassFromString(@"_ANEProgramProcedureInfo");
    if (!cls) { printf("_ANEProgramProcedureInfo: not found\n\n"); return; }
    printf("=== _ANEProgramProcedureInfo methods ===\n");
    unsigned int count = 0;
    Method *methods = class_copyMethodList(cls, &count);
    for (unsigned int i = 0; i < count; i++)
        printf("  %s\n", sel_getName(method_getName(methods[i])));
    free(methods);
    printf("\n");
}

int main(int argc, char *argv[]) {
    @autoreleasepool {
        setbuf(stdout, NULL);
        mach_timebase_info(&g_tb);
        ane_init();

        printf("=== ANE IOSurface Mapping & Chaining Probe ===\n\n");

        dump_io_methods();
        dump_procedure_info();

        // Look for any chaining-related classes
        const char *classNames[] = {
            "_ANEChainingRequest", "_ANEChainRequest", "_ANEChain",
            "_ANEPipeline", "_ANEPipelineRequest", "_ANEModelChain",
            "_ANEDaemonConnection", "_ANEDeviceController",
            "_ANESharedEvent", "_ANETransactionHandle",
            "_ANEModelLoadInstParams", "_ANEInputSet", "_ANEOutputSet",
            NULL
        };
        printf("=== Probing for related classes ===\n");
        for (int i = 0; classNames[i]; i++) {
            Class cls = NSClassFromString(@(classNames[i]));
            if (cls) {
                printf("  FOUND: %s\n", classNames[i]);
                unsigned int count = 0;
                Method *methods = class_copyMethodList(cls, &count);
                for (unsigned int j = 0; j < count; j++)
                    printf("    %s\n", sel_getName(method_getName(methods[j])));
                free(methods);
            }
        }
        printf("\n");

        // Enumerate ALL classes in AppleNeuralEngine framework
        printf("=== All _ANE classes ===\n");
        unsigned int numClasses = 0;
        Class *allClasses = objc_copyClassList(&numClasses);
        for (unsigned int i = 0; i < numClasses; i++) {
            const char *name = class_getName(allClasses[i]);
            if (strncmp(name, "_ANE", 4) == 0 || strncmp(name, "ANE", 3) == 0) {
                printf("  %s\n", name);
            }
        }
        free(allClasses);
        printf("\n");

        int ic = 256, oc = 256, seq = 64;  // Small kernel for fast testing
        NSData *mil = gen_matmul_mil(ic, oc, seq);
        int sp = seq + oc;

        id desc = ((id(*)(Class,SEL,id,id,id))objc_msgSend)(g_D, @selector(modelWithMILText:weights:optionsPlist:), mil, @{}, nil);
        id model = ((id(*)(Class,SEL,id))objc_msgSend)(g_I, @selector(inMemoryModelWithDescriptor:), desc);
        id hx = ((id(*)(id,SEL))objc_msgSend)(model, @selector(hexStringIdentifier));
        NSString *td = [NSTemporaryDirectory() stringByAppendingPathComponent:hx];
        [[NSFileManager defaultManager] createDirectoryAtPath:[td stringByAppendingPathComponent:@"weights"]
            withIntermediateDirectories:YES attributes:nil error:nil];
        [mil writeToFile:[td stringByAppendingPathComponent:@"model.mil"] atomically:YES];

        NSError *e = nil;
        ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(model, @selector(compileWithQoS:options:error:), 21, @{}, &e);
        ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(model, @selector(loadWithQoS:options:error:), 21, @{}, &e);
        id aneModel = ((id(*)(id,SEL))objc_msgSend)(model, @selector(model));

        // Get procedure info
        printf("=== Model procedure info ===\n");
        @try {
            id procInfo = ((id(*)(id,SEL,id))objc_msgSend)(aneModel, @selector(procedureInfoForProcedureIndex:), @0);
            if (procInfo) {
                printf("  procInfo class: %s\n", class_getName([procInfo class]));
                printf("  procInfo: %s\n", [[procInfo description] UTF8String]);

                // Dump its methods
                unsigned int count = 0;
                Method *methods = class_copyMethodList([procInfo class], &count);
                for (unsigned int i = 0; i < count; i++)
                    printf("    %s\n", sel_getName(method_getName(methods[i])));
                free(methods);
            } else {
                printf("  procInfo: nil\n");
            }
        } @catch (NSException *ex) {
            printf("  procInfo threw: %s\n", [[ex description] UTF8String]);
        }

        // Try different IOSurface formats for mapping
        printf("\n=== IOSurface format tests for mapIOSurfaces ===\n");

        typedef struct { const char *name; IOSurfaceRef surf; } SurfTest;
        SurfTest tests[] = {
            {"flat bytes", make_surf(ic * sp * 2)},
            {"tensor [IC,SP] fp16", make_tensor_surf(ic, sp)},
            {"tensor [1, IC*SP]", make_tensor_surf(1, ic*sp)},
        };
        int n_tests = sizeof(tests)/sizeof(tests[0]);

        for (int t = 0; t < n_tests; t++) {
            IOSurfaceRef outS = make_surf(oc * seq * 2);
            id wI = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_AIO, @selector(objectWithIOSurface:), tests[t].surf);
            id wO = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_AIO, @selector(objectWithIOSurface:), outS);
            id req = ((id(*)(Class,SEL,id,id,id,id,id,id,id))objc_msgSend)(g_AR,
                @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:),
                @[wI], @[@0], @[wO], @[@0], nil, nil, @0);

            // First verify it works for eval
            BOOL evalOk = ((BOOL(*)(id,SEL,id,id,id,NSError**))objc_msgSend)(
                g_client, @selector(evaluateRealTimeWithModel:options:request:error:),
                aneModel, @{}, req, &e);

            // Then try mapping
            BOOL mapOk = ((BOOL(*)(id,SEL,id,BOOL,NSError**))objc_msgSend)(
                model, @selector(mapIOSurfacesWithRequest:cacheInference:error:),
                req, YES, &e);

            printf("  %s: eval=%s map=%s\n", tests[t].name,
                   evalOk?"OK":"FAIL", mapOk?"OK":"FAIL");
            if (!mapOk && e)
                printf("    map error: %s\n", [[[e description] componentsSeparatedByString:@"UserInfo"][0] UTF8String]);
            e = nil;

            if (mapOk) {
                // Benchmark with mapping
                int N = 500;
                uint64_t t0 = mach_absolute_time();
                for (int i = 0; i < N; i++) {
                    ((BOOL(*)(id,SEL,id,id,id,NSError**))objc_msgSend)(
                        g_client, @selector(evaluateRealTimeWithModel:options:request:error:),
                        aneModel, @{}, req, &e);
                }
                double mapped_ms = ms_t(mach_absolute_time() - t0);
                printf("    mapped: %.1f ms (%5.3f ms/eval)\n", mapped_ms, mapped_ms/N);
                ((void(*)(id,SEL,id))objc_msgSend)(model, @selector(unmapIOSurfacesWithRequest:), req);
            }

            CFRelease(tests[t].surf);
            CFRelease(outS);
        }

        // Also try mapIOSurfaces on the client with the right args
        printf("\n=== Client-side mapping with model ===\n");
        IOSurfaceRef testIn = make_surf(ic * sp * 2);
        IOSurfaceRef testOut = make_surf(oc * seq * 2);
        id twI = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_AIO, @selector(objectWithIOSurface:), testIn);
        id twO = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_AIO, @selector(objectWithIOSurface:), testOut);
        id treq = ((id(*)(Class,SEL,id,id,id,id,id,id,id))objc_msgSend)(g_AR,
            @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:),
            @[twI], @[@0], @[twO], @[@0], nil, nil, @0);

        // Try mapIOSurfaces with cacheInference=NO
        BOOL map2 = ((BOOL(*)(id,SEL,id,BOOL,NSError**))objc_msgSend)(
            model, @selector(mapIOSurfacesWithRequest:cacheInference:error:),
            treq, NO, &e);
        printf("  cacheInference=NO: %s\n", map2 ? "OK" : "FAIL");
        if (!map2 && e) printf("    %s\n", [[[e description] componentsSeparatedByString:@"UserInfo"][0] UTF8String]);
        e = nil;

        // Try via the _ANEClient path
        BOOL map3 = ((BOOL(*)(id,SEL,id,id,BOOL,NSError**))objc_msgSend)(
            g_client, @selector(mapIOSurfacesWithModel:request:cacheInference:error:),
            aneModel, treq, NO, &e);
        printf("  Client map (cache=NO): %s\n", map3 ? "OK" : "FAIL");
        if (!map3 && e) printf("    %s\n", [[[e description] componentsSeparatedByString:@"UserInfo"][0] UTF8String]);
        e = nil;

        CFRelease(testIn); CFRelease(testOut);

        // === Probe chaining with actual _ANERequest ===
        printf("\n=== Chaining with _ANERequest ===\n");

        // Compile a chain: A(256→256) → B(256→256)
        NSData *milA = gen_matmul_mil(256, 256, 64);
        NSData *milB = gen_matmul_mil(256, 256, 64);

        id descA = ((id(*)(Class,SEL,id,id,id))objc_msgSend)(g_D, @selector(modelWithMILText:weights:optionsPlist:), milA, @{}, nil);
        id modelA = ((id(*)(Class,SEL,id))objc_msgSend)(g_I, @selector(inMemoryModelWithDescriptor:), descA);
        id hxA = ((id(*)(id,SEL))objc_msgSend)(modelA, @selector(hexStringIdentifier));
        NSString *tdA = [NSTemporaryDirectory() stringByAppendingPathComponent:hxA];
        [[NSFileManager defaultManager] createDirectoryAtPath:[tdA stringByAppendingPathComponent:@"weights"] withIntermediateDirectories:YES attributes:nil error:nil];
        [milA writeToFile:[tdA stringByAppendingPathComponent:@"model.mil"] atomically:YES];
        ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(modelA, @selector(compileWithQoS:options:error:), 21, @{}, &e);
        ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(modelA, @selector(loadWithQoS:options:error:), 21, @{}, &e);
        id aneModelA = ((id(*)(id,SEL))objc_msgSend)(modelA, @selector(model));

        // Shared IOSurface: A's output = B's input
        // A outputs [1,256,1,64], B expects [1,256,1,320]
        // This won't work because B needs weights too
        // Instead: test chaining where A→B share an intermediate
        size_t inA_bytes = 256 * (64 + 256) * 2;
        size_t outA_bytes = 256 * 64 * 2;  // also inB activation part

        IOSurfaceRef sInA = make_surf(inA_bytes);
        IOSurfaceRef sChain = make_surf(outA_bytes);  // A's output = shared
        _Float16 *pA = (void*)IOSurfaceGetBaseAddress(sInA);
        for (size_t i = 0; i < inA_bytes/2; i++) pA[i] = (_Float16)(0.01f * (i % 100));

        id ioInA = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_AIO, @selector(objectWithIOSurface:), sInA);
        id ioChain = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_AIO, @selector(objectWithIOSurface:), sChain);
        id reqA = ((id(*)(Class,SEL,id,id,id,id,id,id,id))objc_msgSend)(g_AR,
            @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:),
            @[ioInA], @[@0], @[ioChain], @[@0], nil, nil, @0);

        // Try prepareChainingWithModel using _ANERequest as chainingReq
        @try {
            BOOL chainOk = ((BOOL(*)(id,SEL,id,id,id,unsigned int,NSError**))objc_msgSend)(
                g_client, @selector(prepareChainingWithModel:options:chainingReq:qos:error:),
                aneModelA, @{}, reqA, 21, &e);
            printf("  prepareChaining(_ANERequest): %s\n", chainOk ? "OK" : "FAIL");
            if (!chainOk && e) printf("    Error: %s\n", [[e description] UTF8String]);
        } @catch (NSException *ex) {
            printf("  prepareChaining threw: %s\n", [[ex description] UTF8String]);
        }
        e = nil;

        // Try doPrepareChainingWithModel
        @try {
            BOOL chainOk = ((BOOL(*)(id,SEL,id,id,id,unsigned int,NSError**))objc_msgSend)(
                g_client, @selector(doPrepareChainingWithModel:options:chainingReq:qos:error:),
                aneModelA, @{}, reqA, 21, &e);
            printf("  doPrepareChainingWithModel: %s\n", chainOk ? "OK" : "FAIL");
            if (!chainOk && e) printf("    Error: %s\n", [[e description] UTF8String]);
        } @catch (NSException *ex) {
            printf("  doPrepareChainingWithModel threw: %s\n", [[ex description] UTF8String]);
        }
        e = nil;

        // Try buffersReady with _ANERequest (not array)
        printf("\n=== buffersReady with _ANERequest ===\n");
        @try {
            BOOL brOk = ((BOOL(*)(id,SEL,id,id,id,unsigned int,NSError**))objc_msgSend)(
                g_client, @selector(buffersReadyWithModel:inputBuffers:options:qos:error:),
                aneModelA, reqA, @{}, 21, &e);
            printf("  buffersReady(req): %s\n", brOk ? "OK" : "FAIL");
            if (!brOk && e) printf("    Error: %s\n", [[e description] UTF8String]);
        } @catch (NSException *ex) {
            printf("  buffersReady threw: %s\n", [[ex description] UTF8String]);
        }
        e = nil;

        // Try enqueueSets with _ANERequest
        printf("\n=== enqueueSets with _ANERequest ===\n");
        @try {
            BOOL esOk = ((BOOL(*)(id,SEL,id,id,id,unsigned int,NSError**))objc_msgSend)(
                g_client, @selector(enqueueSetsWithModel:outputSet:options:qos:error:),
                aneModelA, reqA, @{}, 21, &e);
            printf("  enqueueSets(req): %s\n", esOk ? "OK" : "FAIL");
            if (!esOk && e) printf("    Error: %s\n", [[e description] UTF8String]);
        } @catch (NSException *ex) {
            printf("  enqueueSets threw: %s\n", [[ex description] UTF8String]);
        }
        e = nil;

        // === Try the buffersReady→enqueueSets pipeline as split eval ===
        // Hypothesis: buffersReady signals inputs are ready, enqueueSets triggers execution
        printf("\n=== Split eval: buffersReady + enqueueSets ===\n");

        // The _ANEIOSurfaceObject might need to be the inputBuffers, not _ANERequest
        @try {
            BOOL brOk = ((BOOL(*)(id,SEL,id,id,id,unsigned int,NSError**))objc_msgSend)(
                g_client, @selector(buffersReadyWithModel:inputBuffers:options:qos:error:),
                aneModelA, ioInA, @{}, 21, &e);
            printf("  buffersReady(IOSurfObj): %s\n", brOk ? "OK" : "FAIL");
            if (!brOk && e) printf("    Error: %s\n", [[e description] UTF8String]);
        } @catch (NSException *ex) {
            printf("  buffersReady threw: %s\n", [[ex description] UTF8String]);
        }
        e = nil;

        // Cleanup
        ((BOOL(*)(id,SEL,unsigned int,NSError**))objc_msgSend)(model, @selector(unloadWithQoS:error:), 21, &e);
        ((BOOL(*)(id,SEL,unsigned int,NSError**))objc_msgSend)(modelA, @selector(unloadWithQoS:error:), 21, &e);
        CFRelease(sInA); CFRelease(sChain);

        printf("\n=== Done ===\n");
    }
    return 0;
}
