// probe_enqueue.m — Systematic probe of enqueueSets parameter permutations
// Focus: after buffersReadyWithModel succeeds, try every possible outputSet format
#import <Foundation/Foundation.h>
#import <IOSurface/IOSurface.h>
#import <objc/runtime.h>
#import <objc/message.h>
#import <dlfcn.h>
#import <mach/mach_time.h>

static mach_timebase_info_data_t g_tb;
static double ms_t(uint64_t t) { return (double)t * g_tb.numer / g_tb.denom / 1e6; }
static Class g_AIO, g_AR;
static id g_client = nil;

static IOSurfaceRef make_surf(size_t bytes) {
    return IOSurfaceCreate((__bridge CFDictionaryRef)@{
        (id)kIOSurfaceWidth:@(bytes), (id)kIOSurfaceHeight:@1,
        (id)kIOSurfaceBytesPerElement:@1, (id)kIOSurfaceBytesPerRow:@(bytes),
        (id)kIOSurfaceAllocSize:@(bytes), (id)kIOSurfacePixelFormat:@0});
}

static NSData *gen_mil(int ic, int oc, int seq) {
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

typedef struct { id model, aneModel; IOSurfaceRef ioIn, ioOut; id request; } Kern;

static Kern compile(int ic, int oc, int seq) {
    Kern k = {0};
    Class D = NSClassFromString(@"_ANEInMemoryModelDescriptor");
    Class I = NSClassFromString(@"_ANEInMemoryModel");
    NSData *mil = gen_mil(ic, oc, seq);
    size_t inB = ic*(seq+oc)*2, outB = oc*seq*2;
    id desc = ((id(*)(Class,SEL,id,id,id))objc_msgSend)(D, @selector(modelWithMILText:weights:optionsPlist:), mil, @{}, nil);
    k.model = ((id(*)(Class,SEL,id))objc_msgSend)(I, @selector(inMemoryModelWithDescriptor:), desc);
    id hx = ((id(*)(id,SEL))objc_msgSend)(k.model, @selector(hexStringIdentifier));
    NSString *td = [NSTemporaryDirectory() stringByAppendingPathComponent:hx];
    [[NSFileManager defaultManager] createDirectoryAtPath:[td stringByAppendingPathComponent:@"weights"]
        withIntermediateDirectories:YES attributes:nil error:nil];
    [mil writeToFile:[td stringByAppendingPathComponent:@"model.mil"] atomically:YES];
    ((void(*)(id,SEL,char))objc_msgSend)(k.model, @selector(setQueueDepth:), (char)4);
    NSError *e = nil;
    ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(k.model, @selector(compileWithQoS:options:error:), 21, @{}, &e);
    if (e) { printf("Compile error: %s\n", [[e description] UTF8String]); }
    ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(k.model, @selector(loadWithQoS:options:error:), 21, @{}, &e);
    if (e) { printf("Load error: %s\n", [[e description] UTF8String]); }
    k.aneModel = ((id(*)(id,SEL))objc_msgSend)(k.model, @selector(model));
    k.ioIn = make_surf(inB); k.ioOut = make_surf(outB);
    _Float16 *p = (void*)IOSurfaceGetBaseAddress(k.ioIn);
    for (size_t i = 0; i < inB/2; i++) p[i] = (_Float16)(0.01f*(i%100));
    id wI = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_AIO, @selector(objectWithIOSurface:), k.ioIn);
    id wO = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_AIO, @selector(objectWithIOSurface:), k.ioOut);
    k.request = ((id(*)(Class,SEL,id,id,id,id,id,id,id))objc_msgSend)(g_AR,
        @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:),
        @[wI], @[@0], @[wO], @[@0], nil, nil, @0);
    return k;
}

// Helper: call buffersReady (direct) and return raw int result
static int do_buffers_ready(id model, id ibr, NSDictionary *opts) {
    NSError *e = nil;
    // Cast as returning int to see the raw value (might not be BOOL)
    int ret = ((int(*)(id,SEL,id,id,id,unsigned int,NSError**))objc_msgSend)(
        g_client, @selector(doBuffersReadyWithModel:inputBuffers:options:qos:error:),
        model, ibr, opts, (unsigned int)21, &e);
    printf("  doBuffersReady ret=%d", ret);
    if (e) printf(" err=%ld (%s)", (long)[e code], [[e localizedDescription] UTF8String]);
    printf("\n");
    return ret;
}

// Helper: try enqueue with a label — direct variant only (XPC can timeout/hang)
static void try_enqueue_direct(const char *label, id model, id outputSet, NSDictionary *opts) {
    NSError *e = nil;
    @try {
        int ret = ((int(*)(id,SEL,id,id,id,unsigned int,NSError**))objc_msgSend)(
            g_client, @selector(doEnqueueSetsWithModel:outputSet:options:qos:error:),
            model, outputSet, opts, (unsigned int)21, &e);
        printf("    [%s]: ret=%d", label, ret);
        if (e) printf(" err=%ld (%s)", (long)[e code], [[e localizedDescription] UTF8String]);
        printf("\n");
    } @catch (NSException *ex) {
        printf("    [%s]: EXCEPTION %s\n", label, [[ex reason] UTF8String]);
    }
}

// Helper: try enqueue with XPC variant
static void try_enqueue_xpc(const char *label, id model, id outputSet, NSDictionary *opts) {
    NSError *e = nil;
    @try {
        int ret = ((int(*)(id,SEL,id,id,id,unsigned int,NSError**))objc_msgSend)(
            g_client, @selector(enqueueSetsWithModel:outputSet:options:qos:error:),
            model, outputSet, opts, (unsigned int)21, &e);
        printf("    [%s]: ret=%d", label, ret);
        if (e) printf(" err=%ld (%s)", (long)[e code], [[e localizedDescription] UTF8String]);
        printf("\n");
    } @catch (NSException *ex) {
        printf("    [%s]: EXCEPTION %s\n", label, [[ex reason] UTF8String]);
    }
}

static void dump_methods(Class cls, const char *name) {
    unsigned int count = 0;
    Method *methods = class_copyMethodList(cls, &count);
    printf("  %s methods (%u):\n", name, count);
    for (unsigned int i = 0; i < count; i++) {
        printf("    %s\n", sel_getName(method_getName(methods[i])));
    }
    free(methods);
}

int main(int argc, char *argv[]) {
    @autoreleasepool {
        setbuf(stdout, NULL);
        mach_timebase_info(&g_tb);
        dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine", RTLD_NOW);
        g_AIO = NSClassFromString(@"_ANEIOSurfaceObject");
        g_AR  = NSClassFromString(@"_ANERequest");
        g_client = ((id(*)(Class,SEL))objc_msgSend)(NSClassFromString(@"_ANEClient"), @selector(sharedConnection));

        printf("=== ANE Enqueue Probe ===\n");

        // Dump relevant class methods for reference
        dump_methods(NSClassFromString(@"_ANEOutputSetEnqueue"), "_ANEOutputSetEnqueue");
        dump_methods(NSClassFromString(@"_ANEInputBuffersReady"), "_ANEInputBuffersReady");

        // Compile 64x64 matmul kernel
        Kern k = compile(64, 64, 32);
        printf("\nKernel compiled: [64,96] -> [64,32]\n");

        // Check queueDepth
        char qd = ((char(*)(id,SEL))objc_msgSend)(k.aneModel, @selector(queueDepth));
        printf("  aneModel queueDepth: %d\n", (int)qd);

        // === Step 1: Verify kernel with normal eval ===
        printf("\n=== Step 1: doEvaluateDirectWithModel (baseline) ===\n");
        {
            NSError *e = nil;
            BOOL ok = ((BOOL(*)(id,SEL,id,id,id,unsigned int,NSError**))objc_msgSend)(
                g_client, @selector(doEvaluateDirectWithModel:options:request:qos:error:),
                k.aneModel, @{}, k.request, (unsigned int)21, &e);
            printf("  eval: %s", ok ? "SUCCESS" : "FAIL");
            if (e) printf(" err=%ld", (long)[e code]);
            printf("\n");
            _Float16 *out = (_Float16*)IOSurfaceGetBaseAddress(k.ioOut);
            printf("  output[0..3]: %.4f %.4f %.4f %.4f\n",
                   (float)out[0], (float)out[1], (float)out[2], (float)out[3]);
        }

        // Prepare helper objects
        Class ANEBuf       = NSClassFromString(@"_ANEBuffer");
        Class ANEOutSets    = NSClassFromString(@"_ANEIOSurfaceOutputSets");
        Class ANEOutSetEnq  = NSClassFromString(@"_ANEOutputSetEnqueue");
        Class ANEInputReady = NSClassFromString(@"_ANEInputBuffersReady");

        id wIn  = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_AIO, @selector(objectWithIOSurface:), k.ioIn);
        id wOut = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_AIO, @selector(objectWithIOSurface:), k.ioOut);
        IOSurfaceRef statsSurf = make_surf(4096);

        id bufOut = ((id(*)(Class,SEL,id,id,long long))objc_msgSend)(ANEBuf,
            @selector(bufferWithIOSurfaceObject:symbolIndex:source:), wOut, @0, (long long)1);

        // Build InputBuffersReady
        id ibr = ((id(*)(id,SEL,unsigned int,unsigned int,unsigned long long,unsigned int))objc_msgSend)(
            [ANEInputReady alloc],
            @selector(initInputsProcedureIndex:inputBufferInfoIndex:inputFreeValue:executionDelay:),
            (unsigned int)0, (unsigned int)0, (unsigned long long)0, (unsigned int)0);
        printf("\n  ibr: %s (class: %s)\n", [[ibr description] UTF8String], class_getName([ibr class]));

        // Build OutputSetEnqueue variants
        // A: basic (signalNotRequired=NO, isOpenLoop=NO)
        id ose_basic = ((id(*)(id,SEL,unsigned int,unsigned int,unsigned long long,BOOL,BOOL))objc_msgSend)(
            [ANEOutSetEnq alloc],
            @selector(initOutputSetWithProcedureIndex:setIndex:signalValue:signalNotRequired:isOpenLoop:),
            (unsigned int)0, (unsigned int)0, (unsigned long long)1, NO, NO);

        // B: signalNotRequired=YES
        id ose_snr = ((id(*)(id,SEL,unsigned int,unsigned int,unsigned long long,BOOL,BOOL))objc_msgSend)(
            [ANEOutSetEnq alloc],
            @selector(initOutputSetWithProcedureIndex:setIndex:signalValue:signalNotRequired:isOpenLoop:),
            (unsigned int)0, (unsigned int)0, (unsigned long long)1, YES, NO);

        // C: isOpenLoop=YES
        id ose_ol = ((id(*)(id,SEL,unsigned int,unsigned int,unsigned long long,BOOL,BOOL))objc_msgSend)(
            [ANEOutSetEnq alloc],
            @selector(initOutputSetWithProcedureIndex:setIndex:signalValue:signalNotRequired:isOpenLoop:),
            (unsigned int)0, (unsigned int)0, (unsigned long long)1, NO, YES);

        // D: both=YES
        id ose_both = ((id(*)(id,SEL,unsigned int,unsigned int,unsigned long long,BOOL,BOOL))objc_msgSend)(
            [ANEOutSetEnq alloc],
            @selector(initOutputSetWithProcedureIndex:setIndex:signalValue:signalNotRequired:isOpenLoop:),
            (unsigned int)0, (unsigned int)0, (unsigned long long)1, YES, YES);

        // E: signalValue=0
        id ose_sv0 = ((id(*)(id,SEL,unsigned int,unsigned int,unsigned long long,BOOL,BOOL))objc_msgSend)(
            [ANEOutSetEnq alloc],
            @selector(initOutputSetWithProcedureIndex:setIndex:signalValue:signalNotRequired:isOpenLoop:),
            (unsigned int)0, (unsigned int)0, (unsigned long long)0, NO, NO);

        // _ANEIOSurfaceOutputSets
        id outSets = ((id(*)(Class,SEL,IOSurfaceRef,id))objc_msgSend)(ANEOutSets,
            @selector(objectWithstatsSurRef:outputBuffer:), statsSurf, @[bufOut]);

        // _ANERequest
        id reqObj = k.request;

        printf("  ose_basic: %s\n", [[ose_basic description] UTF8String]);
        printf("  outSets:   %s (class: %s)\n", class_getName([outSets class]), class_getName([outSets class]));
        printf("  request:   %s (class: %s)\n", class_getName([reqObj class]), class_getName([reqObj class]));

        // Introspect: does _ANEOutputSetEnqueue respond to procedureIndex?
        printf("\n  ose responds to procedureIndex: %s\n",
               [ose_basic respondsToSelector:@selector(procedureIndex)] ? "YES" : "NO");
        printf("  outSets responds to procedureIndex: %s\n",
               [outSets respondsToSelector:@selector(procedureIndex)] ? "YES" : "NO");
        printf("  request responds to procedureIndex: %s\n",
               [reqObj respondsToSelector:@selector(procedureIndex)] ? "YES" : "NO");

        // === Step 2: doBuffersReadyWithModel (direct) ===
        printf("\n=== Step 2: doBuffersReadyWithModel (direct) ===\n");
        do_buffers_ready(k.aneModel, ibr, @{});

        // === Step 3: XPC buffersReady ===
        printf("\n=== Step 3: buffersReadyWithModel (XPC) ===\n");
        {
            NSError *e = nil;
            int ret = ((int(*)(id,SEL,id,id,id,unsigned int,NSError**))objc_msgSend)(
                g_client, @selector(buffersReadyWithModel:inputBuffers:options:qos:error:),
                k.aneModel, ibr, @{}, (unsigned int)21, &e);
            printf("  buffersReady ret=%d", ret);
            if (e) printf(" err=%ld (%s)", (long)[e code], [[e localizedDescription] UTF8String]);
            printf("\n");
        }

        // === Step 4: doEnqueueSetsWithModel — all permutations ===
        // First re-prime with doBuffersReady before each attempt
        printf("\n=== Step 4: doEnqueueSetsWithModel (direct) — all permutations ===\n");

        // 4a: raw _ANEOutputSetEnqueue (basic)
        printf("  --- 4a: raw ose_basic ---\n");
        do_buffers_ready(k.aneModel, ibr, @{});
        try_enqueue_direct("ose_basic", k.aneModel, ose_basic, @{});

        // 4b: signalNotRequired=YES
        printf("  --- 4b: ose signalNotRequired=YES ---\n");
        do_buffers_ready(k.aneModel, ibr, @{});
        try_enqueue_direct("ose_snr", k.aneModel, ose_snr, @{});

        // 4c: isOpenLoop=YES
        printf("  --- 4c: ose isOpenLoop=YES ---\n");
        do_buffers_ready(k.aneModel, ibr, @{});
        try_enqueue_direct("ose_ol", k.aneModel, ose_ol, @{});

        // 4d: both=YES
        printf("  --- 4d: ose both=YES ---\n");
        do_buffers_ready(k.aneModel, ibr, @{});
        try_enqueue_direct("ose_both", k.aneModel, ose_both, @{});

        // 4e: signalValue=0
        printf("  --- 4e: ose signalValue=0 ---\n");
        do_buffers_ready(k.aneModel, ibr, @{});
        try_enqueue_direct("ose_sv0", k.aneModel, ose_sv0, @{});

        // 4f: _ANEIOSurfaceOutputSets (wrong type, but try)
        printf("  --- 4f: _ANEIOSurfaceOutputSets ---\n");
        do_buffers_ready(k.aneModel, ibr, @{});
        try_enqueue_direct("outSets", k.aneModel, outSets, @{});

        // 4g: _ANERequest object
        printf("  --- 4g: _ANERequest ---\n");
        do_buffers_ready(k.aneModel, ibr, @{});
        try_enqueue_direct("request", k.aneModel, reqObj, @{});

        // 4h: array of ose (will likely crash but try)
        printf("  --- 4h: array of ose ---\n");
        do_buffers_ready(k.aneModel, ibr, @{});
        try_enqueue_direct("array[ose]", k.aneModel, @[ose_basic], @{});

        // === Step 5: enqueueSetsWithModel (XPC) — safe permutations only ===
        printf("\n=== Step 5: enqueueSetsWithModel (XPC) — raw ose only ===\n");
        // Only test the ones that won't crash (raw ose, not arrays/wrong types)
        printf("  --- 5a: raw ose_basic ---\n");
        {
            NSError *e = nil;
            ((int(*)(id,SEL,id,id,id,unsigned int,NSError**))objc_msgSend)(
                g_client, @selector(buffersReadyWithModel:inputBuffers:options:qos:error:),
                k.aneModel, ibr, @{}, (unsigned int)21, &e);
        }
        try_enqueue_xpc("ose_basic", k.aneModel, ose_basic, @{});

        printf("  --- 5b: ose signalNotRequired=YES ---\n");
        {
            NSError *e = nil;
            ((int(*)(id,SEL,id,id,id,unsigned int,NSError**))objc_msgSend)(
                g_client, @selector(buffersReadyWithModel:inputBuffers:options:qos:error:),
                k.aneModel, ibr, @{}, (unsigned int)21, &e);
        }
        try_enqueue_xpc("ose_snr", k.aneModel, ose_snr, @{});

        printf("  --- 5c: ose isOpenLoop=YES ---\n");
        {
            NSError *e = nil;
            ((int(*)(id,SEL,id,id,id,unsigned int,NSError**))objc_msgSend)(
                g_client, @selector(buffersReadyWithModel:inputBuffers:options:qos:error:),
                k.aneModel, ibr, @{}, (unsigned int)21, &e);
        }
        try_enqueue_xpc("ose_ol", k.aneModel, ose_ol, @{});

        printf("  --- 5d: ose both=YES ---\n");
        {
            NSError *e = nil;
            ((int(*)(id,SEL,id,id,id,unsigned int,NSError**))objc_msgSend)(
                g_client, @selector(buffersReadyWithModel:inputBuffers:options:qos:error:),
                k.aneModel, ibr, @{}, (unsigned int)21, &e);
        }
        try_enqueue_xpc("ose_both", k.aneModel, ose_both, @{});

        // === Step 6: Without prior buffersReady ===
        printf("\n=== Step 6: doEnqueueSets WITHOUT prior buffersReady ===\n");
        // Reset with normal eval
        { NSError *e=nil; ((BOOL(*)(id,SEL,id,id,id,unsigned int,NSError**))objc_msgSend)(
            g_client, @selector(doEvaluateDirectWithModel:options:request:qos:error:),
            k.aneModel, @{}, k.request, (unsigned int)21, &e); }
        try_enqueue_direct("cold ose_basic", k.aneModel, ose_basic, @{});
        try_enqueue_direct("cold ose_snr",   k.aneModel, ose_snr,   @{});

        // === Step 7: Mixed XPC/direct ===
        printf("\n=== Step 7: XPC buffersReady + direct doEnqueue ===\n");
        {
            NSError *e = nil;
            int ret = ((int(*)(id,SEL,id,id,id,unsigned int,NSError**))objc_msgSend)(
                g_client, @selector(buffersReadyWithModel:inputBuffers:options:qos:error:),
                k.aneModel, ibr, @{}, (unsigned int)21, &e);
            printf("  XPC buffersReady ret=%d\n", ret);
        }
        try_enqueue_direct("ose_basic after XPC bufReady", k.aneModel, ose_basic, @{});

        printf("\n=== Step 8: direct doBuffersReady + XPC enqueueSets ===\n");
        do_buffers_ready(k.aneModel, ibr, @{});
        try_enqueue_xpc("ose_basic after direct bufReady", k.aneModel, ose_basic, @{});

        // === Step 9: Check if output changed after any enqueue ===
        printf("\n=== Step 9: Check output after enqueue attempts ===\n");
        {
            _Float16 *out = (_Float16*)IOSurfaceGetBaseAddress(k.ioOut);
            printf("  output[0..3]: %.4f %.4f %.4f %.4f\n",
                   (float)out[0], (float)out[1], (float)out[2], (float)out[3]);
        }

        printf("\nDone.\n");
        CFRelease(statsSurf);
    }
    return 0;
}
