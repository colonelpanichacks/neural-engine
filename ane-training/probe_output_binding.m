// probe_output_binding.m — Test whether ANE respects per-request output IOSurface bindings
// Hypothesis: when creating an _ANERequest with a DIFFERENT output IOSurface than the one
// used during compile, does ANE write to the new surface or the original?
// This determines whether double-buffering is possible without compiling two kernels.
#import <Foundation/Foundation.h>
#import <IOSurface/IOSurface.h>
#import <objc/runtime.h>
#import <objc/message.h>
#import <dlfcn.h>

static Class g_AIO, g_AR;
static id g_client = nil;

static IOSurfaceRef make_surf(size_t bytes) {
    return IOSurfaceCreate((__bridge CFDictionaryRef)@{
        (id)kIOSurfaceWidth:@(bytes), (id)kIOSurfaceHeight:@1,
        (id)kIOSurfaceBytesPerElement:@1, (id)kIOSurfaceBytesPerRow:@(bytes),
        (id)kIOSurfaceAllocSize:@(bytes), (id)kIOSurfacePixelFormat:@0});
}

// Generate a 64x64 matmul kernel: input [1,64,1,128] (64 act + 64 weight), output [1,64,1,64]
static NSData *gen_mil(void) {
    int ic = 64, oc = 64, seq = 64;
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

static Kern compile_kernel(void) {
    Kern k = {0};
    int ic = 64, oc = 64, seq = 64;
    Class D = NSClassFromString(@"_ANEInMemoryModelDescriptor");
    Class I = NSClassFromString(@"_ANEInMemoryModel");
    NSData *mil = gen_mil();
    size_t inB = ic * (seq + oc) * 2;  // 64 * 128 * 2 = 16384
    size_t outB = oc * seq * 2;        // 64 * 64 * 2 = 8192

    id desc = ((id(*)(Class,SEL,id,id,id))objc_msgSend)(D, @selector(modelWithMILText:weights:optionsPlist:), mil, @{}, nil);
    k.model = ((id(*)(Class,SEL,id))objc_msgSend)(I, @selector(inMemoryModelWithDescriptor:), desc);
    id hx = ((id(*)(id,SEL))objc_msgSend)(k.model, @selector(hexStringIdentifier));
    NSString *td = [NSTemporaryDirectory() stringByAppendingPathComponent:hx];
    [[NSFileManager defaultManager] createDirectoryAtPath:[td stringByAppendingPathComponent:@"weights"]
        withIntermediateDirectories:YES attributes:nil error:nil];
    [mil writeToFile:[td stringByAppendingPathComponent:@"model.mil"] atomically:YES];

    NSError *e = nil;
    BOOL ok = ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(k.model, @selector(compileWithQoS:options:error:), 21, @{}, &e);
    if (!ok) { printf("COMPILE FAIL: %s\n", e ? [[e description] UTF8String] : "unknown"); return k; }
    ok = ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(k.model, @selector(loadWithQoS:options:error:), 21, @{}, &e);
    if (!ok) { printf("LOAD FAIL: %s\n", e ? [[e description] UTF8String] : "unknown"); return k; }

    k.aneModel = ((id(*)(id,SEL))objc_msgSend)(k.model, @selector(model));
    k.ioIn = make_surf(inB);
    k.ioOut = make_surf(outB);

    // Build default request with original IOSurfaces
    id wI = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_AIO, @selector(objectWithIOSurface:), k.ioIn);
    id wO = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_AIO, @selector(objectWithIOSurface:), k.ioOut);
    k.request = ((id(*)(Class,SEL,id,id,id,id,id,id,id))objc_msgSend)(g_AR,
        @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:),
        @[wI], @[@0], @[wO], @[@0], nil, nil, @0);
    return k;
}

// Create a new _ANERequest with specified input and output IOSurfaces
static id make_request(IOSurfaceRef ioIn, IOSurfaceRef ioOut) {
    id wI = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_AIO, @selector(objectWithIOSurface:), ioIn);
    id wO = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_AIO, @selector(objectWithIOSurface:), ioOut);
    return ((id(*)(Class,SEL,id,id,id,id,id,id,id))objc_msgSend)(g_AR,
        @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:),
        @[wI], @[@0], @[wO], @[@0], nil, nil, @0);
}

static BOOL ane_eval(id aneModel, id request) {
    NSError *e = nil;
    BOOL ok = ((BOOL(*)(id,SEL,id,id,id,unsigned int,NSError**))objc_msgSend)(
        g_client, @selector(doEvaluateDirectWithModel:options:request:qos:error:),
        aneModel, @{}, request, 21, &e);
    if (!ok) printf("  [ANE EVAL FAIL] %s\n", e ? [[e description] UTF8String] : "no error");
    return ok;
}

static void print_surface_fp16(const char *label, IOSurfaceRef s, int count) {
    _Float16 *p = (_Float16*)IOSurfaceGetBaseAddress(s);
    printf("  %s:", label);
    for (int i = 0; i < count; i++) printf(" %.4f", (float)p[i]);
    printf("\n");
}

static void fill_surface_pattern(IOSurfaceRef s, size_t bytes, uint16_t pattern) {
    uint16_t *p = (uint16_t*)IOSurfaceGetBaseAddress(s);
    for (size_t i = 0; i < bytes / 2; i++) p[i] = pattern;
}

static BOOL surface_has_pattern(IOSurfaceRef s, size_t bytes, uint16_t pattern) {
    uint16_t *p = (uint16_t*)IOSurfaceGetBaseAddress(s);
    for (size_t i = 0; i < bytes / 2; i++) {
        if (p[i] != pattern) return NO;
    }
    return YES;
}

static BOOL surfaces_match(IOSurfaceRef a, IOSurfaceRef b, size_t bytes) {
    return memcmp(IOSurfaceGetBaseAddress(a), IOSurfaceGetBaseAddress(b), bytes) == 0;
}

int main(int argc, char *argv[]) {
    @autoreleasepool {
        setbuf(stdout, NULL);
        dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine", RTLD_NOW);
        g_AIO = NSClassFromString(@"_ANEIOSurfaceObject");
        g_AR  = NSClassFromString(@"_ANERequest");
        g_client = ((id(*)(Class,SEL))objc_msgSend)(NSClassFromString(@"_ANEClient"), @selector(sharedConnection));

        printf("=== ANE Output IOSurface Binding Probe ===\n\n");

        int ic = 64, oc = 64, seq = 64;
        size_t outB = oc * seq * 2;  // 8192 bytes

        // ===== Step 1: Compile kernel =====
        printf("--- Step 1: Compile 64x64 matmul kernel ---\n");
        Kern k = compile_kernel();
        if (!k.aneModel) { printf("FATAL: compilation failed\n"); return 1; }
        printf("  Compiled OK. ioIn=%p, ioOut=%p (original)\n",
               (void*)k.ioIn, (void*)k.ioOut);

        // ===== Step 2: Fill input with 1.0, eval with original request =====
        printf("\n--- Step 2: Baseline eval (input=1.0, original output surface) ---\n");
        {
            _Float16 *inp = (_Float16*)IOSurfaceGetBaseAddress(k.ioIn);
            size_t inB = ic * (seq + oc) * 2;
            // Fill activation region [0:ic*seq] with 1.0
            for (int i = 0; i < ic * seq; i++) inp[i] = (_Float16)1.0f;
            // Fill weight region [ic*seq:end] with identity-like values
            // Weight is [64,64], use small values so output is predictable
            for (int ch = 0; ch < ic; ch++) {
                for (int j = 0; j < oc; j++) {
                    // Weight at channel ch, spatial offset (seq + j)
                    inp[ch * (seq + oc) + seq + j] = (_Float16)(ch == j ? 1.0f : 0.0f);
                }
            }
        }
        ane_eval(k.aneModel, k.request);
        printf("  Original ioOut after eval:\n");
        print_surface_fp16("ioOut[0:7]", k.ioOut, 8);

        // Save reference output for comparison
        uint8_t refOut[8192];
        memcpy(refOut, IOSurfaceGetBaseAddress(k.ioOut), outB);

        // ===== Step 3: Create alternate output surface =====
        printf("\n--- Step 3: Create alternate output surface (altOut) ---\n");
        IOSurfaceRef altOut = make_surf(outB);
        printf("  altOut=%p, size=%zu\n", (void*)altOut, outB);

        // ===== Step 4: Fill altOut with 0xDEAD sentinel =====
        printf("\n--- Step 4: Fill altOut with 0xDEAD sentinel ---\n");
        fill_surface_pattern(altOut, outB, 0xDEAD);
        printf("  altOut still has sentinel: %s\n",
               surface_has_pattern(altOut, outB, 0xDEAD) ? "YES" : "NO");
        print_surface_fp16("altOut[0:7]", altOut, 8);

        // Also clear original ioOut to 0xBEEF so we can tell if ANE writes to it
        printf("  Clearing original ioOut to 0xBEEF sentinel...\n");
        fill_surface_pattern(k.ioOut, outB, 0xBEEF);
        printf("  ioOut still has 0xBEEF: %s\n",
               surface_has_pattern(k.ioOut, outB, 0xBEEF) ? "YES" : "NO");

        // ===== Step 5: Create new request with same input, different output =====
        printf("\n--- Step 5: Create new _ANERequest with altOut ---\n");
        id altReq = make_request(k.ioIn, altOut);
        printf("  altReq=%p (original req=%p)\n", (__bridge void*)altReq, (__bridge void*)k.request);

        // ===== Step 6: Eval with the alternate request =====
        printf("\n--- Step 6: Eval with alternate request ---\n");
        BOOL evalOk = ane_eval(k.aneModel, altReq);
        printf("  Eval returned: %s\n", evalOk ? "SUCCESS" : "FAIL");

        // ===== Step 7: Read altOut — did ANE write here? =====
        printf("\n--- Step 7: Check altOut (did ANE write correct output?) ---\n");
        BOOL altStillSentinel = surface_has_pattern(altOut, outB, 0xDEAD);
        printf("  altOut still has 0xDEAD sentinel: %s\n", altStillSentinel ? "YES (ANE did NOT write)" : "NO (ANE wrote!)");
        print_surface_fp16("altOut[0:7]", altOut, 8);

        // Check if altOut matches reference output
        BOOL altMatchesRef = (memcmp(IOSurfaceGetBaseAddress(altOut), refOut, outB) == 0);
        printf("  altOut matches reference output: %s\n", altMatchesRef ? "YES" : "NO");

        // ===== Step 8: Read original ioOut — did ANE also write here? =====
        printf("\n--- Step 8: Check original ioOut (did ANE write here too?) ---\n");
        BOOL origStillSentinel = surface_has_pattern(k.ioOut, outB, 0xBEEF);
        printf("  ioOut still has 0xBEEF sentinel: %s\n", origStillSentinel ? "YES (ANE did NOT write)" : "NO (ANE wrote!)");
        print_surface_fp16("ioOut[0:7]", k.ioOut, 8);

        BOOL origMatchesRef = (memcmp(IOSurfaceGetBaseAddress(k.ioOut), refOut, outB) == 0);
        printf("  ioOut matches reference output: %s\n", origMatchesRef ? "YES" : "NO");

        // ===== Step 9: Summary =====
        printf("\n========== RESULTS ==========\n");
        if (!altStillSentinel && altMatchesRef && origStillSentinel) {
            printf("  DOUBLE-BUFFERING WORKS!\n");
            printf("  ANE respects per-request output IOSurface binding.\n");
            printf("  altOut received correct output, original ioOut untouched.\n");
        } else if (!altStillSentinel && altMatchesRef && !origStillSentinel && origMatchesRef) {
            printf("  ANE WRITES TO BOTH SURFACES!\n");
            printf("  altOut got correct output, but ioOut was also written.\n");
            printf("  Double-buffering is BROKEN — ANE writes to compile-time surface too.\n");
        } else if (altStillSentinel && !origStillSentinel && origMatchesRef) {
            printf("  ANE IGNORES REQUEST BINDING!\n");
            printf("  ANE wrote to the ORIGINAL ioOut, ignoring altOut in the request.\n");
            printf("  Double-buffering requires compiling separate kernels.\n");
        } else if (altStillSentinel && origStillSentinel) {
            printf("  ANE WROTE TO NEITHER SURFACE!\n");
            printf("  Eval may have failed silently or written to a DMA buffer.\n");
        } else {
            printf("  UNEXPECTED RESULT — manual inspection needed.\n");
            printf("  altOut sentinel cleared: %s, matches ref: %s\n",
                   altStillSentinel ? "NO" : "YES", altMatchesRef ? "YES" : "NO");
            printf("  ioOut sentinel cleared: %s, matches ref: %s\n",
                   origStillSentinel ? "NO" : "YES", origMatchesRef ? "YES" : "NO");
        }

        // ===== Bonus: Eval again with original request to confirm it still works =====
        printf("\n--- Bonus: Re-eval with original request (sanity check) ---\n");
        fill_surface_pattern(k.ioOut, outB, 0x0000);
        ane_eval(k.aneModel, k.request);
        BOOL origWorks = (memcmp(IOSurfaceGetBaseAddress(k.ioOut), refOut, outB) == 0);
        printf("  Original request still produces correct output: %s\n", origWorks ? "YES" : "NO");
        print_surface_fp16("ioOut[0:7]", k.ioOut, 8);

        printf("\nDone.\n");
        CFRelease(altOut);
    }
    return 0;
}
