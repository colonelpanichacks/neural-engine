// probe_privileged.m — Explore the privileged ANE controller path
// Key question: Can we use sharedPrivilegedConnection's controller to create
// a PFE that dispatches differently (faster, chaining-enabled, etc.)?
//
// Previous findings:
// - sharedPrivilegedConnection returns a DIFFERENT controller with isPrivileged=1
// - initWithProgramHandle:priviledged:YES gives NULL device pointer
// - Normal controller from [pfe controller] has a valid device pointer
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

typedef struct { id model, aneModel; IOSurfaceRef ioIn, ioOut; id request; size_t outB; } Kern;

static Kern compile(int ic, int oc, int seq) {
    Kern k = {0};
    Class D = NSClassFromString(@"_ANEInMemoryModelDescriptor");
    Class I = NSClassFromString(@"_ANEInMemoryModel");
    NSData *mil = gen_mil(ic, oc, seq);
    size_t inB = ic*(seq+oc)*2;
    k.outB = oc*seq*2;
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
    k.ioIn = make_surf(inB); k.ioOut = make_surf(k.outB);
    _Float16 *p = (void*)IOSurfaceGetBaseAddress(k.ioIn);
    for (size_t i = 0; i < inB/2; i++) p[i] = (_Float16)(0.01f*(i%100));
    id wI = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_AIO, @selector(objectWithIOSurface:), k.ioIn);
    id wO = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_AIO, @selector(objectWithIOSurface:), k.ioOut);
    k.request = ((id(*)(Class,SEL,id,id,id,id,id,id,id))objc_msgSend)(g_AR,
        @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:),
        @[wI], @[@0], @[wO], @[@0], nil, nil, @0);
    return k;
}

static void dump_methods(Class cls, const char *name) {
    unsigned int count = 0;
    Method *methods = class_copyMethodList(cls, &count);
    printf("  %s instance methods (%u):\n", name, count);
    for (unsigned int i = 0; i < count; i++) {
        const char *sel = sel_getName(method_getName(methods[i]));
        const char *enc = method_getTypeEncoding(methods[i]);
        printf("    %-80s  %s\n", sel, enc ? enc : "(nil)");
    }
    free(methods);
    unsigned int ccount = 0;
    Method *cmethods = class_copyMethodList(object_getClass(cls), &ccount);
    printf("  %s class methods (%u):\n", name, ccount);
    for (unsigned int i = 0; i < ccount; i++) {
        const char *sel = sel_getName(method_getName(cmethods[i]));
        const char *enc = method_getTypeEncoding(cmethods[i]);
        printf("    %-80s  %s\n", sel, enc ? enc : "(nil)");
    }
    free(cmethods);
}

static void dump_properties(Class cls, const char *name) {
    unsigned int count = 0;
    objc_property_t *props = class_copyPropertyList(cls, &count);
    printf("  %s properties (%u):\n", name, count);
    for (unsigned int i = 0; i < count; i++) {
        const char *pname = property_getName(props[i]);
        const char *attr = property_getAttributes(props[i]);
        printf("    %-40s  %s\n", pname, attr ? attr : "(nil)");
    }
    free(props);
}

static void dump_ivars(Class cls, const char *name) {
    unsigned int count = 0;
    Ivar *ivars = class_copyIvarList(cls, &count);
    printf("  %s ivars (%u):\n", name, count);
    for (unsigned int i = 0; i < count; i++) {
        const char *iname = ivar_getName(ivars[i]);
        const char *itype = ivar_getTypeEncoding(ivars[i]);
        ptrdiff_t off = ivar_getOffset(ivars[i]);
        printf("    offset=%-4td  %-40s  %s\n", off, iname ? iname : "(anon)", itype ? itype : "(nil)");
    }
    free(ivars);
}

// Read an ivar value as uint64 from an object
static uint64_t read_ivar_u64(id obj, const char *ivarName) {
    Ivar iv = class_getInstanceVariable([obj class], ivarName);
    if (!iv) return 0;
    ptrdiff_t off = ivar_getOffset(iv);
    return *(uint64_t *)((uint8_t *)(__bridge void *)obj + off);
}

// Print first N output values
static void print_output(IOSurfaceRef surf, int n, const char *label) {
    _Float16 *out = (_Float16 *)IOSurfaceGetBaseAddress(surf);
    printf("  %s output[0..%d]:", label, n-1);
    for (int i = 0; i < n; i++) printf(" %.4f", (float)out[i]);
    printf("\n");
}

int main(int argc, char *argv[]) {
    @autoreleasepool {
        setbuf(stdout, NULL);
        mach_timebase_info(&g_tb);
        dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine", RTLD_NOW);
        g_AIO = NSClassFromString(@"_ANEIOSurfaceObject");
        g_AR  = NSClassFromString(@"_ANERequest");
        g_client = ((id(*)(Class,SEL))objc_msgSend)(NSClassFromString(@"_ANEClient"), @selector(sharedConnection));

        Class dcClass = NSClassFromString(@"_ANEDeviceController");
        Class PFECls  = NSClassFromString(@"_ANEProgramForEvaluation");

        printf("=== ANE Privileged Controller Probe ===\n\n");

        // ============================================================
        // Step 1: Compile 64x64 matmul kernel
        // ============================================================
        printf("=== Step 1: Compile 64x64 matmul kernel ===\n");
        Kern k = compile(64, 64, 64);
        printf("  aneModel: %s <%p>\n", class_getName([k.aneModel class]), k.aneModel);

        // Baseline eval for reference output
        {
            NSError *e = nil;
            BOOL ok = ((BOOL(*)(id,SEL,id,id,id,unsigned int,NSError**))objc_msgSend)(
                g_client, @selector(doEvaluateDirectWithModel:options:request:qos:error:),
                k.aneModel, @{}, k.request, (unsigned int)21, &e);
            printf("  Baseline eval: %s\n", ok ? "OK" : "FAIL");
            if (e) printf("  Error: %s\n", [[e description] UTF8String]);
        }
        // Save reference output
        _Float16 ref_out[4];
        memcpy(ref_out, IOSurfaceGetBaseAddress(k.ioOut), sizeof(ref_out));
        print_output(k.ioOut, 4, "Reference");

        // ============================================================
        // Step 2: Get normal PFE via [aneModel program]
        // ============================================================
        printf("\n=== Step 2: Get normal PFE ===\n");
        id pfe = ((id(*)(id,SEL))objc_msgSend)(k.aneModel, @selector(program));
        printf("  [aneModel program] -> %s <%p>\n",
               pfe ? class_getName([pfe class]) : "nil", pfe);

        if (!pfe) {
            printf("  FATAL: Could not get PFE. Aborting.\n");
            return 1;
        }

        // Get normal controller from PFE
        id normalCtrl = ((id(*)(id,SEL))objc_msgSend)(pfe, @selector(controller));
        printf("  [pfe controller] -> %s <%p>\n",
               normalCtrl ? class_getName([normalCtrl class]) : "nil", normalCtrl);

        // Read key properties from normal PFE
        uint64_t pfeProgHandle = ((uint64_t(*)(id,SEL))objc_msgSend)(pfe, @selector(programHandle));
        uint64_t pfeIntBufHandle = 0;
        if ([pfe respondsToSelector:@selector(intermediateBufferHandle)]) {
            pfeIntBufHandle = ((uint64_t(*)(id,SEL))objc_msgSend)(pfe, @selector(intermediateBufferHandle));
        }
        char pfeQD = ((char(*)(id,SEL))objc_msgSend)(pfe, @selector(queueDepth));
        printf("  Normal PFE: programHandle=%llu, intermediateBufferHandle=%llu, queueDepth=%d\n",
               pfeProgHandle, pfeIntBufHandle, (int)pfeQD);

        // Read normal controller properties
        if (normalCtrl) {
            printf("\n  Normal controller properties:\n");
            if ([normalCtrl respondsToSelector:@selector(isPrivileged)]) {
                BOOL ip = ((BOOL(*)(id,SEL))objc_msgSend)(normalCtrl, @selector(isPrivileged));
                printf("    isPrivileged = %d\n", ip);
            }
            if ([normalCtrl respondsToSelector:@selector(device)]) {
                void *dp = ((void*(*)(id,SEL))objc_msgSend)(normalCtrl, @selector(device));
                printf("    device = %p\n", dp);
            }
            if ([normalCtrl respondsToSelector:@selector(programHandle)]) {
                long long ph = ((long long(*)(id,SEL))objc_msgSend)(normalCtrl, @selector(programHandle));
                printf("    programHandle = %lld\n", ph);
            }
        }

        // ============================================================
        // Step 3: Get privileged controller via sharedPrivilegedConnection
        // ============================================================
        printf("\n=== Step 3: Get privileged controller ===\n");
        id privCtrl = nil;
        if ([dcClass respondsToSelector:@selector(sharedPrivilegedConnection)]) {
            @try {
                privCtrl = ((id(*)(Class,SEL))objc_msgSend)(dcClass, @selector(sharedPrivilegedConnection));
                printf("  +sharedPrivilegedConnection -> %s <%p>\n",
                       privCtrl ? class_getName([privCtrl class]) : "nil", privCtrl);
            } @catch (NSException *ex) {
                printf("  sharedPrivilegedConnection threw: %s\n", [[ex description] UTF8String]);
            }
        } else {
            printf("  _ANEDeviceController does NOT respond to +sharedPrivilegedConnection\n");
        }

        if (!privCtrl) {
            printf("  FATAL: Could not get privileged controller. Aborting.\n");
            return 1;
        }

        // Compare controllers
        printf("  Same as normalCtrl? %s\n", privCtrl == normalCtrl ? "YES" : "NO");
        if ([privCtrl respondsToSelector:@selector(isPrivileged)]) {
            BOOL ip = ((BOOL(*)(id,SEL))objc_msgSend)(privCtrl, @selector(isPrivileged));
            printf("  privCtrl.isPrivileged = %d\n", ip);
        }
        if ([privCtrl respondsToSelector:@selector(device)]) {
            void *dp = ((void*(*)(id,SEL))objc_msgSend)(privCtrl, @selector(device));
            printf("  privCtrl.device = %p\n", dp);
        }
        if ([privCtrl respondsToSelector:@selector(programHandle)]) {
            long long ph = ((long long(*)(id,SEL))objc_msgSend)(privCtrl, @selector(programHandle));
            printf("  privCtrl.programHandle = %lld\n", ph);
        }

        // Dump privileged controller ivars to see what differs
        printf("\n  Privileged controller ivar values:\n");
        {
            unsigned int icount = 0;
            Ivar *ivars = class_copyIvarList([privCtrl class], &icount);
            for (unsigned int i = 0; i < icount; i++) {
                const char *iname = ivar_getName(ivars[i]);
                const char *itype = ivar_getTypeEncoding(ivars[i]);
                ptrdiff_t off = ivar_getOffset(ivars[i]);
                if (!iname) continue;
                // Read value based on type
                void *base = (__bridge void *)privCtrl;
                if (itype && (itype[0] == 'Q' || itype[0] == 'q' || itype[0] == '^')) {
                    uint64_t val = *(uint64_t *)((uint8_t *)base + off);
                    printf("    [priv] %-30s = 0x%llx\n", iname, val);
                } else if (itype && (itype[0] == 'I' || itype[0] == 'i')) {
                    uint32_t val = *(uint32_t *)((uint8_t *)base + off);
                    printf("    [priv] %-30s = %u (0x%x)\n", iname, val, val);
                } else if (itype && itype[0] == 'B') {
                    BOOL val = *(BOOL *)((uint8_t *)base + off);
                    printf("    [priv] %-30s = %d\n", iname, val);
                } else if (itype && itype[0] == 'c') {
                    char val = *(char *)((uint8_t *)base + off);
                    printf("    [priv] %-30s = %d\n", iname, val);
                } else if (itype && itype[0] == '@') {
                    id val = (__bridge id)(*(void **)((uint8_t *)base + off));
                    printf("    [priv] %-30s = %s <%p>\n", iname,
                           val ? class_getName([val class]) : "nil", val);
                }
            }
            free(ivars);

            // Same for normal controller
            if (normalCtrl) {
                printf("\n  Normal controller ivar values:\n");
                ivars = class_copyIvarList([normalCtrl class], &icount);
                for (unsigned int i = 0; i < icount; i++) {
                    const char *iname = ivar_getName(ivars[i]);
                    const char *itype = ivar_getTypeEncoding(ivars[i]);
                    ptrdiff_t off = ivar_getOffset(ivars[i]);
                    if (!iname) continue;
                    void *base = (__bridge void *)normalCtrl;
                    if (itype && (itype[0] == 'Q' || itype[0] == 'q' || itype[0] == '^')) {
                        uint64_t val = *(uint64_t *)((uint8_t *)base + off);
                        printf("    [norm] %-30s = 0x%llx\n", iname, val);
                    } else if (itype && (itype[0] == 'I' || itype[0] == 'i')) {
                        uint32_t val = *(uint32_t *)((uint8_t *)base + off);
                        printf("    [norm] %-30s = %u (0x%x)\n", iname, val, val);
                    } else if (itype && itype[0] == 'B') {
                        BOOL val = *(BOOL *)((uint8_t *)base + off);
                        printf("    [norm] %-30s = %d\n", iname, val);
                    } else if (itype && itype[0] == 'c') {
                        char val = *(char *)((uint8_t *)base + off);
                        printf("    [norm] %-30s = %d\n", iname, val);
                    } else if (itype && itype[0] == '@') {
                        id val = (__bridge id)(*(void **)((uint8_t *)base + off));
                        printf("    [norm] %-30s = %s <%p>\n", iname,
                               val ? class_getName([val class]) : "nil", val);
                    }
                }
                free(ivars);
            }
        }

        // ============================================================
        // Step 4: Try to create a NEW PFE using privileged controller
        // ============================================================
        printf("\n=== Step 4: Create new PFE with privileged controller ===\n");

        // First, dump PFE init methods
        printf("  PFE init methods:\n");
        {
            unsigned int count = 0;
            Method *methods = class_copyMethodList(PFECls, &count);
            for (unsigned int i = 0; i < count; i++) {
                const char *sel = sel_getName(method_getName(methods[i]));
                if (strstr(sel, "init") || strstr(sel, "With") || strstr(sel, "create")) {
                    const char *enc = method_getTypeEncoding(methods[i]);
                    printf("    %-80s  %s\n", sel, enc ? enc : "(nil)");
                }
            }
            free(methods);
        }

        // Try initWithController:intermediateBufferHandle:queueDepth: (correct signature from introspection)
        SEL initPFE1 = @selector(initWithController:intermediateBufferHandle:queueDepth:);
        if (class_getInstanceMethod(PFECls, initPFE1)) {
            printf("\n  Trying initWithController:intermediateBufferHandle:queueDepth:\n");
            @try {
                id newPFE = ((id(*)(id,SEL,id,uint64_t,char))objc_msgSend)(
                    [PFECls alloc], initPFE1,
                    privCtrl, pfeIntBufHandle, pfeQD);
                printf("    result -> %s <%p>\n",
                       newPFE ? class_getName([newPFE class]) : "nil", newPFE);
                if (newPFE) {
                    // Check its properties
                    id newCtrl = ((id(*)(id,SEL))objc_msgSend)(newPFE, @selector(controller));
                    printf("    [newPFE controller] -> %s <%p> (same as privCtrl? %s)\n",
                           newCtrl ? class_getName([newCtrl class]) : "nil", newCtrl,
                           newCtrl == privCtrl ? "YES" : "NO");
                    uint64_t newPH = ((uint64_t(*)(id,SEL))objc_msgSend)(newPFE, @selector(programHandle));
                    printf("    [newPFE programHandle] = %llu\n", newPH);

                    // ============================================================
                    // Step 5: Try processRequest on the privileged PFE
                    // ============================================================
                    printf("\n=== Step 5: processRequest on privileged PFE ===\n");

                    // Get string_id
                    uint64_t stringId = 0;
                    @try { stringId = ((uint64_t(*)(id,SEL))objc_msgSend)(k.aneModel, @selector(string_id)); }
                    @catch (NSException *ex) { printf("  string_id threw: %s\n", [[ex reason] UTF8String]); }

                    // Zero output
                    memset(IOSurfaceGetBaseAddress(k.ioOut), 0, k.outB);

                    @try {
                        NSError *err = nil;
                        unsigned int retVal = 0;
                        uint64_t t0 = mach_absolute_time();
                        BOOL ok = ((BOOL(*)(id,SEL,id,id,unsigned int,uint64_t,uint64_t,id,unsigned int *,NSError**))objc_msgSend)(
                            newPFE,
                            @selector(processRequest:model:qos:qIndex:modelStringID:options:returnValue:error:),
                            k.request, k.aneModel, (unsigned int)21, (uint64_t)0,
                            stringId, @{}, &retVal, &err);
                        double dt = ms_t(mach_absolute_time() - t0);
                        printf("  processRequest: ret=%d retVal=%u  %.3fms\n", (int)ok, retVal, dt);
                        if (err) printf("  Error: %s\n", [[err description] UTF8String]);
                        print_output(k.ioOut, 4, "Priv PFE");

                        // Compare to reference
                        _Float16 *out = (_Float16 *)IOSurfaceGetBaseAddress(k.ioOut);
                        BOOL match = YES;
                        for (int i = 0; i < 4; i++) {
                            if (fabsf((float)out[i] - (float)ref_out[i]) > 0.01f) { match = NO; break; }
                        }
                        printf("  Output matches reference: %s\n", match ? "YES" : "NO");
                    } @catch (NSException *ex) {
                        printf("  processRequest threw: %s\n", [[ex reason] UTF8String]);
                    }

                    // ============================================================
                    // Step 6: Compare speed: normal PFE vs privileged PFE
                    // ============================================================
                    printf("\n=== Step 6: Speed comparison (100 iters) ===\n");
                    int ITERS = 100;

                    // Normal PFE
                    {
                        uint64_t t0 = mach_absolute_time();
                        for (int i = 0; i < ITERS; i++) {
                            NSError *err = nil;
                            unsigned int rv = 0;
                            ((BOOL(*)(id,SEL,id,id,unsigned int,uint64_t,uint64_t,id,unsigned int *,NSError**))objc_msgSend)(
                                pfe,
                                @selector(processRequest:model:qos:qIndex:modelStringID:options:returnValue:error:),
                                k.request, k.aneModel, (unsigned int)21, (uint64_t)(i % (int)pfeQD),
                                stringId, @{}, &rv, &err);
                        }
                        double dt = ms_t(mach_absolute_time() - t0);
                        printf("  Normal PFE:     %d iters in %.2fms (%.3fms/iter)\n", ITERS, dt, dt/ITERS);
                    }

                    // Privileged PFE
                    {
                        uint64_t t0 = mach_absolute_time();
                        for (int i = 0; i < ITERS; i++) {
                            NSError *err = nil;
                            unsigned int rv = 0;
                            ((BOOL(*)(id,SEL,id,id,unsigned int,uint64_t,uint64_t,id,unsigned int *,NSError**))objc_msgSend)(
                                newPFE,
                                @selector(processRequest:model:qos:qIndex:modelStringID:options:returnValue:error:),
                                k.request, k.aneModel, (unsigned int)21, (uint64_t)(i % (int)pfeQD),
                                stringId, @{}, &rv, &err);
                        }
                        double dt = ms_t(mach_absolute_time() - t0);
                        printf("  Priv PFE:       %d iters in %.2fms (%.3fms/iter)\n", ITERS, dt, dt/ITERS);
                    }

                    // doEvaluateDirectWithModel for comparison
                    {
                        uint64_t t0 = mach_absolute_time();
                        for (int i = 0; i < ITERS; i++) {
                            NSError *err = nil;
                            ((BOOL(*)(id,SEL,id,id,id,unsigned int,NSError**))objc_msgSend)(
                                g_client, @selector(doEvaluateDirectWithModel:options:request:qos:error:),
                                k.aneModel, @{}, k.request, (unsigned int)21, &err);
                        }
                        double dt = ms_t(mach_absolute_time() - t0);
                        printf("  doEvalDirect:   %d iters in %.2fms (%.3fms/iter)\n", ITERS, dt, dt/ITERS);
                    }

                    // ============================================================
                    // Step 7: Try two-phase dispatch on privileged PFE
                    // ============================================================
                    printf("\n=== Step 7: Two-phase dispatch on privileged PFE ===\n");

                    Class ANEInputReady = NSClassFromString(@"_ANEInputBuffersReady");
                    Class ANEOutSetEnq  = NSClassFromString(@"_ANEOutputSetEnqueue");

                    // Create InputBuffersReady (with object params — type encoding shows @ not I)
                    id ibr = ((id(*)(id,SEL,unsigned int,id,id,unsigned long long))objc_msgSend)(
                        [ANEInputReady alloc],
                        @selector(initInputsProcedureIndex:inputBufferInfoIndex:inputFreeValue:executionDelay:),
                        (unsigned int)0, @0, @0, (unsigned long long)0);
                    printf("  InputBuffersReady: %s\n", ibr ? "OK" : "nil");

                    // Create OutputSetEnqueue
                    id ose = ((id(*)(id,SEL,unsigned int,unsigned int,unsigned long long,BOOL,BOOL))objc_msgSend)(
                        [ANEOutSetEnq alloc],
                        @selector(initOutputSetWithProcedureIndex:setIndex:signalValue:signalNotRequired:isOpenLoop:),
                        (unsigned int)0, (unsigned int)0, (unsigned long long)1, NO, NO);
                    printf("  OutputSetEnqueue:  %s\n", ose ? "OK" : "nil");

                    if (ibr && ose) {
                        // Zero output
                        memset(IOSurfaceGetBaseAddress(k.ioOut), 0, k.outB);

                        // Phase 1: processInputBuffers on privileged PFE
                        @try {
                            NSError *err = nil;
                            BOOL ok1 = ((BOOL(*)(id,SEL,id,id,id,NSError**))objc_msgSend)(
                                newPFE, @selector(processInputBuffers:model:options:error:),
                                ibr, k.aneModel, @{}, &err);
                            printf("  [privPFE processInputBuffers]: %s", ok1 ? "OK" : "FAIL");
                            if (err) printf(" err=%ld (%s)", (long)[err code], [[err localizedDescription] UTF8String]);
                            printf("\n");
                        } @catch (NSException *ex) {
                            printf("  processInputBuffers threw: %s\n", [[ex reason] UTF8String]);
                        }

                        // Phase 2: processOutputSet on privileged PFE
                        @try {
                            NSError *err = nil;
                            BOOL ok2 = ((BOOL(*)(id,SEL,id,id,id,NSError**))objc_msgSend)(
                                newPFE, @selector(processOutputSet:model:options:error:),
                                ose, k.aneModel, @{}, &err);
                            printf("  [privPFE processOutputSet]:   %s", ok2 ? "OK" : "FAIL");
                            if (err) printf(" err=%ld (%s)", (long)[err code], [[err localizedDescription] UTF8String]);
                            printf("\n");
                        } @catch (NSException *ex) {
                            printf("  processOutputSet threw: %s\n", [[ex reason] UTF8String]);
                        }

                        // Wait and check output
                        usleep(5000);
                        print_output(k.ioOut, 4, "Two-phase priv");
                        _Float16 *out2 = (_Float16 *)IOSurfaceGetBaseAddress(k.ioOut);
                        BOOL match2 = YES;
                        for (int i = 0; i < 4; i++) {
                            if (fabsf((float)out2[i] - (float)ref_out[i]) > 0.01f) { match2 = NO; break; }
                        }
                        printf("  Two-phase output matches reference: %s\n", match2 ? "YES" : "NO");

                        // Also try via _ANEClient two-phase with privileged PFE in play
                        printf("\n  Trying _ANEClient two-phase after priv PFE setup:\n");
                        memset(IOSurfaceGetBaseAddress(k.ioOut), 0, k.outB);
                        @try {
                            NSError *err = nil;
                            long ret1 = ((long(*)(id,SEL,id,id,id,unsigned int,NSError**))objc_msgSend)(
                                g_client, @selector(doBuffersReadyWithModel:inputBuffers:options:qos:error:),
                                k.aneModel, ibr, @{}, (unsigned int)21, &err);
                            printf("    doBuffersReady ret=%ld", ret1);
                            if (err) printf(" err=%ld", (long)[err code]);
                            printf("\n");

                            err = nil;
                            long ret2 = ((long(*)(id,SEL,id,id,id,unsigned int,NSError**))objc_msgSend)(
                                g_client, @selector(doEnqueueSetsWithModel:outputSet:options:qos:error:),
                                k.aneModel, ose, @{}, (unsigned int)21, &err);
                            printf("    doEnqueueSets ret=%ld", ret2);
                            if (err) printf(" err=%ld", (long)[err code]);
                            printf("\n");

                            usleep(5000);
                            print_output(k.ioOut, 4, "Client two-phase");
                        } @catch (NSException *ex) {
                            printf("    Client two-phase threw: %s\n", [[ex reason] UTF8String]);
                        }
                    }
                } else {
                    printf("    PFE creation returned nil\n");
                }
            } @catch (NSException *ex) {
                printf("    initWithController:... threw: %s\n", [[ex reason] UTF8String]);
            }
        } else {
            printf("  initWithController:intermediateBufferHandle:queueDepth: NOT FOUND\n");
        }

        // ============================================================
        // Step 8: Check privileged controller for additional methods/properties
        // ============================================================
        printf("\n=== Step 8: Privileged controller — unique methods/properties ===\n");
        if (privCtrl) {
            dump_properties([privCtrl class], "PrivCtrl");
            dump_ivars([privCtrl class], "PrivCtrl");
            dump_methods([privCtrl class], "PrivCtrl");

            // Try methods that might only work on privileged controller
            printf("\n  Probing privileged-only methods:\n");

            // Try prepareChainingWithModel on the privileged controller
            if ([privCtrl respondsToSelector:@selector(prepareChainingWithModel:options:error:)]) {
                @try {
                    NSError *err = nil;
                    BOOL ok = ((BOOL(*)(id,SEL,id,id,NSError**))objc_msgSend)(
                        privCtrl, @selector(prepareChainingWithModel:options:error:),
                        k.aneModel, @{}, &err);
                    printf("    [privCtrl prepareChainingWithModel]: %s", ok ? "OK" : "FAIL");
                    if (err) printf(" err=%ld (%s)", (long)[err code], [[err localizedDescription] UTF8String]);
                    printf("\n");
                } @catch (NSException *ex) {
                    printf("    prepareChainingWithModel threw: %s\n", [[ex reason] UTF8String]);
                }
            }

            // Check for privileged dispatch methods
            SEL privMethods[] = {
                @selector(evaluateWithRequest:error:),
                @selector(evaluateWithModel:request:error:),
                @selector(dispatchWithModel:request:error:),
                @selector(submitRequest:model:error:),
                @selector(processRequest:model:error:),
                @selector(executeWithModel:request:error:),
                @selector(evaluateDirectWithModel:request:error:),
                @selector(evaluateDirectWithModel:options:request:qos:error:),
            };
            const char *privMethodNames[] = {
                "evaluateWithRequest:error:",
                "evaluateWithModel:request:error:",
                "dispatchWithModel:request:error:",
                "submitRequest:model:error:",
                "processRequest:model:error:",
                "executeWithModel:request:error:",
                "evaluateDirectWithModel:request:error:",
                "evaluateDirectWithModel:options:request:qos:error:",
            };
            for (int i = 0; i < 8; i++) {
                if ([privCtrl respondsToSelector:privMethods[i]]) {
                    printf("    HAS -%s\n", privMethodNames[i]);
                }
            }

            // Try swapping the controller on the existing PFE
            printf("\n  Trying to swap controller on existing PFE:\n");
            if ([pfe respondsToSelector:@selector(setController:)]) {
                @try {
                    printf("    PFE responds to setController: — swapping...\n");
                    ((void(*)(id,SEL,id))objc_msgSend)(pfe, @selector(setController:), privCtrl);
                    id swappedCtrl = ((id(*)(id,SEL))objc_msgSend)(pfe, @selector(controller));
                    printf("    After swap: [pfe controller] -> %s <%p> (isPriv=%d)\n",
                           swappedCtrl ? class_getName([swappedCtrl class]) : "nil", swappedCtrl,
                           [swappedCtrl respondsToSelector:@selector(isPrivileged)] ?
                               (int)((BOOL(*)(id,SEL))objc_msgSend)(swappedCtrl, @selector(isPrivileged)) : -1);

                    // Try processRequest on the swapped PFE
                    uint64_t stringId = 0;
                    @try { stringId = ((uint64_t(*)(id,SEL))objc_msgSend)(k.aneModel, @selector(string_id)); }
                    @catch (NSException *ex) {}

                    memset(IOSurfaceGetBaseAddress(k.ioOut), 0, k.outB);
                    NSError *err = nil;
                    unsigned int retVal = 0;
                    uint64_t t0 = mach_absolute_time();
                    BOOL ok = ((BOOL(*)(id,SEL,id,id,unsigned int,uint64_t,uint64_t,id,unsigned int *,NSError**))objc_msgSend)(
                        pfe,
                        @selector(processRequest:model:qos:qIndex:modelStringID:options:returnValue:error:),
                        k.request, k.aneModel, (unsigned int)21, (uint64_t)0,
                        stringId, @{}, &retVal, &err);
                    double dt = ms_t(mach_absolute_time() - t0);
                    printf("    processRequest (swapped ctrl): ret=%d retVal=%u  %.3fms\n", (int)ok, retVal, dt);
                    if (err) printf("    Error: %s\n", [[err description] UTF8String]);
                    print_output(k.ioOut, 4, "Swapped PFE");

                    // Restore original controller
                    ((void(*)(id,SEL,id))objc_msgSend)(pfe, @selector(setController:), normalCtrl);
                    printf("    Restored original controller\n");
                } @catch (NSException *ex) {
                    printf("    setController threw: %s\n", [[ex reason] UTF8String]);
                    // Restore just in case
                    @try { ((void(*)(id,SEL,id))objc_msgSend)(pfe, @selector(setController:), normalCtrl); }
                    @catch (NSException *ex2) {}
                }
            } else {
                printf("    PFE does NOT respond to setController: (readonly)\n");

                // Try direct ivar write
                printf("    Trying direct ivar write of _controller...\n");
                Ivar ctrlIvar = class_getInstanceVariable(PFECls, "_controller");
                if (ctrlIvar) {
                    ptrdiff_t off = ivar_getOffset(ctrlIvar);
                    void *pfeBase = (__bridge void *)pfe;

                    // Save original
                    void *origCtrl = *(void **)((uint8_t *)pfeBase + off);

                    // Swap in privileged controller
                    *(void **)((uint8_t *)pfeBase + off) = (__bridge void *)privCtrl;
                    printf("    Wrote privCtrl to _controller ivar\n");

                    // Verify
                    id readBack = ((id(*)(id,SEL))objc_msgSend)(pfe, @selector(controller));
                    printf("    [pfe controller] now -> %s <%p>\n",
                           readBack ? class_getName([readBack class]) : "nil", readBack);

                    // Try processRequest
                    uint64_t stringId = 0;
                    @try { stringId = ((uint64_t(*)(id,SEL))objc_msgSend)(k.aneModel, @selector(string_id)); }
                    @catch (NSException *ex) {}

                    memset(IOSurfaceGetBaseAddress(k.ioOut), 0, k.outB);
                    @try {
                        NSError *err = nil;
                        unsigned int retVal = 0;
                        uint64_t t0 = mach_absolute_time();
                        BOOL ok = ((BOOL(*)(id,SEL,id,id,unsigned int,uint64_t,uint64_t,id,unsigned int *,NSError**))objc_msgSend)(
                            pfe,
                            @selector(processRequest:model:qos:qIndex:modelStringID:options:returnValue:error:),
                            k.request, k.aneModel, (unsigned int)21, (uint64_t)0,
                            stringId, @{}, &retVal, &err);
                        double dt = ms_t(mach_absolute_time() - t0);
                        printf("    processRequest (ivar-swapped): ret=%d retVal=%u  %.3fms\n", (int)ok, retVal, dt);
                        if (err) printf("    Error: %s\n", [[err description] UTF8String]);
                        print_output(k.ioOut, 4, "Ivar-swapped PFE");

                        _Float16 *out = (_Float16 *)IOSurfaceGetBaseAddress(k.ioOut);
                        BOOL match = YES;
                        for (int i = 0; i < 4; i++) {
                            if (fabsf((float)out[i] - (float)ref_out[i]) > 0.01f) { match = NO; break; }
                        }
                        printf("    Output matches reference: %s\n", match ? "YES" : "NO");

                        // Bench swapped
                        int ITERS2 = 100;
                        t0 = mach_absolute_time();
                        for (int i = 0; i < ITERS2; i++) {
                            err = nil;
                            unsigned int rv = 0;
                            ((BOOL(*)(id,SEL,id,id,unsigned int,uint64_t,uint64_t,id,unsigned int *,NSError**))objc_msgSend)(
                                pfe,
                                @selector(processRequest:model:qos:qIndex:modelStringID:options:returnValue:error:),
                                k.request, k.aneModel, (unsigned int)21, (uint64_t)(i % (int)pfeQD),
                                stringId, @{}, &rv, &err);
                        }
                        double dt2 = ms_t(mach_absolute_time() - t0);
                        printf("    Ivar-swapped: %d iters in %.2fms (%.3fms/iter)\n", ITERS2, dt2, dt2/ITERS2);
                    } @catch (NSException *ex) {
                        printf("    processRequest (ivar-swapped) threw: %s\n", [[ex reason] UTF8String]);
                    }

                    // Restore original
                    *(void **)((uint8_t *)pfeBase + off) = origCtrl;
                    printf("    Restored original _controller ivar\n");

                    // Also try two-phase on ivar-swapped PFE
                    printf("\n    Two-phase on ivar-swapped PFE:\n");
                    *(void **)((uint8_t *)pfeBase + off) = (__bridge void *)privCtrl;

                    Class ANEInputReady = NSClassFromString(@"_ANEInputBuffersReady");
                    Class ANEOutSetEnq  = NSClassFromString(@"_ANEOutputSetEnqueue");
                    id ibr2 = ((id(*)(id,SEL,unsigned int,id,id,unsigned long long))objc_msgSend)(
                        [ANEInputReady alloc],
                        @selector(initInputsProcedureIndex:inputBufferInfoIndex:inputFreeValue:executionDelay:),
                        (unsigned int)0, nil, nil, (unsigned long long)0);
                    id ose2 = ((id(*)(id,SEL,unsigned int,unsigned int,unsigned long long,BOOL,BOOL))objc_msgSend)(
                        [ANEOutSetEnq alloc],
                        @selector(initOutputSetWithProcedureIndex:setIndex:signalValue:signalNotRequired:isOpenLoop:),
                        (unsigned int)0, (unsigned int)0, (unsigned long long)1, NO, NO);

                    memset(IOSurfaceGetBaseAddress(k.ioOut), 0, k.outB);
                    @try {
                        NSError *err = nil;
                        BOOL ok1 = ((BOOL(*)(id,SEL,id,id,id,NSError**))objc_msgSend)(
                            pfe, @selector(processInputBuffers:model:options:error:),
                            ibr2, k.aneModel, @{}, &err);
                        printf("      processInputBuffers: %s", ok1 ? "OK" : "FAIL");
                        if (err) printf(" err=%ld", (long)[err code]);
                        printf("\n");

                        err = nil;
                        BOOL ok2 = ((BOOL(*)(id,SEL,id,id,id,NSError**))objc_msgSend)(
                            pfe, @selector(processOutputSet:model:options:error:),
                            ose2, k.aneModel, @{}, &err);
                        printf("      processOutputSet:   %s", ok2 ? "OK" : "FAIL");
                        if (err) printf(" err=%ld", (long)[err code]);
                        printf("\n");

                        usleep(5000);
                        print_output(k.ioOut, 4, "Two-phase ivar-swapped");
                    } @catch (NSException *ex) {
                        printf("      Two-phase threw: %s\n", [[ex reason] UTF8String]);
                    }

                    // Restore
                    *(void **)((uint8_t *)pfeBase + off) = origCtrl;
                    printf("    Restored original _controller ivar\n");
                } else {
                    printf("    _controller ivar not found in PFE\n");
                }
            }
        }

        // ============================================================
        // Step 9: Search for any methods on privileged ctrl not on normal
        // ============================================================
        printf("\n=== Step 9: Methods unique to each controller instance ===\n");
        if (privCtrl && normalCtrl) {
            // Both are same class, but check if there are selector differences
            // (there shouldn't be for ObjC, but check runtime)
            printf("  privCtrl class: %s\n", class_getName([privCtrl class]));
            printf("  normalCtrl class: %s\n", class_getName([normalCtrl class]));
            printf("  Same class? %s\n",
                   [privCtrl class] == [normalCtrl class] ? "YES" : "NO");

            // Check if privileged controller has a different superclass/isa
            Class pc = object_getClass(privCtrl);
            Class nc = object_getClass(normalCtrl);
            printf("  privCtrl isa: %s\n", class_getName(pc));
            printf("  normalCtrl isa: %s\n", class_getName(nc));

            // Try processRequest directly on the privileged controller
            printf("\n  Trying dispatch methods on privileged controller directly:\n");
            SEL directSels[] = {
                @selector(processRequest:model:qos:qIndex:modelStringID:options:returnValue:error:),
                @selector(evaluateWithModel:options:request:qos:error:),
            };
            const char *directNames[] = {
                "processRequest:model:qos:qIndex:modelStringID:options:returnValue:error:",
                "evaluateWithModel:options:request:qos:error:",
            };
            for (int i = 0; i < 2; i++) {
                if ([privCtrl respondsToSelector:directSels[i]]) {
                    printf("    privCtrl HAS -%s\n", directNames[i]);
                } else {
                    printf("    privCtrl NO  -%s\n", directNames[i]);
                }
            }
        }

        // ============================================================
        // Cleanup
        // ============================================================
        printf("\n=== Cleanup ===\n");
        {
            NSError *e = nil;
            ((BOOL(*)(id,SEL,unsigned int,NSError**))objc_msgSend)(k.model, @selector(unloadWithQoS:error:), 21, &e);
            if (e) printf("  Unload error: %s\n", [[e description] UTF8String]);
        }
        CFRelease(k.ioIn); CFRelease(k.ioOut);

        printf("\n=== DONE ===\n");
    }
    return 0;
}
