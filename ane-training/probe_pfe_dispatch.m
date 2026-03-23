// probe_pfe_dispatch.m — Explore _ANEProgramForEvaluation two-phase dispatch
// This is the executor-level dispatch path that CoreML uses internally,
// below the _ANEClient layer.
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

static void dump_methods_with_encodings(Class cls, const char *name) {
    unsigned int count = 0;
    Method *methods = class_copyMethodList(cls, &count);
    printf("  %s instance methods (%u):\n", name, count);
    for (unsigned int i = 0; i < count; i++) {
        const char *sel = sel_getName(method_getName(methods[i]));
        const char *enc = method_getTypeEncoding(methods[i]);
        printf("    %-80s  %s\n", sel, enc ? enc : "(nil)");
    }
    free(methods);

    // Also class methods
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

int main(int argc, char *argv[]) {
    @autoreleasepool {
        setbuf(stdout, NULL);
        mach_timebase_info(&g_tb);
        dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine", RTLD_NOW);
        g_AIO = NSClassFromString(@"_ANEIOSurfaceObject");
        g_AR  = NSClassFromString(@"_ANERequest");
        g_client = ((id(*)(Class,SEL))objc_msgSend)(NSClassFromString(@"_ANEClient"), @selector(sharedConnection));

        printf("=== ANE _ANEProgramForEvaluation Dispatch Probe ===\n\n");

        // ============================================================
        // Step 1: Compile a simple 64x64 matmul kernel
        // ============================================================
        printf("=== Step 1: Compile 64x64 matmul kernel ===\n");
        Kern k = compile(64, 64, 64);
        printf("  Kernel compiled and loaded\n");
        printf("  aneModel class: %s\n", class_getName([k.aneModel class]));

        // Verify with baseline eval
        {
            NSError *e = nil;
            BOOL ok = ((BOOL(*)(id,SEL,id,id,id,unsigned int,NSError**))objc_msgSend)(
                g_client, @selector(doEvaluateDirectWithModel:options:request:qos:error:),
                k.aneModel, @{}, k.request, (unsigned int)21, &e);
            _Float16 *out = (_Float16*)IOSurfaceGetBaseAddress(k.ioOut);
            printf("  Baseline eval: %s, output[0..3]: %.4f %.4f %.4f %.4f\n",
                   ok ? "OK" : "FAIL", (float)out[0], (float)out[1], (float)out[2], (float)out[3]);
            if (e) printf("  Error: %s\n", [[e description] UTF8String]);
        }

        // ============================================================
        // Step 2: Get the _ANEProgramForEvaluation from aneModel.program
        // ============================================================
        printf("\n=== Step 2: Get _ANEProgramForEvaluation ===\n");

        // First dump _ANEModel methods to find the right accessor
        dump_methods_with_encodings([k.aneModel class], class_getName([k.aneModel class]));
        dump_properties([k.aneModel class], class_getName([k.aneModel class]));

        id pfe = nil;
        @try {
            pfe = ((id(*)(id,SEL))objc_msgSend)(k.aneModel, @selector(program));
            printf("\n  [aneModel program] => %s (class: %s)\n",
                   pfe ? [[pfe description] UTF8String] : "nil",
                   pfe ? class_getName([pfe class]) : "nil");
        } @catch (NSException *ex) {
            printf("  [aneModel program] threw: %s\n", [[ex reason] UTF8String]);
        }

        // If program didn't work, try programForEvaluation
        if (!pfe) {
            @try {
                pfe = ((id(*)(id,SEL))objc_msgSend)(k.aneModel, @selector(programForEvaluation));
                printf("  [aneModel programForEvaluation] => %s (class: %s)\n",
                       pfe ? [[pfe description] UTF8String] : "nil",
                       pfe ? class_getName([pfe class]) : "nil");
            } @catch (NSException *ex) {
                printf("  [aneModel programForEvaluation] threw: %s\n", [[ex reason] UTF8String]);
            }
        }

        if (!pfe) {
            printf("  Trying programHandle instead...\n");
            @try {
                pfe = ((id(*)(id,SEL))objc_msgSend)(k.aneModel, @selector(programHandle));
                printf("  [aneModel programHandle] => %s (class: %s)\n",
                       pfe ? [[pfe description] UTF8String] : "nil",
                       pfe ? class_getName([pfe class]) : "nil");
            } @catch (NSException *ex) {
                printf("  [aneModel programHandle] threw: %s\n", [[ex reason] UTF8String]);
            }
        }

        // ============================================================
        // Step 3: Dump _ANEProgramForEvaluation methods with type encodings
        // ============================================================
        printf("\n=== Step 3: _ANEProgramForEvaluation class introspection ===\n");
        Class PFECls = NSClassFromString(@"_ANEProgramForEvaluation");
        if (PFECls) {
            dump_methods_with_encodings(PFECls, "_ANEProgramForEvaluation");
            dump_properties(PFECls, "_ANEProgramForEvaluation");

            // Check superclass
            Class super = class_getSuperclass(PFECls);
            printf("\n  Superclass: %s\n", super ? class_getName(super) : "nil");
            if (super && super != [NSObject class]) {
                dump_methods_with_encodings(super, class_getName(super));
            }
        } else {
            printf("  _ANEProgramForEvaluation class NOT FOUND\n");
        }

        // If we couldn't get pfe from aneModel, try to find it another way
        if (!pfe && PFECls) {
            printf("\n  Trying to find PFE via KVC on aneModel...\n");
            @try {
                // Try various KVC paths
                NSArray *keys = @[@"program", @"programForEvaluation", @"_program",
                                  @"_programForEvaluation", @"evaluationProgram"];
                for (NSString *key in keys) {
                    @try {
                        id val = [k.aneModel valueForKey:key];
                        if (val) {
                            printf("  [aneModel valueForKey:@\"%s\"] => %s (class: %s)\n",
                                   [key UTF8String], [[val description] UTF8String],
                                   class_getName([val class]));
                            if ([val isKindOfClass:PFECls]) {
                                pfe = val;
                                break;
                            }
                        }
                    } @catch (NSException *ex) {
                        // silently skip
                    }
                }
            } @catch (NSException *ex) {
                printf("  KVC threw: %s\n", [[ex reason] UTF8String]);
            }
        }

        // Also try getting it from _ANEInMemoryModel
        if (!pfe) {
            printf("  Trying [inMemoryModel program]...\n");
            @try {
                pfe = ((id(*)(id,SEL))objc_msgSend)(k.model, @selector(program));
                printf("  [inMemoryModel program] => %s (class: %s)\n",
                       pfe ? [[pfe description] UTF8String] : "nil",
                       pfe ? class_getName([pfe class]) : "nil");
            } @catch (NSException *ex) {
                printf("  threw: %s\n", [[ex reason] UTF8String]);
            }
        }

        // ============================================================
        // Step 4: Try processRequest:model:qos:qIndex:modelStringID:options:returnValue:error:
        // ============================================================
        printf("\n=== Step 4: Try processRequest on PFE ===\n");
        // Type encoding: B76@0:8@16@24I32Q36Q44@52^I60^@68
        // Args: request(@), model(@), qos(I/uint), qIndex(Q/uint64), modelStringID(Q/uint64), options(@), returnValue(^I/uint*), error(^@)
        if (pfe) {
            // First get the string_id from the model
            uint64_t stringId = 0;
            @try {
                stringId = ((uint64_t(*)(id,SEL))objc_msgSend)(k.aneModel, @selector(string_id));
                printf("  aneModel string_id: %llu\n", stringId);
            } @catch (NSException *ex) {
                printf("  string_id threw: %s (using 0)\n", [[ex reason] UTF8String]);
            }

            @try {
                NSError *err = nil;
                unsigned int retVal = 0;
                BOOL ok = ((BOOL(*)(id,SEL,id,id,unsigned int,uint64_t,uint64_t,id,unsigned int *,NSError**))objc_msgSend)(
                    pfe,
                    @selector(processRequest:model:qos:qIndex:modelStringID:options:returnValue:error:),
                    k.request, k.aneModel, (unsigned int)21, (uint64_t)0,
                    stringId, @{}, &retVal, &err);
                printf("  processRequest ret=%d retVal=%u\n", (int)ok, retVal);
                if (err) printf("  Error: %s\n", [[err description] UTF8String]);

                // Check output after processRequest
                _Float16 *out = (_Float16*)IOSurfaceGetBaseAddress(k.ioOut);
                printf("  output[0..3]: %.4f %.4f %.4f %.4f\n",
                       (float)out[0], (float)out[1], (float)out[2], (float)out[3]);
            } @catch (NSException *ex) {
                printf("  processRequest threw: %s\n", [[ex reason] UTF8String]);
            }

            // Also try with qIndex=1
            @try {
                NSError *err = nil;
                unsigned int retVal = 0;
                BOOL ok = ((BOOL(*)(id,SEL,id,id,unsigned int,uint64_t,uint64_t,id,unsigned int *,NSError**))objc_msgSend)(
                    pfe,
                    @selector(processRequest:model:qos:qIndex:modelStringID:options:returnValue:error:),
                    k.request, k.aneModel, (unsigned int)21, (uint64_t)1,
                    stringId, @{}, &retVal, &err);
                printf("  processRequest (qIndex=1) ret=%d retVal=%u\n", (int)ok, retVal);
                if (err) printf("  Error: %s\n", [[err description] UTF8String]);
            } @catch (NSException *ex) {
                printf("  processRequest (qIndex=1) threw: %s\n", [[ex reason] UTF8String]);
            }
        } else {
            printf("  SKIPPED (no PFE object)\n");
        }

        // ============================================================
        // Step 5: Try processInputBuffers:model:options:error:
        // ============================================================
        printf("\n=== Step 5: Try processInputBuffers (phase 1) ===\n");
        Class ANEInputReady = NSClassFromString(@"_ANEInputBuffersReady");
        id ibr = nil;
        @try {
            ibr = ((id(*)(id,SEL,unsigned int,unsigned int,unsigned long long,unsigned int))objc_msgSend)(
                [ANEInputReady alloc],
                @selector(initInputsProcedureIndex:inputBufferInfoIndex:inputFreeValue:executionDelay:),
                (unsigned int)0, (unsigned int)0, (unsigned long long)0, (unsigned int)0);
            printf("  InputBuffersReady: %s\n", ibr ? "OK" : "nil");
        } @catch (NSException *ex) {
            printf("  InputBuffersReady init threw: %s\n", [[ex reason] UTF8String]);
        }

        if (pfe && ibr) {
            @try {
                NSError *err = nil;
                BOOL ok = ((BOOL(*)(id,SEL,id,id,id,NSError**))objc_msgSend)(
                    pfe,
                    @selector(processInputBuffers:model:options:error:),
                    ibr, k.aneModel, @{}, &err);
                printf("  processInputBuffers ret=%d\n", (int)ok);
                if (err) printf("  Error: %s\n", [[err description] UTF8String]);
            } @catch (NSException *ex) {
                printf("  processInputBuffers threw: %s\n", [[ex reason] UTF8String]);
            }
        } else {
            printf("  SKIPPED (pfe=%s, ibr=%s)\n", pfe ? "yes" : "no", ibr ? "yes" : "no");
        }

        // ============================================================
        // Step 6: Try processOutputSet:model:options:error: (phase 2)
        // ============================================================
        printf("\n=== Step 6: Try processOutputSet (phase 2) ===\n");
        Class ANEOutSetEnq = NSClassFromString(@"_ANEOutputSetEnqueue");
        id ose = nil;
        @try {
            ose = ((id(*)(id,SEL,unsigned int,unsigned int,unsigned long long,BOOL,BOOL))objc_msgSend)(
                [ANEOutSetEnq alloc],
                @selector(initOutputSetWithProcedureIndex:setIndex:signalValue:signalNotRequired:isOpenLoop:),
                (unsigned int)0, (unsigned int)0, (unsigned long long)1, NO, NO);
            printf("  OutputSetEnqueue: %s\n", ose ? "OK" : "nil");
        } @catch (NSException *ex) {
            printf("  OutputSetEnqueue init threw: %s\n", [[ex reason] UTF8String]);
        }

        if (pfe && ose) {
            @try {
                NSError *err = nil;
                BOOL ok = ((BOOL(*)(id,SEL,id,id,id,NSError**))objc_msgSend)(
                    pfe,
                    @selector(processOutputSet:model:options:error:),
                    ose, k.aneModel, @{}, &err);
                printf("  processOutputSet ret=%d\n", (int)ok);
                if (err) printf("  Error: %s\n", [[err description] UTF8String]);
            } @catch (NSException *ex) {
                printf("  processOutputSet threw: %s\n", [[ex reason] UTF8String]);
            }
        } else {
            printf("  SKIPPED (pfe=%s, ose=%s)\n", pfe ? "yes" : "no", ose ? "yes" : "no");
        }

        // ============================================================
        // Step 7: Check concurrency tracking properties
        // ============================================================
        printf("\n=== Step 7: Concurrency tracking ===\n");
        if (pfe) {
            // queueDepth returns 'c' (char), currentAsyncRequestsInFlight returns 'q' (long long)
            // requestsInFlight returns '@' (dispatch_semaphore)
            @try {
                char qd = ((char(*)(id,SEL))objc_msgSend)(pfe, @selector(queueDepth));
                printf("  [pfe queueDepth] = %d\n", (int)qd);
            } @catch (NSException *ex) {
                printf("  [pfe queueDepth] threw: %s\n", [[ex reason] UTF8String]);
            }

            @try {
                id rif = ((id(*)(id,SEL))objc_msgSend)(pfe, @selector(requestsInFlight));
                printf("  [pfe requestsInFlight] = %s (class: %s)\n",
                       rif ? [[rif description] UTF8String] : "nil",
                       rif ? class_getName([rif class]) : "nil");
            } @catch (NSException *ex) {
                printf("  [pfe requestsInFlight] threw: %s\n", [[ex reason] UTF8String]);
            }

            @try {
                long long caif = ((long long(*)(id,SEL))objc_msgSend)(pfe, @selector(currentAsyncRequestsInFlight));
                printf("  [pfe currentAsyncRequestsInFlight] = %lld\n", caif);
            } @catch (NSException *ex) {
                printf("  [pfe currentAsyncRequestsInFlight] threw: %s\n", [[ex reason] UTF8String]);
            }

            // Check controller
            @try {
                id ctrl = ((id(*)(id,SEL))objc_msgSend)(pfe, @selector(controller));
                printf("  [pfe controller] = %s (class: %s)\n",
                       ctrl ? [[ctrl description] UTF8String] : "nil",
                       ctrl ? class_getName([ctrl class]) : "nil");
                if (ctrl) {
                    dump_methods_with_encodings([ctrl class], class_getName([ctrl class]));
                }
            } @catch (NSException *ex) {
                printf("  [pfe controller] threw: %s\n", [[ex reason] UTF8String]);
            }

            // programHandle and intermediateBufferHandle
            @try {
                uint64_t ph = ((uint64_t(*)(id,SEL))objc_msgSend)(pfe, @selector(programHandle));
                uint64_t ibh = ((uint64_t(*)(id,SEL))objc_msgSend)(pfe, @selector(intermediateBufferHandle));
                printf("  [pfe programHandle] = %llu\n", ph);
                printf("  [pfe intermediateBufferHandle] = %llu\n", ibh);
            } @catch (NSException *ex) {
                printf("  [pfe handles] threw: %s\n", [[ex reason] UTF8String]);
            }
        }

        // Also check on aneModel for comparison
        printf("\n  On aneModel:\n");
        {
            SEL selectors[] = {
                @selector(queueDepth),
                @selector(requestsInFlight),
                @selector(currentAsyncRequestsInFlight),
            };
            const char *names[] = {
                "queueDepth", "requestsInFlight", "currentAsyncRequestsInFlight",
            };
            for (int i = 0; i < 3; i++) {
                @try {
                    if ([k.aneModel respondsToSelector:selectors[i]]) {
                        long long val = ((long long(*)(id,SEL))objc_msgSend)(k.aneModel, selectors[i]);
                        printf("    [aneModel %s] = %lld\n", names[i], val);
                    } else {
                        printf("    [aneModel %s] — not recognized\n", names[i]);
                    }
                } @catch (NSException *ex) {
                    printf("    [aneModel %s] threw: %s\n", names[i], [[ex reason] UTF8String]);
                }
            }
        }

        // ============================================================
        // Step 8: Try setting queueDepth to different values and re-test
        // ============================================================
        printf("\n=== Step 8: Set queueDepth and re-test ===\n");
        // PFE queueDepth is readonly (Tc,R,N). Set via aneModel and check PFE reflects it.
        {
            uint64_t stringId = 0;
            @try { stringId = ((uint64_t(*)(id,SEL))objc_msgSend)(k.aneModel, @selector(string_id)); } @catch(NSException *x) {}

            int depths[] = {1, 2, 4, 8, 127};
            for (int d = 0; d < 5; d++) {
                @try {
                    ((void(*)(id,SEL,char))objc_msgSend)(k.aneModel, @selector(setQueueDepth:), (char)depths[d]);
                    char modelQD = ((char(*)(id,SEL))objc_msgSend)(k.aneModel, @selector(queueDepth));

                    // Re-read PFE from aneModel (might be regenerated)
                    id pfe2 = ((id(*)(id,SEL))objc_msgSend)(k.aneModel, @selector(program));
                    char pfeQD = pfe2 ? ((char(*)(id,SEL))objc_msgSend)(pfe2, @selector(queueDepth)) : -1;

                    printf("  depth=%d: aneModel.qd=%d, pfe.qd=%d\n", depths[d], (int)modelQD, (int)pfeQD);

                    // Eval via doEvaluateDirect
                    NSError *err = nil;
                    BOOL ok = ((BOOL(*)(id,SEL,id,id,id,unsigned int,NSError**))objc_msgSend)(
                        g_client, @selector(doEvaluateDirectWithModel:options:request:qos:error:),
                        k.aneModel, @{}, k.request, (unsigned int)21, &err);
                    printf("    doEvalDirect: %s\n", ok ? "OK" : "FAIL");
                    if (err) printf("    err: %s\n", [[err description] UTF8String]);

                    // Also try processRequest via PFE at this depth
                    if (pfe2) {
                        err = nil;
                        unsigned int retVal = 0;
                        ok = ((BOOL(*)(id,SEL,id,id,unsigned int,uint64_t,uint64_t,id,unsigned int *,NSError**))objc_msgSend)(
                            pfe2,
                            @selector(processRequest:model:qos:qIndex:modelStringID:options:returnValue:error:),
                            k.request, k.aneModel, (unsigned int)21, (uint64_t)0,
                            stringId, @{}, &retVal, &err);
                        printf("    pfe.processRequest: ret=%d retVal=%u\n", (int)ok, retVal);
                        if (err) printf("    err: %s\n", [[err description] UTF8String]);
                    }
                } @catch (NSException *ex) {
                    printf("  depth=%d threw: %s\n", depths[d], [[ex reason] UTF8String]);
                }
            }
        }

        // ============================================================
        // Step 9: Try processSessionHint:options:report:error:
        // ============================================================
        printf("\n=== Step 9: Try processSessionHint ===\n");
        // processSessionHint:options:report:error: encoding: B48@0:8@16@24@32^@40
        // All args are objects: sessionHint(@), options(@), report(@), error(^@)
        if (pfe) {
            SEL sessHintSel = @selector(processSessionHint:options:report:error:);
            if ([pfe respondsToSelector:sessHintSel]) {
                printf("  PFE responds to processSessionHint:options:report:error:\n");

                // Try with nil hint
                @try {
                    NSError *err = nil;
                    BOOL ok = ((BOOL(*)(id,SEL,id,id,id,NSError**))objc_msgSend)(
                        pfe, sessHintSel,
                        nil, @{}, nil, &err);
                    printf("    hint=nil: ret=%d\n", (int)ok);
                    if (err) printf("      err: %s\n", [[err description] UTF8String]);
                } @catch (NSException *ex) {
                    printf("    hint=nil threw: %s\n", [[ex reason] UTF8String]);
                }

                // Try with NSNumber hints
                NSArray *hintVals = @[@0, @1, @2, @3];
                for (NSNumber *hv in hintVals) {
                    @try {
                        NSError *err = nil;
                        BOOL ok = ((BOOL(*)(id,SEL,id,id,id,NSError**))objc_msgSend)(
                            pfe, sessHintSel,
                            hv, @{}, nil, &err);
                        printf("    hint=@%d: ret=%d\n", [hv intValue], (int)ok);
                        if (err) printf("      err: %s\n", [[err description] UTF8String]);
                    } @catch (NSException *ex) {
                        printf("    hint=@%d threw: %s\n", [hv intValue], [[ex reason] UTF8String]);
                    }
                }

                // Try with string hints
                NSArray *strHints = @[@"begin", @"end", @"reset", @"prepare"];
                for (NSString *sh in strHints) {
                    @try {
                        NSError *err = nil;
                        BOOL ok = ((BOOL(*)(id,SEL,id,id,id,NSError**))objc_msgSend)(
                            pfe, sessHintSel,
                            sh, @{}, nil, &err);
                        printf("    hint=@\"%s\": ret=%d\n", [sh UTF8String], (int)ok);
                        if (err) printf("      err: %s\n", [[err description] UTF8String]);
                    } @catch (NSException *ex) {
                        printf("    hint=@\"%s\" threw: %s\n", [sh UTF8String], [[ex reason] UTF8String]);
                    }
                }

                // Try with dict hint
                @try {
                    NSError *err = nil;
                    BOOL ok = ((BOOL(*)(id,SEL,id,id,id,NSError**))objc_msgSend)(
                        pfe, sessHintSel,
                        @{@"type": @"begin"}, @{}, nil, &err);
                    printf("    hint=dict: ret=%d\n", (int)ok);
                    if (err) printf("      err: %s\n", [[err description] UTF8String]);
                } @catch (NSException *ex) {
                    printf("    hint=dict threw: %s\n", [[ex reason] UTF8String]);
                }

                // Check if there's a _ANESessionHint class
                Class sessHintCls = NSClassFromString(@"_ANESessionHint");
                if (sessHintCls) {
                    printf("  _ANESessionHint class exists!\n");
                    dump_methods_with_encodings(sessHintCls, "_ANESessionHint");
                }
            } else {
                printf("  PFE does NOT respond to processSessionHint:options:report:error:\n");
            }
        } else {
            printf("  SKIPPED (no PFE object)\n");
        }

        // ============================================================
        // Extra: Also check _ANEClient for processSessionHint
        // ============================================================
        printf("\n=== Extra: _ANEClient session hint methods ===\n");
        {
            SEL clientSessHints[] = {
                @selector(processSessionHint:model:options:report:error:),
                @selector(processSessionHint:model:options:error:),
                @selector(setSessionHint:forModel:options:error:),
            };
            const char *clientSessHintNames[] = {
                "processSessionHint:model:options:report:error:",
                "processSessionHint:model:options:error:",
                "setSessionHint:forModel:options:error:",
            };
            for (int i = 0; i < 3; i++) {
                if ([g_client respondsToSelector:clientSessHints[i]]) {
                    printf("  _ANEClient responds to %s\n", clientSessHintNames[i]);
                }
            }
        }

        // ============================================================
        // Extra: Check if PFE can be constructed directly
        // ============================================================
        printf("\n=== Extra: Try constructing PFE directly ===\n");
        if (PFECls) {
            // List all init methods
            unsigned int count = 0;
            Method *methods = class_copyMethodList(PFECls, &count);
            printf("  PFE init/alloc methods:\n");
            for (unsigned int i = 0; i < count; i++) {
                const char *sel = sel_getName(method_getName(methods[i]));
                if (strstr(sel, "init") || strstr(sel, "alloc") || strstr(sel, "create") || strstr(sel, "Create")) {
                    const char *enc = method_getTypeEncoding(methods[i]);
                    printf("    %-80s  %s\n", sel, enc ? enc : "(nil)");
                }
            }
            free(methods);

            // Try alloc/init
            @try {
                id newPfe = [[PFECls alloc] init];
                printf("  [[PFE alloc] init] => %s (class: %s)\n",
                       newPfe ? [[newPfe description] UTF8String] : "nil",
                       newPfe ? class_getName([newPfe class]) : "nil");
            } @catch (NSException *ex) {
                printf("  [[PFE alloc] init] threw: %s\n", [[ex reason] UTF8String]);
            }
        }

        // ============================================================
        // Final: Check output to see if any PFE dispatch actually ran
        // ============================================================
        printf("\n=== Final: Check output ===\n");
        {
            _Float16 *out = (_Float16*)IOSurfaceGetBaseAddress(k.ioOut);
            printf("  output[0..3]: %.4f %.4f %.4f %.4f\n",
                   (float)out[0], (float)out[1], (float)out[2], (float)out[3]);
        }

        // Cleanup
        {
            NSError *e = nil;
            ((BOOL(*)(id,SEL,unsigned int,NSError**))objc_msgSend)(k.model, @selector(unloadWithQoS:error:), 21, &e);
        }
        CFRelease(k.ioIn); CFRelease(k.ioOut);

        printf("\n=== Done ===\n");
    }
    return 0;
}
