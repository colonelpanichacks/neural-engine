// probe_device_ctrl.m — Deep probe of _ANEDeviceController for direct hardware access
// Explores device struct, privileged connections, IOKit handles, and dispatch methods
#import <Foundation/Foundation.h>
#import <IOSurface/IOSurface.h>
#import <IOKit/IOKitLib.h>
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

// Search all loaded ANE classes for methods matching keywords
static void search_all_ane_methods(void) {
    printf("\n=== GLOBAL SEARCH: ANE methods containing submit/dispatch/fire/trigger/execute/run/kick ===\n");
    const char *keywords[] = {"submit", "dispatch", "fire", "trigger", "execute", "run", "kick",
                              "enqueue", "eval", "invoke", "send", "perform", "launch", "schedule", NULL};
    unsigned int numClasses = 0;
    Class *classes = objc_copyClassList(&numClasses);
    for (unsigned int c = 0; c < numClasses; c++) {
        const char *cname = class_getName(classes[c]);
        if (strstr(cname, "ANE") == NULL && strstr(cname, "ane") == NULL) continue;

        // Instance methods
        unsigned int mcount = 0;
        Method *methods = class_copyMethodList(classes[c], &mcount);
        for (unsigned int m = 0; m < mcount; m++) {
            const char *sel = sel_getName(method_getName(methods[m]));
            for (int k = 0; keywords[k]; k++) {
                if (strcasestr(sel, keywords[k])) {
                    const char *enc = method_getTypeEncoding(methods[m]);
                    printf("  [%s] -%s  %s\n", cname, sel, enc ? enc : "");
                    break;
                }
            }
        }
        free(methods);

        // Class methods
        unsigned int cmcount = 0;
        Method *cmethods = class_copyMethodList(object_getClass(classes[c]), &cmcount);
        for (unsigned int m = 0; m < cmcount; m++) {
            const char *sel = sel_getName(method_getName(cmethods[m]));
            for (int k = 0; keywords[k]; k++) {
                if (strcasestr(sel, keywords[k])) {
                    const char *enc = method_getTypeEncoding(cmethods[m]);
                    printf("  [%s] +%s  %s\n", cname, sel, enc ? enc : "");
                    break;
                }
            }
        }
        free(cmethods);
    }
    free(classes);
}

int main(int argc, char *argv[]) {
    @autoreleasepool {
        setbuf(stdout, NULL);
        mach_timebase_info(&g_tb);
        dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine", RTLD_NOW);
        g_AIO = NSClassFromString(@"_ANEIOSurfaceObject");
        g_AR  = NSClassFromString(@"_ANERequest");
        g_client = ((id(*)(Class,SEL))objc_msgSend)(NSClassFromString(@"_ANEClient"), @selector(sharedConnection));

        printf("=== ANE _ANEDeviceController Deep Probe ===\n\n");

        // ============================================================
        // Step 1: Compile 64x64 matmul
        // ============================================================
        printf("=== Step 1: Compile 64x64 matmul kernel ===\n");
        Kern k = compile(64, 64, 64);
        printf("  aneModel class: %s\n", class_getName([k.aneModel class]));
        printf("  aneModel: %s\n", [[k.aneModel description] UTF8String]);

        // Verify baseline eval works
        {
            NSError *e2 = nil;
            BOOL ok = ((BOOL(*)(id,SEL,id,id,id,unsigned int,NSError**))objc_msgSend)(
                g_client, @selector(doEvaluateDirectWithModel:options:request:qos:error:),
                k.aneModel, @{}, k.request, (unsigned int)21, &e2);
            printf("  Baseline eval: %s%s\n", ok ? "OK" : "FAIL", e2 ? [[e2 description] UTF8String] : "");
        }

        // ============================================================
        // Step 2: Get _ANEProgramForEvaluation via [aneModel program]
        // ============================================================
        printf("\n=== Step 2: Get _ANEProgramForEvaluation ===\n");
        id pfe = nil;
        if ([k.aneModel respondsToSelector:@selector(program)]) {
            pfe = ((id(*)(id,SEL))objc_msgSend)(k.aneModel, @selector(program));
            printf("  [aneModel program] -> %s <%p>\n", class_getName([pfe class]), pfe);
        } else {
            printf("  aneModel does not respond to 'program'\n");
            // Try alternatives
            SEL alts[] = {@selector(programForEvaluation), @selector(evaluationProgram), @selector(aneProgram)};
            for (int i = 0; i < 3; i++) {
                if ([k.aneModel respondsToSelector:alts[i]]) {
                    pfe = ((id(*)(id,SEL))objc_msgSend)(k.aneModel, alts[i]);
                    printf("  [aneModel %s] -> %s <%p>\n", sel_getName(alts[i]),
                           class_getName([pfe class]), pfe);
                    break;
                }
            }
        }
        if (!pfe) {
            printf("  ERROR: Could not get PFE from aneModel\n");
            printf("  Dumping aneModel methods/properties for clues:\n");
            dump_methods([k.aneModel class], class_getName([k.aneModel class]));
            dump_properties([k.aneModel class], class_getName([k.aneModel class]));
        }

        // ============================================================
        // Step 3: Get _ANEDeviceController from [pfe controller]
        // ============================================================
        printf("\n=== Step 3: Get _ANEDeviceController ===\n");
        Class dcClass = NSClassFromString(@"_ANEDeviceController");
        printf("  _ANEDeviceController class exists: %s\n", dcClass ? "YES" : "NO");

        id controller = nil;
        if (pfe) {
            if ([pfe respondsToSelector:@selector(controller)]) {
                controller = ((id(*)(id,SEL))objc_msgSend)(pfe, @selector(controller));
                printf("  [pfe controller] -> %s <%p>\n",
                       controller ? class_getName([controller class]) : "nil", controller);
            } else {
                printf("  pfe does NOT respond to 'controller'\n");
            }
        }

        // ============================================================
        // Step 4: Dump ALL properties, methods, ivars of _ANEDeviceController
        // ============================================================
        printf("\n=== Step 4: Full _ANEDeviceController introspection ===\n");
        if (dcClass) {
            dump_properties(dcClass, "_ANEDeviceController");
            dump_ivars(dcClass, "_ANEDeviceController");
            dump_methods(dcClass, "_ANEDeviceController");
        } else {
            printf("  _ANEDeviceController class not found!\n");
        }

        // Also dump _ANEProgramForEvaluation
        if (pfe) {
            printf("\n=== _ANEProgramForEvaluation introspection ===\n");
            dump_properties([pfe class], "_ANEProgramForEvaluation");
            dump_ivars([pfe class], "_ANEProgramForEvaluation");
            dump_methods([pfe class], "_ANEProgramForEvaluation");
        }

        // ============================================================
        // Step 5: Try sharedPrivilegedConnection
        // ============================================================
        printf("\n=== Step 5: sharedPrivilegedConnection ===\n");
        if (dcClass) {
            if ([dcClass respondsToSelector:@selector(sharedPrivilegedConnection)]) {
                @try {
                    id priv = ((id(*)(Class,SEL))objc_msgSend)(dcClass, @selector(sharedPrivilegedConnection));
                    printf("  +sharedPrivilegedConnection -> %s <%p>\n",
                           priv ? class_getName([priv class]) : "nil", priv);
                    if (priv && controller) {
                        printf("  Same as pfe.controller? %s\n", priv == controller ? "YES" : "NO");
                    }
                    // Check isPrivileged on it
                    if (priv && [priv respondsToSelector:@selector(isPrivileged)]) {
                        BOOL ip = ((BOOL(*)(id,SEL))objc_msgSend)(priv, @selector(isPrivileged));
                        printf("  priv.isPrivileged = %d\n", ip);
                    }
                } @catch (NSException *ex) {
                    printf("  sharedPrivilegedConnection threw: %s\n", [[ex description] UTF8String]);
                }
            } else {
                printf("  _ANEDeviceController does NOT respond to +sharedPrivilegedConnection\n");
            }
        }

        // ============================================================
        // Step 6: Check device property — read raw struct pointer
        // ============================================================
        printf("\n=== Step 6: device property (raw ANEDeviceStruct*) ===\n");
        if (controller) {
            if ([controller respondsToSelector:@selector(device)]) {
                void *devicePtr = ((void*(*)(id,SEL))objc_msgSend)(controller, @selector(device));
                printf("  [controller device] -> %p\n", devicePtr);
                if (devicePtr) {
                    printf("  Attempting to read first 16 words of device struct:\n");
                    @try {
                        uint64_t *words = (uint64_t *)devicePtr;
                        for (int i = 0; i < 16; i++) {
                            printf("    [%2d] 0x%016llx\n", i, words[i]);
                        }
                    } @catch (NSException *ex) {
                        printf("  Exception reading device struct: %s\n", [[ex description] UTF8String]);
                    }
                }
            } else {
                printf("  controller does NOT respond to 'device'\n");
            }

            // Also check other interesting properties
            const char *propNames[] = {"programHandle", "usecount", "isPrivileged",
                                        "connection", "port", "ioConnection", NULL};
            for (int i = 0; propNames[i]; i++) {
                SEL s = sel_registerName(propNames[i]);
                if ([controller respondsToSelector:s]) {
                    // Use long long to capture any integer-like return
                    long long val = ((long long(*)(id,SEL))objc_msgSend)(controller, s);
                    printf("  controller.%s = 0x%llx (%lld)\n", propNames[i], val, val);
                } else {
                    printf("  controller does NOT respond to '%s'\n", propNames[i]);
                }
            }
        } else {
            printf("  No controller instance available\n");

            // Try getting one from the class directly
            if (dcClass) {
                printf("  Trying class-level accessors...\n");
                SEL classAccessors[] = {
                    @selector(sharedController), @selector(defaultController),
                    @selector(sharedConnection), @selector(new)
                };
                const char *classAccessorNames[] = {"sharedController", "defaultController", "sharedConnection", "new"};
                for (int i = 0; i < 4; i++) {
                    if ([dcClass respondsToSelector:classAccessors[i]]) {
                        @try {
                            id obj = ((id(*)(Class,SEL))objc_msgSend)(dcClass, classAccessors[i]);
                            printf("    +%s -> %s <%p>\n", classAccessorNames[i],
                                   obj ? class_getName([obj class]) : "nil", obj);
                            if (obj) controller = obj; // Use first one we find
                        } @catch (NSException *ex) {
                            printf("    +%s threw: %s\n", classAccessorNames[i], [[ex description] UTF8String]);
                        }
                    }
                }
            }
        }

        // ============================================================
        // Step 7: Create NEW controller via initWithProgramHandle:priviledged:YES
        // ============================================================
        printf("\n=== Step 7: Create new _ANEDeviceController ===\n");
        if (dcClass) {
            // Get programHandle from aneModel
            long long programHandle = 0;
            if ([k.aneModel respondsToSelector:@selector(programHandle)]) {
                programHandle = ((long long(*)(id,SEL))objc_msgSend)(k.aneModel, @selector(programHandle));
                printf("  aneModel.programHandle = 0x%llx (%lld)\n", programHandle, programHandle);
            } else {
                printf("  aneModel does not respond to programHandle\n");
            }

            // Try initWithProgramHandle:priviledged: (note: typo in API — "priviledged" not "privileged")
            SEL initSels[] = {
                @selector(initWithProgramHandle:priviledged:),
                @selector(initWithProgramHandle:privileged:),
                @selector(initWithProgramHandle:),
            };
            const char *initNames[] = {
                "initWithProgramHandle:priviledged:",
                "initWithProgramHandle:privileged:",
                "initWithProgramHandle:",
            };
            for (int i = 0; i < 3; i++) {
                if (class_getInstanceMethod(dcClass, initSels[i])) {
                    printf("  Found init method: %s\n", initNames[i]);
                    @try {
                        id newCtrl = [dcClass alloc];
                        id result = nil;
                        if (i < 2) {
                            // Two-arg init
                            result = ((id(*)(id,SEL,long long,BOOL))objc_msgSend)(
                                newCtrl, initSels[i], programHandle, YES);
                        } else {
                            result = ((id(*)(id,SEL,long long))objc_msgSend)(
                                newCtrl, initSels[i], programHandle);
                        }
                        printf("    result -> %s <%p>\n",
                               result ? class_getName([result class]) : "nil", result);
                        if (result) {
                            if ([result respondsToSelector:@selector(isPrivileged)]) {
                                BOOL ip = ((BOOL(*)(id,SEL))objc_msgSend)(result, @selector(isPrivileged));
                                printf("    isPrivileged = %d\n", ip);
                            }
                            if ([result respondsToSelector:@selector(device)]) {
                                void *dp = ((void*(*)(id,SEL))objc_msgSend)(result, @selector(device));
                                printf("    device = %p\n", dp);
                            }
                        }
                    } @catch (NSException *ex) {
                        printf("    init threw: %s\n", [[ex description] UTF8String]);
                    }
                } else {
                    printf("  No method: %s\n", initNames[i]);
                }
            }
        }

        // ============================================================
        // Step 8: Check for submit/enqueue/dispatch methods on controller
        // ============================================================
        printf("\n=== Step 8: submit/enqueue/dispatch methods on controller ===\n");
        if (dcClass) {
            SEL dispatchSels[] = {
                @selector(submitRequest:), @selector(enqueue:), @selector(dispatch:),
                @selector(evaluateWithRequest:), @selector(executeRequest:),
                @selector(runRequest:), @selector(submitRequest:error:),
                @selector(evaluateWithRequest:error:),
                @selector(doEvaluateWithRequest:error:),
                @selector(evaluateDirectWithModel:request:error:),
                @selector(start), @selector(stop),
                @selector(startWithCompletionHandler:), @selector(stopWithCompletionHandler:),
            };
            const char *dispatchNames[] = {
                "submitRequest:", "enqueue:", "dispatch:",
                "evaluateWithRequest:", "executeRequest:",
                "runRequest:", "submitRequest:error:",
                "evaluateWithRequest:error:",
                "doEvaluateWithRequest:error:",
                "evaluateDirectWithModel:request:error:",
                "start", "stop",
                "startWithCompletionHandler:", "stopWithCompletionHandler:",
            };
            for (int i = 0; i < 14; i++) {
                Method m = class_getInstanceMethod(dcClass, dispatchSels[i]);
                if (m) {
                    const char *enc = method_getTypeEncoding(m);
                    printf("  HAS -%s  %s\n", dispatchNames[i], enc ? enc : "");
                } else {
                    printf("  NO  -%s\n", dispatchNames[i]);
                }
            }
        }

        // ============================================================
        // Step 9: Try controllerWithProgramHandle:
        // ============================================================
        printf("\n=== Step 9: controllerWithProgramHandle: ===\n");
        if (dcClass) {
            long long ph = 0;
            if ([k.aneModel respondsToSelector:@selector(programHandle)]) {
                ph = ((long long(*)(id,SEL))objc_msgSend)(k.aneModel, @selector(programHandle));
            }

            SEL factorySels[] = {
                @selector(controllerWithProgramHandle:),
                @selector(controllerWithProgramHandle:priviledged:),
                @selector(controllerForProgram:),
                @selector(controllerForModel:),
            };
            const char *factoryNames[] = {
                "controllerWithProgramHandle:",
                "controllerWithProgramHandle:priviledged:",
                "controllerForProgram:",
                "controllerForModel:",
            };
            for (int i = 0; i < 4; i++) {
                if ([dcClass respondsToSelector:factorySels[i]]) {
                    @try {
                        id r = nil;
                        if (i == 1) {
                            r = ((id(*)(Class,SEL,long long,BOOL))objc_msgSend)(dcClass, factorySels[i], ph, YES);
                        } else if (i >= 2) {
                            r = ((id(*)(Class,SEL,id))objc_msgSend)(dcClass, factorySels[i], k.aneModel);
                        } else {
                            r = ((id(*)(Class,SEL,long long))objc_msgSend)(dcClass, factorySels[i], ph);
                        }
                        printf("  +%s -> %s <%p>\n", factoryNames[i],
                               r ? class_getName([r class]) : "nil", r);
                    } @catch (NSException *ex) {
                        printf("  +%s threw: %s\n", factoryNames[i], [[ex description] UTF8String]);
                    }
                } else {
                    printf("  NO +%s\n", factoryNames[i]);
                }
            }
        }

        // ============================================================
        // Step 10: Check for _ANEDevice class and IOKit service connections
        // ============================================================
        printf("\n=== Step 10: _ANEDevice class and IOKit services ===\n");
        Class aneDevice = NSClassFromString(@"_ANEDevice");
        printf("  _ANEDevice class: %s\n", aneDevice ? "EXISTS" : "not found");
        if (aneDevice) {
            dump_properties(aneDevice, "_ANEDevice");
            dump_ivars(aneDevice, "_ANEDevice");
            dump_methods(aneDevice, "_ANEDevice");
        }

        // Check for other device-like classes
        const char *deviceClasses[] = {
            "_ANEDeviceInfo", "_ANEHardware", "_ANEHWDevice",
            "_ANEDeviceManager", "_ANEDeviceProxy",
            "_ANEService", "_ANEServiceClient",
            "_ANEDeviceConfig", "_ANEDeviceState",
            NULL
        };
        for (int i = 0; deviceClasses[i]; i++) {
            Class c = NSClassFromString([NSString stringWithUTF8String:deviceClasses[i]]);
            if (c) {
                printf("\n  Found: %s\n", deviceClasses[i]);
                dump_properties(c, deviceClasses[i]);
                dump_methods(c, deviceClasses[i]);
            }
        }

        // Check IOKit for ANE service
        printf("\n  --- IOKit ANE service search ---\n");
        io_iterator_t iter;
        kern_return_t kr;

        // Try common ANE service names
        const char *svcNames[] = {
            "AppleH16ANE", "AppleH13ANE", "AppleH11ANE",
            "AppleANE", "AppleNeuralEngine", "H16ANEIn",
            "AppleH16ANEInterface", "AppleANEInterface",
            NULL
        };
        for (int i = 0; svcNames[i]; i++) {
            io_service_t svc = IOServiceGetMatchingService(kIOMainPortDefault,
                IOServiceMatching(svcNames[i]));
            if (svc) {
                printf("  Found IOKit service: %s (port=%u)\n", svcNames[i], svc);
                // Try to get properties
                CFMutableDictionaryRef props = NULL;
                kr = IORegistryEntryCreateCFProperties(svc, &props, kCFAllocatorDefault, 0);
                if (kr == KERN_SUCCESS && props) {
                    NSString *desc = [(__bridge NSDictionary *)props description];
                    // Print first 2000 chars
                    if ([desc length] > 2000) desc = [desc substringToIndex:2000];
                    printf("  Properties: %s...\n", [desc UTF8String]);
                    CFRelease(props);
                }
                IOObjectRelease(svc);
            }
        }

        // Broader search: iterate ANE-related services
        kr = IOServiceGetMatchingServices(kIOMainPortDefault,
            IOServiceMatching("IOService"), &iter);
        if (kr == KERN_SUCCESS) {
            io_service_t svc;
            printf("\n  IOKit services containing 'ANE' or 'NeuralEngine':\n");
            while ((svc = IOIteratorNext(iter))) {
                io_name_t name;
                IORegistryEntryGetName(svc, name);
                if (strcasestr(name, "ANE") || strcasestr(name, "Neural")) {
                    io_name_t className;
                    IOObjectGetClass(svc, className);
                    printf("    %s (class: %s, port=%u)\n", name, className, svc);
                }
                IOObjectRelease(svc);
            }
            IOObjectRelease(iter);
        }

        // Check if controller has any io_connect_t ivar
        if (controller) {
            printf("\n  Checking controller for IOKit connection handles...\n");
            unsigned int icount = 0;
            Ivar *ivars = class_copyIvarList([controller class], &icount);
            for (unsigned int i = 0; i < icount; i++) {
                const char *iname = ivar_getName(ivars[i]);
                const char *itype = ivar_getTypeEncoding(ivars[i]);
                if (iname && (strstr(iname, "connect") || strstr(iname, "port") ||
                              strstr(iname, "service") || strstr(iname, "io") ||
                              strstr(iname, "mach"))) {
                    ptrdiff_t off = ivar_getOffset(ivars[i]);
                    void *base = (__bridge void *)controller;
                    uint64_t val = *(uint64_t *)((uint8_t *)base + off);
                    printf("    %s (%s) at offset %td = 0x%llx\n", iname, itype ? itype : "?", off, val);
                }
            }
            free(ivars);
        }

        // ============================================================
        // Step 11: Global search for dispatch/submit/execute methods
        // ============================================================
        search_all_ane_methods();

        // ============================================================
        // Step 12: Check _ANEClient for IOKit connection info
        // ============================================================
        printf("\n=== Step 12: _ANEClient connection details ===\n");
        if (g_client) {
            printf("  g_client class: %s <%p>\n", class_getName([g_client class]), g_client);
            dump_ivars([g_client class], "_ANEClient");

            // Read IOKit-related ivars
            unsigned int icount = 0;
            Ivar *ivars = class_copyIvarList([g_client class], &icount);
            for (unsigned int i = 0; i < icount; i++) {
                const char *iname = ivar_getName(ivars[i]);
                const char *itype = ivar_getTypeEncoding(ivars[i]);
                if (iname && (strstr(iname, "connect") || strstr(iname, "port") ||
                              strstr(iname, "service") || strstr(iname, "xpc") ||
                              strstr(iname, "mach") || strstr(iname, "io_"))) {
                    ptrdiff_t off = ivar_getOffset(ivars[i]);
                    void *base = (__bridge void *)g_client;
                    uint64_t val = *(uint64_t *)((uint8_t *)base + off);
                    printf("    %s (%s) at offset %td = 0x%llx\n", iname, itype ? itype : "?", off, val);
                }
            }
            free(ivars);
        }

        printf("\n=== DONE ===\n");
    }
    return 0;
}
