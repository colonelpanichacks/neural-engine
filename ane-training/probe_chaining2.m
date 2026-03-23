// probe_chaining2.m — Explore _ANEBuffer and chaining with correct input types
#import <Foundation/Foundation.h>
#import <IOSurface/IOSurface.h>
#import <objc/runtime.h>
#import <objc/message.h>
#import <dlfcn.h>
#import <mach/mach_time.h>

static mach_timebase_info_data_t g_tb;
static double ms_t(uint64_t t) { return (double)t * g_tb.numer / g_tb.denom / 1e6; }
static id g_client;
static Class g_D, g_I, g_AR, g_AIO;
static NSMutableArray *g_keep;

static void ane_init(void) {
    dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine", RTLD_NOW);
    g_D  = NSClassFromString(@"_ANEInMemoryModelDescriptor");
    g_I  = NSClassFromString(@"_ANEInMemoryModel");
    g_AR = NSClassFromString(@"_ANERequest");
    g_AIO= NSClassFromString(@"_ANEIOSurfaceObject");
    g_client = ((id(*)(Class,SEL))objc_msgSend)(NSClassFromString(@"_ANEClient"), @selector(sharedConnection));
    g_keep = [NSMutableArray array];
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

static void dump_class(Class cls, const char *label) {
    if (!cls) { printf("  %s: class not found\n", label); return; }
    printf("  %s:\n", label);

    unsigned int count = 0;
    objc_property_t *props = class_copyPropertyList(cls, &count);
    printf("    Properties (%u):\n", count);
    for (unsigned int i = 0; i < count; i++) {
        printf("      .%s [%s]\n", property_getName(props[i]),
               property_getAttributes(props[i]));
    }
    free(props);

    unsigned int mcount = 0;
    Method *methods = class_copyMethodList(object_getClass(cls), &mcount);
    printf("    Class methods (%u):\n", mcount);
    for (unsigned int i = 0; i < mcount; i++) {
        printf("      +%s\n", sel_getName(method_getName(methods[i])));
    }
    free(methods);

    mcount = 0;
    methods = class_copyMethodList(cls, &mcount);
    printf("    Instance methods (%u):\n", mcount);
    for (unsigned int i = 0; i < mcount; i++) {
        printf("      -%s\n", sel_getName(method_getName(methods[i])));
    }
    free(methods);
}

int main(int argc, char *argv[]) {
    @autoreleasepool {
        setbuf(stdout, NULL);
        mach_timebase_info(&g_tb);
        ane_init();

        printf("=== ANE Chaining Probe v2 ===\n\n");

        // ===== 1. Introspect _ANEBuffer =====
        printf("--- _ANEBuffer ---\n");
        dump_class(NSClassFromString(@"_ANEBuffer"), "_ANEBuffer");

        // ===== 2. All ANE buffer/symbol/input/output classes =====
        printf("\n--- ANE buffer/symbol/input/output classes ---\n");
        {
            unsigned int classCount = 0;
            Class *classes = objc_copyClassList(&classCount);
            for (unsigned int i = 0; i < classCount; i++) {
                const char *cn = class_getName(classes[i]);
                if (strstr(cn, "ANE") && (strstr(cn, "uffer") || strstr(cn, "ymbol") ||
                    strstr(cn, "nput") || strstr(cn, "utput") || strstr(cn, "Enqueue"))) {
                    printf("  %s\n", cn);
                }
            }
            free(classes);
        }

        // ===== 3. Compile model and try _ANEBuffer creation =====
        printf("\n--- _ANEBuffer creation test ---\n");
        {
            int IC=256, OC=512, SEQ=64;
            NSData *mil = gen_matmul(IC, OC, SEQ);
            id desc = ((id(*)(Class,SEL,id,id,id))objc_msgSend)(g_D,@selector(modelWithMILText:weights:optionsPlist:),mil,@{},nil);
            id mdl = ((id(*)(Class,SEL,id))objc_msgSend)(g_I,@selector(inMemoryModelWithDescriptor:),desc);
            id hx = ((id(*)(id,SEL))objc_msgSend)(mdl,@selector(hexStringIdentifier));
            NSString *td = [NSTemporaryDirectory() stringByAppendingPathComponent:hx];
            [[NSFileManager defaultManager] createDirectoryAtPath:[td stringByAppendingPathComponent:@"weights"] withIntermediateDirectories:YES attributes:nil error:nil];
            [mil writeToFile:[td stringByAppendingPathComponent:@"model.mil"] atomically:YES];
            NSError *e = nil;
            ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(mdl,@selector(compileWithQoS:options:error:),21,@{},&e);
            ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(mdl,@selector(loadWithQoS:options:error:),21,@{},&e);
            printf("Compiled+loaded OK\n");

            id aneModel = ((id(*)(id,SEL))objc_msgSend)(mdl, @selector(model));
            size_t in_bytes = (size_t)IC*(SEQ+OC)*2;
            size_t out_bytes = (size_t)OC*SEQ*2;
            IOSurfaceRef ioIn = make_surf(in_bytes);
            IOSurfaceRef ioOut = make_surf(out_bytes);

            Class bufClass = NSClassFromString(@"_ANEBuffer");
            if (bufClass) {
                // Correct factory: bufferWithIOSurfaceObject:symbolIndex:source:
                id wI = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_AIO, @selector(objectWithIOSurface:), ioIn);
                id wO = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_AIO, @selector(objectWithIOSurface:), ioOut);
                [g_keep addObject:wI];
                [g_keep addObject:wO];

                // Try source values 0, 1, 2
                for (int src = 0; src < 3; src++) {
                    @try {
                        id buf = ((id(*)(Class,SEL,id,id,long))objc_msgSend)(
                            bufClass, @selector(bufferWithIOSurfaceObject:symbolIndex:source:),
                            wI, @0, (long)src);
                        printf("  bufferWithIOSurfaceObject(symbolIndex=0,source=%d): %s\n", src, buf ? "OK" : "nil");
                        if (buf) {
                            [g_keep addObject:buf];
                            unsigned int count = 0;
                            objc_property_t *props = class_copyPropertyList([buf class], &count);
                            for (unsigned int j = 0; j < count; j++) {
                                const char *name = property_getName(props[j]);
                                @try {
                                    id val = [buf valueForKey:[NSString stringWithUTF8String:name]];
                                    printf("    .%s = %s\n", name, val ? [[val description] UTF8String] : "nil");
                                } @catch (id ex) { printf("    .%s = <err>\n", name); }
                            }
                            free(props);
                        }
                    } @catch (NSException *ex) {
                        printf("  source=%d: exception %s\n", src, [[ex reason] UTF8String]);
                    }
                }

                // Also introspect _ANEInputBuffersReady and _ANEIOSurfaceOutputSets
                printf("\n--- _ANEInputBuffersReady ---\n");
                dump_class(NSClassFromString(@"_ANEInputBuffersReady"), "_ANEInputBuffersReady");
                printf("\n--- _ANEIOSurfaceOutputSets ---\n");
                dump_class(NSClassFromString(@"_ANEIOSurfaceOutputSets"), "_ANEIOSurfaceOutputSets");

                // ===== 4. Try chaining with _ANEBuffer + _ANEIOSurfaceOutputSets =====
                printf("\n--- Chaining with _ANEBuffer ---\n");
                @try {
                    id bufIn = ((id(*)(Class,SEL,id,id,long))objc_msgSend)(
                        bufClass, @selector(bufferWithIOSurfaceObject:symbolIndex:source:),
                        wI, @0, (long)0);
                    id bufOut = ((id(*)(Class,SEL,id,id,long))objc_msgSend)(
                        bufClass, @selector(bufferWithIOSurfaceObject:symbolIndex:source:),
                        wO, @0, (long)0);

                    if (bufIn && bufOut) {
                        [g_keep addObject:bufIn];
                        [g_keep addObject:bufOut];

                        // Create _ANEIOSurfaceOutputSets with output buffer
                        Class ioOutSetsClass = NSClassFromString(@"_ANEIOSurfaceOutputSets");
                        IOSurfaceRef statsSurf = make_surf(4096);  // stats surface
                        id outSets = ((id(*)(Class,SEL,IOSurfaceRef,id))objc_msgSend)(
                            ioOutSetsClass,
                            @selector(objectWithstatsSurRef:outputBuffer:),
                            statsSurf, @[bufOut]);
                        [g_keep addObject:outSets];
                        printf("  IOSurfaceOutputSets: %s\n", outSets ? "OK" : "nil");
                        if (outSets) printf("    desc: %s\n", [[outSets description] UTF8String]);

                        Class chainReqClass = NSClassFromString(@"_ANEChainingRequest");
                        id cr = ((id(*)(Class,SEL,id,id,id,id,id,id,id,id,id))objc_msgSend)(chainReqClass,
                            @selector(chainingRequestWithInputs:outputSets:lbInputSymbolId:lbOutputSymbolId:procedureIndex:signalEvents:transactionHandle:fwEnqueueDelay:memoryPoolId:),
                            @[bufIn], @[outSets], @[], @[], @0, @[], nil, @0, @0);
                        printf("  ChainingRequest: %s\n", cr ? "OK" : "nil");

                        if (cr) {
                            [g_keep addObject:cr];
                            e = nil;
                            BOOL ok = ((BOOL(*)(id,SEL,id,id,id,unsigned int,NSError**))objc_msgSend)(g_client,
                                @selector(doPrepareChainingWithModel:options:chainingReq:qos:error:),
                                aneModel, @{}, cr, 21, &e);
                            printf("  prepareChaining: %d\n", ok);
                            if (!ok && e) printf("    error: %s\n", [[e description] UTF8String]);

                            if (ok) {
                                // Try doEnqueueSetsWithModel with _ANEOutputSetEnqueue
                                Class outSetEnqClass = NSClassFromString(@"_ANEOutputSetEnqueue");
                                id outSetEnq = ((id(*)(Class,SEL,unsigned int,unsigned int,uint64_t,BOOL,BOOL))objc_msgSend)(outSetEnqClass,
                                    @selector(outputSetWithProcedureIndex:setIndex:signalValue:signalNotRequired:isOpenLoop:),
                                    (unsigned int)0, (unsigned int)0, (uint64_t)1, NO, NO);
                                [g_keep addObject:outSetEnq];

                                e = nil;
                                BOOL ok2 = ((BOOL(*)(id,SEL,id,id,id,unsigned int,NSError**))objc_msgSend)(g_client,
                                    @selector(doEnqueueSetsWithModel:outputSet:options:qos:error:),
                                    aneModel, outSetEnq, @{}, 21, &e);
                                printf("  enqueueSets: %d\n", ok2);
                                if (!ok2 && e) printf("    error: %s\n", [[e description] UTF8String]);

                                if (ok2) {
                                    // Verify output
                                    _Float16 *out = (_Float16*)IOSurfaceGetBaseAddress(ioOut);
                                    printf("  output[0..3]: %.4f %.4f %.4f %.4f\n",
                                           (float)out[0], (float)out[1], (float)out[2], (float)out[3]);

                                    // Benchmark
                                    uint64_t t0 = mach_absolute_time();
                                    for (int i = 0; i < 200; i++) {
                                        ((BOOL(*)(id,SEL,id,id,id,unsigned int,NSError**))objc_msgSend)(g_client,
                                            @selector(doEnqueueSetsWithModel:outputSet:options:qos:error:),
                                            aneModel, outSetEnq, @{}, 21, &e);
                                    }
                                    double ms = ms_t(mach_absolute_time() - t0);
                                    printf("  200 enqueues: %.1f ms (%.3f ms/enqueue)\n", ms, ms/200);
                                }
                            }
                        }
                    } else {
                        printf("  Could not create _ANEBuffer objects\n");
                    }
                } @catch (NSException *ex) {
                    printf("  Exception: %s\n", [[ex reason] UTF8String]);
                }
            }

            CFRelease(ioIn);
            CFRelease(ioOut);
        }

        printf("\n=== Done ===\n");
    }
    return 0;
}
