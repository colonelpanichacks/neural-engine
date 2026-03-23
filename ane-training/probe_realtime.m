// probe_realtime.m — Benchmark evaluateRealTimeWithModel at training scale
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

static void ane_init(void) {
    dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine", RTLD_NOW);
    g_D  = NSClassFromString(@"_ANEInMemoryModelDescriptor");
    g_I  = NSClassFromString(@"_ANEInMemoryModel");
    g_AR = NSClassFromString(@"_ANERequest");
    g_AIO= NSClassFromString(@"_ANEIOSurfaceObject");
    g_client = ((id(*)(Class,SEL))objc_msgSend)(NSClassFromString(@"_ANEClient"), @selector(sharedConnection));
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

        printf("=== evaluateRealTimeWithModel Benchmark ===\n\n");

        // Test at multiple kernel sizes
        typedef struct { int IC, OC, SEQ; const char *name; } Config;
        Config configs[] = {
            {256,  512,  64,  "small (256x512,s64)"},
            {1024, 2048, 256, "training (1024x2048,s256)"},
            {1024, 5504, 256, "ffn-scale (1024x5504,s256)"},
        };
        int nconfigs = sizeof(configs)/sizeof(configs[0]);

        for (int ci = 0; ci < nconfigs; ci++) {
            Config c = configs[ci];
            printf("--- %s ---\n", c.name);

            NSData *mil = gen_matmul(c.IC, c.OC, c.SEQ);
            id desc = ((id(*)(Class,SEL,id,id,id))objc_msgSend)(g_D,@selector(modelWithMILText:weights:optionsPlist:),mil,@{},nil);
            id mdl = ((id(*)(Class,SEL,id))objc_msgSend)(g_I,@selector(inMemoryModelWithDescriptor:),desc);
            id hx = ((id(*)(id,SEL))objc_msgSend)(mdl,@selector(hexStringIdentifier));
            NSString *td = [NSTemporaryDirectory() stringByAppendingPathComponent:hx];
            [[NSFileManager defaultManager] createDirectoryAtPath:[td stringByAppendingPathComponent:@"weights"] withIntermediateDirectories:YES attributes:nil error:nil];
            [mil writeToFile:[td stringByAppendingPathComponent:@"model.mil"] atomically:YES];
            NSError *e = nil;
            ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(mdl,@selector(compileWithQoS:options:error:),21,@{},&e);
            ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(mdl,@selector(loadWithQoS:options:error:),21,@{},&e);

            id aneModel = ((id(*)(id,SEL))objc_msgSend)(mdl, @selector(model));
            size_t in_bytes = (size_t)c.IC*(c.SEQ+c.OC)*2;
            size_t out_bytes = (size_t)c.OC*c.SEQ*2;
            IOSurfaceRef ioIn = make_surf(in_bytes);
            IOSurfaceRef ioOut = make_surf(out_bytes);

            id wI=((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_AIO,@selector(objectWithIOSurface:),ioIn);
            id wO=((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_AIO,@selector(objectWithIOSurface:),ioOut);

            id req = ((id(*)(Class,SEL,id,id,id,id,id,id))objc_msgSend)(g_AR,
                @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:procedureIndex:),
                @[wI],@[@0],@[wO],@[@0],nil,@0);

            int N = 200;

            // Warmup both paths
            for (int i=0; i<50; i++) {
                ((BOOL(*)(id,SEL,id,id,id,unsigned int,NSError**))objc_msgSend)(g_client,
                    @selector(doEvaluateDirectWithModel:options:request:qos:error:),
                    aneModel,@{},req,21,&e);
            }
            for (int i=0; i<50; i++) {
                ((BOOL(*)(id,SEL,id,id,id,NSError**))objc_msgSend)(g_client,
                    @selector(evaluateRealTimeWithModel:options:request:error:),
                    aneModel,@{},req,&e);
            }

            // Benchmark doEvaluateDirectWithModel
            uint64_t t0 = mach_absolute_time();
            for (int i=0; i<N; i++)
                ((BOOL(*)(id,SEL,id,id,id,unsigned int,NSError**))objc_msgSend)(g_client,
                    @selector(doEvaluateDirectWithModel:options:request:qos:error:),
                    aneModel,@{},req,21,&e);
            double ms_direct = ms_t(mach_absolute_time()-t0);

            // Benchmark evaluateRealTimeWithModel
            t0 = mach_absolute_time();
            for (int i=0; i<N; i++)
                ((BOOL(*)(id,SEL,id,id,id,NSError**))objc_msgSend)(g_client,
                    @selector(evaluateRealTimeWithModel:options:request:error:),
                    aneModel,@{},req,&e);
            double ms_rt = ms_t(mach_absolute_time()-t0);

            // Benchmark evaluateWithModel (XPC path)
            t0 = mach_absolute_time();
            for (int i=0; i<N; i++)
                ((BOOL(*)(id,SEL,id,id,id,unsigned int,NSError**))objc_msgSend)(g_client,
                    @selector(evaluateWithModel:options:request:qos:error:),
                    aneModel,@{},req,21,&e);
            double ms_xpc = ms_t(mach_absolute_time()-t0);

            printf("  doEvaluateDirectWithModel:  %.1f ms (%.3f ms/eval)\n", ms_direct, ms_direct/N);
            printf("  evaluateRealTimeWithModel:  %.1f ms (%.3f ms/eval)\n", ms_rt, ms_rt/N);
            printf("  evaluateWithModel (XPC):    %.1f ms (%.3f ms/eval)\n", ms_xpc, ms_xpc/N);
            printf("  Speedup RT vs Direct: %.1f%%\n", (1.0 - ms_rt/ms_direct)*100);
            printf("  Savings per 196 evals: %.1f ms\n\n", (ms_direct/N - ms_rt/N)*196);

            CFRelease(ioIn);
            CFRelease(ioOut);
        }

        // Also look for classes with symbolIndex for chaining
        printf("--- Classes with symbolIndex property ---\n");
        {
            unsigned int classCount = 0;
            Class *classes = objc_copyClassList(&classCount);
            for (unsigned int i = 0; i < classCount; i++) {
                const char *cn = class_getName(classes[i]);
                if (strstr(cn, "ANE") || strstr(cn, "IOSurface")) {
                    unsigned int propCount = 0;
                    objc_property_t *props = class_copyPropertyList(classes[i], &propCount);
                    for (unsigned int j = 0; j < propCount; j++) {
                        if (strstr(property_getName(props[j]), "symbolIndex") ||
                            strstr(property_getName(props[j]), "SymbolIndex")) {
                            printf("  %s.%s\n", cn, property_getName(props[j]));
                        }
                    }
                    free(props);
                }
            }
            free(classes);
        }

        printf("\n=== Done ===\n");
    }
    return 0;
}
