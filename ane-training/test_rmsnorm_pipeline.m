// test_rmsnorm_pipeline.m — Test if RMSNorm ANE kernel can pipeline with matmul
// Concurrent submit: RMSNorm layer N+1 while matmul layer N is executing
#import <Foundation/Foundation.h>
#import <IOSurface/IOSurface.h>
#import <objc/runtime.h>
#import <objc/message.h>
#import <dlfcn.h>
#import <mach/mach_time.h>
#import <dispatch/dispatch.h>

static mach_timebase_info_data_t g_tb;
static double ms_t(uint64_t t) { return (double)t * g_tb.numer / g_tb.denom / 1e6; }
static Class g_D, g_I, g_AR, g_AIO;
static id g_client;
static void ane_init(void) {
    dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine", RTLD_NOW);
    g_D=NSClassFromString(@"_ANEInMemoryModelDescriptor");
    g_I=NSClassFromString(@"_ANEInMemoryModel");
    g_AR=NSClassFromString(@"_ANERequest");
    g_AIO=NSClassFromString(@"_ANEIOSurfaceObject");
    g_client=((id(*)(Class,SEL))objc_msgSend)(NSClassFromString(@"_ANEClient"),@selector(sharedConnection));
}
static IOSurfaceRef make_surf(size_t b) {
    return IOSurfaceCreate((__bridge CFDictionaryRef)@{
        (id)kIOSurfaceWidth:@(b),(id)kIOSurfaceHeight:@1,
        (id)kIOSurfaceBytesPerElement:@1,(id)kIOSurfaceBytesPerRow:@(b),
        (id)kIOSurfaceAllocSize:@(b),(id)kIOSurfacePixelFormat:@0});
}

typedef struct { id model, aneModel; IOSurfaceRef ioIn, ioOut; id request; } Kern;

static NSData *gen_rmsnorm(int DIM, int SEQ) {
    NSMutableString *m = [NSMutableString string];
    [m appendString:@"program(1.3)\n[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"},{\"coremlc-version\", \"3505.4.1\"},{\"coremltools-component-milinternal\", \"\"},{\"coremltools-version\", \"9.0\"}})]\n{\n"];
    [m appendFormat:@"    func main<ios18>(tensor<fp16, [1, %d, 1, %d]> x) {\n", DIM, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> x2 = mul(x=x, y=x)[name=string(\"x2\")];\n", DIM, SEQ];
    [m appendString:@"        tensor<int32, [1]> ax = const()[name=string(\"ax\"), val=tensor<int32, [1]>([1])];\n"];
    [m appendString:@"        bool kd = const()[name=string(\"kd\"), val=bool(true)];\n"];
    [m appendFormat:@"        tensor<fp16, [1,1,1,%d]> ms = reduce_mean(x=x2, axes=ax, keep_dims=kd)[name=string(\"ms\")];\n", SEQ];
    [m appendString:@"        fp16 epsv = const()[name=string(\"epsv\"), val=fp16(1e-5)];\n"];
    [m appendFormat:@"        tensor<fp16, [1,1,1,%d]> mse = add(x=ms, y=epsv)[name=string(\"mse\")];\n", SEQ];
    [m appendString:@"        fp16 nhalf = const()[name=string(\"nh\"), val=fp16(-0.5)];\n"];
    [m appendFormat:@"        tensor<fp16, [1,1,1,%d]> rrms = pow(x=mse, y=nhalf)[name=string(\"rrms\")];\n", SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> y = mul(x=x, y=rrms)[name=string(\"y\")];\n", DIM, SEQ];
    [m appendString:@"    } -> (y);\n}\n"];
    return [m dataUsingEncoding:NSUTF8StringEncoding];
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

static Kern compile_kern(NSData *mil, size_t in_bytes, size_t out_bytes) {
    Kern k = {0};
    id desc = ((id(*)(Class,SEL,id,id,id))objc_msgSend)(g_D,@selector(modelWithMILText:weights:optionsPlist:),mil,@{},nil);
    k.model = ((id(*)(Class,SEL,id))objc_msgSend)(g_I,@selector(inMemoryModelWithDescriptor:),desc);
    id hx = ((id(*)(id,SEL))objc_msgSend)(k.model,@selector(hexStringIdentifier));
    NSString *td = [NSTemporaryDirectory() stringByAppendingPathComponent:hx];
    [[NSFileManager defaultManager] createDirectoryAtPath:[td stringByAppendingPathComponent:@"weights"] withIntermediateDirectories:YES attributes:nil error:nil];
    [mil writeToFile:[td stringByAppendingPathComponent:@"model.mil"] atomically:YES];
    NSError *e=nil;
    ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(k.model,@selector(compileWithQoS:options:error:),21,@{},&e);
    ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(k.model,@selector(loadWithQoS:options:error:),21,@{},&e);
    k.ioIn = make_surf(in_bytes); k.ioOut = make_surf(out_bytes);
    _Float16 *inp = (_Float16*)IOSurfaceGetBaseAddress(k.ioIn);
    for (size_t i=0;i<in_bytes/2;i++) inp[i]=(_Float16)(0.01f*(i%100));
    id wI=((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_AIO,@selector(objectWithIOSurface:),k.ioIn);
    id wO=((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_AIO,@selector(objectWithIOSurface:),k.ioOut);
    k.request=((id(*)(Class,SEL,id,id,id,id,id,id,id))objc_msgSend)(g_AR,
        @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:),
        @[wI],@[@0],@[wO],@[@0],nil,nil,@0);
    k.aneModel=((id(*)(id,SEL))objc_msgSend)(k.model,@selector(model));
    return k;
}

static void sync_eval(Kern *k) {
    NSError *e=nil;
    ((BOOL(*)(id,SEL,id,id,id,NSError**))objc_msgSend)(g_client,
        @selector(evaluateRealTimeWithModel:options:request:error:),k->aneModel,@{},k->request,&e);
}

int main(int argc, char *argv[]) {
    @autoreleasepool {
        setbuf(stdout,NULL);
        mach_timebase_info(&g_tb);
        ane_init();

        int DIM=1024, OC=2048, SEQ=256;
        printf("=== RMSNorm+Matmul Pipeline Test ===\n");
        printf("Simulating: RMSNorm(DIM=%d) → matmul(%d→%d) per layer\n\n", DIM, DIM, OC);

        // Compile kernels
        Kern rms = compile_kern(gen_rmsnorm(DIM, SEQ), DIM*SEQ*2, DIM*SEQ*2);
        Kern mm = compile_kern(gen_matmul(DIM, OC, SEQ), DIM*(SEQ+OC)*2, OC*SEQ*2);
        // Second copy for pipelining (separate surfaces)
        Kern rms2 = compile_kern(gen_rmsnorm(DIM, SEQ), DIM*SEQ*2, DIM*SEQ*2);

        int LAYERS = 28, N = 5;

        // Warmup
        for (int i=0;i<20;i++) { sync_eval(&rms); sync_eval(&mm); }

        dispatch_queue_t q1 = dispatch_queue_create("q1",
            dispatch_queue_attr_make_with_qos_class(DISPATCH_QUEUE_SERIAL, QOS_CLASS_USER_INTERACTIVE, 0));
        dispatch_queue_t q2 = dispatch_queue_create("q2",
            dispatch_queue_attr_make_with_qos_class(DISPATCH_QUEUE_SERIAL, QOS_CLASS_USER_INTERACTIVE, 0));

        // Test 1: Sequential (current pattern) — CPU RMSNorm + ANE matmul
        // Simulates 0.09ms CPU RMSNorm + 0.28ms ANE matmul per layer
        printf("Test 1: Sequential RMSNorm(ANE) + Matmul(ANE) per layer\n");
        uint64_t t0 = mach_absolute_time();
        for (int n=0;n<N;n++) {
            for (int L=0;L<LAYERS;L++) {
                sync_eval(&rms);
                sync_eval(&mm);
            }
        }
        double seq_ms = ms_t(mach_absolute_time()-t0)/(N*LAYERS);
        printf("  %.3f ms/layer (%.1f ms/step)\n\n", seq_ms, seq_ms*LAYERS);

        // Test 2: Pipeline — submit next layer's RMSNorm during current layer's matmul
        printf("Test 2: Pipelined — RMSNorm(L+1) overlaps with matmul(L)\n");
        t0 = mach_absolute_time();
        for (int n=0;n<N;n++) {
            // Layer 0: sequential RMSNorm
            sync_eval(&rms);
            for (int L=0;L<LAYERS;L++) {
                // Submit matmul for current layer
                dispatch_semaphore_t sm = dispatch_semaphore_create(0);
                dispatch_async(q1, ^{ sync_eval(&mm); dispatch_semaphore_signal(sm); });
                if (L < LAYERS-1) {
                    // Pipeline: submit next layer's RMSNorm concurrently
                    dispatch_semaphore_t sr = dispatch_semaphore_create(0);
                    dispatch_async(q2, ^{ sync_eval(&rms2); dispatch_semaphore_signal(sr); });
                    dispatch_semaphore_wait(sm, DISPATCH_TIME_FOREVER);
                    dispatch_semaphore_wait(sr, DISPATCH_TIME_FOREVER);
                } else {
                    dispatch_semaphore_wait(sm, DISPATCH_TIME_FOREVER);
                }
            }
        }
        double pipe_ms = ms_t(mach_absolute_time()-t0)/(N*LAYERS);
        printf("  %.3f ms/layer (%.1f ms/step)\n", pipe_ms, pipe_ms*LAYERS);
        printf("  Speedup: %.2fx\n\n", seq_ms/pipe_ms);

        // Test 3: Current approach — CPU RMSNorm (~0.09ms) + ANE matmul
        printf("Test 3: CPU RMSNorm (0.09ms sleep) + ANE matmul\n");
        t0 = mach_absolute_time();
        for (int n=0;n<N;n++) {
            for (int L=0;L<LAYERS;L++) {
                usleep(90);  // simulate 0.09ms CPU RMSNorm
                sync_eval(&mm);
            }
        }
        double cpu_ms = ms_t(mach_absolute_time()-t0)/(N*LAYERS);
        printf("  %.3f ms/layer (%.1f ms/step)\n", cpu_ms, cpu_ms*LAYERS);

        printf("\n  Pipeline benefit: RMSNorm hidden behind matmul = %.3f ms saved/layer\n",
            seq_ms - pipe_ms);

        printf("\n=== Done ===\n");
    }
    return 0;
}
