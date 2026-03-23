// test_rmsnorm_fuse2.m — Find the resource limit boundary for RMSNorm+matmul
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

// Generate: RMSNorm(x) → x_norm @ W → y
static NSData *gen_fused(int DIM, int OC, int SEQ) {
    int SP = SEQ + 1 + OC;
    NSMutableString *m = [NSMutableString string];
    [m appendString:@"program(1.3)\n[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"},{\"coremlc-version\", \"3505.4.1\"},{\"coremltools-component-milinternal\", \"\"},{\"coremltools-version\", \"9.0\"}})]\n{\n"];
    [m appendFormat:@"    func main<ios18>(tensor<fp16, [1, %d, 1, %d]> inp) {\n", DIM, SP];

    // Slice x, rms_w, W
    [m appendString:@"        tensor<int32, [4]> bx = const()[name=string(\"bx\"), val=tensor<int32, [4]>([0,0,0,0])];\n"];
    [m appendFormat:@"        tensor<int32, [4]> sx = const()[name=string(\"sx\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n", DIM, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> x = slice_by_size(x=inp,begin=bx,size=sx)[name=string(\"x\")];\n", DIM, SEQ];
    [m appendFormat:@"        tensor<int32, [4]> brw = const()[name=string(\"brw\"), val=tensor<int32, [4]>([0,0,0,%d])];\n", SEQ];
    [m appendFormat:@"        tensor<int32, [4]> srw = const()[name=string(\"srw\"), val=tensor<int32, [4]>([1,%d,1,1])];\n", DIM];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,1]> rw = slice_by_size(x=inp,begin=brw,size=srw)[name=string(\"rw\")];\n", DIM];

    // RMSNorm
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> x2 = mul(x=x, y=x)[name=string(\"x2\")];\n", DIM, SEQ];
    [m appendString:@"        tensor<int32, [1]> ax = const()[name=string(\"ax\"), val=tensor<int32, [1]>([1])];\n"];
    [m appendString:@"        bool kd = const()[name=string(\"kd\"), val=bool(true)];\n"];
    [m appendFormat:@"        tensor<fp16, [1,1,1,%d]> ms = reduce_mean(x=x2, axes=ax, keep_dims=kd)[name=string(\"ms\")];\n", SEQ];
    [m appendString:@"        fp16 epsv = const()[name=string(\"epsv\"), val=fp16(1e-5)];\n"];
    [m appendFormat:@"        tensor<fp16, [1,1,1,%d]> mse = add(x=ms, y=epsv)[name=string(\"mse\")];\n", SEQ];
    [m appendString:@"        fp16 nhalf = const()[name=string(\"nh\"), val=fp16(-0.5)];\n"];
    [m appendFormat:@"        tensor<fp16, [1,1,1,%d]> rrms = pow(x=mse, y=nhalf)[name=string(\"rrms\")];\n", SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> xn = mul(x=x, y=rrms)[name=string(\"xn\")];\n", DIM, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> xnw = mul(x=xn, y=rw)[name=string(\"xnw\")];\n", DIM, SEQ];

    // Matmul
    [m appendFormat:@"        tensor<int32, [4]> bm = const()[name=string(\"bm\"), val=tensor<int32, [4]>([0,0,0,%d])];\n", SEQ+1];
    [m appendFormat:@"        tensor<int32, [4]> sm = const()[name=string(\"sm\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n", DIM, OC];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> W = slice_by_size(x=inp,begin=bm,size=sm)[name=string(\"W\")];\n", DIM, OC];
    [m appendFormat:@"        tensor<int32, [4]> ra = const()[name=string(\"ra\"), val=tensor<int32, [4]>([1,1,%d,%d])];\n", DIM, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> a2 = reshape(shape=ra,x=xnw)[name=string(\"a2\")];\n", DIM, SEQ];
    [m appendString:@"        tensor<int32, [4]> pm = const()[name=string(\"pm\"), val=tensor<int32, [4]>([0,1,3,2])];\n"];
    [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> a3 = transpose(perm=pm,x=a2)[name=string(\"a3\")];\n", SEQ, DIM];
    [m appendFormat:@"        tensor<int32, [4]> rw2 = const()[name=string(\"rw2\"), val=tensor<int32, [4]>([1,1,%d,%d])];\n", DIM, OC];
    [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> Wr = reshape(shape=rw2,x=W)[name=string(\"Wr\")];\n", DIM, OC];
    [m appendString:@"        bool bF = const()[name=string(\"bF\"), val=bool(false)];\n"];
    [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> yh = matmul(transpose_x=bF,transpose_y=bF,x=a3,y=Wr)[name=string(\"yh\")];\n", SEQ, OC];
    [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> yt = transpose(perm=pm,x=yh)[name=string(\"yt\")];\n", OC, SEQ];
    [m appendFormat:@"        tensor<int32, [4]> ro = const()[name=string(\"ro\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n", OC, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> y = reshape(shape=ro,x=yt)[name=string(\"y\")];\n", OC, SEQ];
    [m appendString:@"    } -> (y);\n}\n"];
    return [m dataUsingEncoding:NSUTF8StringEncoding];
}

static int test(int DIM, int OC, int SEQ) {
    printf("DIM=%d OC=%d SEQ=%d: ", DIM, OC, SEQ);
    int SP = SEQ + 1 + OC;
    NSData *mil = gen_fused(DIM, OC, SEQ);
    id desc = ((id(*)(Class,SEL,id,id,id))objc_msgSend)(g_D,@selector(modelWithMILText:weights:optionsPlist:),mil,@{},nil);
    id model = ((id(*)(Class,SEL,id))objc_msgSend)(g_I,@selector(inMemoryModelWithDescriptor:),desc);
    id hx = ((id(*)(id,SEL))objc_msgSend)(model,@selector(hexStringIdentifier));
    NSString *td = [NSTemporaryDirectory() stringByAppendingPathComponent:hx];
    [[NSFileManager defaultManager] createDirectoryAtPath:[td stringByAppendingPathComponent:@"weights"] withIntermediateDirectories:YES attributes:nil error:nil];
    [mil writeToFile:[td stringByAppendingPathComponent:@"model.mil"] atomically:YES];
    NSError *e=nil;
    BOOL ok=((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(model,@selector(compileWithQoS:options:error:),21,@{},&e);
    if (!ok) { printf("COMPILE FAIL\n"); return 0; }
    ok=((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(model,@selector(loadWithQoS:options:error:),21,@{},&e);
    if (!ok) { printf("LOAD FAIL (0x1d)\n"); return 0; }
    IOSurfaceRef ioIn=make_surf((size_t)DIM*SP*2), ioOut=make_surf((size_t)OC*SEQ*2);
    _Float16 *inp=(_Float16*)IOSurfaceGetBaseAddress(ioIn);
    memset(inp, 0, (size_t)DIM*SP*2);
    for (int c=0;c<DIM;c++) for(int s=0;s<SEQ;s++) inp[c*SP+s]=(_Float16)(0.01f*((c+s)%20+1));
    for (int c=0;c<DIM;c++) inp[c*SP+SEQ]=(_Float16)1.0f;
    for (int c=0;c<DIM;c++) for(int o=0;o<OC&&o<DIM;o++) inp[c*SP+SEQ+1+o]=(_Float16)(c==o?0.1f:0.0f);
    id wI=((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_AIO,@selector(objectWithIOSurface:),ioIn);
    id wO=((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_AIO,@selector(objectWithIOSurface:),ioOut);
    id req=((id(*)(Class,SEL,id,id,id,id,id,id,id))objc_msgSend)(g_AR,
        @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:),
        @[wI],@[@0],@[wO],@[@0],nil,nil,@0);
    id aneModel=((id(*)(id,SEL))objc_msgSend)(model,@selector(model));
    ok=((BOOL(*)(id,SEL,id,id,id,NSError**))objc_msgSend)(g_client,
        @selector(evaluateRealTimeWithModel:options:request:error:),aneModel,@{},req,&e);
    if (!ok) { printf("EVAL FAIL (0x1d)\n"); CFRelease(ioIn);CFRelease(ioOut); return 0; }
    int N=100;
    for(int i=0;i<10;i++) ((BOOL(*)(id,SEL,id,id,id,NSError**))objc_msgSend)(g_client,@selector(evaluateRealTimeWithModel:options:request:error:),aneModel,@{},req,&e);
    uint64_t t0=mach_absolute_time();
    for(int i=0;i<N;i++) ((BOOL(*)(id,SEL,id,id,id,NSError**))objc_msgSend)(g_client,@selector(evaluateRealTimeWithModel:options:request:error:),aneModel,@{},req,&e);
    printf("OK! %.3f ms/eval\n", ms_t(mach_absolute_time()-t0)/N);
    CFRelease(ioIn);CFRelease(ioOut);
    return 1;
}

int main(int argc, char *argv[]) {
    @autoreleasepool {
        setbuf(stdout,NULL);
        mach_timebase_info(&g_tb);
        ane_init();
        printf("=== RMSNorm+Matmul Fused Size Sweep ===\n\n");

        // Test various OC sizes with DIM=1024, SEQ=256
        int DIM=1024, SEQ=256;
        test(DIM, 64, SEQ);
        test(DIM, 128, SEQ);
        test(DIM, 256, SEQ);
        test(DIM, 512, SEQ);
        test(DIM, 1024, SEQ);
        test(DIM, 2048, SEQ);

        // Try smaller DIM
        printf("\n--- Smaller DIM ---\n");
        test(512, 1024, SEQ);
        test(512, 2048, SEQ);
        test(256, 2048, SEQ);

        // Try with smaller SEQ
        printf("\n--- Smaller SEQ ---\n");
        test(1024, 2048, 128);
        test(1024, 2048, 64);
        test(1024, 1024, 128);

        printf("\n=== Done ===\n");
    }
    return 0;
}
