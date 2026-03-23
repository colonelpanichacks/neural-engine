// test_mil_ops.m — Probe which MIL ops are valid on ANE
#import <Foundation/Foundation.h>
#import <IOSurface/IOSurface.h>
#import <objc/runtime.h>
#import <objc/message.h>
#import <dlfcn.h>

static Class g_D, g_I;

static void ane_init(void) {
    dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine", RTLD_NOW);
    g_D = NSClassFromString(@"_ANEInMemoryModelDescriptor");
    g_I = NSClassFromString(@"_ANEInMemoryModel");
}

static BOOL try_op(const char *op_code, const char *name) {
    NSMutableString *m = [NSMutableString string];
    [m appendString:@"program(1.3)\n"
        "[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}, "
        "{\"coremlc-version\", \"3505.4.1\"}, {\"coremltools-component-milinternal\", \"\"}, "
        "{\"coremltools-version\", \"9.0\"}})]\n{\n"
        "    func main<ios18>(tensor<fp16, [1, 16, 1, 64]> x) {\n"];
    [m appendFormat:@"        %s\n", op_code];
    [m appendString:@"    } -> (y);\n}\n"];

    NSData *mil = [m dataUsingEncoding:NSUTF8StringEncoding];
    id desc = ((id(*)(Class,SEL,id,id,id))objc_msgSend)(g_D,
        @selector(modelWithMILText:weights:optionsPlist:), mil, @{}, nil);
    id model = ((id(*)(Class,SEL,id))objc_msgSend)(g_I,
        @selector(inMemoryModelWithDescriptor:), desc);
    id hx = ((id(*)(id,SEL))objc_msgSend)(model, @selector(hexStringIdentifier));
    NSString *td = [NSTemporaryDirectory() stringByAppendingPathComponent:hx];
    [[NSFileManager defaultManager] createDirectoryAtPath:[td stringByAppendingPathComponent:@"weights"]
        withIntermediateDirectories:YES attributes:nil error:nil];
    [mil writeToFile:[td stringByAppendingPathComponent:@"model.mil"] atomically:YES];

    NSError *e = nil;
    BOOL ok = ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(
        model, @selector(compileWithQoS:options:error:), 21, @{}, &e);
    printf("  %-25s %s\n", name, ok ? "OK" : "FAIL");
    return ok;
}

int main(int argc, char *argv[]) {
    @autoreleasepool {
        setbuf(stdout, NULL);
        ane_init();

        printf("=== MIL Op Probe (16ch, 64sp, fp16) ===\n\n");

        // Unary ops
        try_op("tensor<fp16, [1,16,1,64]> y = mul(x=x, y=x)[name=string(\"y\")];", "mul(x,x)");
        try_op("tensor<fp16, [1,16,1,64]> y = add(x=x, y=x)[name=string(\"y\")];", "add(x,x)");
        try_op("tensor<fp16, [1,16,1,64]> y = sub(x=x, y=x)[name=string(\"y\")];", "sub(x,x)");
        try_op("tensor<fp16, [1,16,1,64]> y = real_div(x=x, y=x)[name=string(\"y\")];", "real_div(x,x)");
        try_op("tensor<fp16, [1,16,1,64]> y = sigmoid(x=x)[name=string(\"y\")];", "sigmoid(x)");
        try_op("tensor<fp16, [1,16,1,64]> y = relu(x=x)[name=string(\"y\")];", "relu(x)");
        try_op("tensor<fp16, [1,16,1,64]> y = tanh(x=x)[name=string(\"y\")];", "tanh(x)");
        try_op("tensor<fp16, [1,16,1,64]> y = rsqrt(x=x)[name=string(\"y\")];", "rsqrt(x)");
        try_op("tensor<fp16, [1,16,1,64]> y = sqrt(x=x)[name=string(\"y\")];", "sqrt(x)");
        try_op("tensor<fp16, [1,16,1,64]> y = exp(x=x)[name=string(\"y\")];", "exp(x)");
        try_op("tensor<fp16, [1,16,1,64]> y = exp2(x=x)[name=string(\"y\")];", "exp2(x)");
        try_op("tensor<fp16, [1,16,1,64]> y = log(x=x)[name=string(\"y\")];", "log(x)");
        try_op("tensor<fp16, [1,16,1,64]> y = abs(x=x)[name=string(\"y\")];", "abs(x)");
        try_op("tensor<fp16, [1,16,1,64]> y = square(x=x)[name=string(\"y\")];", "square(x)");
        try_op("tensor<fp16, [1,16,1,64]> y = inverse(x=x)[name=string(\"y\")];", "inverse(x)");
        try_op("tensor<fp16, [1,16,1,64]> y = floor(x=x)[name=string(\"y\")];", "floor(x)");
        try_op("tensor<fp16, [1,16,1,64]> y = ceil(x=x)[name=string(\"y\")];", "ceil(x)");
        try_op("tensor<fp16, [1,16,1,64]> y = clip(x=x, alpha=fp16(0.0), beta=fp16(6.0))[name=string(\"y\")];", "clip(x,0,6)");
        try_op("tensor<fp16, [1,16,1,64]> y = sign(x=x)[name=string(\"y\")];", "sign(x)");
        try_op("tensor<fp16, [1,16,1,64]> y = round(x=x)[name=string(\"y\")];", "round(x)");
        try_op("tensor<fp16, [1,16,1,64]> y = sin(x=x)[name=string(\"y\")];", "sin(x)");
        try_op("tensor<fp16, [1,16,1,64]> y = cos(x=x)[name=string(\"y\")];", "cos(x)");
        try_op("tensor<fp16, [1,16,1,64]> y = erf(x=x)[name=string(\"y\")];", "erf(x)");

        // Pow
        try_op("fp16 pv = const()[name=string(\"pv\"), val=fp16(-0.5)];\n"
               "        tensor<fp16, [1,16,1,64]> y = pow(x=x, y=pv)[name=string(\"y\")];", "pow(x, -0.5)");
        try_op("fp16 pv = const()[name=string(\"pv\"), val=fp16(2.0)];\n"
               "        tensor<fp16, [1,16,1,64]> y = pow(x=x, y=pv)[name=string(\"y\")];", "pow(x, 2.0)");

        // Reduce ops on different axes
        printf("\n--- Reduce ops ---\n");
        try_op("tensor<int32, [1]> ax = const()[name=string(\"ax\"), val=tensor<int32, [1]>([-1])];\n"
               "        bool kd = const()[name=string(\"kd\"), val=bool(true)];\n"
               "        tensor<fp16, [1,16,1,1]> y = reduce_sum(x=x, axes=ax, keep_dims=kd)[name=string(\"y\")];",
               "reduce_sum(axis=-1)");
        try_op("tensor<int32, [1]> ax = const()[name=string(\"ax\"), val=tensor<int32, [1]>([1])];\n"
               "        bool kd = const()[name=string(\"kd\"), val=bool(true)];\n"
               "        tensor<fp16, [1,1,1,64]> y = reduce_sum(x=x, axes=ax, keep_dims=kd)[name=string(\"y\")];",
               "reduce_sum(axis=1/chan)");
        try_op("tensor<int32, [1]> ax = const()[name=string(\"ax\"), val=tensor<int32, [1]>([-1])];\n"
               "        bool kd = const()[name=string(\"kd\"), val=bool(true)];\n"
               "        tensor<fp16, [1,16,1,1]> y = reduce_mean(x=x, axes=ax, keep_dims=kd)[name=string(\"y\")];",
               "reduce_mean(axis=-1)");
        try_op("tensor<int32, [1]> ax = const()[name=string(\"ax\"), val=tensor<int32, [1]>([1])];\n"
               "        bool kd = const()[name=string(\"kd\"), val=bool(true)];\n"
               "        tensor<fp16, [1,1,1,64]> y = reduce_mean(x=x, axes=ax, keep_dims=kd)[name=string(\"y\")];",
               "reduce_mean(axis=1/chan)");

        // Softmax
        try_op("tensor<int32, [1]> ax = const()[name=string(\"ax\"), val=tensor<int32, [1]>([-1])];\n"
               "        tensor<fp16, [1,16,1,64]> y = softmax(x=x, axis=ax)[name=string(\"y\")];",
               "softmax(axis=-1)");

        // Layer norm / instance norm
        try_op("tensor<fp16, [1,16,1,1]> gam = const()[name=string(\"g\"), val=tensor<fp16, [1,16,1,1]>([1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1])];\n"
               "        tensor<fp16, [1,16,1,1]> bet = const()[name=string(\"b\"), val=tensor<fp16, [1,16,1,1]>([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])];\n"
               "        fp16 eps = const()[name=string(\"eps\"), val=fp16(1e-5)];\n"
               "        tensor<int32, [1]> nax = const()[name=string(\"nax\"), val=tensor<int32, [1]>([-1])];\n"
               "        tensor<fp16, [1,16,1,64]> y = instance_norm(x=x, gamma=gam, beta=bet, epsilon=eps)[name=string(\"y\")];",
               "instance_norm");

        try_op("tensor<fp16, [64]> gam = const()[name=string(\"g\"), val=tensor<fp16, [64]>(["
               "1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,"
               "1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,"
               "1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,"
               "1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1])];\n"
               "        tensor<fp16, [64]> bet = const()[name=string(\"b\"), val=tensor<fp16, [64]>(["
               "0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,"
               "0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,"
               "0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,"
               "0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])];\n"
               "        fp16 eps = const()[name=string(\"eps\"), val=fp16(1e-5)];\n"
               "        tensor<int32, [1]> nax = const()[name=string(\"nax\"), val=tensor<int32, [1]>([-1])];\n"
               "        tensor<fp16, [1,16,1,64]> y = layer_norm(x=x, axes=nax, gamma=gam, beta=bet, epsilon=eps)[name=string(\"y\")];",
               "layer_norm(axis=-1)");

        printf("\n=== Done ===\n");
    }
    return 0;
}
