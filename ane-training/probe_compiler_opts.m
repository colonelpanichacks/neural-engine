// probe_compiler_opts.m — Explore ANE compiler options for enabling chaining
// Findings from probe_coreml_chain:
//   - MLModelConfiguration.neuralEngineCompilerOptions (private)
//   - MLModelConfiguration.e5rtCustomANECompilerOptions (private)
//   - _ANEInMemoryModel.compilerOptionsWithOptions:isCompiledModelCached: (returns dict)
//   - _ANEInMemoryModel.compilerOptionsFileName (returns string)
#import <Foundation/Foundation.h>
#import <CoreML/CoreML.h>
#import <IOSurface/IOSurface.h>
#import <objc/runtime.h>
#import <objc/message.h>
#import <dlfcn.h>

// ── helpers ──────────────────────────────────────────────────────────────────

static IOSurfaceRef make_surf(size_t bytes) {
    return IOSurfaceCreate((__bridge CFDictionaryRef)@{
        (id)kIOSurfaceWidth:@(bytes), (id)kIOSurfaceHeight:@1,
        (id)kIOSurfaceBytesPerElement:@1, (id)kIOSurfaceBytesPerRow:@(bytes),
        (id)kIOSurfaceAllocSize:@(bytes), (id)kIOSurfacePixelFormat:@0});
}

// Simple matmul MIL: y = x @ W (from probe_chaining4)
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

// Deep-print a dictionary (recursive for nested dicts/arrays)
static void print_dict(NSDictionary *d, int indent) {
    NSArray *keys = [[d allKeys] sortedArrayUsingSelector:@selector(compare:)];
    for (NSString *key in keys) {
        id val = d[key];
        for (int i = 0; i < indent; i++) printf("  ");
        if ([val isKindOfClass:[NSDictionary class]]) {
            printf("%s: {\n", [key UTF8String]);
            print_dict(val, indent + 1);
            for (int i = 0; i < indent; i++) printf("  ");
            printf("}\n");
        } else if ([val isKindOfClass:[NSArray class]]) {
            printf("%s: [\n", [key UTF8String]);
            for (id item in val) {
                for (int i = 0; i < indent + 1; i++) printf("  ");
                printf("%s\n", [[item description] UTF8String]);
            }
            for (int i = 0; i < indent; i++) printf("  ");
            printf("]\n");
        } else if ([val isKindOfClass:[NSData class]]) {
            printf("%s: <NSData, %lu bytes>\n", [key UTF8String], (unsigned long)[(NSData *)val length]);
        } else {
            printf("%s: %s  (%s)\n", [key UTF8String],
                   [[val description] UTF8String], class_getName([val class]));
        }
    }
}

// ── compile + load a model, return the _ANEInMemoryModel ──

static id compile_model(int ic, int oc, int seq, NSDictionary *compileOpts) {
    Class D = NSClassFromString(@"_ANEInMemoryModelDescriptor");
    Class I = NSClassFromString(@"_ANEInMemoryModel");
    NSData *mil = gen_mil(ic, oc, seq);

    id desc = ((id(*)(Class,SEL,id,id,id))objc_msgSend)(D,
        @selector(modelWithMILText:weights:optionsPlist:), mil, @{}, nil);
    id model = ((id(*)(Class,SEL,id))objc_msgSend)(I,
        @selector(inMemoryModelWithDescriptor:), desc);

    // Write MIL to temp dir so compiler can find it
    id hx = ((id(*)(id,SEL))objc_msgSend)(model, @selector(hexStringIdentifier));
    NSString *td = [NSTemporaryDirectory() stringByAppendingPathComponent:hx];
    [[NSFileManager defaultManager] createDirectoryAtPath:[td stringByAppendingPathComponent:@"weights"]
        withIntermediateDirectories:YES attributes:nil error:nil];
    [mil writeToFile:[td stringByAppendingPathComponent:@"model.mil"] atomically:YES];

    NSError *e = nil;
    BOOL ok = ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(
        model, @selector(compileWithQoS:options:error:), 21, compileOpts ? compileOpts : @{}, &e);
    if (!ok) {
        printf("    compile FAILED: %s\n", e ? [[e description] UTF8String] : "unknown");
        return nil;
    }
    e = nil;
    ok = ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(
        model, @selector(loadWithQoS:options:error:), 21, @{}, &e);
    if (!ok) {
        printf("    load FAILED: %s\n", e ? [[e description] UTF8String] : "unknown");
        return nil;
    }
    return model;
}

// ── try chaining on a model ──

static void try_chaining(id model, const char *label) {
    Class AIO = NSClassFromString(@"_ANEIOSurfaceObject");
    Class AR  = NSClassFromString(@"_ANERequest");
    Class ANEBuf = NSClassFromString(@"_ANEBuffer");
    Class ANEOutSets = NSClassFromString(@"_ANEIOSurfaceOutputSets");
    Class ANEChainReq = NSClassFromString(@"_ANEChainingRequest");

    id client = ((id(*)(Class,SEL))objc_msgSend)(NSClassFromString(@"_ANEClient"), @selector(sharedConnection));
    id aneModel = ((id(*)(id,SEL))objc_msgSend)(model, @selector(model));

    IOSurfaceRef ioIn = make_surf(64*96*2);
    IOSurfaceRef ioOut = make_surf(64*32*2);
    IOSurfaceRef statsSurf = make_surf(4096);

    id wIn  = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(AIO, @selector(objectWithIOSurface:), ioIn);
    id wOut = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(AIO, @selector(objectWithIOSurface:), ioOut);

    id bufIn  = ((id(*)(Class,SEL,id,id,long long))objc_msgSend)(ANEBuf,
        @selector(bufferWithIOSurfaceObject:symbolIndex:source:), wIn, @0, (long long)0);
    id bufOut = ((id(*)(Class,SEL,id,id,long long))objc_msgSend)(ANEBuf,
        @selector(bufferWithIOSurfaceObject:symbolIndex:source:), wOut, @0, (long long)1);

    id outSets = ((id(*)(Class,SEL,IOSurfaceRef,id))objc_msgSend)(ANEOutSets,
        @selector(objectWithstatsSurRef:outputBuffer:), statsSurf, @[bufOut]);

    @try {
        id chainReq = ((id(*)(id,SEL,id,id,id,id,id,id,id,id,id))objc_msgSend)(
            [ANEChainReq alloc],
            @selector(initWithInputs:outputs:lbInputSymbolId:lbOutputSymbolId:procedureIndex:signalEvents:transactionHandle:fwEnqueueDelay:memoryPoolId:),
            @[bufIn], @[outSets], @[@(-1)], @[@(-1)], @0, @[], @1, @0, @0);

        NSError *e = nil;
        BOOL ok = ((BOOL(*)(id,SEL,id,id,id,unsigned int,NSError**))objc_msgSend)(
            client, @selector(doPrepareChainingWithModel:options:chainingReq:qos:error:),
            aneModel, @{}, chainReq, (unsigned int)21, &e);
        printf("    [%s] prepareChaining: %s", label, ok ? "SUCCESS" : "FAIL");
        if (e) printf(" err=%ld", (long)[e code]);
        printf("\n");
    } @catch (NSException *ex) {
        printf("    [%s] prepareChaining EXCEPTION: %s\n", label, [[ex reason] UTF8String]);
    }

    CFRelease(ioIn); CFRelease(ioOut); CFRelease(statsSurf);
}

// ── main ─────────────────────────────────────────────────────────────────────

int main(int argc, char *argv[]) {
    @autoreleasepool {
        setbuf(stdout, NULL);

        dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine", RTLD_NOW);

        printf("=== ANE Compiler Options Probe ===\n\n");

        // ════════════════════════════════════════════════════════════════
        // 1. Get default compiler options
        // ════════════════════════════════════════════════════════════════
        printf("═══ 1. Default Compiler Options ═══\n\n");

        id baseModel = compile_model(64, 64, 32, @{});
        if (!baseModel) {
            printf("FATAL: base model compile failed\n");
            return 1;
        }

        // compilerOptionsWithOptions:isCompiledModelCached:
        @try {
            id opts = ((id(*)(id,SEL,id,BOOL))objc_msgSend)(baseModel,
                @selector(compilerOptionsWithOptions:isCompiledModelCached:), @{}, NO);
            printf("compilerOptionsWithOptions:@{} isCompiledModelCached:NO:\n");
            if ([opts isKindOfClass:[NSDictionary class]]) {
                print_dict(opts, 1);
            } else {
                printf("  returned: %s (%s)\n", [[opts description] UTF8String],
                       opts ? class_getName([opts class]) : "nil");
            }
        } @catch (NSException *ex) {
            printf("  EXCEPTION: %s\n", [[ex reason] UTF8String]);
        }

        // Try with cached=YES
        printf("\ncompilerOptionsWithOptions:@{} isCompiledModelCached:YES:\n");
        @try {
            id opts2 = ((id(*)(id,SEL,id,BOOL))objc_msgSend)(baseModel,
                @selector(compilerOptionsWithOptions:isCompiledModelCached:), @{}, YES);
            if ([opts2 isKindOfClass:[NSDictionary class]]) {
                print_dict(opts2, 1);
            } else {
                printf("  returned: %s (%s)\n", [[opts2 description] UTF8String],
                       opts2 ? class_getName([opts2 class]) : "nil");
            }
        } @catch (NSException *ex) {
            printf("  EXCEPTION: %s\n", [[ex reason] UTF8String]);
        }

        // Try passing some options to see if they get merged
        printf("\ncompilerOptionsWithOptions:@{@\"enableChaining\":@YES} isCompiledModelCached:NO:\n");
        @try {
            id opts3 = ((id(*)(id,SEL,id,BOOL))objc_msgSend)(baseModel,
                @selector(compilerOptionsWithOptions:isCompiledModelCached:),
                @{@"enableChaining":@YES}, NO);
            if ([opts3 isKindOfClass:[NSDictionary class]]) {
                print_dict(opts3, 1);
            } else {
                printf("  returned: %s (%s)\n", [[opts3 description] UTF8String],
                       opts3 ? class_getName([opts3 class]) : "nil");
            }
        } @catch (NSException *ex) {
            printf("  EXCEPTION: %s\n", [[ex reason] UTF8String]);
        }

        // ════════════════════════════════════════════════════════════════
        // 2. compilerOptionsFileName
        // ════════════════════════════════════════════════════════════════
        printf("\n═══ 2. compilerOptionsFileName ═══\n\n");

        @try {
            id fname = ((id(*)(id,SEL))objc_msgSend)(baseModel,
                @selector(compilerOptionsFileName));
            printf("compilerOptionsFileName: %s (%s)\n",
                   fname ? [[fname description] UTF8String] : "(nil)",
                   fname ? class_getName([fname class]) : "nil");
        } @catch (NSException *ex) {
            printf("  EXCEPTION: %s\n", [[ex reason] UTF8String]);
        }

        // Try setting it
        @try {
            ((void(*)(id,SEL,id))objc_msgSend)(baseModel,
                @selector(setCompilerOptionsFileName:), @"custom_opts.plist");
            id fname2 = ((id(*)(id,SEL))objc_msgSend)(baseModel,
                @selector(compilerOptionsFileName));
            printf("After setCompilerOptionsFileName: %s\n",
                   fname2 ? [[fname2 description] UTF8String] : "(nil)");
        } @catch (NSException *ex) {
            printf("  set EXCEPTION: %s\n", [[ex reason] UTF8String]);
        }

        // ════════════════════════════════════════════════════════════════
        // 3. Read compiler options plist from temp dir
        // ════════════════════════════════════════════════════════════════
        printf("\n═══ 3. Compiler Options Plist from Temp Dir ═══\n\n");

        id hx = ((id(*)(id,SEL))objc_msgSend)(baseModel, @selector(hexStringIdentifier));
        NSString *td = [NSTemporaryDirectory() stringByAppendingPathComponent:hx];
        printf("Model temp dir: %s\n", [td UTF8String]);

        // List all files in temp dir
        NSFileManager *fm = [NSFileManager defaultManager];
        NSError *err = nil;
        NSArray *files = [fm subpathsOfDirectoryAtPath:td error:&err];
        if (files) {
            printf("Files in temp dir:\n");
            for (NSString *f in files) {
                NSDictionary *attrs = [fm attributesOfItemAtPath:[td stringByAppendingPathComponent:f] error:nil];
                unsigned long long sz = [attrs[NSFileSize] unsignedLongLongValue];
                printf("  %s (%llu bytes)\n", [f UTF8String], sz);
            }
        } else {
            printf("Could not list temp dir: %s\n", [[err description] UTF8String]);
        }

        // Look for plist files
        for (NSString *f in files) {
            if ([f hasSuffix:@".plist"] || [f containsString:@"compiler"] || [f containsString:@"option"]) {
                NSString *fullPath = [td stringByAppendingPathComponent:f];
                printf("\nReading %s:\n", [f UTF8String]);
                NSDictionary *plist = [NSDictionary dictionaryWithContentsOfFile:fullPath];
                if (plist) {
                    print_dict(plist, 1);
                } else {
                    // Try as data
                    NSData *data = [NSData dataWithContentsOfFile:fullPath];
                    if (data) {
                        // Try property list deserialization
                        id pobj = [NSPropertyListSerialization propertyListWithData:data
                            options:NSPropertyListImmutable format:NULL error:nil];
                        if (pobj) {
                            printf("  (deserialized as %s):\n", class_getName([pobj class]));
                            if ([pobj isKindOfClass:[NSDictionary class]])
                                print_dict(pobj, 2);
                            else
                                printf("  %s\n", [[pobj description] UTF8String]);
                        } else {
                            printf("  <binary data, %lu bytes>\n", (unsigned long)[data length]);
                            // Print first 256 bytes as hex
                            const uint8_t *bytes = [data bytes];
                            NSUInteger len = MIN([data length], 256);
                            for (NSUInteger i = 0; i < len; i++) {
                                if (i % 32 == 0) printf("  ");
                                printf("%02x", bytes[i]);
                                if (i % 32 == 31 || i == len-1) printf("\n");
                            }
                        }
                    }
                }
            }
        }

        // Also look for any model.mil.anec or compiled artifacts that might have options
        for (NSString *f in files) {
            if ([f hasSuffix:@".anec"] || [f containsString:@"net.plist"] || [f containsString:@"config"]) {
                NSString *fullPath = [td stringByAppendingPathComponent:f];
                printf("\nFound artifact: %s\n", [f UTF8String]);
                NSDictionary *plist = [NSDictionary dictionaryWithContentsOfFile:fullPath];
                if (plist) {
                    printf("  Contents (plist):\n");
                    print_dict(plist, 2);
                }
            }
        }

        // ════════════════════════════════════════════════════════════════
        // 4. Compile with various option dicts
        // ════════════════════════════════════════════════════════════════
        printf("\n═══ 4. Compile with Various Options ═══\n\n");

        struct { const char *name; NSDictionary *opts; } trials[] = {
            {"enableChaining=YES", @{@"enableChaining": @YES}},
            {"queueDepth=4", @{@"queueDepth": @4}},
            {"enablePipeline=YES", @{@"enablePipeline": @YES}},
            {"ANEChainingEnabled=YES", @{@"ANEChainingEnabled": @YES}},
            {"kANEFEnableFWToFWSignal=YES", @{@"kANEFEnableFWToFWSignal": @YES}},
            {"chainingEnabled=YES", @{@"chainingEnabled": @YES}},
            {"pipelineEnabled=YES", @{@"pipelineEnabled": @YES}},
            {"enableFirmwareChaining=YES", @{@"enableFirmwareChaining": @YES}},
            {"ANECompilerOptionsEnableChaining=YES", @{@"ANECompilerOptionsEnableChaining": @YES}},
            {"loopbackEnabled=YES", @{@"loopbackEnabled": @YES}},
            {"enableLoopback=YES", @{@"enableLoopback": @YES}},
        };
        int nTrials = sizeof(trials) / sizeof(trials[0]);

        for (int i = 0; i < nTrials; i++) {
            printf("[%d/%d] %s\n", i+1, nTrials, trials[i].name);
            id m = compile_model(64, 64, 32, trials[i].opts);
            if (m) {
                printf("  compile+load OK\n");

                // Check if the options appear in compilerOptionsWithOptions
                @try {
                    id merged = ((id(*)(id,SEL,id,BOOL))objc_msgSend)(m,
                        @selector(compilerOptionsWithOptions:isCompiledModelCached:),
                        trials[i].opts, NO);
                    if ([merged isKindOfClass:[NSDictionary class]]) {
                        NSDictionary *md = (NSDictionary *)merged;
                        // Only print keys that differ from default or are new
                        printf("  compiler options dict has %lu keys\n", (unsigned long)[md count]);
                        // Check for our key
                        NSString *ourKey = [[trials[i].opts allKeys] firstObject];
                        if (md[ourKey]) {
                            printf("  -> our key '%s' present: %s\n",
                                   [ourKey UTF8String], [[md[ourKey] description] UTF8String]);
                        }
                    }
                } @catch (NSException *ex) {}

                // Try chaining
                try_chaining(m, trials[i].name);
            } else {
                printf("  compile FAILED\n");
            }
            printf("\n");
        }

        // ════════════════════════════════════════════════════════════════
        // 5. Get default compiler options keys and try them
        // ════════════════════════════════════════════════════════════════
        printf("\n═══ 5. Default Keys as Compile Options ═══\n\n");

        // Get the default options dict
        @try {
            id defaultOpts = ((id(*)(id,SEL,id,BOOL))objc_msgSend)(baseModel,
                @selector(compilerOptionsWithOptions:isCompiledModelCached:), @{}, NO);
            if ([defaultOpts isKindOfClass:[NSDictionary class]]) {
                NSDictionary *dd = (NSDictionary *)defaultOpts;
                printf("Default compiler options has %lu keys.\n", (unsigned long)[dd count]);
                printf("Keys:\n");
                for (NSString *key in [[dd allKeys] sortedArrayUsingSelector:@selector(compare:)]) {
                    id val = dd[key];
                    printf("  %s = %s (%s)\n", [key UTF8String],
                           [[val description] UTF8String], class_getName([val class]));
                }

                // For each boolean key set to NO/0, try setting to YES/1
                printf("\nTrying to flip boolean keys to YES:\n");
                for (NSString *key in [[dd allKeys] sortedArrayUsingSelector:@selector(compare:)]) {
                    id val = dd[key];
                    if ([val isKindOfClass:[NSNumber class]]) {
                        NSNumber *num = (NSNumber *)val;
                        // If it's 0/NO, try setting to 1/YES
                        if ([num intValue] == 0) {
                            printf("\n  Trying %s = YES (was %s):\n", [key UTF8String], [[num description] UTF8String]);
                            NSDictionary *testOpts = @{key: @YES};
                            id m = compile_model(64, 64, 32, testOpts);
                            if (m) {
                                printf("    compile+load OK\n");
                                try_chaining(m, [key UTF8String]);
                            } else {
                                printf("    compile FAILED\n");
                            }
                        }
                    }
                }
            }
        } @catch (NSException *ex) {
            printf("  EXCEPTION: %s\n", [[ex reason] UTF8String]);
        }

        // ════════════════════════════════════════════════════════════════
        // 6. Set queueDepth before compile + try chaining
        // ════════════════════════════════════════════════════════════════
        printf("\n═══ 6. queueDepth Before Compile + Chaining ═══\n\n");

        for (int qd = 1; qd <= 8; qd++) {
            printf("queueDepth=%d:\n", qd);
            Class D = NSClassFromString(@"_ANEInMemoryModelDescriptor");
            Class I = NSClassFromString(@"_ANEInMemoryModel");
            NSData *mil = gen_mil(64, 64, 32);

            id desc = ((id(*)(Class,SEL,id,id,id))objc_msgSend)(D,
                @selector(modelWithMILText:weights:optionsPlist:), mil, @{}, nil);
            id model = ((id(*)(Class,SEL,id))objc_msgSend)(I,
                @selector(inMemoryModelWithDescriptor:), desc);

            id hx2 = ((id(*)(id,SEL))objc_msgSend)(model, @selector(hexStringIdentifier));
            NSString *td2 = [NSTemporaryDirectory() stringByAppendingPathComponent:hx2];
            [fm createDirectoryAtPath:[td2 stringByAppendingPathComponent:@"weights"]
                withIntermediateDirectories:YES attributes:nil error:nil];
            [mil writeToFile:[td2 stringByAppendingPathComponent:@"model.mil"] atomically:YES];

            // Set queueDepth BEFORE compile
            ((void(*)(id,SEL,char))objc_msgSend)(model, @selector(setQueueDepth:), (char)qd);
            char readBack = ((char(*)(id,SEL))objc_msgSend)(model, @selector(queueDepth));
            printf("  queueDepth readback: %d\n", readBack);

            NSError *e = nil;
            BOOL ok = ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(
                model, @selector(compileWithQoS:options:error:), 21, @{}, &e);
            if (!ok) { printf("  compile FAILED\n\n"); continue; }
            e = nil;
            ok = ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(
                model, @selector(loadWithQoS:options:error:), 21, @{}, &e);
            if (!ok) { printf("  load FAILED\n\n"); continue; }

            printf("  compile+load OK\n");

            // Check compiler options after compile
            @try {
                id opts = ((id(*)(id,SEL,id,BOOL))objc_msgSend)(model,
                    @selector(compilerOptionsWithOptions:isCompiledModelCached:), @{}, YES);
                if ([opts isKindOfClass:[NSDictionary class]]) {
                    NSDictionary *od = (NSDictionary *)opts;
                    // Look for queueDepth or chaining related keys
                    for (NSString *key in od) {
                        NSString *lk = [key lowercaseString];
                        if ([lk containsString:@"queue"] || [lk containsString:@"chain"] ||
                            [lk containsString:@"pipe"] || [lk containsString:@"loop"] ||
                            [lk containsString:@"depth"]) {
                            printf("  opts[%s] = %s\n", [key UTF8String], [[od[key] description] UTF8String]);
                        }
                    }
                }
            } @catch (NSException *ex) {}

            try_chaining(model, [[NSString stringWithFormat:@"qd=%d", qd] UTF8String]);
            printf("\n");
        }

        // ════════════════════════════════════════════════════════════════
        // 7. MLModelConfiguration neuralEngineCompilerOptions
        // ════════════════════════════════════════════════════════════════
        printf("\n═══ 7. MLModelConfiguration Private Properties ═══\n\n");

        MLModelConfiguration *config = [[MLModelConfiguration alloc] init];
        config.computeUnits = MLComputeUnitsCPUAndNeuralEngine;

        // Probe neuralEngineCompilerOptions
        @try {
            id val = [config valueForKey:@"neuralEngineCompilerOptions"];
            printf("neuralEngineCompilerOptions (default): %s (%s)\n",
                   val ? [[val description] UTF8String] : "(nil)",
                   val ? class_getName([val class]) : "nil");
        } @catch (NSException *ex) {
            printf("neuralEngineCompilerOptions: NOT A VALID KEY (%s)\n", [[ex reason] UTF8String]);
        }

        // Try setting it
        @try {
            NSDictionary *aneOpts = @{
                @"enableChaining": @YES,
                @"queueDepth": @4,
                @"enablePipeline": @YES,
            };
            [config setValue:aneOpts forKey:@"neuralEngineCompilerOptions"];
            id readBack = [config valueForKey:@"neuralEngineCompilerOptions"];
            printf("After setting neuralEngineCompilerOptions: %s\n",
                   readBack ? [[readBack description] UTF8String] : "(nil)");
        } @catch (NSException *ex) {
            printf("Setting neuralEngineCompilerOptions FAILED: %s\n", [[ex reason] UTF8String]);
        }

        // Probe e5rtCustomANECompilerOptions
        @try {
            id val = [config valueForKey:@"e5rtCustomANECompilerOptions"];
            printf("\ne5rtCustomANECompilerOptions (default): %s (%s)\n",
                   val ? [[val description] UTF8String] : "(nil)",
                   val ? class_getName([val class]) : "nil");
        } @catch (NSException *ex) {
            printf("\ne5rtCustomANECompilerOptions: NOT A VALID KEY (%s)\n", [[ex reason] UTF8String]);
        }

        // Try setting it
        @try {
            NSDictionary *e5Opts = @{
                @"enableChaining": @YES,
                @"ANEChainingEnabled": @YES,
            };
            [config setValue:e5Opts forKey:@"e5rtCustomANECompilerOptions"];
            id readBack = [config valueForKey:@"e5rtCustomANECompilerOptions"];
            printf("After setting e5rtCustomANECompilerOptions: %s\n",
                   readBack ? [[readBack description] UTF8String] : "(nil)");
        } @catch (NSException *ex) {
            printf("Setting e5rtCustomANECompilerOptions FAILED: %s\n", [[ex reason] UTF8String]);
        }

        // Dump all properties on MLModelConfiguration for completeness
        printf("\nAll MLModelConfiguration properties:\n");
        unsigned int pcount = 0;
        objc_property_t *props = class_copyPropertyList([MLModelConfiguration class], &pcount);
        for (unsigned int i = 0; i < pcount; i++) {
            const char *name = property_getName(props[i]);
            @try {
                id val = [config valueForKey:[NSString stringWithUTF8String:name]];
                printf("  %s = %s\n", name, val ? [[val description] UTF8String] : "(nil)");
            } @catch (NSException *ex) {
                printf("  %s = <inaccessible>\n", name);
            }
        }
        free(props);

        // Also probe ivars
        printf("\nMLModelConfiguration ivars:\n");
        unsigned int icount = 0;
        Ivar *ivars = class_copyIvarList([MLModelConfiguration class], &icount);
        for (unsigned int i = 0; i < icount; i++) {
            const char *name = ivar_getName(ivars[i]);
            const char *type = ivar_getTypeEncoding(ivars[i]);
            printf("  %s [%s]\n", name, type ? type : "?");
        }
        free(ivars);

        // ════════════════════════════════════════════════════════════════
        // 8. Dump _ANEInMemoryModel methods related to compiler
        // ════════════════════════════════════════════════════════════════
        printf("\n═══ 8. _ANEInMemoryModel Compiler-Related Methods ═══\n\n");

        Class IMM = NSClassFromString(@"_ANEInMemoryModel");
        unsigned int mcount = 0;
        Method *methods = class_copyMethodList(IMM, &mcount);
        for (unsigned int i = 0; i < mcount; i++) {
            const char *name = sel_getName(method_getName(methods[i]));
            NSString *ns = [NSString stringWithUTF8String:name];
            if ([ns containsString:@"ompil"] || [ns containsString:@"ption"] ||
                [ns containsString:@"hain"] || [ns containsString:@"queue"] ||
                [ns containsString:@"ipe"]) {
                const char *types = method_getTypeEncoding(methods[i]);
                printf("  - %s  [%s]\n", name, types ? types : "?");
            }
        }
        free(methods);

        // Same for _ANEModel
        Class AM = NSClassFromString(@"_ANEModel");
        mcount = 0;
        methods = class_copyMethodList(AM, &mcount);
        printf("\n_ANEModel compiler/chain/pipe methods:\n");
        for (unsigned int i = 0; i < mcount; i++) {
            const char *name = sel_getName(method_getName(methods[i]));
            NSString *ns = [NSString stringWithUTF8String:name];
            if ([ns containsString:@"ompil"] || [ns containsString:@"ption"] ||
                [ns containsString:@"hain"] || [ns containsString:@"queue"] ||
                [ns containsString:@"ipe"]) {
                const char *types = method_getTypeEncoding(methods[i]);
                printf("  - %s  [%s]\n", name, types ? types : "?");
            }
        }
        free(methods);

        printf("\nDone.\n");
    }
    return 0;
}
