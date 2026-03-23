// probe_coreml_chain.m — Research how CoreML dispatches to ANE internally
// Dumps methods on CoreML classes related to ANE dispatch, plus private ANE classes
#import <Foundation/Foundation.h>
#import <CoreML/CoreML.h>
#import <objc/runtime.h>
#import <dlfcn.h>

// ── helpers ──────────────────────────────────────────────────────────────────

static void dump_methods(Class cls, const char *label) {
    if (!cls) {
        printf("  [%s] class not found\n", label);
        return;
    }
    printf("\n═══ %s ═══\n", label);

    // instance methods
    unsigned int icount = 0;
    Method *imethods = class_copyMethodList(cls, &icount);
    if (icount > 0) {
        printf("  Instance methods (%u):\n", icount);
        for (unsigned int i = 0; i < icount; i++) {
            SEL sel = method_getName(imethods[i]);
            const char *types = method_getTypeEncoding(imethods[i]);
            printf("    - %s  [%s]\n", sel_getName(sel), types ? types : "?");
        }
    }
    free(imethods);

    // class methods
    Class meta = object_getClass(cls);
    unsigned int ccount = 0;
    Method *cmethods = class_copyMethodList(meta, &ccount);
    if (ccount > 0) {
        printf("  Class methods (%u):\n", ccount);
        for (unsigned int i = 0; i < ccount; i++) {
            SEL sel = method_getName(cmethods[i]);
            const char *types = method_getTypeEncoding(cmethods[i]);
            printf("    + %s  [%s]\n", sel_getName(sel), types ? types : "?");
        }
    }
    free(cmethods);

    if (icount == 0 && ccount == 0)
        printf("  (no methods)\n");
}

static void dump_properties(Class cls, const char *label) {
    if (!cls) return;
    unsigned int pcount = 0;
    objc_property_t *props = class_copyPropertyList(cls, &pcount);
    if (pcount > 0) {
        printf("  Properties (%u):\n", pcount);
        for (unsigned int i = 0; i < pcount; i++) {
            const char *name = property_getName(props[i]);
            const char *attrs = property_getAttributes(props[i]);
            printf("    . %s  [%s]\n", name, attrs ? attrs : "?");
        }
    }
    free(props);
}

static void dump_filtered_methods(Class cls, const char *label, const char **keywords, int nkw) {
    if (!cls) {
        printf("  [%s] class not found\n", label);
        return;
    }

    printf("\n─── %s: filtered methods (pipeline/chain/queue/batch/ane/neural/dispatch) ───\n", label);
    int found = 0;

    // instance
    unsigned int icount = 0;
    Method *imethods = class_copyMethodList(cls, &icount);
    for (unsigned int i = 0; i < icount; i++) {
        const char *name = sel_getName(method_getName(imethods[i]));
        NSString *ns = [NSString stringWithUTF8String:name];
        for (int k = 0; k < nkw; k++) {
            if ([ns rangeOfString:[NSString stringWithUTF8String:keywords[k]]
                          options:NSCaseInsensitiveSearch].location != NSNotFound) {
                printf("    - %s\n", name);
                found++;
                break;
            }
        }
    }
    free(imethods);

    // class
    Class meta = object_getClass(cls);
    unsigned int ccount = 0;
    Method *cmethods = class_copyMethodList(meta, &ccount);
    for (unsigned int i = 0; i < ccount; i++) {
        const char *name = sel_getName(method_getName(cmethods[i]));
        NSString *ns = [NSString stringWithUTF8String:name];
        for (int k = 0; k < nkw; k++) {
            if ([ns rangeOfString:[NSString stringWithUTF8String:keywords[k]]
                          options:NSCaseInsensitiveSearch].location != NSNotFound) {
                printf("    + %s\n", name);
                found++;
                break;
            }
        }
    }
    free(cmethods);

    if (!found) printf("    (none matched)\n");
}

// ── scan all loaded classes for ANE/NeuralEngine in name ─────────────────────

static void scan_all_ane_classes(void) {
    printf("\n\n╔══════════════════════════════════════════════════════╗\n");
    printf("║  SCANNING ALL CLASSES FOR 'ANE' OR 'NeuralEngine'   ║\n");
    printf("╚══════════════════════════════════════════════════════╝\n");

    unsigned int classCount = 0;
    Class *classes = objc_copyClassList(&classCount);
    int found = 0;

    for (unsigned int i = 0; i < classCount; i++) {
        const char *name = class_getName(classes[i]);
        if (!name) continue;
        NSString *ns = [NSString stringWithUTF8String:name];
        if ([ns rangeOfString:@"ANE" options:0].location != NSNotFound ||
            [ns rangeOfString:@"NeuralEngine" options:NSCaseInsensitiveSearch].location != NSNotFound) {
            printf("  [%d] %s\n", ++found, name);
        }
    }
    free(classes);
    printf("  Total ANE/NeuralEngine classes found: %d\n", found);
}

// ── main ─────────────────────────────────────────────────────────────────────

int main(int argc, const char *argv[]) {
    @autoreleasepool {
        printf("probe_coreml_chain — CoreML <-> ANE dispatch research\n");
        printf("=========================================================\n");

        // Load private ANE framework
        void *ane = dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine", RTLD_NOW);
        printf("ANE framework loaded: %s\n", ane ? "YES" : "NO");

        // ── 1. CoreML public classes ──

        printf("\n\n╔══════════════════════════════════════════════════════╗\n");
        printf("║           CoreML PUBLIC CLASSES                      ║\n");
        printf("╚══════════════════════════════════════════════════════╝\n");

        // MLComputePlan (iOS 17+ / macOS 14+)
        Class mlComputePlan = NSClassFromString(@"MLComputePlan");
        printf("\nMLComputePlan exists: %s\n", mlComputePlan ? "YES" : "NO");
        if (mlComputePlan) {
            dump_methods(mlComputePlan, "MLComputePlan");
            dump_properties(mlComputePlan, "MLComputePlan");
        }

        // MLModelConfiguration
        Class mlModelConfig = [MLModelConfiguration class];
        printf("\nMLModelConfiguration exists: YES\n");
        dump_methods(mlModelConfig, "MLModelConfiguration");
        dump_properties(mlModelConfig, "MLModelConfiguration");

        // MLModel
        Class mlModel = [MLModel class];
        dump_methods(mlModel, "MLModel");
        dump_properties(mlModel, "MLModel");

        // MLModelAsset (if exists)
        Class mlModelAsset = NSClassFromString(@"MLModelAsset");
        if (mlModelAsset) {
            dump_methods(mlModelAsset, "MLModelAsset");
            dump_properties(mlModelAsset, "MLModelAsset");
        }

        // ── 2. Filter CoreML classes for dispatch-related methods ──

        printf("\n\n╔══════════════════════════════════════════════════════╗\n");
        printf("║  CoreML DISPATCH-RELATED METHODS (filtered)          ║\n");
        printf("╚══════════════════════════════════════════════════════╝\n");

        const char *kw[] = {"pipeline", "chain", "queue", "batch", "ane",
                            "neural", "dispatch", "compute", "plan", "device",
                            "engine", "accelerate", "predict", "evaluate"};
        int nkw = sizeof(kw) / sizeof(kw[0]);

        dump_filtered_methods(mlModelConfig, "MLModelConfiguration", kw, nkw);
        dump_filtered_methods(mlModel, "MLModel", kw, nkw);
        if (mlComputePlan)
            dump_filtered_methods(mlComputePlan, "MLComputePlan", kw, nkw);

        // ── 3. MLModelConfiguration: computeUnits and other properties ──

        printf("\n\n╔══════════════════════════════════════════════════════╗\n");
        printf("║  MLModelConfiguration DETAILED INSPECTION            ║\n");
        printf("╚══════════════════════════════════════════════════════╝\n");

        MLModelConfiguration *config = [[MLModelConfiguration alloc] init];
        printf("  Default computeUnits: %ld\n", (long)config.computeUnits);
        printf("    MLComputeUnitsAll = %ld\n", (long)MLComputeUnitsAll);
        printf("    MLComputeUnitsCPUOnly = %ld\n", (long)MLComputeUnitsCPUOnly);
        printf("    MLComputeUnitsCPUAndGPU = %ld\n", (long)MLComputeUnitsCPUAndGPU);
        printf("    MLComputeUnitsCPUAndNeuralEngine = %ld\n", (long)MLComputeUnitsCPUAndNeuralEngine);

        // Check for private properties via KVC
        NSArray *privateKeys = @[@"useNeuralEngine", @"pipelineMode", @"chainMode",
                                  @"batchSize", @"queuePriority", @"allowBackgroundAccess",
                                  @"experimentalMLE5EngineUsage", @"prefersSPIPipeline",
                                  @"allowLowPrecisionAccumulationOnGPU",
                                  @"functionName", @"modelDisplayName",
                                  @"optimizationHints", @"modelLifecycle"];
        printf("\n  Probing private/undocumented KVC keys on MLModelConfiguration:\n");
        for (NSString *key in privateKeys) {
            @try {
                id val = [config valueForKey:key];
                printf("    %-50s = %s\n", [key UTF8String],
                       val ? [[val description] UTF8String] : "(nil)");
            } @catch (NSException *e) {
                // not a valid key
            }
        }

        // ── 4. Private ANE classes ──

        printf("\n\n╔══════════════════════════════════════════════════════╗\n");
        printf("║  PRIVATE ANE CLASSES                                 ║\n");
        printf("╚══════════════════════════════════════════════════════╝\n");

        // _ANEDeviceController
        Class devCtrl = NSClassFromString(@"_ANEDeviceController");
        printf("\n_ANEDeviceController exists: %s\n", devCtrl ? "YES" : "NO");
        if (devCtrl) {
            dump_methods(devCtrl, "_ANEDeviceController");
            dump_properties(devCtrl, "_ANEDeviceController");
        }

        // _ANEProgramForEvaluation
        Class progEval = NSClassFromString(@"_ANEProgramForEvaluation");
        printf("\n_ANEProgramForEvaluation exists: %s\n", progEval ? "YES" : "NO");
        if (progEval) {
            dump_methods(progEval, "_ANEProgramForEvaluation");
            dump_properties(progEval, "_ANEProgramForEvaluation");
        }

        // _ANECompiler
        Class compiler = NSClassFromString(@"_ANECompiler");
        printf("\n_ANECompiler exists: %s\n", compiler ? "YES" : "NO");
        if (compiler) {
            dump_methods(compiler, "_ANECompiler");
            dump_properties(compiler, "_ANECompiler");
        }

        // Additional interesting private classes
        const char *extraClasses[] = {
            "_ANEClient", "_ANEModel", "_ANERequest",
            "_ANEChainingRequest", "_ANEBuffer",
            "_ANEIOSurfaceOutputSets", "_ANEOutputSetEnqueue",
            "_ANEVirtualClient", "_ANEInMemoryModel",
            "_ANEInMemoryModelDescriptor", "_ANEPipelineModel",
            "_ANEModelCompilationResult",
        };
        int nExtra = sizeof(extraClasses) / sizeof(extraClasses[0]);
        for (int i = 0; i < nExtra; i++) {
            Class c = NSClassFromString([NSString stringWithUTF8String:extraClasses[i]]);
            if (c) {
                dump_methods(c, extraClasses[i]);
                dump_properties(c, extraClasses[i]);
            } else {
                printf("\n  [%s] class not found\n", extraClasses[i]);
            }
        }

        // ── 5. Scan all loaded classes ──
        scan_all_ane_classes();

        printf("\n\nDone.\n");
    }
    return 0;
}
