#import <Foundation/Foundation.h>
#import <objc/runtime.h>
#import <dlfcn.h>

int main() {
    @autoreleasepool {
        setbuf(stdout, NULL);
        dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine", RTLD_NOW);

        // Search all classes for statType
        unsigned int count = 0;
        Class *classes = objc_copyClassList(&count);
        printf("Classes with 'statType' method or property:\n");
        for (unsigned int i = 0; i < count; i++) {
            const char *cn = class_getName(classes[i]);
            // Check instance methods
            unsigned int mcount = 0;
            Method *methods = class_copyMethodList(classes[i], &mcount);
            for (unsigned int j = 0; j < mcount; j++) {
                const char *mn = sel_getName(method_getName(methods[j]));
                if (strstr(mn, "statType") || strstr(mn, "StatType")) {
                    printf("  %s -%s\n", cn, mn);
                }
            }
            free(methods);
            // Check class methods
            methods = class_copyMethodList(object_getClass(classes[i]), &mcount);
            for (unsigned int j = 0; j < mcount; j++) {
                const char *mn = sel_getName(method_getName(methods[j]));
                if (strstr(mn, "statType") || strstr(mn, "StatType")) {
                    printf("  %s +%s\n", cn, mn);
                }
            }
            free(methods);
            // Check protocols
            if (strstr(cn, "ANE") && strstr(cn, "tat")) {
                printf("  Found ANE stats class: %s\n", cn);
            }
        }
        free(classes);

        // Search protocols
        printf("\nProtocols with 'stat' in name:\n");
        unsigned int pcount = 0;
        Protocol * __unsafe_unretained *protocols = objc_copyProtocolList(&pcount);
        for (unsigned int i = 0; i < pcount; i++) {
            const char *pn = protocol_getName(protocols[i]);
            if (strstr(pn, "tat") || strstr(pn, "Perf")) {
                printf("  Protocol: %s\n", pn);
                // Dump methods
                unsigned int pmcount = 0;
                struct objc_method_description *pms = protocol_copyMethodDescriptionList(protocols[i], YES, YES, &pmcount);
                for (unsigned int j = 0; j < pmcount; j++) {
                    printf("    required: %s\n", sel_getName(pms[j].name));
                }
                free(pms);
                pms = protocol_copyMethodDescriptionList(protocols[i], NO, YES, &pmcount);
                for (unsigned int j = 0; j < pmcount; j++) {
                    printf("    optional: %s\n", sel_getName(pms[j].name));
                }
                free(pms);
            }
        }
        free(protocols);

        // Check _ANERequest for how it uses statType
        Class reqClass = NSClassFromString(@"_ANERequest");
        printf("\n_ANERequest methods with 'stat' or 'perf':\n");
        unsigned int rmcount = 0;
        Method *rmethods = class_copyMethodList(reqClass, &rmcount);
        for (unsigned int i = 0; i < rmcount; i++) {
            const char *mn = sel_getName(method_getName(rmethods[i]));
            if (strstr(mn, "stat") || strstr(mn, "Stat") || strstr(mn, "perf") || strstr(mn, "Perf")) {
                printf("  -%s\n", mn);
            }
        }
        free(rmethods);

        printf("\nDone\n");
    }
    return 0;
}
