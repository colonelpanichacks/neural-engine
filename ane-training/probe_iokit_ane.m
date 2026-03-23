/*
 * probe_iokit_ane.m — Direct IOKit probing of Apple Neural Engine services
 *
 * Attempts to open ANE user clients and enumerate external methods.
 * Target services: H11ANEIn, AppleT6041ANEHAL, H1xANELoadBalancer
 *
 * Build:
 *   xcrun clang -O2 -framework Foundation -framework IOKit \
 *     -isysroot $(xcrun --show-sdk-path) -fobjc-arc -o probe_iokit_ane probe_iokit_ane.m
 */

#import <Foundation/Foundation.h>
#import <IOKit/IOKitLib.h>
#include <mach/mach.h>

// ---------------------------------------------------------------------------
// Helper: dump an IORegistry properties dictionary
// ---------------------------------------------------------------------------
static void dumpProperties(io_service_t service, const char *label) {
    CFMutableDictionaryRef props = NULL;
    kern_return_t kr = IORegistryEntryCreateCFProperties(service, &props,
                                                          kCFAllocatorDefault, 0);
    if (kr != KERN_SUCCESS || !props) {
        printf("  [%s] Could not read properties (kr=0x%x)\n", label, kr);
        return;
    }

    printf("\n  === Properties for %s ===\n", label);

    // Convert to NSDict for easy enumeration
    NSDictionary *dict = (__bridge_transfer NSDictionary *)props;
    for (NSString *key in [dict.allKeys sortedArrayUsingSelector:@selector(compare:)]) {
        id val = dict[key];
        // Truncate long data blobs
        NSString *desc;
        if ([val isKindOfClass:[NSData class]]) {
            NSData *d = (NSData *)val;
            if (d.length > 64) {
                desc = [NSString stringWithFormat:@"<Data %lu bytes>", (unsigned long)d.length];
            } else {
                desc = [d description];
            }
        } else {
            desc = [NSString stringWithFormat:@"%@", val];
            if (desc.length > 200) {
                desc = [[desc substringToIndex:197] stringByAppendingString:@"..."];
            }
        }
        printf("    %-40s = %s\n", key.UTF8String, desc.UTF8String);
    }
    printf("  === End Properties ===\n\n");
}

// ---------------------------------------------------------------------------
// Helper: try to open a connection to a named service
// ---------------------------------------------------------------------------
static io_connect_t tryOpenService(const char *className, const char *label) {
    printf("\n--- Trying to open service: %s (match: %s) ---\n", label, className);

    // Try matching by class name
    CFMutableDictionaryRef matching = IOServiceMatching(className);
    if (!matching) {
        printf("  IOServiceMatching returned NULL for '%s'\n", className);
        return IO_OBJECT_NULL;
    }

    io_service_t service = IOServiceGetMatchingService(kIOMainPortDefault, matching);
    // matching is consumed by the call above

    if (service == IO_OBJECT_NULL) {
        printf("  No matching service found for '%s'\n", className);
        return IO_OBJECT_NULL;
    }

    printf("  Found service (port=%u)\n", service);

    // Dump properties before trying to open
    dumpProperties(service, label);

    // Get the class name to confirm
    io_name_t name;
    kern_return_t kr = IOObjectGetClass(service, name);
    if (kr == KERN_SUCCESS) {
        printf("  Actual class: %s\n", name);
    }

    // Try to open a user client connection
    io_connect_t conn = IO_OBJECT_NULL;

    // Try type 0 first (default user client)
    for (uint32_t type = 0; type < 4; type++) {
        kr = IOServiceOpen(service, mach_task_self(), type, &conn);
        if (kr == KERN_SUCCESS) {
            printf("  SUCCESS: Opened connection (type=%u, conn=%u)\n", type, conn);
            IOObjectRelease(service);
            return conn;
        } else {
            printf("  IOServiceOpen(type=%u) failed: 0x%x (%s)\n",
                   type, kr, mach_error_string(kr));
        }
    }

    IOObjectRelease(service);
    return IO_OBJECT_NULL;
}

// ---------------------------------------------------------------------------
// Helper: return code name for common IOKit returns
// ---------------------------------------------------------------------------
static const char *ioReturnName(kern_return_t kr) {
    switch (kr) {
        case KERN_SUCCESS:              return "kIOReturnSuccess";
        case 0xe00002bc:                return "kIOReturnBadArgument";
        case 0xe00002c2:                return "kIOReturnUnsupported";
        case 0xe00002be:                return "kIOReturnNotPrivileged";
        case 0xe00002cd:                return "kIOReturnNotPermitted";
        case 0xe00002c7:                return "kIOReturnNoDevice";
        case 0xe00002ed:                return "kIOReturnInternalError";
        case 0xe00002c1:                return "kIOReturnExclusiveAccess";
        case 0xe00002ce:                return "kIOReturnInvalid";
        case 0xe00002d8:                return "kIOReturnNotReady";
        case 0xe00002eb:                return "kIOReturnAborted";
        case 0xe0000001:                return "kIOReturnError";
        default:                        return "unknown";
    }
}

// ---------------------------------------------------------------------------
// Probe: enumerate external methods on an open connection
// ---------------------------------------------------------------------------
static void probeExternalMethods(io_connect_t conn) {
    printf("\n=== Probing IOConnectCallMethod selectors 0-31 (null params) ===\n");

    int validCount = 0;

    for (uint32_t sel = 0; sel < 32; sel++) {
        kern_return_t kr;

        // IOConnectCallMethod with all-NULL params
        // This is the safest call — no input, no output
        uint64_t scalarOut[16] = {0};
        uint32_t scalarOutCnt = 16;
        char structOut[4096] = {0};
        size_t structOutSize = sizeof(structOut);

        kr = IOConnectCallMethod(conn, sel,
                                 NULL, 0,        // scalar input
                                 NULL, 0,        // struct input
                                 scalarOut, &scalarOutCnt,
                                 structOut, &structOutSize);

        if (kr == 0xe00002bc) {
            // kIOReturnBadArgument — selector likely doesn't exist or needs params
            // Still print for completeness but mark as "needs args"
            printf("  sel %2u: 0x%08x %-28s (needs arguments or invalid)\n",
                   sel, kr, ioReturnName(kr));
        } else {
            printf("  sel %2u: 0x%08x %-28s *** VALID ***", sel, kr, ioReturnName(kr));
            if (kr == KERN_SUCCESS) {
                printf(" | scalarOut[%u]", scalarOutCnt);
                for (uint32_t i = 0; i < scalarOutCnt && i < 4; i++) {
                    printf(" [%u]=0x%llx", i, scalarOut[i]);
                }
                if (structOutSize > 0 && structOutSize != sizeof(structOut)) {
                    printf(" | structOut %zu bytes", structOutSize);
                }
            }
            printf("\n");
            validCount++;
        }
    }
    printf("  Total valid selectors: %d\n", validCount);

    // --- Try IOConnectCallScalarMethod on discovered selectors ---
    printf("\n=== Probing IOConnectCallScalarMethod selectors 0-31 ===\n");
    for (uint32_t sel = 0; sel < 32; sel++) {
        uint64_t output[16] = {0};
        uint32_t outputCnt = 16;

        kern_return_t kr = IOConnectCallScalarMethod(conn, sel,
                                                      NULL, 0,
                                                      output, &outputCnt);
        if (kr != 0xe00002bc) {  // skip BadArgument (likely invalid)
            printf("  sel %2u: 0x%08x %-28s", sel, kr, ioReturnName(kr));
            if (kr == KERN_SUCCESS && outputCnt > 0) {
                printf(" | out[%u]:", outputCnt);
                for (uint32_t i = 0; i < outputCnt && i < 4; i++) {
                    printf(" 0x%llx", output[i]);
                }
            }
            printf("\n");
        }
    }

    // --- Try IOConnectCallStructMethod on discovered selectors ---
    printf("\n=== Probing IOConnectCallStructMethod selectors 0-31 ===\n");
    for (uint32_t sel = 0; sel < 32; sel++) {
        char outBuf[4096] = {0};
        size_t outSize = sizeof(outBuf);

        kern_return_t kr = IOConnectCallStructMethod(conn, sel,
                                                      NULL, 0,
                                                      outBuf, &outSize);
        if (kr != 0xe00002bc) {  // skip BadArgument
            printf("  sel %2u: 0x%08x %-28s", sel, kr, ioReturnName(kr));
            if (kr == KERN_SUCCESS && outSize > 0) {
                printf(" | %zu bytes out", outSize);
                // Hex dump first 64 bytes
                size_t dumpLen = outSize < 64 ? outSize : 64;
                printf(" | hex:");
                for (size_t i = 0; i < dumpLen; i++) {
                    if (i % 16 == 0) printf("\n    ");
                    printf("%02x ", (unsigned char)outBuf[i]);
                }
            }
            printf("\n");
        }
    }
}

// ---------------------------------------------------------------------------
// Walk IORegistry tree looking for ANE-related entries
// ---------------------------------------------------------------------------
static void walkRegistryForANE(void) {
    printf("\n=== Walking IORegistry for ANE-related entries ===\n");

    io_iterator_t iter;
    kern_return_t kr = IORegistryEntryCreateIterator(
        IORegistryGetRootEntry(kIOMainPortDefault),
        kIOServicePlane,
        kIORegistryIterateRecursively,
        &iter);

    if (kr != KERN_SUCCESS) {
        printf("  Failed to create iterator: 0x%x\n", kr);
        return;
    }

    io_object_t entry;
    int found = 0;
    while ((entry = IOIteratorNext(iter)) != IO_OBJECT_NULL) {
        io_name_t name;
        IORegistryEntryGetName(entry, name);

        // Check if name contains "ANE" or "ane"
        NSString *nameStr = [NSString stringWithUTF8String:name];
        if ([nameStr rangeOfString:@"ANE" options:NSCaseInsensitiveSearch].location != NSNotFound ||
            [nameStr rangeOfString:@"ane" options:NSCaseInsensitiveSearch].location != NSNotFound) {

            io_name_t className;
            IOObjectGetClass(entry, className);
            io_string_t path;
            IORegistryEntryGetPath(entry, kIOServicePlane, path);

            printf("  [%d] Name: %-30s Class: %-30s\n", found, name, className);
            printf("       Path: %s\n", path);
            found++;

            // Dump a few key properties
            CFMutableDictionaryRef props = NULL;
            if (IORegistryEntryCreateCFProperties(entry, &props,
                                                   kCFAllocatorDefault, 0) == KERN_SUCCESS && props) {
                NSDictionary *dict = (__bridge_transfer NSDictionary *)props;
                // Look for interesting keys
                NSArray *interestingKeys = @[
                    @"IOClass", @"IONameMatch", @"IOProviderClass",
                    @"ane-type", @"firmware-version", @"fw-version",
                    @"ane-count", @"num-engines", @"capabilities",
                    @"chaining-support", @"max-batch-size",
                    @"performance-state", @"power-domain",
                    @"model", @"compatible", @"device-type"
                ];
                for (NSString *key in interestingKeys) {
                    id val = dict[key];
                    if (val) {
                        printf("       %-30s = %s\n", key.UTF8String,
                               [[NSString stringWithFormat:@"%@", val] UTF8String]);
                    }
                }
            }
        }
        IOObjectRelease(entry);
    }
    IOObjectRelease(iter);
    printf("  Found %d ANE-related registry entries\n", found);
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int main(int argc, const char *argv[]) {
    @autoreleasepool {
        printf("=== ANE IOKit Direct Probe ===\n");
        printf("PID: %d, UID: %d\n\n", getpid(), getuid());

        // First, walk the registry for all ANE entries
        walkRegistryForANE();

        // Service names to try (in priority order)
        const char *services[][2] = {
            {"H11ANEIn",             "H11ANE UserClient"},
            {"AppleT6041ANEHAL",     "M4 Pro HAL"},
            {"H1xANELoadBalancer",   "ANE Load Balancer"},
            {"H11ANEController",     "ANE Controller"},
            {"AppleARMIODevice",     "ARM IO Device (ANE)"},
        };
        int numServices = sizeof(services) / sizeof(services[0]);

        io_connect_t conn = IO_OBJECT_NULL;
        const char *connectedName = NULL;

        for (int i = 0; i < numServices; i++) {
            conn = tryOpenService(services[i][0], services[i][1]);
            if (conn != IO_OBJECT_NULL) {
                connectedName = services[i][1];
                break;
            }
        }

        if (conn == IO_OBJECT_NULL) {
            printf("\n*** Could not open any ANE service. ***\n");
            printf("This is expected on modern macOS due to sandbox/entitlement\n");
            printf("requirements. The ANE typically requires:\n");
            printf("  - com.apple.ane.user-access entitlement\n");
            printf("  - com.apple.private.ane.* entitlements\n");
            printf("  - Or running via AppleNeuralEngine.framework\n\n");

            // Even without a connection, try to get properties from the service
            printf("--- Attempting property reads without user client ---\n");
            const char *classNames[] = {"H11ANEIn", "AppleT6041ANEHAL",
                                         "H1xANELoadBalancer", "H11ANEController"};
            for (int i = 0; i < 4; i++) {
                CFMutableDictionaryRef matching = IOServiceMatching(classNames[i]);
                if (!matching) continue;
                io_service_t svc = IOServiceGetMatchingService(kIOMainPortDefault, matching);
                if (svc != IO_OBJECT_NULL) {
                    printf("\n  Found service '%s' — reading properties:\n", classNames[i]);
                    dumpProperties(svc, classNames[i]);

                    // Also check for child entries
                    io_iterator_t childIter;
                    kern_return_t kr = IORegistryEntryGetChildIterator(svc, kIOServicePlane, &childIter);
                    if (kr == KERN_SUCCESS) {
                        io_object_t child;
                        while ((child = IOIteratorNext(childIter)) != IO_OBJECT_NULL) {
                            io_name_t childName, childClass;
                            IORegistryEntryGetName(child, childName);
                            IOObjectGetClass(child, childClass);
                            printf("  Child: %s (class: %s)\n", childName, childClass);
                            IOObjectRelease(child);
                        }
                        IOObjectRelease(childIter);
                    }

                    IOObjectRelease(svc);
                }
            }
        } else {
            printf("\n*** Successfully connected to: %s ***\n", connectedName);
            probeExternalMethods(conn);
            IOServiceClose(conn);
            printf("\nConnection closed.\n");
        }

        printf("\n=== Probe complete ===\n");
    }
    return 0;
}
