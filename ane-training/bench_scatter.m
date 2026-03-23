// bench_scatter.m — Microbenchmark: scatter strategies for qkv_w staging
// Source: contiguous fp16 [2048 ch x 256 elem] = 512 bytes/channel
// Dest: strided fp16 with stride 3840 (QKV_BWD_SP)
// Each channel: 512 bytes copied with 7680-byte gaps (stride * sizeof(fp16) = 3840*2)
//
// Build: xcrun clang -O2 -framework Foundation -isysroot $(xcrun --show-sdk-path) -o bench_scatter bench_scatter.m

#import <Foundation/Foundation.h>
#import <mach/mach_time.h>
#import <stdlib.h>
#import <string.h>
#import <dispatch/dispatch.h>
#import <arm_neon.h>

#define CHANNELS     2048
#define ELEMENTS     256
#define BYTES_PER_CH (ELEMENTS * 2)   // 512 bytes per channel (fp16)
#define QKV_BWD_SP   3840
#define DST_STRIDE   (QKV_BWD_SP * 2) // 7680 bytes between channel starts in dest

#define WARMUP       50
#define ITERATIONS   1000
#define BLOCK        64

// ---- helpers ----

static double ticks_to_us(uint64_t ticks) {
    static mach_timebase_info_data_t tb;
    if (tb.denom == 0) mach_timebase_info(&tb);
    return (double)ticks * tb.numer / tb.denom / 1000.0;
}

static void fill_random_fp16(uint16_t *buf, size_t count) {
    for (size_t i = 0; i < count; i++)
        buf[i] = (uint16_t)(arc4random() & 0xFFFF);
}

// ---- benchmark strategies ----

// 1. memcpy per channel, single-threaded
static void scatter_memcpy_serial(const uint8_t *src, uint8_t *dst) {
    for (int ch = 0; ch < CHANNELS; ch++) {
        memcpy(dst + (size_t)ch * DST_STRIDE,
               src + (size_t)ch * BYTES_PER_CH,
               BYTES_PER_CH);
    }
}

// 2. memcpy with dispatch_apply BLOCK=64
static void scatter_memcpy_dispatch(const uint8_t *src, uint8_t *dst) {
    int nblocks = (CHANNELS + BLOCK - 1) / BLOCK;
    dispatch_apply(nblocks, dispatch_get_global_queue(QOS_CLASS_USER_INTERACTIVE, 0),
                   ^(size_t blk) {
        int start = (int)blk * BLOCK;
        int end = start + BLOCK;
        if (end > CHANNELS) end = CHANNELS;
        for (int ch = start; ch < end; ch++) {
            memcpy(dst + (size_t)ch * DST_STRIDE,
                   src + (size_t)ch * BYTES_PER_CH,
                   BYTES_PER_CH);
        }
    });
}

// 3. STNP (non-temporal store pair) per channel, single-threaded
static void scatter_stnp_serial(const uint8_t *src, uint8_t *dst) {
    for (int ch = 0; ch < CHANNELS; ch++) {
        const uint8_t *s = src + (size_t)ch * BYTES_PER_CH;
        uint8_t *d = dst + (size_t)ch * DST_STRIDE;
        // 512 bytes = 32 x 16-byte pairs via STNP (each stnp stores 2x16=32 bytes)
        // 512 / 32 = 16 stnp iterations
        for (int i = 0; i < 16; i++) {
            uint8x16_t a = vld1q_u8(s + i * 32);
            uint8x16_t b = vld1q_u8(s + i * 32 + 16);
            __asm__ volatile(
                "stnp q0, q1, [%0]"
                :
                : "r"(d + i * 32), "w"(a), "w"(b)
                : "memory"
            );
        }
    }
}

// Fixed STNP using explicit register binding
static inline void stnp_32bytes(const uint8_t *s, uint8_t *d) {
    // Load 32 bytes
    uint8x16_t va = vld1q_u8(s);
    uint8x16_t vb = vld1q_u8(s + 16);
    // Store non-temporal pair
    __asm__ volatile(
        "stnp %q[a], %q[b], [%[dst]]"
        :
        : [a]"w"(va), [b]"w"(vb), [dst]"r"(d)
        : "memory"
    );
}

static void scatter_stnp_serial_v2(const uint8_t *src, uint8_t *dst) {
    for (int ch = 0; ch < CHANNELS; ch++) {
        const uint8_t *s = src + (size_t)ch * BYTES_PER_CH;
        uint8_t *d = dst + (size_t)ch * DST_STRIDE;
        // 512 bytes = 16 x 32-byte STNP stores
        for (int i = 0; i < 16; i++) {
            stnp_32bytes(s + i * 32, d + i * 32);
        }
    }
}

// 4. STNP with dispatch_apply BLOCK=64
static void scatter_stnp_dispatch(const uint8_t *src, uint8_t *dst) {
    int nblocks = (CHANNELS + BLOCK - 1) / BLOCK;
    dispatch_apply(nblocks, dispatch_get_global_queue(QOS_CLASS_USER_INTERACTIVE, 0),
                   ^(size_t blk) {
        int start = (int)blk * BLOCK;
        int end = start + BLOCK;
        if (end > CHANNELS) end = CHANNELS;
        for (int ch = start; ch < end; ch++) {
            const uint8_t *s = src + (size_t)ch * BYTES_PER_CH;
            uint8_t *d = dst + (size_t)ch * DST_STRIDE;
            for (int i = 0; i < 16; i++) {
                stnp_32bytes(s + i * 32, d + i * 32);
            }
        }
    });
}

// 5. Write-prefetch + memcpy, dispatch_apply
static void scatter_prefetch_dispatch(const uint8_t *src, uint8_t *dst) {
    int nblocks = (CHANNELS + BLOCK - 1) / BLOCK;
    dispatch_apply(nblocks, dispatch_get_global_queue(QOS_CLASS_USER_INTERACTIVE, 0),
                   ^(size_t blk) {
        int start = (int)blk * BLOCK;
        int end = start + BLOCK;
        if (end > CHANNELS) end = CHANNELS;
        for (int ch = start; ch < end; ch++) {
            uint8_t *d = dst + (size_t)ch * DST_STRIDE;
            // Write-prefetch: locality 0 = non-temporal hint
            __builtin_prefetch(d, 1, 0);
            __builtin_prefetch(d + 64, 1, 0);
            __builtin_prefetch(d + 128, 1, 0);
            __builtin_prefetch(d + 192, 1, 0);
            __builtin_prefetch(d + 256, 1, 0);
            __builtin_prefetch(d + 320, 1, 0);
            __builtin_prefetch(d + 384, 1, 0);
            __builtin_prefetch(d + 448, 1, 0);
            memcpy(d, src + (size_t)ch * BYTES_PER_CH, BYTES_PER_CH);
        }
    });
}

// 6. DC ZVA + memcpy, dispatch_apply
static void scatter_dczva_dispatch(const uint8_t *src, uint8_t *dst) {
    int nblocks = (CHANNELS + BLOCK - 1) / BLOCK;
    dispatch_apply(nblocks, dispatch_get_global_queue(QOS_CLASS_USER_INTERACTIVE, 0),
                   ^(size_t blk) {
        int start = (int)blk * BLOCK;
        int end = start + BLOCK;
        if (end > CHANNELS) end = CHANNELS;
        for (int ch = start; ch < end; ch++) {
            uint8_t *d = dst + (size_t)ch * DST_STRIDE;
            // Zero cache lines (64 bytes each on Apple Silicon) before writing
            // 512 bytes = 8 cache lines
            for (int cl = 0; cl < 8; cl++) {
                __asm__ volatile("dc zva, %0" : : "r"(d + cl * 64));
            }
            memcpy(d, src + (size_t)ch * BYTES_PER_CH, BYTES_PER_CH);
        }
    });
}

// ---- runner ----

typedef void (*scatter_fn)(const uint8_t *, uint8_t *);

static double run_bench(const char *name, scatter_fn fn,
                        const uint8_t *src, uint8_t *dst, size_t dst_size) {
    // warmup
    for (int i = 0; i < WARMUP; i++) {
        memset(dst, 0, dst_size);
        fn(src, dst);
    }

    // timed
    uint64_t total = 0;
    double min_us = 1e12, max_us = 0;
    for (int i = 0; i < ITERATIONS; i++) {
        memset(dst, 0, dst_size);
        uint64_t t0 = mach_absolute_time();
        fn(src, dst);
        uint64_t t1 = mach_absolute_time();
        double us = ticks_to_us(t1 - t0);
        total += (t1 - t0);
        if (us < min_us) min_us = us;
        if (us > max_us) max_us = us;
    }
    double avg_us = ticks_to_us(total) / ITERATIONS;
    double mb = (double)(CHANNELS * BYTES_PER_CH) / (1024.0 * 1024.0);
    double gbps = mb / (avg_us / 1e6) / 1024.0;

    printf("  %-45s  avg=%7.1f us  min=%7.1f us  max=%7.1f us  %.2f GB/s\n",
           name, avg_us, min_us, max_us, gbps);

    // verify correctness (spot check)
    fn(src, dst);
    int ok = 1;
    for (int ch = 0; ch < CHANNELS; ch += 512) {
        if (memcmp(dst + (size_t)ch * DST_STRIDE,
                   src + (size_t)ch * BYTES_PER_CH,
                   BYTES_PER_CH) != 0) {
            ok = 0;
            break;
        }
    }
    if (!ok) printf("    *** CORRECTNESS FAILURE ***\n");

    return avg_us;
}

int main(int argc, const char *argv[]) {
    @autoreleasepool {
        printf("=== Scatter Benchmark: qkv_w staging pattern ===\n");
        printf("Source: contiguous fp16 [%d ch x %d elem] = %d bytes/ch\n",
               CHANNELS, ELEMENTS, BYTES_PER_CH);
        printf("Dest:   strided, stride=%d bytes (QKV_BWD_SP=%d)\n",
               DST_STRIDE, QKV_BWD_SP);
        printf("Total source: %d KB, dest span: %lu KB\n",
               CHANNELS * BYTES_PER_CH / 1024,
               (unsigned long)((size_t)(CHANNELS - 1) * DST_STRIDE + BYTES_PER_CH) / 1024);
        printf("Warmup: %d, Iterations: %d, dispatch BLOCK: %d\n\n", WARMUP, ITERATIONS, BLOCK);

        // Allocate source (contiguous)
        size_t src_size = (size_t)CHANNELS * BYTES_PER_CH;
        uint8_t *src = NULL;
        posix_memalign((void **)&src, 128, src_size);
        fill_random_fp16((uint16_t *)src, src_size / 2);

        // Allocate dest (large enough for strided layout)
        size_t dst_size = (size_t)(CHANNELS - 1) * DST_STRIDE + BYTES_PER_CH;
        uint8_t *dst = NULL;
        posix_memalign((void **)&dst, 128, dst_size);

        printf("Results:\n");

        double t1 = run_bench("1. memcpy serial (baseline)",
                              scatter_memcpy_serial, src, dst, dst_size);

        double t2 = run_bench("2. memcpy dispatch_apply BLOCK=64",
                              scatter_memcpy_dispatch, src, dst, dst_size);

        double t3 = run_bench("3. STNP serial (v2, register-bound)",
                              scatter_stnp_serial_v2, src, dst, dst_size);

        double t4 = run_bench("4. STNP dispatch_apply BLOCK=64",
                              scatter_stnp_dispatch, src, dst, dst_size);

        double t5 = run_bench("5. write-prefetch + memcpy dispatch",
                              scatter_prefetch_dispatch, src, dst, dst_size);

        double t6 = run_bench("6. DC ZVA + memcpy dispatch",
                              scatter_dczva_dispatch, src, dst, dst_size);

        printf("\n--- Summary (vs baseline) ---\n");
        printf("  1. memcpy serial:          %7.1f us (1.00x)\n", t1);
        printf("  2. memcpy dispatch:        %7.1f us (%.2fx)\n", t2, t1/t2);
        printf("  3. STNP serial:            %7.1f us (%.2fx)\n", t3, t1/t3);
        printf("  4. STNP dispatch:          %7.1f us (%.2fx)\n", t4, t1/t4);
        printf("  5. prefetch+memcpy disp:   %7.1f us (%.2fx)\n", t5, t1/t5);
        printf("  6. DC ZVA+memcpy disp:     %7.1f us (%.2fx)\n", t6, t1/t6);

        free(src);
        free(dst);
    }
    return 0;
}
