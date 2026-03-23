// io.h — IOSurface helpers, NEON conversion, kernel compile/eval
// Updated for GQA (Qwen3-0.6B): Q_DIM != DIM, separate KV heads
#pragma once
#include "config.h"

static IOSurfaceRef make_surface(size_t bytes) {
    return IOSurfaceCreate((__bridge CFDictionaryRef)@{
        (id)kIOSurfaceWidth:@(bytes), (id)kIOSurfaceHeight:@1,
        (id)kIOSurfaceBytesPerElement:@1, (id)kIOSurfaceBytesPerRow:@(bytes),
        (id)kIOSurfaceAllocSize:@(bytes), (id)kIOSurfacePixelFormat:@0});
}

// Blob builder for const weights (fp16 data ready)
static NSData *build_blob_fp16(_Float16 *d, int cnt) {
    int ws=cnt*2, tot=128+ws;
    uint8_t *b=(uint8_t*)calloc(tot,1);
    b[0]=1;b[4]=2;b[64]=0xEF;b[65]=0xBE;b[66]=0xAD;b[67]=0xDE;b[68]=1;
    *(uint32_t*)(b+72)=ws;*(uint32_t*)(b+80)=128;
    memcpy(b+128,d,ws);
    return [NSData dataWithBytesNoCopy:b length:tot freeWhenDone:YES];
}

// Fused residual: out = alpha * fp16_to_fp32(src_fp16) + x, single pass
// Eliminates separate cvt_f16_f32 + vDSP_vsma by fusing conversion into FMA
static void residual_cvt_f16(float *out, const _Float16 *src_fp16, const float *x,
                              float alpha, int n) {
    float32x4_t va = vdupq_n_f32(alpha);
    int i = 0;
    for (; i + 7 < n; i += 8) {
        float16x8_t h = vld1q_f16((const __fp16*)(src_fp16 + i));
        float32x4_t lo = vcvt_f32_f16(vget_low_f16(h));
        float32x4_t hi = vcvt_f32_f16(vget_high_f16(h));
        vst1q_f32(out + i,     vfmaq_f32(vld1q_f32(x + i),     va, lo));
        vst1q_f32(out + i + 4, vfmaq_f32(vld1q_f32(x + i + 4), va, hi));
    }
    for (; i < n; i++) out[i] = alpha * (float)src_fp16[i] + x[i];
}

// NEON vectorized conversion
static void cvt_f16_f32(float *dst, const _Float16 *src, int n) {
    int i = 0;
    for (; i+7 < n; i += 8) {
        float16x8_t h = vld1q_f16((const __fp16*)(src+i));
        vst1q_f32(dst+i,   vcvt_f32_f16(vget_low_f16(h)));
        vst1q_f32(dst+i+4, vcvt_f32_f16(vget_high_f16(h)));
    }
    for (; i < n; i++) dst[i] = (float)src[i];
}
static void cvt_f32_f16(_Float16 *dst, const float *src, int n) {
    int i = 0;
    for (; i+7 < n; i += 8) {
        float16x8_t h = vcombine_f16(vcvt_f16_f32(vld1q_f32(src+i)),
                                      vcvt_f16_f32(vld1q_f32(src+i+4)));
        vst1q_f16((__fp16*)(dst+i), h);
    }
    for (; i < n; i++) dst[i] = (_Float16)src[i];
}

// Fused fp32→fp16 conversion + strided scatter into IOSurface [C,S] layout.
// Eliminates intermediate fp16 staging buffers — converts and writes in one pass.
// src: [channels, seq] fp32 contiguous. dst: [C, S] fp16, writing at dst[ch*stride + offset].
static void cvt_scatter_f32_f16(_Float16 *dst, const float *src,
                                 int channels, int seq, int stride, int sp_offset) {
    for (int ch = 0; ch < channels; ch++) {
        _Float16 *d = dst + ch * stride + sp_offset;
        const float *s = src + ch * seq;
        int i = 0;
        for (; i + 7 < seq; i += 8) {
            float16x8_t h = vcombine_f16(vcvt_f16_f32(vld1q_f32(s + i)),
                                          vcvt_f16_f32(vld1q_f32(s + i + 4)));
            vst1q_f16((__fp16*)(d + i), h);
        }
        for (; i < seq; i++) d[i] = (_Float16)s[i];
    }
}

// Parallel version for large scatters (>512 channels). Uses dispatch_apply to spread
// across P-cores. Block size tuned to amortize dispatch overhead vs cache thrashing.
static dispatch_queue_t g_scatter_q = NULL;
static void cvt_scatter_f32_f16_par(_Float16 *dst, const float *src,
                                     int channels, int seq, int stride, int sp_offset) {
    if (!g_scatter_q) g_scatter_q = dispatch_get_global_queue(QOS_CLASS_USER_INTERACTIVE, 0);
    const int BLOCK = 64;  // channels per work unit
    int nblocks = (channels + BLOCK - 1) / BLOCK;
    dispatch_apply((size_t)nblocks, g_scatter_q, ^(size_t b) {
        int ch_start = (int)b * BLOCK;
        int ch_end = ch_start + BLOCK;
        if (ch_end > channels) ch_end = channels;
        for (int ch = ch_start; ch < ch_end; ch++) {
            _Float16 *d = dst + ch * stride + sp_offset;
            const float *s = src + ch * seq;
            int i = 0;
            // STNP: convert 16 fp32 → 16 fp16 (32 bytes), non-temporal store pair
            for (; i + 15 < seq; i += 16) {
                float16x8_t h0 = vcombine_f16(vcvt_f16_f32(vld1q_f32(s + i)),
                                               vcvt_f16_f32(vld1q_f32(s + i + 4)));
                float16x8_t h1 = vcombine_f16(vcvt_f16_f32(vld1q_f32(s + i + 8)),
                                               vcvt_f16_f32(vld1q_f32(s + i + 12)));
                __asm__ volatile (
                    "stnp q0, q1, [%0]"
                    : : "r"(d + i), "w"(h0), "w"(h1)
                    : "memory"
                );
            }
            for (; i + 7 < seq; i += 8) {
                float16x8_t h = vcombine_f16(vcvt_f16_f32(vld1q_f32(s + i)),
                                              vcvt_f16_f32(vld1q_f32(s + i + 4)));
                vst1q_f16((__fp16*)(d + i), h);
            }
            for (; i < seq; i++) d[i] = (_Float16)s[i];
        }
    });
}

// IOSurface I/O (channel-first [C,S] layout, fp16 on surface)
// No IOSurface locks — ANE eval call provides synchronization
static void io_read_fp16(IOSurfaceRef s, float *data, int ch_off, int channels, int sp) {
    cvt_f16_f32(data, (_Float16*)IOSurfaceGetBaseAddress(s) + ch_off * sp, channels * sp);
}
static void io_read_dyn(IOSurfaceRef s, float *out, int oc, int seq) {
    cvt_f16_f32(out, (_Float16*)IOSurfaceGetBaseAddress(s), oc * seq);
}
static void io_read_ffn_bwd_fused(IOSurfaceRef s, float *dx_ffn, float *dh1, float *dh3) {
    _Float16 *buf = (_Float16*)IOSurfaceGetBaseAddress(s);
    cvt_f16_f32(dx_ffn, buf, DIM*SEQ);
    cvt_f16_f32(dh1, buf + DIM*SEQ, HIDDEN*SEQ);
    cvt_f16_f32(dh3, buf + (DIM+HIDDEN)*SEQ, HIDDEN*SEQ);
}

// Compile MIL to ANE kernel
static Kern *compile_kern_mil_w(NSString *mil, NSDictionary *weights, int ic_bytes, int oc_bytes) {
    @autoreleasepool {
    NSData *md = [mil dataUsingEncoding:NSUTF8StringEncoding];
    id desc = ((id(*)(Class,SEL,id,id,id))objc_msgSend)(g_D, @selector(modelWithMILText:weights:optionsPlist:), md, weights, nil);
    if (!desc) { printf("  [compile] desc=NULL\n"); return NULL; }
    id mdl = ((id(*)(Class,SEL,id))objc_msgSend)(g_I, @selector(inMemoryModelWithDescriptor:), desc);
    id hx = ((id(*)(id,SEL))objc_msgSend)(mdl, @selector(hexStringIdentifier));
    NSString *td = [NSTemporaryDirectory() stringByAppendingPathComponent:hx];
    [[NSFileManager defaultManager] createDirectoryAtPath:[td stringByAppendingPathComponent:@"weights"] withIntermediateDirectories:YES attributes:nil error:nil];
    [md writeToFile:[td stringByAppendingPathComponent:@"model.mil"] atomically:YES];
    for (NSString *path in weights) {
        NSString *rel = [path stringByReplacingOccurrencesOfString:@"@model_path/" withString:@""];
        [weights[path][@"data"] writeToFile:[td stringByAppendingPathComponent:rel] atomically:YES];
    }
    NSError *e = nil;
    if (!((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(mdl, @selector(compileWithQoS:options:error:), 21, @{}, &e)) {
        printf("  [compile] FAIL: %s\n", e ? [[e description] UTF8String] : "no error"); return NULL;
    }
    if (!((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(mdl, @selector(loadWithQoS:options:error:), 21, @{}, &e)) {
        printf("  [compile] load FAIL\n"); return NULL;
    }
    __sync_fetch_and_add(&g_compile_count, 1);
    Kern *k = (Kern*)calloc(1, sizeof(Kern));
    k->model = (void*)CFBridgingRetain(mdl);
    k->ioIn = make_surface(ic_bytes);
    k->ioOut = make_surface(oc_bytes);
    id wI = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_AIO, @selector(objectWithIOSurface:), k->ioIn);
    id wO = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_AIO, @selector(objectWithIOSurface:), k->ioOut);
    k->request = (void*)CFBridgingRetain(((id(*)(Class,SEL,id,id,id,id,id,id,id))objc_msgSend)(g_AR,
        @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:),
        @[wI], @[@0], @[wO], @[@0], nil, nil, @0));
    k->tmpDir = (void*)CFBridgingRetain(td);
    // Get underlying _ANEModel for evaluateRealTime
    k->aneModel = (void*)CFBridgingRetain(((id(*)(id,SEL))objc_msgSend)(mdl, @selector(model)));
    return k;
    }
}
static void free_kern(Kern *k) {
    if (!k) return;
    id mdl = (__bridge id)k->model; NSError *e = nil;
    ((BOOL(*)(id,SEL,unsigned int,NSError**))objc_msgSend)(mdl, @selector(unloadWithQoS:error:), 21, &e);
    CFRelease(k->ioIn); CFRelease(k->ioOut);
    [[NSFileManager defaultManager] removeItemAtPath:(__bridge id)k->tmpDir error:nil];
    if (k->aneModel) CFRelease(k->aneModel);
    CFRelease(k->model); CFRelease(k->request); CFRelease(k->tmpDir);
    free(k);
}
// doEvaluateDirectWithModel: fastest dispatch path in real async training workload
// (evaluateRealTimeWithModel is faster in serial benchmarks but ~14% slower with async dispatch)
static void ane_eval(Kern *k) {
    id aneModel = (__bridge id)k->aneModel; id req = (__bridge id)k->request; NSError *e = nil;
    ((BOOL(*)(id,SEL,id,id,id,unsigned int,NSError**))objc_msgSend)(
        g_ane_client, @selector(doEvaluateDirectWithModel:options:request:qos:error:),
        aneModel, @{}, req, 21, &e);
}
static void ane_eval_req(Kern *k, void *request) {
    id aneModel = (__bridge id)k->aneModel; id req = (__bridge id)request; NSError *e = nil;
    BOOL ok = ((BOOL(*)(id,SEL,id,id,id,unsigned int,NSError**))objc_msgSend)(
        g_ane_client, @selector(doEvaluateDirectWithModel:options:request:qos:error:),
        aneModel, @{}, req, 21, &e);
    if (!ok) printf("  [ANE EVAL FAIL] %s\n", e ? [[e description] UTF8String] : "no error");
}
// Async ANE eval: dispatch eval to dedicated queue, overlap CPU work
static dispatch_queue_t g_ane_q = NULL;
static dispatch_queue_t ane_dispatch_queue(void) {
    if (!g_ane_q) {
        dispatch_queue_attr_t attr = dispatch_queue_attr_make_with_qos_class(
            DISPATCH_QUEUE_SERIAL, QOS_CLASS_USER_INTERACTIVE, 0);
        g_ane_q = dispatch_queue_create("ane_eval", attr);
    }
    return g_ane_q;
}

// Async eval: dispatches ANE eval to background, signals semaphore on completion.
// Call ane_eval_wait(sem) when you need the result.
static dispatch_semaphore_t ane_eval_req_async(Kern *k, void *request) {
    dispatch_semaphore_t sem = dispatch_semaphore_create(0);
    id aneModel = (__bridge id)k->aneModel;
    id req = (__bridge id)request;
    id client = g_ane_client;
    dispatch_async(ane_dispatch_queue(), ^{
        NSError *e = nil;
        ((BOOL(*)(id,SEL,id,id,id,unsigned int,NSError**))objc_msgSend)(
            client, @selector(doEvaluateDirectWithModel:options:request:qos:error:),
            aneModel, @{}, req, 21, &e);
        dispatch_semaphore_signal(sem);
    });
    return sem;
}
static dispatch_semaphore_t ane_eval_async(Kern *k) {
    return ane_eval_req_async(k, k->request);
}
static void ane_eval_wait(dispatch_semaphore_t sem) {
    dispatch_semaphore_wait(sem, DISPATCH_TIME_FOREVER);
}

// Rebind a kernel's output IOSurface (for IOSurface sharing between kernels)
// Replaces ioOut and rebuilds the default request with the new output surface
static void rebind_kern_output(Kern *k, IOSurfaceRef newOut) {
    CFRelease(k->ioOut);
    k->ioOut = (IOSurfaceRef)CFRetain(newOut);
    // Rebuild default request with new output
    CFRelease(k->request);
    id wI = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_AIO, @selector(objectWithIOSurface:), k->ioIn);
    id wO = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_AIO, @selector(objectWithIOSurface:), k->ioOut);
    k->request = (void*)CFBridgingRetain(((id(*)(Class,SEL,id,id,id,id,id,id,id))objc_msgSend)(g_AR,
        @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:),
        @[wI], @[@0], @[wO], @[@0], nil, nil, @0));
}

static void *make_request(Kern *k, IOSurfaceRef ioIn) {
    id wI = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_AIO, @selector(objectWithIOSurface:), ioIn);
    id wO = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_AIO, @selector(objectWithIOSurface:), k->ioOut);
    id req = ((id(*)(Class,SEL,id,id,id,id,id,id,id))objc_msgSend)(g_AR,
        @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:),
        @[wI], @[@0], @[wO], @[@0], nil, nil, @0);
    return (void*)CFBridgingRetain(req);
}

// ===== Per-layer weight staging for GQA =====
// sdpaFwd: [1, DIM, 1, SEQ + Q_DIM + KV_DIM + KV_DIM] fp16 — no Wo (separate kernel)
//   Wq: [DIM, Q_DIM], Wk: [DIM, KV_DIM], Wv: [DIM, KV_DIM]
#define SDPA_FWD_SP (SEQ + Q_DIM + KV_DIM + KV_DIM + Q_DIM)
static void stage_sdpa_fwd_weights(IOSurfaceRef s, const float *Wq, const float *Wk, const float *Wv, const float *Wo) {
    _Float16 *buf = (_Float16*)IOSurfaceGetBaseAddress(s);
    // Wq [DIM, Q_DIM] at sp[SEQ:SEQ+Q_DIM]
    cvt_scatter_f32_f16(buf, Wq, DIM, Q_DIM, SDPA_FWD_SP, SEQ);
    // Wk [DIM, KV_DIM] at sp[SEQ+Q_DIM:SEQ+Q_DIM+KV_DIM]
    cvt_scatter_f32_f16(buf, Wk, DIM, KV_DIM, SDPA_FWD_SP, SEQ+Q_DIM);
    // Wv [DIM, KV_DIM] at sp[SEQ+Q_DIM+KV_DIM:SEQ+Q_DIM+2*KV_DIM]
    cvt_scatter_f32_f16(buf, Wv, DIM, KV_DIM, SDPA_FWD_SP, SEQ+Q_DIM+KV_DIM);
    // Wo [DIM, Q_DIM] at sp[SEQ+Q_DIM+2*KV_DIM:SEQ+2*Q_DIM+2*KV_DIM]
    cvt_scatter_f32_f16(buf, Wo, DIM, Q_DIM, SDPA_FWD_SP, SEQ+Q_DIM+2*KV_DIM);
}
// Fused version: xnorm is fp32 (converted+scattered in one pass)
static void write_sdpa_fwd_acts(IOSurfaceRef s, const float *xnorm) {
    _Float16 *buf = (_Float16*)IOSurfaceGetBaseAddress(s);
    cvt_scatter_f32_f16_par(buf, xnorm, DIM, SEQ, SDPA_FWD_SP, 0);
}

// ffnFused: [1, DIM, 1, 2*SEQ+3*HIDDEN] fp16
#define FFN_FUSED_SP (2*SEQ + 3*HIDDEN)
static void stage_ffn_fused_weights(IOSurfaceRef s,
                                     const float *W1t, const float *W3t, const float *W2_orig) {
    _Float16 *buf = (_Float16*)IOSurfaceGetBaseAddress(s);
    cvt_scatter_f32_f16(buf, W1t, DIM, HIDDEN, FFN_FUSED_SP, 2*SEQ);
    cvt_scatter_f32_f16(buf, W3t, DIM, HIDDEN, FFN_FUSED_SP, 2*SEQ+HIDDEN);
    cvt_scatter_f32_f16(buf, W2_orig, DIM, HIDDEN, FFN_FUSED_SP, 2*SEQ+2*HIDDEN);
}
// Fused version: x2norm and x2 are fp32 (converted+scattered in one pass)
static void write_ffn_fused_acts(IOSurfaceRef s, const float *x2norm, const float *x2) {
    _Float16 *buf = (_Float16*)IOSurfaceGetBaseAddress(s);
    cvt_scatter_f32_f16_par(buf, x2norm, DIM, SEQ, FFN_FUSED_SP, 0);
    cvt_scatter_f32_f16_par(buf, x2, DIM, SEQ, FFN_FUSED_SP, SEQ);
}

// ffnBwdFull: [1, HIDDEN, 1, 3*SEQ+3*DIM] fp16 — fully fused W2^T + SiLU bwd + W1^T/W3^T
// Replaces separate ffnBwdW2t + ffnBwdFused. dsilu_raw stays on ANE.
//   sp[0:SEQ]               = dffn [DIM, SEQ] (only channels 0:DIM used)
//   sp[SEQ:SEQ+DIM]         = W2^T [HIDDEN, DIM] (transposed weight)
//   sp[SEQ+DIM:SEQ+DIM+SEQ] = h1 [HIDDEN, SEQ]
//   sp[SEQ+DIM+SEQ:SEQ+DIM+2S] = h3 [HIDDEN, SEQ]
//   sp[SEQ+DIM+2S:SEQ+2D+2S]   = W1 [HIDDEN, DIM]
//   sp[SEQ+2D+2S:SEQ+3D+2S]    = W3 [HIDDEN, DIM]
#define FFN_BWD_FULL_SP (3*SEQ + 3*DIM)
// W2t is W2^T [HIDDEN, DIM] already transposed (from transpose_weight in Adam update)
static void stage_ffn_bwd_full_weights(IOSurfaceRef s, const float *W2t, const float *W1, const float *W3) {
    _Float16 *buf = (_Float16*)IOSurfaceGetBaseAddress(s);
    // W2^T at sp[SEQ:SEQ+DIM]: W2t is already [HIDDEN, DIM] row-major
    cvt_scatter_f32_f16(buf, W2t, HIDDEN, DIM, FFN_BWD_FULL_SP, SEQ);
    // W1 at sp[SEQ+DIM+2*SEQ:SEQ+2*DIM+2*SEQ]
    cvt_scatter_f32_f16(buf, W1, HIDDEN, DIM, FFN_BWD_FULL_SP, SEQ+DIM+2*SEQ);
    // W3 at sp[SEQ+2*DIM+2*SEQ:SEQ+3*DIM+2*SEQ]
    cvt_scatter_f32_f16(buf, W3, HIDDEN, DIM, FFN_BWD_FULL_SP, SEQ+2*DIM+2*SEQ);
}
// Full version: dffn + h1 + h3 (used when h1/h3 not pre-staged)
static void write_ffn_bwd_full_acts(IOSurfaceRef s, const float *dffn,
                                     const _Float16 *h1_fp16, const _Float16 *h3_fp16) {
    _Float16 *buf = (_Float16*)IOSurfaceGetBaseAddress(s);
    cvt_scatter_f32_f16(buf, dffn, DIM, SEQ, FFN_BWD_FULL_SP, 0);
    for (int h = 0; h < HIDDEN; h++)
        memcpy(buf + h*FFN_BWD_FULL_SP + SEQ+DIM, h1_fp16 + h*SEQ, SEQ*2);
    for (int h = 0; h < HIDDEN; h++)
        memcpy(buf + h*FFN_BWD_FULL_SP + SEQ+DIM+SEQ, h3_fp16 + h*SEQ, SEQ*2);
}
// dffn-only version: h1/h3 already pre-staged during forward pass
static void write_ffn_bwd_dffn_only(IOSurfaceRef s, const float *dffn) {
    _Float16 *buf = (_Float16*)IOSurfaceGetBaseAddress(s);
    cvt_scatter_f32_f16_par(buf, dffn, DIM, SEQ, FFN_BWD_FULL_SP, 0);
}
// Pre-stage h1/h3 into ffnBwdFull input from ffnFused output (called async during forward)
static void prestage_ffn_bwd_h1h3(IOSurfaceRef bwd_in, const _Float16 *ffn_out) {
    _Float16 *buf = (_Float16*)IOSurfaceGetBaseAddress(bwd_in);
    const _Float16 *h1_src = ffn_out + DIM*SEQ;
    const _Float16 *h3_src = ffn_out + (DIM+HIDDEN)*SEQ;
    for (int h = 0; h < HIDDEN; h++) {
        memcpy(buf + h*FFN_BWD_FULL_SP + SEQ+DIM, h1_src + h*SEQ, SEQ*2);
        memcpy(buf + h*FFN_BWD_FULL_SP + SEQ+DIM+SEQ, h3_src + h*SEQ, SEQ*2);
    }
}
// wotBwd: [1, DIM, 1, SEQ+Q_DIM] fp16 — Wo is [DIM, Q_DIM], matmul gives Wo^T @ dy
#define WOT_BWD_SP (SEQ + Q_DIM)
static void stage_wot_bwd_weights(IOSurfaceRef s, const float *Wo) {
    _Float16 *buf = (_Float16*)IOSurfaceGetBaseAddress(s);
    cvt_scatter_f32_f16(buf, Wo, DIM, Q_DIM, WOT_BWD_SP, SEQ);
}
// Fused version: dy is fp32 (converted+scattered in one pass)
static void write_wot_bwd_acts(IOSurfaceRef s, const float *dy) {
    _Float16 *buf = (_Float16*)IOSurfaceGetBaseAddress(s);
    cvt_scatter_f32_f16_par(buf, dy, DIM, SEQ, WOT_BWD_SP, 0);
}

#define WOT_SDPA_BWD1_SP (4*SEQ + Q_DIM)
// wotSdpaBwd1 FUSED: [1, Q_DIM, 1, 4*SEQ+Q_DIM] fp16
// Wo weight at sp[SEQ:SEQ+Q_DIM] (first DIM channels), pre-staged once
// Per-step: dx2_scaled at sp[0:SEQ], Q/K/V at sp[SEQ+Q_DIM:4*SEQ+Q_DIM]
static void stage_wot_sdpa_bwd1_weights(IOSurfaceRef s, const float *Wo) {
    _Float16 *buf = (_Float16*)IOSurfaceGetBaseAddress(s);
    // Wo [DIM, Q_DIM] at sp[SEQ:SEQ+Q_DIM], first DIM channels
    cvt_scatter_f32_f16(buf, Wo, DIM, Q_DIM, WOT_SDPA_BWD1_SP, SEQ);
}
// Write dx2_scaled activation for the wotBwd part
static void write_wot_sdpa_bwd1_acts(IOSurfaceRef s, const float *dx2_scaled) {
    _Float16 *buf = (_Float16*)IOSurfaceGetBaseAddress(s);
    // dx2_scaled [DIM, SEQ] at sp[0:SEQ], first DIM channels
    cvt_scatter_f32_f16_par(buf, dx2_scaled, DIM, SEQ, WOT_SDPA_BWD1_SP, 0);
}
// Pre-stage Q/K_tiled/V_tiled (called during overlap with previous ANE kernel)
static void prestage_wot_sdpa_bwd1_qkv(IOSurfaceRef s,
                                         const _Float16 *Q_fp16,
                                         const _Float16 *k_tiled_fp16,
                                         const _Float16 *v_tiled_fp16) {
    _Float16 *buf = (_Float16*)IOSurfaceGetBaseAddress(s);
    int sp = WOT_SDPA_BWD1_SP;
    // Q at sp[SEQ+Q_DIM:2*SEQ+Q_DIM] — all Q_DIM channels
    for (int ch = 0; ch < Q_DIM; ch++)
        memcpy(buf + ch*sp + SEQ+Q_DIM, Q_fp16 + ch*SEQ, SEQ*2);
    // K_tiled at sp[2*SEQ+Q_DIM:3*SEQ+Q_DIM]
    for (int ch = 0; ch < Q_DIM; ch++)
        memcpy(buf + ch*sp + 2*SEQ+Q_DIM, k_tiled_fp16 + ch*SEQ, SEQ*2);
    // V_tiled at sp[3*SEQ+Q_DIM:4*SEQ+Q_DIM]
    for (int ch = 0; ch < Q_DIM; ch++)
        memcpy(buf + ch*sp + 3*SEQ+Q_DIM, v_tiled_fp16 + ch*SEQ, SEQ*2);
}

// qkvBwd: fused dq@Wq + dk@Wk + dv@Wv → dx_attn (single kernel, single output)
// Input: [1, Q_DIM, 1, 3*SEQ+3*DIM] fp16
//   sp[0:SEQ]                    = dq [Q_DIM, SEQ]
//   sp[SEQ:SEQ+DIM]              = Wq [Q_DIM, DIM] (pre-staged)
//   sp[SEQ+DIM:SEQ+DIM+SEQ]      = dk [KV_DIM, SEQ] (channels 0:KV_DIM only)
//   sp[SEQ+DIM+SEQ:SEQ+DIM+2S]   = dv [KV_DIM, SEQ] (channels 0:KV_DIM only)
//   sp[SEQ+DIM+2S:SEQ+2D+2S]     = Wk [KV_DIM, DIM] (pre-staged)
//   sp[SEQ+2D+2S:SEQ+3D+2S]      = Wv [KV_DIM, DIM] (pre-staged)
// Output: [1, DIM, 1, SEQ] fp16 = dx_q + dx_k + dx_v (summed)
#define QKV_BWD_SP (3*SEQ + 3*DIM)
static void stage_qkv_bwd_weights(IOSurfaceRef s, const float *Wq, const float *Wk, const float *Wv) {
    _Float16 *buf = (_Float16*)IOSurfaceGetBaseAddress(s);
    cvt_scatter_f32_f16(buf, Wq, Q_DIM, DIM, QKV_BWD_SP, SEQ);
    cvt_scatter_f32_f16(buf, Wk, KV_DIM, DIM, QKV_BWD_SP, SEQ+DIM+2*SEQ);
    cvt_scatter_f32_f16(buf, Wv, KV_DIM, DIM, QKV_BWD_SP, SEQ+2*DIM+2*SEQ);
}
// Pre-stage dv into qkvBwd input (called during sdpaBwd2 ANE overlap)
static void prestage_qkv_bwd_dv(IOSurfaceRef s, const float *dv) {
    _Float16 *buf = (_Float16*)IOSurfaceGetBaseAddress(s);
    cvt_scatter_f32_f16_par(buf, dv, KV_DIM, SEQ, QKV_BWD_SP, SEQ+DIM+SEQ);
}

// Free per-layer surfaces and requests
static void free_per_layer(PerLayerSurfaces *pls, PerLayerRequests *plr) {
    for (int L = 0; L < NLAYERS; L++) {
        CFRelease(pls[L].sdpaFwd_in); CFRelease(pls[L].ffnFused_in);
        CFRelease(pls[L].ffnBwdFull_in); CFRelease(pls[L].wotBwd_in); CFRelease(pls[L].qkvBwd_in);
        CFRelease(plr[L].sdpaFwd); CFRelease(plr[L].ffnFused);
        CFRelease(plr[L].ffnBwdFull); CFRelease(plr[L].wotBwd); CFRelease(plr[L].qkvBwd);
    }
}

// GQA helpers: tile KV from KV_HEADS to HEADS, and reduce HEADS to KV_HEADS
// tile_kv_fp16: input [KV_DIM, SEQ], output [Q_DIM, SEQ] in fp16
static void gqa_tile_kv_fp16(_Float16 *out, const _Float16 *in, int seq) {
    for (int kv = 0; kv < KV_HEADS; kv++) {
        for (int r = 0; r < GQA_RATIO; r++) {
            int q_head = kv * GQA_RATIO + r;
            memcpy(out + q_head * HD * seq, in + kv * HD * seq, HD * seq * 2);
        }
    }
}
// reduce_kv: input [Q_DIM, SEQ], output [KV_DIM, SEQ]
// Sum contributions from Q heads sharing each KV head
static void gqa_reduce_kv(float *out, const float *in, int seq) {
    memset(out, 0, KV_DIM * seq * sizeof(float));
    for (int kv = 0; kv < KV_HEADS; kv++) {
        for (int r = 0; r < GQA_RATIO; r++) {
            int q_head = kv * GQA_RATIO + r;
            const float *src = in + q_head * HD * seq;
            float *dst = out + kv * HD * seq;
            vDSP_vadd(src, 1, dst, 1, dst, 1, (vDSP_Length)(HD * seq));
        }
    }
}
