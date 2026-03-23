// cpu_ops.h — CPU operations: RMSNorm, cross-entropy, Adam, embedding
#pragma once
#include "config.h"

static float *g_rms_tmp = NULL;
static float *g_rms_ss = NULL, *g_rms_rrms = NULL, *g_rms_dot = NULL;
static int g_rms_S = 0;

static void rms_ensure_bufs(int S) {
    if (g_rms_S >= S) return;
    free(g_rms_tmp); free(g_rms_ss); free(g_rms_rrms); free(g_rms_dot);
    g_rms_tmp  = (float*)malloc(S*4);
    g_rms_ss   = (float*)malloc(S*4);
    g_rms_rrms = (float*)malloc(S*4);
    g_rms_dot  = (float*)malloc(S*4);
    g_rms_S = S;
}

// NEON-optimized RMSNorm forward — eliminates ~4K vDSP calls per invocation
static void rmsnorm(float *out, const float *x, const float *w, int d, int S) {
    rms_ensure_bufs(S);
    float *ss = g_rms_ss;

    // Accumulate sum of squares per position: ss[s] = sum_i(x[i,s]^2) — 8-wide
    memset(ss, 0, S*4);
    for (int i = 0; i < d; i++) {
        const float *xi = x + i*S;
        int s = 0;
        for (; s + 7 < S; s += 8) {
            float32x4_t xv0 = vld1q_f32(xi + s), xv1 = vld1q_f32(xi + s + 4);
            vst1q_f32(ss + s,     vfmaq_f32(vld1q_f32(ss + s),     xv0, xv0));
            vst1q_f32(ss + s + 4, vfmaq_f32(vld1q_f32(ss + s + 4), xv1, xv1));
        }
        for (; s + 3 < S; s += 4) {
            float32x4_t xv = vld1q_f32(xi + s);
            vst1q_f32(ss + s, vfmaq_f32(vld1q_f32(ss + s), xv, xv));
        }
        for (; s < S; s++) ss[s] += xi[s] * xi[s];
    }

    // ss = rsqrt(ss / d + eps)
    float invd = 1.0f/d, eps = 1e-5f;
    vDSP_vsmsa(ss, 1, &invd, &eps, ss, 1, (vDSP_Length)S);
    int n = S; vvrsqrtf(ss, ss, &n);

    // out[i,s] = x[i,s] * ss[s] * w[i] — 8-wide
    for (int i = 0; i < d; i++) {
        const float *xi = x + i*S;
        float *oi = out + i*S;
        float32x4_t wi = vdupq_n_f32(w[i]);
        int s = 0;
        for (; s + 7 < S; s += 8) {
            vst1q_f32(oi + s,     vmulq_f32(vmulq_f32(vld1q_f32(xi + s),     vld1q_f32(ss + s)),     wi));
            vst1q_f32(oi + s + 4, vmulq_f32(vmulq_f32(vld1q_f32(xi + s + 4), vld1q_f32(ss + s + 4)), wi));
        }
        for (; s + 3 < S; s += 4)
            vst1q_f32(oi + s, vmulq_f32(vmulq_f32(vld1q_f32(xi + s), vld1q_f32(ss + s)), wi));
        for (; s < S; s++) oi[s] = xi[s] * ss[s] * w[i];
    }
}

// Fused RMSNorm + fp16 scatter: writes fp32 output AND fp16 to IOSurface in single pass.
// Eliminates separate write_*_fwd_acts scatter (saves one full DIM*SEQ read+convert pass).
static void rmsnorm_scatter(float *out, const float *x, const float *w, int d, int S,
                             _Float16 *ios, int ios_stride, int ios_sp_off) {
    rms_ensure_bufs(S);
    float *ss = g_rms_ss;
    memset(ss, 0, S*4);
    for (int i = 0; i < d; i++) {
        const float *xi = x + i*S;
        int s = 0;
        for (; s + 7 < S; s += 8) {
            float32x4_t xv0 = vld1q_f32(xi + s), xv1 = vld1q_f32(xi + s + 4);
            vst1q_f32(ss + s,     vfmaq_f32(vld1q_f32(ss + s),     xv0, xv0));
            vst1q_f32(ss + s + 4, vfmaq_f32(vld1q_f32(ss + s + 4), xv1, xv1));
        }
        for (; s + 3 < S; s += 4) {
            float32x4_t xv = vld1q_f32(xi + s);
            vst1q_f32(ss + s, vfmaq_f32(vld1q_f32(ss + s), xv, xv));
        }
        for (; s < S; s++) ss[s] += xi[s] * xi[s];
    }
    float invd = 1.0f/d, eps = 1e-5f;
    vDSP_vsmsa(ss, 1, &invd, &eps, ss, 1, (vDSP_Length)S);
    int n = S; vvrsqrtf(ss, ss, &n);
    for (int i = 0; i < d; i++) {
        const float *xi = x + i*S;
        float *oi = out + i*S;
        _Float16 *di = ios + i*ios_stride + ios_sp_off;
        float32x4_t wi = vdupq_n_f32(w[i]);
        int s = 0;
        for (; s + 7 < S; s += 8) {
            float32x4_t r0 = vmulq_f32(vmulq_f32(vld1q_f32(xi + s),     vld1q_f32(ss + s)),     wi);
            float32x4_t r1 = vmulq_f32(vmulq_f32(vld1q_f32(xi + s + 4), vld1q_f32(ss + s + 4)), wi);
            vst1q_f32(oi + s, r0);
            vst1q_f32(oi + s + 4, r1);
            vst1q_f16((__fp16*)(di + s), vcombine_f16(vcvt_f16_f32(r0), vcvt_f16_f32(r1)));
        }
        for (; s < S; s++) {
            float v = xi[s] * ss[s] * w[i];
            oi[s] = v;
            di[s] = (_Float16)v;
        }
    }
}

// NEON-optimized RMSNorm backward — eliminates ~11K vDSP calls per invocation
static void rmsnorm_bwd(float *dx, float *dw, const float *dy, const float *x, const float *w, int d, int S) {
    rms_ensure_bufs(S);
    float *ss = g_rms_ss;
    float *rrms = g_rms_rrms;
    float *dot = g_rms_dot;

    // Fused loop 1+2: ss[s] = sum_i(x[i,s]^2), dot[s] = sum_i(dy[i,s]*x[i,s]*w[i])
    // Single pass over x saves 1MB/call of L2 traffic (56 calls/step = 56MB saved)
    memset(ss, 0, S*4);
    memset(dot, 0, S*4);
    for (int i = 0; i < d; i++) {
        const float *xi = x + i*S;
        const float *dyi = dy + i*S;
        float32x4_t wi = vdupq_n_f32(w[i]);
        int s = 0;
        for (; s + 7 < S; s += 8) {
            float32x4_t xv0 = vld1q_f32(xi + s), xv1 = vld1q_f32(xi + s + 4);
            vst1q_f32(ss + s,     vfmaq_f32(vld1q_f32(ss + s),     xv0, xv0));
            vst1q_f32(ss + s + 4, vfmaq_f32(vld1q_f32(ss + s + 4), xv1, xv1));
            vst1q_f32(dot + s,     vfmaq_f32(vld1q_f32(dot + s),     vmulq_f32(vld1q_f32(dyi + s),     xv0), wi));
            vst1q_f32(dot + s + 4, vfmaq_f32(vld1q_f32(dot + s + 4), vmulq_f32(vld1q_f32(dyi + s + 4), xv1), wi));
        }
        for (; s + 3 < S; s += 4) {
            float32x4_t xv = vld1q_f32(xi + s);
            vst1q_f32(ss + s, vfmaq_f32(vld1q_f32(ss + s), xv, xv));
            vst1q_f32(dot + s, vfmaq_f32(vld1q_f32(dot + s), vmulq_f32(vld1q_f32(dyi + s), xv), wi));
        }
        for (; s < S; s++) {
            ss[s] += xi[s] * xi[s];
            dot[s] += dyi[s] * xi[s] * w[i];
        }
    }

    // ss = ss / d + eps, then rrms = rsqrt(ss)
    float invd = 1.0f/d, eps = 1e-5f;
    vDSP_vsmsa(ss, 1, &invd, &eps, ss, 1, (vDSP_Length)S);
    int n = S; vvrsqrtf(rrms, ss, &n);

    // dot = dot * rrms^2 / d  (precompute common factor)
    vDSP_vmul(rrms, 1, rrms, 1, ss, 1, (vDSP_Length)S);
    vDSP_vsmul(ss, 1, &invd, ss, 1, (vDSP_Length)S);
    vDSP_vmul(dot, 1, ss, 1, dot, 1, (vDSP_Length)S);

    // Loop 3: per-channel gradient computation — 8-wide NEON unrolled
    // dx[i,s] = (dy[i,s] - x[i,s] * dot[s]) * rrms[s] * w[i]
    // dw[i] += sum_s(dy[i,s] * x[i,s] * rrms[s])
    for (int i = 0; i < d; i++) {
        const float *dyi = dy + i*S;
        const float *xi = x + i*S;
        float *dxi = dx + i*S;
        float32x4_t wi = vdupq_n_f32(w[i]);
        float32x4_t acc0 = vdupq_n_f32(0.0f);
        float32x4_t acc1 = vdupq_n_f32(0.0f);
        int s = 0;
        for (; s + 7 < S; s += 8) {
            float32x4_t xv0 = vld1q_f32(xi + s);
            float32x4_t xv1 = vld1q_f32(xi + s + 4);
            float32x4_t dyv0 = vld1q_f32(dyi + s);
            float32x4_t dyv1 = vld1q_f32(dyi + s + 4);
            float32x4_t rv0 = vld1q_f32(rrms + s);
            float32x4_t rv1 = vld1q_f32(rrms + s + 4);
            float32x4_t dv0 = vld1q_f32(dot + s);
            float32x4_t dv1 = vld1q_f32(dot + s + 4);
            float32x4_t t0 = vmulq_f32(vfmsq_f32(dyv0, xv0, dv0), rv0);
            float32x4_t t1 = vmulq_f32(vfmsq_f32(dyv1, xv1, dv1), rv1);
            vst1q_f32(dxi + s,     vmulq_f32(t0, wi));
            vst1q_f32(dxi + s + 4, vmulq_f32(t1, wi));
            acc0 = vfmaq_f32(acc0, vmulq_f32(dyv0, xv0), rv0);
            acc1 = vfmaq_f32(acc1, vmulq_f32(dyv1, xv1), rv1);
        }
        for (; s + 3 < S; s += 4) {
            float32x4_t xv = vld1q_f32(xi + s);
            float32x4_t dyv = vld1q_f32(dyi + s);
            float32x4_t rv = vld1q_f32(rrms + s);
            float32x4_t dv = vld1q_f32(dot + s);
            vst1q_f32(dxi + s, vmulq_f32(vmulq_f32(vfmsq_f32(dyv, xv, dv), rv), wi));
            acc0 = vfmaq_f32(acc0, vmulq_f32(dyv, xv), rv);
        }
        float dw_sum = vaddvq_f32(vaddq_f32(acc0, acc1));
        for (; s < S; s++) {
            float tmp = (dyi[s] - xi[s] * dot[s]) * rrms[s];
            dxi[s] = tmp * w[i];
            dw_sum += dyi[s] * xi[s] * rrms[s];
        }
        dw[i] += dw_sum;
    }
}

static float *g_adam_tmp1 = NULL, *g_adam_tmp2 = NULL;
static size_t g_adam_sz = 0;

static void adam_update(float *w, const float *g, AdamState *s, int t, float lr, float b1, float b2, float eps, float wd) {
    size_t n = s->n;
    if (n > g_adam_sz) {
        free(g_adam_tmp1); free(g_adam_tmp2);
        g_adam_tmp1 = (float*)malloc(n * 4);
        g_adam_tmp2 = (float*)malloc(n * 4);
        g_adam_sz = n;
    }
    float bc1 = 1.0f - powf(b1, t), bc2 = 1.0f - powf(b2, t);
    float ob1 = 1.0f - b1, ob2 = 1.0f - b2;
    float inv_bc1 = 1.0f / bc1, inv_bc2 = 1.0f / bc2;
    float neg_lr = -lr;
    vDSP_Length vn = (vDSP_Length)n;

    // m = b1*m + (1-b1)*g
    vDSP_vsmul(s->m, 1, &b1, s->m, 1, vn);
    vDSP_vsma(g, 1, &ob1, s->m, 1, s->m, 1, vn);

    // v = b2*v + (1-b2)*g*g
    vDSP_vsmul(s->v, 1, &b2, s->v, 1, vn);
    vDSP_vmul(g, 1, g, 1, g_adam_tmp1, 1, vn);
    vDSP_vsma(g_adam_tmp1, 1, &ob2, s->v, 1, s->v, 1, vn);

    // mh = m / bc1
    vDSP_vsmul(s->m, 1, &inv_bc1, g_adam_tmp1, 1, vn);

    // vh = v / bc2, then sqrt(vh) + eps
    vDSP_vsmul(s->v, 1, &inv_bc2, g_adam_tmp2, 1, vn);
    int nn = (int)n; vvsqrtf(g_adam_tmp2, g_adam_tmp2, &nn);
    vDSP_vsadd(g_adam_tmp2, 1, &eps, g_adam_tmp2, 1, vn);

    // update = mh / (sqrt(vh) + eps) + wd * w
    vDSP_vdiv(g_adam_tmp2, 1, g_adam_tmp1, 1, g_adam_tmp1, 1, vn);
    vDSP_vsma(w, 1, &wd, g_adam_tmp1, 1, g_adam_tmp1, 1, vn);

    // w -= lr * update
    vDSP_vsma(g_adam_tmp1, 1, &neg_lr, w, 1, w, 1, vn);
}

// Cross-entropy loss: operates on logits[S, V] row-major (each row = one token, contiguous)
// Eliminates strided gather/scatter — each token's V logits are contiguous in memory.
// grad_scale folds loss_scale into gradient computation (avoids separate vDSP_vsmul pass).
// Parallelized across tokens with dispatch_apply.
static float cross_entropy_loss(float *dlogits, const float *logits, const uint16_t *targets, int V, int S, float grad_scale) {
    float *losses = (float*)alloca(S * 4);
    float invS = grad_scale / S;  // folds loss_scale into gradient normalization

    dispatch_apply((size_t)S, dispatch_get_global_queue(QOS_CLASS_USER_INTERACTIVE, 0), ^(size_t t) {
        const float *row_in = logits + t * V;
        float *row_out = dlogits + t * V;
        // Copy to output buffer only if not aliased (in-place mode skips this)
        if (row_out != row_in) memcpy(row_out, row_in, V * 4);
        // Softmax
        float maxv; vDSP_maxv(row_out, 1, &maxv, (vDSP_Length)V);
        float neg_max = -maxv;
        vDSP_vsadd(row_out, 1, &neg_max, row_out, 1, (vDSP_Length)V);
        int n = V; vvexpf(row_out, row_out, &n);
        float sum; vDSP_sve(row_out, 1, &sum, (vDSP_Length)V);
        float inv_sum = 1.0f / sum;
        vDSP_vsmul(row_out, 1, &inv_sum, row_out, 1, (vDSP_Length)V);
        // Loss + gradient
        int tgt = targets[t];
        losses[t] = -logf(row_out[tgt] + 1e-10f);
        row_out[tgt] -= 1.0f;
        vDSP_vsmul(row_out, 1, &invS, row_out, 1, (vDSP_Length)V);
    });

    float total_loss;
    vDSP_sve(losses, 1, &total_loss, (vDSP_Length)S);
    return total_loss / S;
}

// Vocab compaction: build mapping from full 32K vocab to compact vocab
typedef struct {
    int compact_vocab;          // number of active tokens
    int *full_to_compact;       // [VOCAB] → compact id (-1 if unused)
    int *compact_to_full;       // [compact_vocab] → full vocab id
} VocabMap;

static VocabMap vocab_map_build(const uint16_t *data, size_t n_tokens, int full_vocab) {
    VocabMap vm;
    vm.full_to_compact = (int*)malloc(full_vocab * sizeof(int));
    memset(vm.full_to_compact, -1, full_vocab * sizeof(int));
    // Scan for used tokens
    for (size_t i = 0; i < n_tokens; i++) {
        vm.full_to_compact[data[i]] = 0;  // mark as used
    }
    // Assign compact IDs
    int cid = 0;
    for (int v = 0; v < full_vocab; v++) {
        if (vm.full_to_compact[v] == 0)
            vm.full_to_compact[v] = cid++;
        else
            vm.full_to_compact[v] = -1;
    }
    vm.compact_vocab = cid;
    vm.compact_to_full = (int*)malloc(cid * sizeof(int));
    for (int v = 0; v < full_vocab; v++) {
        if (vm.full_to_compact[v] >= 0)
            vm.compact_to_full[vm.full_to_compact[v]] = v;
    }
    return vm;
}

// Create compact embedding from full embedding
static float *vocab_compact_embed(const float *full_embed, const VocabMap *vm, int dim) {
    float *ce = (float*)malloc((size_t)vm->compact_vocab * dim * 4);
    for (int c = 0; c < vm->compact_vocab; c++)
        memcpy(ce + c*dim, full_embed + vm->compact_to_full[c]*dim, dim*4);
    return ce;
}

// Scatter compact embed gradients back to full embed
static void vocab_scatter_grads(float *full_gembed, const float *compact_gembed, const VocabMap *vm, int dim) {
    float one = 1.0f;
    for (int c = 0; c < vm->compact_vocab; c++) {
        int fv = vm->compact_to_full[c];
        vDSP_vsma(compact_gembed + c*dim, 1, &one, full_gembed + fv*dim, 1, full_gembed + fv*dim, 1, (vDSP_Length)dim);
    }
}

static void embed_lookup(float *x, const float *embed, const uint16_t *tokens, int dim, int seq) {
    for (int t = 0; t < seq; t++) {
        int tok = tokens[t];
        for (int d = 0; d < dim; d++)
            x[d*seq + t] = embed[tok*dim + d];
    }
}

static void embed_backward(float *d_embed, const float *dx, const uint16_t *tokens, int dim, int seq) {
    for (int t = 0; t < seq; t++) {
        int tok = tokens[t];
        for (int d = 0; d < dim; d++)
            d_embed[tok*dim + d] += dx[d*seq + t];
    }
}

// RoPE backward (in-place): inverse rotation on dQ/dK gradients
// Data layout: [DIM, SEQ] channel-first, DIM = nheads * hd
// Precomputed cos/sin tables + vDSP vectorized rotation
static float *g_rope_cos = NULL, *g_rope_sin = NULL;
static _Float16 *g_rope_cos_f16 = NULL, *g_rope_sin_f16 = NULL;
static int g_rope_hd = 0, g_rope_seq = 0;

static void rope_ensure_table(int hd, int seq) {
    if (g_rope_hd == hd && g_rope_seq == seq) return;
    free(g_rope_cos); free(g_rope_sin);
    free(g_rope_cos_f16); free(g_rope_sin_f16);
    int table_sz = (hd / 2) * seq;
    g_rope_cos = (float*)malloc(table_sz * 4);
    g_rope_sin = (float*)malloc(table_sz * 4);
    g_rope_cos_f16 = (_Float16*)malloc(table_sz * 2);
    g_rope_sin_f16 = (_Float16*)malloc(table_sz * 2);
    for (int i = 0; i < hd / 2; i++) {
        float freq = 1.0f / powf(10000.0f, 2.0f * i / (float)hd);
        for (int p = 0; p < seq; p++) {
            float theta = p * freq;
            __sincosf(theta, &g_rope_sin[i * seq + p], &g_rope_cos[i * seq + p]);
            g_rope_cos_f16[i * seq + p] = (_Float16)g_rope_cos[i * seq + p];
            g_rope_sin_f16[i * seq + p] = (_Float16)g_rope_sin[i * seq + p];
        }
    }
    g_rope_hd = hd; g_rope_seq = seq;
}

// NEON-optimized RoPE backward: direct FMA, no vDSP overhead
// Backward rotation: (v0,v1) → (v0*cos + v1*sin, v1*cos - v0*sin)
static void rope_backward_inplace(float *dx, int seq, int dim, int hd) {
    rope_ensure_table(hd, seq);
    int nheads = dim / hd;
    for (int h = 0; h < nheads; h++) {
        for (int i = 0; i < hd / 2; i++) {
            float *v0 = dx + (h * hd + 2 * i) * seq;
            float *v1 = dx + (h * hd + 2 * i + 1) * seq;
            const float *cos_t = g_rope_cos + i * seq;
            const float *sin_t = g_rope_sin + i * seq;
            int s = 0;
            for (; s + 3 < seq; s += 4) {
                float32x4_t c = vld1q_f32(cos_t + s);
                float32x4_t sn = vld1q_f32(sin_t + s);
                float32x4_t x0 = vld1q_f32(v0 + s);
                float32x4_t x1 = vld1q_f32(v1 + s);
                // new_v0 = x0*c + x1*sn
                vst1q_f32(v0 + s, vfmaq_f32(vmulq_f32(x0, c), x1, sn));
                // new_v1 = x1*c - x0*sn
                vst1q_f32(v1 + s, vfmsq_f32(vmulq_f32(x1, c), x0, sn));
            }
            for (; s < seq; s++) {
                float x0 = v0[s], x1 = v1[s];
                v0[s] = x0 * cos_t[s] + x1 * sin_t[s];
                v1[s] = x1 * cos_t[s] - x0 * sin_t[s];
            }
        }
    }
}

// fp16 RoPE backward: same rotation, native fp16 NEON (8-wide, no conversion)
// Data layout: [DIM, SEQ] channel-first, DIM = nheads * hd
static void rope_backward_inplace_f16(_Float16 *dx, int seq, int dim, int hd) {
    rope_ensure_table(hd, seq);
    int nheads = dim / hd;
    for (int h = 0; h < nheads; h++) {
        for (int i = 0; i < hd / 2; i++) {
            _Float16 *v0 = dx + (h * hd + 2 * i) * seq;
            _Float16 *v1 = dx + (h * hd + 2 * i + 1) * seq;
            const _Float16 *cos_t = g_rope_cos_f16 + i * seq;
            const _Float16 *sin_t = g_rope_sin_f16 + i * seq;
            int s = 0;
            for (; s + 7 < seq; s += 8) {
                float16x8_t c = vld1q_f16((const __fp16*)(cos_t + s));
                float16x8_t sn = vld1q_f16((const __fp16*)(sin_t + s));
                float16x8_t x0 = vld1q_f16((const __fp16*)(v0 + s));
                float16x8_t x1 = vld1q_f16((const __fp16*)(v1 + s));
                // new_v0 = x0*c + x1*sn
                vst1q_f16((__fp16*)(v0 + s), vfmaq_f16(vmulq_f16(x0, c), x1, sn));
                // new_v1 = x1*c - x0*sn
                vst1q_f16((__fp16*)(v1 + s), vfmsq_f16(vmulq_f16(x1, c), x0, sn));
            }
            for (; s < seq; s++) {
                _Float16 x0 = v0[s], x1 = v1[s];
                v0[s] = x0 * cos_t[s] + x1 * sin_t[s];
                v1[s] = x1 * cos_t[s] - x0 * sin_t[s];
            }
        }
    }
}

// Fused RoPE backward + scatter to IOSurface: computes RoPE rotation in fp16,
// writes result to BOTH src buffer (in-place, for fp32 conversion in overlap window)
// AND IOSurface destination (strided, for ANE input staging).
// Eliminates standalone scatter pass by combining data traversals.
static void rope_backward_scatter_f16(_Float16 *dx, int seq, int dim, int hd,
                                       _Float16 *ios_dst, int ios_stride, int ios_sp_offset) {
    rope_ensure_table(hd, seq);
    int nheads = dim / hd;
    for (int h = 0; h < nheads; h++) {
        for (int i = 0; i < hd / 2; i++) {
            int ch0 = h * hd + 2 * i;
            int ch1 = ch0 + 1;
            _Float16 *v0 = dx + ch0 * seq;
            _Float16 *v1 = dx + ch1 * seq;
            _Float16 *d0 = ios_dst + ch0 * ios_stride + ios_sp_offset;
            _Float16 *d1 = ios_dst + ch1 * ios_stride + ios_sp_offset;
            const _Float16 *cos_t = g_rope_cos_f16 + i * seq;
            const _Float16 *sin_t = g_rope_sin_f16 + i * seq;
            int s = 0;
            for (; s + 7 < seq; s += 8) {
                float16x8_t c = vld1q_f16((const __fp16*)(cos_t + s));
                float16x8_t sn = vld1q_f16((const __fp16*)(sin_t + s));
                float16x8_t x0 = vld1q_f16((const __fp16*)(v0 + s));
                float16x8_t x1 = vld1q_f16((const __fp16*)(v1 + s));
                float16x8_t r0 = vfmaq_f16(vmulq_f16(x0, c), x1, sn);
                float16x8_t r1 = vfmsq_f16(vmulq_f16(x1, c), x0, sn);
                // Write to contiguous source (for fp32 conversion in overlap)
                vst1q_f16((__fp16*)(v0 + s), r0);
                vst1q_f16((__fp16*)(v1 + s), r1);
                // Write to IOSurface (strided, ANE input staging)
                vst1q_f16((__fp16*)(d0 + s), r0);
                vst1q_f16((__fp16*)(d1 + s), r1);
            }
            for (; s < seq; s++) {
                _Float16 x0v = v0[s], x1v = v1[s];
                _Float16 r0 = x0v * cos_t[s] + x1v * sin_t[s];
                _Float16 r1 = x1v * cos_t[s] - x0v * sin_t[s];
                v0[s] = r0; v1[s] = r1;
                d0[s] = r0; d1[s] = r1;
            }
        }
    }
}

// fp16 GQA reduce: sum GQA_RATIO query heads per KV head, fp16 in/out
// in: [Q_DIM, seq] fp16, out: [KV_DIM, seq] fp16
static void gqa_reduce_kv_f16(_Float16 *out, const _Float16 *in, int seq) {
    for (int kv = 0; kv < KV_HEADS; kv++) {
        for (int d = 0; d < HD; d++) {
            _Float16 *dst = out + (kv * HD + d) * seq;
            const _Float16 *src0 = in + (kv * GQA_RATIO * HD + d) * seq;
            // First contribution: copy
            memcpy(dst, src0, seq * 2);
            // Sum remaining GQA_RATIO-1 contributions
            for (int r = 1; r < GQA_RATIO; r++) {
                const _Float16 *srci = in + ((kv * GQA_RATIO + r) * HD + d) * seq;
                int s = 0;
                for (; s + 7 < seq; s += 8) {
                    float16x8_t a = vld1q_f16((const __fp16*)(dst + s));
                    float16x8_t b = vld1q_f16((const __fp16*)(srci + s));
                    vst1q_f16((__fp16*)(dst + s), vaddq_f16(a, b));
                }
                for (; s < seq; s++) dst[s] += srci[s];
            }
        }
    }
}

// Parallel fp16 strided scatter with non-temporal stores: multi-threaded STNP
// bypasses read-for-ownership cache miss on write-only IOSurface data
static void scatter_f16_par(_Float16 *dst, const _Float16 *src,
                            int channels, int seq, int stride, int sp_offset) {
    if (!g_scatter_q) g_scatter_q = dispatch_get_global_queue(QOS_CLASS_USER_INTERACTIVE, 0);
    const int BLOCK = 64;
    int nblocks = (channels + BLOCK - 1) / BLOCK;
    int bytes = seq * 2;
    dispatch_apply((size_t)nblocks, g_scatter_q, ^(size_t b) {
        int ch_start = (int)b * BLOCK;
        int ch_end = ch_start + BLOCK;
        if (ch_end > channels) ch_end = channels;
        for (int ch = ch_start; ch < ch_end; ch++) {
            const uint8_t *s = (const uint8_t*)(src + ch * seq);
            uint8_t *d = (uint8_t*)(dst + ch * stride + sp_offset);
            int i = 0;
            for (; i + 31 < bytes; i += 32) {
                __asm__ volatile (
                    "ldp q0, q1, [%0]   \n"
                    "stnp q0, q1, [%1]  \n"
                    : : "r"(s + i), "r"(d + i)
                    : "v0", "v1", "memory"
                );
            }
            for (; i < bytes; i++) d[i] = s[i];
        }
    });
}
