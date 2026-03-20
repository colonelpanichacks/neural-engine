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

static void rmsnorm(float *out, const float *x, const float *w, int d, int S) {
    rms_ensure_bufs(S);
    float *ss = g_rms_ss;
    memset(ss, 0, S*4);
    for (int i=0; i<d; i++) {
        vDSP_vmul(x+i*S, 1, x+i*S, 1, g_rms_tmp, 1, (vDSP_Length)S);
        vDSP_vadd(g_rms_tmp, 1, ss, 1, ss, 1, (vDSP_Length)S);
    }
    float invd = 1.0f/d, eps=1e-5f;
    vDSP_vsmsa(ss, 1, &invd, &eps, ss, 1, (vDSP_Length)S);
    int n = S; vvrsqrtf(ss, ss, &n);
    for (int i=0; i<d; i++) {
        vDSP_vmul(x+i*S, 1, ss, 1, out+i*S, 1, (vDSP_Length)S);
        vDSP_vsmul(out+i*S, 1, &w[i], out+i*S, 1, (vDSP_Length)S);
    }
}

static void rmsnorm_bwd(float *dx, float *dw, const float *dy, const float *x, const float *w, int d, int S) {
    rms_ensure_bufs(S);
    float *ss = g_rms_ss; memset(ss, 0, S*4);
    for (int i=0; i<d; i++) {
        vDSP_vmul(x+i*S, 1, x+i*S, 1, g_rms_tmp, 1, (vDSP_Length)S);
        vDSP_vadd(g_rms_tmp, 1, ss, 1, ss, 1, (vDSP_Length)S);
    }
    float invd = 1.0f/d, eps=1e-5f;
    vDSP_vsmsa(ss, 1, &invd, &eps, ss, 1, (vDSP_Length)S);
    float *rrms = g_rms_rrms;
    int n = S; vvrsqrtf(rrms, ss, &n);
    float *dot = g_rms_dot; memset(dot, 0, S*4);
    for (int i=0; i<d; i++) {
        vDSP_vmul(dy+i*S, 1, x+i*S, 1, g_rms_tmp, 1, (vDSP_Length)S);
        vDSP_vsma(g_rms_tmp, 1, &w[i], dot, 1, dot, 1, (vDSP_Length)S);
    }
    vDSP_vmul(rrms, 1, rrms, 1, ss, 1, (vDSP_Length)S);
    vDSP_vsmul(ss, 1, &invd, ss, 1, (vDSP_Length)S);
    vDSP_vmul(dot, 1, ss, 1, dot, 1, (vDSP_Length)S);
    for (int i=0; i<d; i++) {
        vDSP_vmul(x+i*S, 1, dot, 1, g_rms_tmp, 1, (vDSP_Length)S);
        vDSP_vsub(g_rms_tmp, 1, dy+i*S, 1, g_rms_tmp, 1, (vDSP_Length)S);
        vDSP_vmul(g_rms_tmp, 1, rrms, 1, g_rms_tmp, 1, (vDSP_Length)S);
        vDSP_vsmul(g_rms_tmp, 1, &w[i], dx+i*S, 1, (vDSP_Length)S);
        vDSP_vmul(dy+i*S, 1, x+i*S, 1, g_rms_tmp, 1, (vDSP_Length)S);
        vDSP_vmul(g_rms_tmp, 1, rrms, 1, g_rms_tmp, 1, (vDSP_Length)S);
        float s; vDSP_sve(g_rms_tmp, 1, &s, (vDSP_Length)S);
        dw[i] += s;
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

// Cross-entropy loss: operates on logits[V, S] column-major (each column = one token)
// Avoids transposing by using a per-token temp buffer
static float *g_ce_col = NULL;
static int g_ce_V = 0;
static float cross_entropy_loss(float *dlogits, const float *logits, const uint16_t *targets, int V, int S) {
    if (V > g_ce_V) { free(g_ce_col); g_ce_col = (float*)malloc(V * 4); g_ce_V = V; }
    float *col = g_ce_col;
    float total_loss = 0;
    float invS = 1.0f / S;
    for (int t = 0; t < S; t++) {
        // Gather column t: logits[v, t] = logits[v*S + t], stride=S
        cblas_scopy(V, logits + t, S, col, 1);
        // Softmax
        float maxv; vDSP_maxv(col, 1, &maxv, (vDSP_Length)V);
        float neg_max = -maxv;
        vDSP_vsadd(col, 1, &neg_max, col, 1, (vDSP_Length)V);
        int n = V; vvexpf(col, col, &n);
        float sum; vDSP_sve(col, 1, &sum, (vDSP_Length)V);
        float inv_sum = 1.0f / sum;
        vDSP_vsmul(col, 1, &inv_sum, col, 1, (vDSP_Length)V);
        // Loss + gradient
        int tgt = targets[t];
        total_loss -= logf(col[tgt] + 1e-10f);
        col[tgt] -= 1.0f;
        vDSP_vsmul(col, 1, &invS, col, 1, (vDSP_Length)V);
        // Scatter back: dlogits[v*S + t] = col[v]
        cblas_scopy(V, col, 1, dlogits + t, S);
    }
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
static int g_rope_hd = 0, g_rope_seq = 0;

static void rope_ensure_table(int hd, int seq) {
    if (g_rope_hd == hd && g_rope_seq == seq) return;
    free(g_rope_cos); free(g_rope_sin);
    int table_sz = (hd / 2) * seq;
    g_rope_cos = (float*)malloc(table_sz * 4);
    g_rope_sin = (float*)malloc(table_sz * 4);
    for (int i = 0; i < hd / 2; i++) {
        float freq = 1.0f / powf(10000.0f, 2.0f * i / (float)hd);
        for (int p = 0; p < seq; p++) {
            float theta = p * freq;
            __sincosf(theta, &g_rope_sin[i * seq + p], &g_rope_cos[i * seq + p]);
        }
    }
    g_rope_hd = hd; g_rope_seq = seq;
}

static float *g_rope_tmp0 = NULL, *g_rope_tmp1 = NULL;
static int g_rope_tmp_seq = 0;
static void rope_backward_inplace(float *dx, int seq, int dim, int hd) {
    rope_ensure_table(hd, seq);
    int nheads = dim / hd;
    if (seq > g_rope_tmp_seq) {
        free(g_rope_tmp0); free(g_rope_tmp1);
        g_rope_tmp0 = (float*)malloc(seq * 4);
        g_rope_tmp1 = (float*)malloc(seq * 4);
        g_rope_tmp_seq = seq;
    }
    float *tmp0 = g_rope_tmp0;
    float *tmp1 = g_rope_tmp1;
    for (int h = 0; h < nheads; h++) {
        for (int i = 0; i < hd / 2; i++) {
            float *v0 = dx + (h * hd + 2 * i) * seq;
            float *v1 = dx + (h * hd + 2 * i + 1) * seq;
            float *cos_t = g_rope_cos + i * seq;
            float *sin_t = g_rope_sin + i * seq;
            // tmp0 = v0*cos + v1*sin (new v0)
            vDSP_vmul(v0, 1, cos_t, 1, tmp0, 1, (vDSP_Length)seq);
            vDSP_vmul(v1, 1, sin_t, 1, tmp1, 1, (vDSP_Length)seq);
            vDSP_vadd(tmp0, 1, tmp1, 1, tmp0, 1, (vDSP_Length)seq);
            // v1 = -v0*sin + v1*cos (new v1, computed before v0 is overwritten)
            vDSP_vmul(v0, 1, sin_t, 1, tmp1, 1, (vDSP_Length)seq);
            vDSP_vneg(tmp1, 1, tmp1, 1, (vDSP_Length)seq);
            vDSP_vmul(v1, 1, cos_t, 1, v1, 1, (vDSP_Length)seq);
            vDSP_vadd(tmp1, 1, v1, 1, v1, 1, (vDSP_Length)seq);
            // Write new v0
            memcpy(v0, tmp0, seq * 4);
        }
    }
}
