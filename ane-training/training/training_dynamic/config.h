// config.h — Model-agnostic structs, derived sizes, ANE init
// Model-specific dims come from models/*.h, selected via -DMODEL_HEADER
#pragma once
#import <Foundation/Foundation.h>
#import <objc/runtime.h>
#import <objc/message.h>
#import <dlfcn.h>
#import <IOSurface/IOSurface.h>
#import <mach/mach_time.h>
#import <Accelerate/Accelerate.h>
#include <math.h>
#include <unistd.h>
#include <dispatch/dispatch.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <arm_neon.h>

// Include selected model config
// MODEL_HEADER is set by Makefile via -include models/xxx.h
#ifndef MODEL_NAME
#error "No model selected. Build with: make MODEL=qwen3_06b (or stories110m)"
#endif

// Derived weight sizes per layer (GQA-aware)
#define WQ_SZ (Q_DIM*DIM)
#define WK_SZ (KV_DIM*DIM)
#define WV_SZ (KV_DIM*DIM)
#define WO_SZ (DIM*Q_DIM)
#define W1_SZ (HIDDEN*DIM)
#define W2_SZ (DIM*HIDDEN)
#define W3_SZ (HIDDEN*DIM)
#define LAYER_PARAMS (WQ_SZ + WK_SZ + WV_SZ + WO_SZ + W1_SZ + W2_SZ + W3_SZ + 2*DIM)

// Attention score channels for SDPA backward
#define SCORE_CH (HEADS*SEQ)

// Per-layer weights
typedef struct {
    float *Wq, *Wk, *Wv, *Wo;
    float *W1, *W2, *W3;
    float *rms_att, *rms_ffn;
} LayerWeights;

// Adam optimizer state
typedef struct { float *m, *v; size_t n; } AdamState;
typedef struct {
    AdamState Wq, Wk, Wv, Wo, W1, W2, W3, rms_att, rms_ffn;
} LayerAdam;

// Per-layer activations (saved for backward)
// fp16 fields: stored as fp16 to avoid fp16→fp32→fp16 roundtrips between ANE kernels
typedef struct {
    float *layer_in, *xnorm;
    _Float16 *attn_out_fp16;  // stored as fp16, converted to fp32 only at dWo capture time
    float *x2, *x2norm, *ffn_out;
    _Float16 *Q_fp16, *K_fp16, *V_fp16;     // saved from sdpaFwd, used in sdpaBwd1/2
    _Float16 *silu_out_fp16;                 // saved from ffnFused, used for dW2 gradient
    // h1/h3 are pre-staged directly into ffnBwdFull input IOSurface during forward
} LayerActs;

// Per-layer gradients
typedef struct {
    float *Wq, *Wk, *Wv, *Wo, *W1, *W2, *W3, *rms_att, *rms_ffn;
} LayerGrads;

// ANE kernel handle
typedef struct { void *model; void *aneModel; IOSurfaceRef ioIn, ioOut; void *request; void *tmpDir; } Kern;

// Per-layer IOSurfaces for pre-staged weights
typedef struct {
    IOSurfaceRef sdpaFwd_in, ffnFused_in;
    IOSurfaceRef ffnBwdFull_in, wotBwd_in, wotSdpaBwd1_in, qkvBwd_in;
} PerLayerSurfaces;

// Per-layer ANE requests (bound to per-layer IOSurfaces)
typedef struct {
    void *sdpaFwd, *ffnFused;
    void *ffnBwdFull, *wotBwd, *wotSdpaBwd1, *qkvBwd;
} PerLayerRequests;

// Checkpoint header
typedef struct {
    int magic, version, step, total_steps;
    int n_layers, vocab_size, dim, hidden_dim, n_heads, seq_len;
    float lr, loss;
    double cum_compile, cum_train, cum_wall;
    int cum_steps, cum_batches, adam_t;
    int kv_heads, head_dim, q_dim;  // GQA fields
    // Note: was int pad[3] in v3, now stores GQA info in v4+
} CkptHdr;

// Globals
static Class g_D, g_I, g_AR, g_AIO;
static mach_timebase_info_data_t g_tb;
static int g_compile_count = 0;
static id g_ane_client = nil;  // _ANEClient for evaluateRealTime

static void ane_init(void) {
    dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine", RTLD_NOW);
    g_D  = NSClassFromString(@"_ANEInMemoryModelDescriptor");
    g_I  = NSClassFromString(@"_ANEInMemoryModel");
    g_AR = NSClassFromString(@"_ANERequest");
    g_AIO= NSClassFromString(@"_ANEIOSurfaceObject");
    Class ANEClient = NSClassFromString(@"_ANEClient");
    if (ANEClient) g_ane_client = ((id(*)(Class,SEL))objc_msgSend)(ANEClient, @selector(sharedConnection));
}
static double tb_ms(uint64_t t) { return (double)t * g_tb.numer / g_tb.denom / 1e6; }

// Alloc helpers
static AdamState adam_alloc(size_t n) { AdamState s; s.m=(float*)calloc(n,4); s.v=(float*)calloc(n,4); s.n=n; return s; }
static void adam_free(AdamState *s) { free(s->m); free(s->v); }

static LayerWeights layer_weights_alloc(void) {
    LayerWeights w;
    w.Wq=(float*)malloc(WQ_SZ*4); w.Wk=(float*)malloc(WK_SZ*4);
    w.Wv=(float*)malloc(WV_SZ*4); w.Wo=(float*)malloc(WO_SZ*4);
    w.W1=(float*)malloc(W1_SZ*4); w.W2=(float*)malloc(W2_SZ*4); w.W3=(float*)malloc(W3_SZ*4);
    w.rms_att=(float*)malloc(DIM*4); w.rms_ffn=(float*)malloc(DIM*4);
    return w;
}
static void layer_weights_free(LayerWeights *w) {
    free(w->Wq);free(w->Wk);free(w->Wv);free(w->Wo);
    free(w->W1);free(w->W2);free(w->W3);free(w->rms_att);free(w->rms_ffn);
}
static LayerAdam layer_adam_alloc(void) {
    LayerAdam a;
    a.Wq=adam_alloc(WQ_SZ); a.Wk=adam_alloc(WK_SZ); a.Wv=adam_alloc(WV_SZ); a.Wo=adam_alloc(WO_SZ);
    a.W1=adam_alloc(W1_SZ); a.W2=adam_alloc(W2_SZ); a.W3=adam_alloc(W3_SZ);
    a.rms_att=adam_alloc(DIM); a.rms_ffn=adam_alloc(DIM);
    return a;
}
static void layer_adam_free(LayerAdam *a) {
    adam_free(&a->Wq);adam_free(&a->Wk);adam_free(&a->Wv);adam_free(&a->Wo);
    adam_free(&a->W1);adam_free(&a->W2);adam_free(&a->W3);
    adam_free(&a->rms_att);adam_free(&a->rms_ffn);
}
static LayerActs layer_acts_alloc(void) {
    LayerActs a;
    a.layer_in=(float*)malloc(SEQ*DIM*4);
    a.xnorm=(float*)malloc(SEQ*DIM*4);
    a.attn_out_fp16=(_Float16*)malloc(SEQ*Q_DIM*2);
    a.x2=(float*)malloc(SEQ*DIM*4); a.x2norm=(float*)malloc(SEQ*DIM*4);
    a.ffn_out=(float*)malloc(SEQ*DIM*4);
    // fp16 activations — no fp32 roundtrip needed
    a.Q_fp16=(_Float16*)malloc(SEQ*Q_DIM*2);
    a.K_fp16=(_Float16*)malloc(SEQ*KV_DIM*2);
    a.V_fp16=(_Float16*)malloc(SEQ*KV_DIM*2);
    a.silu_out_fp16=(_Float16*)malloc(SEQ*HIDDEN*2);
    return a;
}
static void layer_acts_free(LayerActs *a) {
    free(a->layer_in);free(a->xnorm);
    free(a->attn_out_fp16);free(a->x2);free(a->x2norm);
    free(a->ffn_out);
    free(a->Q_fp16);free(a->K_fp16);free(a->V_fp16);
    free(a->silu_out_fp16);
}
static LayerGrads layer_grads_alloc(void) {
    LayerGrads g;
    g.Wq=(float*)calloc(WQ_SZ,4); g.Wk=(float*)calloc(WK_SZ,4);
    g.Wv=(float*)calloc(WV_SZ,4); g.Wo=(float*)calloc(WO_SZ,4);
    g.W1=(float*)calloc(W1_SZ,4); g.W2=(float*)calloc(W2_SZ,4); g.W3=(float*)calloc(W3_SZ,4);
    g.rms_att=(float*)calloc(DIM,4); g.rms_ffn=(float*)calloc(DIM,4);
    return g;
}
static void layer_grads_zero(LayerGrads *g) {
    memset(g->Wq,0,WQ_SZ*4);memset(g->Wk,0,WK_SZ*4);
    memset(g->Wv,0,WV_SZ*4);memset(g->Wo,0,WO_SZ*4);
    memset(g->W1,0,W1_SZ*4);memset(g->W2,0,W2_SZ*4);memset(g->W3,0,W3_SZ*4);
    memset(g->rms_att,0,DIM*4);memset(g->rms_ffn,0,DIM*4);
}
static void layer_grads_free(LayerGrads *g) {
    free(g->Wq);free(g->Wk);free(g->Wv);free(g->Wo);
    free(g->W1);free(g->W2);free(g->W3);free(g->rms_att);free(g->rms_ffn);
}
