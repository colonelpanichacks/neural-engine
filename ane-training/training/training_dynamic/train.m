// train.m — Dynamic weight ANE training (model-agnostic GQA support)
// Model selected at compile time via: make MODEL=qwen3_06b (or stories110m)
// Compile kernels ONCE at startup, update weights via IOSurface every step.
#include "mil_dynamic.h"
#include "cpu_ops.h"

// Dynamic kernel set per layer
typedef struct {
    Kern *sdpaFwd;     // QKV matmul + RoPE + GQA tile + SDPA (no Wo)
    Kern *woFwd;       // attn_out @ Wo^T → o_out (Q_DIM → DIM)
    Kern *ffnFused;    // W1,W3 + SiLU + W2 + residual (fused)
    Kern *ffnBwdW2t;   // dffn @ W2^T → dsilu_raw (DIM → HIDDEN)
    Kern *ffnBwdW13t;  // dh1@W1^T + dh3@W3^T → dx_ffn (HIDDEN → DIM) [legacy]
    Kern *ffnBwdFused; // SiLU backward + W13t fused (HIDDEN → DIM+2*HIDDEN)
    Kern *wotBwd;      // dx2 @ Wo → da (DIM → Q_DIM)
    Kern *sdpaBwd1;    // Q,K,V,da → dV_full,probs,dp (weight-free, has mask)
    Kern *sdpaBwd2;    // probs,dp,Q,K → dQ,dK_full (weight-free)
    Kern *qBwd;        // dq @ Wq → dx_q (Q_DIM → DIM)
    Kern *kvBwd;       // dk@Wk + dv@Wv → dx_kv (KV_DIM → DIM)
} DynLayerKernels;

// Transpose W[rows,cols] → W^T[cols,rows] stored as [cols channels, rows spatial]
static void transpose_weight(float *dst, const float *src, int rows, int cols) {
    vDSP_mtrans(src, 1, dst, 1, (vDSP_Length)cols, (vDSP_Length)rows);
}

// ===== Compile all dynamic kernels (ONCE) =====
static bool compile_dynamic_kernels(DynLayerKernels *dk) {
    NSDictionary *mask_w = @{@"@model_path/weights/mask.bin": @{@"offset":@0, @"data":get_mask_blob()}};
    NSDictionary *sdpa_fwd_w = @{
        @"@model_path/weights/mask.bin": @{@"offset":@0, @"data":get_mask_blob()},
        @"@model_path/weights/rope_cos.bin": @{@"offset":@0, @"data":get_rope_cos_blob()},
        @"@model_path/weights/rope_sin.bin": @{@"offset":@0, @"data":get_rope_sin_blob()}
    };

    int sdpa_out_ch = Q_DIM + Q_DIM + KV_DIM + KV_DIM + DIM;

    // SDPA forward (no Wo): [1, DIM, 1, SDPA_FWD_SP] → [1, sdpa_out_ch, 1, SEQ]
    printf("  Compiling sdpaFwd (GQA)...\n");
    dk->sdpaFwd = compile_kern_mil_w(gen_sdpa_fwd_dynamic(), sdpa_fwd_w,
        DIM*SDPA_FWD_SP*2, sdpa_out_ch*SEQ*2);
    if (!dk->sdpaFwd) return false;

    // Wo forward: [1, Q_DIM, 1, SEQ+DIM] → [1, DIM, 1, SEQ]
    printf("  Compiling woFwd...\n");
    dk->woFwd = compile_kern_mil_w(gen_wo_fwd_dynamic(), @{},
        Q_DIM*WO_FWD_SP*2, DIM*SEQ*2);
    if (!dk->woFwd) return false;

    // Fused FFN: [1, DIM, 1, FFN_FUSED_SP] → [1, DIM+3*HIDDEN, 1, SEQ]
    printf("  Compiling ffnFused...\n");
    int ffn_fused_och = DIM + 3*HIDDEN;
    dk->ffnFused = compile_kern_mil_w(gen_ffn_fused_dynamic(), @{},
        DIM*FFN_FUSED_SP*2, ffn_fused_och*SEQ*2);
    if (!dk->ffnFused) return false;

    // FFN backward W2^T: [1, DIM, 1, SEQ+HIDDEN] → [1, HIDDEN, 1, SEQ]
    printf("  Compiling ffnBwdW2t...\n");
    dk->ffnBwdW2t = compile_kern_mil_w(gen_ffn_bwd_w2t_dynamic(), @{},
        DIM*FFN_BWD_W2T_SP*2, HIDDEN*SEQ*2);
    if (!dk->ffnBwdW2t) return false;

    // FFN backward W1^T+W3^T: [1, HIDDEN, 1, 2*SEQ+2*DIM] → [1, DIM, 1, SEQ]
    printf("  Compiling ffnBwdW13t...\n");
    dk->ffnBwdW13t = compile_kern_mil_w(gen_ffn_bwd_w13t_dynamic(), @{},
        HIDDEN*FFN_BWD_W13T_SP*2, DIM*SEQ*2);
    if (!dk->ffnBwdW13t) return false;

    // FFN backward FUSED (SiLU bwd + W13t): [1, HIDDEN, 1, 3*SEQ+2*DIM] → [1, DIM+2*HIDDEN, 1, SEQ]
    printf("  Compiling ffnBwdFused (SiLU+W13t)...\n");
    int fused_out_ch = DIM + 2*HIDDEN;
    dk->ffnBwdFused = compile_kern_mil_w(gen_ffn_bwd_fused_dynamic(), @{},
        HIDDEN*FFN_BWD_FUSED_SP*2, fused_out_ch*SEQ*2);
    if (!dk->ffnBwdFused) return false;

    // Wo^T backward: [1, DIM, 1, SEQ+Q_DIM] → [1, Q_DIM, 1, SEQ]
    printf("  Compiling wotBwd...\n");
    dk->wotBwd = compile_kern_mil_w(gen_wot_dynamic(), @{},
        DIM*WOT_BWD_SP*2, Q_DIM*SEQ*2);
    if (!dk->wotBwd) return false;

    // SDPA bwd1 (weight-free, has mask): [1, 4*Q_DIM, 1, SEQ] → [1, Q_DIM+2*SCORE_CH, 1, SEQ]
    printf("  Compiling sdpaBwd1 (GQA)...\n");
    dk->sdpaBwd1 = compile_kern_mil_w(gen_sdpa_bwd1_noweight(), mask_w,
        4*Q_DIM*SEQ*2, (Q_DIM+2*SCORE_CH)*SEQ*2);
    if (!dk->sdpaBwd1) return false;

    // SDPA bwd2 (weight-free): [1, 2*SCORE_CH+2*Q_DIM, 1, SEQ] → [1, 2*Q_DIM, 1, SEQ]
    printf("  Compiling sdpaBwd2 (GQA)...\n");
    dk->sdpaBwd2 = compile_kern_mil_w(gen_sdpa_bwd2(), @{},
        (2*SCORE_CH+2*Q_DIM)*SEQ*2, 2*Q_DIM*SEQ*2);
    if (!dk->sdpaBwd2) return false;

    // Q backward: [1, Q_DIM, 1, SEQ+DIM] → [1, DIM, 1, SEQ]
    printf("  Compiling qBwd...\n");
    dk->qBwd = compile_kern_mil_w(gen_q_bwd_dynamic(), @{},
        Q_DIM*Q_BWD_SP*2, DIM*SEQ*2);
    if (!dk->qBwd) return false;

    // KV backward: [1, KV_DIM, 1, 2*SEQ+2*DIM] → [1, DIM, 1, SEQ]
    printf("  Compiling kvBwd...\n");
    dk->kvBwd = compile_kern_mil_w(gen_kv_bwd_dynamic(), @{},
        KV_DIM*KV_BWD_SP*2, DIM*SEQ*2);
    if (!dk->kvBwd) return false;

    return true;
}

// ===== Checkpoint =====
static void save_checkpoint(const char *path, int step, int total_steps, float lr, float loss,
                            double ct, double cw, int cs, int adam_t,
                            LayerWeights *lw, LayerAdam *la, float *rms_final, AdamState *arms_final,
                            float *embed, AdamState *aembed) {
    FILE *f = fopen(path, "wb");
    CkptHdr h = {0};
    h.magic = 0x424C5A54; h.version = 4;
    h.step = step; h.total_steps = total_steps;
    h.n_layers = NLAYERS; h.vocab_size = VOCAB; h.dim = DIM;
    h.hidden_dim = HIDDEN; h.n_heads = HEADS; h.seq_len = SEQ;
    h.lr = lr; h.loss = loss;
    h.cum_train = ct; h.cum_wall = cw; h.cum_steps = cs; h.adam_t = adam_t;
    h.kv_heads = KV_HEADS; h.head_dim = HD; h.q_dim = Q_DIM;
    fwrite(&h, sizeof(h), 1, f);
    for (int L = 0; L < NLAYERS; L++) {
        fwrite(lw[L].Wq,4,WQ_SZ,f); fwrite(lw[L].Wk,4,WK_SZ,f);
        fwrite(lw[L].Wv,4,WV_SZ,f); fwrite(lw[L].Wo,4,WO_SZ,f);
        fwrite(lw[L].W1,4,W1_SZ,f); fwrite(lw[L].W2,4,W2_SZ,f); fwrite(lw[L].W3,4,W3_SZ,f);
        fwrite(lw[L].rms_att,4,DIM,f); fwrite(lw[L].rms_ffn,4,DIM,f);
        fwrite(la[L].Wq.m,4,WQ_SZ,f); fwrite(la[L].Wq.v,4,WQ_SZ,f);
        fwrite(la[L].Wk.m,4,WK_SZ,f); fwrite(la[L].Wk.v,4,WK_SZ,f);
        fwrite(la[L].Wv.m,4,WV_SZ,f); fwrite(la[L].Wv.v,4,WV_SZ,f);
        fwrite(la[L].Wo.m,4,WO_SZ,f); fwrite(la[L].Wo.v,4,WO_SZ,f);
        fwrite(la[L].W1.m,4,W1_SZ,f); fwrite(la[L].W1.v,4,W1_SZ,f);
        fwrite(la[L].W2.m,4,W2_SZ,f); fwrite(la[L].W2.v,4,W2_SZ,f);
        fwrite(la[L].W3.m,4,W3_SZ,f); fwrite(la[L].W3.v,4,W3_SZ,f);
        fwrite(la[L].rms_att.m,4,DIM,f); fwrite(la[L].rms_att.v,4,DIM,f);
        fwrite(la[L].rms_ffn.m,4,DIM,f); fwrite(la[L].rms_ffn.v,4,DIM,f);
    }
    fwrite(rms_final,4,DIM,f);
    fwrite(arms_final->m,4,DIM,f); fwrite(arms_final->v,4,DIM,f);
    fwrite(embed,4,VOCAB*DIM,f);
    fwrite(aembed->m,4,VOCAB*DIM,f); fwrite(aembed->v,4,VOCAB*DIM,f);
    fclose(f);
}

static bool load_checkpoint(const char *path, int *step, int *total_steps, float *lr, float *loss,
                             double *ct, double *cw, int *cs, int *adam_t,
                             LayerWeights *lw, LayerAdam *la, float *rms_final, AdamState *arms_final,
                             float *embed, AdamState *aembed) {
    FILE *f = fopen(path, "rb");
    if (!f) return false;
    CkptHdr h;
    fread(&h, sizeof(h), 1, f);
    if (h.magic != 0x424C5A54 || h.version != 4) { fclose(f); return false; }
    *step = h.step; *total_steps = h.total_steps; *lr = h.lr; *loss = h.loss;
    *ct = h.cum_train; *cw = h.cum_wall; *cs = h.cum_steps; *adam_t = h.adam_t;
    for (int L = 0; L < NLAYERS; L++) {
        fread(lw[L].Wq,4,WQ_SZ,f); fread(lw[L].Wk,4,WK_SZ,f);
        fread(lw[L].Wv,4,WV_SZ,f); fread(lw[L].Wo,4,WO_SZ,f);
        fread(lw[L].W1,4,W1_SZ,f); fread(lw[L].W2,4,W2_SZ,f); fread(lw[L].W3,4,W3_SZ,f);
        fread(lw[L].rms_att,4,DIM,f); fread(lw[L].rms_ffn,4,DIM,f);
        fread(la[L].Wq.m,4,WQ_SZ,f); fread(la[L].Wq.v,4,WQ_SZ,f);
        fread(la[L].Wk.m,4,WK_SZ,f); fread(la[L].Wk.v,4,WK_SZ,f);
        fread(la[L].Wv.m,4,WV_SZ,f); fread(la[L].Wv.v,4,WV_SZ,f);
        fread(la[L].Wo.m,4,WO_SZ,f); fread(la[L].Wo.v,4,WO_SZ,f);
        fread(la[L].W1.m,4,W1_SZ,f); fread(la[L].W1.v,4,W1_SZ,f);
        fread(la[L].W2.m,4,W2_SZ,f); fread(la[L].W2.v,4,W2_SZ,f);
        fread(la[L].W3.m,4,W3_SZ,f); fread(la[L].W3.v,4,W3_SZ,f);
        fread(la[L].rms_att.m,4,DIM,f); fread(la[L].rms_att.v,4,DIM,f);
        fread(la[L].rms_ffn.m,4,DIM,f); fread(la[L].rms_ffn.v,4,DIM,f);
    }
    fread(rms_final,4,DIM,f);
    fread(arms_final->m,4,DIM,f); fread(arms_final->v,4,DIM,f);
    fread(embed,4,VOCAB*DIM,f);
    fread(aembed->m,4,VOCAB*DIM,f); fread(aembed->v,4,VOCAB*DIM,f);
    fclose(f);
    return true;
}

int main(int argc, char *argv[]) {
    @autoreleasepool {
        setbuf(stdout, NULL);
        ane_init();
        mach_timebase_info(&g_tb);

        int total_steps = 10000;
        float max_lr = 3e-4f;
        float adam_b1=0.9f, adam_b2=0.95f, adam_eps=1e-8f, wd=0.1f;
        int adam_t = 0, start_step = 0;
        int accum_steps = 10;
        int warmup_steps = 100;
        float grad_clip = 1.0f;
        float loss_scale = 256.0f;
        float res_alpha = 1.0f / sqrtf(2.0f * NLAYERS);
        float min_lr_frac = 0.1f;

        bool do_resume = false, from_scratch = false;
        const char *data_path = DEFAULT_DATA_PATH;
        for (int i=1; i<argc; i++) {
            if (strcmp(argv[i], "--resume") == 0) do_resume = true;
            else if (strcmp(argv[i], "--scratch") == 0) from_scratch = true;
            else if (strcmp(argv[i], "--steps") == 0 && i+1<argc) total_steps = atoi(argv[++i]);
            else if (strcmp(argv[i], "--lr") == 0 && i+1<argc) max_lr = atof(argv[++i]);
            else if (strcmp(argv[i], "--accum") == 0 && i+1<argc) accum_steps = atoi(argv[++i]);
            else if (strcmp(argv[i], "--warmup") == 0 && i+1<argc) warmup_steps = atoi(argv[++i]);
            else if (strcmp(argv[i], "--clip") == 0 && i+1<argc) grad_clip = atof(argv[++i]);
            else if (strcmp(argv[i], "--data") == 0 && i+1<argc) data_path = argv[++i];
        }
        float lr = max_lr;

        // Allocate per-layer state
        LayerWeights lw[NLAYERS]; LayerAdam la[NLAYERS];
        LayerActs acts[NLAYERS]; LayerGrads grads[NLAYERS];
        for (int L=0; L<NLAYERS; L++) {
            lw[L] = layer_weights_alloc(); la[L] = layer_adam_alloc();
            acts[L] = layer_acts_alloc(); grads[L] = layer_grads_alloc();
        }
        float *rms_final = (float*)malloc(DIM*4);
        float *embed = (float*)malloc(VOCAB*DIM*4);
        float *grms_final = (float*)calloc(DIM, 4);
        float *gembed = (float*)calloc(VOCAB*DIM, 4);
        AdamState arms_final = adam_alloc(DIM);
        AdamState aembed = adam_alloc((size_t)VOCAB*DIM);

        double cum_train=0, cum_wall=0; int cum_steps=0;
        float resume_loss = 0;
        bool resuming = false;
        if (do_resume) {
            resuming = load_checkpoint(CKPT_PATH, &start_step, &total_steps, &lr, &resume_loss,
                &cum_train, &cum_wall, &cum_steps, &adam_t,
                lw, la, rms_final, &arms_final, embed, &aembed);
            if (resuming) printf("[RESUMED step %d, loss=%.4f]\n", start_step, resume_loss);
        }
        if (!resuming) {
            printf("=== ANE Dynamic Training: %s (%d layers, GQA %d/%d heads) ===\n",
                   MODEL_NAME, NLAYERS, HEADS, KV_HEADS);
            printf("dim=%d q_dim=%d kv_dim=%d hd=%d hidden=%d seq=%d vocab=%d\n",
                   DIM, Q_DIM, KV_DIM, HD, HIDDEN, SEQ, VOCAB);
            double xformer_m = (double)NLAYERS*(WQ_SZ + WK_SZ + WV_SZ + (double)WO_SZ + W1_SZ + W2_SZ + W3_SZ + 2.0*DIM) / 1e6;
            double embed_m = (double)VOCAB*DIM / 1e6;
            printf("Params: %.1fM (transformer %.1fM + embed %.1fM)\n", xformer_m+embed_m, xformer_m, embed_m);
            printf("Kernels: 11 compiled (sdpaFwd+woFwd, ffnFused, ffnBwdW2t+W13t+Fused, wotBwd, sdpaBwd1+2, qBwd+kvBwd)\n");
            printf("Accum %d steps, LR=%g\n", accum_steps, max_lr);
            double fwd_flops = 2.0*NLAYERS*((double)WQ_SZ + WK_SZ + WV_SZ + WO_SZ + W1_SZ + W2_SZ + W3_SZ) * SEQ;
            double total_flops = 3.0 * fwd_flops;
            printf("FLOPs/step: fwd=%.1fM total=%.1fM\n", fwd_flops/1e6, total_flops/1e6);
            if (from_scratch) {
                printf("  Training from scratch (random init)\n");
                srand48(42);
                float scale_d=1.0f/sqrtf(DIM), scale_qd=1.0f/sqrtf(Q_DIM), scale_h=1.0f/sqrtf(HIDDEN);
                float res_scale = 1.0f/sqrtf(2.0f*NLAYERS);
                for (int L=0; L<NLAYERS; L++) {
                    for(size_t i=0;i<WQ_SZ;i++) lw[L].Wq[i]=scale_d*(2*drand48()-1);
                    for(size_t i=0;i<WK_SZ;i++) lw[L].Wk[i]=scale_d*(2*drand48()-1);
                    for(size_t i=0;i<WV_SZ;i++) lw[L].Wv[i]=scale_d*(2*drand48()-1);
                    for(size_t i=0;i<WO_SZ;i++) lw[L].Wo[i]=scale_qd*res_scale*(2*drand48()-1);
                    for(size_t i=0;i<W1_SZ;i++) lw[L].W1[i]=scale_h*(2*drand48()-1);
                    for(size_t i=0;i<W2_SZ;i++) lw[L].W2[i]=scale_d*res_scale*(2*drand48()-1);
                    for(size_t i=0;i<W3_SZ;i++) lw[L].W3[i]=scale_h*(2*drand48()-1);
                    for(int i=0;i<DIM;i++){lw[L].rms_att[i]=1.0f; lw[L].rms_ffn[i]=1.0f;}
                }
                for(int i=0;i<DIM;i++) rms_final[i]=1.0f;
                float escale = 0.02f;
                for(size_t i=0;i<(size_t)VOCAB*DIM;i++) embed[i]=escale*(2*drand48()-1);
            } else {
                printf("  ERROR: Pretrained weight loading not implemented for Qwen3. Use --scratch.\n");
                return 1;
            }
        }

        // Precompute transposed weights for forward/backward kernels
        // Forward: sdpaFwd needs Wq^T[Q_DIM,DIM], Wk^T[KV_DIM,DIM], Wv^T[KV_DIM,DIM]
        //          woFwd needs Wo^T[DIM,Q_DIM]
        // Backward uses original (non-transposed) weights
        float *Wqt_buf[NLAYERS], *Wkt_buf[NLAYERS], *Wvt_buf[NLAYERS], *Wot_buf[NLAYERS];
        float *W1t_buf[NLAYERS], *W2t_buf[NLAYERS], *W3t_buf[NLAYERS];
        for (int L=0; L<NLAYERS; L++) {
            Wqt_buf[L]=(float*)malloc(WQ_SZ*4); Wkt_buf[L]=(float*)malloc(WK_SZ*4);
            Wvt_buf[L]=(float*)malloc(WV_SZ*4); Wot_buf[L]=(float*)malloc(WO_SZ*4);
            W1t_buf[L]=(float*)malloc(W1_SZ*4); W2t_buf[L]=(float*)malloc(W2_SZ*4);
            W3t_buf[L]=(float*)malloc(W3_SZ*4);
            // Wq is [Q_DIM, DIM] → Wq^T is [DIM, Q_DIM] (staged as [DIM channels, Q_DIM spatial])
            transpose_weight(Wqt_buf[L], lw[L].Wq, Q_DIM, DIM);
            // Wk is [KV_DIM, DIM] → Wk^T is [DIM, KV_DIM]
            transpose_weight(Wkt_buf[L], lw[L].Wk, KV_DIM, DIM);
            // Wv is [KV_DIM, DIM] → Wv^T is [DIM, KV_DIM]
            transpose_weight(Wvt_buf[L], lw[L].Wv, KV_DIM, DIM);
            // Wo is [DIM, Q_DIM] → Wo^T is [Q_DIM, DIM]
            transpose_weight(Wot_buf[L], lw[L].Wo, DIM, Q_DIM);
            transpose_weight(W1t_buf[L], lw[L].W1, HIDDEN, DIM);
            transpose_weight(W2t_buf[L], lw[L].W2, DIM, HIDDEN);
            transpose_weight(W3t_buf[L], lw[L].W3, HIDDEN, DIM);
        }

        // mmap token data
        int data_fd = open(data_path, O_RDONLY);
        if (data_fd < 0) { printf("Cannot open %s\n", data_path); return 1; }
        struct stat st; fstat(data_fd, &st);
        size_t data_len = st.st_size;
        uint16_t *token_data = (uint16_t*)mmap(NULL, data_len, PROT_READ, MAP_PRIVATE, data_fd, 0);
        if (token_data == MAP_FAILED) { printf("mmap failed\n"); return 1; }
        size_t n_tokens = data_len / 2;
        printf("Token data: %zu tokens (%.1f MB)\n", n_tokens, data_len/1e6);

        // Vocab compaction
        VocabMap vm = vocab_map_build(token_data, n_tokens, VOCAB);
        int CV = vm.compact_vocab;
        printf("Vocab compaction: %d → %d active tokens (%.1fx reduction)\n", VOCAB, CV, (float)VOCAB/CV);

        float *cembed = vocab_compact_embed(embed, &vm, DIM);
        float *gcembed = (float*)calloc((size_t)CV*DIM, 4);

        // ===== Compile all kernels ONCE =====
        printf("Compiling 10 dynamic kernels (one-time)...\n");
        uint64_t tc = mach_absolute_time();
        DynLayerKernels dk;
        if (!compile_dynamic_kernels(&dk)) {
            printf("Compilation failed!\n"); return 1;
        }
        double compile_ms = tb_ms(mach_absolute_time() - tc);
        printf("Compiled 10 kernels in %.0fms (shared across all %d layers)\n", compile_ms, NLAYERS);

        // Allocate per-layer IOSurfaces + requests
        printf("Allocating per-layer IOSurfaces...\n");
        PerLayerSurfaces pls[NLAYERS];
        PerLayerRequests plr[NLAYERS];
        for (int L = 0; L < NLAYERS; L++) {
            pls[L].sdpaFwd_in    = make_surface(DIM*SDPA_FWD_SP*2);
            pls[L].woFwd_in      = make_surface(Q_DIM*WO_FWD_SP*2);
            pls[L].ffnFused_in   = make_surface(DIM*FFN_FUSED_SP*2);
            pls[L].ffnBwdW2t_in  = make_surface(DIM*FFN_BWD_W2T_SP*2);
            pls[L].ffnBwdW13t_in = make_surface(HIDDEN*FFN_BWD_W13T_SP*2);
            pls[L].ffnBwdFused_in = make_surface(HIDDEN*FFN_BWD_FUSED_SP*2);
            pls[L].wotBwd_in     = make_surface(DIM*WOT_BWD_SP*2);
            pls[L].qBwd_in       = make_surface(Q_DIM*Q_BWD_SP*2);
            pls[L].kvBwd_in      = make_surface(KV_DIM*KV_BWD_SP*2);

            plr[L].sdpaFwd   = make_request(dk.sdpaFwd,   pls[L].sdpaFwd_in);
            plr[L].woFwd     = make_request(dk.woFwd,     pls[L].woFwd_in);
            plr[L].ffnFused  = make_request(dk.ffnFused,  pls[L].ffnFused_in);
            plr[L].ffnBwdW2t = make_request(dk.ffnBwdW2t, pls[L].ffnBwdW2t_in);
            plr[L].ffnBwdW13t= make_request(dk.ffnBwdW13t,pls[L].ffnBwdW13t_in);
            plr[L].ffnBwdFused = make_request(dk.ffnBwdFused, pls[L].ffnBwdFused_in);
            plr[L].wotBwd    = make_request(dk.wotBwd,    pls[L].wotBwd_in);
            plr[L].qBwd      = make_request(dk.qBwd,      pls[L].qBwd_in);
            plr[L].kvBwd     = make_request(dk.kvBwd,     pls[L].kvBwd_in);
        }

        // Stage weights into per-layer surfaces
        for (int L = 0; L < NLAYERS; L++) {
            stage_sdpa_fwd_weights(pls[L].sdpaFwd_in, Wqt_buf[L], Wkt_buf[L], Wvt_buf[L]);
            stage_wo_fwd_weights(pls[L].woFwd_in, Wot_buf[L]);
            stage_ffn_fused_weights(pls[L].ffnFused_in, W1t_buf[L], W3t_buf[L], lw[L].W2);
            stage_ffn_bwd_w2t_weights(pls[L].ffnBwdW2t_in, lw[L].W2);
            stage_ffn_bwd_w13t_weights(pls[L].ffnBwdW13t_in, lw[L].W1, lw[L].W3);
            stage_ffn_bwd_fused_weights(pls[L].ffnBwdFused_in, lw[L].W1, lw[L].W3);
            stage_wot_bwd_weights(pls[L].wotBwd_in, lw[L].Wo);
            stage_q_bwd_weights(pls[L].qBwd_in, lw[L].Wq);
            stage_kv_bwd_weights(pls[L].kvBwd_in, lw[L].Wk, lw[L].Wv);
        }
        printf("Per-layer weight staging complete\n\n");

        // Gradient + work buffers (GQA: Q has Q_DIM, K/V have KV_DIM)
        float *dy = (float*)malloc(SEQ*DIM*4);
        float *dffn = (float*)malloc(SEQ*DIM*4);
        float *dx_ffn = (float*)malloc(SEQ*DIM*4);
        float *dx2 = (float*)malloc(SEQ*DIM*4);
        float *dx_attn = (float*)malloc(SEQ*DIM*4);
        float *dk_buf = (float*)malloc(SEQ*KV_DIM*4); // KV_DIM for K grads
        float *dv = (float*)malloc(SEQ*KV_DIM*4);     // KV_DIM for V grads
        // da_buf eliminated — direct fp16 copy from wotBwd→sdpaBwd1
        float *x_cur = (float*)malloc(SEQ*DIM*4);
        float *x_final = (float*)malloc(SEQ*DIM*4);
        float *xnorm_buf = (float*)malloc(SEQ*DIM*4);
        float *logits = (float*)malloc(SEQ*CV*4);
        float *dlogits = (float*)malloc(SEQ*CV*4);
        float *dh1 = (float*)malloc(SEQ*HIDDEN*4);
        float *dh3 = (float*)malloc(SEQ*HIDDEN*4);
        // dsilu no longer needed — direct fp16 copy from ffnBwdW2t→ffnBwdFused
        // GQA tile/reduce buffers
        _Float16 *k_tiled_fp16 = (_Float16*)malloc(SEQ*Q_DIM*2);  // KV_DIM → Q_DIM, fp16
        _Float16 *v_tiled_fp16 = (_Float16*)malloc(SEQ*Q_DIM*2);
        float *dq_full = (float*)malloc(SEQ*Q_DIM*4);  // from sdpaBwd2
        float *dk_full = (float*)malloc(SEQ*Q_DIM*4);  // from sdpaBwd2 (needs reduce)
        float *dv_full = (float*)malloc(SEQ*Q_DIM*4);  // from sdpaBwd1 (needs reduce)

        // Pre-allocated backward buffers (avoid hot-path malloc/free)
        float *dx2_scaled = (float*)malloc(SEQ*DIM*4);
        float *dx_kv = (float*)malloc(SEQ*DIM*4);
        float *dx_rms_final = (float*)malloc(SEQ*DIM*4);
        float *dx_rms1 = (float*)malloc(SEQ*DIM*4);

        dispatch_queue_t dw_q = dispatch_queue_create("dw_cblas", DISPATCH_QUEUE_CONCURRENT);
        dispatch_group_t dw_grp = dispatch_group_create();

        float last_loss = 999.0f;
        float best_loss = resume_loss > 0 ? resume_loss : 999.0f;
        double total_train_ms = 0;
        int total_steps_done = 0;
        uint64_t t_wall_start = mach_absolute_time();
        srand48(42 + start_step);

        for (int step = start_step; step < total_steps; step++) {
            uint64_t t0, t_step = mach_absolute_time();

            // Sample data
            size_t max_pos = n_tokens - SEQ - 1;
            size_t pos = (size_t)(drand48() * max_pos);
            uint16_t *input_tokens = token_data + pos;
            uint16_t *target_tokens_raw = token_data + pos + 1;

            uint16_t ctargets[SEQ];
            for (int t = 0; t < SEQ; t++) ctargets[t] = (uint16_t)vm.full_to_compact[target_tokens_raw[t]];

            embed_lookup(x_cur, embed, input_tokens, DIM, SEQ);

            double t_rms=0, t_ane_fwd=0, t_io_fwd=0, t_cblas_wait=0;
            double t_ane_bwd=0, t_io_bwd=0, t_rms_bwd=0, t_cls=0, t_dw_copy=0;

            // ===== FORWARD (28 layers) =====
            for (int L=0; L<NLAYERS; L++) {
                LayerActs *ac = &acts[L];
                memcpy(ac->layer_in, x_cur, SEQ*DIM*4);

                // RMSNorm1 (CPU)
                t0 = mach_absolute_time();
                rmsnorm(xnorm_buf, x_cur, lw[L].rms_att, DIM, SEQ);
                memcpy(ac->xnorm, xnorm_buf, SEQ*DIM*4);
                t_rms += tb_ms(mach_absolute_time() - t0);

                // Wait for any pending dW cblas
                t0 = mach_absolute_time();
                dispatch_group_wait(dw_grp, DISPATCH_TIME_FOREVER);
                t_cblas_wait += tb_ms(mach_absolute_time() - t0);

                // SDPA forward (ANE): xnorm + Wq,Wk,Wv → attn_out[Q_DIM], Q_rope[Q_DIM], K_rope[KV_DIM], V[KV_DIM], xnorm[DIM]
                t0 = mach_absolute_time();
                write_sdpa_fwd_acts(pls[L].sdpaFwd_in, xnorm_buf);
                t_io_fwd += tb_ms(mach_absolute_time() - t0);
                t0 = mach_absolute_time();
                ane_eval_req(dk.sdpaFwd, plr[L].sdpaFwd);
                t_ane_fwd += tb_ms(mach_absolute_time() - t0);

                // Direct fp16 copy: attn_out from sdpaFwd → woFwd (skip fp32 roundtrip)
                t0 = mach_absolute_time();
                IOSurfaceLock(dk.sdpaFwd->ioOut, kIOSurfaceLockReadOnly, NULL);
                _Float16 *fwd_out = (_Float16*)IOSurfaceGetBaseAddress(dk.sdpaFwd->ioOut);
                IOSurfaceLock(pls[L].woFwd_in, 0, NULL);
                copy_attn_out_fp16(pls[L].woFwd_in, fwd_out);
                IOSurfaceUnlock(pls[L].woFwd_in, 0, NULL);
                t_io_fwd += tb_ms(mach_absolute_time() - t0);

                // Launch woFwd async, overlap with fp16 saves for backward
                t0 = mach_absolute_time();
                dispatch_semaphore_t sem_wo = ane_eval_req_async(dk.woFwd, plr[L].woFwd);
                // CPU work overlapped with ANE: save attn_out(fp32), Q/K/V(fp16) for backward
                int off = 0;
                cvt_f16_f32(ac->attn_out, fwd_out + off, Q_DIM*SEQ); off += Q_DIM*SEQ;
                memcpy(ac->Q_fp16, fwd_out + off, Q_DIM*SEQ*2);      off += Q_DIM*SEQ;
                memcpy(ac->K_fp16, fwd_out + off, KV_DIM*SEQ*2);     off += KV_DIM*SEQ;
                memcpy(ac->V_fp16, fwd_out + off, KV_DIM*SEQ*2);
                IOSurfaceUnlock(dk.sdpaFwd->ioOut, kIOSurfaceLockReadOnly, NULL);
                ane_eval_wait(sem_wo);
                t_ane_fwd += tb_ms(mach_absolute_time() - t0);

                // Read woFwd output
                t0 = mach_absolute_time();
                io_read_dyn(dk.woFwd->ioOut, ac->o_out, DIM, SEQ);
                t_io_fwd += tb_ms(mach_absolute_time() - t0);

                // CPU: scaled residual
                vDSP_vsma(ac->o_out, 1, &res_alpha, x_cur, 1, ac->x2, 1, (vDSP_Length)(SEQ*DIM));

                // CPU: RMSNorm for FFN
                t0 = mach_absolute_time();
                rmsnorm(ac->x2norm, ac->x2, lw[L].rms_ffn, DIM, SEQ);
                t_rms += tb_ms(mach_absolute_time() - t0);

                // Fused FFN (ANE): x2norm,x2 → W1,W3,SiLU,W2,residual
                t0 = mach_absolute_time();
                write_ffn_fused_acts(pls[L].ffnFused_in, ac->x2norm, ac->x2);
                t_io_fwd += tb_ms(mach_absolute_time() - t0);
                t0 = mach_absolute_time();
                ane_eval_req(dk.ffnFused, plr[L].ffnFused);
                t_ane_fwd += tb_ms(mach_absolute_time() - t0);

                // Read fused output: [1, DIM+3*HIDDEN, 1, SEQ]
                t0 = mach_absolute_time();
                IOSurfaceLock(dk.ffnFused->ioOut, kIOSurfaceLockReadOnly, NULL);
                _Float16 *ffn_out = (_Float16*)IOSurfaceGetBaseAddress(dk.ffnFused->ioOut);
                off = 0;
                cvt_f16_f32(x_cur,       ffn_out + off, DIM*SEQ);     off += DIM*SEQ;
                memcpy(ac->h1_fp16,       ffn_out + off, HIDDEN*SEQ*2); off += HIDDEN*SEQ;
                memcpy(ac->h3_fp16,       ffn_out + off, HIDDEN*SEQ*2); off += HIDDEN*SEQ;
                memcpy(ac->silu_out_fp16, ffn_out + off, HIDDEN*SEQ*2);
                IOSurfaceUnlock(dk.ffnFused->ioOut, kIOSurfaceLockReadOnly, NULL);
                t_io_fwd += tb_ms(mach_absolute_time() - t0);
            }

            // Final RMSNorm + classifier + loss (CPU cblas — needs FP32 precision for softmax)
            t0 = mach_absolute_time();
            rmsnorm(x_final, x_cur, rms_final, DIM, SEQ);
            t_rms += tb_ms(mach_absolute_time() - t0);
            t0 = mach_absolute_time();
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                        CV, SEQ, DIM, 1.0f, cembed, DIM, x_final, SEQ, 0.0f, logits, SEQ);
            float loss = cross_entropy_loss(dlogits, logits, ctargets, CV, SEQ);
            t_cls += tb_ms(mach_absolute_time() - t0);
            last_loss = loss;

            // ===== BACKWARD =====
            vDSP_vsmul(dlogits, 1, &loss_scale, dlogits, 1, (vDSP_Length)(SEQ*CV));

            // Classifier backward (CPU cblas)
            t0 = mach_absolute_time();
            cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                        DIM, SEQ, CV, 1.0f, cembed, DIM, dlogits, SEQ, 0.0f, dy, SEQ);
            t_cls += tb_ms(mach_absolute_time() - t0);

            // dEmbed async
            dispatch_group_async(dw_grp, dw_q, ^{
                cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                            CV, DIM, SEQ, 1.0f, dlogits, SEQ, x_final, SEQ, 1.0f, gcembed, DIM);
            });

            // Final RMSNorm backward
            rmsnorm_bwd(dx_rms_final, grms_final, dy, x_cur, rms_final, DIM, SEQ);
            memcpy(dy, dx_rms_final, SEQ*DIM*4);

            // ===== BACKWARD (28 layers, reverse) — ASYNC OVERLAP OPTIMIZED =====
            for (int L=NLAYERS-1; L>=0; L--) {
                LayerActs *ac = &acts[L];
                LayerGrads *gr = &grads[L];

                // dffn = alpha * dy
                vDSP_vsmul(dy, 1, &res_alpha, dffn, 1, (vDSP_Length)(SEQ*DIM));

                // FFN backward: dffn @ W2^T → dsilu_raw (sync — need result for SiLU)
                t0 = mach_absolute_time();
                write_ffn_bwd_w2t_acts(pls[L].ffnBwdW2t_in, dffn);
                t_io_bwd += tb_ms(mach_absolute_time() - t0);
                t0 = mach_absolute_time();
                ane_eval_req(dk.ffnBwdW2t, plr[L].ffnBwdW2t);
                t_ane_bwd += tb_ms(mach_absolute_time() - t0);
                t0 = mach_absolute_time();
                // Direct fp16 strided copy: ffnBwdW2t output → ffnBwdFused input
                // Eliminates dsilu fp16→fp32→fp16 roundtrip
                IOSurfaceLock(dk.ffnBwdW2t->ioOut, kIOSurfaceLockReadOnly, NULL);
                IOSurfaceLock(pls[L].ffnBwdFused_in, 0, NULL);
                _Float16 *fused_buf = (_Float16*)IOSurfaceGetBaseAddress(pls[L].ffnBwdFused_in);
                _Float16 *dsilu_src = (_Float16*)IOSurfaceGetBaseAddress(dk.ffnBwdW2t->ioOut);
                for (int d = 0; d < HIDDEN; d++) {
                    memcpy(fused_buf + d*FFN_BWD_FUSED_SP,         dsilu_src + d*SEQ, SEQ*2);
                    memcpy(fused_buf + d*FFN_BWD_FUSED_SP + SEQ,   ac->h1_fp16 + d*SEQ, SEQ*2);
                    memcpy(fused_buf + d*FFN_BWD_FUSED_SP + 2*SEQ, ac->h3_fp16 + d*SEQ, SEQ*2);
                }
                IOSurfaceUnlock(pls[L].ffnBwdFused_in, 0, NULL);
                IOSurfaceUnlock(dk.ffnBwdW2t->ioOut, kIOSurfaceLockReadOnly, NULL);
                t_io_bwd += tb_ms(mach_absolute_time() - t0);

                // OVERLAP 1: Dispatch fused kernel ASYNC, copy dW_W2 data while ANE runs
                t0 = mach_absolute_time();
                dispatch_semaphore_t sem_fused = ane_eval_req_async(dk.ffnBwdFused, plr[L].ffnBwdFused);
                // CPU work overlapped with ANE: copy dffn + silu_out for W2 gradient
                float *capt_dffn = (float*)malloc(SEQ*DIM*4); memcpy(capt_dffn, dffn, SEQ*DIM*4);
                float *capt_silu = (float*)malloc(SEQ*HIDDEN*4); cvt_f16_f32(capt_silu, ac->silu_out_fp16, SEQ*HIDDEN);
                float *capt_x2n = (float*)malloc(SEQ*DIM*4); memcpy(capt_x2n, ac->x2norm, SEQ*DIM*4);
                // Dispatch W2 gradient immediately (doesn't need dh1/dh3)
                dispatch_group_async(dw_grp, dw_q, ^{
                    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, DIM, HIDDEN, SEQ,
                                1.0f, capt_dffn, SEQ, capt_silu, SEQ, 1.0f, gr->W2, HIDDEN);
                    free(capt_dffn); free(capt_silu);
                });
                ane_eval_wait(sem_fused);
                t_ane_bwd += tb_ms(mach_absolute_time() - t0);

                // Read fused output: dx_ffn, dh1, dh3
                t0 = mach_absolute_time();
                io_read_ffn_bwd_fused(dk.ffnBwdFused->ioOut, dx_ffn, dh1, dh3);
                t_io_bwd += tb_ms(mach_absolute_time() - t0);

                // Dispatch W1/W3 gradients (need dh1/dh3 from fused kernel)
                float *capt_dh1 = (float*)malloc(SEQ*HIDDEN*4); memcpy(capt_dh1, dh1, SEQ*HIDDEN*4);
                float *capt_dh3 = (float*)malloc(SEQ*HIDDEN*4); memcpy(capt_dh3, dh3, SEQ*HIDDEN*4);
                dispatch_group_async(dw_grp, dw_q, ^{
                    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, HIDDEN, DIM, SEQ,
                                1.0f, capt_dh1, SEQ, capt_x2n, SEQ, 1.0f, gr->W1, DIM);
                    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, HIDDEN, DIM, SEQ,
                                1.0f, capt_dh3, SEQ, capt_x2n, SEQ, 1.0f, gr->W3, DIM);
                    free(capt_dh1); free(capt_dh3); free(capt_x2n);
                });

                // RMSNorm2 backward
                t0 = mach_absolute_time();
                rmsnorm_bwd(dx2, gr->rms_ffn, dx_ffn, ac->x2, lw[L].rms_ffn, DIM, SEQ);
                vDSP_vadd(dx2, 1, dy, 1, dx2, 1, (vDSP_Length)(SEQ*DIM));
                t_rms_bwd += tb_ms(mach_absolute_time() - t0);

                // OVERLAP 2: Dispatch wotBwd ASYNC, do dWo copy + GQA tile while ANE runs
                vDSP_vsmul(dx2, 1, &res_alpha, dx2_scaled, 1, (vDSP_Length)(SEQ*DIM));
                t0 = mach_absolute_time();
                write_wot_bwd_acts(pls[L].wotBwd_in, dx2_scaled);
                t_io_bwd += tb_ms(mach_absolute_time() - t0);
                t0 = mach_absolute_time();
                dispatch_semaphore_t sem_wot = ane_eval_req_async(dk.wotBwd, plr[L].wotBwd);
                // CPU work overlapped with ANE:
                // dWo copy+dispatch (needs dx2_scaled + attn_out, not da)
                float *capt_do = (float*)malloc(SEQ*DIM*4); memcpy(capt_do, dx2_scaled, SEQ*DIM*4);
                float *capt_attn = (float*)malloc(SEQ*Q_DIM*4); memcpy(capt_attn, ac->attn_out, SEQ*Q_DIM*4);
                dispatch_group_async(dw_grp, dw_q, ^{
                    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, DIM, Q_DIM, SEQ,
                                1.0f, capt_do, SEQ, capt_attn, SEQ, 1.0f, gr->Wo, Q_DIM);
                    free(capt_do); free(capt_attn);
                });
                // GQA tile K,V in fp16 (just memcpy, no conversion)
                gqa_tile_kv_fp16(k_tiled_fp16, ac->K_fp16, SEQ);
                gqa_tile_kv_fp16(v_tiled_fp16, ac->V_fp16, SEQ);
                ane_eval_wait(sem_wot);
                t_ane_bwd += tb_ms(mach_absolute_time() - t0);

                // SDPA backward part 1: all fp16 direct copies (Q, K_tiled, V_tiled, da)
                t0 = mach_absolute_time();
                {
                    IOSurfaceLock(dk.sdpaBwd1->ioIn, 0, NULL);
                    _Float16 *b = (_Float16*)IOSurfaceGetBaseAddress(dk.sdpaBwd1->ioIn);
                    memcpy(b,                  ac->Q_fp16,     Q_DIM*SEQ*2);
                    memcpy(b + Q_DIM*SEQ,      k_tiled_fp16,   Q_DIM*SEQ*2);
                    memcpy(b + 2*Q_DIM*SEQ,    v_tiled_fp16,   Q_DIM*SEQ*2);
                    // Direct fp16 copy: da from wotBwd output
                    IOSurfaceLock(dk.wotBwd->ioOut, kIOSurfaceLockReadOnly, NULL);
                    memcpy(b + 3*Q_DIM*SEQ, (_Float16*)IOSurfaceGetBaseAddress(dk.wotBwd->ioOut), Q_DIM*SEQ*2);
                    IOSurfaceUnlock(dk.wotBwd->ioOut, kIOSurfaceLockReadOnly, NULL);
                    IOSurfaceUnlock(dk.sdpaBwd1->ioIn, 0, NULL);
                }
                t_io_bwd += tb_ms(mach_absolute_time() - t0);
                t0 = mach_absolute_time();
                ane_eval(dk.sdpaBwd1);
                t_ane_bwd += tb_ms(mach_absolute_time() - t0);

                // OVERLAP 3: Dispatch sdpaBwd2 ASYNC, read dV + GQA reduce while ANE runs
                t0 = mach_absolute_time();
                {   // Batch copy + 2 writes under minimal locks
                    IOSurfaceLock(dk.sdpaBwd2->ioIn, 0, NULL);
                    IOSurfaceLock(dk.sdpaBwd1->ioOut, kIOSurfaceLockReadOnly, NULL);
                    _Float16 *dst = (_Float16*)IOSurfaceGetBaseAddress(dk.sdpaBwd2->ioIn);
                    _Float16 *src = (_Float16*)IOSurfaceGetBaseAddress(dk.sdpaBwd1->ioOut);
                    memcpy(dst, src + Q_DIM*SEQ, 2*SCORE_CH*SEQ*2);
                    IOSurfaceUnlock(dk.sdpaBwd1->ioOut, kIOSurfaceLockReadOnly, NULL);
                    memcpy(dst + 2*SCORE_CH*SEQ,           ac->Q_fp16,     Q_DIM*SEQ*2);
                    memcpy(dst + 2*SCORE_CH*SEQ+Q_DIM*SEQ, k_tiled_fp16,   Q_DIM*SEQ*2);
                    IOSurfaceUnlock(dk.sdpaBwd2->ioIn, 0, NULL);
                }
                t_io_bwd += tb_ms(mach_absolute_time() - t0);
                t0 = mach_absolute_time();
                dispatch_semaphore_t sem_bwd2 = ane_eval_async(dk.sdpaBwd2);
                // CPU work overlapped with ANE: read dV from sdpaBwd1 (already done) + GQA reduce
                io_read_fp16(dk.sdpaBwd1->ioOut, dv_full, 0,     Q_DIM, SEQ);
                gqa_reduce_kv(dv, dv_full, SEQ);
                ane_eval_wait(sem_bwd2);
                t_ane_bwd += tb_ms(mach_absolute_time() - t0);

                // Read sdpaBwd2 outputs (dQ, dK) — single lock
                t0 = mach_absolute_time();
                {
                    IOSurfaceLock(dk.sdpaBwd2->ioOut, kIOSurfaceLockReadOnly, NULL);
                    _Float16 *b = (_Float16*)IOSurfaceGetBaseAddress(dk.sdpaBwd2->ioOut);
                    cvt_f16_f32(dq_full, b, Q_DIM*SEQ);
                    cvt_f16_f32(dk_full, b + Q_DIM*SEQ, Q_DIM*SEQ);
                    IOSurfaceUnlock(dk.sdpaBwd2->ioOut, kIOSurfaceLockReadOnly, NULL);
                }
                t_io_bwd += tb_ms(mach_absolute_time() - t0);

                // GQA reduce dK (dV already done above during overlap)
                gqa_reduce_kv(dk_buf, dk_full, SEQ);

                // RoPE backward on dQ[Q_DIM] and dK[KV_DIM]
                rope_backward_inplace(dq_full, SEQ, Q_DIM, HD);
                rope_backward_inplace(dk_buf, SEQ, KV_DIM, HD);

                // dWq/dWk/dWv async copy+dispatch
                t0 = mach_absolute_time();
                float *capt_dq = (float*)malloc(SEQ*Q_DIM*4); memcpy(capt_dq, dq_full, SEQ*Q_DIM*4);
                float *capt_dk = (float*)malloc(SEQ*KV_DIM*4); memcpy(capt_dk, dk_buf, SEQ*KV_DIM*4);
                float *capt_dv = (float*)malloc(SEQ*KV_DIM*4); memcpy(capt_dv, dv, SEQ*KV_DIM*4);
                float *capt_xn = (float*)malloc(SEQ*DIM*4); memcpy(capt_xn, ac->xnorm, SEQ*DIM*4);
                t_dw_copy += tb_ms(mach_absolute_time() - t0);
                dispatch_group_async(dw_grp, dw_q, ^{
                    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, Q_DIM, DIM, SEQ,
                                1.0f, capt_dq, SEQ, capt_xn, SEQ, 1.0f, gr->Wq, DIM);
                    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, KV_DIM, DIM, SEQ,
                                1.0f, capt_dk, SEQ, capt_xn, SEQ, 1.0f, gr->Wk, DIM);
                    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, KV_DIM, DIM, SEQ,
                                1.0f, capt_dv, SEQ, capt_xn, SEQ, 1.0f, gr->Wv, DIM);
                    free(capt_dq); free(capt_dk); free(capt_dv); free(capt_xn);
                });

                // OVERLAP 4: Dispatch qBwd + kvBwd CONCURRENTLY on ANE
                t0 = mach_absolute_time();
                write_q_bwd_acts(pls[L].qBwd_in, dq_full);
                write_kv_bwd_acts(pls[L].kvBwd_in, dk_buf, dv);
                t_io_bwd += tb_ms(mach_absolute_time() - t0);
                t0 = mach_absolute_time();
                dispatch_semaphore_t sem_q = ane_eval_req_async(dk.qBwd, plr[L].qBwd);
                dispatch_semaphore_t sem_kv = ane_eval_req_async(dk.kvBwd, plr[L].kvBwd);
                ane_eval_wait(sem_q);
                ane_eval_wait(sem_kv);
                t_ane_bwd += tb_ms(mach_absolute_time() - t0);

                // Read both results
                t0 = mach_absolute_time();
                io_read_dyn(dk.qBwd->ioOut, dx_attn, DIM, SEQ);
                io_read_dyn(dk.kvBwd->ioOut, dx_kv, DIM, SEQ);
                t_io_bwd += tb_ms(mach_absolute_time() - t0);

                // dx_attn = dx_q + dx_kv
                vDSP_vadd(dx_attn, 1, dx_kv, 1, dx_attn, 1, (vDSP_Length)(SEQ*DIM));

                // RMSNorm1 backward
                t0 = mach_absolute_time();
                rmsnorm_bwd(dx_rms1, gr->rms_att, dx_attn, ac->layer_in, lw[L].rms_att, DIM, SEQ);
                vDSP_vadd(dx_rms1, 1, dx2, 1, dy, 1, (vDSP_Length)(SEQ*DIM));
                t_rms_bwd += tb_ms(mach_absolute_time() - t0);
            }

            // Embedding backward
            dispatch_group_wait(dw_grp, DISPATCH_TIME_FOREVER);
            embed_backward(gembed, dy, input_tokens, DIM, SEQ);

            double step_ms = tb_ms(mach_absolute_time() - t_step);
            total_train_ms += step_ms;
            total_steps_done++;

            if (step % 10 == 0 || step == start_step) {
                printf("  timing: ane_fwd=%.1f io_fwd=%.1f rms=%.1f ane_bwd=%.1f io_bwd=%.1f rms_bwd=%.1f cls=%.1f cblas_wait=%.1f dw_copy=%.1f\n",
                       t_ane_fwd, t_io_fwd, t_rms, t_ane_bwd, t_io_bwd, t_rms_bwd, t_cls, t_cblas_wait, t_dw_copy);
                float xmx, xmn;
                vDSP_maxv(x_cur,1,&xmx,(vDSP_Length)(SEQ*DIM));
                vDSP_minv(x_cur,1,&xmn,(vDSP_Length)(SEQ*DIM));
                float dmx, dmn;
                vDSP_maxv(dy,1,&dmx,(vDSP_Length)(SEQ*DIM));
                vDSP_minv(dy,1,&dmn,(vDSP_Length)(SEQ*DIM));
                printf("step %-4d loss=%.4f  lr=%.2e  %.1fms/step  x[%.2f,%.2f] dy[%.3e,%.3e]\n",
                       step, loss, lr, step_ms, xmn, xmx, dmn, dmx);
            }

            // Adam update every accum_steps
            if ((step+1) % accum_steps == 0 || step == total_steps-1) {
                dispatch_group_wait(dw_grp, DISPATCH_TIME_FOREVER);
                float gsc = 1.0f / (accum_steps * loss_scale);
                adam_t++;

                // Scale gradients (vDSP vectorized)
                for (int L=0; L<NLAYERS; L++) {
                    LayerGrads *g = &grads[L];
                    vDSP_vsmul(g->Wq,1,&gsc,g->Wq,1,(vDSP_Length)WQ_SZ);
                    vDSP_vsmul(g->Wk,1,&gsc,g->Wk,1,(vDSP_Length)WK_SZ);
                    vDSP_vsmul(g->Wv,1,&gsc,g->Wv,1,(vDSP_Length)WV_SZ);
                    vDSP_vsmul(g->Wo,1,&gsc,g->Wo,1,(vDSP_Length)WO_SZ);
                    vDSP_vsmul(g->W1,1,&gsc,g->W1,1,(vDSP_Length)W1_SZ);
                    vDSP_vsmul(g->W2,1,&gsc,g->W2,1,(vDSP_Length)W2_SZ);
                    vDSP_vsmul(g->W3,1,&gsc,g->W3,1,(vDSP_Length)W3_SZ);
                    vDSP_vsmul(g->rms_att,1,&gsc,g->rms_att,1,(vDSP_Length)DIM);
                    vDSP_vsmul(g->rms_ffn,1,&gsc,g->rms_ffn,1,(vDSP_Length)DIM);
                }
                vDSP_vsmul(grms_final,1,&gsc,grms_final,1,(vDSP_Length)DIM);
                vocab_scatter_grads(gembed, gcembed, &vm, DIM);
                vDSP_vsmul(gembed,1,&gsc,gembed,1,(vDSP_Length)(VOCAB*DIM));

                // Global gradient norm (single pass, accumulate attn/ffn/rms breakdown)
                float attn_sq=0, ffn_sq=0, rms_sq=0;
                for (int L=0; L<NLAYERS; L++) {
                    LayerGrads *g = &grads[L];
                    float s;
                    vDSP_dotpr(g->Wq,1,g->Wq,1,&s,(vDSP_Length)WQ_SZ); attn_sq+=s;
                    vDSP_dotpr(g->Wk,1,g->Wk,1,&s,(vDSP_Length)WK_SZ); attn_sq+=s;
                    vDSP_dotpr(g->Wv,1,g->Wv,1,&s,(vDSP_Length)WV_SZ); attn_sq+=s;
                    vDSP_dotpr(g->Wo,1,g->Wo,1,&s,(vDSP_Length)WO_SZ); attn_sq+=s;
                    vDSP_dotpr(g->W1,1,g->W1,1,&s,(vDSP_Length)W1_SZ); ffn_sq+=s;
                    vDSP_dotpr(g->W2,1,g->W2,1,&s,(vDSP_Length)W2_SZ); ffn_sq+=s;
                    vDSP_dotpr(g->W3,1,g->W3,1,&s,(vDSP_Length)W3_SZ); ffn_sq+=s;
                    vDSP_dotpr(g->rms_att,1,g->rms_att,1,&s,(vDSP_Length)DIM); rms_sq+=s;
                    vDSP_dotpr(g->rms_ffn,1,g->rms_ffn,1,&s,(vDSP_Length)DIM); rms_sq+=s;
                }
                float embed_sq;
                { float s;
                  vDSP_dotpr(grms_final,1,grms_final,1,&s,(vDSP_Length)DIM); rms_sq+=s;
                  vDSP_dotpr(gembed,1,gembed,1,&s,(vDSP_Length)(VOCAB*DIM)); embed_sq=s;
                }
                float grad_norm_sq = attn_sq + ffn_sq + rms_sq + embed_sq;
                float grad_norm = sqrtf(grad_norm_sq);
                if ((step+1) % 10 == 0) {
                    printf("  grad_norm=%.4f  attn=%.4f ffn=%.4f embed=%.4f\n",
                           grad_norm, sqrtf(attn_sq), sqrtf(ffn_sq), sqrtf(embed_sq));
                }

                // Gradient clipping
                if (grad_clip > 0 && grad_norm > grad_clip) {
                    float clip_scale = grad_clip / grad_norm;
                    for (int L=0; L<NLAYERS; L++) {
                        LayerGrads *g = &grads[L];
                        vDSP_vsmul(g->Wq,1,&clip_scale,g->Wq,1,(vDSP_Length)WQ_SZ);
                        vDSP_vsmul(g->Wk,1,&clip_scale,g->Wk,1,(vDSP_Length)WK_SZ);
                        vDSP_vsmul(g->Wv,1,&clip_scale,g->Wv,1,(vDSP_Length)WV_SZ);
                        vDSP_vsmul(g->Wo,1,&clip_scale,g->Wo,1,(vDSP_Length)WO_SZ);
                        vDSP_vsmul(g->W1,1,&clip_scale,g->W1,1,(vDSP_Length)W1_SZ);
                        vDSP_vsmul(g->W2,1,&clip_scale,g->W2,1,(vDSP_Length)W2_SZ);
                        vDSP_vsmul(g->W3,1,&clip_scale,g->W3,1,(vDSP_Length)W3_SZ);
                        vDSP_vsmul(g->rms_att,1,&clip_scale,g->rms_att,1,(vDSP_Length)DIM);
                        vDSP_vsmul(g->rms_ffn,1,&clip_scale,g->rms_ffn,1,(vDSP_Length)DIM);
                    }
                    vDSP_vsmul(grms_final,1,&clip_scale,grms_final,1,(vDSP_Length)DIM);
                    vDSP_vsmul(gembed,1,&clip_scale,gembed,1,(vDSP_Length)(VOCAB*DIM));
                }

                // Cosine LR schedule with warmup
                if (step < warmup_steps) {
                    lr = max_lr * ((float)(step + 1)) / warmup_steps;
                } else {
                    float decay_ratio = (float)(step - warmup_steps) / (float)(total_steps - warmup_steps);
                    float min_lr = max_lr * min_lr_frac;
                    lr = min_lr + 0.5f * (1.0f + cosf(M_PI * decay_ratio)) * (max_lr - min_lr);
                }

                // Adam update
                for (int L=0; L<NLAYERS; L++) {
                    LayerGrads *g = &grads[L];
                    adam_update(lw[L].Wq, g->Wq, &la[L].Wq, adam_t, lr, adam_b1, adam_b2, adam_eps, wd);
                    adam_update(lw[L].Wk, g->Wk, &la[L].Wk, adam_t, lr, adam_b1, adam_b2, adam_eps, wd);
                    adam_update(lw[L].Wv, g->Wv, &la[L].Wv, adam_t, lr, adam_b1, adam_b2, adam_eps, wd);
                    adam_update(lw[L].Wo, g->Wo, &la[L].Wo, adam_t, lr, adam_b1, adam_b2, adam_eps, wd);
                    adam_update(lw[L].W1, g->W1, &la[L].W1, adam_t, lr, adam_b1, adam_b2, adam_eps, wd);
                    adam_update(lw[L].W2, g->W2, &la[L].W2, adam_t, lr, adam_b1, adam_b2, adam_eps, wd);
                    adam_update(lw[L].W3, g->W3, &la[L].W3, adam_t, lr, adam_b1, adam_b2, adam_eps, wd);
                    adam_update(lw[L].rms_att, g->rms_att, &la[L].rms_att, adam_t, lr, adam_b1, adam_b2, adam_eps, 0.0f);
                    adam_update(lw[L].rms_ffn, g->rms_ffn, &la[L].rms_ffn, adam_t, lr, adam_b1, adam_b2, adam_eps, 0.0f);

                    // Update transposed weight buffers
                    transpose_weight(Wqt_buf[L], lw[L].Wq, Q_DIM, DIM);
                    transpose_weight(Wkt_buf[L], lw[L].Wk, KV_DIM, DIM);
                    transpose_weight(Wvt_buf[L], lw[L].Wv, KV_DIM, DIM);
                    transpose_weight(Wot_buf[L], lw[L].Wo, DIM, Q_DIM);
                    transpose_weight(W1t_buf[L], lw[L].W1, HIDDEN, DIM);
                    transpose_weight(W2t_buf[L], lw[L].W2, DIM, HIDDEN);
                    transpose_weight(W3t_buf[L], lw[L].W3, HIDDEN, DIM);

                    // Re-stage weights
                    stage_sdpa_fwd_weights(pls[L].sdpaFwd_in, Wqt_buf[L], Wkt_buf[L], Wvt_buf[L]);
                    stage_wo_fwd_weights(pls[L].woFwd_in, Wot_buf[L]);
                    stage_ffn_fused_weights(pls[L].ffnFused_in, W1t_buf[L], W3t_buf[L], lw[L].W2);
                    stage_ffn_bwd_w2t_weights(pls[L].ffnBwdW2t_in, lw[L].W2);
                    stage_ffn_bwd_w13t_weights(pls[L].ffnBwdW13t_in, lw[L].W1, lw[L].W3);
                    stage_ffn_bwd_fused_weights(pls[L].ffnBwdFused_in, lw[L].W1, lw[L].W3);
                    stage_wot_bwd_weights(pls[L].wotBwd_in, lw[L].Wo);
                    stage_q_bwd_weights(pls[L].qBwd_in, lw[L].Wq);
                    stage_kv_bwd_weights(pls[L].kvBwd_in, lw[L].Wk, lw[L].Wv);
                }
                adam_update(rms_final, grms_final, &arms_final, adam_t, lr, adam_b1, adam_b2, adam_eps, 0.0f);
                adam_update(embed, gembed, &aembed, adam_t, lr, adam_b1, adam_b2, adam_eps, wd);
                free(cembed);
                cembed = vocab_compact_embed(embed, &vm, DIM);
                // Zero grads
                for (int L=0; L<NLAYERS; L++) layer_grads_zero(&grads[L]);
                memset(grms_final, 0, DIM*4);
                memset(gembed, 0, (size_t)VOCAB*DIM*4);
                memset(gcembed, 0, (size_t)CV*DIM*4);

                // Checkpoint — only save on best loss
                if ((step+1) % 100 == 0 && last_loss < best_loss) {
                    best_loss = last_loss;
                    double wall = tb_ms(mach_absolute_time() - t_wall_start);
                    save_checkpoint(CKPT_PATH, step+1, total_steps, lr, last_loss,
                        total_train_ms+cum_train, wall+cum_wall, total_steps_done+cum_steps, adam_t,
                        lw, la, rms_final, &arms_final, embed, &aembed);
                    printf("  [ckpt saved, best_loss=%.4f]\n", best_loss);
                }
            }
        }

        // Report
        double wall = tb_ms(mach_absolute_time() - t_wall_start);
        printf("\n=== Efficiency Report ===\n");
        printf("Total steps:  %d\n", total_steps_done);
        printf("Compile:      %.0fms (one-time, %.1f%%)\n", compile_ms, 100*compile_ms/(wall+cum_wall));
        printf("Train time:   %.0fms (%.1fms/step)\n", total_train_ms, total_train_ms/total_steps_done);
        printf("Wall time:    %.1fs\n", (wall+cum_wall)/1000);

        // Cleanup
        for (int L=0; L<NLAYERS; L++) {
            layer_weights_free(&lw[L]); layer_adam_free(&la[L]);
            layer_acts_free(&acts[L]); layer_grads_free(&grads[L]);
            free(Wqt_buf[L]); free(Wkt_buf[L]); free(Wvt_buf[L]); free(Wot_buf[L]);
            free(W1t_buf[L]); free(W2t_buf[L]); free(W3t_buf[L]);
        }
        free_per_layer(pls, plr);
        free_kern(dk.sdpaFwd); free_kern(dk.woFwd); free_kern(dk.ffnFused);
        free_kern(dk.ffnBwdW2t); free_kern(dk.ffnBwdW13t); free_kern(dk.ffnBwdFused); free_kern(dk.wotBwd);
        free_kern(dk.sdpaBwd1); free_kern(dk.sdpaBwd2);
        free_kern(dk.qBwd); free_kern(dk.kvBwd);
        free(k_tiled_fp16); free(v_tiled_fp16);
        free(dq_full); free(dk_full); free(dv_full);
        free(dk_buf); free(dv);
        munmap(token_data, data_len); close(data_fd);
    }
    return 0;
}
