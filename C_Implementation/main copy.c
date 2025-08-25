// slice_test.c
#include <stdio.h>
#include <math.h>
#include <stdint.h>
#include "weights_io.h"

// ======= Compile-time sizes (edit to match your slice) =======
enum {
    VOCAB_SIZE = 128,  // Emb: (128, 32)
    E  = 32,           // embedding_dims
    T  = 44,           // hist_len
    K  = 3,            // ConvW: (3, 32, 2)
    F  = 2,            // conv filters (Cout)
    P  = 7,            // pool_width
    H  = 16,           // LSTM hidden size
    M  = 8             // <<< head hidden_neurons (Dense 16->8)
};

// ======= utils =======
static inline float sigmoidf(float x){ return 1.0f/(1.0f+expf(-x)); }
static inline float tanhf_fast(float x){ return tanhf(x); } // alias
static inline float reluf(float x){ return x > 0.0f ? x : 0.0f; }

// ------- Dense (fully-connected): y[Out] = x[In] * W[In,Out] + b[Out] ------
static inline void dense_forward(const float *x, int In,
                                 const float *W, const float *b, int Out,
                                 float *y)
{
    for (int j=0; j<Out; ++j) {
        float acc = b ? b[j] : 0.0f;
        for (int i=0; i<In; ++i) {
            acc += x[i] * W[i*Out + j]; // W row-major [In][Out]
        }
        y[j] = acc;
    }
}

// ======= BatchNorm over last dim =======
static inline void bn_time_channel(float *Y, int Tlen, int C,
                                   const float *gamma, const float *beta,
                                   const float *mean, const float *var, float eps)
{
    for (int t=0; t<Tlen; ++t) {
        float *yt = Y + t*C;
        for (int c=0; c<C; ++c) {
            float norm = (yt[c] - mean[c]) / sqrtf(var[c] + eps);
            yt[c] = gamma[c]*norm + beta[c];
        }
    }
}
static inline void bn_vector(float *v, int C,
                             const float *gamma, const float *beta,
                             const float *mean, const float *var, float eps)
{
    for (int c=0; c<C; ++c) {
        float norm = (v[c] - mean[c]) / sqrtf(var[c] + eps);
        v[c] = gamma[c]*norm + beta[c];
    }
}

// ======= Activation =======
typedef enum { ACT_RELU, ACT_TANH, ACT_SIGMOID } act_t;
static inline void apply_activation_time(float *Y, int Tlen, int C, act_t a) {
    for (int t=0; t<Tlen; ++t) {
        float *yt = Y + t*C;
        for (int c=0; c<C; ++c) {
            float x = yt[c];
            if (a == ACT_RELU)      yt[c] = (x>0.0f)? x : 0.0f;
            else if (a == ACT_TANH) yt[c] = tanhf_fast(x);
            else                     yt[c] = sigmoidf(x);
        }
    }
}
// vector helper
static inline void apply_activation_vec(float *v, int C, act_t a) {
    for (int c=0; c<C; ++c) {
        float x = v[c];
        if      (a == ACT_RELU)    v[c] = reluf(x);
        else if (a == ACT_TANH)    v[c] = tanhf_fast(x);
        else                       v[c] = sigmoidf(x);
    }
}

// ======= AveragePooling1D(pool=P, stride=P, valid) =======
static inline void avgpool1d_P(const float *Z, int Tlen, int C, int Psz, float *U) {
    int Tout = Tlen / Psz; // stride=P, valid
    for (int u=0; u<Tout; ++u) {
        for (int c=0; c<C; ++c) {
            float acc = 0.0f;
            for (int p=0; p<Psz; ++p) acc += Z[(u*Psz + p)*C + c];
            U[u*C + c] = acc / (float)Psz;
        }
    }
}

// ======= LSTM (single layer, unidir), gate order [i,f,o,g] internally =======
static inline void lstm_forward_unidir(const float *x, int Tlen, int D,
                                       const float *W_ifog, // [D][4H]
                                       const float *R_ifog, // [H][4H]
                                       const float *b_ifog, // [4H]
                                       int Hsz,
                                       float *h_last, float *c_last)
{
    for (int j=0;j<Hsz;++j){ h_last[j]=0.0f; c_last[j]=0.0f; }

    for (int t=0; t<Tlen; ++t) {
        float z[4*H] = {0};
        for (int gk=0; gk<4*Hsz; ++gk) z[gk] = b_ifog[gk];

        const float *xt = x + t*D;
        for (int d=0; d<D; ++d) {
            const float xv = xt[d];
            const float *Wd = W_ifog + d*(4*Hsz);
            for (int gk=0; gk<4*Hsz; ++gk) z[gk] += xv * Wd[gk];
        }
        for (int hp=0; hp<Hsz; ++hp) {
            const float hv = h_last[hp];
            const float *Rh = R_ifog + hp*(4*Hsz);
            for (int gk=0; gk<4*Hsz; ++gk) z[gk] += hv * Rh[gk];
        }
        for (int j=0; j<Hsz; ++j) {
            float i = sigmoidf(z[0*Hsz + j]);
            float f = sigmoidf(z[1*Hsz + j]);
            float o = sigmoidf(z[2*Hsz + j]);
            float g = tanhf_fast(z[3*Hsz + j]);
            float c = f * c_last[j] + i * g;
            float h = o * tanhf_fast(c);
            c_last[j] = c;
            h_last[j] = h;
        }
    }
}

// --- Embedding: X[t,E] = Emb[tokens[t], :] ---
static inline void embedding_forward(const int *tokens, float *X, const float *Emb)
{
    for (int t = 0; t < T; ++t) {
        int idx = tokens[t];
        if (idx < 0 || idx >= VOCAB_SIZE) idx = 0; // clamp
        const float *row = Emb + idx * E;
        float *dst = X + t * E;
        for (int e = 0; e < E; ++e) dst[e] = row[e];
    }
}

// --- Conv1D (Keras 'valid'): in X[T,Cin], W[K,Cin,Cout], B[Cout] -> Y[T-K+1,Cout] ---
static inline void conv1d_keras_valid(const float *X, int Tlen, int Cin,
                                      const float *W, int Ksz, int Cout,
                                      const float *B, float *Y)
{
    const int T1 = Tlen - Ksz + 1;
    for (int t = 0; t < T1; ++t) {
        for (int f = 0; f < Cout; ++f) {
            float acc = B ? B[f] : 0.0f;
            for (int k = 0; k < Ksz; ++k) {
                const float *xrow = X + (t + k) * Cin;
                const float *wf   = W + (k * Cin) * Cout + f; // W[k,e,f] packed as [K][Cin][Cout]
                for (int e = 0; e < Cin; ++e) acc += xrow[e] * wf[e * Cout];
            }
            Y[t * Cout + f] = acc;
        }
    }
}

// ======= Slice forward =======
static void slice_forward(const int *tokens,
                          // Embedding
                          const float *Emb,
                          // Conv1D
                          const float *ConvW, const float *ConvB,
                          const float *BN1_gamma, const float *BN1_beta,
                          const float *BN1_mean,  const float *BN1_var, float BN1_eps,
                          act_t conv_act,
                          // LSTM
                          const float *LSTM_W_ifog, const float *LSTM_R_ifog, const float *LSTM_b_ifog,
                          const float *BN2_gamma, const float *BN2_beta,
                          const float *BN2_mean,  const float *BN2_var, float BN2_eps,
                          // output buffer
                          float *out_tanh // [H]
                          )
{
    // 1) Embedding -> X0[T][E]
    float X0[T*E];
    embedding_forward(tokens, X0, Emb);

    // 2) Conv1D VALID -> Y[T1][F]
    const int T1 = T - K + 1;
    float Y[T1*F];
    conv1d_keras_valid(X0, T, E, ConvW, K, F, ConvB, Y);

    // 3) BN + Activation
    bn_time_channel(Y, T1, F, BN1_gamma, BN1_beta, BN1_mean, BN1_var, BN1_eps);
    apply_activation_time(Y, T1, F, conv_act);

    // 4) AvgPool1D(pool=P, stride=P) -> U[T2][F]
    const int T2 = T1 / P;
    float U[T2*F];
    avgpool1d_P(Y, T1, F, P, U);

    // 5) LSTM (D=F, T=T2)
    float h_last[H], c_last[H];
    lstm_forward_unidir(U, T2, F, LSTM_W_ifog, LSTM_R_ifog, LSTM_b_ifog, H, h_last, c_last);

    // 6) BN (vector) + tanh
    bn_vector(h_last, H, BN2_gamma, BN2_beta, BN2_mean, BN2_var, BN2_eps);
    for (int j=0; j<H; ++j) out_tanh[j] = tanhf_fast(h_last[j]);
}

static void path_join(char *dst, size_t cap, const char *dir, const char *file) {
    size_t n = snprintf(dst, cap, "%s/%s", dir, file);
    if (n >= cap) { fprintf(stderr, "path too long\n"); exit(1); }
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <weights_dir>\n", argv[0]);
        fprintf(stderr, "example: %s weights_out/slice1\n", argv[0]);
        return 1;
    }
    const char *WDIR = argv[1];

    // ---- allocate static buffers for weights ----
    static float Emb[VOCAB_SIZE * E];
    static float ConvW[K * E * F];
    static float ConvB[F];
    static float BN1_gamma[F], BN1_beta[F], BN1_mean[F], BN1_var[F];
    static float LSTM_W_ifog[F * 4 * H];
    static float LSTM_R_ifog[H * 4 * H];
    static float LSTM_b_ifog[4 * H];
    static float BN2_gamma[H], BN2_beta[H], BN2_mean[H], BN2_var[H];
    const float BN1_eps = 1e-3f;
    const float BN2_eps = 1e-3f;

    // ---- head weights: fc_0 (16->8), BN over 8, output (8->1) ----
    static float FC0_W[H * M];      // [16,8]
    static float FC0_b[M];          // [8]
    static float FC0_BN_gamma[M];   // [8]
    static float FC0_BN_beta[M];    // [8]
    static float FC0_BN_mean[M];    // [8]
    static float FC0_BN_var[M];     // [8]
    const float FC0_BN_eps = 1e-3f;

    static float OUT_W[M * 1];      // [8,1]
    static float OUT_b[1];          // [1]

    // ---- load .bin files ----
    char path[512];
    path_join(path, sizeof(path), WDIR, "Emb.bin");                 load_bin_f32(path, Emb, VOCAB_SIZE*E);
    path_join(path, sizeof(path), WDIR, "ConvW.bin");               load_bin_f32(path, ConvW, K*E*F);
    path_join(path, sizeof(path), WDIR, "ConvB.bin");               load_bin_f32(path, ConvB, F);
    path_join(path, sizeof(path), WDIR, "BN1_gamma.bin");           load_bin_f32(path, BN1_gamma, F);
    path_join(path, sizeof(path), WDIR, "BN1_beta.bin");            load_bin_f32(path, BN1_beta,  F);
    path_join(path, sizeof(path), WDIR, "BN1_mean.bin");            load_bin_f32(path, BN1_mean,  F);
    path_join(path, sizeof(path), WDIR, "BN1_var.bin");             load_bin_f32(path, BN1_var,   F);
    path_join(path, sizeof(path), WDIR, "LSTM_W_ifog.bin");         load_bin_f32(path, LSTM_W_ifog, F*4*H);
    path_join(path, sizeof(path), WDIR, "LSTM_R_ifog.bin");         load_bin_f32(path, LSTM_R_ifog, H*4*H);
    path_join(path, sizeof(path), WDIR, "LSTM_b_ifog.bin");         load_bin_f32(path, LSTM_b_ifog, 4*H);
    path_join(path, sizeof(path), WDIR, "BN2_gamma.bin");           load_bin_f32(path, BN2_gamma, H);
    path_join(path, sizeof(path), WDIR, "BN2_beta.bin");            load_bin_f32(path, BN2_beta,  H);
    path_join(path, sizeof(path), WDIR, "BN2_mean.bin");            load_bin_f32(path, BN2_mean,  H);
    path_join(path, sizeof(path), WDIR, "BN2_var.bin");             load_bin_f32(path, BN2_var,   H);

    // ---- load head (match your exporter filenames) ----
    path_join(path, sizeof(path), WDIR, "fc_0_W.bin");              load_bin_f32(path, FC0_W, H*M);
    path_join(path, sizeof(path), WDIR, "fc_0_b.bin");              load_bin_f32(path, FC0_b, M);
    path_join(path, sizeof(path), WDIR, "fc_0_bn_gamma.bin");       load_bin_f32(path, FC0_BN_gamma, M);
    path_join(path, sizeof(path), WDIR, "fc_0_bn_beta.bin");        load_bin_f32(path, FC0_BN_beta,  M);
    path_join(path, sizeof(path), WDIR, "fc_0_bn_mean.bin");        load_bin_f32(path, FC0_BN_mean,  M);
    path_join(path, sizeof(path), WDIR, "fc_0_bn_var.bin");         load_bin_f32(path, FC0_BN_var,   M);

    path_join(path, sizeof(path), WDIR, "output_W.bin");            load_bin_f32(path, OUT_W, M*1);
    path_join(path, sizeof(path), WDIR, "output_b.bin");            load_bin_f32(path, OUT_b, 1);

    // ---- tokens: prefer tokens.bin if present; else synthesize simple pattern ----
    static int tokens[T];
    FILE *ft = NULL;
    path_join(path, sizeof(path), WDIR, "tokens.bin");
    ft = fopen(path, "rb");
    if (ft) {
        size_t got = fread(tokens, sizeof(int), T, ft);
        fclose(ft);
        if (got != (size_t)T) { fprintf(stderr, "tokens.bin length != T\n"); return 1; }
    } else {
        for (int t=0; t<T; ++t) tokens[t] = t % VOCAB_SIZE;
    }

    // ---- run the slice ----
    float slice_vec[H] = {0}; // tanh(BN(h_last)) from LSTM
    slice_forward(tokens,
                  Emb,
                  ConvW, ConvB,
                  BN1_gamma, BN1_beta, BN1_mean, BN1_var, BN1_eps,
                  /*conv_act=*/ACT_RELU,   // matches conv_act='relu'
                  LSTM_W_ifog, LSTM_R_ifog, LSTM_b_ifog,
                  BN2_gamma, BN2_beta, BN2_mean, BN2_var, BN2_eps,
                  slice_vec);

    // ---- head: Dense(16->8) -> BN -> ReLU ----
    float z1[M];
    dense_forward(slice_vec, H, FC0_W, FC0_b, M, z1);                 // fc_0
    bn_vector(z1, M, FC0_BN_gamma, FC0_BN_beta,                       // fc_0_bn
              FC0_BN_mean, FC0_BN_var, FC0_BN_eps);
    apply_activation_vec(z1, M, ACT_RELU);                            // fc_0_act (relu)

    // ---- output: Dense(8->1) -> Sigmoid ----
    float y_lin[1];
    dense_forward(z1, M, OUT_W, OUT_b, 1, y_lin);                     // output (pre-sigmoid)
    float y_hat = sigmoidf(y_lin[0]);                                  // final probability

    // ---- print outputs ----
    printf("slice_out = [");
    for (int j=0; j<H; ++j) printf("%s%.6f", j?", ":"", slice_vec[j]);
    printf("]\n");
    printf("y_hat = %.6f\n", y_hat);

    return 0;
}
