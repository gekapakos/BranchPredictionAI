// main_multi.c  (bin-free, header-backed)

// ===== includes =====
#include <stdio.h>
#include <stdint.h>
#include <math.h>

// bring in the generated arrays
#include "slice0_weights.h"
#include "slice1_weights.h"
#include "slice2_weights.h"
#include "slice3_weights.h"
#include "slice4_weights.h"
#include "head_weights.h"

// ===== model constants (match Python config) =====
enum { VOCAB_SIZE=4096, E=32, K=7, F=32, H=32 };
enum { M0=128, M1=128 };

// Fixed T/P per slice (no VLA)
enum { T0=42,  P0=3,  T10=T0-K+1,  T20=T10/P0 };
enum { T1=78,  P1=6,  T11=T1-K+1,  T21=T11/P1 };
enum { T2=150, P2=12, T12=T2-K+1,  T22=T12/P2 };
enum { T3=294, P3=24, T13=T3-K+1,  T23=T13/P3 };
enum { T4=582, P4=48, T14=T4-K+1,  T24=T14/P4 };

// global arrays
static float h_slice[H], c_slice[H];

// ===== small utils =====
static inline float sigmoidf(float x) { 
    #pragma HLS INLINE
    return 1.0f/(1.0f+expf(-x));
}

static inline float tanhf_fast(float x) { 
    #pragma HLS INLINE
    return tanhf(x); 
}

static inline float reluf(float x) { 
    #pragma HLS INLINE
    return x>0.0f? x:0.0f;
}

// ===== layers (dimensioned by args, not globals) =====
static inline void embedding_forward_dyn(const int32_t *tokens, int Tlen,
                                         const float *Emb, float *X) // X[Tlen,E]
{
    for(int t=0;t<Tlen;++t){
        int idx = (int)tokens[t];
        if(idx < 0 || idx >= VOCAB_SIZE) idx=0;

        const float *row = Emb + idx*E;
        float *dst = X + t*E;
        for(int e=0;e<E;++e) { 
            dst[e]=row[e];
        }
    }
}

typedef enum { ACT_RELU, ACT_TANH, ACT_SIGMOID } act_t;
static inline void conv_bn_act_pool(const float *X,int Tlen,
                                    const float *W,int Ksz, // W[Ksz,E,F]
                                    const float *B, float *Y, // Y[(Tlen-Ksz+1),F]
                                    int C, const float *gamma,
                                    const float *beta, const float *mean,
                                    const float *var, float eps, act_t a,
                                    int Psz, float *U) {
    const int T1 = Tlen - Ksz + 1;
    int t, c;

    for(t=0;t<T1;++t) {
        for(int f=0; f<F; ++f){
            float acc = B ? B[f] : 0.0f;
            for(int k=0;k<Ksz;++k){
                const float *xrow = X + (t+k)*E;
                const float *wf   = W + (k*E)*F + f; // stride by Cout on e
                for(int e=0;e<E;++e) acc += xrow[e] * wf[e*F];
            }
            Y[t*F + f] = acc;
        }
    }

    for(t=0;t<Tlen;++t) {
        float *yt = Y + t*C;
        for(c=0;c<C;++c){
            float nrm = (yt[c]-mean[c]) / sqrtf(var[c]+eps);
            yt[c] = gamma[c]*nrm + beta[c];
            float x = yt[c];
            yt[c] = (a == ACT_RELU) ? reluf(x) : (a == ACT_TANH? tanhf_fast(x) : sigmoidf(x));
        }
    }

    // for(t = 0;t < Tlen; ++t) {
    //     float *yt = Y + t*C;
    //     for(c = 0;c < C;++c) {
    //         float x = yt[c];
    //         yt[c] = (a == ACT_RELU) ? reluf(x) : (a == ACT_TANH? tanhf_fast(x) : sigmoidf(x));
    //     }
    // }

    const int Tout = Tlen / Psz;
    for(int u = 0;u < Tout; ++u) {
        for(c = 0;c < C; ++c){
            float acc=0.0f;
            for(int p=0;p<Psz;++p) acc += Y[(u*Psz + p)*C + c];
            U[u*C + c] = acc / (float)Psz;
        }
    }
}

// LSTM (expects gate order [i,f,o,g] in W,R,b)
static inline void lstm_forward_unidir(const float *x,int Tlen,int D,
    const float *W_ifog,const float *R_ifog,const float *b_ifog,
    float *h_last,float *c_last)
{
    int j;
    for(j=0;j<H;++j){ h_last[j]=0.0f; c_last[j]=0.0f; }
    float z[4*H];

    for(int t=0;t<Tlen;++t){
        // z = b
        for(int g=0; g<4*H; ++g) z[g] = b_ifog[g];

        // input part
        const float *xt = x + t*D;
        for(int d=0; d<D; ++d){
            float xv = xt[d];
            const float *Wd = W_ifog + d*(4*H);
            for(int g=0; g<4*H; ++g) z[g] += xv * Wd[g];
        }
        // recurrent part
        for(int hp=0; hp<H; ++hp){
            float hv = h_last[hp];
            const float *Rh = R_ifog + hp*(4*H);
            for(int g=0; g<4*H; ++g) z[g] += hv * Rh[g];
        }
        // gates
        for(j=0;j<H;++j){
            float i = 1.f/(1.f+expf(-z[0*H + j]));
            float f = 1.f/(1.f+expf(-z[1*H + j]));
            float o = 1.f/(1.f+expf(-z[2*H + j]));
            float g = tanhf_fast(z[3*H + j]);
            float c = f * c_last[j] + i * g;
            float h = o * tanhf_fast(c);
            c_last[j]=c; h_last[j]=h;
        }
    }
}

static inline void bn_vector(float *v,int C, const float *gamma,const float *beta,const float *mean,const float *var,float eps)
{
    #pragma HLS INLINE
    for(int c = 0; c < C; ++c){
        float n = (v[c]-mean[c]) / sqrtf(var[c]+eps);
        v[c] = gamma[c]*n + beta[c];
    }
}

static inline void dense_forward(const float *x,int In,
    const float *W,const float *b,int Out,float *y)
{
    float acc;
    for(int j = 0; j < Out; ++j) {
        acc = b ? b[j] : 0.0f;
        for(int i = 0; i < In; ++i) {
            acc += x[i] * W[i * Out + j];
        }
        y[j] = acc;
    }
}

static float Y[T14*F];
static float U_slice[T24*F];

// ======================= run all 5 slices (using header arrays) =======================
static void run_all_slices_unrolled(float merged[H]) {
    const float BN_eps = 1e-3f;
    int j;
    for (j = 0; j < H; ++j) merged[j] = 0.0f;

    // -------- slice 0 --------
    {
        float X_slice[T0*E];
        // #pragma HLS ARRAY_PARTITION variable=Emb0 block factor=128
        embedding_forward_dyn(tokens0, T0, Emb0, X_slice);
        conv_bn_act_pool(X_slice, T0, ConvW0, K, ConvB0, Y, T10,
        BN1_gamma0, BN1_beta0, BN1_mean0, BN1_var0, BN_eps,
        ACT_RELU, P0, U_slice);
        lstm_forward_unidir(U_slice, T20, F, LSTM_W_ifog0, LSTM_R_ifog0, LSTM_b_ifog0, h_slice, c_slice);
        bn_vector(h_slice, H, BN2_gamma0, BN2_beta0, BN2_mean0, BN2_var0, BN_eps);
        for (j=0;j<H;++j) merged[j] += tanhf_fast(h_slice[j]);
    }

    // -------- slice 1 --------
    {
        float X_slice[T1*E];
		// #pragma HLS ARRAY_PARTITION variable=Emb1 block factor=128
        embedding_forward_dyn(tokens1, T1, Emb1, X_slice);
        conv_bn_act_pool(X_slice, T1, ConvW1, K, ConvB1, Y, T11,
        BN1_gamma1, BN1_beta1, BN1_mean1, BN1_var1, BN_eps,
        ACT_RELU, P1, U_slice);
        lstm_forward_unidir(U_slice, T21, F, LSTM_W_ifog1, LSTM_R_ifog1, LSTM_b_ifog1, h_slice, c_slice);
        bn_vector(h_slice, H, BN2_gamma1, BN2_beta1, BN2_mean1, BN2_var1, BN_eps);
        for (j=0;j<H;++j) merged[j] += tanhf_fast(h_slice[j]);
    }

    // -------- slice 2 --------
    {
        float X_slice[T2*E];
        // #pragma HLS ARRAY_PARTITION variable=Emb1 block factor=128
        embedding_forward_dyn(tokens2, T2, Emb2, X_slice);
        conv_bn_act_pool(X_slice, T2, ConvW2, K, ConvB2, Y, T12,
        BN1_gamma2, BN1_beta2, BN1_mean2, BN1_var2, BN_eps,
        ACT_RELU, P2, U_slice);
        lstm_forward_unidir(U_slice, T22, F, LSTM_W_ifog2, LSTM_R_ifog2, LSTM_b_ifog2, h_slice, c_slice);
        bn_vector(h_slice, H, BN2_gamma2, BN2_beta2, BN2_mean2, BN2_var2, BN_eps);
        for (j=0;j<H;++j) merged[j] += tanhf_fast(h_slice[j]);
    }

    // -------- slice 3 --------
    {
        float X_slice[T3*E];
        // #pragma HLS ARRAY_PARTITION variable=Emb1 block factor=128
        embedding_forward_dyn(tokens3, T3, Emb3, X_slice);
        conv_bn_act_pool(X_slice, T3, ConvW3, K, ConvB3, Y, T13,
        BN1_gamma3, BN1_beta3, BN1_mean3, BN1_var3, BN_eps,
        ACT_RELU, P3, U_slice);
        lstm_forward_unidir(U_slice, T23, F, LSTM_W_ifog3, LSTM_R_ifog3, LSTM_b_ifog3, h_slice, c_slice);
        bn_vector(h_slice, H, BN2_gamma3, BN2_beta3, BN2_mean3, BN2_var3, BN_eps);
        for (j=0;j<H;++j) merged[j] += tanhf_fast(h_slice[j]);
    }

    // -------- slice 4 --------
    {
        float X_slice[T4*E];
        // #pragma HLS ARRAY_PARTITION variable=Emb1 block factor=128
        embedding_forward_dyn(tokens4, T4, Emb4, X_slice);
        conv_bn_act_pool(X_slice, T4, ConvW4, K, ConvB4, Y, T14,
        BN1_gamma4, BN1_beta4, BN1_mean4, BN1_var4, BN_eps,
        ACT_RELU, P4, U_slice);
        lstm_forward_unidir(U_slice, T24, F, LSTM_W_ifog4, LSTM_R_ifog4, LSTM_b_ifog4, h_slice, c_slice);
        bn_vector(h_slice, H, BN2_gamma4, BN2_beta4, BN2_mean4, BN2_var4, BN_eps);
        for (j=0;j<H;++j) merged[j] += tanhf_fast(h_slice[j]);
    }
}

// ======================= main =======================
int main(void) {
    #pragma HLS TOP
    float merged[H];
    int j;
    run_all_slices_unrolled(merged);

    // ---- head forward (uses arrays from head_weights.h) ----
    const float BN_eps = 1e-3f;

    float z0[M0];
    dense_forward(merged, H, fc_0_W, fc_0_b, M0, z0);
    bn_vector(z0, M0, fc_0_bn_gamma, fc_0_bn_beta, fc_0_bn_mean, fc_0_bn_var, BN_eps);
    for (j=0;j<M0;++j) z0[j] = reluf(z0[j]);

    float z1[M1];
    dense_forward(z0, M0, fc_1_W, fc_1_b, M1, z1);
    bn_vector(z1, M1, fc_1_bn_gamma, fc_1_bn_beta, fc_1_bn_mean, fc_1_bn_var, BN_eps);
    for (j=0;j<M1;++j) z1[j] = reluf(z1[j]);

    float y_lin[1];
    dense_forward(z1, M1, output_W, output_b, 1, y_lin);
    float y_hat = sigmoidf(y_lin[0]);

    printf("y_hat = %.7e\n", y_hat);
    return 0;
}
