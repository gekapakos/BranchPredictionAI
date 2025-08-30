// main_multi.c  (bin-free, header-backed)

// ===== includes =====
#include <stdio.h>
#include <stdint.h>
#include <math.h>

// HLS 
#include <ap_fixed.h>
#include "hls_math.h"

typedef ap_fixed<16,6, AP_RND, AP_SAT> fx16;
typedef ap_fixed<32,12, AP_RND, AP_SAT> acc32;

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

static inline fx16 tanh_fx(fx16 x) { return fx16(hls::tanh((float)x)); }
static inline fx16 exp_fx (fx16 x) { return fx16(hls::exp ((float)x)); }
static inline fx16 sqrt_fx(fx16 x) { return fx16(hls::sqrt((float)x)); }

// ===== small utils =====
static inline fx16 sigmoidf(fx16 x){
    const fx16 one = fx16(1);
    return one / (one + exp_fx(-x));
}
static inline fx16 tanhf_fast(fx16 x){ return tanh_fx(x); }
static inline fx16 reluf(fx16 x){ return x > fx16(0) ? x : fx16(0); }

// ===== layers (dimensioned by args, not globals) =====
// ---- embedding ----
static inline void embedding_forward_dyn(const int32_t *tokens, int Tlen,
                                         const fx16 *Emb, fx16 *X) // X[Tlen,E]
{
    for(int t=0; t<Tlen; ++t){
        int idx = (int)tokens[t];
        if(idx < 0 || idx >= VOCAB_SIZE) idx = 0;
        const fx16 *row = Emb + idx*E;
        fx16 *dst = X + t*E;                 // <-- add this line
        for(int e=0; e<E; ++e) dst[e] = row[e];
    }
}


static inline void conv1d_valid_dyn(const fx16 *X,int Tlen,
                                    const fx16 *W,int Ksz,    // W[Ksz,E,F]
                                    const fx16 *bias, fx16 *Y)// Y[(Tlen-Ksz+1),F]
{
    const int T1 = Tlen - Ksz + 1;
    for(int t=0; t<T1; ++t){
        for(int f=0; f<F; ++f){
            fx16 acc = bias ? bias[f] : fx16(0);  // <-- was B ? B[f]
            for(int k=0; k<Ksz; ++k){
                const fx16 *xrow = X + (t+k)*E;
                const fx16 *wf   = W + (k*E)*F + f; // stride by Cout on e
                for(int e=0; e<E; ++e) acc += xrow[e] * wf[e*F];
            }
            Y[t*F + f] = acc;
        }
    }
}


static inline void bn_time_channel(fx16 *Y,int Tlen,int C,
    const fx16 *gamma,const fx16 *beta,const fx16 *mean,const fx16 *var,fx16 eps)
{
    for(int t=0;t<Tlen;++t) {
        fx16 *yt = Y + t*C;
        for(int c=0;c<C;++c) {
            fx16 nrm = (yt[c]-mean[c]) / sqrt_fx(var[c] + eps);
            yt[c] = gamma[c]*nrm + beta[c];
        }
    }
}

typedef enum { ACT_RELU, ACT_TANH, ACT_SIGMOID } act_t;
static inline void apply_activation_time(fx16 *Y,int Tlen,int C,act_t a){
    for(int t=0;t<Tlen;++t){
        fx16 *yt = Y + t*C;
        for(int c=0;c<C;++c){
            fx16 x=yt[c];
            yt[c] = (a==ACT_RELU)? reluf(x) : (a==ACT_TANH? tanhf_fast(x) : sigmoidf(x));
        }
    }
}

static inline void avgpool1d_P(const fx16 *Z,int Tlen,int C,int Psz,fx16 *U){
    const int Tout = Tlen / Psz;
    for(int u=0;u<Tout;++u){
        // for(int c=0;c<C;++c){
        for(int c=0;c<C;++c) { 
            fx16 acc = fx16(0);
            for(int p=0;p<Psz;++p) acc += Z[(u*Psz + p)*C + c];
            U[u*C + c] = acc / (fx16)Psz;
        }
    }
}

// LSTM (expects gate order [i,f,o,g] in W,R,b)
static inline void lstm_forward_unidir(const fx16 *x,int Tlen,int D,
    const fx16 *W_ifog,const fx16 *R_ifog,const fx16 *b_ifog,
    fx16 *h_last,fx16 *c_last)
{
    for(int j=0;j<H;++j){ h_last[j]=fx16(0); c_last[j]=fx16(0); }
    fx16 z[4*H];

    for(int t=0;t<Tlen;++t){
        for(int g=0; g<4*H; ++g) z[g] = b_ifog[g];

        const fx16 *xt = x + t*D;
        for(int d=0; d<D; ++d) {
            fx16 xv = xt[d];
            const fx16 *Wd = W_ifog + d*(4*H);
            for(int g=0; g<4*H; ++g) z[g] += xv * Wd[g];
        }
        for(int hp=0; hp<H; ++hp) {
            fx16 hv = h_last[hp];
            const fx16 *Rh = R_ifog + hp*(4*H);
            for(int g=0; g<4*H; ++g) z[g] += hv * Rh[g];
        }
        for(int j=0;j<H;++j) {
            fx16 i = sigmoidf(z[0*H + j]);
            fx16 f = sigmoidf(z[1*H + j]);
            fx16 o = sigmoidf(z[2*H + j]);
            fx16 g = tanh_fx (z[3*H + j]);
            fx16 c = f * c_last[j] + i * g;
            fx16 h = o * tanh_fx(c);
            c_last[j]=c; h_last[j]=h;
        }
    }
}


static inline void bn_vector(fx16 *v,int C,
    const fx16 *gamma,const fx16 *beta,const fx16 *mean,const fx16 *var,fx16 eps)
{
    for(int c=0;c<C;++c){
        fx16 n = (v[c]-mean[c]) / sqrt_fx(var[c] + eps);
        v[c] = gamma[c]*n + beta[c];
    }
}

static inline void dense_forward(const fx16 *x,int In,
    const fx16 *W,const fx16 *b,int Out,fx16 *y)
{
    for(int j=0;j<Out;++j) { 
        fx16 acc = b ? b[j] : fx16(0);
        for(int i=0;i<In;++i) acc += x[i]*W[i*Out + j];
        y[j]=acc;
    }
}

// ======================= run all 5 slices (using header arrays) =======================
static void run_all_slices_unrolled(fx16 merged[H]) {
    const fx16 BN_eps = 1e-3f;
    int j;
    for (j=0;j<H;++j) merged[j] = fx16(0);

    // -------- slice 0 --------
    {
        fx16 X0[T0*E], Y[T10*F], U[T20*F], h[H], c[H];
        embedding_forward_dyn(tokens0, T0, Emb0, X0);
        conv1d_valid_dyn(X0, T0, ConvW0, K, ConvB0, Y);
        bn_time_channel(Y, T10, F, BN1_gamma0, BN1_beta0, BN1_mean0, BN1_var0, BN_eps);
        apply_activation_time(Y, T10, F, ACT_RELU);
        avgpool1d_P(Y, T10, F, P0, U);
        lstm_forward_unidir(U, T20, F, LSTM_W_ifog0, LSTM_R_ifog0, LSTM_b_ifog0, h, c);
        bn_vector(h, H, BN2_gamma0, BN2_beta0, BN2_mean0, BN2_var0, BN_eps);
        for (j=0;j<H;++j) merged[j] += tanhf_fast(h[j]);
    }

    // -------- slice 1 --------
    {
        fx16 X0[T1*E], Y[T11*F], U[T21*F], h[H], c[H];
        embedding_forward_dyn(tokens1, T1, Emb1, X0);
        conv1d_valid_dyn(X0, T1, ConvW1, K, ConvB1, Y);
        bn_time_channel(Y, T11, F, BN1_gamma1, BN1_beta1, BN1_mean1, BN1_var1, BN_eps);
        apply_activation_time(Y, T11, F, ACT_RELU);
        avgpool1d_P(Y, T11, F, P1, U);
        lstm_forward_unidir(U, T21, F, LSTM_W_ifog1, LSTM_R_ifog1, LSTM_b_ifog1, h, c);
        bn_vector(h, H, BN2_gamma1, BN2_beta1, BN2_mean1, BN2_var1, BN_eps);
        for (j=0;j<H;++j) merged[j] += tanhf_fast(h[j]);
    }

    // -------- slice 2 --------
    {
        fx16 X0[T2*E], Y[T12*F], U[T22*F], h[H], c[H];
        embedding_forward_dyn(tokens2, T2, Emb2, X0);
        conv1d_valid_dyn(X0, T2, ConvW2, K, ConvB2, Y);
        bn_time_channel(Y, T12, F, BN1_gamma2, BN1_beta2, BN1_mean2, BN1_var2, BN_eps);
        apply_activation_time(Y, T12, F, ACT_RELU);
        avgpool1d_P(Y, T12, F, P2, U);
        lstm_forward_unidir(U, T22, F, LSTM_W_ifog2, LSTM_R_ifog2, LSTM_b_ifog2, h, c);
        bn_vector(h, H, BN2_gamma2, BN2_beta2, BN2_mean2, BN2_var2, BN_eps);
        for (j=0;j<H;++j) merged[j] += tanhf_fast(h[j]);
    }

    // -------- slice 3 --------
    {
        fx16 X0[T3*E], Y[T13*F], U[T23*F], h[H], c[H];
        embedding_forward_dyn(tokens3, T3, Emb3, X0);
        conv1d_valid_dyn(X0, T3, ConvW3, K, ConvB3, Y);
        bn_time_channel(Y, T13, F, BN1_gamma3, BN1_beta3, BN1_mean3, BN1_var3, BN_eps);
        apply_activation_time(Y, T13, F, ACT_RELU);
        avgpool1d_P(Y, T13, F, P3, U);
        lstm_forward_unidir(U, T23, F, LSTM_W_ifog3, LSTM_R_ifog3, LSTM_b_ifog3, h, c);
        bn_vector(h, H, BN2_gamma3, BN2_beta3, BN2_mean3, BN2_var3, BN_eps);
        for (j=0;j<H;++j) merged[j] += tanhf_fast(h[j]);
    }

    // -------- slice 4 --------
    {
        fx16 X0[T4*E], Y[T14*F], U[T24*F], h[H], c[H];
        embedding_forward_dyn(tokens4, T4, Emb4, X0);
        conv1d_valid_dyn(X0, T4, ConvW4, K, ConvB4, Y);
        bn_time_channel(Y, T14, F, BN1_gamma4, BN1_beta4, BN1_mean4, BN1_var4, BN_eps);
        apply_activation_time(Y, T14, F, ACT_RELU);
        avgpool1d_P(Y, T14, F, P4, U);
        lstm_forward_unidir(U, T24, F, LSTM_W_ifog4, LSTM_R_ifog4, LSTM_b_ifog4, h, c);
        bn_vector(h, H, BN2_gamma4, BN2_beta4, BN2_mean4, BN2_var4, BN_eps);
        for (j=0;j<H;++j) merged[j] += tanhf_fast(h[j]);
    }
}

// ======================= main =======================
int main(void) {
    fx16 merged[H];
    int j;
    run_all_slices_unrolled(merged);

    // ---- head forward (uses arrays from head_weights.h) ----
    const fx16 BN_eps = 1e-3f;

    fx16 z0[M0];
    dense_forward(merged, H, fc_0_W, fc_0_b, M0, z0);
    bn_vector(z0, M0, fc_0_bn_gamma, fc_0_bn_beta, fc_0_bn_mean, fc_0_bn_var, BN_eps);
    for (j=0;j<M0;++j) z0[j] = reluf(z0[j]);

    fx16 z1[M1];
    dense_forward(z0, M0, fc_1_W, fc_1_b, M1, z1);
    bn_vector(z1, M1, fc_1_bn_gamma, fc_1_bn_beta, fc_1_bn_mean, fc_1_bn_var, BN_eps);
    for (j=0;j<M1;++j) z1[j] = reluf(z1[j]);

    fx16 y_lin[1];
    dense_forward(z1, M1, output_W, output_b, 1, y_lin);
    fx16 y_hat = sigmoidf(y_lin[0]);

    // printf("y_hat = %.7e\n", y_hat);
    printf("y_hat = %.7e\n", (double)y_hat);
    return 0;
}