// slice_test.c
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdint.h>
#include <string.h>
#include "weights_io.h"
#include <time.h>

enum { VOCAB_SIZE=128, E=32, T=44, K=3, F=2, P=7, H=16, M=8 };
static inline float sigmoidf(float x){ return 1.0f/(1.0f+expf(-x)); }
static inline float tanhf_fast(float x){ return tanhf(x); }
static inline float reluf(float x){ return x > 0.0f ? x : 0.0f; }

// Dense y = xW + b, W row-major [In][Out]
static inline void dense_forward(const float *x, int In, const float *W, const float *b, int Out, float *y){
    for (int j=0; j<Out; ++j){
        float acc = b ? b[j] : 0.0f;
        for (int i=0;i<In;++i) acc += x[i]*W[i*Out + j];
        y[j]=acc;
    }
}

// BN (time,channel)
static inline void bn_time_channel(float *Y, int Tlen, int C, const float *gamma,const float *beta,const float *mean,const float *var,float eps){
    for (int t=0;t<Tlen;++t){
        float *yt = Y + t*C;
        for (int c=0;c<C;++c){
            float norm = (yt[c]-mean[c]) / sqrtf(var[c]+eps);
            yt[c] = gamma[c]*norm + beta[c];
        }
    }
}
static inline void bn_vector(float *v,int C,const float *gamma,const float *beta,const float *mean,const float *var,float eps){
    for (int c=0;c<C;++c){
        float norm = (v[c]-mean[c]) / sqrtf(var[c]+eps);
        v[c] = gamma[c]*norm + beta[c];
    }
}
typedef enum { ACT_RELU, ACT_TANH, ACT_SIGMOID } act_t;
static inline void apply_activation_time(float *Y,int Tlen,int C,act_t a){
    for(int t=0;t<Tlen;++t){
        float *yt = Y + t*C;
        for(int c=0;c<C;++c){
            float x = yt[c];
            yt[c] = (a==ACT_RELU)? reluf(x) : (a==ACT_TANH? tanhf_fast(x) : sigmoidf(x));
        }
    }
}
static inline void avgpool1d_P(const float *Z,int Tlen,int C,int Psz,float *U){
    int Tout = Tlen / Psz;
    for(int u=0;u<Tout;++u){
        for(int c=0;c<C;++c){
            float acc=0.0f;
            for(int p=0;p<Psz;++p) acc += Z[(u*Psz + p)*C + c];
            U[u*C + c] = acc / (float)Psz;
        }
    }
}

// LSTM forward (gate order [i,f,o,g] expected in W,R,b)
static inline void lstm_forward_unidir(const float *x,int Tlen,int D,
                                       const float *W_ifog,const float *R_ifog,const float *b_ifog,
                                       int Hsz,float *h_last,float *c_last)
{
    for(int j=0;j<Hsz;++j){ h_last[j]=0.0f; c_last[j]=0.0f; }

    for(int t=0;t<Tlen;++t){
        float z[4*Hsz];                                      // FIX: use Hsz, no VLA initializer
        for(int gk=0; gk<4*Hsz; ++gk) z[gk] = b_ifog[gk];    // start from bias

        const float *xt = x + t*D;
        for(int d=0; d<D; ++d){
            float xv = xt[d];
            const float *Wd = W_ifog + d*(4*Hsz);
            for(int gk=0; gk<4*Hsz; ++gk) z[gk] += xv * Wd[gk];
        }
        for(int hp=0; hp<Hsz; ++hp){
            float hv = h_last[hp];
            const float *Rh = R_ifog + hp*(4*Hsz);
            for(int gk=0; gk<4*Hsz; ++gk) z[gk] += hv * Rh[gk];
        }
        for(int j=0;j<Hsz;++j){
            float i = sigmoidf(z[0*Hsz + j]);
            float f = sigmoidf(z[1*Hsz + j]);
            float o = sigmoidf(z[2*Hsz + j]);
            float g = tanhf_fast(z[3*Hsz + j]);              // g = candidate
            float c = f * c_last[j] + i * g;
            float h = o * tanhf_fast(c);
            c_last[j]=c; h_last[j]=h;
        }
    }
}

static inline void conv1d_keras_valid(const float *X,int Tlen,int Cin,const float *W,int Ksz,int Cout,const float *B,float *Y){
    const int T1 = Tlen - Ksz + 1;
    for(int t=0;t<T1;++t){
        for(int f=0; f<Cout; ++f){
            float acc = B ? B[f] : 0.0f;
            for(int k=0;k<Ksz;++k){
                const float *xrow = X + (t+k)*Cin;
                const float *wf   = W + (k*Cin)*Cout + f; // W[k,e,f] contiguous in f
                for(int e=0;e<Cin;++e) acc += xrow[e] * wf[e*Cout];
            }
            Y[t*Cout + f] = acc;
        }
    }
}
static inline void embedding_forward(const int *tokens,float *X,const float *Emb){
    for(int t=0;t<T;++t){
        int idx = tokens[t];
        if(idx<0 || idx>=VOCAB_SIZE) idx=0;
        const float *row = Emb + idx*E;
        float *dst = X + t*E;
        for(int e=0;e<E;++e) dst[e]=row[e];
    }
}

static void print_mat_sci(const char *name,const float *X,int Tm,int Cm){
    printf("%s:\n", name);
    for(int t=0;t<Tm;++t){
        printf(" [");
        for(int c=0;c<Cm;++c) printf("%s%.7e", c?" ":"", X[t*Cm + c]);
        printf("]\n");
    }
}

static void slice_forward(const int *tokens,
                          const float *Emb,
                          const float *ConvW,const float *ConvB,
                          const float *BN1_gamma,const float *BN1_beta,const float *BN1_mean,const float *BN1_var,float BN1_eps,
                          const float *LSTM_W_ifog,const float *LSTM_R_ifog,const float *LSTM_b_ifog,
                          const float *BN2_gamma,const float *BN2_beta,const float *BN2_mean,const float *BN2_var,float BN2_eps,
                          float *out_tanh)
{
    float X0[T*E];
    embedding_forward(tokens, X0, Emb);

    const int T1 = T - K + 1;
    float Y[T1*F];
    conv1d_keras_valid(X0, T, E, ConvW, K, F, ConvB, Y);

    bn_time_channel(Y, T1, F, BN1_gamma, BN1_beta, BN1_mean, BN1_var, BN1_eps);
    apply_activation_time(Y, T1, F, ACT_RELU);

    const int T2 = T1 / P;
    float U[T2*F];
    avgpool1d_P(Y, T1, F, P, U);

    float h_last[H], c_last[H];
    lstm_forward_unidir(U, T2, F, LSTM_W_ifog, LSTM_R_ifog, LSTM_b_ifog, H, h_last, c_last);

    bn_vector(h_last, H, BN2_gamma, BN2_beta, BN2_mean, BN2_var, BN2_eps);
    for(int j=0;j<H;++j) out_tanh[j] = tanhf_fast(h_last[j]);
}

static void path_join(char *dst,size_t cap,const char *dir,const char *file){
    size_t n = snprintf(dst, cap, "%s/%s", dir, file);
    if(n >= cap){ fprintf(stderr,"path too long\n"); exit(1); }
}

int main(int argc,char **argv){
    if(argc<2){ fprintf(stderr,"usage: %s <weights_dir>\n", argv[0]); return 1; }
    const char *WDIR = argv[1];

    // buffers
    static float Emb[VOCAB_SIZE*E];
    static float ConvW[K*E*F], ConvB[F];
    static float BN1_gamma[F], BN1_beta[F], BN1_mean[F], BN1_var[F];
    static float LSTM_W_ifog[F*4*H], LSTM_R_ifog[H*4*H], LSTM_b_ifog[4*H]; 
    static float BN2_gamma[H], BN2_beta[H], BN2_mean[H], BN2_var[H];

    const float BN1_eps = 1e-3f, BN2_eps = 1e-3f;

    char path[512];
    #define LOAD(name,buf,count) do{ path_join(path,sizeof(path),WDIR,name); load_bin_f32(path,buf,count); }while(0)

    LOAD("Emb.bin", Emb, VOCAB_SIZE*E);
    LOAD("ConvW.bin", ConvW, K*E*F);
    LOAD("ConvB.bin", ConvB, F);
    LOAD("BN1_gamma.bin", BN1_gamma, F);
    LOAD("BN1_beta.bin",  BN1_beta,  F);
    LOAD("BN1_mean.bin",  BN1_mean,  F);
    LOAD("BN1_var.bin",   BN1_var,   F);

    LOAD("LSTM_W_ifog.bin", LSTM_W_ifog, F*4*H);
    LOAD("LSTM_R_ifog.bin", LSTM_R_ifog, H*4*H);
    LOAD("LSTM_b_ifog.bin", LSTM_b_ifog, 4*H);

    LOAD("BN2_gamma.bin", BN2_gamma, H);
    LOAD("BN2_beta.bin",  BN2_beta,  H);
    LOAD("BN2_mean.bin",  BN2_mean,  H);
    LOAD("BN2_var.bin",   BN2_var,   H);

    // tokens
    static int tokens[T];
    path_join(path, sizeof(path), WDIR, "tokens.bin");
    FILE *ft = fopen(path, "rb");
    if(ft){
        size_t got = fread(tokens, sizeof(int), T, ft); fclose(ft);
        if(got != (size_t)T){ fprintf(stderr,"tokens.bin length != T\n"); return 1; }
    }else{
        for(int t=0;t<T;++t) tokens[t]=t % VOCAB_SIZE;
    }

    float slice_vec[H];

    clock_t start, end;
    double cpu_time_used;

    start = clock();

    slice_forward(tokens, Emb,
                  ConvW, ConvB,
                  BN1_gamma, BN1_beta, BN1_mean, BN1_var, BN1_eps,
                  LSTM_W_ifog, LSTM_R_ifog, LSTM_b_ifog,
                  BN2_gamma, BN2_beta, BN2_mean, BN2_var, BN2_eps,
                  slice_vec);

    // print_mat_sci("slice_out", slice_vec, 1, H);

    // --- head weights: Dense(16->8) -> BN(8) -> ReLU -> Dense(8->1) ---
    static float FC0_W[H * M];    // [16,8] row-major [In][Out]
    static float FC0_b[M];        // [8]
    static float FC0_BN_gamma[M]; // [8]
    static float FC0_BN_beta[M];  // [8]
    static float FC0_BN_mean[M];  // [8]
    static float FC0_BN_var[M];   // [8]
    const float FC0_BN_eps = 1e-3f;

    static float OUT_W[M * 1];    // [8,1]
    static float OUT_b[1];        // [1]

    LOAD("fc_0_W.bin",        FC0_W,       H*M);
    LOAD("fc_0_b.bin",        FC0_b,       M);
    LOAD("fc_0_bn_gamma.bin", FC0_BN_gamma, M);
    LOAD("fc_0_bn_beta.bin",  FC0_BN_beta,  M);
    LOAD("fc_0_bn_mean.bin",  FC0_BN_mean,  M);
    LOAD("fc_0_bn_var.bin",   FC0_BN_var,   M);

    LOAD("output_W.bin",      OUT_W,       M*1);
    LOAD("output_b.bin",      OUT_b,       1);

    float merged[H];
    for (int j=0;j<H;++j) merged[j] = slice_vec[j];

    float z1[M];
    dense_forward(merged, H, FC0_W, FC0_b, M, z1);                    // fc_0
    bn_vector(z1, M, FC0_BN_gamma, FC0_BN_beta,                       // fc_0_bn
            FC0_BN_mean, FC0_BN_var, FC0_BN_eps);
    for (int j=0;j<M;++j) z1[j] = reluf(z1[j]);                       // fc_0_act (relu)

    float y_lin[1];
    dense_forward(z1, M, OUT_W, OUT_b, 1, y_lin);                     // output (pre-sigmoid)
    float y_hat = sigmoidf(y_lin[0]);                                  // final prob

    // print_mat_sci("slice_out", merged, 1, H);
    // printf("y_hat = %.7e\n", y_hat);

    end = clock();

    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("Elapsed time: %.6f seconds\n", cpu_time_used);

    return 0;
}
