// main_multi.c
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <string.h>
#include "weights_io.h"
#include <time.h>


// ===== model constants (match Python config) =====
enum { SLICES=5, VOCAB_SIZE=4096, E=32, K=7, F=32, H=32 };
static const int Tlens[SLICES] = {42, 78, 150, 294, 582};
static const int Pools[SLICES] = { 3,  6,  12,  24,  48};

// head sizes: Dense(H->M0) -> BN -> ReLU -> Dense(M0->M1) -> BN -> ReLU -> Dense(M1->1)
enum { M0=128, M1=128 };

// ===== small utils =====
static inline float sigmoidf(float x){ return 1.0f/(1.0f+expf(-x)); }
static inline float tanhf_fast(float x){ return tanhf(x); }
static inline float reluf(float x){ return x>0.0f? x:0.0f; }

static void path_join3(char* dst, size_t cap, const char* a, const char* b, const char* c){
    int n = snprintf(dst, cap, "%s/%s/%s", a, b, c);
    if(n<0 || (size_t)n>=cap){ fprintf(stderr,"path too long\n"); exit(1); }
}
static void path_join2(char* dst, size_t cap, const char* a, const char* b){
    int n = snprintf(dst, cap, "%s/%s", a, b);
    if(n<0 || (size_t)n>=cap){ fprintf(stderr,"path too long\n"); exit(1); }
}

// ===== layers (dimensioned by args, not globals) =====
static inline void embedding_forward_dyn(const int *tokens, int Tlen,
                                         const float *Emb, float *X) // X[Tlen,E]
{
    for(int t=0;t<Tlen;++t){
        int idx = tokens[t];
        if(idx<0 || idx>=VOCAB_SIZE) idx=0;
        const float *row = Emb + idx*E;
        float *dst = X + t*E;
        for(int e=0;e<E;++e) dst[e]=row[e];
    }
}

static inline void conv1d_valid_dyn(const float *X,int Tlen,
                                    const float *W,int Ksz, // W[Ksz,E,F]
                                    const float *B, float *Y) // Y[(Tlen-Ksz+1),F]
{
    const int T1 = Tlen - Ksz + 1;
    for(int t=0;t<T1;++t){
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
}

static inline void bn_time_channel(float *Y,int Tlen,int C,
    const float *gamma,const float *beta,const float *mean,const float *var,float eps)
{
    for(int t=0;t<Tlen;++t){
        float *yt = Y + t*C;
        for(int c=0;c<C;++c){
            float nrm = (yt[c]-mean[c]) / sqrtf(var[c]+eps);
            yt[c] = gamma[c]*nrm + beta[c];
        }
    }
}

typedef enum { ACT_RELU, ACT_TANH, ACT_SIGMOID } act_t;
static inline void apply_activation_time(float *Y,int Tlen,int C,act_t a){
    for(int t=0;t<Tlen;++t){
        float *yt = Y + t*C;
        for(int c=0;c<C;++c){
            float x=yt[c];
            yt[c] = (a==ACT_RELU)? reluf(x) : (a==ACT_TANH? tanhf_fast(x) : sigmoidf(x));
        }
    }
}

static inline void avgpool1d_P(const float *Z,int Tlen,int C,int Psz,float *U){
    const int Tout = Tlen / Psz;
    for(int u=0;u<Tout;++u){
        for(int c=0;c<C;++c){
            float acc=0.0f;
            for(int p=0;p<Psz;++p) acc += Z[(u*Psz + p)*C + c];
            U[u*C + c] = acc / (float)Psz;
        }
    }
}

// LSTM (expects gate order [i,f,o,g] in W,R,b; H is global 32)
static inline void lstm_forward_unidir(const float *x,int Tlen,int D,
    const float *W_ifog,const float *R_ifog,const float *b_ifog,
    float *h_last,float *c_last)
{
    for(int j=0;j<H;++j){ h_last[j]=0.0f; c_last[j]=0.0f; }
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
        for(int j=0;j<H;++j){
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

static inline void bn_vector(float *v,int C,
    const float *gamma,const float *beta,const float *mean,const float *var,float eps)
{
    for(int c=0;c<C;++c){
        float n = (v[c]-mean[c]) / sqrtf(var[c]+eps);
        v[c] = gamma[c]*n + beta[c];
    }
}

static inline void dense_forward(const float *x,int In,
    const float *W,const float *b,int Out,float *y)
{
    for(int j=0;j<Out;++j){
        float acc = b? b[j] : 0.0f;
        for(int i=0;i<In;++i) acc += x[i]*W[i*Out + j];
        y[j]=acc;
    }
}

static void print_vec(const char* name,const float* v,int n){
    printf("%s = [", name);
    for(int i=0;i<n;++i) printf("%s%.7e", i?", ":"", v[i]);
    printf("]\n");
}

// forward one slice i; reads weights from base_dir/slice{i}/
static void run_slice_i(const char* base_dir, int i, float out_tanh[H]){
    char path[512], sdir[64];
    snprintf(sdir, sizeof sdir, "slice%d", i);

    // ---- allocate per-slice weights ----
    static float Emb[VOCAB_SIZE*E];
    static float ConvW[K*E*F], ConvB[F];
    static float BN1_gamma[F], BN1_beta[F], BN1_mean[F], BN1_var[F];
    static float LSTM_W_ifog[F*4*H], LSTM_R_ifog[H*4*H], LSTM_b_ifog[4*H];
    static float BN2_gamma[H], BN2_beta[H], BN2_mean[H], BN2_var[H];
    const float BN_eps = 1e-3f;

    // ---- load weights ----
    path_join3(path,sizeof path, base_dir,sdir,"Emb.bin");            load_bin_f32(path, Emb, VOCAB_SIZE*E);
    path_join3(path,sizeof path, base_dir,sdir,"ConvW.bin");          load_bin_f32(path, ConvW, K*E*F);
    path_join3(path,sizeof path, base_dir,sdir,"ConvB.bin");          load_bin_f32(path, ConvB, F);
    path_join3(path,sizeof path, base_dir,sdir,"BN1_gamma.bin");      load_bin_f32(path, BN1_gamma, F);
    path_join3(path,sizeof path, base_dir,sdir,"BN1_beta.bin");       load_bin_f32(path, BN1_beta,  F);
    path_join3(path,sizeof path, base_dir,sdir,"BN1_mean.bin");       load_bin_f32(path, BN1_mean,  F);
    path_join3(path,sizeof path, base_dir,sdir,"BN1_var.bin");        load_bin_f32(path, BN1_var,   F);
    path_join3(path,sizeof path, base_dir,sdir,"LSTM_W_ifog.bin");    load_bin_f32(path, LSTM_W_ifog, F*4*H);
    path_join3(path,sizeof path, base_dir,sdir,"LSTM_R_ifog.bin");    load_bin_f32(path, LSTM_R_ifog, H*4*H);
    path_join3(path,sizeof path, base_dir,sdir,"LSTM_b_ifog.bin");    load_bin_f32(path, LSTM_b_ifog, 4*H);
    path_join3(path,sizeof path, base_dir,sdir,"BN2_gamma.bin");      load_bin_f32(path, BN2_gamma, H);
    path_join3(path,sizeof path, base_dir,sdir,"BN2_beta.bin");       load_bin_f32(path, BN2_beta,  H);
    path_join3(path,sizeof path, base_dir,sdir,"BN2_mean.bin");       load_bin_f32(path, BN2_mean,  H);
    path_join3(path,sizeof path, base_dir,sdir,"BN2_var.bin");        load_bin_f32(path, BN2_var,   H);

    // ---- load tokens ----
    int T = Tlens[i], P = Pools[i];
    int *tokens = (int*)malloc(sizeof(int)*T);
    path_join3(path,sizeof path, base_dir,sdir,"tokens.bin");
    FILE *ft = fopen(path,"rb");
    if(ft){
        size_t got = fread(tokens,sizeof(int),T,ft); fclose(ft);
        if(got != (size_t)T){ fprintf(stderr,"tokens.bin len != %d for %s\n", T, sdir); exit(1); }
    }else{
        for(int t=0;t<T;++t) tokens[t] = t % VOCAB_SIZE;
    }

    // ---- forward ----
    float *X0 = (float*)malloc(sizeof(float)*T*E);
    embedding_forward_dyn(tokens, T, Emb, X0);

    const int T1 = T - K + 1;
    float *Y = (float*)malloc(sizeof(float)*T1*F);
    conv1d_valid_dyn(X0, T, ConvW, K, ConvB, Y);

    bn_time_channel(Y, T1, F, BN1_gamma, BN1_beta, BN1_mean, BN1_var, BN_eps);
    apply_activation_time(Y, T1, F, ACT_RELU);

    const int T2 = T1 / P;
    float *U = (float*)malloc(sizeof(float)*T2*F);
    avgpool1d_P(Y, T1, F, P, U);

    float h_last[H], c_last[H];
    lstm_forward_unidir(U, T2, F, LSTM_W_ifog, LSTM_R_ifog, LSTM_b_ifog, h_last, c_last);

    bn_vector(h_last, H, BN2_gamma, BN2_beta, BN2_mean, BN2_var, BN_eps);
    for(int j=0;j<H;++j) out_tanh[j] = tanhf_fast(h_last[j]);

    free(tokens); free(X0); free(Y); free(U);
}

int main(int argc,char** argv){
    if(argc<2){ fprintf(stderr,"usage: %s <weights_out/multi>\n", argv[0]); return 1; }
    const char* WROOT = argv[1];

    clock_t start, end;
    double cpu_time_used;

    start = clock();

    // ---- run all slices & sum (Add) ----
    float merged[H]; for(int j=0;j<H;++j) merged[j]=0.0f;
    for(int i=0;i<SLICES;++i){
        float vec[H];
        run_slice_i(WROOT, i, vec);
        for(int j=0;j<H;++j) merged[j] += vec[j];
    }

    // ---- load head (from root) ----
    char path[512];
    // fc_0
    static float FC0_W[H*M0], FC0_b[M0], FC0_g[M0], FC0_be[M0], FC0_mm[M0], FC0_mv[M0];
    path_join2(path,sizeof path,WROOT,"fc_0_W.bin");        load_bin_f32(path, FC0_W, H*M0);
    path_join2(path,sizeof path,WROOT,"fc_0_b.bin");        load_bin_f32(path, FC0_b, M0);
    path_join2(path,sizeof path,WROOT,"fc_0_bn_gamma.bin"); load_bin_f32(path, FC0_g, M0);
    path_join2(path,sizeof path,WROOT,"fc_0_bn_beta.bin");  load_bin_f32(path, FC0_be, M0);
    path_join2(path,sizeof path,WROOT,"fc_0_bn_mean.bin");  load_bin_f32(path, FC0_mm, M0);
    path_join2(path,sizeof path,WROOT,"fc_0_bn_var.bin");   load_bin_f32(path, FC0_mv, M0);

    // fc_1
    static float FC1_W[M0*M1], FC1_b[M1], FC1_g[M1], FC1_be[M1], FC1_mm[M1], FC1_mv[M1];
    path_join2(path,sizeof path,WROOT,"fc_1_W.bin");        load_bin_f32(path, FC1_W, M0*M1);
    path_join2(path,sizeof path,WROOT,"fc_1_b.bin");        load_bin_f32(path, FC1_b, M1);
    path_join2(path,sizeof path,WROOT,"fc_1_bn_gamma.bin"); load_bin_f32(path, FC1_g, M1);
    path_join2(path,sizeof path,WROOT,"fc_1_bn_beta.bin");  load_bin_f32(path, FC1_be, M1);
    path_join2(path,sizeof path,WROOT,"fc_1_bn_mean.bin");  load_bin_f32(path, FC1_mm, M1);
    path_join2(path,sizeof path,WROOT,"fc_1_bn_var.bin");   load_bin_f32(path, FC1_mv, M1);

    // output
    static float OUT_W[M1*1], OUT_b[1];
    path_join2(path,sizeof path,WROOT,"output_W.bin");      load_bin_f32(path, OUT_W, M1*1);
    path_join2(path,sizeof path,WROOT,"output_b.bin");      load_bin_f32(path, OUT_b, 1);

    const float BN_eps = 1e-3f;

    // ---- head forward ----
    float z0[M0];
    dense_forward(merged, H, FC0_W, FC0_b, M0, z0);
    bn_vector(z0, M0, FC0_g, FC0_be, FC0_mm, FC0_mv, BN_eps);
    for(int j=0;j<M0;++j) z0[j] = reluf(z0[j]);

    float z1[M1];
    dense_forward(z0, M0, FC1_W, FC1_b, M1, z1);
    bn_vector(z1, M1, FC1_g, FC1_be, FC1_mm, FC1_mv, BN_eps);
    for(int j=0;j<M1;++j) z1[j] = reluf(z1[j]);

    float y_lin[1];
    dense_forward(z1, M1, OUT_W, OUT_b, 1, y_lin);
    float y_hat = sigmoidf(y_lin[0]);

    // print_vec("merged", merged, H); // optional
    printf("y_hat = %.7e\n", y_hat);

    end = clock();

    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("Elapsed time: %.6f seconds\n", cpu_time_used);

    return 0;
}
