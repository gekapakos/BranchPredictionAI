// conv1d_valid_test.c
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdbool.h>

// --- Under test: Conv1D (Keras 'valid'), X[T,Cin], W[K,Cin,Cout], B[Cout] -> Y[T-K+1,Cout]
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
                const float *wf   = W + (k * Cin) * Cout + f; // W[k,e,f], Cout is fastest
                for (int e = 0; e < Cin; ++e) acc += xrow[e] * wf[e * Cout];
            }
            Y[t * Cout + f] = acc;
        }
    }
}

// --- helpers ---
static bool allclose(const float *a, const float *b, int n, float atol, float rtol){
    for (int i=0;i<n;++i){
        float diff=fabsf(a[i]-b[i]);
        float tol = atol + rtol * fmaxf(fabsf(a[i]), fabsf(b[i]));
        if (diff > tol) return false;
    }
    return true;
}
static void print_mat_TxC(const char *name, const float *Y, int T, int C){
    printf("%s:\n", name);
    for (int t=0;t<T;++t){
        printf("  t=%d: [", t);
        for (int c=0;c<C;++c) printf("%s%.3f", c?", ":"", Y[t*C+c]);
        printf("]\n");
    }
}

int main(void){
    // =======================
    // Test 1: Cin=1, Cout=1, K=3 (checks cross-correlation, no flip)
    // x[t] = t, w = [1, 0, -1]  => y[t] = x[t]*1 + x[t+1]*0 + x[t+2]*(-1) = -2 (constant)
    // Then add bias 0.5 to confirm bias path -> -1.5
    // =======================
    enum { T1=6, Cin1=1, K1=3, F1=1, Tout1=T1-K1+1 };

    float X1[T1*Cin1] = { 0,1,2,3,4,5 };
    float W1[K1*Cin1*F1] = {
        // W[k,e,f] with Cout fastest -> for Cin=1,F=1 this is just [w0, w1, w2]
        1.0f, 0.0f, -1.0f
    };

    float Y1[Tout1*F1], Y1b[Tout1*F1], Y1_exp[Tout1*F1], Y1b_exp[Tout1*F1];
    // expected without bias: all -2; with bias 0.5: all -1.5
    for (int t=0;t<Tout1;++t){ Y1_exp[t]= -2.0f; Y1b_exp[t]= -1.5f; }

    // run without bias
    conv1d_keras_valid(X1, T1, Cin1, W1, K1, F1, NULL, Y1);
    // run with bias
    float B1[F1] = { 0.5f };
    conv1d_keras_valid(X1, T1, Cin1, W1, K1, F1, B1, Y1b);

    print_mat_TxC("Test1 - Y (no bias)", Y1, Tout1, F1);
    print_mat_TxC("Test1 - Y (with bias=0.5)", Y1b, Tout1, F1);
    printf("Test1 no-bias: %s\n", allclose(Y1,  Y1_exp,  Tout1*F1, 1e-6f, 0.0f) ? "PASS":"FAIL");
    printf("Test1 +bias:   %s\n\n", allclose(Y1b, Y1b_exp, Tout1*F1, 1e-6f, 0.0f) ? "PASS":"FAIL");

    // =======================
    // Test 2: Cin=2, Cout=2, K=3, T=6  (checks multi-channel & multi-filter indexing)
    // X[t,e] = 10*t + e
    // Filter 0: all ones -> Y0[t] = sum_{k=0..2} sum_{e=0..1} (10*(t+k)+e)
    //   = 2*10*(3t+3) + 3*(sum_e e=1) = 60t + 63
    // Filter 1: only pick (k=1,e=1) with weight 2 -> Y1[t] = 2*(10*(t+1)+1) = 20t + 22
    // Bias B = [0, -5] to test bias per-filter
    // =======================
    enum { T2=6, Cin2=2, K2=3, F2=2, Tout2=T2-K2+1 };

    float X2[T2*Cin2];
    for (int t=0;t<T2;++t) for (int e=0;e<Cin2;++e) X2[t*Cin2 + e] = 10.0f*t + (float)e;

    float W2[K2*Cin2*F2];
    memset(W2, 0, sizeof(W2));
    // Filter 0 (f=0): all ones
    for (int k=0;k<K2;++k){
        for (int e=0;e<Cin2;++e){
            // index W[k,e,0]
            W2[(k*Cin2)*F2 + e*F2 + 0] = 1.0f;
        }
    }
    // Filter 1 (f=1): only weight at (k=1,e=1) = 2
    W2[(1*Cin2)*F2 + 1*F2 + 1] = 2.0f;

    float B2[F2] = { 0.0f, -5.0f };
    float Y2[Tout2*F2], Y2_exp[Tout2*F2];

    // compute expected from the closed forms above
    for (int t=0;t<Tout2;++t){
        float y0 = 60.0f*t + 63.0f;
        float y1 = 20.0f*t + 22.0f - 5.0f;
        Y2_exp[t*F2 + 0] = y0;
        Y2_exp[t*F2 + 1] = y1;
    }

    conv1d_keras_valid(X2, T2, Cin2, W2, K2, F2, B2, Y2);

    print_mat_TxC("Test2 - Y", Y2, Tout2, F2);
    print_mat_TxC("Test2 - Y_expected", Y2_exp, Tout2, F2);
    printf("Test2: %s\n", allclose(Y2, Y2_exp, Tout2*F2, 1e-6f, 0.0f) ? "PASS":"FAIL");

    return 0;
}
