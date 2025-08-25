// avgpool1d_test.c
#include <stdio.h>
#include <math.h>
#include <stdbool.h>

// ======= Under test =======
// AveragePooling1D(pool=P, stride=P, valid), layout [Tlen, C] (time-major, channels-last)
static inline void avgpool1d_P(const float *Z, int Tlen, int C, int Psz, float *U) {
    int Tout = Tlen / Psz; // stride=P, valid (tail is dropped)
    for (int u=0; u<Tout; ++u) {
        for (int c=0; c<C; ++c) {
            float acc = 0.0f;
            for (int p=0; p<Psz; ++p) acc += Z[(u*Psz + p)*C + c];
            U[u*C + c] = acc / (float)Psz;
        }
    }
}

// ======= Test helpers =======
static bool allclose(const float *a, const float *b, int n, float atol, float rtol){
    for (int i=0;i<n;++i){
        float diff=fabsf(a[i]-b[i]);
        float tol = atol + rtol * fmaxf(fabsf(a[i]), fabsf(b[i]));
        if (diff > tol) return false;
    }
    return true;
}
static void print_mat_TxC(const char *name, const float *X, int T, int C){
    printf("%s:\n", name);
    for (int t=0;t<T;++t){
        printf("  t=%d: [", t);
        for (int c=0;c<C;++c) printf("%s%.3f", c?", ":"", X[t*C+c]);
        printf("]\n");
    }
}

int main(void){
    // ---------- Test 1: simple divisible case ----------
    // Tlen=8, C=2, Psz=2  -> Tout=4
    // Z[t,c] laid out as [Tlen, C], time-major (channels-last).
    const int T1=8, C1=2, P1=2, Tout1=T1/P1;
    float Z1[] = {
        1,  3,
        3,  5,
        5,  7,
        7,  9,
        9, 11,
        11, 13,
        13, 15,
        15, 17
    };
    // Windows of size 2 with stride 2:
    // [ (1,3),(3,5) ] -> avg = ( (1+3)/2, (3+5)/2 ) = (2,4)
    // [ (5,7),(7,9) ] -> (6,8)
    // [ (9,11),(11,13) ] -> (10,12)
    // [ (13,15),(15,17) ] -> (14,16)
    float U1_exp[] = {
        2,  4,
        6,  8,
       10, 12,
       14, 16
    };
    float U1[Tout1*C1];
    avgpool1d_P(Z1, T1, C1, P1, U1);

    print_mat_TxC("Test1 - input Z1", Z1, T1, C1);
    print_mat_TxC("Test1 - output U1", U1, Tout1, C1);
    printf("Test1: %s\n\n", allclose(U1, U1_exp, Tout1*C1, 1e-6f, 0.0f) ? "PASS" : "FAIL");

    // ---------- Test 2: non-divisible length (tail dropped) ----------
    // Tlen=7, C=3, Psz=3  -> Tout=floor(7/3)=2, last row (t=6) ignored.
    const int T2=7, C2=3, P2=3, Tout2=T2/P2;
    float Z2[T2*C2];
    // Fill Z2[t,c] = 10*t + c  (easy to average by hand)
    for (int t=0;t<T2;++t) for (int c=0;c<C2;++c) Z2[t*C2 + c] = 10.0f*t + (float)c;

    // Window 0: t=0..2  -> channel-wise means: [10,11,12]
    // Window 1: t=3..5  ->                      [40,41,42]
    float U2_exp[] = {
        10, 11, 12,
        40, 41, 42
    };
    float U2[Tout2*C2];
    avgpool1d_P(Z2, T2, C2, P2, U2);

    print_mat_TxC("Test2 - input Z2", Z2, T2, C2);
    print_mat_TxC("Test2 - output U2", U2, Tout2, C2);
    printf("Test2: %s\n", allclose(U2, U2_exp, Tout2*C2, 1e-6f, 0.0f) ? "PASS" : "FAIL");

    return 0;
}
