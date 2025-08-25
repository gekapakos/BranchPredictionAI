// lstm_test.c
#include <stdio.h>
#include <math.h>
#include <stdbool.h>
#include <string.h>

// ---------- helpers ----------
static inline float sigmoidf(float x){ return 1.0f / (1.0f + expf(-x)); }
static inline float tanhf_fast(float x){ return tanhf(x); }

static bool allclose(const float *a, const float *b, int n, float atol, float rtol){
    for (int i=0;i<n;++i){
        float diff=fabsf(a[i]-b[i]);
        float tol = atol + rtol*fmaxf(fabsf(a[i]), fabsf(b[i]));
        if (diff > tol) return false;
    }
    return true;
}

static void print_vec(const char *name, const float *v, int n){
    printf("%s = [", name);
    for (int i=0;i<n;++i) printf("%s%.6f", i?", ":"", v[i]);
    printf("]\n");
}

// ---------- LSTM under test (gate order [i,f,o,g]) ----------
static inline void lstm_forward_unidir(const float *x, int Tlen, int D,
                                       const float *W_ifog, // [D][4H]
                                       const float *R_ifog, // [H][4H]
                                       const float *b_ifog, // [4H]
                                       int Hsz,
                                       float *h_last, float *c_last)
{
    for (int j=0;j<Hsz;++j){ h_last[j]=0.0f; c_last[j]=0.0f; }

    for (int t=0; t<Tlen; ++t) {
        float z[4*Hsz];
        // start from bias
        for (int gk=0; gk<4*Hsz; ++gk) z[gk] = b_ifog ? b_ifog[gk] : 0.0f;

        // input contribution
        const float *xt = x + t*D;
        for (int d=0; d<D; ++d) {
            float xv = xt[d];
            const float *Wd = W_ifog + d*(4*Hsz);
            for (int gk=0; gk<4*Hsz; ++gk) z[gk] += xv * Wd[gk];
        }
        // recurrent contribution
        for (int hp=0; hp<Hsz; ++hp) {
            float hv = h_last[hp];
            const float *Rh = R_ifog + hp*(4*Hsz);
            for (int gk=0; gk<4*Hsz; ++gk) z[gk] += hv * Rh[gk];
        }
        // elementwise update
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

// ---------- tests ----------
int main(void){
    // ----- Test 1: bias-only dynamics (W=R=0), easy hand-check -----
    // H=1, D=1, T=3. i=f=o = sigmoid(0)=0.5. g = tanh(atanh(0.2))=0.2.
    const int H1=1, D1=1, T1=3;
    float x1[T1*D1]; memset(x1, 0, sizeof(x1));
    float W1[D1*4*H1]; memset(W1, 0, sizeof(W1));
    float R1[H1*4*H1]; memset(R1, 0, sizeof(R1));
    float b1[] = { 0.0f, 0.0f, 0.0f, atanhf(0.2f) }; // [i,f,o,g]

    float h1[H1], c1[H1];
    lstm_forward_unidir(x1, T1, D1, W1, R1, b1, H1, h1, c1);

    // expected: c: 0.1, 0.15, 0.175 ; h_t = 0.5 * tanh(c_t)
    float c1_exp_last = 0.0f;
    float h1_exp_last = 0.0f;
    float c=0.0f;
    for(int t=0;t<T1;++t){ c = 0.5f*c + 0.5f*0.2f; } // final c
    c1_exp_last = c;
    h1_exp_last = 0.5f * tanhf(c1_exp_last);

    print_vec("Test1 h_last", h1, H1);
    print_vec("Test1 c_last", c1, H1);
    printf("Test1: %s\n\n",
        (fabsf(h1[0]-h1_exp_last)<1e-6f && fabsf(c1[0]-c1_exp_last)<1e-6f) ? "PASS" : "FAIL");

    // ----- Test 2: input-weight mapping sanity (R=0) -----
    // H=1, D=2, T=1. Set distinct weights per gate:
    // z_i = 1*x0 + 0*x1 ; z_f = 0*x0 + 2*x1 ; z_o = -1*x0 + 0*x1 ; z_g = 0*x0 + 0.5*x1
    const int H2=1, D2=2, T2=1;
    float x2[] = { 1.0f, 2.0f }; // [x0, x1]
    float W2[] = {
        1.0f, 0.0f, -1.0f, 0.0f,   // d=0 -> [i,f,o,g]
        0.0f, 2.0f,  0.0f, 0.5f    // d=1 -> [i,f,o,g]
    };
    float R2[H2*4*H2]; memset(R2, 0, sizeof(R2));
    float b2[4*H2]; memset(b2, 0, sizeof(b2));

    float h2[H2], c2[H2];
    lstm_forward_unidir(x2, T2, D2, W2, R2, b2, H2, h2, c2);

    // expected (do the math directly):
    float zi = 1.0f;
    float zf = 4.0f;
    float zo = -1.0f;
    float zg = 1.0f;
    float i = sigmoidf(zi), f = sigmoidf(zf), o = sigmoidf(zo), g = tanhf(zg);
    float c_exp = f*0.0f + i*g;
    float h_exp = o*tanhf(c_exp);

    print_vec("Test2 h_last", h2, H2);
    print_vec("Test2 c_last", c2, H2);
    printf("Test2: %s\n\n",
        (fabsf(h2[0]-h_exp)<1e-6f && fabsf(c2[0]-c_exp)<1e-6f) ? "PASS" : "FAIL");

    // ----- Test 3: recurrent-weight mapping (kick with bias, then use R on g) -----
    // H=1, D=1, T=2. W=0. R_g = 1.0. Bias g=atanh(0.2), i=f=o=0.
    const int H3=1, D3=1, T3=2;
    float x3[T3*D3]; memset(x3, 0, sizeof(x3));
    float W3[D3*4*H3]; memset(W3, 0, sizeof(W3));
    float R3[] = { 0.0f, 0.0f, 0.0f, 1.0f }; // only g gate gets h_{t-1}
    float b3[] = { 0.0f, 0.0f, 0.0f, atanhf(0.2f) };

    float h3[H3], c3[H3];
    lstm_forward_unidir(x3, T3, D3, W3, R3, b3, H3, h3, c3);

    // expected: step0 uses bias-only (like Test1 first step), step1 uses z_g += h0
    float c_prev=0.0f, h_prev=0.0f;
    // t=0
    float i0=sigmoidf(0.0f), f0=sigmoidf(0.0f), o0=sigmoidf(0.0f), g0=tanhf(atanhf(0.2f));
    float c0=f0*c_prev + i0*g0;
    float h0=o0*tanhf(c0);
    // t=1
    float i1=i0, f1=f0, o1=o0, g1=tanhf(atanhf(0.2f) + h0);
    float c11=f1*c0 + i1*g1;
    float h11=o1*tanhf(c11);

    print_vec("Test3 h_last", h3, H3);
    print_vec("Test3 c_last", c3, H3);
    printf("Test3: %s\n",
        (fabsf(h3[0]-h11)<1e-6f && fabsf(c3[0]-c11)<1e-6f) ? "PASS" : "FAIL");

    return 0;
}
