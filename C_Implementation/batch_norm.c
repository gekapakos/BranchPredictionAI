// bn_inference_example.c
#include <stdio.h>
#include <math.h>
#include <stdbool.h>

// ======= Your BN kernels (inference over last dim) =======
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

// ======= Test helpers =======
static bool allclose(const float *a, const float *b, int n, float atol, float rtol)
{
    for (int i = 0; i < n; ++i) {
        float diff = fabsf(a[i] - b[i]);
        float tol  = atol + rtol * fmaxf(fabsf(a[i]), fabsf(b[i]));
        if (diff > tol) return false;
    }
    return true;
}

static void print_vec(const char *name, const float *v, int n) {
    printf("%s = [", name);
    for (int i = 0; i < n; ++i) printf("%s%.6f", i?", ":"", v[i]);
    printf("]\n");
}

int main(void)
{
    // Channels C=3, two time steps (Tlen=2). We choose friendly numbers.
    // Per-channel running stats and affine params (from training):
    // mean = [1, -1, 0], var = [1, 4, 9], gamma = [1, 2, 3], beta = [0, 0.5, -1]
    // eps = 0 for exact arithmetic in this demo.
    const int C = 3, Tlen = 2;
    float gamma[3] = {1.0f, 2.0f, 3.0f};
    float beta [3] = {0.0f, 0.5f, -1.0f};
    float mean [3] = {1.0f,-1.0f, 0.0f};
    float var  [3] = {1.0f, 4.0f, 9.0f};
    float eps = 0.0f;

    // Input Y (shape [Tlen, C], row-major):
    // t=0: [2, -1,  9]
    // t=1: [1,  3, -3]
    float Y[2*3] = { 2.f,-1.f, 9.f,
                     1.f, 3.f,-3.f };

    // Apply BN over last dim (per-channel)
    bn_time_channel(Y, Tlen, C, gamma, beta, mean, var, eps);

    // Hand-computed expected outputs:
    // inv_std = [1, 1/2, 1/3]
    // t=0: [(2-1)*1*1 + 0   , (-1+1)*.5*2 + .5, (9-0)*(1/3)*3 - 1] = [1, 0.5, 8]
    // t=1: [(1-1)*1*1 + 0   , (3+1)*.5*2 + .5, (-3-0)*(1/3)*3 - 1] = [0, 4.5, -4]
    float Y_exp[2*3] = { 1.f, 0.5f,  8.f,
                         0.f, 4.5f, -4.f };

    // Show results
    print_vec("Y_out[t0]", &Y[0], C);
    print_vec("exp  [t0]", &Y_exp[0], C);
    print_vec("Y_out[t1]", &Y[3], C);
    print_vec("exp  [t1]", &Y_exp[3], C);

    printf("\nTime-channel test: %s\n\n",
           allclose(Y, Y_exp, 2*C, 1e-6f, 0.0f) ? "PASS" : "FAIL");

    // Also test the vector version on t0's original input
    float v[3] = {2.f, -1.f, 9.f};
    bn_vector(v, C, gamma, beta, mean, var, eps);
    print_vec("bn_vector(v)", v, C);
    print_vec("expected    ", Y_exp, C);
    printf("Vector test: %s\n", allclose(v, Y_exp, C, 1e-6f, 0.0f) ? "PASS" : "FAIL");

    return 0;
}
