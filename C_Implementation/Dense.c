// slice_test.c
#include <stdio.h>
#include <math.h>
#include <stdint.h>
#include <stdbool.h>
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
    for (int j = 0; j < Out; ++j) {
        float acc = b ? b[j] : 0.0f;
        for (int i = 0; i < In; ++i) {
            acc += x[i] * W[i*Out + j]; // W row-major [In][Out]
        }
        y[j] = acc;
    }
}

static bool allclose(const float *a, const float *b, int n, float atol, float rtol)
{
    for (int k = 0; k < n; ++k) {
        float diff = fabsf(a[k] - b[k]);
        float tol  = atol + rtol * fmaxf(fabsf(a[k]), fabsf(b[k]));
        if (diff > tol) return false;
    }
    return true;
}

static void print_vec(const char *name, const float *v, int n){
    printf("%s = [", name);
    for (int i = 0; i < n; ++i) {
        printf("%s%.6f", (i ? ", " : ""), v[i]);
    }
    printf("]\n");
}

int main(void)
{
    // Test 1: small 3x2 layer with bias
    // W row-major as [In][Out]:
    //   [ [1,2],
    //     [3,4],
    //     [5,6] ]
    const int In = 3, Out = 2;
    const float x[3] = {1.0f, 2.0f, 3.0f};
    const float W[6] = {1,2, 3,4, 5,6}; // row-major: W[i*Out + j]
    const float b[2] = {0.5f, -1.0f};

    float y[2];
    dense_forward(x, In, W, b, Out, y);

    // Expected:
    // y0 = 1*1 + 2*3 + 3*5 + 0.5 = 22.5
    // y1 = 1*2 + 2*4 + 3*6 - 1.0 = 27.0
    const float y_exp[2] = {22.5f, 27.0f};

    print_vec("y (with bias)", y, Out);
    print_vec("expected      ", y_exp, Out);
    printf("Test 1 %s\n\n", allclose(y, y_exp, Out, 1e-6f, 0.0f) ? "PASS" : "FAIL");

    // Test 2: same but b == NULL (no bias)
    float y2[2];
    dense_forward(x, In, W, NULL, Out, y2);
    const float y2_exp[2] = {22.0f, 28.0f}; // same sums without bias
    print_vec("y (no bias)  ", y2, Out);
    print_vec("expected      ", y2_exp, Out);
    printf("Test 2 %s\n\n", allclose(y2, y2_exp, Out, 1e-6f, 0.0f) ? "PASS" : "FAIL");

    // Test 3: edge-ish case In=1, Out=3
    const int In3 = 1, Out3 = 3;
    const float x3[1] = {2.0f};
    const float W3[3] = {10.0f, -3.0f, 0.25f}; // shape [1][3]
    const float b3[3] = {0.0f, 1.0f, -2.0f};
    float y3[3];
    dense_forward(x3, In3, W3, b3, Out3, y3);
    const float y3_exp[3] = {20.0f, -5.0f, -1.5f}; // 2*W3 + b3
    print_vec("y3           ", y3, Out3);
    print_vec("expected     ", y3_exp, Out3);
    printf("Test 3 %s\n", allclose(y3, y3_exp, Out3, 1e-6f, 0.0f) ? "PASS" : "FAIL");

    return 0;
}