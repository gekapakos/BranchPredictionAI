// tb_main_multi.cpp  (test bench)
#include <cstdio>
#include <cmath>
#include "hls_half.h"

// The design is compiled as C; tell C++ about it:
extern "C" void run_model(half *y_hat_out);

// === Put your known-good reference here (from Python/float pipeline, etc.) ===
// If you already have the golden, paste it below:
static const float GOLDEN_Y = 0.5058176517486572f;   // <-- replace with your golden
static const float EPS       = 1e-3f;       // tolerance; relax to 5e-3 or 1e-2 if needed

int main() {
    half yh;
    run_model(&yh);

    float y = (float)yh;  // compare in float
    float diff = std::fabs(y - GOLDEN_Y);

    std::printf("TB: y_hat = %.7e, golden = %.7e, |diff|=%.3e, eps=%.3e\n", y, GOLDEN_Y, diff, EPS);

    if (diff > EPS) {
        std::puts("TB: FAIL");
        return 1;   // non-zero means failure
    } 
    else {
        std::puts("TB: PASS");
        return 0;   // zero means success
    }
}
