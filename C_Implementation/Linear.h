#ifndef LINEAR_H
#define LINEAR_H

#include <stdio.h>
#include <stdlib.h>

int *linear(int *input, int D_in, int *W, int *bias, int D_out, int *output) {
    int *output = (int *)malloc(D_out * sizeof(int));
    for (int j = 0; j < D_out; j++) {
        int sum = bias[j];
        const int *row = W + j * D_in;
        for (int i = 0; i < D_in; i++) {
            sum += row[i] * input[i];
        }

        output[j] = sum;
    }

    return output;
}

#endif // LINEAR_H