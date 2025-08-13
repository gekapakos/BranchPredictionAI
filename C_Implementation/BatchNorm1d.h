#ifndef BATCHNORM1D_H
#define BATCHNORM1D_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

int *batch_norm_1d(int size, int *mean, int* var, int *input, float epsilon)
{
    int *output = (int *) malloc(size * sizeof(int));
    for(int i = 0; i < size; i++) {
        output[i] = (int)((input[i] - mean[i]) / (sqrt(var[i] + epsilon)));
    }

    return output;
}

#endif // BATCHNORM1D_H