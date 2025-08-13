#ifndef CONV1D_H
#define CONV1D_H

#include <stdio.h>
#include <stdlib.h>

float *Conv1d(float *input, int input_size,
            float *kernel, int kernel_size)
{
    int output_size = input_size - kernel_size + 1;
    
    if (output_size <= 0) 
        return NULL;

    float *output = (float*) calloc(output_size, sizeof(float));
    if (!output) {
        fprintf(stderr, "Memory allocation failed\n");
        return NULL;
    }

    for (int i = 0; i < output_size; i++) {
        int acc = 0;
        for (int j = 0; j < kernel_size; j++) {
            acc += input[i + j] * kernel[j];
        }
        output[i] = acc;
    }

    return output;
}

#endif // CONV1D_H
