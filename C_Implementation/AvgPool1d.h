#ifndef AVGPOOL1D_H
#define AVGPOOL1D_H

#include <stdio.h>
#include <stdlib.h>

float *avg_pooling_1d(float *input, int input_size, int kernel_size, int stride) {
    // Calculate the output size based on kernel size and stride
    int output_size = (input_size - kernel_size) / stride + 1;

    float *output = (float*) malloc(output_size * sizeof(float));

    // Perform average pooling operation
    for(int i = 0; i < output_size; i++) {
        float sum = 0;
        // Calculate the sum for each window with the given kernel size and stride
        for(int j = 0; j < kernel_size; j++) {
            sum += input[i * stride + j];  // Move by stride each time
        }
        output[i] = sum / kernel_size;  // Average the values in the window
    }

    return output;
}

#endif // AVGPOOL1D_H