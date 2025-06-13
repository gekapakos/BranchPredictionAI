#ifndef AVGPOOL1D_H
#define AVGPOOL1D_H

#include <stdio.h>
#include <stdlib.h>

int *avg_pooling_1d(int *input, int input_size, int kernel_size) {
    
    int output_size = input_size / kernel_size;
    int *output = (int*) malloc(output_size * sizeof(int));
    for(int i = 0; i < output_size; i++) {
        int sum = 0;
        for(int j= 0; j < kernel_size; j++) {
            sum += input[i * kernel_size + j];
        }
        output[i] = sum / kernel_size;
    }

    return output;
}

#endif // AVGPOOL1D_H
