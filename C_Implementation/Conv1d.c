#include <stdio.h>
#include <stdlib.h>
#include "Conv1d.h"

int main() {
    int input_size = 12;
    int kernel_size = 2;
    int output_size = input_size - kernel_size + 1; 

    int *input = (int *) malloc(input_size * sizeof(int));
    int *kernel = (int *) malloc(kernel_size * sizeof(int));
    int temp_input[] = {9, 7, 2, 4, 8, 7, 3, 1, 5, 9, 8, 4};
    int temp_kernel[] = {3, 6};

    for (int i = 0; i < input_size; i++) {
        input[i] = temp_input[i];
    }
    for (int i = 0; i < kernel_size; i++) {
        kernel[i] = temp_kernel[i];
    }
    int *output = Conv1d(input, input_size, kernel, kernel_size);
    if (output == NULL) {
        fprintf(stderr, "Error in convolution operation\n");
        return 1;
    }
    printf("Output of Conv1d:\n");
    for (int i = 0; i < output_size; i++) {
        printf("%d ", output[i]);
    }
    printf("\n");
    free(output);
    output = NULL;

    return 0;
}