#ifndef KLADI_H
#define KLADI_H

#include <stdio.h>
#include <stdlib.h>
#include "Relu.h"
#include "Conv1d.h"
#include "BatchNorm1d.h"
#include "LSTM.h"
#include "Linear.h"
#include "AvgPool1d.h"
#include "Embedding.h"

int HISTORY_LENGTHS[] = {42, 78, 150, 294, 582};
int CONV_FILTERS[]    = {32, 32, 32, 32, 32};
int CONV_WIDTHS[]     = {7, 7, 7, 7, 7};
int POOL_WIDTHS[]     = {3, 6, 12, 24, 48};
int PC_HASH_BITS    = 12;            // bits of hashed PC value
int HASH_DIR_WITH_PC = true;         // include branch direction bit
int EMBED_DIM       = 32;
int USE_LSTM        = true;
int LSTM_HIDDEN     = 128;
int BIDIRECTIONAL   = true;
int HIDDEN_NEURONS[]  = {128, 128};

float Kladi(float *input, int input_size,
            float *kernel, int kernel_size,
            int pool_width) {
    // Embedding first
    /*
    
void embedding(const float *weight,
                      int num_embeddings,
                      int embedding_dim,
                      const int *indices,
                      int N)
    
    */
    float *embedded_input = embedding(input, input_size, EMBED_DIM);
    if (!embedded_input) {
        fprintf(stderr, "Embedding failed\n");
        return NULL;
    }

    // Conv1d
    float *conv1d_out = Conv1d(input, EMBED_DIM, kernel, kernel_size);
    if (!conv1d_out) {
        fprintf(stderr, "Conv1d failed\n");
        return NULL;
    }
    // Relu activation
    float *relu_out = RELU(conv1d_out);
    free(conv1d_out);
    if (!relu_out) {
        fprintf(stderr, "Relu activation failed\n");
        return NULL;
    }
    // AvgPool1d
    float *pool_out = avg_pooling_1d(relu_out, input_size, pool_width, pool_width);

    free(relu_out);
    if (!pool_out) {
        fprintf(stderr, "AvgPool1d failed\n");
        return NULL;
    }

    // Bi-LSTM
    // float *lstm(float *input, int input_size, float *weight, float *bias, int hidden_size, int num_layers, int bidirectional, int batch_first)
    float *lstm_out = lstm(pool_out, input_size / pool_width, kernel, NULL, LSTM_HIDDEN, 1, BIDIRECTIONAL, 0);
    free(pool_out);
    if (!lstm_out) {
        fprintf(stderr, "LSTM failed\n");
        return NULL;
    }
}

#endif // KLADI_H
