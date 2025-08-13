#ifndef LSTM_H
#define LSTM_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#define SIGMOID(x) (1 / (1 + exp(-x)))
#define TANH(x) (tanh(x))
#define RELU(x) ((x) > 0 ? (x) : 0)

float *lstm(float *input, int input_size, float *weight, float *bias, int hidden_size, int num_layers, int bidirectional, int batch_first) {
    // # Initialize hidden state and cell state for each layer
    float *h_state = (float *)calloc(num_layers * (2 * bidirectional + 1), hidden_size * sizeof(float));
    float *c_state = (float *)calloc(num_layers * (2 * bidirectional + 1), hidden_size * sizeof(float));

    for(int layer=0; layer < num_layers; layer++) {
        // Get the previous layer’s hidden and cell state (or zero for the first layer)
        float *h_prev = &h_state[layer * hidden_size];
        float *c_prev = &c_state[layer * hidden_size];

        for(int t=0; t < input_size; t++) {
            // Compute the gates (forget, input, output, and cell update) for the current time step
            float f_t = SIGMOID(weight[0] * input[t] + weight[1] * h_prev[t] + bias[0]);
            float i_t = SIGMOID(weight[2] * input[t] + weight[3] * h_prev[t] + bias[1]);
            float o_t = SIGMOID(weight[4] * input[t] + weight[5] * h_prev[t] + bias[2]);
            float g_t = TANH(weight[6] * input[t] + weight[7] * h_prev[t] + bias[3]);

            // Update the cell state
            c_state[layer * hidden_size + t] = f_t * c_prev[t] + i_t * g_t;

            // Compute the new hidden state
            h_state[layer * hidden_size + t] = o_t * TANH(c_state[layer * hidden_size + t]);
        }

        if(bidirectional) {
            // Reverse the sequence and apply the same process for the reverse direction
            float *h_t_reverse = (float *)calloc(hidden_size, sizeof(float));
            for(int t=0; t < input_size; t++) {
                h_t_reverse[t] = h_state[layer * hidden_size + (input_size - 1 - t)];
            }
            // Concatenate the forward and reverse hidden states for each time step
            for(int t=0; t < input_size; t++) {
                h_state[layer * hidden_size + t] = (h_state[layer * hidden_size + t] + h_t_reverse[t]) / 2; // Simple average for demonstration
            }
            free(h_t_reverse);
        }
    }
}

#endif // LSTM_H
