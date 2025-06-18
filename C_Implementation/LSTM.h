#ifndef LSTM_H
#define LSTM_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#define SIGMOID(x) (1 / (1 + exp(-x)))
#define TANH(x) (tanh(x))
#define RELU(x) ((x) > 0 ? (x) : 0)

/*
function lstm_inference(input, weight, bias, hidden_size, num_layers, bidirectional, batch_first):
    # input: tensor of shape (L_in, C_in)  (single sequence, no batch size)
    # weight: contains the weight matrices for input-to-hidden and hidden-to-hidden
    # bias: contains the bias vectors for each gate (for each layer)
    # hidden_size: number of units in the hidden state
    # num_layers: number of LSTM layers (default is 1)
    # bidirectional: flag to determine if it's bidirectional
    # batch_first: flag to specify input format

    # Initialize hidden state and cell state for each layer
    h_state = zeros(num_layers * (2 if bidirectional else 1), hidden_size)  # Initialize hidden state
    c_state = zeros(num_layers * (2 if bidirectional else 1), hidden_size)  # Initialize cell state

    # Process the input sequence through the LSTM layers
    for layer in 1 to num_layers:  # Iterate through each LSTM layer
        # Get the previous layer’s hidden and cell state (or zero for the first layer)
        h_prev = h_state[layer - 1]
        c_prev = c_state[layer - 1]

        # Compute the gates (forget, input, output, and cell update) for the current time step
        f_t = sigmoid(W_f * input + U_f * h_prev + b_f)    # Forget gate
        i_t = sigmoid(W_i * input + U_i * h_prev + b_i)    # Input gate
        o_t = sigmoid(W_o * input + U_o * h_prev + b_o)    # Output gate
        g_t = tanh(W_g * input + U_g * h_prev + b_g)       # Candidate cell state

        # Update the cell state
        c_t = f_t * c_prev + i_t * g_t  # New cell state

        # Compute the new hidden state
        h_t = o_t * tanh(c_t)  # Hidden state

        # Store the updated hidden and cell state for the current layer
        h_state[layer] = h_t
        c_state[layer] = c_t

        if bidirectional is True:
            # If bidirectional, reverse the sequence and apply the same process for the reverse direction
            h_t_reverse = reverse_sequence(input)
            # Concatenate the forward and reverse hidden states for each time step
            h_state[layer] = concatenate(h_t, h_t_reverse)
                
    return h_state

*/

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
