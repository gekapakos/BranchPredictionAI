#include <iostream>

#include "mnist.h"
#include "parameters.h"


void mnist(
    input_t x[N_INPUT_1_1*N_INPUT_2_1*N_INPUT_3_1],
    result_t layer5_out[N_LAYER_5]
) {

    // hls-fpga-machine-learning insert IO
    #pragma HLS ARRAY_RESHAPE variable=x complete dim=0
    #pragma HLS ARRAY_PARTITION variable=layer5_out complete dim=0
    #pragma HLS INTERFACE ap_vld port=x,layer5_out 
    #pragma HLS DATAFLOW

    // hls-fpga-machine-learning insert load weights
#ifndef __SYNTHESIS__
    static bool loaded_weights = false;
    if (!loaded_weights) {
        nnet::load_weights_from_txt<conv_weight_t, 72>(w2, "w2.txt");
        nnet::load_weights_from_txt<conv_bias_t, 8>(b2, "b2.txt");
        nnet::load_weights_from_txt<fc_weight_t, 54080>(w5, "w5.txt");
        nnet::load_weights_from_txt<fc_bias_t, 10>(b5, "b5.txt");
        loaded_weights = true;    }
#endif
    // ****************************************
    // NETWORK INSTANTIATION
    // ****************************************

    // hls-fpga-machine-learning insert layers

    conv_result_t layer2_out[OUT_HEIGHT_2*OUT_WIDTH_2*N_FILT_2];
    #pragma HLS ARRAY_PARTITION variable=layer2_out complete dim=0
    nnet::conv_2d_cl<input_t, conv_result_t, config2>(x, layer2_out, w2, b2); // conv

    layer3_t layer3_out[OUT_HEIGHT_2*OUT_WIDTH_2*N_FILT_2];
    #pragma HLS ARRAY_PARTITION variable=layer3_out complete dim=0
    nnet::relu<conv_result_t, layer3_t, relu_config3>(layer2_out, layer3_out); // relu

    auto& layer4_out = layer3_out;
    nnet::dense<layer3_t, result_t, config5>(layer4_out, layer5_out, w5, b5); // fc

}

