#ifndef DEFINES_H_
#define DEFINES_H_

#include "ap_fixed.h"
#include "ap_int.h"
#include "nnet_utils/nnet_types.h"
#include <cstddef>
#include <cstdio>

// hls-fpga-machine-learning insert numbers
#define N_INPUT_1_1 28
#define N_INPUT_2_1 28
#define N_INPUT_3_1 1
#define OUT_HEIGHT_2 26
#define OUT_WIDTH_2 26
#define N_FILT_2 8
#define OUT_HEIGHT_2 26
#define OUT_WIDTH_2 26
#define N_FILT_2 8
#define N_SIZE_0_4 5408
#define N_LAYER_5 10


// hls-fpga-machine-learning insert layer-precision
typedef ap_fixed<16,6> input_t;
typedef ap_fixed<16,6> model_default_t;
typedef ap_fixed<29,14> conv_result_t;
typedef ap_fixed<8,3> conv_weight_t;
typedef ap_fixed<16,6> conv_bias_t;
typedef ap_fixed<16,6> layer3_t;
typedef ap_fixed<18,8> relu_table_t;
typedef ap_fixed<38,23> result_t;
typedef ap_fixed<8,3> fc_weight_t;
typedef ap_fixed<16,6> fc_bias_t;
typedef ap_uint<1> layer5_index;


#endif
