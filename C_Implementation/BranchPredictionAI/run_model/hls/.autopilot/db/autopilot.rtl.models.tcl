set SynModuleInfo {
  {SRCNAME run_all_slices_unrolled_Pipeline_MergedLoop MODELNAME run_all_slices_unrolled_Pipeline_MergedLoop RTLNAME run_model_run_all_slices_unrolled_Pipeline_MergedLoop
    SUBMODULES {
      {MODELNAME run_model_flow_control_loop_pipe_sequential_init RTLNAME run_model_flow_control_loop_pipe_sequential_init BINDTYPE interface TYPE internal_upc_flow_control INSTNAME run_model_flow_control_loop_pipe_sequential_init_U}
    }
  }
  {SRCNAME lstm_forward_unidir_Pipeline_VITIS_LOOP_145_1 MODELNAME lstm_forward_unidir_Pipeline_VITIS_LOOP_145_1 RTLNAME run_model_lstm_forward_unidir_Pipeline_VITIS_LOOP_145_1}
  {SRCNAME lstm_forward_unidir_Pipeline_Loop2_1LSTM MODELNAME lstm_forward_unidir_Pipeline_Loop2_1LSTM RTLNAME run_model_lstm_forward_unidir_Pipeline_Loop2_1LSTM
    SUBMODULES {
      {MODELNAME run_model_mul_7ns_9ns_15_1_1 RTLNAME run_model_mul_7ns_9ns_15_1_1 BINDTYPE op TYPE mul IMPL auto LATENCY 0 ALLOW_PRAGMA 1}
      {MODELNAME run_model_sparsemux_257_7_16_1_1 RTLNAME run_model_sparsemux_257_7_16_1_1 BINDTYPE op TYPE sparsemux IMPL compactencoding_dontcare}
    }
  }
  {SRCNAME lstm_forward_unidir_Pipeline_Loop2_4LSTM MODELNAME lstm_forward_unidir_Pipeline_Loop2_4LSTM RTLNAME run_model_lstm_forward_unidir_Pipeline_Loop2_4LSTM
    SUBMODULES {
      {MODELNAME run_model_fadd_32ns_32ns_32_4_full_dsp_1 RTLNAME run_model_fadd_32ns_32ns_32_4_full_dsp_1 BINDTYPE op TYPE fadd IMPL fulldsp LATENCY 3 ALLOW_PRAGMA 1}
      {MODELNAME run_model_fdiv_32ns_32ns_32_10_no_dsp_1 RTLNAME run_model_fdiv_32ns_32ns_32_10_no_dsp_1 BINDTYPE op TYPE fdiv IMPL fabric LATENCY 9 ALLOW_PRAGMA 1}
      {MODELNAME run_model_fexp_32ns_32ns_32_8_full_dsp_1 RTLNAME run_model_fexp_32ns_32ns_32_8_full_dsp_1 BINDTYPE op TYPE fexp IMPL fulldsp LATENCY 7 ALLOW_PRAGMA 1}
      {MODELNAME run_model_sptohp_32ns_16_2_no_dsp_1 RTLNAME run_model_sptohp_32ns_16_2_no_dsp_1 BINDTYPE op TYPE sptohp IMPL auto LATENCY 1 ALLOW_PRAGMA 1}
      {MODELNAME run_model_hptosp_16ns_32_2_no_dsp_1 RTLNAME run_model_hptosp_16ns_32_2_no_dsp_1 BINDTYPE op TYPE hptosp IMPL auto LATENCY 1 ALLOW_PRAGMA 1}
      {MODELNAME run_model_hmul_16ns_16ns_16_4_max_dsp_1 RTLNAME run_model_hmul_16ns_16ns_16_4_max_dsp_1 BINDTYPE op TYPE hmul IMPL maxdsp LATENCY 3 ALLOW_PRAGMA 1}
      {MODELNAME run_model_hcmp_16ns_16ns_1_2_no_dsp_1 RTLNAME run_model_hcmp_16ns_16ns_1_2_no_dsp_1 BINDTYPE op TYPE hcmp IMPL auto LATENCY 1 ALLOW_PRAGMA 1}
      {MODELNAME run_model_mul_5ns_7ns_11_1_1 RTLNAME run_model_mul_5ns_7ns_11_1_1 BINDTYPE op TYPE mul IMPL auto LATENCY 0 ALLOW_PRAGMA 1}
      {MODELNAME run_model_mul_6ns_8ns_13_1_1 RTLNAME run_model_mul_6ns_8ns_13_1_1 BINDTYPE op TYPE mul IMPL auto LATENCY 0 ALLOW_PRAGMA 1}
      {MODELNAME run_model_sparsemux_11_3_16_1_1 RTLNAME run_model_sparsemux_11_3_16_1_1 BINDTYPE op TYPE sparsemux IMPL compactencoding_dontcare}
      {MODELNAME run_model_sparsemux_7_2_16_1_1 RTLNAME run_model_sparsemux_7_2_16_1_1 BINDTYPE op TYPE sparsemux IMPL onehotencoding_realdef}
    }
  }
  {SRCNAME lstm_forward_unidir MODELNAME lstm_forward_unidir RTLNAME run_model_lstm_forward_unidir
    SUBMODULES {
      {MODELNAME run_model_hadd_16ns_16ns_16_5_full_dsp_1 RTLNAME run_model_hadd_16ns_16ns_16_5_full_dsp_1 BINDTYPE op TYPE hadd IMPL fulldsp LATENCY 4 ALLOW_PRAGMA 1}
      {MODELNAME run_model_urem_8ns_4ns_3_12_1 RTLNAME run_model_urem_8ns_4ns_3_12_1 BINDTYPE op TYPE urem IMPL auto LATENCY 11 ALLOW_PRAGMA 1}
      {MODELNAME run_model_mul_8ns_10ns_17_1_1 RTLNAME run_model_mul_8ns_10ns_17_1_1 BINDTYPE op TYPE mul IMPL auto LATENCY 0 ALLOW_PRAGMA 1}
      {MODELNAME run_model_lstm_forward_unidir_c_slice_RAM_AUTO_1R1W RTLNAME run_model_lstm_forward_unidir_c_slice_RAM_AUTO_1R1W BINDTYPE storage TYPE ram IMPL auto LATENCY 2 ALLOW_PRAGMA 1}
      {MODELNAME run_model_lstm_forward_unidir_z_RAM_AUTO_1R1W RTLNAME run_model_lstm_forward_unidir_z_RAM_AUTO_1R1W BINDTYPE storage TYPE ram IMPL auto LATENCY 2 ALLOW_PRAGMA 1}
    }
  }
  {SRCNAME run_all_slices_unrolled_Pipeline_Loop_BN MODELNAME run_all_slices_unrolled_Pipeline_Loop_BN RTLNAME run_model_run_all_slices_unrolled_Pipeline_Loop_BN
    SUBMODULES {
      {MODELNAME run_model_run_all_slices_unrolled_Pipeline_Loop_BN_BN2_var0_ROM_AUTO_1R RTLNAME run_model_run_all_slices_unrolled_Pipeline_Loop_BN_BN2_var0_ROM_AUTO_1R BINDTYPE storage TYPE rom IMPL auto LATENCY 2 ALLOW_PRAGMA 1}
    }
  }
  {SRCNAME run_all_slices_unrolled_Pipeline_MergedLoop0 MODELNAME run_all_slices_unrolled_Pipeline_MergedLoop0 RTLNAME run_model_run_all_slices_unrolled_Pipeline_MergedLoop0}
  {SRCNAME conv_bn_act_pool_Pipeline_BNParamsLoop MODELNAME conv_bn_act_pool_Pipeline_BNParamsLoop RTLNAME run_model_conv_bn_act_pool_Pipeline_BNParamsLoop
    SUBMODULES {
      {MODELNAME run_model_sparsemux_65_5_16_1_1 RTLNAME run_model_sparsemux_65_5_16_1_1 BINDTYPE op TYPE sparsemux IMPL compactencoding_dontcare}
    }
  }
  {SRCNAME conv_bn_act_pool_Pipeline_Loop3_2Big MODELNAME conv_bn_act_pool_Pipeline_Loop3_2Big RTLNAME run_model_conv_bn_act_pool_Pipeline_Loop3_2Big}
  {SRCNAME conv_bn_act_pool MODELNAME conv_bn_act_pool RTLNAME run_model_conv_bn_act_pool
    SUBMODULES {
      {MODELNAME run_model_conv_bn_act_pool_X_slice12_ROM_AUTO_3R RTLNAME run_model_conv_bn_act_pool_X_slice12_ROM_AUTO_3R BINDTYPE storage TYPE rom IMPL auto LATENCY 4 ALLOW_PRAGMA 1}
      {MODELNAME run_model_conv_bn_act_pool_ConvW1_ROM_AUTO_3R RTLNAME run_model_conv_bn_act_pool_ConvW1_ROM_AUTO_3R BINDTYPE storage TYPE rom IMPL auto LATENCY 4 ALLOW_PRAGMA 1}
    }
  }
  {SRCNAME run_all_slices_unrolled_Pipeline_Loop_BN2 MODELNAME run_all_slices_unrolled_Pipeline_Loop_BN2 RTLNAME run_model_run_all_slices_unrolled_Pipeline_Loop_BN2}
  {SRCNAME run_all_slices_unrolled_Pipeline_MergedLoop1 MODELNAME run_all_slices_unrolled_Pipeline_MergedLoop1 RTLNAME run_model_run_all_slices_unrolled_Pipeline_MergedLoop1}
  {SRCNAME conv_bn_act_pool.2_Pipeline_BNParamsLoop MODELNAME conv_bn_act_pool_2_Pipeline_BNParamsLoop RTLNAME run_model_conv_bn_act_pool_2_Pipeline_BNParamsLoop}
  {SRCNAME conv_bn_act_pool.2_Pipeline_Loop3_2Big MODELNAME conv_bn_act_pool_2_Pipeline_Loop3_2Big RTLNAME run_model_conv_bn_act_pool_2_Pipeline_Loop3_2Big}
  {SRCNAME conv_bn_act_pool.2 MODELNAME conv_bn_act_pool_2 RTLNAME run_model_conv_bn_act_pool_2
    SUBMODULES {
      {MODELNAME run_model_conv_bn_act_pool_2_X_slice27_ROM_AUTO_3R RTLNAME run_model_conv_bn_act_pool_2_X_slice27_ROM_AUTO_3R BINDTYPE storage TYPE rom IMPL auto LATENCY 4 ALLOW_PRAGMA 1}
      {MODELNAME run_model_conv_bn_act_pool_2_ConvW2_ROM_AUTO_3R RTLNAME run_model_conv_bn_act_pool_2_ConvW2_ROM_AUTO_3R BINDTYPE storage TYPE rom IMPL auto LATENCY 4 ALLOW_PRAGMA 1}
    }
  }
  {SRCNAME run_all_slices_unrolled_Pipeline_Loop_BN3 MODELNAME run_all_slices_unrolled_Pipeline_Loop_BN3 RTLNAME run_model_run_all_slices_unrolled_Pipeline_Loop_BN3}
  {SRCNAME run_all_slices_unrolled_Pipeline_MergedLoop2 MODELNAME run_all_slices_unrolled_Pipeline_MergedLoop2 RTLNAME run_model_run_all_slices_unrolled_Pipeline_MergedLoop2}
  {SRCNAME conv_bn_act_pool.3_Pipeline_BNParamsLoop MODELNAME conv_bn_act_pool_3_Pipeline_BNParamsLoop RTLNAME run_model_conv_bn_act_pool_3_Pipeline_BNParamsLoop}
  {SRCNAME conv_bn_act_pool.3_Pipeline_Loop3_2Big MODELNAME conv_bn_act_pool_3_Pipeline_Loop3_2Big RTLNAME run_model_conv_bn_act_pool_3_Pipeline_Loop3_2Big}
  {SRCNAME conv_bn_act_pool.3 MODELNAME conv_bn_act_pool_3 RTLNAME run_model_conv_bn_act_pool_3
    SUBMODULES {
      {MODELNAME run_model_conv_bn_act_pool_3_X_slice42_ROM_AUTO_3R RTLNAME run_model_conv_bn_act_pool_3_X_slice42_ROM_AUTO_3R BINDTYPE storage TYPE rom IMPL auto LATENCY 4 ALLOW_PRAGMA 1}
      {MODELNAME run_model_conv_bn_act_pool_3_ConvW3_ROM_AUTO_3R RTLNAME run_model_conv_bn_act_pool_3_ConvW3_ROM_AUTO_3R BINDTYPE storage TYPE rom IMPL auto LATENCY 4 ALLOW_PRAGMA 1}
    }
  }
  {SRCNAME run_all_slices_unrolled_Pipeline_Loop_BN4 MODELNAME run_all_slices_unrolled_Pipeline_Loop_BN4 RTLNAME run_model_run_all_slices_unrolled_Pipeline_Loop_BN4}
  {SRCNAME run_all_slices_unrolled_Pipeline_MergedLoop3 MODELNAME run_all_slices_unrolled_Pipeline_MergedLoop3 RTLNAME run_model_run_all_slices_unrolled_Pipeline_MergedLoop3}
  {SRCNAME conv_bn_act_pool.4_Pipeline_BNParamsLoop MODELNAME conv_bn_act_pool_4_Pipeline_BNParamsLoop RTLNAME run_model_conv_bn_act_pool_4_Pipeline_BNParamsLoop}
  {SRCNAME conv_bn_act_pool.4_Pipeline_Loop3_2Big MODELNAME conv_bn_act_pool_4_Pipeline_Loop3_2Big RTLNAME run_model_conv_bn_act_pool_4_Pipeline_Loop3_2Big}
  {SRCNAME conv_bn_act_pool.4 MODELNAME conv_bn_act_pool_4 RTLNAME run_model_conv_bn_act_pool_4
    SUBMODULES {
      {MODELNAME run_model_conv_bn_act_pool_4_X_slice57_ROM_AUTO_3R RTLNAME run_model_conv_bn_act_pool_4_X_slice57_ROM_AUTO_3R BINDTYPE storage TYPE rom IMPL auto LATENCY 4 ALLOW_PRAGMA 1}
      {MODELNAME run_model_conv_bn_act_pool_4_ConvW4_ROM_AUTO_3R RTLNAME run_model_conv_bn_act_pool_4_ConvW4_ROM_AUTO_3R BINDTYPE storage TYPE rom IMPL auto LATENCY 4 ALLOW_PRAGMA 1}
    }
  }
  {SRCNAME run_all_slices_unrolled_Pipeline_Loop_BN5 MODELNAME run_all_slices_unrolled_Pipeline_Loop_BN5 RTLNAME run_model_run_all_slices_unrolled_Pipeline_Loop_BN5}
  {SRCNAME run_all_slices_unrolled_Pipeline_MergedLoop4 MODELNAME run_all_slices_unrolled_Pipeline_MergedLoop4 RTLNAME run_model_run_all_slices_unrolled_Pipeline_MergedLoop4}
  {SRCNAME run_all_slices_unrolled MODELNAME run_all_slices_unrolled RTLNAME run_model_run_all_slices_unrolled
    SUBMODULES {
      {MODELNAME run_model_fsqrt_32ns_32ns_32_10_no_dsp_1 RTLNAME run_model_fsqrt_32ns_32ns_32_10_no_dsp_1 BINDTYPE op TYPE fsqrt IMPL fabric LATENCY 9 ALLOW_PRAGMA 1}
      {MODELNAME run_model_fsub_32ns_32ns_32_4_full_dsp_1 RTLNAME run_model_fsub_32ns_32ns_32_4_full_dsp_1 BINDTYPE op TYPE fsub IMPL fulldsp LATENCY 3 ALLOW_PRAGMA 1}
      {MODELNAME run_model_fmul_32ns_32ns_32_3_max_dsp_1 RTLNAME run_model_fmul_32ns_32ns_32_3_max_dsp_1 BINDTYPE op TYPE fmul IMPL maxdsp LATENCY 2 ALLOW_PRAGMA 1}
      {MODELNAME run_model_hdiv_16ns_16ns_16_6_no_dsp_1 RTLNAME run_model_hdiv_16ns_16ns_16_6_no_dsp_1 BINDTYPE op TYPE hdiv IMPL fabric LATENCY 5 ALLOW_PRAGMA 1}
      {MODELNAME run_model_run_all_slices_unrolled_h_slice_RAM_AUTO_1R1W RTLNAME run_model_run_all_slices_unrolled_h_slice_RAM_AUTO_1R1W BINDTYPE storage TYPE ram IMPL auto LATENCY 2 ALLOW_PRAGMA 1}
      {MODELNAME run_model_run_all_slices_unrolled_U_slice_RAM_AUTO_3R2W RTLNAME run_model_run_all_slices_unrolled_U_slice_RAM_AUTO_3R2W BINDTYPE storage TYPE ram IMPL auto LATENCY 4 ALLOW_PRAGMA 1}
      {MODELNAME run_model_run_all_slices_unrolled_LSTM_W_ifog0_ROM_AUTO_3R RTLNAME run_model_run_all_slices_unrolled_LSTM_W_ifog0_ROM_AUTO_3R BINDTYPE storage TYPE rom IMPL auto LATENCY 4 ALLOW_PRAGMA 1}
      {MODELNAME run_model_run_all_slices_unrolled_LSTM_R_ifog0_ROM_AUTO_3R RTLNAME run_model_run_all_slices_unrolled_LSTM_R_ifog0_ROM_AUTO_3R BINDTYPE storage TYPE rom IMPL auto LATENCY 4 ALLOW_PRAGMA 1}
      {MODELNAME run_model_run_all_slices_unrolled_LSTM_W_ifog1_ROM_AUTO_3R RTLNAME run_model_run_all_slices_unrolled_LSTM_W_ifog1_ROM_AUTO_3R BINDTYPE storage TYPE rom IMPL auto LATENCY 4 ALLOW_PRAGMA 1}
      {MODELNAME run_model_run_all_slices_unrolled_LSTM_R_ifog1_ROM_AUTO_3R RTLNAME run_model_run_all_slices_unrolled_LSTM_R_ifog1_ROM_AUTO_3R BINDTYPE storage TYPE rom IMPL auto LATENCY 4 ALLOW_PRAGMA 1}
      {MODELNAME run_model_run_all_slices_unrolled_LSTM_W_ifog2_ROM_AUTO_3R RTLNAME run_model_run_all_slices_unrolled_LSTM_W_ifog2_ROM_AUTO_3R BINDTYPE storage TYPE rom IMPL auto LATENCY 4 ALLOW_PRAGMA 1}
      {MODELNAME run_model_run_all_slices_unrolled_LSTM_R_ifog2_ROM_AUTO_3R RTLNAME run_model_run_all_slices_unrolled_LSTM_R_ifog2_ROM_AUTO_3R BINDTYPE storage TYPE rom IMPL auto LATENCY 4 ALLOW_PRAGMA 1}
      {MODELNAME run_model_run_all_slices_unrolled_LSTM_W_ifog3_ROM_AUTO_3R RTLNAME run_model_run_all_slices_unrolled_LSTM_W_ifog3_ROM_AUTO_3R BINDTYPE storage TYPE rom IMPL auto LATENCY 4 ALLOW_PRAGMA 1}
      {MODELNAME run_model_run_all_slices_unrolled_LSTM_R_ifog3_ROM_AUTO_3R RTLNAME run_model_run_all_slices_unrolled_LSTM_R_ifog3_ROM_AUTO_3R BINDTYPE storage TYPE rom IMPL auto LATENCY 4 ALLOW_PRAGMA 1}
      {MODELNAME run_model_run_all_slices_unrolled_LSTM_W_ifog4_ROM_AUTO_3R RTLNAME run_model_run_all_slices_unrolled_LSTM_W_ifog4_ROM_AUTO_3R BINDTYPE storage TYPE rom IMPL auto LATENCY 4 ALLOW_PRAGMA 1}
      {MODELNAME run_model_run_all_slices_unrolled_LSTM_R_ifog4_ROM_AUTO_3R RTLNAME run_model_run_all_slices_unrolled_LSTM_R_ifog4_ROM_AUTO_3R BINDTYPE storage TYPE rom IMPL auto LATENCY 4 ALLOW_PRAGMA 1}
    }
  }
  {SRCNAME run_model_Pipeline_Loop_BN MODELNAME run_model_Pipeline_Loop_BN RTLNAME run_model_run_model_Pipeline_Loop_BN
    SUBMODULES {
      {MODELNAME run_model_run_model_Pipeline_Loop_BN_fc_0_bn_var_ROM_AUTO_3R RTLNAME run_model_run_model_Pipeline_Loop_BN_fc_0_bn_var_ROM_AUTO_3R BINDTYPE storage TYPE rom IMPL auto LATENCY 4 ALLOW_PRAGMA 1}
    }
  }
  {SRCNAME run_model_Pipeline_ReLULoop1 MODELNAME run_model_Pipeline_ReLULoop1 RTLNAME run_model_run_model_Pipeline_ReLULoop1}
  {SRCNAME run_model_Pipeline_Loop_BN1 MODELNAME run_model_Pipeline_Loop_BN1 RTLNAME run_model_run_model_Pipeline_Loop_BN1}
  {SRCNAME run_model_Pipeline_ReLULoop2 MODELNAME run_model_Pipeline_ReLULoop2 RTLNAME run_model_run_model_Pipeline_ReLULoop2}
  {SRCNAME run_model MODELNAME run_model RTLNAME run_model IS_TOP 1
    SUBMODULES {
      {MODELNAME run_model_fc_0_W_ROM_AUTO_3R RTLNAME run_model_fc_0_W_ROM_AUTO_3R BINDTYPE storage TYPE rom IMPL auto LATENCY 4 ALLOW_PRAGMA 1}
      {MODELNAME run_model_fc_1_W_ROM_AUTO_3R RTLNAME run_model_fc_1_W_ROM_AUTO_3R BINDTYPE storage TYPE rom IMPL auto LATENCY 4 ALLOW_PRAGMA 1}
      {MODELNAME run_model_merged_RAM_AUTO_1R1W RTLNAME run_model_merged_RAM_AUTO_1R1W BINDTYPE storage TYPE ram IMPL auto LATENCY 2 ALLOW_PRAGMA 1}
      {MODELNAME run_model_z0_RAM_AUTO_3R2W RTLNAME run_model_z0_RAM_AUTO_3R2W BINDTYPE storage TYPE ram IMPL auto LATENCY 4 ALLOW_PRAGMA 1}
      {MODELNAME run_model_z1_RAM_AUTO_3R2W RTLNAME run_model_z1_RAM_AUTO_3R2W BINDTYPE storage TYPE ram IMPL auto LATENCY 4 ALLOW_PRAGMA 1}
    }
  }
}
