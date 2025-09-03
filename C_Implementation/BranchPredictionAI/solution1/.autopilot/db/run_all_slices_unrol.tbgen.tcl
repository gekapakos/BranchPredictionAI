set moduleName run_all_slices_unrol
set isTopModule 0
set isTaskLevelControl 1
set isCombinational 0
set isDatapathOnly 0
set isFreeRunPipelineModule 0
set isPipelined 0
set pipeline_type none
set FunctionProtocol ap_ctrl_hs
set isOneStateSeq 0
set ProfileFlag 0
set StallSigGenFlag 0
set isEnableWaveformDebug 1
set C_modelName {run_all_slices_unrol}
set C_modelType { void 0 }
set C_modelArgList {
	{ merged float 32 regular {array 32 { 2 3 } 1 1 }  }
}
set C_modelArgMapList {[ 
	{ "Name" : "merged", "interface" : "memory", "bitwidth" : 32, "direction" : "READWRITE"} ]}
# RTL Port declarations: 
set portNum 11
set portList { 
	{ ap_clk sc_in sc_logic 1 clock -1 } 
	{ ap_rst sc_in sc_logic 1 reset -1 active_high_sync } 
	{ ap_start sc_in sc_logic 1 start -1 } 
	{ ap_done sc_out sc_logic 1 predone -1 } 
	{ ap_idle sc_out sc_logic 1 done -1 } 
	{ ap_ready sc_out sc_logic 1 ready -1 } 
	{ merged_address0 sc_out sc_lv 5 signal 0 } 
	{ merged_ce0 sc_out sc_logic 1 signal 0 } 
	{ merged_we0 sc_out sc_logic 1 signal 0 } 
	{ merged_d0 sc_out sc_lv 32 signal 0 } 
	{ merged_q0 sc_in sc_lv 32 signal 0 } 
}
set NewPortList {[ 
	{ "name": "ap_clk", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "clock", "bundle":{"name": "ap_clk", "role": "default" }} , 
 	{ "name": "ap_rst", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "reset", "bundle":{"name": "ap_rst", "role": "default" }} , 
 	{ "name": "ap_start", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "start", "bundle":{"name": "ap_start", "role": "default" }} , 
 	{ "name": "ap_done", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "predone", "bundle":{"name": "ap_done", "role": "default" }} , 
 	{ "name": "ap_idle", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "done", "bundle":{"name": "ap_idle", "role": "default" }} , 
 	{ "name": "ap_ready", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "ready", "bundle":{"name": "ap_ready", "role": "default" }} , 
 	{ "name": "merged_address0", "direction": "out", "datatype": "sc_lv", "bitwidth":5, "type": "signal", "bundle":{"name": "merged", "role": "address0" }} , 
 	{ "name": "merged_ce0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "merged", "role": "ce0" }} , 
 	{ "name": "merged_we0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "merged", "role": "we0" }} , 
 	{ "name": "merged_d0", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "merged", "role": "d0" }} , 
 	{ "name": "merged_q0", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "merged", "role": "q0" }}  ]}

set RtlHierarchyInfo {[
	{"ID" : "0", "Level" : "0", "Path" : "`AUTOTB_DUT_INST", "Parent" : "", "Child" : ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21", "22", "23", "24", "25", "26", "27", "28", "29", "30", "31", "32", "33", "34", "35", "36", "37", "38", "39", "40", "41", "42", "43", "44", "45", "46", "47", "48", "49", "50", "51", "52", "53", "54", "55", "56", "57", "58", "59", "60", "61", "62", "63", "64", "65", "66", "67", "68", "69", "97", "114", "124", "125", "126", "127"],
		"CDFG" : "run_all_slices_unrol",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "0", "ap_idle" : "1",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "20767760", "EstimateLatencyMax" : "541751140",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"WaitState" : [
			{"State" : "ap_ST_fsm_state9", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_lstm_forward_unidir_fu_929"},
			{"State" : "ap_ST_fsm_state71", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_lstm_forward_unidir_fu_929"},
			{"State" : "ap_ST_fsm_state133", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_lstm_forward_unidir_fu_929"},
			{"State" : "ap_ST_fsm_state195", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_lstm_forward_unidir_fu_929"},
			{"State" : "ap_ST_fsm_state257", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_lstm_forward_unidir_fu_929"},
			{"State" : "ap_ST_fsm_state58", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_generic_tanh_float_s_fu_963"},
			{"State" : "ap_ST_fsm_state120", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_generic_tanh_float_s_fu_963"},
			{"State" : "ap_ST_fsm_state182", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_generic_tanh_float_s_fu_963"},
			{"State" : "ap_ST_fsm_state244", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_generic_tanh_float_s_fu_963"},
			{"State" : "ap_ST_fsm_state306", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_generic_tanh_float_s_fu_963"},
			{"State" : "ap_ST_fsm_state7", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_conv_bn_act_pool_fu_974"},
			{"State" : "ap_ST_fsm_state69", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_conv_bn_act_pool_fu_974"},
			{"State" : "ap_ST_fsm_state131", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_conv_bn_act_pool_fu_974"},
			{"State" : "ap_ST_fsm_state193", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_conv_bn_act_pool_fu_974"},
			{"State" : "ap_ST_fsm_state255", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_conv_bn_act_pool_fu_974"}],
		"Port" : [
			{"Name" : "merged", "Type" : "Memory", "Direction" : "IO"},
			{"Name" : "tokens0", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "Emb0", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "X_slice", "Type" : "Memory", "Direction" : "IO",
				"SubConnect" : [
					{"ID" : "114", "SubInstance" : "grp_conv_bn_act_pool_fu_974", "Port" : "X_slice"}]},
			{"Name" : "B_arg_index", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "Y", "Type" : "Memory", "Direction" : "IO",
				"SubConnect" : [
					{"ID" : "114", "SubInstance" : "grp_conv_bn_act_pool_fu_974", "Port" : "Y"}]},
			{"Name" : "ConvW0", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "114", "SubInstance" : "grp_conv_bn_act_pool_fu_974", "Port" : "W"}]},
			{"Name" : "ConvB0", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "114", "SubInstance" : "grp_conv_bn_act_pool_fu_974", "Port" : "B"}]},
			{"Name" : "BN1_gamma0", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "114", "SubInstance" : "grp_conv_bn_act_pool_fu_974", "Port" : "gamma"}]},
			{"Name" : "BN1_beta0", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "114", "SubInstance" : "grp_conv_bn_act_pool_fu_974", "Port" : "beta"}]},
			{"Name" : "BN1_mean0", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "114", "SubInstance" : "grp_conv_bn_act_pool_fu_974", "Port" : "mean"}]},
			{"Name" : "BN1_var0", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "114", "SubInstance" : "grp_conv_bn_act_pool_fu_974", "Port" : "var"}]},
			{"Name" : "U_slice", "Type" : "Memory", "Direction" : "IO",
				"SubConnect" : [
					{"ID" : "114", "SubInstance" : "grp_conv_bn_act_pool_fu_974", "Port" : "U"},
					{"ID" : "69", "SubInstance" : "grp_lstm_forward_unidir_fu_929", "Port" : "U_slice"}]},
			{"Name" : "c_slice", "Type" : "Memory", "Direction" : "IO",
				"SubConnect" : [
					{"ID" : "69", "SubInstance" : "grp_lstm_forward_unidir_fu_929", "Port" : "c_slice"}]},
			{"Name" : "table_exp_Z1_array_s", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "97", "SubInstance" : "grp_generic_tanh_float_s_fu_963", "Port" : "table_exp_Z1_array_s"},
					{"ID" : "69", "SubInstance" : "grp_lstm_forward_unidir_fu_929", "Port" : "table_exp_Z1_array_s"}]},
			{"Name" : "table_f_Z3_array_V", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "97", "SubInstance" : "grp_generic_tanh_float_s_fu_963", "Port" : "table_f_Z3_array_V"},
					{"ID" : "69", "SubInstance" : "grp_lstm_forward_unidir_fu_929", "Port" : "table_f_Z3_array_V"}]},
			{"Name" : "table_f_Z2_array_V", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "97", "SubInstance" : "grp_generic_tanh_float_s_fu_963", "Port" : "table_f_Z2_array_V"},
					{"ID" : "69", "SubInstance" : "grp_lstm_forward_unidir_fu_929", "Port" : "table_f_Z2_array_V"}]},
			{"Name" : "LSTM_W_ifog0", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "69", "SubInstance" : "grp_lstm_forward_unidir_fu_929", "Port" : "W_ifog"}]},
			{"Name" : "LSTM_R_ifog0", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "69", "SubInstance" : "grp_lstm_forward_unidir_fu_929", "Port" : "R_ifog"}]},
			{"Name" : "LSTM_b_ifog0", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "69", "SubInstance" : "grp_lstm_forward_unidir_fu_929", "Port" : "b_ifog"}]},
			{"Name" : "h_slice", "Type" : "Memory", "Direction" : "IO",
				"SubConnect" : [
					{"ID" : "69", "SubInstance" : "grp_lstm_forward_unidir_fu_929", "Port" : "h_last"}]},
			{"Name" : "BN2_var0", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "BN2_gamma0", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "tokens1", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "Emb1", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "ConvW1", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "114", "SubInstance" : "grp_conv_bn_act_pool_fu_974", "Port" : "W"}]},
			{"Name" : "ConvB1", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "114", "SubInstance" : "grp_conv_bn_act_pool_fu_974", "Port" : "B"}]},
			{"Name" : "BN1_gamma1", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "114", "SubInstance" : "grp_conv_bn_act_pool_fu_974", "Port" : "gamma"}]},
			{"Name" : "BN1_beta1", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "114", "SubInstance" : "grp_conv_bn_act_pool_fu_974", "Port" : "beta"}]},
			{"Name" : "BN1_mean1", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "114", "SubInstance" : "grp_conv_bn_act_pool_fu_974", "Port" : "mean"}]},
			{"Name" : "BN1_var1", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "114", "SubInstance" : "grp_conv_bn_act_pool_fu_974", "Port" : "var"}]},
			{"Name" : "LSTM_W_ifog1", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "69", "SubInstance" : "grp_lstm_forward_unidir_fu_929", "Port" : "W_ifog"}]},
			{"Name" : "LSTM_R_ifog1", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "69", "SubInstance" : "grp_lstm_forward_unidir_fu_929", "Port" : "R_ifog"}]},
			{"Name" : "LSTM_b_ifog1", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "69", "SubInstance" : "grp_lstm_forward_unidir_fu_929", "Port" : "b_ifog"}]},
			{"Name" : "BN2_var1", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "BN2_gamma1", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "tokens2", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "Emb2", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "ConvW2", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "114", "SubInstance" : "grp_conv_bn_act_pool_fu_974", "Port" : "W"}]},
			{"Name" : "ConvB2", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "114", "SubInstance" : "grp_conv_bn_act_pool_fu_974", "Port" : "B"}]},
			{"Name" : "BN1_gamma2", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "114", "SubInstance" : "grp_conv_bn_act_pool_fu_974", "Port" : "gamma"}]},
			{"Name" : "BN1_beta2", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "114", "SubInstance" : "grp_conv_bn_act_pool_fu_974", "Port" : "beta"}]},
			{"Name" : "BN1_mean2", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "114", "SubInstance" : "grp_conv_bn_act_pool_fu_974", "Port" : "mean"}]},
			{"Name" : "BN1_var2", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "114", "SubInstance" : "grp_conv_bn_act_pool_fu_974", "Port" : "var"}]},
			{"Name" : "LSTM_W_ifog2", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "69", "SubInstance" : "grp_lstm_forward_unidir_fu_929", "Port" : "W_ifog"}]},
			{"Name" : "LSTM_R_ifog2", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "69", "SubInstance" : "grp_lstm_forward_unidir_fu_929", "Port" : "R_ifog"}]},
			{"Name" : "LSTM_b_ifog2", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "69", "SubInstance" : "grp_lstm_forward_unidir_fu_929", "Port" : "b_ifog"}]},
			{"Name" : "BN2_var2", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "BN2_gamma2", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "tokens3", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "Emb3", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "ConvW3", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "114", "SubInstance" : "grp_conv_bn_act_pool_fu_974", "Port" : "W"}]},
			{"Name" : "ConvB3", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "114", "SubInstance" : "grp_conv_bn_act_pool_fu_974", "Port" : "B"}]},
			{"Name" : "BN1_gamma3", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "114", "SubInstance" : "grp_conv_bn_act_pool_fu_974", "Port" : "gamma"}]},
			{"Name" : "BN1_beta3", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "114", "SubInstance" : "grp_conv_bn_act_pool_fu_974", "Port" : "beta"}]},
			{"Name" : "BN1_mean3", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "114", "SubInstance" : "grp_conv_bn_act_pool_fu_974", "Port" : "mean"}]},
			{"Name" : "BN1_var3", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "114", "SubInstance" : "grp_conv_bn_act_pool_fu_974", "Port" : "var"}]},
			{"Name" : "LSTM_W_ifog3", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "69", "SubInstance" : "grp_lstm_forward_unidir_fu_929", "Port" : "W_ifog"}]},
			{"Name" : "LSTM_R_ifog3", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "69", "SubInstance" : "grp_lstm_forward_unidir_fu_929", "Port" : "R_ifog"}]},
			{"Name" : "LSTM_b_ifog3", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "69", "SubInstance" : "grp_lstm_forward_unidir_fu_929", "Port" : "b_ifog"}]},
			{"Name" : "BN2_var3", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "BN2_gamma3", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "tokens4", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "Emb4", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "ConvW4", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "114", "SubInstance" : "grp_conv_bn_act_pool_fu_974", "Port" : "W"}]},
			{"Name" : "ConvB4", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "114", "SubInstance" : "grp_conv_bn_act_pool_fu_974", "Port" : "B"}]},
			{"Name" : "BN1_gamma4", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "114", "SubInstance" : "grp_conv_bn_act_pool_fu_974", "Port" : "gamma"}]},
			{"Name" : "BN1_beta4", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "114", "SubInstance" : "grp_conv_bn_act_pool_fu_974", "Port" : "beta"}]},
			{"Name" : "BN1_mean4", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "114", "SubInstance" : "grp_conv_bn_act_pool_fu_974", "Port" : "mean"}]},
			{"Name" : "BN1_var4", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "114", "SubInstance" : "grp_conv_bn_act_pool_fu_974", "Port" : "var"}]},
			{"Name" : "LSTM_W_ifog4", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "69", "SubInstance" : "grp_lstm_forward_unidir_fu_929", "Port" : "W_ifog"}]},
			{"Name" : "LSTM_R_ifog4", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "69", "SubInstance" : "grp_lstm_forward_unidir_fu_929", "Port" : "R_ifog"}]},
			{"Name" : "LSTM_b_ifog4", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "69", "SubInstance" : "grp_lstm_forward_unidir_fu_929", "Port" : "b_ifog"}]},
			{"Name" : "BN2_var4", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "BN2_gamma4", "Type" : "Memory", "Direction" : "I"}]},
	{"ID" : "1", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.tokens0_U", "Parent" : "0"},
	{"ID" : "2", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.Emb0_U", "Parent" : "0"},
	{"ID" : "3", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.X_slice_U", "Parent" : "0"},
	{"ID" : "4", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.ConvW0_U", "Parent" : "0"},
	{"ID" : "5", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.ConvB0_U", "Parent" : "0"},
	{"ID" : "6", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.BN1_gamma0_U", "Parent" : "0"},
	{"ID" : "7", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.BN1_beta0_U", "Parent" : "0"},
	{"ID" : "8", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.BN1_mean0_U", "Parent" : "0"},
	{"ID" : "9", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.BN1_var0_U", "Parent" : "0"},
	{"ID" : "10", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.U_slice_U", "Parent" : "0"},
	{"ID" : "11", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.LSTM_W_ifog0_U", "Parent" : "0"},
	{"ID" : "12", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.LSTM_R_ifog0_U", "Parent" : "0"},
	{"ID" : "13", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.LSTM_b_ifog0_U", "Parent" : "0"},
	{"ID" : "14", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.h_slice_U", "Parent" : "0"},
	{"ID" : "15", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.BN2_var0_U", "Parent" : "0"},
	{"ID" : "16", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.BN2_gamma0_U", "Parent" : "0"},
	{"ID" : "17", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.tokens1_U", "Parent" : "0"},
	{"ID" : "18", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.Emb1_U", "Parent" : "0"},
	{"ID" : "19", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.ConvW1_U", "Parent" : "0"},
	{"ID" : "20", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.ConvB1_U", "Parent" : "0"},
	{"ID" : "21", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.BN1_gamma1_U", "Parent" : "0"},
	{"ID" : "22", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.BN1_beta1_U", "Parent" : "0"},
	{"ID" : "23", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.BN1_mean1_U", "Parent" : "0"},
	{"ID" : "24", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.BN1_var1_U", "Parent" : "0"},
	{"ID" : "25", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.LSTM_W_ifog1_U", "Parent" : "0"},
	{"ID" : "26", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.LSTM_R_ifog1_U", "Parent" : "0"},
	{"ID" : "27", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.LSTM_b_ifog1_U", "Parent" : "0"},
	{"ID" : "28", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.BN2_var1_U", "Parent" : "0"},
	{"ID" : "29", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.BN2_gamma1_U", "Parent" : "0"},
	{"ID" : "30", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.tokens2_U", "Parent" : "0"},
	{"ID" : "31", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.Emb2_U", "Parent" : "0"},
	{"ID" : "32", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.ConvW2_U", "Parent" : "0"},
	{"ID" : "33", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.ConvB2_U", "Parent" : "0"},
	{"ID" : "34", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.BN1_gamma2_U", "Parent" : "0"},
	{"ID" : "35", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.BN1_beta2_U", "Parent" : "0"},
	{"ID" : "36", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.BN1_mean2_U", "Parent" : "0"},
	{"ID" : "37", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.BN1_var2_U", "Parent" : "0"},
	{"ID" : "38", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.LSTM_W_ifog2_U", "Parent" : "0"},
	{"ID" : "39", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.LSTM_R_ifog2_U", "Parent" : "0"},
	{"ID" : "40", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.LSTM_b_ifog2_U", "Parent" : "0"},
	{"ID" : "41", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.BN2_var2_U", "Parent" : "0"},
	{"ID" : "42", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.BN2_gamma2_U", "Parent" : "0"},
	{"ID" : "43", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.tokens3_U", "Parent" : "0"},
	{"ID" : "44", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.Emb3_U", "Parent" : "0"},
	{"ID" : "45", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.ConvW3_U", "Parent" : "0"},
	{"ID" : "46", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.ConvB3_U", "Parent" : "0"},
	{"ID" : "47", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.BN1_gamma3_U", "Parent" : "0"},
	{"ID" : "48", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.BN1_beta3_U", "Parent" : "0"},
	{"ID" : "49", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.BN1_mean3_U", "Parent" : "0"},
	{"ID" : "50", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.BN1_var3_U", "Parent" : "0"},
	{"ID" : "51", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.LSTM_W_ifog3_U", "Parent" : "0"},
	{"ID" : "52", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.LSTM_R_ifog3_U", "Parent" : "0"},
	{"ID" : "53", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.LSTM_b_ifog3_U", "Parent" : "0"},
	{"ID" : "54", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.BN2_var3_U", "Parent" : "0"},
	{"ID" : "55", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.BN2_gamma3_U", "Parent" : "0"},
	{"ID" : "56", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.tokens4_U", "Parent" : "0"},
	{"ID" : "57", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.Emb4_U", "Parent" : "0"},
	{"ID" : "58", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.ConvW4_U", "Parent" : "0"},
	{"ID" : "59", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.ConvB4_U", "Parent" : "0"},
	{"ID" : "60", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.BN1_gamma4_U", "Parent" : "0"},
	{"ID" : "61", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.BN1_beta4_U", "Parent" : "0"},
	{"ID" : "62", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.BN1_mean4_U", "Parent" : "0"},
	{"ID" : "63", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.BN1_var4_U", "Parent" : "0"},
	{"ID" : "64", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.LSTM_W_ifog4_U", "Parent" : "0"},
	{"ID" : "65", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.LSTM_R_ifog4_U", "Parent" : "0"},
	{"ID" : "66", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.LSTM_b_ifog4_U", "Parent" : "0"},
	{"ID" : "67", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.BN2_var4_U", "Parent" : "0"},
	{"ID" : "68", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.BN2_gamma4_U", "Parent" : "0"},
	{"ID" : "69", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.grp_lstm_forward_unidir_fu_929", "Parent" : "0", "Child" : ["70", "71", "72", "89", "90", "91", "92", "93", "94", "95", "96"],
		"CDFG" : "lstm_forward_unidir",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "0", "ap_idle" : "1",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "1203934", "EstimateLatencyMax" : "1250014",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"WaitState" : [
			{"State" : "ap_ST_fsm_state66", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_generic_tanh_float_s_fu_325"},
			{"State" : "ap_ST_fsm_state76", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_generic_tanh_float_s_fu_325"}],
		"Port" : [
			{"Name" : "W_ifog", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "R_ifog", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "b_ifog", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "h_last", "Type" : "Memory", "Direction" : "IO"},
			{"Name" : "c_slice", "Type" : "Memory", "Direction" : "IO"},
			{"Name" : "U_slice", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "table_exp_Z1_array_s", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "72", "SubInstance" : "grp_generic_tanh_float_s_fu_325", "Port" : "table_exp_Z1_array_s"}]},
			{"Name" : "table_f_Z3_array_V", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "72", "SubInstance" : "grp_generic_tanh_float_s_fu_325", "Port" : "table_f_Z3_array_V"}]},
			{"Name" : "table_f_Z2_array_V", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "72", "SubInstance" : "grp_generic_tanh_float_s_fu_325", "Port" : "table_f_Z2_array_V"}]}]},
	{"ID" : "70", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_lstm_forward_unidir_fu_929.c_slice_U", "Parent" : "69"},
	{"ID" : "71", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_lstm_forward_unidir_fu_929.z_U", "Parent" : "69"},
	{"ID" : "72", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_lstm_forward_unidir_fu_929.grp_generic_tanh_float_s_fu_325", "Parent" : "69", "Child" : ["73", "82", "83", "84", "85", "86", "87", "88"],
		"CDFG" : "generic_tanh_float_s",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "0", "ap_idle" : "1",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "1", "EstimateLatencyMax" : "61",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"Port" : [
			{"Name" : "t_in", "Type" : "None", "Direction" : "I"},
			{"Name" : "table_exp_Z1_array_s", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "73", "SubInstance" : "grp_exp_generic_double_s_fu_89", "Port" : "table_exp_Z1_array_s"}]},
			{"Name" : "table_f_Z3_array_V", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "73", "SubInstance" : "grp_exp_generic_double_s_fu_89", "Port" : "table_f_Z3_array_V"}]},
			{"Name" : "table_f_Z2_array_V", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "73", "SubInstance" : "grp_exp_generic_double_s_fu_89", "Port" : "table_f_Z2_array_V"}]}]},
	{"ID" : "73", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_lstm_forward_unidir_fu_929.grp_generic_tanh_float_s_fu_325.grp_exp_generic_double_s_fu_89", "Parent" : "72", "Child" : ["74", "75", "76", "77", "78", "79", "80", "81"],
		"CDFG" : "exp_generic_double_s",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "0", "ap_idle" : "1",
		"Pipeline" : "Aligned", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "1",
		"VariableLatency" : "0", "ExactLatency" : "19", "EstimateLatencyMin" : "19", "EstimateLatencyMax" : "19",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"Port" : [
			{"Name" : "x", "Type" : "None", "Direction" : "I"},
			{"Name" : "table_exp_Z1_array_s", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "table_f_Z3_array_V", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "table_f_Z2_array_V", "Type" : "Memory", "Direction" : "I"}]},
	{"ID" : "74", "Level" : "4", "Path" : "`AUTOTB_DUT_INST.grp_lstm_forward_unidir_fu_929.grp_generic_tanh_float_s_fu_325.grp_exp_generic_double_s_fu_89.table_exp_Z1_array_s_U", "Parent" : "73"},
	{"ID" : "75", "Level" : "4", "Path" : "`AUTOTB_DUT_INST.grp_lstm_forward_unidir_fu_929.grp_generic_tanh_float_s_fu_325.grp_exp_generic_double_s_fu_89.table_f_Z3_array_V_U", "Parent" : "73"},
	{"ID" : "76", "Level" : "4", "Path" : "`AUTOTB_DUT_INST.grp_lstm_forward_unidir_fu_929.grp_generic_tanh_float_s_fu_325.grp_exp_generic_double_s_fu_89.table_f_Z2_array_V_U", "Parent" : "73"},
	{"ID" : "77", "Level" : "4", "Path" : "`AUTOTB_DUT_INST.grp_lstm_forward_unidir_fu_929.grp_generic_tanh_float_s_fu_325.grp_exp_generic_double_s_fu_89.main_mul_72ns_13s_84_5_1_U29", "Parent" : "73"},
	{"ID" : "78", "Level" : "4", "Path" : "`AUTOTB_DUT_INST.grp_lstm_forward_unidir_fu_929.grp_generic_tanh_float_s_fu_325.grp_exp_generic_double_s_fu_89.main_mul_36ns_43ns_79_2_1_U30", "Parent" : "73"},
	{"ID" : "79", "Level" : "4", "Path" : "`AUTOTB_DUT_INST.grp_lstm_forward_unidir_fu_929.grp_generic_tanh_float_s_fu_325.grp_exp_generic_double_s_fu_89.main_mul_44ns_49ns_93_2_1_U31", "Parent" : "73"},
	{"ID" : "80", "Level" : "4", "Path" : "`AUTOTB_DUT_INST.grp_lstm_forward_unidir_fu_929.grp_generic_tanh_float_s_fu_325.grp_exp_generic_double_s_fu_89.main_mul_50ns_50ns_100_2_1_U32", "Parent" : "73"},
	{"ID" : "81", "Level" : "4", "Path" : "`AUTOTB_DUT_INST.grp_lstm_forward_unidir_fu_929.grp_generic_tanh_float_s_fu_325.grp_exp_generic_double_s_fu_89.main_mac_muladd_16ns_16s_19s_31_1_1_U33", "Parent" : "73"},
	{"ID" : "82", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_lstm_forward_unidir_fu_929.grp_generic_tanh_float_s_fu_325.main_faddfsub_32ns_32ns_32_5_full_dsp_1_U43", "Parent" : "72"},
	{"ID" : "83", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_lstm_forward_unidir_fu_929.grp_generic_tanh_float_s_fu_325.main_fmul_32ns_32ns_32_4_max_dsp_1_U44", "Parent" : "72"},
	{"ID" : "84", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_lstm_forward_unidir_fu_929.grp_generic_tanh_float_s_fu_325.main_fdiv_32ns_32ns_32_16_1_U45", "Parent" : "72"},
	{"ID" : "85", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_lstm_forward_unidir_fu_929.grp_generic_tanh_float_s_fu_325.main_fptrunc_64ns_32_2_1_U46", "Parent" : "72"},
	{"ID" : "86", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_lstm_forward_unidir_fu_929.grp_generic_tanh_float_s_fu_325.main_fpext_32ns_64_2_1_U47", "Parent" : "72"},
	{"ID" : "87", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_lstm_forward_unidir_fu_929.grp_generic_tanh_float_s_fu_325.main_fcmp_32ns_32ns_1_2_1_U48", "Parent" : "72"},
	{"ID" : "88", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_lstm_forward_unidir_fu_929.grp_generic_tanh_float_s_fu_325.main_dadd_64ns_64ns_64_5_full_dsp_1_U49", "Parent" : "72"},
	{"ID" : "89", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_lstm_forward_unidir_fu_929.main_fadd_32ns_32ns_32_5_full_dsp_1_U54", "Parent" : "69"},
	{"ID" : "90", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_lstm_forward_unidir_fu_929.main_fadd_32ns_32ns_32_5_full_dsp_1_U55", "Parent" : "69"},
	{"ID" : "91", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_lstm_forward_unidir_fu_929.main_fmul_32ns_32ns_32_4_max_dsp_1_U56", "Parent" : "69"},
	{"ID" : "92", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_lstm_forward_unidir_fu_929.main_fmul_32ns_32ns_32_4_max_dsp_1_U57", "Parent" : "69"},
	{"ID" : "93", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_lstm_forward_unidir_fu_929.main_fdiv_32ns_32ns_32_16_1_U58", "Parent" : "69"},
	{"ID" : "94", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_lstm_forward_unidir_fu_929.main_fdiv_32ns_32ns_32_16_1_U59", "Parent" : "69"},
	{"ID" : "95", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_lstm_forward_unidir_fu_929.main_fexp_32ns_32ns_32_9_full_dsp_1_U60", "Parent" : "69"},
	{"ID" : "96", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_lstm_forward_unidir_fu_929.main_fexp_32ns_32ns_32_9_full_dsp_1_U61", "Parent" : "69"},
	{"ID" : "97", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.grp_generic_tanh_float_s_fu_963", "Parent" : "0", "Child" : ["98", "107", "108", "109", "110", "111", "112", "113"],
		"CDFG" : "generic_tanh_float_s",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "0", "ap_idle" : "1",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "1", "EstimateLatencyMax" : "61",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"Port" : [
			{"Name" : "t_in", "Type" : "None", "Direction" : "I"},
			{"Name" : "table_exp_Z1_array_s", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "98", "SubInstance" : "grp_exp_generic_double_s_fu_89", "Port" : "table_exp_Z1_array_s"}]},
			{"Name" : "table_f_Z3_array_V", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "98", "SubInstance" : "grp_exp_generic_double_s_fu_89", "Port" : "table_f_Z3_array_V"}]},
			{"Name" : "table_f_Z2_array_V", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "98", "SubInstance" : "grp_exp_generic_double_s_fu_89", "Port" : "table_f_Z2_array_V"}]}]},
	{"ID" : "98", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_generic_tanh_float_s_fu_963.grp_exp_generic_double_s_fu_89", "Parent" : "97", "Child" : ["99", "100", "101", "102", "103", "104", "105", "106"],
		"CDFG" : "exp_generic_double_s",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "0", "ap_idle" : "1",
		"Pipeline" : "Aligned", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "1",
		"VariableLatency" : "0", "ExactLatency" : "19", "EstimateLatencyMin" : "19", "EstimateLatencyMax" : "19",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"Port" : [
			{"Name" : "x", "Type" : "None", "Direction" : "I"},
			{"Name" : "table_exp_Z1_array_s", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "table_f_Z3_array_V", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "table_f_Z2_array_V", "Type" : "Memory", "Direction" : "I"}]},
	{"ID" : "99", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_generic_tanh_float_s_fu_963.grp_exp_generic_double_s_fu_89.table_exp_Z1_array_s_U", "Parent" : "98"},
	{"ID" : "100", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_generic_tanh_float_s_fu_963.grp_exp_generic_double_s_fu_89.table_f_Z3_array_V_U", "Parent" : "98"},
	{"ID" : "101", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_generic_tanh_float_s_fu_963.grp_exp_generic_double_s_fu_89.table_f_Z2_array_V_U", "Parent" : "98"},
	{"ID" : "102", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_generic_tanh_float_s_fu_963.grp_exp_generic_double_s_fu_89.main_mul_72ns_13s_84_5_1_U29", "Parent" : "98"},
	{"ID" : "103", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_generic_tanh_float_s_fu_963.grp_exp_generic_double_s_fu_89.main_mul_36ns_43ns_79_2_1_U30", "Parent" : "98"},
	{"ID" : "104", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_generic_tanh_float_s_fu_963.grp_exp_generic_double_s_fu_89.main_mul_44ns_49ns_93_2_1_U31", "Parent" : "98"},
	{"ID" : "105", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_generic_tanh_float_s_fu_963.grp_exp_generic_double_s_fu_89.main_mul_50ns_50ns_100_2_1_U32", "Parent" : "98"},
	{"ID" : "106", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_generic_tanh_float_s_fu_963.grp_exp_generic_double_s_fu_89.main_mac_muladd_16ns_16s_19s_31_1_1_U33", "Parent" : "98"},
	{"ID" : "107", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_generic_tanh_float_s_fu_963.main_faddfsub_32ns_32ns_32_5_full_dsp_1_U43", "Parent" : "97"},
	{"ID" : "108", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_generic_tanh_float_s_fu_963.main_fmul_32ns_32ns_32_4_max_dsp_1_U44", "Parent" : "97"},
	{"ID" : "109", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_generic_tanh_float_s_fu_963.main_fdiv_32ns_32ns_32_16_1_U45", "Parent" : "97"},
	{"ID" : "110", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_generic_tanh_float_s_fu_963.main_fptrunc_64ns_32_2_1_U46", "Parent" : "97"},
	{"ID" : "111", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_generic_tanh_float_s_fu_963.main_fpext_32ns_64_2_1_U47", "Parent" : "97"},
	{"ID" : "112", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_generic_tanh_float_s_fu_963.main_fcmp_32ns_32ns_1_2_1_U48", "Parent" : "97"},
	{"ID" : "113", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_generic_tanh_float_s_fu_963.main_dadd_64ns_64ns_64_5_full_dsp_1_U49", "Parent" : "97"},
	{"ID" : "114", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.grp_conv_bn_act_pool_fu_974", "Parent" : "0", "Child" : ["115", "116", "117", "118", "119", "120", "121", "122", "123"],
		"CDFG" : "conv_bn_act_pool",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "0", "ap_idle" : "1",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "2932457", "EstimateLatencyMax" : "107081133",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"Port" : [
			{"Name" : "Tlen", "Type" : "None", "Direction" : "I"},
			{"Name" : "W", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "B", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "C", "Type" : "None", "Direction" : "I"},
			{"Name" : "gamma", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "beta", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "mean", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "var", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "Psz", "Type" : "None", "Direction" : "I"},
			{"Name" : "U", "Type" : "Memory", "Direction" : "O"},
			{"Name" : "X_slice", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "Y", "Type" : "Memory", "Direction" : "IO"}]},
	{"ID" : "115", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_bn_act_pool_fu_974.Y_U", "Parent" : "114"},
	{"ID" : "116", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_bn_act_pool_fu_974.main_faddfsub_32ns_32ns_32_5_full_dsp_1_U1", "Parent" : "114"},
	{"ID" : "117", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_bn_act_pool_fu_974.main_fmul_32ns_32ns_32_4_max_dsp_1_U2", "Parent" : "114"},
	{"ID" : "118", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_bn_act_pool_fu_974.main_fdiv_32ns_32ns_32_16_1_U3", "Parent" : "114"},
	{"ID" : "119", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_bn_act_pool_fu_974.main_sitofp_32s_32_6_1_U4", "Parent" : "114"},
	{"ID" : "120", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_bn_act_pool_fu_974.main_fcmp_32ns_32ns_1_2_1_U5", "Parent" : "114"},
	{"ID" : "121", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_bn_act_pool_fu_974.main_fsqrt_32ns_32ns_32_12_1_U6", "Parent" : "114"},
	{"ID" : "122", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_bn_act_pool_fu_974.main_udiv_10ns_7ns_9_14_seq_1_U7", "Parent" : "114"},
	{"ID" : "123", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_bn_act_pool_fu_974.main_mac_muladd_11ns_14ns_10ns_23_1_1_U8", "Parent" : "114"},
	{"ID" : "124", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.main_fadd_32ns_32ns_32_5_full_dsp_1_U71", "Parent" : "0"},
	{"ID" : "125", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.main_fmul_32ns_32ns_32_4_max_dsp_1_U72", "Parent" : "0"},
	{"ID" : "126", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.main_fdiv_32ns_32ns_32_16_1_U73", "Parent" : "0"},
	{"ID" : "127", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.main_fsqrt_32ns_32ns_32_12_1_U74", "Parent" : "0"}]}


set ArgLastReadFirstWriteLatency {
	run_all_slices_unrol {
		merged {Type IO LastRead 33 FirstWrite 1}
		tokens0 {Type I LastRead -1 FirstWrite -1}
		Emb0 {Type I LastRead -1 FirstWrite -1}
		X_slice {Type IO LastRead -1 FirstWrite -1}
		B_arg_index {Type O LastRead -1 FirstWrite -1}
		Y {Type IO LastRead -1 FirstWrite -1}
		ConvW0 {Type I LastRead -1 FirstWrite -1}
		ConvB0 {Type I LastRead -1 FirstWrite -1}
		BN1_gamma0 {Type I LastRead -1 FirstWrite -1}
		BN1_beta0 {Type I LastRead -1 FirstWrite -1}
		BN1_mean0 {Type I LastRead -1 FirstWrite -1}
		BN1_var0 {Type I LastRead -1 FirstWrite -1}
		U_slice {Type IO LastRead -1 FirstWrite -1}
		c_slice {Type IO LastRead -1 FirstWrite -1}
		table_exp_Z1_array_s {Type I LastRead -1 FirstWrite -1}
		table_f_Z3_array_V {Type I LastRead -1 FirstWrite -1}
		table_f_Z2_array_V {Type I LastRead -1 FirstWrite -1}
		LSTM_W_ifog0 {Type I LastRead -1 FirstWrite -1}
		LSTM_R_ifog0 {Type I LastRead -1 FirstWrite -1}
		LSTM_b_ifog0 {Type I LastRead -1 FirstWrite -1}
		h_slice {Type IO LastRead -1 FirstWrite -1}
		BN2_var0 {Type I LastRead -1 FirstWrite -1}
		BN2_gamma0 {Type I LastRead -1 FirstWrite -1}
		tokens1 {Type I LastRead -1 FirstWrite -1}
		Emb1 {Type I LastRead -1 FirstWrite -1}
		ConvW1 {Type I LastRead -1 FirstWrite -1}
		ConvB1 {Type I LastRead -1 FirstWrite -1}
		BN1_gamma1 {Type I LastRead -1 FirstWrite -1}
		BN1_beta1 {Type I LastRead -1 FirstWrite -1}
		BN1_mean1 {Type I LastRead -1 FirstWrite -1}
		BN1_var1 {Type I LastRead -1 FirstWrite -1}
		LSTM_W_ifog1 {Type I LastRead -1 FirstWrite -1}
		LSTM_R_ifog1 {Type I LastRead -1 FirstWrite -1}
		LSTM_b_ifog1 {Type I LastRead -1 FirstWrite -1}
		BN2_var1 {Type I LastRead -1 FirstWrite -1}
		BN2_gamma1 {Type I LastRead -1 FirstWrite -1}
		tokens2 {Type I LastRead -1 FirstWrite -1}
		Emb2 {Type I LastRead -1 FirstWrite -1}
		ConvW2 {Type I LastRead -1 FirstWrite -1}
		ConvB2 {Type I LastRead -1 FirstWrite -1}
		BN1_gamma2 {Type I LastRead -1 FirstWrite -1}
		BN1_beta2 {Type I LastRead -1 FirstWrite -1}
		BN1_mean2 {Type I LastRead -1 FirstWrite -1}
		BN1_var2 {Type I LastRead -1 FirstWrite -1}
		LSTM_W_ifog2 {Type I LastRead -1 FirstWrite -1}
		LSTM_R_ifog2 {Type I LastRead -1 FirstWrite -1}
		LSTM_b_ifog2 {Type I LastRead -1 FirstWrite -1}
		BN2_var2 {Type I LastRead -1 FirstWrite -1}
		BN2_gamma2 {Type I LastRead -1 FirstWrite -1}
		tokens3 {Type I LastRead -1 FirstWrite -1}
		Emb3 {Type I LastRead -1 FirstWrite -1}
		ConvW3 {Type I LastRead -1 FirstWrite -1}
		ConvB3 {Type I LastRead -1 FirstWrite -1}
		BN1_gamma3 {Type I LastRead -1 FirstWrite -1}
		BN1_beta3 {Type I LastRead -1 FirstWrite -1}
		BN1_mean3 {Type I LastRead -1 FirstWrite -1}
		BN1_var3 {Type I LastRead -1 FirstWrite -1}
		LSTM_W_ifog3 {Type I LastRead -1 FirstWrite -1}
		LSTM_R_ifog3 {Type I LastRead -1 FirstWrite -1}
		LSTM_b_ifog3 {Type I LastRead -1 FirstWrite -1}
		BN2_var3 {Type I LastRead -1 FirstWrite -1}
		BN2_gamma3 {Type I LastRead -1 FirstWrite -1}
		tokens4 {Type I LastRead -1 FirstWrite -1}
		Emb4 {Type I LastRead -1 FirstWrite -1}
		ConvW4 {Type I LastRead -1 FirstWrite -1}
		ConvB4 {Type I LastRead -1 FirstWrite -1}
		BN1_gamma4 {Type I LastRead -1 FirstWrite -1}
		BN1_beta4 {Type I LastRead -1 FirstWrite -1}
		BN1_mean4 {Type I LastRead -1 FirstWrite -1}
		BN1_var4 {Type I LastRead -1 FirstWrite -1}
		LSTM_W_ifog4 {Type I LastRead -1 FirstWrite -1}
		LSTM_R_ifog4 {Type I LastRead -1 FirstWrite -1}
		LSTM_b_ifog4 {Type I LastRead -1 FirstWrite -1}
		BN2_var4 {Type I LastRead -1 FirstWrite -1}
		BN2_gamma4 {Type I LastRead -1 FirstWrite -1}}
	lstm_forward_unidir {
		W_ifog {Type I LastRead 6 FirstWrite -1}
		R_ifog {Type I LastRead 7 FirstWrite -1}
		b_ifog {Type I LastRead 3 FirstWrite -1}
		h_last {Type IO LastRead 5 FirstWrite 1}
		c_slice {Type IO LastRead -1 FirstWrite -1}
		U_slice {Type I LastRead 4 FirstWrite -1}
		table_exp_Z1_array_s {Type I LastRead -1 FirstWrite -1}
		table_f_Z3_array_V {Type I LastRead -1 FirstWrite -1}
		table_f_Z2_array_V {Type I LastRead -1 FirstWrite -1}}
	generic_tanh_float_s {
		t_in {Type I LastRead 0 FirstWrite -1}
		table_exp_Z1_array_s {Type I LastRead -1 FirstWrite -1}
		table_f_Z3_array_V {Type I LastRead -1 FirstWrite -1}
		table_f_Z2_array_V {Type I LastRead -1 FirstWrite -1}}
	exp_generic_double_s {
		x {Type I LastRead 0 FirstWrite -1}
		table_exp_Z1_array_s {Type I LastRead -1 FirstWrite -1}
		table_f_Z3_array_V {Type I LastRead -1 FirstWrite -1}
		table_f_Z2_array_V {Type I LastRead -1 FirstWrite -1}}
	generic_tanh_float_s {
		t_in {Type I LastRead 0 FirstWrite -1}
		table_exp_Z1_array_s {Type I LastRead -1 FirstWrite -1}
		table_f_Z3_array_V {Type I LastRead -1 FirstWrite -1}
		table_f_Z2_array_V {Type I LastRead -1 FirstWrite -1}}
	exp_generic_double_s {
		x {Type I LastRead 0 FirstWrite -1}
		table_exp_Z1_array_s {Type I LastRead -1 FirstWrite -1}
		table_f_Z3_array_V {Type I LastRead -1 FirstWrite -1}
		table_f_Z2_array_V {Type I LastRead -1 FirstWrite -1}}
	conv_bn_act_pool {
		Tlen {Type I LastRead 0 FirstWrite -1}
		W {Type I LastRead 5 FirstWrite -1}
		B {Type I LastRead 2 FirstWrite -1}
		C {Type I LastRead 0 FirstWrite -1}
		gamma {Type I LastRead 36 FirstWrite -1}
		beta {Type I LastRead 36 FirstWrite -1}
		mean {Type I LastRead 15 FirstWrite -1}
		var {Type I LastRead 3 FirstWrite -1}
		Psz {Type I LastRead 0 FirstWrite -1}
		U {Type O LastRead -1 FirstWrite 35}
		X_slice {Type I LastRead 5 FirstWrite -1}
		Y {Type IO LastRead -1 FirstWrite -1}}}

set hasDtUnsupportedChannel 0

set PerformanceInfo {[
	{"Name" : "Latency", "Min" : "20767760", "Max" : "541751140"}
	, {"Name" : "Interval", "Min" : "20767760", "Max" : "541751140"}
]}

set PipelineEnableSignalInfo {[
]}

set Spec2ImplPortList { 
	merged { ap_memory {  { merged_address0 mem_address 1 5 }  { merged_ce0 mem_ce 1 1 }  { merged_we0 mem_we 1 1 }  { merged_d0 mem_din 1 32 }  { merged_q0 mem_dout 0 32 } } }
}
