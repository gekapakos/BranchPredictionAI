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
	{ merged int 16 regular {array 32 { 2 3 } 1 1 }  }
}
set C_modelArgMapList {[ 
	{ "Name" : "merged", "interface" : "memory", "bitwidth" : 16, "direction" : "READWRITE"} ]}
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
	{ merged_d0 sc_out sc_lv 16 signal 0 } 
	{ merged_q0 sc_in sc_lv 16 signal 0 } 
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
 	{ "name": "merged_d0", "direction": "out", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "merged", "role": "d0" }} , 
 	{ "name": "merged_q0", "direction": "in", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "merged", "role": "q0" }}  ]}

set RtlHierarchyInfo {[
	{"ID" : "0", "Level" : "0", "Path" : "`AUTOTB_DUT_INST", "Parent" : "", "Child" : ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21", "22", "23", "24", "25", "26", "27", "28", "29", "30", "31", "32", "68", "85", "97", "110", "123", "136", "149", "150", "151", "152", "153", "154"],
		"CDFG" : "run_all_slices_unrol",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "0", "ap_idle" : "1",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "121199628", "EstimateLatencyMax" : "121439628",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"WaitState" : [
			{"State" : "ap_ST_fsm_state5", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_lstm_forward_unidir_fu_600"},
			{"State" : "ap_ST_fsm_state69", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_lstm_forward_unidir_fu_600"},
			{"State" : "ap_ST_fsm_state133", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_lstm_forward_unidir_fu_600"},
			{"State" : "ap_ST_fsm_state197", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_lstm_forward_unidir_fu_600"},
			{"State" : "ap_ST_fsm_state265", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_lstm_forward_unidir_fu_600"},
			{"State" : "ap_ST_fsm_state59", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_generic_tanh_float_s_fu_634"},
			{"State" : "ap_ST_fsm_state123", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_generic_tanh_float_s_fu_634"},
			{"State" : "ap_ST_fsm_state187", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_generic_tanh_float_s_fu_634"},
			{"State" : "ap_ST_fsm_state251", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_generic_tanh_float_s_fu_634"},
			{"State" : "ap_ST_fsm_state319", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_generic_tanh_float_s_fu_634"},
			{"State" : "ap_ST_fsm_state263", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_conv_bn_act_pool_fu_645"},
			{"State" : "ap_ST_fsm_state195", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_conv_bn_act_pool_1_fu_660"},
			{"State" : "ap_ST_fsm_state131", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_conv_bn_act_pool_2_fu_676"},
			{"State" : "ap_ST_fsm_state67", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_conv_bn_act_pool_3_fu_692"},
			{"State" : "ap_ST_fsm_state3", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_conv_bn_act_pool_4_fu_708"}],
		"Port" : [
			{"Name" : "merged", "Type" : "Memory", "Direction" : "IO"},
			{"Name" : "X_slice", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "136", "SubInstance" : "grp_conv_bn_act_pool_4_fu_708", "Port" : "X_slice"}]},
			{"Name" : "ConvW0", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "136", "SubInstance" : "grp_conv_bn_act_pool_4_fu_708", "Port" : "ConvW0"}]},
			{"Name" : "BN1_var0", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "136", "SubInstance" : "grp_conv_bn_act_pool_4_fu_708", "Port" : "BN1_var0"}]},
			{"Name" : "BN1_gamma0", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "136", "SubInstance" : "grp_conv_bn_act_pool_4_fu_708", "Port" : "BN1_gamma0"}]},
			{"Name" : "Y", "Type" : "Memory", "Direction" : "IO",
				"SubConnect" : [
					{"ID" : "123", "SubInstance" : "grp_conv_bn_act_pool_3_fu_692", "Port" : "Y"},
					{"ID" : "136", "SubInstance" : "grp_conv_bn_act_pool_4_fu_708", "Port" : "Y"},
					{"ID" : "85", "SubInstance" : "grp_conv_bn_act_pool_fu_645", "Port" : "Y"},
					{"ID" : "97", "SubInstance" : "grp_conv_bn_act_pool_1_fu_660", "Port" : "Y"},
					{"ID" : "110", "SubInstance" : "grp_conv_bn_act_pool_2_fu_676", "Port" : "Y"}]},
			{"Name" : "U_slice", "Type" : "Memory", "Direction" : "IO",
				"SubConnect" : [
					{"ID" : "123", "SubInstance" : "grp_conv_bn_act_pool_3_fu_692", "Port" : "U"},
					{"ID" : "136", "SubInstance" : "grp_conv_bn_act_pool_4_fu_708", "Port" : "U"},
					{"ID" : "85", "SubInstance" : "grp_conv_bn_act_pool_fu_645", "Port" : "U"},
					{"ID" : "32", "SubInstance" : "grp_lstm_forward_unidir_fu_600", "Port" : "U_slice"},
					{"ID" : "97", "SubInstance" : "grp_conv_bn_act_pool_1_fu_660", "Port" : "U"},
					{"ID" : "110", "SubInstance" : "grp_conv_bn_act_pool_2_fu_676", "Port" : "U"}]},
			{"Name" : "c_slice", "Type" : "Memory", "Direction" : "IO",
				"SubConnect" : [
					{"ID" : "32", "SubInstance" : "grp_lstm_forward_unidir_fu_600", "Port" : "c_slice"}]},
			{"Name" : "table_exp_Z1_array_s", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "68", "SubInstance" : "grp_generic_tanh_float_s_fu_634", "Port" : "table_exp_Z1_array_s"},
					{"ID" : "32", "SubInstance" : "grp_lstm_forward_unidir_fu_600", "Port" : "table_exp_Z1_array_s"}]},
			{"Name" : "table_f_Z3_array_V", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "68", "SubInstance" : "grp_generic_tanh_float_s_fu_634", "Port" : "table_f_Z3_array_V"},
					{"ID" : "32", "SubInstance" : "grp_lstm_forward_unidir_fu_600", "Port" : "table_f_Z3_array_V"}]},
			{"Name" : "table_f_Z2_array_V", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "68", "SubInstance" : "grp_generic_tanh_float_s_fu_634", "Port" : "table_f_Z2_array_V"},
					{"ID" : "32", "SubInstance" : "grp_lstm_forward_unidir_fu_600", "Port" : "table_f_Z2_array_V"}]},
			{"Name" : "LSTM_W_ifog0", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "32", "SubInstance" : "grp_lstm_forward_unidir_fu_600", "Port" : "W_ifog"}]},
			{"Name" : "LSTM_R_ifog0", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "32", "SubInstance" : "grp_lstm_forward_unidir_fu_600", "Port" : "R_ifog"}]},
			{"Name" : "LSTM_b_ifog0", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "32", "SubInstance" : "grp_lstm_forward_unidir_fu_600", "Port" : "b_ifog"}]},
			{"Name" : "h_slice", "Type" : "Memory", "Direction" : "IO",
				"SubConnect" : [
					{"ID" : "32", "SubInstance" : "grp_lstm_forward_unidir_fu_600", "Port" : "h_last"}]},
			{"Name" : "BN2_var0", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "BN2_gamma0", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "X_slice1", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "123", "SubInstance" : "grp_conv_bn_act_pool_3_fu_692", "Port" : "X_slice1"}]},
			{"Name" : "ConvW1", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "123", "SubInstance" : "grp_conv_bn_act_pool_3_fu_692", "Port" : "ConvW1"}]},
			{"Name" : "BN1_var1", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "123", "SubInstance" : "grp_conv_bn_act_pool_3_fu_692", "Port" : "BN1_var1"}]},
			{"Name" : "BN1_gamma1", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "123", "SubInstance" : "grp_conv_bn_act_pool_3_fu_692", "Port" : "BN1_gamma1"}]},
			{"Name" : "LSTM_W_ifog1", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "32", "SubInstance" : "grp_lstm_forward_unidir_fu_600", "Port" : "W_ifog"}]},
			{"Name" : "LSTM_R_ifog1", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "32", "SubInstance" : "grp_lstm_forward_unidir_fu_600", "Port" : "R_ifog"}]},
			{"Name" : "LSTM_b_ifog1", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "32", "SubInstance" : "grp_lstm_forward_unidir_fu_600", "Port" : "b_ifog"}]},
			{"Name" : "BN2_var1", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "BN2_gamma1", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "X_slice2", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "110", "SubInstance" : "grp_conv_bn_act_pool_2_fu_676", "Port" : "X_slice2"}]},
			{"Name" : "ConvW2", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "110", "SubInstance" : "grp_conv_bn_act_pool_2_fu_676", "Port" : "ConvW2"}]},
			{"Name" : "BN1_var2", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "110", "SubInstance" : "grp_conv_bn_act_pool_2_fu_676", "Port" : "BN1_var2"}]},
			{"Name" : "BN1_gamma2", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "110", "SubInstance" : "grp_conv_bn_act_pool_2_fu_676", "Port" : "BN1_gamma2"}]},
			{"Name" : "LSTM_W_ifog2", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "32", "SubInstance" : "grp_lstm_forward_unidir_fu_600", "Port" : "W_ifog"}]},
			{"Name" : "LSTM_R_ifog2", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "32", "SubInstance" : "grp_lstm_forward_unidir_fu_600", "Port" : "R_ifog"}]},
			{"Name" : "LSTM_b_ifog2", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "32", "SubInstance" : "grp_lstm_forward_unidir_fu_600", "Port" : "b_ifog"}]},
			{"Name" : "BN2_var2", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "BN2_gamma2", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "X_slice3", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "97", "SubInstance" : "grp_conv_bn_act_pool_1_fu_660", "Port" : "X_slice3"}]},
			{"Name" : "ConvW3", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "97", "SubInstance" : "grp_conv_bn_act_pool_1_fu_660", "Port" : "ConvW3"}]},
			{"Name" : "BN1_var3", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "97", "SubInstance" : "grp_conv_bn_act_pool_1_fu_660", "Port" : "BN1_var3"}]},
			{"Name" : "BN1_gamma3", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "97", "SubInstance" : "grp_conv_bn_act_pool_1_fu_660", "Port" : "BN1_gamma3"}]},
			{"Name" : "LSTM_W_ifog3", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "32", "SubInstance" : "grp_lstm_forward_unidir_fu_600", "Port" : "W_ifog"}]},
			{"Name" : "LSTM_R_ifog3", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "32", "SubInstance" : "grp_lstm_forward_unidir_fu_600", "Port" : "R_ifog"}]},
			{"Name" : "LSTM_b_ifog3", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "32", "SubInstance" : "grp_lstm_forward_unidir_fu_600", "Port" : "b_ifog"}]},
			{"Name" : "BN2_var3", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "BN2_gamma3", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "tokens4", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "Emb4", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "ConvW4", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "85", "SubInstance" : "grp_conv_bn_act_pool_fu_645", "Port" : "ConvW4"}]},
			{"Name" : "BN1_var4", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "85", "SubInstance" : "grp_conv_bn_act_pool_fu_645", "Port" : "BN1_var4"}]},
			{"Name" : "BN1_gamma4", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "85", "SubInstance" : "grp_conv_bn_act_pool_fu_645", "Port" : "BN1_gamma4"}]},
			{"Name" : "LSTM_W_ifog4", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "32", "SubInstance" : "grp_lstm_forward_unidir_fu_600", "Port" : "W_ifog"}]},
			{"Name" : "LSTM_R_ifog4", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "32", "SubInstance" : "grp_lstm_forward_unidir_fu_600", "Port" : "R_ifog"}]},
			{"Name" : "LSTM_b_ifog4", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "32", "SubInstance" : "grp_lstm_forward_unidir_fu_600", "Port" : "b_ifog"}]},
			{"Name" : "BN2_var4", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "BN2_gamma4", "Type" : "Memory", "Direction" : "I"}]},
	{"ID" : "1", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.Y_U", "Parent" : "0"},
	{"ID" : "2", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.U_slice_U", "Parent" : "0"},
	{"ID" : "3", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.LSTM_W_ifog0_U", "Parent" : "0"},
	{"ID" : "4", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.LSTM_R_ifog0_U", "Parent" : "0"},
	{"ID" : "5", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.LSTM_b_ifog0_U", "Parent" : "0"},
	{"ID" : "6", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.h_slice_U", "Parent" : "0"},
	{"ID" : "7", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.BN2_var0_U", "Parent" : "0"},
	{"ID" : "8", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.BN2_gamma0_U", "Parent" : "0"},
	{"ID" : "9", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.LSTM_W_ifog1_U", "Parent" : "0"},
	{"ID" : "10", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.LSTM_R_ifog1_U", "Parent" : "0"},
	{"ID" : "11", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.LSTM_b_ifog1_U", "Parent" : "0"},
	{"ID" : "12", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.BN2_var1_U", "Parent" : "0"},
	{"ID" : "13", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.BN2_gamma1_U", "Parent" : "0"},
	{"ID" : "14", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.LSTM_W_ifog2_U", "Parent" : "0"},
	{"ID" : "15", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.LSTM_R_ifog2_U", "Parent" : "0"},
	{"ID" : "16", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.LSTM_b_ifog2_U", "Parent" : "0"},
	{"ID" : "17", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.BN2_var2_U", "Parent" : "0"},
	{"ID" : "18", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.BN2_gamma2_U", "Parent" : "0"},
	{"ID" : "19", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.LSTM_W_ifog3_U", "Parent" : "0"},
	{"ID" : "20", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.LSTM_R_ifog3_U", "Parent" : "0"},
	{"ID" : "21", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.LSTM_b_ifog3_U", "Parent" : "0"},
	{"ID" : "22", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.BN2_var3_U", "Parent" : "0"},
	{"ID" : "23", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.BN2_gamma3_U", "Parent" : "0"},
	{"ID" : "24", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.tokens4_U", "Parent" : "0"},
	{"ID" : "25", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.Emb4_U", "Parent" : "0"},
	{"ID" : "26", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.LSTM_W_ifog4_U", "Parent" : "0"},
	{"ID" : "27", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.LSTM_R_ifog4_U", "Parent" : "0"},
	{"ID" : "28", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.LSTM_b_ifog4_U", "Parent" : "0"},
	{"ID" : "29", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.BN2_var4_U", "Parent" : "0"},
	{"ID" : "30", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.BN2_gamma4_U", "Parent" : "0"},
	{"ID" : "31", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.X_slice_1_U", "Parent" : "0"},
	{"ID" : "32", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.grp_lstm_forward_unidir_fu_600", "Parent" : "0", "Child" : ["33", "34", "35", "52", "53", "54", "55", "56", "57", "58", "59", "60", "61", "62", "63", "64", "65", "66", "67"],
		"CDFG" : "lstm_forward_unidir",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "0", "ap_idle" : "1",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "1208926", "EstimateLatencyMax" : "1255006",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"WaitState" : [
			{"State" : "ap_ST_fsm_state73", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_generic_tanh_float_s_fu_325"},
			{"State" : "ap_ST_fsm_state87", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_generic_tanh_float_s_fu_325"}],
		"Port" : [
			{"Name" : "W_ifog", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "R_ifog", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "b_ifog", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "h_last", "Type" : "Memory", "Direction" : "IO"},
			{"Name" : "c_slice", "Type" : "Memory", "Direction" : "IO"},
			{"Name" : "U_slice", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "table_exp_Z1_array_s", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "35", "SubInstance" : "grp_generic_tanh_float_s_fu_325", "Port" : "table_exp_Z1_array_s"}]},
			{"Name" : "table_f_Z3_array_V", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "35", "SubInstance" : "grp_generic_tanh_float_s_fu_325", "Port" : "table_f_Z3_array_V"}]},
			{"Name" : "table_f_Z2_array_V", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "35", "SubInstance" : "grp_generic_tanh_float_s_fu_325", "Port" : "table_f_Z2_array_V"}]}]},
	{"ID" : "33", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_lstm_forward_unidir_fu_600.c_slice_U", "Parent" : "32"},
	{"ID" : "34", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_lstm_forward_unidir_fu_600.z_U", "Parent" : "32"},
	{"ID" : "35", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_lstm_forward_unidir_fu_600.grp_generic_tanh_float_s_fu_325", "Parent" : "32", "Child" : ["36", "45", "46", "47", "48", "49", "50", "51"],
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
					{"ID" : "36", "SubInstance" : "grp_exp_generic_double_s_fu_89", "Port" : "table_exp_Z1_array_s"}]},
			{"Name" : "table_f_Z3_array_V", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "36", "SubInstance" : "grp_exp_generic_double_s_fu_89", "Port" : "table_f_Z3_array_V"}]},
			{"Name" : "table_f_Z2_array_V", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "36", "SubInstance" : "grp_exp_generic_double_s_fu_89", "Port" : "table_f_Z2_array_V"}]}]},
	{"ID" : "36", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_lstm_forward_unidir_fu_600.grp_generic_tanh_float_s_fu_325.grp_exp_generic_double_s_fu_89", "Parent" : "35", "Child" : ["37", "38", "39", "40", "41", "42", "43", "44"],
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
	{"ID" : "37", "Level" : "4", "Path" : "`AUTOTB_DUT_INST.grp_lstm_forward_unidir_fu_600.grp_generic_tanh_float_s_fu_325.grp_exp_generic_double_s_fu_89.table_exp_Z1_array_s_U", "Parent" : "36"},
	{"ID" : "38", "Level" : "4", "Path" : "`AUTOTB_DUT_INST.grp_lstm_forward_unidir_fu_600.grp_generic_tanh_float_s_fu_325.grp_exp_generic_double_s_fu_89.table_f_Z3_array_V_U", "Parent" : "36"},
	{"ID" : "39", "Level" : "4", "Path" : "`AUTOTB_DUT_INST.grp_lstm_forward_unidir_fu_600.grp_generic_tanh_float_s_fu_325.grp_exp_generic_double_s_fu_89.table_f_Z2_array_V_U", "Parent" : "36"},
	{"ID" : "40", "Level" : "4", "Path" : "`AUTOTB_DUT_INST.grp_lstm_forward_unidir_fu_600.grp_generic_tanh_float_s_fu_325.grp_exp_generic_double_s_fu_89.main_mul_72ns_13s_84_5_1_U22", "Parent" : "36"},
	{"ID" : "41", "Level" : "4", "Path" : "`AUTOTB_DUT_INST.grp_lstm_forward_unidir_fu_600.grp_generic_tanh_float_s_fu_325.grp_exp_generic_double_s_fu_89.main_mul_36ns_43ns_79_2_1_U23", "Parent" : "36"},
	{"ID" : "42", "Level" : "4", "Path" : "`AUTOTB_DUT_INST.grp_lstm_forward_unidir_fu_600.grp_generic_tanh_float_s_fu_325.grp_exp_generic_double_s_fu_89.main_mul_44ns_49ns_93_2_1_U24", "Parent" : "36"},
	{"ID" : "43", "Level" : "4", "Path" : "`AUTOTB_DUT_INST.grp_lstm_forward_unidir_fu_600.grp_generic_tanh_float_s_fu_325.grp_exp_generic_double_s_fu_89.main_mul_50ns_50ns_100_2_1_U25", "Parent" : "36"},
	{"ID" : "44", "Level" : "4", "Path" : "`AUTOTB_DUT_INST.grp_lstm_forward_unidir_fu_600.grp_generic_tanh_float_s_fu_325.grp_exp_generic_double_s_fu_89.main_mac_muladd_16ns_16s_19s_31_1_1_U26", "Parent" : "36"},
	{"ID" : "45", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_lstm_forward_unidir_fu_600.grp_generic_tanh_float_s_fu_325.main_faddfsub_32ns_32ns_32_5_full_dsp_1_U36", "Parent" : "35"},
	{"ID" : "46", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_lstm_forward_unidir_fu_600.grp_generic_tanh_float_s_fu_325.main_fmul_32ns_32ns_32_4_max_dsp_1_U37", "Parent" : "35"},
	{"ID" : "47", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_lstm_forward_unidir_fu_600.grp_generic_tanh_float_s_fu_325.main_fdiv_32ns_32ns_32_16_1_U38", "Parent" : "35"},
	{"ID" : "48", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_lstm_forward_unidir_fu_600.grp_generic_tanh_float_s_fu_325.main_fptrunc_64ns_32_2_1_U39", "Parent" : "35"},
	{"ID" : "49", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_lstm_forward_unidir_fu_600.grp_generic_tanh_float_s_fu_325.main_fpext_32ns_64_2_1_U40", "Parent" : "35"},
	{"ID" : "50", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_lstm_forward_unidir_fu_600.grp_generic_tanh_float_s_fu_325.main_fcmp_32ns_32ns_1_2_1_U41", "Parent" : "35"},
	{"ID" : "51", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_lstm_forward_unidir_fu_600.grp_generic_tanh_float_s_fu_325.main_dadd_64ns_64ns_64_5_full_dsp_1_U42", "Parent" : "35"},
	{"ID" : "52", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_lstm_forward_unidir_fu_600.main_fadd_32ns_32ns_32_5_full_dsp_1_U50", "Parent" : "32"},
	{"ID" : "53", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_lstm_forward_unidir_fu_600.main_fadd_32ns_32ns_32_5_full_dsp_1_U51", "Parent" : "32"},
	{"ID" : "54", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_lstm_forward_unidir_fu_600.main_fdiv_32ns_32ns_32_16_1_U52", "Parent" : "32"},
	{"ID" : "55", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_lstm_forward_unidir_fu_600.main_fdiv_32ns_32ns_32_16_1_U53", "Parent" : "32"},
	{"ID" : "56", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_lstm_forward_unidir_fu_600.main_fexp_32ns_32ns_32_9_full_dsp_1_U54", "Parent" : "32"},
	{"ID" : "57", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_lstm_forward_unidir_fu_600.main_fexp_32ns_32ns_32_9_full_dsp_1_U55", "Parent" : "32"},
	{"ID" : "58", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_lstm_forward_unidir_fu_600.main_sptohp_32ns_16_2_1_U56", "Parent" : "32"},
	{"ID" : "59", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_lstm_forward_unidir_fu_600.main_sptohp_32ns_16_2_1_U57", "Parent" : "32"},
	{"ID" : "60", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_lstm_forward_unidir_fu_600.main_sptohp_32ns_16_2_1_U58", "Parent" : "32"},
	{"ID" : "61", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_lstm_forward_unidir_fu_600.main_hptosp_16ns_32_2_1_U59", "Parent" : "32"},
	{"ID" : "62", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_lstm_forward_unidir_fu_600.main_hptosp_16ns_32_2_1_U60", "Parent" : "32"},
	{"ID" : "63", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_lstm_forward_unidir_fu_600.main_hadd_16ns_16ns_16_5_full_dsp_1_U61", "Parent" : "32"},
	{"ID" : "64", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_lstm_forward_unidir_fu_600.main_hsub_16ns_16ns_16_5_full_dsp_1_U62", "Parent" : "32"},
	{"ID" : "65", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_lstm_forward_unidir_fu_600.main_hsub_16ns_16ns_16_5_full_dsp_1_U63", "Parent" : "32"},
	{"ID" : "66", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_lstm_forward_unidir_fu_600.main_hmul_16ns_16ns_16_4_max_dsp_1_U64", "Parent" : "32"},
	{"ID" : "67", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_lstm_forward_unidir_fu_600.main_hmul_16ns_16ns_16_4_max_dsp_1_U65", "Parent" : "32"},
	{"ID" : "68", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.grp_generic_tanh_float_s_fu_634", "Parent" : "0", "Child" : ["69", "78", "79", "80", "81", "82", "83", "84"],
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
					{"ID" : "69", "SubInstance" : "grp_exp_generic_double_s_fu_89", "Port" : "table_exp_Z1_array_s"}]},
			{"Name" : "table_f_Z3_array_V", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "69", "SubInstance" : "grp_exp_generic_double_s_fu_89", "Port" : "table_f_Z3_array_V"}]},
			{"Name" : "table_f_Z2_array_V", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "69", "SubInstance" : "grp_exp_generic_double_s_fu_89", "Port" : "table_f_Z2_array_V"}]}]},
	{"ID" : "69", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_generic_tanh_float_s_fu_634.grp_exp_generic_double_s_fu_89", "Parent" : "68", "Child" : ["70", "71", "72", "73", "74", "75", "76", "77"],
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
	{"ID" : "70", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_generic_tanh_float_s_fu_634.grp_exp_generic_double_s_fu_89.table_exp_Z1_array_s_U", "Parent" : "69"},
	{"ID" : "71", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_generic_tanh_float_s_fu_634.grp_exp_generic_double_s_fu_89.table_f_Z3_array_V_U", "Parent" : "69"},
	{"ID" : "72", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_generic_tanh_float_s_fu_634.grp_exp_generic_double_s_fu_89.table_f_Z2_array_V_U", "Parent" : "69"},
	{"ID" : "73", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_generic_tanh_float_s_fu_634.grp_exp_generic_double_s_fu_89.main_mul_72ns_13s_84_5_1_U22", "Parent" : "69"},
	{"ID" : "74", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_generic_tanh_float_s_fu_634.grp_exp_generic_double_s_fu_89.main_mul_36ns_43ns_79_2_1_U23", "Parent" : "69"},
	{"ID" : "75", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_generic_tanh_float_s_fu_634.grp_exp_generic_double_s_fu_89.main_mul_44ns_49ns_93_2_1_U24", "Parent" : "69"},
	{"ID" : "76", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_generic_tanh_float_s_fu_634.grp_exp_generic_double_s_fu_89.main_mul_50ns_50ns_100_2_1_U25", "Parent" : "69"},
	{"ID" : "77", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_generic_tanh_float_s_fu_634.grp_exp_generic_double_s_fu_89.main_mac_muladd_16ns_16s_19s_31_1_1_U26", "Parent" : "69"},
	{"ID" : "78", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_generic_tanh_float_s_fu_634.main_faddfsub_32ns_32ns_32_5_full_dsp_1_U36", "Parent" : "68"},
	{"ID" : "79", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_generic_tanh_float_s_fu_634.main_fmul_32ns_32ns_32_4_max_dsp_1_U37", "Parent" : "68"},
	{"ID" : "80", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_generic_tanh_float_s_fu_634.main_fdiv_32ns_32ns_32_16_1_U38", "Parent" : "68"},
	{"ID" : "81", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_generic_tanh_float_s_fu_634.main_fptrunc_64ns_32_2_1_U39", "Parent" : "68"},
	{"ID" : "82", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_generic_tanh_float_s_fu_634.main_fpext_32ns_64_2_1_U40", "Parent" : "68"},
	{"ID" : "83", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_generic_tanh_float_s_fu_634.main_fcmp_32ns_32ns_1_2_1_U41", "Parent" : "68"},
	{"ID" : "84", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_generic_tanh_float_s_fu_634.main_dadd_64ns_64ns_64_5_full_dsp_1_U42", "Parent" : "68"},
	{"ID" : "85", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.grp_conv_bn_act_pool_fu_645", "Parent" : "0", "Child" : ["86", "87", "88", "89", "90", "91", "92", "93", "94", "95", "96"],
		"CDFG" : "conv_bn_act_pool",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "0", "ap_idle" : "1",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "65526951", "EstimateLatencyMax" : "65526951",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"Port" : [
			{"Name" : "X", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "U", "Type" : "Memory", "Direction" : "O"},
			{"Name" : "ConvW4", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "Y", "Type" : "Memory", "Direction" : "IO"},
			{"Name" : "BN1_var4", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "BN1_gamma4", "Type" : "Memory", "Direction" : "I"}]},
	{"ID" : "86", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_bn_act_pool_fu_645.ConvW4_U", "Parent" : "85"},
	{"ID" : "87", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_bn_act_pool_fu_645.BN1_var4_U", "Parent" : "85"},
	{"ID" : "88", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_bn_act_pool_fu_645.BN1_gamma4_U", "Parent" : "85"},
	{"ID" : "89", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_bn_act_pool_fu_645.main_fdiv_32ns_32ns_32_16_1_U112", "Parent" : "85"},
	{"ID" : "90", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_bn_act_pool_fu_645.main_fsqrt_32ns_32ns_32_12_1_U113", "Parent" : "85"},
	{"ID" : "91", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_bn_act_pool_fu_645.main_sptohp_32ns_16_2_1_U114", "Parent" : "85"},
	{"ID" : "92", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_bn_act_pool_fu_645.main_hptosp_16ns_32_2_1_U115", "Parent" : "85"},
	{"ID" : "93", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_bn_act_pool_fu_645.main_hadd_16ns_16ns_16_5_full_dsp_1_U116", "Parent" : "85"},
	{"ID" : "94", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_bn_act_pool_fu_645.main_hmul_16ns_16ns_16_4_max_dsp_1_U117", "Parent" : "85"},
	{"ID" : "95", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_bn_act_pool_fu_645.main_hdiv_16ns_16ns_16_7_1_U118", "Parent" : "85"},
	{"ID" : "96", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_bn_act_pool_fu_645.main_hcmp_16ns_16ns_1_2_1_U119", "Parent" : "85"},
	{"ID" : "97", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.grp_conv_bn_act_pool_1_fu_660", "Parent" : "0", "Child" : ["98", "99", "100", "101", "102", "103", "104", "105", "106", "107", "108", "109"],
		"CDFG" : "conv_bn_act_pool_1",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "0", "ap_idle" : "1",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "27869799", "EstimateLatencyMax" : "27869799",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"Port" : [
			{"Name" : "Y", "Type" : "Memory", "Direction" : "IO"},
			{"Name" : "U", "Type" : "Memory", "Direction" : "O"},
			{"Name" : "X_slice3", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "ConvW3", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "BN1_var3", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "BN1_gamma3", "Type" : "Memory", "Direction" : "I"}]},
	{"ID" : "98", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_bn_act_pool_1_fu_660.X_slice3_U", "Parent" : "97"},
	{"ID" : "99", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_bn_act_pool_1_fu_660.ConvW3_U", "Parent" : "97"},
	{"ID" : "100", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_bn_act_pool_1_fu_660.BN1_var3_U", "Parent" : "97"},
	{"ID" : "101", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_bn_act_pool_1_fu_660.BN1_gamma3_U", "Parent" : "97"},
	{"ID" : "102", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_bn_act_pool_1_fu_660.main_fdiv_32ns_32ns_32_16_1_U100", "Parent" : "97"},
	{"ID" : "103", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_bn_act_pool_1_fu_660.main_fsqrt_32ns_32ns_32_12_1_U101", "Parent" : "97"},
	{"ID" : "104", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_bn_act_pool_1_fu_660.main_sptohp_32ns_16_2_1_U102", "Parent" : "97"},
	{"ID" : "105", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_bn_act_pool_1_fu_660.main_hptosp_16ns_32_2_1_U103", "Parent" : "97"},
	{"ID" : "106", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_bn_act_pool_1_fu_660.main_hadd_16ns_16ns_16_5_full_dsp_1_U104", "Parent" : "97"},
	{"ID" : "107", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_bn_act_pool_1_fu_660.main_hmul_16ns_16ns_16_4_max_dsp_1_U105", "Parent" : "97"},
	{"ID" : "108", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_bn_act_pool_1_fu_660.main_hdiv_16ns_16ns_16_7_1_U106", "Parent" : "97"},
	{"ID" : "109", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_bn_act_pool_1_fu_660.main_hcmp_16ns_16ns_1_2_1_U107", "Parent" : "97"},
	{"ID" : "110", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.grp_conv_bn_act_pool_2_fu_676", "Parent" : "0", "Child" : ["111", "112", "113", "114", "115", "116", "117", "118", "119", "120", "121", "122"],
		"CDFG" : "conv_bn_act_pool_2",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "0", "ap_idle" : "1",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "12711495", "EstimateLatencyMax" : "12711495",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"Port" : [
			{"Name" : "Y", "Type" : "Memory", "Direction" : "IO"},
			{"Name" : "U", "Type" : "Memory", "Direction" : "O"},
			{"Name" : "X_slice2", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "ConvW2", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "BN1_var2", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "BN1_gamma2", "Type" : "Memory", "Direction" : "I"}]},
	{"ID" : "111", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_bn_act_pool_2_fu_676.X_slice2_U", "Parent" : "110"},
	{"ID" : "112", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_bn_act_pool_2_fu_676.ConvW2_U", "Parent" : "110"},
	{"ID" : "113", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_bn_act_pool_2_fu_676.BN1_var2_U", "Parent" : "110"},
	{"ID" : "114", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_bn_act_pool_2_fu_676.BN1_gamma2_U", "Parent" : "110"},
	{"ID" : "115", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_bn_act_pool_2_fu_676.main_fdiv_32ns_32ns_32_16_1_U88", "Parent" : "110"},
	{"ID" : "116", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_bn_act_pool_2_fu_676.main_fsqrt_32ns_32ns_32_12_1_U89", "Parent" : "110"},
	{"ID" : "117", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_bn_act_pool_2_fu_676.main_sptohp_32ns_16_2_1_U90", "Parent" : "110"},
	{"ID" : "118", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_bn_act_pool_2_fu_676.main_hptosp_16ns_32_2_1_U91", "Parent" : "110"},
	{"ID" : "119", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_bn_act_pool_2_fu_676.main_hadd_16ns_16ns_16_5_full_dsp_1_U92", "Parent" : "110"},
	{"ID" : "120", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_bn_act_pool_2_fu_676.main_hmul_16ns_16ns_16_4_max_dsp_1_U93", "Parent" : "110"},
	{"ID" : "121", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_bn_act_pool_2_fu_676.main_hdiv_16ns_16ns_16_7_1_U94", "Parent" : "110"},
	{"ID" : "122", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_bn_act_pool_2_fu_676.main_hcmp_16ns_16ns_1_2_1_U95", "Parent" : "110"},
	{"ID" : "123", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.grp_conv_bn_act_pool_3_fu_692", "Parent" : "0", "Child" : ["124", "125", "126", "127", "128", "129", "130", "131", "132", "133", "134", "135"],
		"CDFG" : "conv_bn_act_pool_3",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "0", "ap_idle" : "1",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "6048401", "EstimateLatencyMax" : "6048401",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"Port" : [
			{"Name" : "Y", "Type" : "Memory", "Direction" : "IO"},
			{"Name" : "U", "Type" : "Memory", "Direction" : "O"},
			{"Name" : "X_slice1", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "ConvW1", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "BN1_var1", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "BN1_gamma1", "Type" : "Memory", "Direction" : "I"}]},
	{"ID" : "124", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_bn_act_pool_3_fu_692.X_slice1_U", "Parent" : "123"},
	{"ID" : "125", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_bn_act_pool_3_fu_692.ConvW1_U", "Parent" : "123"},
	{"ID" : "126", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_bn_act_pool_3_fu_692.BN1_var1_U", "Parent" : "123"},
	{"ID" : "127", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_bn_act_pool_3_fu_692.BN1_gamma1_U", "Parent" : "123"},
	{"ID" : "128", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_bn_act_pool_3_fu_692.main_fdiv_32ns_32ns_32_16_1_U76", "Parent" : "123"},
	{"ID" : "129", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_bn_act_pool_3_fu_692.main_fsqrt_32ns_32ns_32_12_1_U77", "Parent" : "123"},
	{"ID" : "130", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_bn_act_pool_3_fu_692.main_sptohp_32ns_16_2_1_U78", "Parent" : "123"},
	{"ID" : "131", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_bn_act_pool_3_fu_692.main_hptosp_16ns_32_2_1_U79", "Parent" : "123"},
	{"ID" : "132", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_bn_act_pool_3_fu_692.main_hadd_16ns_16ns_16_5_full_dsp_1_U80", "Parent" : "123"},
	{"ID" : "133", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_bn_act_pool_3_fu_692.main_hmul_16ns_16ns_16_4_max_dsp_1_U81", "Parent" : "123"},
	{"ID" : "134", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_bn_act_pool_3_fu_692.main_hdiv_16ns_16ns_16_7_1_U82", "Parent" : "123"},
	{"ID" : "135", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_bn_act_pool_3_fu_692.main_hcmp_16ns_16ns_1_2_1_U83", "Parent" : "123"},
	{"ID" : "136", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.grp_conv_bn_act_pool_4_fu_708", "Parent" : "0", "Child" : ["137", "138", "139", "140", "141", "142", "143", "144", "145", "146", "147", "148"],
		"CDFG" : "conv_bn_act_pool_4",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "0", "ap_idle" : "1",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "2949379", "EstimateLatencyMax" : "2949379",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"Port" : [
			{"Name" : "Y", "Type" : "Memory", "Direction" : "IO"},
			{"Name" : "U", "Type" : "Memory", "Direction" : "O"},
			{"Name" : "X_slice", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "ConvW0", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "BN1_var0", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "BN1_gamma0", "Type" : "Memory", "Direction" : "I"}]},
	{"ID" : "137", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_bn_act_pool_4_fu_708.X_slice_U", "Parent" : "136"},
	{"ID" : "138", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_bn_act_pool_4_fu_708.ConvW0_U", "Parent" : "136"},
	{"ID" : "139", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_bn_act_pool_4_fu_708.BN1_var0_U", "Parent" : "136"},
	{"ID" : "140", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_bn_act_pool_4_fu_708.BN1_gamma0_U", "Parent" : "136"},
	{"ID" : "141", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_bn_act_pool_4_fu_708.main_fdiv_32ns_32ns_32_16_1_U1", "Parent" : "136"},
	{"ID" : "142", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_bn_act_pool_4_fu_708.main_fsqrt_32ns_32ns_32_12_1_U2", "Parent" : "136"},
	{"ID" : "143", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_bn_act_pool_4_fu_708.main_sptohp_32ns_16_2_1_U3", "Parent" : "136"},
	{"ID" : "144", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_bn_act_pool_4_fu_708.main_hptosp_16ns_32_2_1_U4", "Parent" : "136"},
	{"ID" : "145", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_bn_act_pool_4_fu_708.main_hadd_16ns_16ns_16_5_full_dsp_1_U5", "Parent" : "136"},
	{"ID" : "146", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_bn_act_pool_4_fu_708.main_hmul_16ns_16ns_16_4_max_dsp_1_U6", "Parent" : "136"},
	{"ID" : "147", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_bn_act_pool_4_fu_708.main_hdiv_16ns_16ns_16_7_1_U7", "Parent" : "136"},
	{"ID" : "148", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_bn_act_pool_4_fu_708.main_hcmp_16ns_16ns_1_2_1_U8", "Parent" : "136"},
	{"ID" : "149", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.main_fdiv_32ns_32ns_32_16_1_U124", "Parent" : "0"},
	{"ID" : "150", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.main_fsqrt_32ns_32ns_32_12_1_U125", "Parent" : "0"},
	{"ID" : "151", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.main_sptohp_32ns_16_2_1_U126", "Parent" : "0"},
	{"ID" : "152", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.main_hptosp_16ns_32_2_1_U127", "Parent" : "0"},
	{"ID" : "153", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.main_hadd_16ns_16ns_16_5_full_dsp_1_U128", "Parent" : "0"},
	{"ID" : "154", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.main_hmul_16ns_16ns_16_4_max_dsp_1_U129", "Parent" : "0"}]}


set ArgLastReadFirstWriteLatency {
	run_all_slices_unrol {
		merged {Type IO LastRead 31 FirstWrite 1}
		X_slice {Type I LastRead -1 FirstWrite -1}
		ConvW0 {Type I LastRead -1 FirstWrite -1}
		BN1_var0 {Type I LastRead -1 FirstWrite -1}
		BN1_gamma0 {Type I LastRead -1 FirstWrite -1}
		Y {Type IO LastRead -1 FirstWrite -1}
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
		X_slice1 {Type I LastRead -1 FirstWrite -1}
		ConvW1 {Type I LastRead -1 FirstWrite -1}
		BN1_var1 {Type I LastRead -1 FirstWrite -1}
		BN1_gamma1 {Type I LastRead -1 FirstWrite -1}
		LSTM_W_ifog1 {Type I LastRead -1 FirstWrite -1}
		LSTM_R_ifog1 {Type I LastRead -1 FirstWrite -1}
		LSTM_b_ifog1 {Type I LastRead -1 FirstWrite -1}
		BN2_var1 {Type I LastRead -1 FirstWrite -1}
		BN2_gamma1 {Type I LastRead -1 FirstWrite -1}
		X_slice2 {Type I LastRead -1 FirstWrite -1}
		ConvW2 {Type I LastRead -1 FirstWrite -1}
		BN1_var2 {Type I LastRead -1 FirstWrite -1}
		BN1_gamma2 {Type I LastRead -1 FirstWrite -1}
		LSTM_W_ifog2 {Type I LastRead -1 FirstWrite -1}
		LSTM_R_ifog2 {Type I LastRead -1 FirstWrite -1}
		LSTM_b_ifog2 {Type I LastRead -1 FirstWrite -1}
		BN2_var2 {Type I LastRead -1 FirstWrite -1}
		BN2_gamma2 {Type I LastRead -1 FirstWrite -1}
		X_slice3 {Type I LastRead -1 FirstWrite -1}
		ConvW3 {Type I LastRead -1 FirstWrite -1}
		BN1_var3 {Type I LastRead -1 FirstWrite -1}
		BN1_gamma3 {Type I LastRead -1 FirstWrite -1}
		LSTM_W_ifog3 {Type I LastRead -1 FirstWrite -1}
		LSTM_R_ifog3 {Type I LastRead -1 FirstWrite -1}
		LSTM_b_ifog3 {Type I LastRead -1 FirstWrite -1}
		BN2_var3 {Type I LastRead -1 FirstWrite -1}
		BN2_gamma3 {Type I LastRead -1 FirstWrite -1}
		tokens4 {Type I LastRead -1 FirstWrite -1}
		Emb4 {Type I LastRead -1 FirstWrite -1}
		ConvW4 {Type I LastRead -1 FirstWrite -1}
		BN1_var4 {Type I LastRead -1 FirstWrite -1}
		BN1_gamma4 {Type I LastRead -1 FirstWrite -1}
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
		X {Type I LastRead 4 FirstWrite -1}
		U {Type O LastRead -1 FirstWrite 12}
		ConvW4 {Type I LastRead -1 FirstWrite -1}
		Y {Type IO LastRead 21 FirstWrite 3}
		BN1_var4 {Type I LastRead -1 FirstWrite -1}
		BN1_gamma4 {Type I LastRead -1 FirstWrite -1}}
	conv_bn_act_pool_1 {
		Y {Type IO LastRead 21 FirstWrite 3}
		U {Type O LastRead -1 FirstWrite 12}
		X_slice3 {Type I LastRead -1 FirstWrite -1}
		ConvW3 {Type I LastRead -1 FirstWrite -1}
		BN1_var3 {Type I LastRead -1 FirstWrite -1}
		BN1_gamma3 {Type I LastRead -1 FirstWrite -1}}
	conv_bn_act_pool_2 {
		Y {Type IO LastRead 21 FirstWrite 3}
		U {Type O LastRead -1 FirstWrite 12}
		X_slice2 {Type I LastRead -1 FirstWrite -1}
		ConvW2 {Type I LastRead -1 FirstWrite -1}
		BN1_var2 {Type I LastRead -1 FirstWrite -1}
		BN1_gamma2 {Type I LastRead -1 FirstWrite -1}}
	conv_bn_act_pool_3 {
		Y {Type IO LastRead 21 FirstWrite 3}
		U {Type O LastRead -1 FirstWrite 12}
		X_slice1 {Type I LastRead -1 FirstWrite -1}
		ConvW1 {Type I LastRead -1 FirstWrite -1}
		BN1_var1 {Type I LastRead -1 FirstWrite -1}
		BN1_gamma1 {Type I LastRead -1 FirstWrite -1}}
	conv_bn_act_pool_4 {
		Y {Type IO LastRead 21 FirstWrite 3}
		U {Type O LastRead -1 FirstWrite 12}
		X_slice {Type I LastRead -1 FirstWrite -1}
		ConvW0 {Type I LastRead -1 FirstWrite -1}
		BN1_var0 {Type I LastRead -1 FirstWrite -1}
		BN1_gamma0 {Type I LastRead -1 FirstWrite -1}}}

set hasDtUnsupportedChannel 0

set PerformanceInfo {[
	{"Name" : "Latency", "Min" : "121199628", "Max" : "121439628"}
	, {"Name" : "Interval", "Min" : "121199628", "Max" : "121439628"}
]}

set PipelineEnableSignalInfo {[
]}

set Spec2ImplPortList { 
	merged { ap_memory {  { merged_address0 mem_address 1 5 }  { merged_ce0 mem_ce 1 1 }  { merged_we0 mem_we 1 1 }  { merged_d0 mem_din 1 16 }  { merged_q0 mem_dout 0 16 } } }
}
