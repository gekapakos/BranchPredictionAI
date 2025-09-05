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
	{"ID" : "0", "Level" : "0", "Path" : "`AUTOTB_DUT_INST", "Parent" : "", "Child" : ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21", "22", "23", "24", "25", "26", "27", "28", "29", "49", "63", "78", "93", "108", "123", "144", "145", "146", "147", "148", "149", "150", "151"],
		"CDFG" : "run_all_slices_unrol",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "0", "ap_idle" : "1",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "47925187", "EstimateLatencyMax" : "48283423",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"WaitState" : [
			{"State" : "ap_ST_fsm_state5", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_lstm_forward_unidir_fu_604"},
			{"State" : "ap_ST_fsm_state66", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_lstm_forward_unidir_fu_604"},
			{"State" : "ap_ST_fsm_state127", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_lstm_forward_unidir_fu_604"},
			{"State" : "ap_ST_fsm_state188", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_lstm_forward_unidir_fu_604"},
			{"State" : "ap_ST_fsm_state251", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_lstm_forward_unidir_fu_604"},
			{"State" : "ap_ST_fsm_state249", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_conv_bn_act_pool_fu_632"},
			{"State" : "ap_ST_fsm_state186", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_conv_bn_act_pool_1_fu_641"},
			{"State" : "ap_ST_fsm_state125", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_conv_bn_act_pool_2_fu_651"},
			{"State" : "ap_ST_fsm_state64", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_conv_bn_act_pool_3_fu_661"},
			{"State" : "ap_ST_fsm_state3", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_conv_bn_act_pool_4_fu_671"},
			{"State" : "ap_ST_fsm_state247", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_embedding_forward_dy_fu_681"}],
		"Port" : [
			{"Name" : "merged", "Type" : "Memory", "Direction" : "IO"},
			{"Name" : "X_slice", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "108", "SubInstance" : "grp_conv_bn_act_pool_4_fu_671", "Port" : "X_slice"}]},
			{"Name" : "ConvW0", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "108", "SubInstance" : "grp_conv_bn_act_pool_4_fu_671", "Port" : "ConvW0"}]},
			{"Name" : "U_slice", "Type" : "Memory", "Direction" : "IO",
				"SubConnect" : [
					{"ID" : "108", "SubInstance" : "grp_conv_bn_act_pool_4_fu_671", "Port" : "U"},
					{"ID" : "78", "SubInstance" : "grp_conv_bn_act_pool_2_fu_651", "Port" : "U"},
					{"ID" : "49", "SubInstance" : "grp_conv_bn_act_pool_fu_632", "Port" : "U"},
					{"ID" : "63", "SubInstance" : "grp_conv_bn_act_pool_1_fu_641", "Port" : "U"},
					{"ID" : "93", "SubInstance" : "grp_conv_bn_act_pool_3_fu_661", "Port" : "U"},
					{"ID" : "29", "SubInstance" : "grp_lstm_forward_unidir_fu_604", "Port" : "U_slice"}]},
			{"Name" : "c_slice", "Type" : "Memory", "Direction" : "IO",
				"SubConnect" : [
					{"ID" : "29", "SubInstance" : "grp_lstm_forward_unidir_fu_604", "Port" : "c_slice"}]},
			{"Name" : "LSTM_W_ifog0", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "29", "SubInstance" : "grp_lstm_forward_unidir_fu_604", "Port" : "W_ifog"}]},
			{"Name" : "LSTM_R_ifog0", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "29", "SubInstance" : "grp_lstm_forward_unidir_fu_604", "Port" : "R_ifog"}]},
			{"Name" : "LSTM_b_ifog0", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "29", "SubInstance" : "grp_lstm_forward_unidir_fu_604", "Port" : "b_ifog"}]},
			{"Name" : "h_slice", "Type" : "Memory", "Direction" : "IO",
				"SubConnect" : [
					{"ID" : "29", "SubInstance" : "grp_lstm_forward_unidir_fu_604", "Port" : "h_last"}]},
			{"Name" : "BN2_var0", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "BN2_gamma0", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "X_slice1", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "93", "SubInstance" : "grp_conv_bn_act_pool_3_fu_661", "Port" : "X_slice1"}]},
			{"Name" : "ConvW1", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "93", "SubInstance" : "grp_conv_bn_act_pool_3_fu_661", "Port" : "ConvW1"}]},
			{"Name" : "LSTM_W_ifog1", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "29", "SubInstance" : "grp_lstm_forward_unidir_fu_604", "Port" : "W_ifog"}]},
			{"Name" : "LSTM_R_ifog1", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "29", "SubInstance" : "grp_lstm_forward_unidir_fu_604", "Port" : "R_ifog"}]},
			{"Name" : "LSTM_b_ifog1", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "29", "SubInstance" : "grp_lstm_forward_unidir_fu_604", "Port" : "b_ifog"}]},
			{"Name" : "BN2_var1", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "BN2_gamma1", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "X_slice2", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "78", "SubInstance" : "grp_conv_bn_act_pool_2_fu_651", "Port" : "X_slice2"}]},
			{"Name" : "ConvW2", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "78", "SubInstance" : "grp_conv_bn_act_pool_2_fu_651", "Port" : "ConvW2"}]},
			{"Name" : "LSTM_W_ifog2", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "29", "SubInstance" : "grp_lstm_forward_unidir_fu_604", "Port" : "W_ifog"}]},
			{"Name" : "LSTM_R_ifog2", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "29", "SubInstance" : "grp_lstm_forward_unidir_fu_604", "Port" : "R_ifog"}]},
			{"Name" : "LSTM_b_ifog2", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "29", "SubInstance" : "grp_lstm_forward_unidir_fu_604", "Port" : "b_ifog"}]},
			{"Name" : "BN2_var2", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "BN2_gamma2", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "X_slice3", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "63", "SubInstance" : "grp_conv_bn_act_pool_1_fu_641", "Port" : "X_slice3"}]},
			{"Name" : "ConvW3", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "63", "SubInstance" : "grp_conv_bn_act_pool_1_fu_641", "Port" : "ConvW3"}]},
			{"Name" : "LSTM_W_ifog3", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "29", "SubInstance" : "grp_lstm_forward_unidir_fu_604", "Port" : "W_ifog"}]},
			{"Name" : "LSTM_R_ifog3", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "29", "SubInstance" : "grp_lstm_forward_unidir_fu_604", "Port" : "R_ifog"}]},
			{"Name" : "LSTM_b_ifog3", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "29", "SubInstance" : "grp_lstm_forward_unidir_fu_604", "Port" : "b_ifog"}]},
			{"Name" : "BN2_var3", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "BN2_gamma3", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "tokens4", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "123", "SubInstance" : "grp_embedding_forward_dy_fu_681", "Port" : "tokens4"}]},
			{"Name" : "Emb4_0", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "123", "SubInstance" : "grp_embedding_forward_dy_fu_681", "Port" : "Emb4_0"}]},
			{"Name" : "Emb4_1", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "123", "SubInstance" : "grp_embedding_forward_dy_fu_681", "Port" : "Emb4_1"}]},
			{"Name" : "Emb4_2", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "123", "SubInstance" : "grp_embedding_forward_dy_fu_681", "Port" : "Emb4_2"}]},
			{"Name" : "Emb4_3", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "123", "SubInstance" : "grp_embedding_forward_dy_fu_681", "Port" : "Emb4_3"}]},
			{"Name" : "Emb4_4", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "123", "SubInstance" : "grp_embedding_forward_dy_fu_681", "Port" : "Emb4_4"}]},
			{"Name" : "Emb4_5", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "123", "SubInstance" : "grp_embedding_forward_dy_fu_681", "Port" : "Emb4_5"}]},
			{"Name" : "Emb4_6", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "123", "SubInstance" : "grp_embedding_forward_dy_fu_681", "Port" : "Emb4_6"}]},
			{"Name" : "Emb4_7", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "123", "SubInstance" : "grp_embedding_forward_dy_fu_681", "Port" : "Emb4_7"}]},
			{"Name" : "Emb4_8", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "123", "SubInstance" : "grp_embedding_forward_dy_fu_681", "Port" : "Emb4_8"}]},
			{"Name" : "Emb4_9", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "123", "SubInstance" : "grp_embedding_forward_dy_fu_681", "Port" : "Emb4_9"}]},
			{"Name" : "Emb4_10", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "123", "SubInstance" : "grp_embedding_forward_dy_fu_681", "Port" : "Emb4_10"}]},
			{"Name" : "Emb4_11", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "123", "SubInstance" : "grp_embedding_forward_dy_fu_681", "Port" : "Emb4_11"}]},
			{"Name" : "Emb4_12", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "123", "SubInstance" : "grp_embedding_forward_dy_fu_681", "Port" : "Emb4_12"}]},
			{"Name" : "Emb4_13", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "123", "SubInstance" : "grp_embedding_forward_dy_fu_681", "Port" : "Emb4_13"}]},
			{"Name" : "Emb4_14", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "123", "SubInstance" : "grp_embedding_forward_dy_fu_681", "Port" : "Emb4_14"}]},
			{"Name" : "Emb4_15", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "123", "SubInstance" : "grp_embedding_forward_dy_fu_681", "Port" : "Emb4_15"}]},
			{"Name" : "Emb4_16", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "123", "SubInstance" : "grp_embedding_forward_dy_fu_681", "Port" : "Emb4_16"}]},
			{"Name" : "Emb4_17", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "123", "SubInstance" : "grp_embedding_forward_dy_fu_681", "Port" : "Emb4_17"}]},
			{"Name" : "Emb4_18", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "123", "SubInstance" : "grp_embedding_forward_dy_fu_681", "Port" : "Emb4_18"}]},
			{"Name" : "ConvW4", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "49", "SubInstance" : "grp_conv_bn_act_pool_fu_632", "Port" : "ConvW4"}]},
			{"Name" : "LSTM_W_ifog4", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "29", "SubInstance" : "grp_lstm_forward_unidir_fu_604", "Port" : "W_ifog"}]},
			{"Name" : "LSTM_R_ifog4", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "29", "SubInstance" : "grp_lstm_forward_unidir_fu_604", "Port" : "R_ifog"}]},
			{"Name" : "LSTM_b_ifog4", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "29", "SubInstance" : "grp_lstm_forward_unidir_fu_604", "Port" : "b_ifog"}]},
			{"Name" : "BN2_var4", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "BN2_gamma4", "Type" : "Memory", "Direction" : "I"}]},
	{"ID" : "1", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.U_slice_U", "Parent" : "0"},
	{"ID" : "2", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.LSTM_W_ifog0_U", "Parent" : "0"},
	{"ID" : "3", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.LSTM_R_ifog0_U", "Parent" : "0"},
	{"ID" : "4", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.LSTM_b_ifog0_U", "Parent" : "0"},
	{"ID" : "5", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.h_slice_U", "Parent" : "0"},
	{"ID" : "6", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.BN2_var0_U", "Parent" : "0"},
	{"ID" : "7", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.BN2_gamma0_U", "Parent" : "0"},
	{"ID" : "8", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.LSTM_W_ifog1_U", "Parent" : "0"},
	{"ID" : "9", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.LSTM_R_ifog1_U", "Parent" : "0"},
	{"ID" : "10", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.LSTM_b_ifog1_U", "Parent" : "0"},
	{"ID" : "11", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.BN2_var1_U", "Parent" : "0"},
	{"ID" : "12", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.BN2_gamma1_U", "Parent" : "0"},
	{"ID" : "13", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.LSTM_W_ifog2_U", "Parent" : "0"},
	{"ID" : "14", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.LSTM_R_ifog2_U", "Parent" : "0"},
	{"ID" : "15", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.LSTM_b_ifog2_U", "Parent" : "0"},
	{"ID" : "16", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.BN2_var2_U", "Parent" : "0"},
	{"ID" : "17", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.BN2_gamma2_U", "Parent" : "0"},
	{"ID" : "18", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.LSTM_W_ifog3_U", "Parent" : "0"},
	{"ID" : "19", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.LSTM_R_ifog3_U", "Parent" : "0"},
	{"ID" : "20", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.LSTM_b_ifog3_U", "Parent" : "0"},
	{"ID" : "21", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.BN2_var3_U", "Parent" : "0"},
	{"ID" : "22", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.BN2_gamma3_U", "Parent" : "0"},
	{"ID" : "23", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.LSTM_W_ifog4_U", "Parent" : "0"},
	{"ID" : "24", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.LSTM_R_ifog4_U", "Parent" : "0"},
	{"ID" : "25", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.LSTM_b_ifog4_U", "Parent" : "0"},
	{"ID" : "26", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.BN2_var4_U", "Parent" : "0"},
	{"ID" : "27", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.BN2_gamma4_U", "Parent" : "0"},
	{"ID" : "28", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.X_slice_1_U", "Parent" : "0"},
	{"ID" : "29", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.grp_lstm_forward_unidir_fu_604", "Parent" : "0", "Child" : ["30", "31", "32", "33", "34", "35", "36", "37", "38", "39", "40", "41", "42", "43", "44", "45", "46", "47", "48"],
		"CDFG" : "lstm_forward_unidir",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "0", "ap_idle" : "1",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "1107166", "EstimateLatencyMax" : "1107166",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"Port" : [
			{"Name" : "W_ifog", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "R_ifog", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "b_ifog", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "h_last", "Type" : "Memory", "Direction" : "IO"},
			{"Name" : "c_slice", "Type" : "Memory", "Direction" : "IO"},
			{"Name" : "U_slice", "Type" : "Memory", "Direction" : "I"}]},
	{"ID" : "30", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_lstm_forward_unidir_fu_604.c_slice_U", "Parent" : "29"},
	{"ID" : "31", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_lstm_forward_unidir_fu_604.z_U", "Parent" : "29"},
	{"ID" : "32", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_lstm_forward_unidir_fu_604.main_fadd_32ns_32ns_32_5_full_dsp_1_U25", "Parent" : "29"},
	{"ID" : "33", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_lstm_forward_unidir_fu_604.main_fadd_32ns_32ns_32_5_full_dsp_1_U26", "Parent" : "29"},
	{"ID" : "34", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_lstm_forward_unidir_fu_604.main_fdiv_32ns_32ns_32_16_1_U27", "Parent" : "29"},
	{"ID" : "35", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_lstm_forward_unidir_fu_604.main_fdiv_32ns_32ns_32_16_1_U28", "Parent" : "29"},
	{"ID" : "36", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_lstm_forward_unidir_fu_604.main_fexp_32ns_32ns_32_9_full_dsp_1_U29", "Parent" : "29"},
	{"ID" : "37", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_lstm_forward_unidir_fu_604.main_fexp_32ns_32ns_32_9_full_dsp_1_U30", "Parent" : "29"},
	{"ID" : "38", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_lstm_forward_unidir_fu_604.main_sptohp_32ns_16_2_1_U31", "Parent" : "29"},
	{"ID" : "39", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_lstm_forward_unidir_fu_604.main_sptohp_32ns_16_2_1_U32", "Parent" : "29"},
	{"ID" : "40", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_lstm_forward_unidir_fu_604.main_hptosp_16ns_32_2_1_U33", "Parent" : "29"},
	{"ID" : "41", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_lstm_forward_unidir_fu_604.main_hptosp_16ns_32_2_1_U34", "Parent" : "29"},
	{"ID" : "42", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_lstm_forward_unidir_fu_604.main_hadd_16ns_16ns_16_5_full_dsp_1_U35", "Parent" : "29"},
	{"ID" : "43", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_lstm_forward_unidir_fu_604.main_hsub_16ns_16ns_16_5_full_dsp_1_U36", "Parent" : "29"},
	{"ID" : "44", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_lstm_forward_unidir_fu_604.main_hsub_16ns_16ns_16_5_full_dsp_1_U37", "Parent" : "29"},
	{"ID" : "45", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_lstm_forward_unidir_fu_604.main_hmul_16ns_16ns_16_4_max_dsp_1_U38", "Parent" : "29"},
	{"ID" : "46", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_lstm_forward_unidir_fu_604.main_hmul_16ns_16ns_16_4_max_dsp_1_U39", "Parent" : "29"},
	{"ID" : "47", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_lstm_forward_unidir_fu_604.main_hcmp_16ns_16ns_1_2_1_U40", "Parent" : "29"},
	{"ID" : "48", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_lstm_forward_unidir_fu_604.main_hcmp_16ns_16ns_1_2_1_U41", "Parent" : "29"},
	{"ID" : "49", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.grp_conv_bn_act_pool_fu_632", "Parent" : "0", "Child" : ["50", "51", "52", "53", "54", "55", "56", "57", "58", "59", "60", "61", "62"],
		"CDFG" : "conv_bn_act_pool",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "0", "ap_idle" : "1",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "21843714", "EstimateLatencyMax" : "22028610",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"Port" : [
			{"Name" : "X", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "U", "Type" : "Memory", "Direction" : "O"},
			{"Name" : "ConvW4", "Type" : "Memory", "Direction" : "I"}]},
	{"ID" : "50", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_bn_act_pool_fu_632.ConvW4_U", "Parent" : "49"},
	{"ID" : "51", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_bn_act_pool_fu_632.main_fadd_32ns_32ns_32_5_full_dsp_1_U118", "Parent" : "49"},
	{"ID" : "52", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_bn_act_pool_fu_632.main_fmul_32ns_32ns_32_4_max_dsp_1_U119", "Parent" : "49"},
	{"ID" : "53", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_bn_act_pool_fu_632.main_sptohp_32ns_16_2_1_U120", "Parent" : "49"},
	{"ID" : "54", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_bn_act_pool_fu_632.main_hptosp_16ns_32_2_1_U121", "Parent" : "49"},
	{"ID" : "55", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_bn_act_pool_fu_632.main_hptosp_16ns_32_2_1_U122", "Parent" : "49"},
	{"ID" : "56", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_bn_act_pool_fu_632.main_hadd_16ns_16ns_16_5_full_dsp_1_U123", "Parent" : "49"},
	{"ID" : "57", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_bn_act_pool_fu_632.main_hmul_16ns_16ns_16_4_max_dsp_1_U124", "Parent" : "49"},
	{"ID" : "58", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_bn_act_pool_fu_632.main_hdiv_16ns_16ns_16_7_1_U125", "Parent" : "49"},
	{"ID" : "59", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_bn_act_pool_fu_632.main_hcmp_16ns_16ns_1_2_1_U126", "Parent" : "49"},
	{"ID" : "60", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_bn_act_pool_fu_632.main_mux_325_16_1_1_U127", "Parent" : "49"},
	{"ID" : "61", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_bn_act_pool_fu_632.main_mux_325_16_1_1_U128", "Parent" : "49"},
	{"ID" : "62", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_bn_act_pool_fu_632.main_mux_325_16_1_1_U129", "Parent" : "49"},
	{"ID" : "63", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.grp_conv_bn_act_pool_1_fu_641", "Parent" : "0", "Child" : ["64", "65", "66", "67", "68", "69", "70", "71", "72", "73", "74", "75", "76", "77"],
		"CDFG" : "conv_bn_act_pool_1",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "0", "ap_idle" : "1",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "10921890", "EstimateLatencyMax" : "11014338",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"Port" : [
			{"Name" : "U", "Type" : "Memory", "Direction" : "O"},
			{"Name" : "X_slice3", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "ConvW3", "Type" : "Memory", "Direction" : "I"}]},
	{"ID" : "64", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_bn_act_pool_1_fu_641.X_slice3_U", "Parent" : "63"},
	{"ID" : "65", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_bn_act_pool_1_fu_641.ConvW3_U", "Parent" : "63"},
	{"ID" : "66", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_bn_act_pool_1_fu_641.main_fadd_32ns_32ns_32_5_full_dsp_1_U82", "Parent" : "63"},
	{"ID" : "67", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_bn_act_pool_1_fu_641.main_fmul_32ns_32ns_32_4_max_dsp_1_U83", "Parent" : "63"},
	{"ID" : "68", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_bn_act_pool_1_fu_641.main_sptohp_32ns_16_2_1_U84", "Parent" : "63"},
	{"ID" : "69", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_bn_act_pool_1_fu_641.main_hptosp_16ns_32_2_1_U85", "Parent" : "63"},
	{"ID" : "70", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_bn_act_pool_1_fu_641.main_hptosp_16ns_32_2_1_U86", "Parent" : "63"},
	{"ID" : "71", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_bn_act_pool_1_fu_641.main_hadd_16ns_16ns_16_5_full_dsp_1_U87", "Parent" : "63"},
	{"ID" : "72", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_bn_act_pool_1_fu_641.main_hmul_16ns_16ns_16_4_max_dsp_1_U88", "Parent" : "63"},
	{"ID" : "73", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_bn_act_pool_1_fu_641.main_hdiv_16ns_16ns_16_7_1_U89", "Parent" : "63"},
	{"ID" : "74", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_bn_act_pool_1_fu_641.main_hcmp_16ns_16ns_1_2_1_U90", "Parent" : "63"},
	{"ID" : "75", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_bn_act_pool_1_fu_641.main_mux_325_16_1_1_U91", "Parent" : "63"},
	{"ID" : "76", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_bn_act_pool_1_fu_641.main_mux_325_16_1_1_U92", "Parent" : "63"},
	{"ID" : "77", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_bn_act_pool_1_fu_641.main_mux_325_16_1_1_U93", "Parent" : "63"},
	{"ID" : "78", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.grp_conv_bn_act_pool_2_fu_651", "Parent" : "0", "Child" : ["79", "80", "81", "82", "83", "84", "85", "86", "87", "88", "89", "90", "91", "92"],
		"CDFG" : "conv_bn_act_pool_2",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "0", "ap_idle" : "1",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "5460978", "EstimateLatencyMax" : "5507202",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"Port" : [
			{"Name" : "U", "Type" : "Memory", "Direction" : "O"},
			{"Name" : "X_slice2", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "ConvW2", "Type" : "Memory", "Direction" : "I"}]},
	{"ID" : "79", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_bn_act_pool_2_fu_651.X_slice2_U", "Parent" : "78"},
	{"ID" : "80", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_bn_act_pool_2_fu_651.ConvW2_U", "Parent" : "78"},
	{"ID" : "81", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_bn_act_pool_2_fu_651.main_fadd_32ns_32ns_32_5_full_dsp_1_U67", "Parent" : "78"},
	{"ID" : "82", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_bn_act_pool_2_fu_651.main_fmul_32ns_32ns_32_4_max_dsp_1_U68", "Parent" : "78"},
	{"ID" : "83", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_bn_act_pool_2_fu_651.main_sptohp_32ns_16_2_1_U69", "Parent" : "78"},
	{"ID" : "84", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_bn_act_pool_2_fu_651.main_hptosp_16ns_32_2_1_U70", "Parent" : "78"},
	{"ID" : "85", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_bn_act_pool_2_fu_651.main_hptosp_16ns_32_2_1_U71", "Parent" : "78"},
	{"ID" : "86", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_bn_act_pool_2_fu_651.main_hadd_16ns_16ns_16_5_full_dsp_1_U72", "Parent" : "78"},
	{"ID" : "87", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_bn_act_pool_2_fu_651.main_hmul_16ns_16ns_16_4_max_dsp_1_U73", "Parent" : "78"},
	{"ID" : "88", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_bn_act_pool_2_fu_651.main_hdiv_16ns_16ns_16_7_1_U74", "Parent" : "78"},
	{"ID" : "89", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_bn_act_pool_2_fu_651.main_hcmp_16ns_16ns_1_2_1_U75", "Parent" : "78"},
	{"ID" : "90", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_bn_act_pool_2_fu_651.main_mux_325_16_1_1_U76", "Parent" : "78"},
	{"ID" : "91", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_bn_act_pool_2_fu_651.main_mux_325_16_1_1_U77", "Parent" : "78"},
	{"ID" : "92", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_bn_act_pool_2_fu_651.main_mux_325_16_1_1_U78", "Parent" : "78"},
	{"ID" : "93", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.grp_conv_bn_act_pool_3_fu_661", "Parent" : "0", "Child" : ["94", "95", "96", "97", "98", "99", "100", "101", "102", "103", "104", "105", "106", "107"],
		"CDFG" : "conv_bn_act_pool_3",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "0", "ap_idle" : "1",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "2730522", "EstimateLatencyMax" : "2753634",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"Port" : [
			{"Name" : "U", "Type" : "Memory", "Direction" : "O"},
			{"Name" : "X_slice1", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "ConvW1", "Type" : "Memory", "Direction" : "I"}]},
	{"ID" : "94", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_bn_act_pool_3_fu_661.X_slice1_U", "Parent" : "93"},
	{"ID" : "95", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_bn_act_pool_3_fu_661.ConvW1_U", "Parent" : "93"},
	{"ID" : "96", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_bn_act_pool_3_fu_661.main_fadd_32ns_32ns_32_5_full_dsp_1_U52", "Parent" : "93"},
	{"ID" : "97", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_bn_act_pool_3_fu_661.main_fmul_32ns_32ns_32_4_max_dsp_1_U53", "Parent" : "93"},
	{"ID" : "98", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_bn_act_pool_3_fu_661.main_sptohp_32ns_16_2_1_U54", "Parent" : "93"},
	{"ID" : "99", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_bn_act_pool_3_fu_661.main_hptosp_16ns_32_2_1_U55", "Parent" : "93"},
	{"ID" : "100", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_bn_act_pool_3_fu_661.main_hptosp_16ns_32_2_1_U56", "Parent" : "93"},
	{"ID" : "101", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_bn_act_pool_3_fu_661.main_hadd_16ns_16ns_16_5_full_dsp_1_U57", "Parent" : "93"},
	{"ID" : "102", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_bn_act_pool_3_fu_661.main_hmul_16ns_16ns_16_4_max_dsp_1_U58", "Parent" : "93"},
	{"ID" : "103", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_bn_act_pool_3_fu_661.main_hdiv_16ns_16ns_16_7_1_U59", "Parent" : "93"},
	{"ID" : "104", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_bn_act_pool_3_fu_661.main_hcmp_16ns_16ns_1_2_1_U60", "Parent" : "93"},
	{"ID" : "105", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_bn_act_pool_3_fu_661.main_mux_325_16_1_1_U61", "Parent" : "93"},
	{"ID" : "106", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_bn_act_pool_3_fu_661.main_mux_325_16_1_1_U62", "Parent" : "93"},
	{"ID" : "107", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_bn_act_pool_3_fu_661.main_mux_325_16_1_1_U63", "Parent" : "93"},
	{"ID" : "108", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.grp_conv_bn_act_pool_4_fu_671", "Parent" : "0", "Child" : ["109", "110", "111", "112", "113", "114", "115", "116", "117", "118", "119", "120", "121", "122"],
		"CDFG" : "conv_bn_act_pool_4",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "0", "ap_idle" : "1",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "1365294", "EstimateLatencyMax" : "1376850",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"Port" : [
			{"Name" : "U", "Type" : "Memory", "Direction" : "O"},
			{"Name" : "X_slice", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "ConvW0", "Type" : "Memory", "Direction" : "I"}]},
	{"ID" : "109", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_bn_act_pool_4_fu_671.X_slice_U", "Parent" : "108"},
	{"ID" : "110", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_bn_act_pool_4_fu_671.ConvW0_U", "Parent" : "108"},
	{"ID" : "111", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_bn_act_pool_4_fu_671.main_fadd_32ns_32ns_32_5_full_dsp_1_U1", "Parent" : "108"},
	{"ID" : "112", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_bn_act_pool_4_fu_671.main_fmul_32ns_32ns_32_4_max_dsp_1_U2", "Parent" : "108"},
	{"ID" : "113", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_bn_act_pool_4_fu_671.main_sptohp_32ns_16_2_1_U3", "Parent" : "108"},
	{"ID" : "114", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_bn_act_pool_4_fu_671.main_hptosp_16ns_32_2_1_U4", "Parent" : "108"},
	{"ID" : "115", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_bn_act_pool_4_fu_671.main_hptosp_16ns_32_2_1_U5", "Parent" : "108"},
	{"ID" : "116", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_bn_act_pool_4_fu_671.main_hadd_16ns_16ns_16_5_full_dsp_1_U6", "Parent" : "108"},
	{"ID" : "117", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_bn_act_pool_4_fu_671.main_hmul_16ns_16ns_16_4_max_dsp_1_U7", "Parent" : "108"},
	{"ID" : "118", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_bn_act_pool_4_fu_671.main_hdiv_16ns_16ns_16_7_1_U8", "Parent" : "108"},
	{"ID" : "119", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_bn_act_pool_4_fu_671.main_hcmp_16ns_16ns_1_2_1_U9", "Parent" : "108"},
	{"ID" : "120", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_bn_act_pool_4_fu_671.main_mux_325_16_1_1_U10", "Parent" : "108"},
	{"ID" : "121", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_bn_act_pool_4_fu_671.main_mux_325_16_1_1_U11", "Parent" : "108"},
	{"ID" : "122", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_bn_act_pool_4_fu_671.main_mux_325_16_1_1_U12", "Parent" : "108"},
	{"ID" : "123", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.grp_embedding_forward_dy_fu_681", "Parent" : "0", "Child" : ["124", "125", "126", "127", "128", "129", "130", "131", "132", "133", "134", "135", "136", "137", "138", "139", "140", "141", "142", "143"],
		"CDFG" : "embedding_forward_dy",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "0", "ap_idle" : "1",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "57619", "EstimateLatencyMax" : "57619",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"Port" : [
			{"Name" : "X", "Type" : "Memory", "Direction" : "O"},
			{"Name" : "tokens4", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "Emb4_0", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "Emb4_1", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "Emb4_2", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "Emb4_3", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "Emb4_4", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "Emb4_5", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "Emb4_6", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "Emb4_7", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "Emb4_8", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "Emb4_9", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "Emb4_10", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "Emb4_11", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "Emb4_12", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "Emb4_13", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "Emb4_14", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "Emb4_15", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "Emb4_16", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "Emb4_17", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "Emb4_18", "Type" : "Memory", "Direction" : "I"}]},
	{"ID" : "124", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_embedding_forward_dy_fu_681.tokens4_U", "Parent" : "123"},
	{"ID" : "125", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_embedding_forward_dy_fu_681.Emb4_0_U", "Parent" : "123"},
	{"ID" : "126", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_embedding_forward_dy_fu_681.Emb4_1_U", "Parent" : "123"},
	{"ID" : "127", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_embedding_forward_dy_fu_681.Emb4_2_U", "Parent" : "123"},
	{"ID" : "128", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_embedding_forward_dy_fu_681.Emb4_3_U", "Parent" : "123"},
	{"ID" : "129", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_embedding_forward_dy_fu_681.Emb4_4_U", "Parent" : "123"},
	{"ID" : "130", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_embedding_forward_dy_fu_681.Emb4_5_U", "Parent" : "123"},
	{"ID" : "131", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_embedding_forward_dy_fu_681.Emb4_6_U", "Parent" : "123"},
	{"ID" : "132", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_embedding_forward_dy_fu_681.Emb4_7_U", "Parent" : "123"},
	{"ID" : "133", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_embedding_forward_dy_fu_681.Emb4_8_U", "Parent" : "123"},
	{"ID" : "134", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_embedding_forward_dy_fu_681.Emb4_9_U", "Parent" : "123"},
	{"ID" : "135", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_embedding_forward_dy_fu_681.Emb4_10_U", "Parent" : "123"},
	{"ID" : "136", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_embedding_forward_dy_fu_681.Emb4_11_U", "Parent" : "123"},
	{"ID" : "137", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_embedding_forward_dy_fu_681.Emb4_12_U", "Parent" : "123"},
	{"ID" : "138", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_embedding_forward_dy_fu_681.Emb4_13_U", "Parent" : "123"},
	{"ID" : "139", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_embedding_forward_dy_fu_681.Emb4_14_U", "Parent" : "123"},
	{"ID" : "140", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_embedding_forward_dy_fu_681.Emb4_15_U", "Parent" : "123"},
	{"ID" : "141", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_embedding_forward_dy_fu_681.Emb4_16_U", "Parent" : "123"},
	{"ID" : "142", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_embedding_forward_dy_fu_681.Emb4_17_U", "Parent" : "123"},
	{"ID" : "143", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_embedding_forward_dy_fu_681.Emb4_18_U", "Parent" : "123"},
	{"ID" : "144", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.main_fdiv_32ns_32ns_32_16_1_U133", "Parent" : "0"},
	{"ID" : "145", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.main_fsqrt_32ns_32ns_32_12_1_U134", "Parent" : "0"},
	{"ID" : "146", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.main_sptohp_32ns_16_2_1_U135", "Parent" : "0"},
	{"ID" : "147", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.main_hptosp_16ns_32_2_1_U136", "Parent" : "0"},
	{"ID" : "148", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.main_hadd_16ns_16ns_16_5_full_dsp_1_U137", "Parent" : "0"},
	{"ID" : "149", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.main_hmul_16ns_16ns_16_4_max_dsp_1_U138", "Parent" : "0"},
	{"ID" : "150", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.main_hcmp_16ns_16ns_1_2_1_U139", "Parent" : "0"},
	{"ID" : "151", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.main_hcmp_16ns_16ns_1_2_1_U140", "Parent" : "0"}]}


set ArgLastReadFirstWriteLatency {
	run_all_slices_unrol {
		merged {Type IO LastRead 29 FirstWrite 1}
		X_slice {Type I LastRead -1 FirstWrite -1}
		ConvW0 {Type I LastRead -1 FirstWrite -1}
		U_slice {Type IO LastRead -1 FirstWrite -1}
		c_slice {Type IO LastRead -1 FirstWrite -1}
		LSTM_W_ifog0 {Type I LastRead -1 FirstWrite -1}
		LSTM_R_ifog0 {Type I LastRead -1 FirstWrite -1}
		LSTM_b_ifog0 {Type I LastRead -1 FirstWrite -1}
		h_slice {Type IO LastRead -1 FirstWrite -1}
		BN2_var0 {Type I LastRead -1 FirstWrite -1}
		BN2_gamma0 {Type I LastRead -1 FirstWrite -1}
		X_slice1 {Type I LastRead -1 FirstWrite -1}
		ConvW1 {Type I LastRead -1 FirstWrite -1}
		LSTM_W_ifog1 {Type I LastRead -1 FirstWrite -1}
		LSTM_R_ifog1 {Type I LastRead -1 FirstWrite -1}
		LSTM_b_ifog1 {Type I LastRead -1 FirstWrite -1}
		BN2_var1 {Type I LastRead -1 FirstWrite -1}
		BN2_gamma1 {Type I LastRead -1 FirstWrite -1}
		X_slice2 {Type I LastRead -1 FirstWrite -1}
		ConvW2 {Type I LastRead -1 FirstWrite -1}
		LSTM_W_ifog2 {Type I LastRead -1 FirstWrite -1}
		LSTM_R_ifog2 {Type I LastRead -1 FirstWrite -1}
		LSTM_b_ifog2 {Type I LastRead -1 FirstWrite -1}
		BN2_var2 {Type I LastRead -1 FirstWrite -1}
		BN2_gamma2 {Type I LastRead -1 FirstWrite -1}
		X_slice3 {Type I LastRead -1 FirstWrite -1}
		ConvW3 {Type I LastRead -1 FirstWrite -1}
		LSTM_W_ifog3 {Type I LastRead -1 FirstWrite -1}
		LSTM_R_ifog3 {Type I LastRead -1 FirstWrite -1}
		LSTM_b_ifog3 {Type I LastRead -1 FirstWrite -1}
		BN2_var3 {Type I LastRead -1 FirstWrite -1}
		BN2_gamma3 {Type I LastRead -1 FirstWrite -1}
		tokens4 {Type I LastRead -1 FirstWrite -1}
		Emb4_0 {Type I LastRead -1 FirstWrite -1}
		Emb4_1 {Type I LastRead -1 FirstWrite -1}
		Emb4_2 {Type I LastRead -1 FirstWrite -1}
		Emb4_3 {Type I LastRead -1 FirstWrite -1}
		Emb4_4 {Type I LastRead -1 FirstWrite -1}
		Emb4_5 {Type I LastRead -1 FirstWrite -1}
		Emb4_6 {Type I LastRead -1 FirstWrite -1}
		Emb4_7 {Type I LastRead -1 FirstWrite -1}
		Emb4_8 {Type I LastRead -1 FirstWrite -1}
		Emb4_9 {Type I LastRead -1 FirstWrite -1}
		Emb4_10 {Type I LastRead -1 FirstWrite -1}
		Emb4_11 {Type I LastRead -1 FirstWrite -1}
		Emb4_12 {Type I LastRead -1 FirstWrite -1}
		Emb4_13 {Type I LastRead -1 FirstWrite -1}
		Emb4_14 {Type I LastRead -1 FirstWrite -1}
		Emb4_15 {Type I LastRead -1 FirstWrite -1}
		Emb4_16 {Type I LastRead -1 FirstWrite -1}
		Emb4_17 {Type I LastRead -1 FirstWrite -1}
		Emb4_18 {Type I LastRead -1 FirstWrite -1}
		ConvW4 {Type I LastRead -1 FirstWrite -1}
		LSTM_W_ifog4 {Type I LastRead -1 FirstWrite -1}
		LSTM_R_ifog4 {Type I LastRead -1 FirstWrite -1}
		LSTM_b_ifog4 {Type I LastRead -1 FirstWrite -1}
		BN2_var4 {Type I LastRead -1 FirstWrite -1}
		BN2_gamma4 {Type I LastRead -1 FirstWrite -1}}
	lstm_forward_unidir {
		W_ifog {Type I LastRead 7 FirstWrite -1}
		R_ifog {Type I LastRead 8 FirstWrite -1}
		b_ifog {Type I LastRead 3 FirstWrite -1}
		h_last {Type IO LastRead 5 FirstWrite 1}
		c_slice {Type IO LastRead -1 FirstWrite -1}
		U_slice {Type I LastRead 4 FirstWrite -1}}
	conv_bn_act_pool {
		X {Type I LastRead 155 FirstWrite -1}
		U {Type O LastRead -1 FirstWrite 12}
		ConvW4 {Type I LastRead -1 FirstWrite -1}}
	conv_bn_act_pool_1 {
		U {Type O LastRead -1 FirstWrite 12}
		X_slice3 {Type I LastRead -1 FirstWrite -1}
		ConvW3 {Type I LastRead -1 FirstWrite -1}}
	conv_bn_act_pool_2 {
		U {Type O LastRead -1 FirstWrite 12}
		X_slice2 {Type I LastRead -1 FirstWrite -1}
		ConvW2 {Type I LastRead -1 FirstWrite -1}}
	conv_bn_act_pool_3 {
		U {Type O LastRead -1 FirstWrite 12}
		X_slice1 {Type I LastRead -1 FirstWrite -1}
		ConvW1 {Type I LastRead -1 FirstWrite -1}}
	conv_bn_act_pool_4 {
		U {Type O LastRead -1 FirstWrite 12}
		X_slice {Type I LastRead -1 FirstWrite -1}
		ConvW0 {Type I LastRead -1 FirstWrite -1}}
	embedding_forward_dy {
		X {Type O LastRead -1 FirstWrite 5}
		tokens4 {Type I LastRead -1 FirstWrite -1}
		Emb4_0 {Type I LastRead -1 FirstWrite -1}
		Emb4_1 {Type I LastRead -1 FirstWrite -1}
		Emb4_2 {Type I LastRead -1 FirstWrite -1}
		Emb4_3 {Type I LastRead -1 FirstWrite -1}
		Emb4_4 {Type I LastRead -1 FirstWrite -1}
		Emb4_5 {Type I LastRead -1 FirstWrite -1}
		Emb4_6 {Type I LastRead -1 FirstWrite -1}
		Emb4_7 {Type I LastRead -1 FirstWrite -1}
		Emb4_8 {Type I LastRead -1 FirstWrite -1}
		Emb4_9 {Type I LastRead -1 FirstWrite -1}
		Emb4_10 {Type I LastRead -1 FirstWrite -1}
		Emb4_11 {Type I LastRead -1 FirstWrite -1}
		Emb4_12 {Type I LastRead -1 FirstWrite -1}
		Emb4_13 {Type I LastRead -1 FirstWrite -1}
		Emb4_14 {Type I LastRead -1 FirstWrite -1}
		Emb4_15 {Type I LastRead -1 FirstWrite -1}
		Emb4_16 {Type I LastRead -1 FirstWrite -1}
		Emb4_17 {Type I LastRead -1 FirstWrite -1}
		Emb4_18 {Type I LastRead -1 FirstWrite -1}}}

set hasDtUnsupportedChannel 0

set PerformanceInfo {[
	{"Name" : "Latency", "Min" : "47925187", "Max" : "48283423"}
	, {"Name" : "Interval", "Min" : "47925187", "Max" : "48283423"}
]}

set PipelineEnableSignalInfo {[
]}

set Spec2ImplPortList { 
	merged { ap_memory {  { merged_address0 mem_address 1 5 }  { merged_ce0 mem_ce 1 1 }  { merged_we0 mem_we 1 1 }  { merged_d0 mem_din 1 16 }  { merged_q0 mem_dout 0 16 } } }
}
