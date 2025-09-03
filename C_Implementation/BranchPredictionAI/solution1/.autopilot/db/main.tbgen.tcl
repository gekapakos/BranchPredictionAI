set moduleName main
set isTopModule 1
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
set C_modelName {main}
set C_modelType { int 32 }
set C_modelArgList {
}
set C_modelArgMapList {[ 
	{ "Name" : "ap_return", "interface" : "wire", "bitwidth" : 32,"bitSlice":[{"low":0,"up":31,"cElement": [{"cName": "return","cData": "int","bit_use": { "low": 0,"up": 31},"cArray": [{"low" : 0,"up" : 1,"step" : 0}]}]}]} ]}
# RTL Port declarations: 
set portNum 7
set portList { 
	{ ap_clk sc_in sc_logic 1 clock -1 } 
	{ ap_rst sc_in sc_logic 1 reset -1 active_high_sync } 
	{ ap_start sc_in sc_logic 1 start -1 } 
	{ ap_done sc_out sc_logic 1 predone -1 } 
	{ ap_idle sc_out sc_logic 1 done -1 } 
	{ ap_ready sc_out sc_logic 1 ready -1 } 
	{ ap_return sc_out sc_lv 32 signal -1 } 
}
set NewPortList {[ 
	{ "name": "ap_clk", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "clock", "bundle":{"name": "ap_clk", "role": "default" }} , 
 	{ "name": "ap_rst", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "reset", "bundle":{"name": "ap_rst", "role": "default" }} , 
 	{ "name": "ap_start", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "start", "bundle":{"name": "ap_start", "role": "default" }} , 
 	{ "name": "ap_done", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "predone", "bundle":{"name": "ap_done", "role": "default" }} , 
 	{ "name": "ap_idle", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "done", "bundle":{"name": "ap_idle", "role": "default" }} , 
 	{ "name": "ap_ready", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "ready", "bundle":{"name": "ap_ready", "role": "default" }} , 
 	{ "name": "ap_return", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "ap_return", "role": "default" }}  ]}

set RtlHierarchyInfo {[
	{"ID" : "0", "Level" : "0", "Path" : "`AUTOTB_DUT_INST", "Parent" : "", "Child" : ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "138", "139", "140", "141", "142"],
		"CDFG" : "main",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "0", "ap_idle" : "1",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "21006103", "EstimateLatencyMax" : "541989483",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"WaitState" : [
			{"State" : "ap_ST_fsm_state2", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_run_all_slices_unrol_fu_485"}],
		"Port" : [
			{"Name" : "tokens0", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "10", "SubInstance" : "grp_run_all_slices_unrol_fu_485", "Port" : "tokens0"}]},
			{"Name" : "Emb0", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "10", "SubInstance" : "grp_run_all_slices_unrol_fu_485", "Port" : "Emb0"}]},
			{"Name" : "X_slice", "Type" : "Memory", "Direction" : "IO",
				"SubConnect" : [
					{"ID" : "10", "SubInstance" : "grp_run_all_slices_unrol_fu_485", "Port" : "X_slice"}]},
			{"Name" : "B_arg_index", "Type" : "Vld", "Direction" : "O",
				"SubConnect" : [
					{"ID" : "10", "SubInstance" : "grp_run_all_slices_unrol_fu_485", "Port" : "B_arg_index"}]},
			{"Name" : "Y", "Type" : "Memory", "Direction" : "IO",
				"SubConnect" : [
					{"ID" : "10", "SubInstance" : "grp_run_all_slices_unrol_fu_485", "Port" : "Y"}]},
			{"Name" : "ConvW0", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "10", "SubInstance" : "grp_run_all_slices_unrol_fu_485", "Port" : "ConvW0"}]},
			{"Name" : "ConvB0", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "10", "SubInstance" : "grp_run_all_slices_unrol_fu_485", "Port" : "ConvB0"}]},
			{"Name" : "BN1_gamma0", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "10", "SubInstance" : "grp_run_all_slices_unrol_fu_485", "Port" : "BN1_gamma0"}]},
			{"Name" : "BN1_beta0", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "10", "SubInstance" : "grp_run_all_slices_unrol_fu_485", "Port" : "BN1_beta0"}]},
			{"Name" : "BN1_mean0", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "10", "SubInstance" : "grp_run_all_slices_unrol_fu_485", "Port" : "BN1_mean0"}]},
			{"Name" : "BN1_var0", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "10", "SubInstance" : "grp_run_all_slices_unrol_fu_485", "Port" : "BN1_var0"}]},
			{"Name" : "U_slice", "Type" : "Memory", "Direction" : "IO",
				"SubConnect" : [
					{"ID" : "10", "SubInstance" : "grp_run_all_slices_unrol_fu_485", "Port" : "U_slice"}]},
			{"Name" : "c_slice", "Type" : "Memory", "Direction" : "IO",
				"SubConnect" : [
					{"ID" : "10", "SubInstance" : "grp_run_all_slices_unrol_fu_485", "Port" : "c_slice"}]},
			{"Name" : "table_exp_Z1_array_s", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "10", "SubInstance" : "grp_run_all_slices_unrol_fu_485", "Port" : "table_exp_Z1_array_s"}]},
			{"Name" : "table_f_Z3_array_V", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "10", "SubInstance" : "grp_run_all_slices_unrol_fu_485", "Port" : "table_f_Z3_array_V"}]},
			{"Name" : "table_f_Z2_array_V", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "10", "SubInstance" : "grp_run_all_slices_unrol_fu_485", "Port" : "table_f_Z2_array_V"}]},
			{"Name" : "LSTM_W_ifog0", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "10", "SubInstance" : "grp_run_all_slices_unrol_fu_485", "Port" : "LSTM_W_ifog0"}]},
			{"Name" : "LSTM_R_ifog0", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "10", "SubInstance" : "grp_run_all_slices_unrol_fu_485", "Port" : "LSTM_R_ifog0"}]},
			{"Name" : "LSTM_b_ifog0", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "10", "SubInstance" : "grp_run_all_slices_unrol_fu_485", "Port" : "LSTM_b_ifog0"}]},
			{"Name" : "h_slice", "Type" : "Memory", "Direction" : "IO",
				"SubConnect" : [
					{"ID" : "10", "SubInstance" : "grp_run_all_slices_unrol_fu_485", "Port" : "h_slice"}]},
			{"Name" : "BN2_var0", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "10", "SubInstance" : "grp_run_all_slices_unrol_fu_485", "Port" : "BN2_var0"}]},
			{"Name" : "BN2_gamma0", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "10", "SubInstance" : "grp_run_all_slices_unrol_fu_485", "Port" : "BN2_gamma0"}]},
			{"Name" : "tokens1", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "10", "SubInstance" : "grp_run_all_slices_unrol_fu_485", "Port" : "tokens1"}]},
			{"Name" : "Emb1", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "10", "SubInstance" : "grp_run_all_slices_unrol_fu_485", "Port" : "Emb1"}]},
			{"Name" : "ConvW1", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "10", "SubInstance" : "grp_run_all_slices_unrol_fu_485", "Port" : "ConvW1"}]},
			{"Name" : "ConvB1", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "10", "SubInstance" : "grp_run_all_slices_unrol_fu_485", "Port" : "ConvB1"}]},
			{"Name" : "BN1_gamma1", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "10", "SubInstance" : "grp_run_all_slices_unrol_fu_485", "Port" : "BN1_gamma1"}]},
			{"Name" : "BN1_beta1", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "10", "SubInstance" : "grp_run_all_slices_unrol_fu_485", "Port" : "BN1_beta1"}]},
			{"Name" : "BN1_mean1", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "10", "SubInstance" : "grp_run_all_slices_unrol_fu_485", "Port" : "BN1_mean1"}]},
			{"Name" : "BN1_var1", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "10", "SubInstance" : "grp_run_all_slices_unrol_fu_485", "Port" : "BN1_var1"}]},
			{"Name" : "LSTM_W_ifog1", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "10", "SubInstance" : "grp_run_all_slices_unrol_fu_485", "Port" : "LSTM_W_ifog1"}]},
			{"Name" : "LSTM_R_ifog1", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "10", "SubInstance" : "grp_run_all_slices_unrol_fu_485", "Port" : "LSTM_R_ifog1"}]},
			{"Name" : "LSTM_b_ifog1", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "10", "SubInstance" : "grp_run_all_slices_unrol_fu_485", "Port" : "LSTM_b_ifog1"}]},
			{"Name" : "BN2_var1", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "10", "SubInstance" : "grp_run_all_slices_unrol_fu_485", "Port" : "BN2_var1"}]},
			{"Name" : "BN2_gamma1", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "10", "SubInstance" : "grp_run_all_slices_unrol_fu_485", "Port" : "BN2_gamma1"}]},
			{"Name" : "tokens2", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "10", "SubInstance" : "grp_run_all_slices_unrol_fu_485", "Port" : "tokens2"}]},
			{"Name" : "Emb2", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "10", "SubInstance" : "grp_run_all_slices_unrol_fu_485", "Port" : "Emb2"}]},
			{"Name" : "ConvW2", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "10", "SubInstance" : "grp_run_all_slices_unrol_fu_485", "Port" : "ConvW2"}]},
			{"Name" : "ConvB2", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "10", "SubInstance" : "grp_run_all_slices_unrol_fu_485", "Port" : "ConvB2"}]},
			{"Name" : "BN1_gamma2", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "10", "SubInstance" : "grp_run_all_slices_unrol_fu_485", "Port" : "BN1_gamma2"}]},
			{"Name" : "BN1_beta2", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "10", "SubInstance" : "grp_run_all_slices_unrol_fu_485", "Port" : "BN1_beta2"}]},
			{"Name" : "BN1_mean2", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "10", "SubInstance" : "grp_run_all_slices_unrol_fu_485", "Port" : "BN1_mean2"}]},
			{"Name" : "BN1_var2", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "10", "SubInstance" : "grp_run_all_slices_unrol_fu_485", "Port" : "BN1_var2"}]},
			{"Name" : "LSTM_W_ifog2", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "10", "SubInstance" : "grp_run_all_slices_unrol_fu_485", "Port" : "LSTM_W_ifog2"}]},
			{"Name" : "LSTM_R_ifog2", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "10", "SubInstance" : "grp_run_all_slices_unrol_fu_485", "Port" : "LSTM_R_ifog2"}]},
			{"Name" : "LSTM_b_ifog2", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "10", "SubInstance" : "grp_run_all_slices_unrol_fu_485", "Port" : "LSTM_b_ifog2"}]},
			{"Name" : "BN2_var2", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "10", "SubInstance" : "grp_run_all_slices_unrol_fu_485", "Port" : "BN2_var2"}]},
			{"Name" : "BN2_gamma2", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "10", "SubInstance" : "grp_run_all_slices_unrol_fu_485", "Port" : "BN2_gamma2"}]},
			{"Name" : "tokens3", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "10", "SubInstance" : "grp_run_all_slices_unrol_fu_485", "Port" : "tokens3"}]},
			{"Name" : "Emb3", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "10", "SubInstance" : "grp_run_all_slices_unrol_fu_485", "Port" : "Emb3"}]},
			{"Name" : "ConvW3", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "10", "SubInstance" : "grp_run_all_slices_unrol_fu_485", "Port" : "ConvW3"}]},
			{"Name" : "ConvB3", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "10", "SubInstance" : "grp_run_all_slices_unrol_fu_485", "Port" : "ConvB3"}]},
			{"Name" : "BN1_gamma3", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "10", "SubInstance" : "grp_run_all_slices_unrol_fu_485", "Port" : "BN1_gamma3"}]},
			{"Name" : "BN1_beta3", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "10", "SubInstance" : "grp_run_all_slices_unrol_fu_485", "Port" : "BN1_beta3"}]},
			{"Name" : "BN1_mean3", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "10", "SubInstance" : "grp_run_all_slices_unrol_fu_485", "Port" : "BN1_mean3"}]},
			{"Name" : "BN1_var3", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "10", "SubInstance" : "grp_run_all_slices_unrol_fu_485", "Port" : "BN1_var3"}]},
			{"Name" : "LSTM_W_ifog3", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "10", "SubInstance" : "grp_run_all_slices_unrol_fu_485", "Port" : "LSTM_W_ifog3"}]},
			{"Name" : "LSTM_R_ifog3", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "10", "SubInstance" : "grp_run_all_slices_unrol_fu_485", "Port" : "LSTM_R_ifog3"}]},
			{"Name" : "LSTM_b_ifog3", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "10", "SubInstance" : "grp_run_all_slices_unrol_fu_485", "Port" : "LSTM_b_ifog3"}]},
			{"Name" : "BN2_var3", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "10", "SubInstance" : "grp_run_all_slices_unrol_fu_485", "Port" : "BN2_var3"}]},
			{"Name" : "BN2_gamma3", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "10", "SubInstance" : "grp_run_all_slices_unrol_fu_485", "Port" : "BN2_gamma3"}]},
			{"Name" : "tokens4", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "10", "SubInstance" : "grp_run_all_slices_unrol_fu_485", "Port" : "tokens4"}]},
			{"Name" : "Emb4", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "10", "SubInstance" : "grp_run_all_slices_unrol_fu_485", "Port" : "Emb4"}]},
			{"Name" : "ConvW4", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "10", "SubInstance" : "grp_run_all_slices_unrol_fu_485", "Port" : "ConvW4"}]},
			{"Name" : "ConvB4", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "10", "SubInstance" : "grp_run_all_slices_unrol_fu_485", "Port" : "ConvB4"}]},
			{"Name" : "BN1_gamma4", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "10", "SubInstance" : "grp_run_all_slices_unrol_fu_485", "Port" : "BN1_gamma4"}]},
			{"Name" : "BN1_beta4", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "10", "SubInstance" : "grp_run_all_slices_unrol_fu_485", "Port" : "BN1_beta4"}]},
			{"Name" : "BN1_mean4", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "10", "SubInstance" : "grp_run_all_slices_unrol_fu_485", "Port" : "BN1_mean4"}]},
			{"Name" : "BN1_var4", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "10", "SubInstance" : "grp_run_all_slices_unrol_fu_485", "Port" : "BN1_var4"}]},
			{"Name" : "LSTM_W_ifog4", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "10", "SubInstance" : "grp_run_all_slices_unrol_fu_485", "Port" : "LSTM_W_ifog4"}]},
			{"Name" : "LSTM_R_ifog4", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "10", "SubInstance" : "grp_run_all_slices_unrol_fu_485", "Port" : "LSTM_R_ifog4"}]},
			{"Name" : "LSTM_b_ifog4", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "10", "SubInstance" : "grp_run_all_slices_unrol_fu_485", "Port" : "LSTM_b_ifog4"}]},
			{"Name" : "BN2_var4", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "10", "SubInstance" : "grp_run_all_slices_unrol_fu_485", "Port" : "BN2_var4"}]},
			{"Name" : "BN2_gamma4", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "10", "SubInstance" : "grp_run_all_slices_unrol_fu_485", "Port" : "BN2_gamma4"}]},
			{"Name" : "fc_0_W", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "fc_0_bn_var", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "fc_0_bn_gamma", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "fc_1_W", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "fc_1_bn_var", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "fc_1_bn_gamma", "Type" : "Memory", "Direction" : "I"}]},
	{"ID" : "1", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fc_0_W_U", "Parent" : "0"},
	{"ID" : "2", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fc_0_bn_var_U", "Parent" : "0"},
	{"ID" : "3", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fc_0_bn_gamma_U", "Parent" : "0"},
	{"ID" : "4", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fc_1_W_U", "Parent" : "0"},
	{"ID" : "5", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fc_1_bn_var_U", "Parent" : "0"},
	{"ID" : "6", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fc_1_bn_gamma_U", "Parent" : "0"},
	{"ID" : "7", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.merged_U", "Parent" : "0"},
	{"ID" : "8", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.z0_U", "Parent" : "0"},
	{"ID" : "9", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.z1_U", "Parent" : "0"},
	{"ID" : "10", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_485", "Parent" : "0", "Child" : ["11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21", "22", "23", "24", "25", "26", "27", "28", "29", "30", "31", "32", "33", "34", "35", "36", "37", "38", "39", "40", "41", "42", "43", "44", "45", "46", "47", "48", "49", "50", "51", "52", "53", "54", "55", "56", "57", "58", "59", "60", "61", "62", "63", "64", "65", "66", "67", "68", "69", "70", "71", "72", "73", "74", "75", "76", "77", "78", "79", "107", "124", "134", "135", "136", "137"],
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
					{"ID" : "124", "SubInstance" : "grp_conv_bn_act_pool_fu_974", "Port" : "X_slice"}]},
			{"Name" : "B_arg_index", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "Y", "Type" : "Memory", "Direction" : "IO",
				"SubConnect" : [
					{"ID" : "124", "SubInstance" : "grp_conv_bn_act_pool_fu_974", "Port" : "Y"}]},
			{"Name" : "ConvW0", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "124", "SubInstance" : "grp_conv_bn_act_pool_fu_974", "Port" : "W"}]},
			{"Name" : "ConvB0", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "124", "SubInstance" : "grp_conv_bn_act_pool_fu_974", "Port" : "B"}]},
			{"Name" : "BN1_gamma0", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "124", "SubInstance" : "grp_conv_bn_act_pool_fu_974", "Port" : "gamma"}]},
			{"Name" : "BN1_beta0", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "124", "SubInstance" : "grp_conv_bn_act_pool_fu_974", "Port" : "beta"}]},
			{"Name" : "BN1_mean0", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "124", "SubInstance" : "grp_conv_bn_act_pool_fu_974", "Port" : "mean"}]},
			{"Name" : "BN1_var0", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "124", "SubInstance" : "grp_conv_bn_act_pool_fu_974", "Port" : "var"}]},
			{"Name" : "U_slice", "Type" : "Memory", "Direction" : "IO",
				"SubConnect" : [
					{"ID" : "124", "SubInstance" : "grp_conv_bn_act_pool_fu_974", "Port" : "U"},
					{"ID" : "79", "SubInstance" : "grp_lstm_forward_unidir_fu_929", "Port" : "U_slice"}]},
			{"Name" : "c_slice", "Type" : "Memory", "Direction" : "IO",
				"SubConnect" : [
					{"ID" : "79", "SubInstance" : "grp_lstm_forward_unidir_fu_929", "Port" : "c_slice"}]},
			{"Name" : "table_exp_Z1_array_s", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "107", "SubInstance" : "grp_generic_tanh_float_s_fu_963", "Port" : "table_exp_Z1_array_s"},
					{"ID" : "79", "SubInstance" : "grp_lstm_forward_unidir_fu_929", "Port" : "table_exp_Z1_array_s"}]},
			{"Name" : "table_f_Z3_array_V", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "107", "SubInstance" : "grp_generic_tanh_float_s_fu_963", "Port" : "table_f_Z3_array_V"},
					{"ID" : "79", "SubInstance" : "grp_lstm_forward_unidir_fu_929", "Port" : "table_f_Z3_array_V"}]},
			{"Name" : "table_f_Z2_array_V", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "107", "SubInstance" : "grp_generic_tanh_float_s_fu_963", "Port" : "table_f_Z2_array_V"},
					{"ID" : "79", "SubInstance" : "grp_lstm_forward_unidir_fu_929", "Port" : "table_f_Z2_array_V"}]},
			{"Name" : "LSTM_W_ifog0", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "79", "SubInstance" : "grp_lstm_forward_unidir_fu_929", "Port" : "W_ifog"}]},
			{"Name" : "LSTM_R_ifog0", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "79", "SubInstance" : "grp_lstm_forward_unidir_fu_929", "Port" : "R_ifog"}]},
			{"Name" : "LSTM_b_ifog0", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "79", "SubInstance" : "grp_lstm_forward_unidir_fu_929", "Port" : "b_ifog"}]},
			{"Name" : "h_slice", "Type" : "Memory", "Direction" : "IO",
				"SubConnect" : [
					{"ID" : "79", "SubInstance" : "grp_lstm_forward_unidir_fu_929", "Port" : "h_last"}]},
			{"Name" : "BN2_var0", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "BN2_gamma0", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "tokens1", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "Emb1", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "ConvW1", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "124", "SubInstance" : "grp_conv_bn_act_pool_fu_974", "Port" : "W"}]},
			{"Name" : "ConvB1", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "124", "SubInstance" : "grp_conv_bn_act_pool_fu_974", "Port" : "B"}]},
			{"Name" : "BN1_gamma1", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "124", "SubInstance" : "grp_conv_bn_act_pool_fu_974", "Port" : "gamma"}]},
			{"Name" : "BN1_beta1", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "124", "SubInstance" : "grp_conv_bn_act_pool_fu_974", "Port" : "beta"}]},
			{"Name" : "BN1_mean1", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "124", "SubInstance" : "grp_conv_bn_act_pool_fu_974", "Port" : "mean"}]},
			{"Name" : "BN1_var1", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "124", "SubInstance" : "grp_conv_bn_act_pool_fu_974", "Port" : "var"}]},
			{"Name" : "LSTM_W_ifog1", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "79", "SubInstance" : "grp_lstm_forward_unidir_fu_929", "Port" : "W_ifog"}]},
			{"Name" : "LSTM_R_ifog1", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "79", "SubInstance" : "grp_lstm_forward_unidir_fu_929", "Port" : "R_ifog"}]},
			{"Name" : "LSTM_b_ifog1", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "79", "SubInstance" : "grp_lstm_forward_unidir_fu_929", "Port" : "b_ifog"}]},
			{"Name" : "BN2_var1", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "BN2_gamma1", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "tokens2", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "Emb2", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "ConvW2", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "124", "SubInstance" : "grp_conv_bn_act_pool_fu_974", "Port" : "W"}]},
			{"Name" : "ConvB2", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "124", "SubInstance" : "grp_conv_bn_act_pool_fu_974", "Port" : "B"}]},
			{"Name" : "BN1_gamma2", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "124", "SubInstance" : "grp_conv_bn_act_pool_fu_974", "Port" : "gamma"}]},
			{"Name" : "BN1_beta2", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "124", "SubInstance" : "grp_conv_bn_act_pool_fu_974", "Port" : "beta"}]},
			{"Name" : "BN1_mean2", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "124", "SubInstance" : "grp_conv_bn_act_pool_fu_974", "Port" : "mean"}]},
			{"Name" : "BN1_var2", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "124", "SubInstance" : "grp_conv_bn_act_pool_fu_974", "Port" : "var"}]},
			{"Name" : "LSTM_W_ifog2", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "79", "SubInstance" : "grp_lstm_forward_unidir_fu_929", "Port" : "W_ifog"}]},
			{"Name" : "LSTM_R_ifog2", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "79", "SubInstance" : "grp_lstm_forward_unidir_fu_929", "Port" : "R_ifog"}]},
			{"Name" : "LSTM_b_ifog2", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "79", "SubInstance" : "grp_lstm_forward_unidir_fu_929", "Port" : "b_ifog"}]},
			{"Name" : "BN2_var2", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "BN2_gamma2", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "tokens3", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "Emb3", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "ConvW3", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "124", "SubInstance" : "grp_conv_bn_act_pool_fu_974", "Port" : "W"}]},
			{"Name" : "ConvB3", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "124", "SubInstance" : "grp_conv_bn_act_pool_fu_974", "Port" : "B"}]},
			{"Name" : "BN1_gamma3", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "124", "SubInstance" : "grp_conv_bn_act_pool_fu_974", "Port" : "gamma"}]},
			{"Name" : "BN1_beta3", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "124", "SubInstance" : "grp_conv_bn_act_pool_fu_974", "Port" : "beta"}]},
			{"Name" : "BN1_mean3", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "124", "SubInstance" : "grp_conv_bn_act_pool_fu_974", "Port" : "mean"}]},
			{"Name" : "BN1_var3", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "124", "SubInstance" : "grp_conv_bn_act_pool_fu_974", "Port" : "var"}]},
			{"Name" : "LSTM_W_ifog3", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "79", "SubInstance" : "grp_lstm_forward_unidir_fu_929", "Port" : "W_ifog"}]},
			{"Name" : "LSTM_R_ifog3", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "79", "SubInstance" : "grp_lstm_forward_unidir_fu_929", "Port" : "R_ifog"}]},
			{"Name" : "LSTM_b_ifog3", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "79", "SubInstance" : "grp_lstm_forward_unidir_fu_929", "Port" : "b_ifog"}]},
			{"Name" : "BN2_var3", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "BN2_gamma3", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "tokens4", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "Emb4", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "ConvW4", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "124", "SubInstance" : "grp_conv_bn_act_pool_fu_974", "Port" : "W"}]},
			{"Name" : "ConvB4", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "124", "SubInstance" : "grp_conv_bn_act_pool_fu_974", "Port" : "B"}]},
			{"Name" : "BN1_gamma4", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "124", "SubInstance" : "grp_conv_bn_act_pool_fu_974", "Port" : "gamma"}]},
			{"Name" : "BN1_beta4", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "124", "SubInstance" : "grp_conv_bn_act_pool_fu_974", "Port" : "beta"}]},
			{"Name" : "BN1_mean4", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "124", "SubInstance" : "grp_conv_bn_act_pool_fu_974", "Port" : "mean"}]},
			{"Name" : "BN1_var4", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "124", "SubInstance" : "grp_conv_bn_act_pool_fu_974", "Port" : "var"}]},
			{"Name" : "LSTM_W_ifog4", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "79", "SubInstance" : "grp_lstm_forward_unidir_fu_929", "Port" : "W_ifog"}]},
			{"Name" : "LSTM_R_ifog4", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "79", "SubInstance" : "grp_lstm_forward_unidir_fu_929", "Port" : "R_ifog"}]},
			{"Name" : "LSTM_b_ifog4", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "79", "SubInstance" : "grp_lstm_forward_unidir_fu_929", "Port" : "b_ifog"}]},
			{"Name" : "BN2_var4", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "BN2_gamma4", "Type" : "Memory", "Direction" : "I"}]},
	{"ID" : "11", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_485.tokens0_U", "Parent" : "10"},
	{"ID" : "12", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_485.Emb0_U", "Parent" : "10"},
	{"ID" : "13", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_485.X_slice_U", "Parent" : "10"},
	{"ID" : "14", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_485.ConvW0_U", "Parent" : "10"},
	{"ID" : "15", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_485.ConvB0_U", "Parent" : "10"},
	{"ID" : "16", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_485.BN1_gamma0_U", "Parent" : "10"},
	{"ID" : "17", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_485.BN1_beta0_U", "Parent" : "10"},
	{"ID" : "18", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_485.BN1_mean0_U", "Parent" : "10"},
	{"ID" : "19", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_485.BN1_var0_U", "Parent" : "10"},
	{"ID" : "20", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_485.U_slice_U", "Parent" : "10"},
	{"ID" : "21", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_485.LSTM_W_ifog0_U", "Parent" : "10"},
	{"ID" : "22", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_485.LSTM_R_ifog0_U", "Parent" : "10"},
	{"ID" : "23", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_485.LSTM_b_ifog0_U", "Parent" : "10"},
	{"ID" : "24", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_485.h_slice_U", "Parent" : "10"},
	{"ID" : "25", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_485.BN2_var0_U", "Parent" : "10"},
	{"ID" : "26", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_485.BN2_gamma0_U", "Parent" : "10"},
	{"ID" : "27", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_485.tokens1_U", "Parent" : "10"},
	{"ID" : "28", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_485.Emb1_U", "Parent" : "10"},
	{"ID" : "29", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_485.ConvW1_U", "Parent" : "10"},
	{"ID" : "30", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_485.ConvB1_U", "Parent" : "10"},
	{"ID" : "31", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_485.BN1_gamma1_U", "Parent" : "10"},
	{"ID" : "32", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_485.BN1_beta1_U", "Parent" : "10"},
	{"ID" : "33", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_485.BN1_mean1_U", "Parent" : "10"},
	{"ID" : "34", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_485.BN1_var1_U", "Parent" : "10"},
	{"ID" : "35", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_485.LSTM_W_ifog1_U", "Parent" : "10"},
	{"ID" : "36", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_485.LSTM_R_ifog1_U", "Parent" : "10"},
	{"ID" : "37", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_485.LSTM_b_ifog1_U", "Parent" : "10"},
	{"ID" : "38", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_485.BN2_var1_U", "Parent" : "10"},
	{"ID" : "39", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_485.BN2_gamma1_U", "Parent" : "10"},
	{"ID" : "40", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_485.tokens2_U", "Parent" : "10"},
	{"ID" : "41", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_485.Emb2_U", "Parent" : "10"},
	{"ID" : "42", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_485.ConvW2_U", "Parent" : "10"},
	{"ID" : "43", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_485.ConvB2_U", "Parent" : "10"},
	{"ID" : "44", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_485.BN1_gamma2_U", "Parent" : "10"},
	{"ID" : "45", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_485.BN1_beta2_U", "Parent" : "10"},
	{"ID" : "46", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_485.BN1_mean2_U", "Parent" : "10"},
	{"ID" : "47", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_485.BN1_var2_U", "Parent" : "10"},
	{"ID" : "48", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_485.LSTM_W_ifog2_U", "Parent" : "10"},
	{"ID" : "49", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_485.LSTM_R_ifog2_U", "Parent" : "10"},
	{"ID" : "50", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_485.LSTM_b_ifog2_U", "Parent" : "10"},
	{"ID" : "51", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_485.BN2_var2_U", "Parent" : "10"},
	{"ID" : "52", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_485.BN2_gamma2_U", "Parent" : "10"},
	{"ID" : "53", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_485.tokens3_U", "Parent" : "10"},
	{"ID" : "54", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_485.Emb3_U", "Parent" : "10"},
	{"ID" : "55", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_485.ConvW3_U", "Parent" : "10"},
	{"ID" : "56", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_485.ConvB3_U", "Parent" : "10"},
	{"ID" : "57", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_485.BN1_gamma3_U", "Parent" : "10"},
	{"ID" : "58", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_485.BN1_beta3_U", "Parent" : "10"},
	{"ID" : "59", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_485.BN1_mean3_U", "Parent" : "10"},
	{"ID" : "60", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_485.BN1_var3_U", "Parent" : "10"},
	{"ID" : "61", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_485.LSTM_W_ifog3_U", "Parent" : "10"},
	{"ID" : "62", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_485.LSTM_R_ifog3_U", "Parent" : "10"},
	{"ID" : "63", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_485.LSTM_b_ifog3_U", "Parent" : "10"},
	{"ID" : "64", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_485.BN2_var3_U", "Parent" : "10"},
	{"ID" : "65", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_485.BN2_gamma3_U", "Parent" : "10"},
	{"ID" : "66", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_485.tokens4_U", "Parent" : "10"},
	{"ID" : "67", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_485.Emb4_U", "Parent" : "10"},
	{"ID" : "68", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_485.ConvW4_U", "Parent" : "10"},
	{"ID" : "69", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_485.ConvB4_U", "Parent" : "10"},
	{"ID" : "70", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_485.BN1_gamma4_U", "Parent" : "10"},
	{"ID" : "71", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_485.BN1_beta4_U", "Parent" : "10"},
	{"ID" : "72", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_485.BN1_mean4_U", "Parent" : "10"},
	{"ID" : "73", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_485.BN1_var4_U", "Parent" : "10"},
	{"ID" : "74", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_485.LSTM_W_ifog4_U", "Parent" : "10"},
	{"ID" : "75", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_485.LSTM_R_ifog4_U", "Parent" : "10"},
	{"ID" : "76", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_485.LSTM_b_ifog4_U", "Parent" : "10"},
	{"ID" : "77", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_485.BN2_var4_U", "Parent" : "10"},
	{"ID" : "78", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_485.BN2_gamma4_U", "Parent" : "10"},
	{"ID" : "79", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_485.grp_lstm_forward_unidir_fu_929", "Parent" : "10", "Child" : ["80", "81", "82", "99", "100", "101", "102", "103", "104", "105", "106"],
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
					{"ID" : "82", "SubInstance" : "grp_generic_tanh_float_s_fu_325", "Port" : "table_exp_Z1_array_s"}]},
			{"Name" : "table_f_Z3_array_V", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "82", "SubInstance" : "grp_generic_tanh_float_s_fu_325", "Port" : "table_f_Z3_array_V"}]},
			{"Name" : "table_f_Z2_array_V", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "82", "SubInstance" : "grp_generic_tanh_float_s_fu_325", "Port" : "table_f_Z2_array_V"}]}]},
	{"ID" : "80", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_485.grp_lstm_forward_unidir_fu_929.c_slice_U", "Parent" : "79"},
	{"ID" : "81", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_485.grp_lstm_forward_unidir_fu_929.z_U", "Parent" : "79"},
	{"ID" : "82", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_485.grp_lstm_forward_unidir_fu_929.grp_generic_tanh_float_s_fu_325", "Parent" : "79", "Child" : ["83", "92", "93", "94", "95", "96", "97", "98"],
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
					{"ID" : "83", "SubInstance" : "grp_exp_generic_double_s_fu_89", "Port" : "table_exp_Z1_array_s"}]},
			{"Name" : "table_f_Z3_array_V", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "83", "SubInstance" : "grp_exp_generic_double_s_fu_89", "Port" : "table_f_Z3_array_V"}]},
			{"Name" : "table_f_Z2_array_V", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "83", "SubInstance" : "grp_exp_generic_double_s_fu_89", "Port" : "table_f_Z2_array_V"}]}]},
	{"ID" : "83", "Level" : "4", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_485.grp_lstm_forward_unidir_fu_929.grp_generic_tanh_float_s_fu_325.grp_exp_generic_double_s_fu_89", "Parent" : "82", "Child" : ["84", "85", "86", "87", "88", "89", "90", "91"],
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
	{"ID" : "84", "Level" : "5", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_485.grp_lstm_forward_unidir_fu_929.grp_generic_tanh_float_s_fu_325.grp_exp_generic_double_s_fu_89.table_exp_Z1_array_s_U", "Parent" : "83"},
	{"ID" : "85", "Level" : "5", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_485.grp_lstm_forward_unidir_fu_929.grp_generic_tanh_float_s_fu_325.grp_exp_generic_double_s_fu_89.table_f_Z3_array_V_U", "Parent" : "83"},
	{"ID" : "86", "Level" : "5", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_485.grp_lstm_forward_unidir_fu_929.grp_generic_tanh_float_s_fu_325.grp_exp_generic_double_s_fu_89.table_f_Z2_array_V_U", "Parent" : "83"},
	{"ID" : "87", "Level" : "5", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_485.grp_lstm_forward_unidir_fu_929.grp_generic_tanh_float_s_fu_325.grp_exp_generic_double_s_fu_89.main_mul_72ns_13s_84_5_1_U29", "Parent" : "83"},
	{"ID" : "88", "Level" : "5", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_485.grp_lstm_forward_unidir_fu_929.grp_generic_tanh_float_s_fu_325.grp_exp_generic_double_s_fu_89.main_mul_36ns_43ns_79_2_1_U30", "Parent" : "83"},
	{"ID" : "89", "Level" : "5", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_485.grp_lstm_forward_unidir_fu_929.grp_generic_tanh_float_s_fu_325.grp_exp_generic_double_s_fu_89.main_mul_44ns_49ns_93_2_1_U31", "Parent" : "83"},
	{"ID" : "90", "Level" : "5", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_485.grp_lstm_forward_unidir_fu_929.grp_generic_tanh_float_s_fu_325.grp_exp_generic_double_s_fu_89.main_mul_50ns_50ns_100_2_1_U32", "Parent" : "83"},
	{"ID" : "91", "Level" : "5", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_485.grp_lstm_forward_unidir_fu_929.grp_generic_tanh_float_s_fu_325.grp_exp_generic_double_s_fu_89.main_mac_muladd_16ns_16s_19s_31_1_1_U33", "Parent" : "83"},
	{"ID" : "92", "Level" : "4", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_485.grp_lstm_forward_unidir_fu_929.grp_generic_tanh_float_s_fu_325.main_faddfsub_32ns_32ns_32_5_full_dsp_1_U43", "Parent" : "82"},
	{"ID" : "93", "Level" : "4", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_485.grp_lstm_forward_unidir_fu_929.grp_generic_tanh_float_s_fu_325.main_fmul_32ns_32ns_32_4_max_dsp_1_U44", "Parent" : "82"},
	{"ID" : "94", "Level" : "4", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_485.grp_lstm_forward_unidir_fu_929.grp_generic_tanh_float_s_fu_325.main_fdiv_32ns_32ns_32_16_1_U45", "Parent" : "82"},
	{"ID" : "95", "Level" : "4", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_485.grp_lstm_forward_unidir_fu_929.grp_generic_tanh_float_s_fu_325.main_fptrunc_64ns_32_2_1_U46", "Parent" : "82"},
	{"ID" : "96", "Level" : "4", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_485.grp_lstm_forward_unidir_fu_929.grp_generic_tanh_float_s_fu_325.main_fpext_32ns_64_2_1_U47", "Parent" : "82"},
	{"ID" : "97", "Level" : "4", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_485.grp_lstm_forward_unidir_fu_929.grp_generic_tanh_float_s_fu_325.main_fcmp_32ns_32ns_1_2_1_U48", "Parent" : "82"},
	{"ID" : "98", "Level" : "4", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_485.grp_lstm_forward_unidir_fu_929.grp_generic_tanh_float_s_fu_325.main_dadd_64ns_64ns_64_5_full_dsp_1_U49", "Parent" : "82"},
	{"ID" : "99", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_485.grp_lstm_forward_unidir_fu_929.main_fadd_32ns_32ns_32_5_full_dsp_1_U54", "Parent" : "79"},
	{"ID" : "100", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_485.grp_lstm_forward_unidir_fu_929.main_fadd_32ns_32ns_32_5_full_dsp_1_U55", "Parent" : "79"},
	{"ID" : "101", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_485.grp_lstm_forward_unidir_fu_929.main_fmul_32ns_32ns_32_4_max_dsp_1_U56", "Parent" : "79"},
	{"ID" : "102", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_485.grp_lstm_forward_unidir_fu_929.main_fmul_32ns_32ns_32_4_max_dsp_1_U57", "Parent" : "79"},
	{"ID" : "103", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_485.grp_lstm_forward_unidir_fu_929.main_fdiv_32ns_32ns_32_16_1_U58", "Parent" : "79"},
	{"ID" : "104", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_485.grp_lstm_forward_unidir_fu_929.main_fdiv_32ns_32ns_32_16_1_U59", "Parent" : "79"},
	{"ID" : "105", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_485.grp_lstm_forward_unidir_fu_929.main_fexp_32ns_32ns_32_9_full_dsp_1_U60", "Parent" : "79"},
	{"ID" : "106", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_485.grp_lstm_forward_unidir_fu_929.main_fexp_32ns_32ns_32_9_full_dsp_1_U61", "Parent" : "79"},
	{"ID" : "107", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_485.grp_generic_tanh_float_s_fu_963", "Parent" : "10", "Child" : ["108", "117", "118", "119", "120", "121", "122", "123"],
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
					{"ID" : "108", "SubInstance" : "grp_exp_generic_double_s_fu_89", "Port" : "table_exp_Z1_array_s"}]},
			{"Name" : "table_f_Z3_array_V", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "108", "SubInstance" : "grp_exp_generic_double_s_fu_89", "Port" : "table_f_Z3_array_V"}]},
			{"Name" : "table_f_Z2_array_V", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "108", "SubInstance" : "grp_exp_generic_double_s_fu_89", "Port" : "table_f_Z2_array_V"}]}]},
	{"ID" : "108", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_485.grp_generic_tanh_float_s_fu_963.grp_exp_generic_double_s_fu_89", "Parent" : "107", "Child" : ["109", "110", "111", "112", "113", "114", "115", "116"],
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
	{"ID" : "109", "Level" : "4", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_485.grp_generic_tanh_float_s_fu_963.grp_exp_generic_double_s_fu_89.table_exp_Z1_array_s_U", "Parent" : "108"},
	{"ID" : "110", "Level" : "4", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_485.grp_generic_tanh_float_s_fu_963.grp_exp_generic_double_s_fu_89.table_f_Z3_array_V_U", "Parent" : "108"},
	{"ID" : "111", "Level" : "4", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_485.grp_generic_tanh_float_s_fu_963.grp_exp_generic_double_s_fu_89.table_f_Z2_array_V_U", "Parent" : "108"},
	{"ID" : "112", "Level" : "4", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_485.grp_generic_tanh_float_s_fu_963.grp_exp_generic_double_s_fu_89.main_mul_72ns_13s_84_5_1_U29", "Parent" : "108"},
	{"ID" : "113", "Level" : "4", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_485.grp_generic_tanh_float_s_fu_963.grp_exp_generic_double_s_fu_89.main_mul_36ns_43ns_79_2_1_U30", "Parent" : "108"},
	{"ID" : "114", "Level" : "4", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_485.grp_generic_tanh_float_s_fu_963.grp_exp_generic_double_s_fu_89.main_mul_44ns_49ns_93_2_1_U31", "Parent" : "108"},
	{"ID" : "115", "Level" : "4", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_485.grp_generic_tanh_float_s_fu_963.grp_exp_generic_double_s_fu_89.main_mul_50ns_50ns_100_2_1_U32", "Parent" : "108"},
	{"ID" : "116", "Level" : "4", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_485.grp_generic_tanh_float_s_fu_963.grp_exp_generic_double_s_fu_89.main_mac_muladd_16ns_16s_19s_31_1_1_U33", "Parent" : "108"},
	{"ID" : "117", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_485.grp_generic_tanh_float_s_fu_963.main_faddfsub_32ns_32ns_32_5_full_dsp_1_U43", "Parent" : "107"},
	{"ID" : "118", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_485.grp_generic_tanh_float_s_fu_963.main_fmul_32ns_32ns_32_4_max_dsp_1_U44", "Parent" : "107"},
	{"ID" : "119", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_485.grp_generic_tanh_float_s_fu_963.main_fdiv_32ns_32ns_32_16_1_U45", "Parent" : "107"},
	{"ID" : "120", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_485.grp_generic_tanh_float_s_fu_963.main_fptrunc_64ns_32_2_1_U46", "Parent" : "107"},
	{"ID" : "121", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_485.grp_generic_tanh_float_s_fu_963.main_fpext_32ns_64_2_1_U47", "Parent" : "107"},
	{"ID" : "122", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_485.grp_generic_tanh_float_s_fu_963.main_fcmp_32ns_32ns_1_2_1_U48", "Parent" : "107"},
	{"ID" : "123", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_485.grp_generic_tanh_float_s_fu_963.main_dadd_64ns_64ns_64_5_full_dsp_1_U49", "Parent" : "107"},
	{"ID" : "124", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_485.grp_conv_bn_act_pool_fu_974", "Parent" : "10", "Child" : ["125", "126", "127", "128", "129", "130", "131", "132", "133"],
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
	{"ID" : "125", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_485.grp_conv_bn_act_pool_fu_974.Y_U", "Parent" : "124"},
	{"ID" : "126", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_485.grp_conv_bn_act_pool_fu_974.main_faddfsub_32ns_32ns_32_5_full_dsp_1_U1", "Parent" : "124"},
	{"ID" : "127", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_485.grp_conv_bn_act_pool_fu_974.main_fmul_32ns_32ns_32_4_max_dsp_1_U2", "Parent" : "124"},
	{"ID" : "128", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_485.grp_conv_bn_act_pool_fu_974.main_fdiv_32ns_32ns_32_16_1_U3", "Parent" : "124"},
	{"ID" : "129", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_485.grp_conv_bn_act_pool_fu_974.main_sitofp_32s_32_6_1_U4", "Parent" : "124"},
	{"ID" : "130", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_485.grp_conv_bn_act_pool_fu_974.main_fcmp_32ns_32ns_1_2_1_U5", "Parent" : "124"},
	{"ID" : "131", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_485.grp_conv_bn_act_pool_fu_974.main_fsqrt_32ns_32ns_32_12_1_U6", "Parent" : "124"},
	{"ID" : "132", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_485.grp_conv_bn_act_pool_fu_974.main_udiv_10ns_7ns_9_14_seq_1_U7", "Parent" : "124"},
	{"ID" : "133", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_485.grp_conv_bn_act_pool_fu_974.main_mac_muladd_11ns_14ns_10ns_23_1_1_U8", "Parent" : "124"},
	{"ID" : "134", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_485.main_fadd_32ns_32ns_32_5_full_dsp_1_U71", "Parent" : "10"},
	{"ID" : "135", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_485.main_fmul_32ns_32ns_32_4_max_dsp_1_U72", "Parent" : "10"},
	{"ID" : "136", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_485.main_fdiv_32ns_32ns_32_16_1_U73", "Parent" : "10"},
	{"ID" : "137", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_485.main_fsqrt_32ns_32ns_32_12_1_U74", "Parent" : "10"},
	{"ID" : "138", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.main_fadd_32ns_32ns_32_5_full_dsp_1_U107", "Parent" : "0"},
	{"ID" : "139", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.main_fmul_32ns_32ns_32_4_max_dsp_1_U108", "Parent" : "0"},
	{"ID" : "140", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.main_fdiv_32ns_32ns_32_16_1_U109", "Parent" : "0"},
	{"ID" : "141", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.main_fcmp_32ns_32ns_1_2_1_U110", "Parent" : "0"},
	{"ID" : "142", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.main_fsqrt_32ns_32ns_32_12_1_U111", "Parent" : "0"}]}


set ArgLastReadFirstWriteLatency {
	main {
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
		BN2_gamma4 {Type I LastRead -1 FirstWrite -1}
		fc_0_W {Type I LastRead -1 FirstWrite -1}
		fc_0_bn_var {Type I LastRead -1 FirstWrite -1}
		fc_0_bn_gamma {Type I LastRead -1 FirstWrite -1}
		fc_1_W {Type I LastRead -1 FirstWrite -1}
		fc_1_bn_var {Type I LastRead -1 FirstWrite -1}
		fc_1_bn_gamma {Type I LastRead -1 FirstWrite -1}}
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
	{"Name" : "Latency", "Min" : "21006103", "Max" : "541989483"}
	, {"Name" : "Interval", "Min" : "21006104", "Max" : "541989484"}
]}

set PipelineEnableSignalInfo {[
]}

set Spec2ImplPortList { 
}

set busDeadlockParameterList { 
}

# RTL port scheduling information:
set fifoSchedulingInfoList { 
}

# RTL bus port read request latency information:
set busReadReqLatencyList { 
}

# RTL bus port write response latency information:
set busWriteResLatencyList { 
}

# RTL array port load latency information:
set memoryLoadLatencyList { 
}
