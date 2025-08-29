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
	{"ID" : "0", "Level" : "0", "Path" : "`AUTOTB_DUT_INST", "Parent" : "", "Child" : ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "176", "181", "182", "183"],
		"CDFG" : "main",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "0", "ap_idle" : "1",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "96913071", "EstimateLatencyMax" : "97153071",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"WaitState" : [
			{"State" : "ap_ST_fsm_state2", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_run_all_slices_unrol_fu_377"},
			{"State" : "ap_ST_fsm_state15", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_bn_vector_1_fu_501"},
			{"State" : "ap_ST_fsm_state32", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_bn_vector_1_fu_501"}],
		"Port" : [
			{"Name" : "X0", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "14", "SubInstance" : "grp_run_all_slices_unrol_fu_377", "Port" : "X0"}]},
			{"Name" : "ConvW0", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "14", "SubInstance" : "grp_run_all_slices_unrol_fu_377", "Port" : "ConvW0"}]},
			{"Name" : "BN1_var0", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "14", "SubInstance" : "grp_run_all_slices_unrol_fu_377", "Port" : "BN1_var0"}]},
			{"Name" : "BN1_gamma0", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "14", "SubInstance" : "grp_run_all_slices_unrol_fu_377", "Port" : "BN1_gamma0"}]},
			{"Name" : "table_exp_Z1_array_s", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "14", "SubInstance" : "grp_run_all_slices_unrol_fu_377", "Port" : "table_exp_Z1_array_s"}]},
			{"Name" : "table_f_Z3_array_V", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "14", "SubInstance" : "grp_run_all_slices_unrol_fu_377", "Port" : "table_f_Z3_array_V"}]},
			{"Name" : "table_f_Z2_array_V", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "14", "SubInstance" : "grp_run_all_slices_unrol_fu_377", "Port" : "table_f_Z2_array_V"}]},
			{"Name" : "LSTM_W_ifog0", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "14", "SubInstance" : "grp_run_all_slices_unrol_fu_377", "Port" : "LSTM_W_ifog0"}]},
			{"Name" : "LSTM_R_ifog0", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "14", "SubInstance" : "grp_run_all_slices_unrol_fu_377", "Port" : "LSTM_R_ifog0"}]},
			{"Name" : "LSTM_b_ifog0", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "14", "SubInstance" : "grp_run_all_slices_unrol_fu_377", "Port" : "LSTM_b_ifog0"}]},
			{"Name" : "BN2_gamma0", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "14", "SubInstance" : "grp_run_all_slices_unrol_fu_377", "Port" : "BN2_gamma0"}]},
			{"Name" : "BN2_beta0", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "14", "SubInstance" : "grp_run_all_slices_unrol_fu_377", "Port" : "BN2_beta0"}]},
			{"Name" : "BN2_mean0", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "14", "SubInstance" : "grp_run_all_slices_unrol_fu_377", "Port" : "BN2_mean0"}]},
			{"Name" : "BN2_var0", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "14", "SubInstance" : "grp_run_all_slices_unrol_fu_377", "Port" : "BN2_var0"}]},
			{"Name" : "X01", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "14", "SubInstance" : "grp_run_all_slices_unrol_fu_377", "Port" : "X01"}]},
			{"Name" : "ConvW1", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "14", "SubInstance" : "grp_run_all_slices_unrol_fu_377", "Port" : "ConvW1"}]},
			{"Name" : "BN1_var1", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "14", "SubInstance" : "grp_run_all_slices_unrol_fu_377", "Port" : "BN1_var1"}]},
			{"Name" : "BN1_gamma1", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "14", "SubInstance" : "grp_run_all_slices_unrol_fu_377", "Port" : "BN1_gamma1"}]},
			{"Name" : "LSTM_W_ifog1", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "14", "SubInstance" : "grp_run_all_slices_unrol_fu_377", "Port" : "LSTM_W_ifog1"}]},
			{"Name" : "LSTM_R_ifog1", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "14", "SubInstance" : "grp_run_all_slices_unrol_fu_377", "Port" : "LSTM_R_ifog1"}]},
			{"Name" : "LSTM_b_ifog1", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "14", "SubInstance" : "grp_run_all_slices_unrol_fu_377", "Port" : "LSTM_b_ifog1"}]},
			{"Name" : "BN2_gamma1", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "14", "SubInstance" : "grp_run_all_slices_unrol_fu_377", "Port" : "BN2_gamma1"}]},
			{"Name" : "BN2_beta1", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "14", "SubInstance" : "grp_run_all_slices_unrol_fu_377", "Port" : "BN2_beta1"}]},
			{"Name" : "BN2_mean1", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "14", "SubInstance" : "grp_run_all_slices_unrol_fu_377", "Port" : "BN2_mean1"}]},
			{"Name" : "BN2_var1", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "14", "SubInstance" : "grp_run_all_slices_unrol_fu_377", "Port" : "BN2_var1"}]},
			{"Name" : "X06", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "14", "SubInstance" : "grp_run_all_slices_unrol_fu_377", "Port" : "X06"}]},
			{"Name" : "ConvW2", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "14", "SubInstance" : "grp_run_all_slices_unrol_fu_377", "Port" : "ConvW2"}]},
			{"Name" : "BN1_var2", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "14", "SubInstance" : "grp_run_all_slices_unrol_fu_377", "Port" : "BN1_var2"}]},
			{"Name" : "BN1_gamma2", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "14", "SubInstance" : "grp_run_all_slices_unrol_fu_377", "Port" : "BN1_gamma2"}]},
			{"Name" : "LSTM_W_ifog2", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "14", "SubInstance" : "grp_run_all_slices_unrol_fu_377", "Port" : "LSTM_W_ifog2"}]},
			{"Name" : "LSTM_R_ifog2", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "14", "SubInstance" : "grp_run_all_slices_unrol_fu_377", "Port" : "LSTM_R_ifog2"}]},
			{"Name" : "LSTM_b_ifog2", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "14", "SubInstance" : "grp_run_all_slices_unrol_fu_377", "Port" : "LSTM_b_ifog2"}]},
			{"Name" : "BN2_gamma2", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "14", "SubInstance" : "grp_run_all_slices_unrol_fu_377", "Port" : "BN2_gamma2"}]},
			{"Name" : "BN2_beta2", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "14", "SubInstance" : "grp_run_all_slices_unrol_fu_377", "Port" : "BN2_beta2"}]},
			{"Name" : "BN2_mean2", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "14", "SubInstance" : "grp_run_all_slices_unrol_fu_377", "Port" : "BN2_mean2"}]},
			{"Name" : "BN2_var2", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "14", "SubInstance" : "grp_run_all_slices_unrol_fu_377", "Port" : "BN2_var2"}]},
			{"Name" : "X011", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "14", "SubInstance" : "grp_run_all_slices_unrol_fu_377", "Port" : "X011"}]},
			{"Name" : "ConvW3", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "14", "SubInstance" : "grp_run_all_slices_unrol_fu_377", "Port" : "ConvW3"}]},
			{"Name" : "BN1_var3", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "14", "SubInstance" : "grp_run_all_slices_unrol_fu_377", "Port" : "BN1_var3"}]},
			{"Name" : "BN1_gamma3", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "14", "SubInstance" : "grp_run_all_slices_unrol_fu_377", "Port" : "BN1_gamma3"}]},
			{"Name" : "LSTM_W_ifog3", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "14", "SubInstance" : "grp_run_all_slices_unrol_fu_377", "Port" : "LSTM_W_ifog3"}]},
			{"Name" : "LSTM_R_ifog3", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "14", "SubInstance" : "grp_run_all_slices_unrol_fu_377", "Port" : "LSTM_R_ifog3"}]},
			{"Name" : "LSTM_b_ifog3", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "14", "SubInstance" : "grp_run_all_slices_unrol_fu_377", "Port" : "LSTM_b_ifog3"}]},
			{"Name" : "BN2_gamma3", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "14", "SubInstance" : "grp_run_all_slices_unrol_fu_377", "Port" : "BN2_gamma3"}]},
			{"Name" : "BN2_beta3", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "14", "SubInstance" : "grp_run_all_slices_unrol_fu_377", "Port" : "BN2_beta3"}]},
			{"Name" : "BN2_mean3", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "14", "SubInstance" : "grp_run_all_slices_unrol_fu_377", "Port" : "BN2_mean3"}]},
			{"Name" : "BN2_var3", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "14", "SubInstance" : "grp_run_all_slices_unrol_fu_377", "Port" : "BN2_var3"}]},
			{"Name" : "tokens4", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "14", "SubInstance" : "grp_run_all_slices_unrol_fu_377", "Port" : "tokens4"}]},
			{"Name" : "Emb4", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "14", "SubInstance" : "grp_run_all_slices_unrol_fu_377", "Port" : "Emb4"}]},
			{"Name" : "ConvW4", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "14", "SubInstance" : "grp_run_all_slices_unrol_fu_377", "Port" : "ConvW4"}]},
			{"Name" : "BN1_var4", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "14", "SubInstance" : "grp_run_all_slices_unrol_fu_377", "Port" : "BN1_var4"}]},
			{"Name" : "BN1_gamma4", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "14", "SubInstance" : "grp_run_all_slices_unrol_fu_377", "Port" : "BN1_gamma4"}]},
			{"Name" : "LSTM_W_ifog4", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "14", "SubInstance" : "grp_run_all_slices_unrol_fu_377", "Port" : "LSTM_W_ifog4"}]},
			{"Name" : "LSTM_R_ifog4", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "14", "SubInstance" : "grp_run_all_slices_unrol_fu_377", "Port" : "LSTM_R_ifog4"}]},
			{"Name" : "LSTM_b_ifog4", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "14", "SubInstance" : "grp_run_all_slices_unrol_fu_377", "Port" : "LSTM_b_ifog4"}]},
			{"Name" : "BN2_gamma4", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "14", "SubInstance" : "grp_run_all_slices_unrol_fu_377", "Port" : "BN2_gamma4"}]},
			{"Name" : "BN2_beta4", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "14", "SubInstance" : "grp_run_all_slices_unrol_fu_377", "Port" : "BN2_beta4"}]},
			{"Name" : "BN2_mean4", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "14", "SubInstance" : "grp_run_all_slices_unrol_fu_377", "Port" : "BN2_mean4"}]},
			{"Name" : "BN2_var4", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "14", "SubInstance" : "grp_run_all_slices_unrol_fu_377", "Port" : "BN2_var4"}]},
			{"Name" : "fc_0_W", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "fc_0_bn_gamma", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "176", "SubInstance" : "grp_bn_vector_1_fu_501", "Port" : "gamma"}]},
			{"Name" : "fc_0_bn_beta", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "176", "SubInstance" : "grp_bn_vector_1_fu_501", "Port" : "beta"}]},
			{"Name" : "fc_0_bn_mean", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "176", "SubInstance" : "grp_bn_vector_1_fu_501", "Port" : "mean"}]},
			{"Name" : "fc_0_bn_var", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "176", "SubInstance" : "grp_bn_vector_1_fu_501", "Port" : "var"}]},
			{"Name" : "fc_1_W", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "fc_1_bn_gamma", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "176", "SubInstance" : "grp_bn_vector_1_fu_501", "Port" : "gamma"}]},
			{"Name" : "fc_1_bn_beta", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "176", "SubInstance" : "grp_bn_vector_1_fu_501", "Port" : "beta"}]},
			{"Name" : "fc_1_bn_mean", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "176", "SubInstance" : "grp_bn_vector_1_fu_501", "Port" : "mean"}]},
			{"Name" : "fc_1_bn_var", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "176", "SubInstance" : "grp_bn_vector_1_fu_501", "Port" : "var"}]}]},
	{"ID" : "1", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fc_0_W_U", "Parent" : "0"},
	{"ID" : "2", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fc_0_bn_gamma_U", "Parent" : "0"},
	{"ID" : "3", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fc_0_bn_beta_U", "Parent" : "0"},
	{"ID" : "4", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fc_0_bn_mean_U", "Parent" : "0"},
	{"ID" : "5", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fc_0_bn_var_U", "Parent" : "0"},
	{"ID" : "6", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fc_1_W_U", "Parent" : "0"},
	{"ID" : "7", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fc_1_bn_gamma_U", "Parent" : "0"},
	{"ID" : "8", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fc_1_bn_beta_U", "Parent" : "0"},
	{"ID" : "9", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fc_1_bn_mean_U", "Parent" : "0"},
	{"ID" : "10", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fc_1_bn_var_U", "Parent" : "0"},
	{"ID" : "11", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.merged_U", "Parent" : "0"},
	{"ID" : "12", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.z0_U", "Parent" : "0"},
	{"ID" : "13", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.z1_U", "Parent" : "0"},
	{"ID" : "14", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_377", "Parent" : "0", "Child" : ["15", "16", "17", "18", "19", "20", "21", "22", "23", "24", "25", "26", "27", "28", "29", "30", "31", "32", "33", "34", "35", "36", "37", "38", "39", "40", "41", "42", "43", "44", "45", "46", "47", "48", "49", "50", "51", "52", "53", "54", "55", "56", "57", "58", "59", "60", "61", "62", "63", "64", "65", "66", "67", "68", "69", "70", "71", "72", "73", "74", "75", "76", "77", "78", "79", "80", "81", "82", "83", "110", "127", "132", "135", "138", "141", "144", "147", "151", "156", "161", "166", "171", "172", "173", "174", "175"],
		"CDFG" : "run_all_slices_unrol",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "0", "ap_idle" : "1",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "96674726", "EstimateLatencyMax" : "96914726",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"WaitState" : [
			{"State" : "ap_ST_fsm_state57", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_lstm_forward_unidir_fu_994"},
			{"State" : "ap_ST_fsm_state124", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_lstm_forward_unidir_fu_994"},
			{"State" : "ap_ST_fsm_state191", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_lstm_forward_unidir_fu_994"},
			{"State" : "ap_ST_fsm_state258", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_lstm_forward_unidir_fu_994"},
			{"State" : "ap_ST_fsm_state329", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_lstm_forward_unidir_fu_994"},
			{"State" : "ap_ST_fsm_state63", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_generic_tanh_float_s_fu_1025"},
			{"State" : "ap_ST_fsm_state130", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_generic_tanh_float_s_fu_1025"},
			{"State" : "ap_ST_fsm_state197", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_generic_tanh_float_s_fu_1025"},
			{"State" : "ap_ST_fsm_state264", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_generic_tanh_float_s_fu_1025"},
			{"State" : "ap_ST_fsm_state335", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_generic_tanh_float_s_fu_1025"},
			{"State" : "ap_ST_fsm_state59", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_bn_vector_fu_1036"},
			{"State" : "ap_ST_fsm_state126", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_bn_vector_fu_1036"},
			{"State" : "ap_ST_fsm_state193", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_bn_vector_fu_1036"},
			{"State" : "ap_ST_fsm_state260", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_bn_vector_fu_1036"},
			{"State" : "ap_ST_fsm_state331", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_bn_vector_fu_1036"},
			{"State" : "ap_ST_fsm_state327", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_avgpool1d_P_fu_1065"},
			{"State" : "ap_ST_fsm_state256", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_avgpool1d_P_1_fu_1071"},
			{"State" : "ap_ST_fsm_state189", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_avgpool1d_P_2_fu_1077"},
			{"State" : "ap_ST_fsm_state122", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_avgpool1d_P_3_fu_1083"},
			{"State" : "ap_ST_fsm_state55", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_avgpool1d_P_4_fu_1089"},
			{"State" : "ap_ST_fsm_state275", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_conv1d_valid_dyn_fu_1095"},
			{"State" : "ap_ST_fsm_state204", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_conv1d_valid_dyn_1_fu_1103"},
			{"State" : "ap_ST_fsm_state137", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_conv1d_valid_dyn_2_fu_1112"},
			{"State" : "ap_ST_fsm_state70", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_conv1d_valid_dyn_3_fu_1121"},
			{"State" : "ap_ST_fsm_state3", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_conv1d_valid_dyn_4_fu_1130"}],
		"Port" : [
			{"Name" : "merged", "Type" : "Memory", "Direction" : "IO"},
			{"Name" : "X0", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "166", "SubInstance" : "grp_conv1d_valid_dyn_4_fu_1130", "Port" : "X0"}]},
			{"Name" : "ConvW0", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "166", "SubInstance" : "grp_conv1d_valid_dyn_4_fu_1130", "Port" : "ConvW0"}]},
			{"Name" : "BN1_var0", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "BN1_gamma0", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "table_exp_Z1_array_s", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "110", "SubInstance" : "grp_generic_tanh_float_s_fu_1025", "Port" : "table_exp_Z1_array_s"},
					{"ID" : "83", "SubInstance" : "grp_lstm_forward_unidir_fu_994", "Port" : "table_exp_Z1_array_s"}]},
			{"Name" : "table_f_Z3_array_V", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "110", "SubInstance" : "grp_generic_tanh_float_s_fu_1025", "Port" : "table_f_Z3_array_V"},
					{"ID" : "83", "SubInstance" : "grp_lstm_forward_unidir_fu_994", "Port" : "table_f_Z3_array_V"}]},
			{"Name" : "table_f_Z2_array_V", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "110", "SubInstance" : "grp_generic_tanh_float_s_fu_1025", "Port" : "table_f_Z2_array_V"},
					{"ID" : "83", "SubInstance" : "grp_lstm_forward_unidir_fu_994", "Port" : "table_f_Z2_array_V"}]},
			{"Name" : "LSTM_W_ifog0", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "83", "SubInstance" : "grp_lstm_forward_unidir_fu_994", "Port" : "W_ifog"}]},
			{"Name" : "LSTM_R_ifog0", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "83", "SubInstance" : "grp_lstm_forward_unidir_fu_994", "Port" : "R_ifog"}]},
			{"Name" : "LSTM_b_ifog0", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "83", "SubInstance" : "grp_lstm_forward_unidir_fu_994", "Port" : "b_ifog"}]},
			{"Name" : "BN2_gamma0", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "127", "SubInstance" : "grp_bn_vector_fu_1036", "Port" : "gamma"}]},
			{"Name" : "BN2_beta0", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "127", "SubInstance" : "grp_bn_vector_fu_1036", "Port" : "beta"}]},
			{"Name" : "BN2_mean0", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "127", "SubInstance" : "grp_bn_vector_fu_1036", "Port" : "mean"}]},
			{"Name" : "BN2_var0", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "127", "SubInstance" : "grp_bn_vector_fu_1036", "Port" : "var"}]},
			{"Name" : "X01", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "161", "SubInstance" : "grp_conv1d_valid_dyn_3_fu_1121", "Port" : "X01"}]},
			{"Name" : "ConvW1", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "161", "SubInstance" : "grp_conv1d_valid_dyn_3_fu_1121", "Port" : "ConvW1"}]},
			{"Name" : "BN1_var1", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "BN1_gamma1", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "LSTM_W_ifog1", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "83", "SubInstance" : "grp_lstm_forward_unidir_fu_994", "Port" : "W_ifog"}]},
			{"Name" : "LSTM_R_ifog1", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "83", "SubInstance" : "grp_lstm_forward_unidir_fu_994", "Port" : "R_ifog"}]},
			{"Name" : "LSTM_b_ifog1", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "83", "SubInstance" : "grp_lstm_forward_unidir_fu_994", "Port" : "b_ifog"}]},
			{"Name" : "BN2_gamma1", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "127", "SubInstance" : "grp_bn_vector_fu_1036", "Port" : "gamma"}]},
			{"Name" : "BN2_beta1", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "127", "SubInstance" : "grp_bn_vector_fu_1036", "Port" : "beta"}]},
			{"Name" : "BN2_mean1", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "127", "SubInstance" : "grp_bn_vector_fu_1036", "Port" : "mean"}]},
			{"Name" : "BN2_var1", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "127", "SubInstance" : "grp_bn_vector_fu_1036", "Port" : "var"}]},
			{"Name" : "X06", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "156", "SubInstance" : "grp_conv1d_valid_dyn_2_fu_1112", "Port" : "X06"}]},
			{"Name" : "ConvW2", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "156", "SubInstance" : "grp_conv1d_valid_dyn_2_fu_1112", "Port" : "ConvW2"}]},
			{"Name" : "BN1_var2", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "BN1_gamma2", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "LSTM_W_ifog2", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "83", "SubInstance" : "grp_lstm_forward_unidir_fu_994", "Port" : "W_ifog"}]},
			{"Name" : "LSTM_R_ifog2", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "83", "SubInstance" : "grp_lstm_forward_unidir_fu_994", "Port" : "R_ifog"}]},
			{"Name" : "LSTM_b_ifog2", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "83", "SubInstance" : "grp_lstm_forward_unidir_fu_994", "Port" : "b_ifog"}]},
			{"Name" : "BN2_gamma2", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "127", "SubInstance" : "grp_bn_vector_fu_1036", "Port" : "gamma"}]},
			{"Name" : "BN2_beta2", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "127", "SubInstance" : "grp_bn_vector_fu_1036", "Port" : "beta"}]},
			{"Name" : "BN2_mean2", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "127", "SubInstance" : "grp_bn_vector_fu_1036", "Port" : "mean"}]},
			{"Name" : "BN2_var2", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "127", "SubInstance" : "grp_bn_vector_fu_1036", "Port" : "var"}]},
			{"Name" : "X011", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "151", "SubInstance" : "grp_conv1d_valid_dyn_1_fu_1103", "Port" : "X011"}]},
			{"Name" : "ConvW3", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "151", "SubInstance" : "grp_conv1d_valid_dyn_1_fu_1103", "Port" : "ConvW3"}]},
			{"Name" : "BN1_var3", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "BN1_gamma3", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "LSTM_W_ifog3", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "83", "SubInstance" : "grp_lstm_forward_unidir_fu_994", "Port" : "W_ifog"}]},
			{"Name" : "LSTM_R_ifog3", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "83", "SubInstance" : "grp_lstm_forward_unidir_fu_994", "Port" : "R_ifog"}]},
			{"Name" : "LSTM_b_ifog3", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "83", "SubInstance" : "grp_lstm_forward_unidir_fu_994", "Port" : "b_ifog"}]},
			{"Name" : "BN2_gamma3", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "127", "SubInstance" : "grp_bn_vector_fu_1036", "Port" : "gamma"}]},
			{"Name" : "BN2_beta3", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "127", "SubInstance" : "grp_bn_vector_fu_1036", "Port" : "beta"}]},
			{"Name" : "BN2_mean3", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "127", "SubInstance" : "grp_bn_vector_fu_1036", "Port" : "mean"}]},
			{"Name" : "BN2_var3", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "127", "SubInstance" : "grp_bn_vector_fu_1036", "Port" : "var"}]},
			{"Name" : "tokens4", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "Emb4", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "ConvW4", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "147", "SubInstance" : "grp_conv1d_valid_dyn_fu_1095", "Port" : "ConvW4"}]},
			{"Name" : "BN1_var4", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "BN1_gamma4", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "LSTM_W_ifog4", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "83", "SubInstance" : "grp_lstm_forward_unidir_fu_994", "Port" : "W_ifog"}]},
			{"Name" : "LSTM_R_ifog4", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "83", "SubInstance" : "grp_lstm_forward_unidir_fu_994", "Port" : "R_ifog"}]},
			{"Name" : "LSTM_b_ifog4", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "83", "SubInstance" : "grp_lstm_forward_unidir_fu_994", "Port" : "b_ifog"}]},
			{"Name" : "BN2_gamma4", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "127", "SubInstance" : "grp_bn_vector_fu_1036", "Port" : "gamma"}]},
			{"Name" : "BN2_beta4", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "127", "SubInstance" : "grp_bn_vector_fu_1036", "Port" : "beta"}]},
			{"Name" : "BN2_mean4", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "127", "SubInstance" : "grp_bn_vector_fu_1036", "Port" : "mean"}]},
			{"Name" : "BN2_var4", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "127", "SubInstance" : "grp_bn_vector_fu_1036", "Port" : "var"}]}]},
	{"ID" : "15", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_377.BN1_var0_U", "Parent" : "14"},
	{"ID" : "16", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_377.BN1_gamma0_U", "Parent" : "14"},
	{"ID" : "17", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_377.LSTM_W_ifog0_U", "Parent" : "14"},
	{"ID" : "18", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_377.LSTM_R_ifog0_U", "Parent" : "14"},
	{"ID" : "19", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_377.LSTM_b_ifog0_U", "Parent" : "14"},
	{"ID" : "20", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_377.BN2_gamma0_U", "Parent" : "14"},
	{"ID" : "21", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_377.BN2_beta0_U", "Parent" : "14"},
	{"ID" : "22", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_377.BN2_mean0_U", "Parent" : "14"},
	{"ID" : "23", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_377.BN2_var0_U", "Parent" : "14"},
	{"ID" : "24", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_377.BN1_var1_U", "Parent" : "14"},
	{"ID" : "25", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_377.BN1_gamma1_U", "Parent" : "14"},
	{"ID" : "26", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_377.LSTM_W_ifog1_U", "Parent" : "14"},
	{"ID" : "27", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_377.LSTM_R_ifog1_U", "Parent" : "14"},
	{"ID" : "28", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_377.LSTM_b_ifog1_U", "Parent" : "14"},
	{"ID" : "29", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_377.BN2_gamma1_U", "Parent" : "14"},
	{"ID" : "30", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_377.BN2_beta1_U", "Parent" : "14"},
	{"ID" : "31", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_377.BN2_mean1_U", "Parent" : "14"},
	{"ID" : "32", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_377.BN2_var1_U", "Parent" : "14"},
	{"ID" : "33", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_377.BN1_var2_U", "Parent" : "14"},
	{"ID" : "34", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_377.BN1_gamma2_U", "Parent" : "14"},
	{"ID" : "35", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_377.LSTM_W_ifog2_U", "Parent" : "14"},
	{"ID" : "36", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_377.LSTM_R_ifog2_U", "Parent" : "14"},
	{"ID" : "37", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_377.LSTM_b_ifog2_U", "Parent" : "14"},
	{"ID" : "38", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_377.BN2_gamma2_U", "Parent" : "14"},
	{"ID" : "39", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_377.BN2_beta2_U", "Parent" : "14"},
	{"ID" : "40", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_377.BN2_mean2_U", "Parent" : "14"},
	{"ID" : "41", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_377.BN2_var2_U", "Parent" : "14"},
	{"ID" : "42", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_377.BN1_var3_U", "Parent" : "14"},
	{"ID" : "43", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_377.BN1_gamma3_U", "Parent" : "14"},
	{"ID" : "44", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_377.LSTM_W_ifog3_U", "Parent" : "14"},
	{"ID" : "45", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_377.LSTM_R_ifog3_U", "Parent" : "14"},
	{"ID" : "46", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_377.LSTM_b_ifog3_U", "Parent" : "14"},
	{"ID" : "47", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_377.BN2_gamma3_U", "Parent" : "14"},
	{"ID" : "48", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_377.BN2_beta3_U", "Parent" : "14"},
	{"ID" : "49", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_377.BN2_mean3_U", "Parent" : "14"},
	{"ID" : "50", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_377.BN2_var3_U", "Parent" : "14"},
	{"ID" : "51", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_377.tokens4_U", "Parent" : "14"},
	{"ID" : "52", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_377.Emb4_U", "Parent" : "14"},
	{"ID" : "53", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_377.BN1_var4_U", "Parent" : "14"},
	{"ID" : "54", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_377.BN1_gamma4_U", "Parent" : "14"},
	{"ID" : "55", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_377.LSTM_W_ifog4_U", "Parent" : "14"},
	{"ID" : "56", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_377.LSTM_R_ifog4_U", "Parent" : "14"},
	{"ID" : "57", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_377.LSTM_b_ifog4_U", "Parent" : "14"},
	{"ID" : "58", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_377.BN2_gamma4_U", "Parent" : "14"},
	{"ID" : "59", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_377.BN2_beta4_U", "Parent" : "14"},
	{"ID" : "60", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_377.BN2_mean4_U", "Parent" : "14"},
	{"ID" : "61", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_377.BN2_var4_U", "Parent" : "14"},
	{"ID" : "62", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_377.Y_1_U", "Parent" : "14"},
	{"ID" : "63", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_377.U_U", "Parent" : "14"},
	{"ID" : "64", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_377.h_U", "Parent" : "14"},
	{"ID" : "65", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_377.c_U", "Parent" : "14"},
	{"ID" : "66", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_377.Y_2_U", "Parent" : "14"},
	{"ID" : "67", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_377.U_1_U", "Parent" : "14"},
	{"ID" : "68", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_377.h_1_U", "Parent" : "14"},
	{"ID" : "69", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_377.c5_U", "Parent" : "14"},
	{"ID" : "70", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_377.Y_3_U", "Parent" : "14"},
	{"ID" : "71", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_377.U_2_U", "Parent" : "14"},
	{"ID" : "72", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_377.h_2_U", "Parent" : "14"},
	{"ID" : "73", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_377.c10_U", "Parent" : "14"},
	{"ID" : "74", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_377.Y_4_U", "Parent" : "14"},
	{"ID" : "75", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_377.U_3_U", "Parent" : "14"},
	{"ID" : "76", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_377.h_3_U", "Parent" : "14"},
	{"ID" : "77", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_377.c15_U", "Parent" : "14"},
	{"ID" : "78", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_377.X0_1_U", "Parent" : "14"},
	{"ID" : "79", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_377.Y_U", "Parent" : "14"},
	{"ID" : "80", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_377.U_4_U", "Parent" : "14"},
	{"ID" : "81", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_377.h_4_U", "Parent" : "14"},
	{"ID" : "82", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_377.c20_U", "Parent" : "14"},
	{"ID" : "83", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_377.grp_lstm_forward_unidir_fu_994", "Parent" : "14", "Child" : ["84", "85", "102", "103", "104", "105", "106", "107", "108", "109"],
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
			{"Name" : "x", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "W_ifog", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "R_ifog", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "b_ifog", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "h_last", "Type" : "Memory", "Direction" : "IO"},
			{"Name" : "c_last", "Type" : "Memory", "Direction" : "IO"},
			{"Name" : "table_exp_Z1_array_s", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "85", "SubInstance" : "grp_generic_tanh_float_s_fu_325", "Port" : "table_exp_Z1_array_s"}]},
			{"Name" : "table_f_Z3_array_V", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "85", "SubInstance" : "grp_generic_tanh_float_s_fu_325", "Port" : "table_f_Z3_array_V"}]},
			{"Name" : "table_f_Z2_array_V", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "85", "SubInstance" : "grp_generic_tanh_float_s_fu_325", "Port" : "table_f_Z2_array_V"}]}]},
	{"ID" : "84", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_377.grp_lstm_forward_unidir_fu_994.z_U", "Parent" : "83"},
	{"ID" : "85", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_377.grp_lstm_forward_unidir_fu_994.grp_generic_tanh_float_s_fu_325", "Parent" : "83", "Child" : ["86", "95", "96", "97", "98", "99", "100", "101"],
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
					{"ID" : "86", "SubInstance" : "grp_exp_generic_double_s_fu_89", "Port" : "table_exp_Z1_array_s"}]},
			{"Name" : "table_f_Z3_array_V", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "86", "SubInstance" : "grp_exp_generic_double_s_fu_89", "Port" : "table_f_Z3_array_V"}]},
			{"Name" : "table_f_Z2_array_V", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "86", "SubInstance" : "grp_exp_generic_double_s_fu_89", "Port" : "table_f_Z2_array_V"}]}]},
	{"ID" : "86", "Level" : "4", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_377.grp_lstm_forward_unidir_fu_994.grp_generic_tanh_float_s_fu_325.grp_exp_generic_double_s_fu_89", "Parent" : "85", "Child" : ["87", "88", "89", "90", "91", "92", "93", "94"],
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
	{"ID" : "87", "Level" : "5", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_377.grp_lstm_forward_unidir_fu_994.grp_generic_tanh_float_s_fu_325.grp_exp_generic_double_s_fu_89.table_exp_Z1_array_s_U", "Parent" : "86"},
	{"ID" : "88", "Level" : "5", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_377.grp_lstm_forward_unidir_fu_994.grp_generic_tanh_float_s_fu_325.grp_exp_generic_double_s_fu_89.table_f_Z3_array_V_U", "Parent" : "86"},
	{"ID" : "89", "Level" : "5", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_377.grp_lstm_forward_unidir_fu_994.grp_generic_tanh_float_s_fu_325.grp_exp_generic_double_s_fu_89.table_f_Z2_array_V_U", "Parent" : "86"},
	{"ID" : "90", "Level" : "5", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_377.grp_lstm_forward_unidir_fu_994.grp_generic_tanh_float_s_fu_325.grp_exp_generic_double_s_fu_89.main_mul_72ns_13s_84_5_1_U13", "Parent" : "86"},
	{"ID" : "91", "Level" : "5", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_377.grp_lstm_forward_unidir_fu_994.grp_generic_tanh_float_s_fu_325.grp_exp_generic_double_s_fu_89.main_mul_36ns_43ns_79_2_1_U14", "Parent" : "86"},
	{"ID" : "92", "Level" : "5", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_377.grp_lstm_forward_unidir_fu_994.grp_generic_tanh_float_s_fu_325.grp_exp_generic_double_s_fu_89.main_mul_44ns_49ns_93_2_1_U15", "Parent" : "86"},
	{"ID" : "93", "Level" : "5", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_377.grp_lstm_forward_unidir_fu_994.grp_generic_tanh_float_s_fu_325.grp_exp_generic_double_s_fu_89.main_mul_50ns_50ns_100_2_1_U16", "Parent" : "86"},
	{"ID" : "94", "Level" : "5", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_377.grp_lstm_forward_unidir_fu_994.grp_generic_tanh_float_s_fu_325.grp_exp_generic_double_s_fu_89.main_mac_muladd_16ns_16s_19s_31_1_1_U17", "Parent" : "86"},
	{"ID" : "95", "Level" : "4", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_377.grp_lstm_forward_unidir_fu_994.grp_generic_tanh_float_s_fu_325.main_faddfsub_32ns_32ns_32_5_full_dsp_1_U27", "Parent" : "85"},
	{"ID" : "96", "Level" : "4", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_377.grp_lstm_forward_unidir_fu_994.grp_generic_tanh_float_s_fu_325.main_fmul_32ns_32ns_32_4_max_dsp_1_U28", "Parent" : "85"},
	{"ID" : "97", "Level" : "4", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_377.grp_lstm_forward_unidir_fu_994.grp_generic_tanh_float_s_fu_325.main_fdiv_32ns_32ns_32_16_1_U29", "Parent" : "85"},
	{"ID" : "98", "Level" : "4", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_377.grp_lstm_forward_unidir_fu_994.grp_generic_tanh_float_s_fu_325.main_fptrunc_64ns_32_2_1_U30", "Parent" : "85"},
	{"ID" : "99", "Level" : "4", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_377.grp_lstm_forward_unidir_fu_994.grp_generic_tanh_float_s_fu_325.main_fpext_32ns_64_2_1_U31", "Parent" : "85"},
	{"ID" : "100", "Level" : "4", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_377.grp_lstm_forward_unidir_fu_994.grp_generic_tanh_float_s_fu_325.main_fcmp_32ns_32ns_1_2_1_U32", "Parent" : "85"},
	{"ID" : "101", "Level" : "4", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_377.grp_lstm_forward_unidir_fu_994.grp_generic_tanh_float_s_fu_325.main_dadd_64ns_64ns_64_5_full_dsp_1_U33", "Parent" : "85"},
	{"ID" : "102", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_377.grp_lstm_forward_unidir_fu_994.main_fadd_32ns_32ns_32_5_full_dsp_1_U40", "Parent" : "83"},
	{"ID" : "103", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_377.grp_lstm_forward_unidir_fu_994.main_fadd_32ns_32ns_32_5_full_dsp_1_U41", "Parent" : "83"},
	{"ID" : "104", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_377.grp_lstm_forward_unidir_fu_994.main_fmul_32ns_32ns_32_4_max_dsp_1_U42", "Parent" : "83"},
	{"ID" : "105", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_377.grp_lstm_forward_unidir_fu_994.main_fmul_32ns_32ns_32_4_max_dsp_1_U43", "Parent" : "83"},
	{"ID" : "106", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_377.grp_lstm_forward_unidir_fu_994.main_fdiv_32ns_32ns_32_16_1_U44", "Parent" : "83"},
	{"ID" : "107", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_377.grp_lstm_forward_unidir_fu_994.main_fdiv_32ns_32ns_32_16_1_U45", "Parent" : "83"},
	{"ID" : "108", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_377.grp_lstm_forward_unidir_fu_994.main_fexp_32ns_32ns_32_9_full_dsp_1_U46", "Parent" : "83"},
	{"ID" : "109", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_377.grp_lstm_forward_unidir_fu_994.main_fexp_32ns_32ns_32_9_full_dsp_1_U47", "Parent" : "83"},
	{"ID" : "110", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_377.grp_generic_tanh_float_s_fu_1025", "Parent" : "14", "Child" : ["111", "120", "121", "122", "123", "124", "125", "126"],
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
					{"ID" : "111", "SubInstance" : "grp_exp_generic_double_s_fu_89", "Port" : "table_exp_Z1_array_s"}]},
			{"Name" : "table_f_Z3_array_V", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "111", "SubInstance" : "grp_exp_generic_double_s_fu_89", "Port" : "table_f_Z3_array_V"}]},
			{"Name" : "table_f_Z2_array_V", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "111", "SubInstance" : "grp_exp_generic_double_s_fu_89", "Port" : "table_f_Z2_array_V"}]}]},
	{"ID" : "111", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_377.grp_generic_tanh_float_s_fu_1025.grp_exp_generic_double_s_fu_89", "Parent" : "110", "Child" : ["112", "113", "114", "115", "116", "117", "118", "119"],
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
	{"ID" : "112", "Level" : "4", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_377.grp_generic_tanh_float_s_fu_1025.grp_exp_generic_double_s_fu_89.table_exp_Z1_array_s_U", "Parent" : "111"},
	{"ID" : "113", "Level" : "4", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_377.grp_generic_tanh_float_s_fu_1025.grp_exp_generic_double_s_fu_89.table_f_Z3_array_V_U", "Parent" : "111"},
	{"ID" : "114", "Level" : "4", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_377.grp_generic_tanh_float_s_fu_1025.grp_exp_generic_double_s_fu_89.table_f_Z2_array_V_U", "Parent" : "111"},
	{"ID" : "115", "Level" : "4", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_377.grp_generic_tanh_float_s_fu_1025.grp_exp_generic_double_s_fu_89.main_mul_72ns_13s_84_5_1_U13", "Parent" : "111"},
	{"ID" : "116", "Level" : "4", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_377.grp_generic_tanh_float_s_fu_1025.grp_exp_generic_double_s_fu_89.main_mul_36ns_43ns_79_2_1_U14", "Parent" : "111"},
	{"ID" : "117", "Level" : "4", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_377.grp_generic_tanh_float_s_fu_1025.grp_exp_generic_double_s_fu_89.main_mul_44ns_49ns_93_2_1_U15", "Parent" : "111"},
	{"ID" : "118", "Level" : "4", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_377.grp_generic_tanh_float_s_fu_1025.grp_exp_generic_double_s_fu_89.main_mul_50ns_50ns_100_2_1_U16", "Parent" : "111"},
	{"ID" : "119", "Level" : "4", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_377.grp_generic_tanh_float_s_fu_1025.grp_exp_generic_double_s_fu_89.main_mac_muladd_16ns_16s_19s_31_1_1_U17", "Parent" : "111"},
	{"ID" : "120", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_377.grp_generic_tanh_float_s_fu_1025.main_faddfsub_32ns_32ns_32_5_full_dsp_1_U27", "Parent" : "110"},
	{"ID" : "121", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_377.grp_generic_tanh_float_s_fu_1025.main_fmul_32ns_32ns_32_4_max_dsp_1_U28", "Parent" : "110"},
	{"ID" : "122", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_377.grp_generic_tanh_float_s_fu_1025.main_fdiv_32ns_32ns_32_16_1_U29", "Parent" : "110"},
	{"ID" : "123", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_377.grp_generic_tanh_float_s_fu_1025.main_fptrunc_64ns_32_2_1_U30", "Parent" : "110"},
	{"ID" : "124", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_377.grp_generic_tanh_float_s_fu_1025.main_fpext_32ns_64_2_1_U31", "Parent" : "110"},
	{"ID" : "125", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_377.grp_generic_tanh_float_s_fu_1025.main_fcmp_32ns_32ns_1_2_1_U32", "Parent" : "110"},
	{"ID" : "126", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_377.grp_generic_tanh_float_s_fu_1025.main_dadd_64ns_64ns_64_5_full_dsp_1_U33", "Parent" : "110"},
	{"ID" : "127", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_377.grp_bn_vector_fu_1036", "Parent" : "14", "Child" : ["128", "129", "130", "131"],
		"CDFG" : "bn_vector",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "0", "ap_idle" : "1",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "1441", "EstimateLatencyMax" : "1441",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"Port" : [
			{"Name" : "v", "Type" : "Memory", "Direction" : "IO"},
			{"Name" : "gamma", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "beta", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "mean", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "var", "Type" : "Memory", "Direction" : "I"}]},
	{"ID" : "128", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_377.grp_bn_vector_fu_1036.main_faddfsub_32ns_32ns_32_5_full_dsp_1_U56", "Parent" : "127"},
	{"ID" : "129", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_377.grp_bn_vector_fu_1036.main_fmul_32ns_32ns_32_4_max_dsp_1_U57", "Parent" : "127"},
	{"ID" : "130", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_377.grp_bn_vector_fu_1036.main_fdiv_32ns_32ns_32_16_1_U58", "Parent" : "127"},
	{"ID" : "131", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_377.grp_bn_vector_fu_1036.main_fsqrt_32ns_32ns_32_12_1_U59", "Parent" : "127"},
	{"ID" : "132", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_377.grp_avgpool1d_P_fu_1065", "Parent" : "14", "Child" : ["133", "134"],
		"CDFG" : "avgpool1d_P",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "0", "ap_idle" : "1",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "135961", "EstimateLatencyMax" : "135961",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"Port" : [
			{"Name" : "Z", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "U", "Type" : "Memory", "Direction" : "O"}]},
	{"ID" : "133", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_377.grp_avgpool1d_P_fu_1065.main_fadd_32ns_32ns_32_5_full_dsp_1_U98", "Parent" : "132"},
	{"ID" : "134", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_377.grp_avgpool1d_P_fu_1065.main_fdiv_32ns_32ns_32_16_1_U99", "Parent" : "132"},
	{"ID" : "135", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_377.grp_avgpool1d_P_1_fu_1071", "Parent" : "14", "Child" : ["136", "137"],
		"CDFG" : "avgpool1d_P_1",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "0", "ap_idle" : "1",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "71449", "EstimateLatencyMax" : "71449",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"Port" : [
			{"Name" : "Z", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "U", "Type" : "Memory", "Direction" : "O"}]},
	{"ID" : "136", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_377.grp_avgpool1d_P_1_fu_1071.main_fadd_32ns_32ns_32_5_full_dsp_1_U89", "Parent" : "135"},
	{"ID" : "137", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_377.grp_avgpool1d_P_1_fu_1071.main_fdiv_32ns_32ns_32_16_1_U90", "Parent" : "135"},
	{"ID" : "138", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_377.grp_avgpool1d_P_2_fu_1077", "Parent" : "14", "Child" : ["139", "140"],
		"CDFG" : "avgpool1d_P_2",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "0", "ap_idle" : "1",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "39193", "EstimateLatencyMax" : "39193",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"Port" : [
			{"Name" : "Z", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "U", "Type" : "Memory", "Direction" : "O"}]},
	{"ID" : "139", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_377.grp_avgpool1d_P_2_fu_1077.main_fadd_32ns_32ns_32_5_full_dsp_1_U80", "Parent" : "138"},
	{"ID" : "140", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_377.grp_avgpool1d_P_2_fu_1077.main_fdiv_32ns_32ns_32_16_1_U81", "Parent" : "138"},
	{"ID" : "141", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_377.grp_avgpool1d_P_3_fu_1083", "Parent" : "14", "Child" : ["142", "143"],
		"CDFG" : "avgpool1d_P_3",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "0", "ap_idle" : "1",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "23065", "EstimateLatencyMax" : "23065",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"Port" : [
			{"Name" : "Z", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "U", "Type" : "Memory", "Direction" : "O"}]},
	{"ID" : "142", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_377.grp_avgpool1d_P_3_fu_1083.main_fadd_32ns_32ns_32_5_full_dsp_1_U71", "Parent" : "141"},
	{"ID" : "143", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_377.grp_avgpool1d_P_3_fu_1083.main_fdiv_32ns_32ns_32_16_1_U72", "Parent" : "141"},
	{"ID" : "144", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_377.grp_avgpool1d_P_4_fu_1089", "Parent" : "14", "Child" : ["145", "146"],
		"CDFG" : "avgpool1d_P_4",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "0", "ap_idle" : "1",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "15001", "EstimateLatencyMax" : "15001",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"Port" : [
			{"Name" : "Z", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "U", "Type" : "Memory", "Direction" : "O"}]},
	{"ID" : "145", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_377.grp_avgpool1d_P_4_fu_1089.main_fadd_32ns_32ns_32_5_full_dsp_1_U8", "Parent" : "144"},
	{"ID" : "146", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_377.grp_avgpool1d_P_4_fu_1089.main_fdiv_32ns_32ns_32_16_1_U9", "Parent" : "144"},
	{"ID" : "147", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_377.grp_conv1d_valid_dyn_fu_1095", "Parent" : "14", "Child" : ["148", "149", "150"],
		"CDFG" : "conv1d_valid_dyn",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "0", "ap_idle" : "1",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "45712513", "EstimateLatencyMax" : "45712513",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"Port" : [
			{"Name" : "X", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "Y", "Type" : "Memory", "Direction" : "O"},
			{"Name" : "ConvW4", "Type" : "Memory", "Direction" : "I"}]},
	{"ID" : "148", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_377.grp_conv1d_valid_dyn_fu_1095.ConvW4_U", "Parent" : "147"},
	{"ID" : "149", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_377.grp_conv1d_valid_dyn_fu_1095.main_fadd_32ns_32ns_32_5_full_dsp_1_U93", "Parent" : "147"},
	{"ID" : "150", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_377.grp_conv1d_valid_dyn_fu_1095.main_fmul_32ns_32ns_32_4_max_dsp_1_U94", "Parent" : "147"},
	{"ID" : "151", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_377.grp_conv1d_valid_dyn_1_fu_1103", "Parent" : "14", "Child" : ["152", "153", "154", "155"],
		"CDFG" : "conv1d_valid_dyn_1",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "0", "ap_idle" : "1",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "22856257", "EstimateLatencyMax" : "22856257",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"Port" : [
			{"Name" : "Y", "Type" : "Memory", "Direction" : "O"},
			{"Name" : "X011", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "ConvW3", "Type" : "Memory", "Direction" : "I"}]},
	{"ID" : "152", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_377.grp_conv1d_valid_dyn_1_fu_1103.X011_U", "Parent" : "151"},
	{"ID" : "153", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_377.grp_conv1d_valid_dyn_1_fu_1103.ConvW3_U", "Parent" : "151"},
	{"ID" : "154", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_377.grp_conv1d_valid_dyn_1_fu_1103.main_fadd_32ns_32ns_32_5_full_dsp_1_U84", "Parent" : "151"},
	{"ID" : "155", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_377.grp_conv1d_valid_dyn_1_fu_1103.main_fmul_32ns_32ns_32_4_max_dsp_1_U85", "Parent" : "151"},
	{"ID" : "156", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_377.grp_conv1d_valid_dyn_2_fu_1112", "Parent" : "14", "Child" : ["157", "158", "159", "160"],
		"CDFG" : "conv1d_valid_dyn_2",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "0", "ap_idle" : "1",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "11428129", "EstimateLatencyMax" : "11428129",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"Port" : [
			{"Name" : "Y", "Type" : "Memory", "Direction" : "O"},
			{"Name" : "X06", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "ConvW2", "Type" : "Memory", "Direction" : "I"}]},
	{"ID" : "157", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_377.grp_conv1d_valid_dyn_2_fu_1112.X06_U", "Parent" : "156"},
	{"ID" : "158", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_377.grp_conv1d_valid_dyn_2_fu_1112.ConvW2_U", "Parent" : "156"},
	{"ID" : "159", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_377.grp_conv1d_valid_dyn_2_fu_1112.main_fadd_32ns_32ns_32_5_full_dsp_1_U75", "Parent" : "156"},
	{"ID" : "160", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_377.grp_conv1d_valid_dyn_2_fu_1112.main_fmul_32ns_32ns_32_4_max_dsp_1_U76", "Parent" : "156"},
	{"ID" : "161", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_377.grp_conv1d_valid_dyn_3_fu_1121", "Parent" : "14", "Child" : ["162", "163", "164", "165"],
		"CDFG" : "conv1d_valid_dyn_3",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "0", "ap_idle" : "1",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "5714065", "EstimateLatencyMax" : "5714065",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"Port" : [
			{"Name" : "Y", "Type" : "Memory", "Direction" : "O"},
			{"Name" : "X01", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "ConvW1", "Type" : "Memory", "Direction" : "I"}]},
	{"ID" : "162", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_377.grp_conv1d_valid_dyn_3_fu_1121.X01_U", "Parent" : "161"},
	{"ID" : "163", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_377.grp_conv1d_valid_dyn_3_fu_1121.ConvW1_U", "Parent" : "161"},
	{"ID" : "164", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_377.grp_conv1d_valid_dyn_3_fu_1121.main_fadd_32ns_32ns_32_5_full_dsp_1_U66", "Parent" : "161"},
	{"ID" : "165", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_377.grp_conv1d_valid_dyn_3_fu_1121.main_fmul_32ns_32ns_32_4_max_dsp_1_U67", "Parent" : "161"},
	{"ID" : "166", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_377.grp_conv1d_valid_dyn_4_fu_1130", "Parent" : "14", "Child" : ["167", "168", "169", "170"],
		"CDFG" : "conv1d_valid_dyn_4",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "0", "ap_idle" : "1",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "2857033", "EstimateLatencyMax" : "2857033",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"Port" : [
			{"Name" : "Y", "Type" : "Memory", "Direction" : "O"},
			{"Name" : "X0", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "ConvW0", "Type" : "Memory", "Direction" : "I"}]},
	{"ID" : "167", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_377.grp_conv1d_valid_dyn_4_fu_1130.X0_U", "Parent" : "166"},
	{"ID" : "168", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_377.grp_conv1d_valid_dyn_4_fu_1130.ConvW0_U", "Parent" : "166"},
	{"ID" : "169", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_377.grp_conv1d_valid_dyn_4_fu_1130.main_fadd_32ns_32ns_32_5_full_dsp_1_U1", "Parent" : "166"},
	{"ID" : "170", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_377.grp_conv1d_valid_dyn_4_fu_1130.main_fmul_32ns_32ns_32_4_max_dsp_1_U2", "Parent" : "166"},
	{"ID" : "171", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_377.main_fadd_32ns_32ns_32_5_full_dsp_1_U102", "Parent" : "14"},
	{"ID" : "172", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_377.main_fmul_32ns_32ns_32_4_max_dsp_1_U103", "Parent" : "14"},
	{"ID" : "173", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_377.main_fdiv_32ns_32ns_32_16_1_U104", "Parent" : "14"},
	{"ID" : "174", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_377.main_fcmp_32ns_32ns_1_2_1_U105", "Parent" : "14"},
	{"ID" : "175", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_377.main_fsqrt_32ns_32ns_32_12_1_U106", "Parent" : "14"},
	{"ID" : "176", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.grp_bn_vector_1_fu_501", "Parent" : "0", "Child" : ["177", "178", "179", "180"],
		"CDFG" : "bn_vector_1",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "0", "ap_idle" : "1",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "5761", "EstimateLatencyMax" : "5761",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"Port" : [
			{"Name" : "v", "Type" : "Memory", "Direction" : "IO"},
			{"Name" : "gamma", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "beta", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "mean", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "var", "Type" : "Memory", "Direction" : "I"}]},
	{"ID" : "177", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_bn_vector_1_fu_501.main_faddfsub_32ns_32ns_32_5_full_dsp_1_U132", "Parent" : "176"},
	{"ID" : "178", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_bn_vector_1_fu_501.main_fmul_32ns_32ns_32_4_max_dsp_1_U133", "Parent" : "176"},
	{"ID" : "179", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_bn_vector_1_fu_501.main_fdiv_32ns_32ns_32_16_1_U134", "Parent" : "176"},
	{"ID" : "180", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_bn_vector_1_fu_501.main_fsqrt_32ns_32ns_32_12_1_U135", "Parent" : "176"},
	{"ID" : "181", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.main_fadd_32ns_32ns_32_5_full_dsp_1_U141", "Parent" : "0"},
	{"ID" : "182", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.main_fmul_32ns_32ns_32_4_max_dsp_1_U142", "Parent" : "0"},
	{"ID" : "183", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.main_fcmp_32ns_32ns_1_2_1_U143", "Parent" : "0"}]}


set ArgLastReadFirstWriteLatency {
	main {
		X0 {Type I LastRead -1 FirstWrite -1}
		ConvW0 {Type I LastRead -1 FirstWrite -1}
		BN1_var0 {Type I LastRead -1 FirstWrite -1}
		BN1_gamma0 {Type I LastRead -1 FirstWrite -1}
		table_exp_Z1_array_s {Type I LastRead -1 FirstWrite -1}
		table_f_Z3_array_V {Type I LastRead -1 FirstWrite -1}
		table_f_Z2_array_V {Type I LastRead -1 FirstWrite -1}
		LSTM_W_ifog0 {Type I LastRead -1 FirstWrite -1}
		LSTM_R_ifog0 {Type I LastRead -1 FirstWrite -1}
		LSTM_b_ifog0 {Type I LastRead -1 FirstWrite -1}
		BN2_gamma0 {Type I LastRead -1 FirstWrite -1}
		BN2_beta0 {Type I LastRead -1 FirstWrite -1}
		BN2_mean0 {Type I LastRead -1 FirstWrite -1}
		BN2_var0 {Type I LastRead -1 FirstWrite -1}
		X01 {Type I LastRead -1 FirstWrite -1}
		ConvW1 {Type I LastRead -1 FirstWrite -1}
		BN1_var1 {Type I LastRead -1 FirstWrite -1}
		BN1_gamma1 {Type I LastRead -1 FirstWrite -1}
		LSTM_W_ifog1 {Type I LastRead -1 FirstWrite -1}
		LSTM_R_ifog1 {Type I LastRead -1 FirstWrite -1}
		LSTM_b_ifog1 {Type I LastRead -1 FirstWrite -1}
		BN2_gamma1 {Type I LastRead -1 FirstWrite -1}
		BN2_beta1 {Type I LastRead -1 FirstWrite -1}
		BN2_mean1 {Type I LastRead -1 FirstWrite -1}
		BN2_var1 {Type I LastRead -1 FirstWrite -1}
		X06 {Type I LastRead -1 FirstWrite -1}
		ConvW2 {Type I LastRead -1 FirstWrite -1}
		BN1_var2 {Type I LastRead -1 FirstWrite -1}
		BN1_gamma2 {Type I LastRead -1 FirstWrite -1}
		LSTM_W_ifog2 {Type I LastRead -1 FirstWrite -1}
		LSTM_R_ifog2 {Type I LastRead -1 FirstWrite -1}
		LSTM_b_ifog2 {Type I LastRead -1 FirstWrite -1}
		BN2_gamma2 {Type I LastRead -1 FirstWrite -1}
		BN2_beta2 {Type I LastRead -1 FirstWrite -1}
		BN2_mean2 {Type I LastRead -1 FirstWrite -1}
		BN2_var2 {Type I LastRead -1 FirstWrite -1}
		X011 {Type I LastRead -1 FirstWrite -1}
		ConvW3 {Type I LastRead -1 FirstWrite -1}
		BN1_var3 {Type I LastRead -1 FirstWrite -1}
		BN1_gamma3 {Type I LastRead -1 FirstWrite -1}
		LSTM_W_ifog3 {Type I LastRead -1 FirstWrite -1}
		LSTM_R_ifog3 {Type I LastRead -1 FirstWrite -1}
		LSTM_b_ifog3 {Type I LastRead -1 FirstWrite -1}
		BN2_gamma3 {Type I LastRead -1 FirstWrite -1}
		BN2_beta3 {Type I LastRead -1 FirstWrite -1}
		BN2_mean3 {Type I LastRead -1 FirstWrite -1}
		BN2_var3 {Type I LastRead -1 FirstWrite -1}
		tokens4 {Type I LastRead -1 FirstWrite -1}
		Emb4 {Type I LastRead -1 FirstWrite -1}
		ConvW4 {Type I LastRead -1 FirstWrite -1}
		BN1_var4 {Type I LastRead -1 FirstWrite -1}
		BN1_gamma4 {Type I LastRead -1 FirstWrite -1}
		LSTM_W_ifog4 {Type I LastRead -1 FirstWrite -1}
		LSTM_R_ifog4 {Type I LastRead -1 FirstWrite -1}
		LSTM_b_ifog4 {Type I LastRead -1 FirstWrite -1}
		BN2_gamma4 {Type I LastRead -1 FirstWrite -1}
		BN2_beta4 {Type I LastRead -1 FirstWrite -1}
		BN2_mean4 {Type I LastRead -1 FirstWrite -1}
		BN2_var4 {Type I LastRead -1 FirstWrite -1}
		fc_0_W {Type I LastRead -1 FirstWrite -1}
		fc_0_bn_gamma {Type I LastRead -1 FirstWrite -1}
		fc_0_bn_beta {Type I LastRead -1 FirstWrite -1}
		fc_0_bn_mean {Type I LastRead -1 FirstWrite -1}
		fc_0_bn_var {Type I LastRead -1 FirstWrite -1}
		fc_1_W {Type I LastRead -1 FirstWrite -1}
		fc_1_bn_gamma {Type I LastRead -1 FirstWrite -1}
		fc_1_bn_beta {Type I LastRead -1 FirstWrite -1}
		fc_1_bn_mean {Type I LastRead -1 FirstWrite -1}
		fc_1_bn_var {Type I LastRead -1 FirstWrite -1}}
	run_all_slices_unrol {
		merged {Type IO LastRead 49 FirstWrite 1}
		X0 {Type I LastRead -1 FirstWrite -1}
		ConvW0 {Type I LastRead -1 FirstWrite -1}
		BN1_var0 {Type I LastRead -1 FirstWrite -1}
		BN1_gamma0 {Type I LastRead -1 FirstWrite -1}
		table_exp_Z1_array_s {Type I LastRead -1 FirstWrite -1}
		table_f_Z3_array_V {Type I LastRead -1 FirstWrite -1}
		table_f_Z2_array_V {Type I LastRead -1 FirstWrite -1}
		LSTM_W_ifog0 {Type I LastRead -1 FirstWrite -1}
		LSTM_R_ifog0 {Type I LastRead -1 FirstWrite -1}
		LSTM_b_ifog0 {Type I LastRead -1 FirstWrite -1}
		BN2_gamma0 {Type I LastRead -1 FirstWrite -1}
		BN2_beta0 {Type I LastRead -1 FirstWrite -1}
		BN2_mean0 {Type I LastRead -1 FirstWrite -1}
		BN2_var0 {Type I LastRead -1 FirstWrite -1}
		X01 {Type I LastRead -1 FirstWrite -1}
		ConvW1 {Type I LastRead -1 FirstWrite -1}
		BN1_var1 {Type I LastRead -1 FirstWrite -1}
		BN1_gamma1 {Type I LastRead -1 FirstWrite -1}
		LSTM_W_ifog1 {Type I LastRead -1 FirstWrite -1}
		LSTM_R_ifog1 {Type I LastRead -1 FirstWrite -1}
		LSTM_b_ifog1 {Type I LastRead -1 FirstWrite -1}
		BN2_gamma1 {Type I LastRead -1 FirstWrite -1}
		BN2_beta1 {Type I LastRead -1 FirstWrite -1}
		BN2_mean1 {Type I LastRead -1 FirstWrite -1}
		BN2_var1 {Type I LastRead -1 FirstWrite -1}
		X06 {Type I LastRead -1 FirstWrite -1}
		ConvW2 {Type I LastRead -1 FirstWrite -1}
		BN1_var2 {Type I LastRead -1 FirstWrite -1}
		BN1_gamma2 {Type I LastRead -1 FirstWrite -1}
		LSTM_W_ifog2 {Type I LastRead -1 FirstWrite -1}
		LSTM_R_ifog2 {Type I LastRead -1 FirstWrite -1}
		LSTM_b_ifog2 {Type I LastRead -1 FirstWrite -1}
		BN2_gamma2 {Type I LastRead -1 FirstWrite -1}
		BN2_beta2 {Type I LastRead -1 FirstWrite -1}
		BN2_mean2 {Type I LastRead -1 FirstWrite -1}
		BN2_var2 {Type I LastRead -1 FirstWrite -1}
		X011 {Type I LastRead -1 FirstWrite -1}
		ConvW3 {Type I LastRead -1 FirstWrite -1}
		BN1_var3 {Type I LastRead -1 FirstWrite -1}
		BN1_gamma3 {Type I LastRead -1 FirstWrite -1}
		LSTM_W_ifog3 {Type I LastRead -1 FirstWrite -1}
		LSTM_R_ifog3 {Type I LastRead -1 FirstWrite -1}
		LSTM_b_ifog3 {Type I LastRead -1 FirstWrite -1}
		BN2_gamma3 {Type I LastRead -1 FirstWrite -1}
		BN2_beta3 {Type I LastRead -1 FirstWrite -1}
		BN2_mean3 {Type I LastRead -1 FirstWrite -1}
		BN2_var3 {Type I LastRead -1 FirstWrite -1}
		tokens4 {Type I LastRead -1 FirstWrite -1}
		Emb4 {Type I LastRead -1 FirstWrite -1}
		ConvW4 {Type I LastRead -1 FirstWrite -1}
		BN1_var4 {Type I LastRead -1 FirstWrite -1}
		BN1_gamma4 {Type I LastRead -1 FirstWrite -1}
		LSTM_W_ifog4 {Type I LastRead -1 FirstWrite -1}
		LSTM_R_ifog4 {Type I LastRead -1 FirstWrite -1}
		LSTM_b_ifog4 {Type I LastRead -1 FirstWrite -1}
		BN2_gamma4 {Type I LastRead -1 FirstWrite -1}
		BN2_beta4 {Type I LastRead -1 FirstWrite -1}
		BN2_mean4 {Type I LastRead -1 FirstWrite -1}
		BN2_var4 {Type I LastRead -1 FirstWrite -1}}
	lstm_forward_unidir {
		x {Type I LastRead 4 FirstWrite -1}
		W_ifog {Type I LastRead 6 FirstWrite -1}
		R_ifog {Type I LastRead 7 FirstWrite -1}
		b_ifog {Type I LastRead 3 FirstWrite -1}
		h_last {Type IO LastRead 5 FirstWrite 1}
		c_last {Type IO LastRead 36 FirstWrite 1}
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
	bn_vector {
		v {Type IO LastRead 13 FirstWrite 45}
		gamma {Type I LastRead 34 FirstWrite -1}
		beta {Type I LastRead 34 FirstWrite -1}
		mean {Type I LastRead 13 FirstWrite -1}
		var {Type I LastRead 1 FirstWrite -1}}
	avgpool1d_P {
		Z {Type I LastRead 3 FirstWrite -1}
		U {Type O LastRead -1 FirstWrite 19}}
	avgpool1d_P_1 {
		Z {Type I LastRead 3 FirstWrite -1}
		U {Type O LastRead -1 FirstWrite 19}}
	avgpool1d_P_2 {
		Z {Type I LastRead 3 FirstWrite -1}
		U {Type O LastRead -1 FirstWrite 19}}
	avgpool1d_P_3 {
		Z {Type I LastRead 3 FirstWrite -1}
		U {Type O LastRead -1 FirstWrite 19}}
	avgpool1d_P_4 {
		Z {Type I LastRead 3 FirstWrite -1}
		U {Type O LastRead -1 FirstWrite 19}}
	conv1d_valid_dyn {
		X {Type I LastRead 4 FirstWrite -1}
		Y {Type O LastRead -1 FirstWrite 3}
		ConvW4 {Type I LastRead -1 FirstWrite -1}}
	conv1d_valid_dyn_1 {
		Y {Type O LastRead -1 FirstWrite 3}
		X011 {Type I LastRead -1 FirstWrite -1}
		ConvW3 {Type I LastRead -1 FirstWrite -1}}
	conv1d_valid_dyn_2 {
		Y {Type O LastRead -1 FirstWrite 3}
		X06 {Type I LastRead -1 FirstWrite -1}
		ConvW2 {Type I LastRead -1 FirstWrite -1}}
	conv1d_valid_dyn_3 {
		Y {Type O LastRead -1 FirstWrite 3}
		X01 {Type I LastRead -1 FirstWrite -1}
		ConvW1 {Type I LastRead -1 FirstWrite -1}}
	conv1d_valid_dyn_4 {
		Y {Type O LastRead -1 FirstWrite 3}
		X0 {Type I LastRead -1 FirstWrite -1}
		ConvW0 {Type I LastRead -1 FirstWrite -1}}
	bn_vector_1 {
		v {Type IO LastRead 13 FirstWrite 45}
		gamma {Type I LastRead 34 FirstWrite -1}
		beta {Type I LastRead 34 FirstWrite -1}
		mean {Type I LastRead 13 FirstWrite -1}
		var {Type I LastRead 1 FirstWrite -1}}}

set hasDtUnsupportedChannel 0

set PerformanceInfo {[
	{"Name" : "Latency", "Min" : "96913071", "Max" : "97153071"}
	, {"Name" : "Interval", "Min" : "96913072", "Max" : "97153072"}
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
