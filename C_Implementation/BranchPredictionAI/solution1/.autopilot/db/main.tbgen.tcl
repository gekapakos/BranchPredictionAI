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
	{"ID" : "0", "Level" : "0", "Path" : "`AUTOTB_DUT_INST", "Parent" : "", "Child" : ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "165", "166", "167", "168", "169", "170", "171"],
		"CDFG" : "main",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "0", "ap_idle" : "1",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "121438995", "EstimateLatencyMax" : "121678995",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"WaitState" : [
			{"State" : "ap_ST_fsm_state2", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_run_all_slices_unrol_fu_433"}],
		"Port" : [
			{"Name" : "X_slice", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "10", "SubInstance" : "grp_run_all_slices_unrol_fu_433", "Port" : "X_slice"}]},
			{"Name" : "ConvW0", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "10", "SubInstance" : "grp_run_all_slices_unrol_fu_433", "Port" : "ConvW0"}]},
			{"Name" : "BN1_var0", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "10", "SubInstance" : "grp_run_all_slices_unrol_fu_433", "Port" : "BN1_var0"}]},
			{"Name" : "BN1_gamma0", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "10", "SubInstance" : "grp_run_all_slices_unrol_fu_433", "Port" : "BN1_gamma0"}]},
			{"Name" : "Y", "Type" : "Memory", "Direction" : "IO",
				"SubConnect" : [
					{"ID" : "10", "SubInstance" : "grp_run_all_slices_unrol_fu_433", "Port" : "Y"}]},
			{"Name" : "U_slice", "Type" : "Memory", "Direction" : "IO",
				"SubConnect" : [
					{"ID" : "10", "SubInstance" : "grp_run_all_slices_unrol_fu_433", "Port" : "U_slice"}]},
			{"Name" : "c_slice", "Type" : "Memory", "Direction" : "IO",
				"SubConnect" : [
					{"ID" : "10", "SubInstance" : "grp_run_all_slices_unrol_fu_433", "Port" : "c_slice"}]},
			{"Name" : "table_exp_Z1_array_s", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "10", "SubInstance" : "grp_run_all_slices_unrol_fu_433", "Port" : "table_exp_Z1_array_s"}]},
			{"Name" : "table_f_Z3_array_V", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "10", "SubInstance" : "grp_run_all_slices_unrol_fu_433", "Port" : "table_f_Z3_array_V"}]},
			{"Name" : "table_f_Z2_array_V", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "10", "SubInstance" : "grp_run_all_slices_unrol_fu_433", "Port" : "table_f_Z2_array_V"}]},
			{"Name" : "LSTM_W_ifog0", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "10", "SubInstance" : "grp_run_all_slices_unrol_fu_433", "Port" : "LSTM_W_ifog0"}]},
			{"Name" : "LSTM_R_ifog0", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "10", "SubInstance" : "grp_run_all_slices_unrol_fu_433", "Port" : "LSTM_R_ifog0"}]},
			{"Name" : "LSTM_b_ifog0", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "10", "SubInstance" : "grp_run_all_slices_unrol_fu_433", "Port" : "LSTM_b_ifog0"}]},
			{"Name" : "h_slice", "Type" : "Memory", "Direction" : "IO",
				"SubConnect" : [
					{"ID" : "10", "SubInstance" : "grp_run_all_slices_unrol_fu_433", "Port" : "h_slice"}]},
			{"Name" : "BN2_var0", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "10", "SubInstance" : "grp_run_all_slices_unrol_fu_433", "Port" : "BN2_var0"}]},
			{"Name" : "BN2_gamma0", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "10", "SubInstance" : "grp_run_all_slices_unrol_fu_433", "Port" : "BN2_gamma0"}]},
			{"Name" : "X_slice1", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "10", "SubInstance" : "grp_run_all_slices_unrol_fu_433", "Port" : "X_slice1"}]},
			{"Name" : "ConvW1", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "10", "SubInstance" : "grp_run_all_slices_unrol_fu_433", "Port" : "ConvW1"}]},
			{"Name" : "BN1_var1", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "10", "SubInstance" : "grp_run_all_slices_unrol_fu_433", "Port" : "BN1_var1"}]},
			{"Name" : "BN1_gamma1", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "10", "SubInstance" : "grp_run_all_slices_unrol_fu_433", "Port" : "BN1_gamma1"}]},
			{"Name" : "LSTM_W_ifog1", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "10", "SubInstance" : "grp_run_all_slices_unrol_fu_433", "Port" : "LSTM_W_ifog1"}]},
			{"Name" : "LSTM_R_ifog1", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "10", "SubInstance" : "grp_run_all_slices_unrol_fu_433", "Port" : "LSTM_R_ifog1"}]},
			{"Name" : "LSTM_b_ifog1", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "10", "SubInstance" : "grp_run_all_slices_unrol_fu_433", "Port" : "LSTM_b_ifog1"}]},
			{"Name" : "BN2_var1", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "10", "SubInstance" : "grp_run_all_slices_unrol_fu_433", "Port" : "BN2_var1"}]},
			{"Name" : "BN2_gamma1", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "10", "SubInstance" : "grp_run_all_slices_unrol_fu_433", "Port" : "BN2_gamma1"}]},
			{"Name" : "X_slice2", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "10", "SubInstance" : "grp_run_all_slices_unrol_fu_433", "Port" : "X_slice2"}]},
			{"Name" : "ConvW2", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "10", "SubInstance" : "grp_run_all_slices_unrol_fu_433", "Port" : "ConvW2"}]},
			{"Name" : "BN1_var2", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "10", "SubInstance" : "grp_run_all_slices_unrol_fu_433", "Port" : "BN1_var2"}]},
			{"Name" : "BN1_gamma2", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "10", "SubInstance" : "grp_run_all_slices_unrol_fu_433", "Port" : "BN1_gamma2"}]},
			{"Name" : "LSTM_W_ifog2", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "10", "SubInstance" : "grp_run_all_slices_unrol_fu_433", "Port" : "LSTM_W_ifog2"}]},
			{"Name" : "LSTM_R_ifog2", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "10", "SubInstance" : "grp_run_all_slices_unrol_fu_433", "Port" : "LSTM_R_ifog2"}]},
			{"Name" : "LSTM_b_ifog2", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "10", "SubInstance" : "grp_run_all_slices_unrol_fu_433", "Port" : "LSTM_b_ifog2"}]},
			{"Name" : "BN2_var2", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "10", "SubInstance" : "grp_run_all_slices_unrol_fu_433", "Port" : "BN2_var2"}]},
			{"Name" : "BN2_gamma2", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "10", "SubInstance" : "grp_run_all_slices_unrol_fu_433", "Port" : "BN2_gamma2"}]},
			{"Name" : "X_slice3", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "10", "SubInstance" : "grp_run_all_slices_unrol_fu_433", "Port" : "X_slice3"}]},
			{"Name" : "ConvW3", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "10", "SubInstance" : "grp_run_all_slices_unrol_fu_433", "Port" : "ConvW3"}]},
			{"Name" : "BN1_var3", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "10", "SubInstance" : "grp_run_all_slices_unrol_fu_433", "Port" : "BN1_var3"}]},
			{"Name" : "BN1_gamma3", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "10", "SubInstance" : "grp_run_all_slices_unrol_fu_433", "Port" : "BN1_gamma3"}]},
			{"Name" : "LSTM_W_ifog3", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "10", "SubInstance" : "grp_run_all_slices_unrol_fu_433", "Port" : "LSTM_W_ifog3"}]},
			{"Name" : "LSTM_R_ifog3", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "10", "SubInstance" : "grp_run_all_slices_unrol_fu_433", "Port" : "LSTM_R_ifog3"}]},
			{"Name" : "LSTM_b_ifog3", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "10", "SubInstance" : "grp_run_all_slices_unrol_fu_433", "Port" : "LSTM_b_ifog3"}]},
			{"Name" : "BN2_var3", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "10", "SubInstance" : "grp_run_all_slices_unrol_fu_433", "Port" : "BN2_var3"}]},
			{"Name" : "BN2_gamma3", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "10", "SubInstance" : "grp_run_all_slices_unrol_fu_433", "Port" : "BN2_gamma3"}]},
			{"Name" : "tokens4", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "10", "SubInstance" : "grp_run_all_slices_unrol_fu_433", "Port" : "tokens4"}]},
			{"Name" : "Emb4", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "10", "SubInstance" : "grp_run_all_slices_unrol_fu_433", "Port" : "Emb4"}]},
			{"Name" : "ConvW4", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "10", "SubInstance" : "grp_run_all_slices_unrol_fu_433", "Port" : "ConvW4"}]},
			{"Name" : "BN1_var4", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "10", "SubInstance" : "grp_run_all_slices_unrol_fu_433", "Port" : "BN1_var4"}]},
			{"Name" : "BN1_gamma4", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "10", "SubInstance" : "grp_run_all_slices_unrol_fu_433", "Port" : "BN1_gamma4"}]},
			{"Name" : "LSTM_W_ifog4", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "10", "SubInstance" : "grp_run_all_slices_unrol_fu_433", "Port" : "LSTM_W_ifog4"}]},
			{"Name" : "LSTM_R_ifog4", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "10", "SubInstance" : "grp_run_all_slices_unrol_fu_433", "Port" : "LSTM_R_ifog4"}]},
			{"Name" : "LSTM_b_ifog4", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "10", "SubInstance" : "grp_run_all_slices_unrol_fu_433", "Port" : "LSTM_b_ifog4"}]},
			{"Name" : "BN2_var4", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "10", "SubInstance" : "grp_run_all_slices_unrol_fu_433", "Port" : "BN2_var4"}]},
			{"Name" : "BN2_gamma4", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "10", "SubInstance" : "grp_run_all_slices_unrol_fu_433", "Port" : "BN2_gamma4"}]},
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
	{"ID" : "10", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_433", "Parent" : "0", "Child" : ["11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21", "22", "23", "24", "25", "26", "27", "28", "29", "30", "31", "32", "33", "34", "35", "36", "37", "38", "39", "40", "41", "42", "78", "95", "107", "120", "133", "146", "159", "160", "161", "162", "163", "164"],
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
					{"ID" : "146", "SubInstance" : "grp_conv_bn_act_pool_4_fu_708", "Port" : "X_slice"}]},
			{"Name" : "ConvW0", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "146", "SubInstance" : "grp_conv_bn_act_pool_4_fu_708", "Port" : "ConvW0"}]},
			{"Name" : "BN1_var0", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "146", "SubInstance" : "grp_conv_bn_act_pool_4_fu_708", "Port" : "BN1_var0"}]},
			{"Name" : "BN1_gamma0", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "146", "SubInstance" : "grp_conv_bn_act_pool_4_fu_708", "Port" : "BN1_gamma0"}]},
			{"Name" : "Y", "Type" : "Memory", "Direction" : "IO",
				"SubConnect" : [
					{"ID" : "133", "SubInstance" : "grp_conv_bn_act_pool_3_fu_692", "Port" : "Y"},
					{"ID" : "146", "SubInstance" : "grp_conv_bn_act_pool_4_fu_708", "Port" : "Y"},
					{"ID" : "95", "SubInstance" : "grp_conv_bn_act_pool_fu_645", "Port" : "Y"},
					{"ID" : "107", "SubInstance" : "grp_conv_bn_act_pool_1_fu_660", "Port" : "Y"},
					{"ID" : "120", "SubInstance" : "grp_conv_bn_act_pool_2_fu_676", "Port" : "Y"}]},
			{"Name" : "U_slice", "Type" : "Memory", "Direction" : "IO",
				"SubConnect" : [
					{"ID" : "133", "SubInstance" : "grp_conv_bn_act_pool_3_fu_692", "Port" : "U"},
					{"ID" : "146", "SubInstance" : "grp_conv_bn_act_pool_4_fu_708", "Port" : "U"},
					{"ID" : "95", "SubInstance" : "grp_conv_bn_act_pool_fu_645", "Port" : "U"},
					{"ID" : "42", "SubInstance" : "grp_lstm_forward_unidir_fu_600", "Port" : "U_slice"},
					{"ID" : "107", "SubInstance" : "grp_conv_bn_act_pool_1_fu_660", "Port" : "U"},
					{"ID" : "120", "SubInstance" : "grp_conv_bn_act_pool_2_fu_676", "Port" : "U"}]},
			{"Name" : "c_slice", "Type" : "Memory", "Direction" : "IO",
				"SubConnect" : [
					{"ID" : "42", "SubInstance" : "grp_lstm_forward_unidir_fu_600", "Port" : "c_slice"}]},
			{"Name" : "table_exp_Z1_array_s", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "78", "SubInstance" : "grp_generic_tanh_float_s_fu_634", "Port" : "table_exp_Z1_array_s"},
					{"ID" : "42", "SubInstance" : "grp_lstm_forward_unidir_fu_600", "Port" : "table_exp_Z1_array_s"}]},
			{"Name" : "table_f_Z3_array_V", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "78", "SubInstance" : "grp_generic_tanh_float_s_fu_634", "Port" : "table_f_Z3_array_V"},
					{"ID" : "42", "SubInstance" : "grp_lstm_forward_unidir_fu_600", "Port" : "table_f_Z3_array_V"}]},
			{"Name" : "table_f_Z2_array_V", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "78", "SubInstance" : "grp_generic_tanh_float_s_fu_634", "Port" : "table_f_Z2_array_V"},
					{"ID" : "42", "SubInstance" : "grp_lstm_forward_unidir_fu_600", "Port" : "table_f_Z2_array_V"}]},
			{"Name" : "LSTM_W_ifog0", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "42", "SubInstance" : "grp_lstm_forward_unidir_fu_600", "Port" : "W_ifog"}]},
			{"Name" : "LSTM_R_ifog0", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "42", "SubInstance" : "grp_lstm_forward_unidir_fu_600", "Port" : "R_ifog"}]},
			{"Name" : "LSTM_b_ifog0", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "42", "SubInstance" : "grp_lstm_forward_unidir_fu_600", "Port" : "b_ifog"}]},
			{"Name" : "h_slice", "Type" : "Memory", "Direction" : "IO",
				"SubConnect" : [
					{"ID" : "42", "SubInstance" : "grp_lstm_forward_unidir_fu_600", "Port" : "h_last"}]},
			{"Name" : "BN2_var0", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "BN2_gamma0", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "X_slice1", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "133", "SubInstance" : "grp_conv_bn_act_pool_3_fu_692", "Port" : "X_slice1"}]},
			{"Name" : "ConvW1", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "133", "SubInstance" : "grp_conv_bn_act_pool_3_fu_692", "Port" : "ConvW1"}]},
			{"Name" : "BN1_var1", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "133", "SubInstance" : "grp_conv_bn_act_pool_3_fu_692", "Port" : "BN1_var1"}]},
			{"Name" : "BN1_gamma1", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "133", "SubInstance" : "grp_conv_bn_act_pool_3_fu_692", "Port" : "BN1_gamma1"}]},
			{"Name" : "LSTM_W_ifog1", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "42", "SubInstance" : "grp_lstm_forward_unidir_fu_600", "Port" : "W_ifog"}]},
			{"Name" : "LSTM_R_ifog1", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "42", "SubInstance" : "grp_lstm_forward_unidir_fu_600", "Port" : "R_ifog"}]},
			{"Name" : "LSTM_b_ifog1", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "42", "SubInstance" : "grp_lstm_forward_unidir_fu_600", "Port" : "b_ifog"}]},
			{"Name" : "BN2_var1", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "BN2_gamma1", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "X_slice2", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "120", "SubInstance" : "grp_conv_bn_act_pool_2_fu_676", "Port" : "X_slice2"}]},
			{"Name" : "ConvW2", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "120", "SubInstance" : "grp_conv_bn_act_pool_2_fu_676", "Port" : "ConvW2"}]},
			{"Name" : "BN1_var2", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "120", "SubInstance" : "grp_conv_bn_act_pool_2_fu_676", "Port" : "BN1_var2"}]},
			{"Name" : "BN1_gamma2", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "120", "SubInstance" : "grp_conv_bn_act_pool_2_fu_676", "Port" : "BN1_gamma2"}]},
			{"Name" : "LSTM_W_ifog2", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "42", "SubInstance" : "grp_lstm_forward_unidir_fu_600", "Port" : "W_ifog"}]},
			{"Name" : "LSTM_R_ifog2", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "42", "SubInstance" : "grp_lstm_forward_unidir_fu_600", "Port" : "R_ifog"}]},
			{"Name" : "LSTM_b_ifog2", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "42", "SubInstance" : "grp_lstm_forward_unidir_fu_600", "Port" : "b_ifog"}]},
			{"Name" : "BN2_var2", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "BN2_gamma2", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "X_slice3", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "107", "SubInstance" : "grp_conv_bn_act_pool_1_fu_660", "Port" : "X_slice3"}]},
			{"Name" : "ConvW3", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "107", "SubInstance" : "grp_conv_bn_act_pool_1_fu_660", "Port" : "ConvW3"}]},
			{"Name" : "BN1_var3", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "107", "SubInstance" : "grp_conv_bn_act_pool_1_fu_660", "Port" : "BN1_var3"}]},
			{"Name" : "BN1_gamma3", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "107", "SubInstance" : "grp_conv_bn_act_pool_1_fu_660", "Port" : "BN1_gamma3"}]},
			{"Name" : "LSTM_W_ifog3", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "42", "SubInstance" : "grp_lstm_forward_unidir_fu_600", "Port" : "W_ifog"}]},
			{"Name" : "LSTM_R_ifog3", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "42", "SubInstance" : "grp_lstm_forward_unidir_fu_600", "Port" : "R_ifog"}]},
			{"Name" : "LSTM_b_ifog3", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "42", "SubInstance" : "grp_lstm_forward_unidir_fu_600", "Port" : "b_ifog"}]},
			{"Name" : "BN2_var3", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "BN2_gamma3", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "tokens4", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "Emb4", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "ConvW4", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "95", "SubInstance" : "grp_conv_bn_act_pool_fu_645", "Port" : "ConvW4"}]},
			{"Name" : "BN1_var4", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "95", "SubInstance" : "grp_conv_bn_act_pool_fu_645", "Port" : "BN1_var4"}]},
			{"Name" : "BN1_gamma4", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "95", "SubInstance" : "grp_conv_bn_act_pool_fu_645", "Port" : "BN1_gamma4"}]},
			{"Name" : "LSTM_W_ifog4", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "42", "SubInstance" : "grp_lstm_forward_unidir_fu_600", "Port" : "W_ifog"}]},
			{"Name" : "LSTM_R_ifog4", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "42", "SubInstance" : "grp_lstm_forward_unidir_fu_600", "Port" : "R_ifog"}]},
			{"Name" : "LSTM_b_ifog4", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "42", "SubInstance" : "grp_lstm_forward_unidir_fu_600", "Port" : "b_ifog"}]},
			{"Name" : "BN2_var4", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "BN2_gamma4", "Type" : "Memory", "Direction" : "I"}]},
	{"ID" : "11", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_433.Y_U", "Parent" : "10"},
	{"ID" : "12", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_433.U_slice_U", "Parent" : "10"},
	{"ID" : "13", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_433.LSTM_W_ifog0_U", "Parent" : "10"},
	{"ID" : "14", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_433.LSTM_R_ifog0_U", "Parent" : "10"},
	{"ID" : "15", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_433.LSTM_b_ifog0_U", "Parent" : "10"},
	{"ID" : "16", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_433.h_slice_U", "Parent" : "10"},
	{"ID" : "17", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_433.BN2_var0_U", "Parent" : "10"},
	{"ID" : "18", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_433.BN2_gamma0_U", "Parent" : "10"},
	{"ID" : "19", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_433.LSTM_W_ifog1_U", "Parent" : "10"},
	{"ID" : "20", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_433.LSTM_R_ifog1_U", "Parent" : "10"},
	{"ID" : "21", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_433.LSTM_b_ifog1_U", "Parent" : "10"},
	{"ID" : "22", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_433.BN2_var1_U", "Parent" : "10"},
	{"ID" : "23", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_433.BN2_gamma1_U", "Parent" : "10"},
	{"ID" : "24", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_433.LSTM_W_ifog2_U", "Parent" : "10"},
	{"ID" : "25", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_433.LSTM_R_ifog2_U", "Parent" : "10"},
	{"ID" : "26", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_433.LSTM_b_ifog2_U", "Parent" : "10"},
	{"ID" : "27", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_433.BN2_var2_U", "Parent" : "10"},
	{"ID" : "28", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_433.BN2_gamma2_U", "Parent" : "10"},
	{"ID" : "29", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_433.LSTM_W_ifog3_U", "Parent" : "10"},
	{"ID" : "30", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_433.LSTM_R_ifog3_U", "Parent" : "10"},
	{"ID" : "31", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_433.LSTM_b_ifog3_U", "Parent" : "10"},
	{"ID" : "32", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_433.BN2_var3_U", "Parent" : "10"},
	{"ID" : "33", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_433.BN2_gamma3_U", "Parent" : "10"},
	{"ID" : "34", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_433.tokens4_U", "Parent" : "10"},
	{"ID" : "35", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_433.Emb4_U", "Parent" : "10"},
	{"ID" : "36", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_433.LSTM_W_ifog4_U", "Parent" : "10"},
	{"ID" : "37", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_433.LSTM_R_ifog4_U", "Parent" : "10"},
	{"ID" : "38", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_433.LSTM_b_ifog4_U", "Parent" : "10"},
	{"ID" : "39", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_433.BN2_var4_U", "Parent" : "10"},
	{"ID" : "40", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_433.BN2_gamma4_U", "Parent" : "10"},
	{"ID" : "41", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_433.X_slice_1_U", "Parent" : "10"},
	{"ID" : "42", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_433.grp_lstm_forward_unidir_fu_600", "Parent" : "10", "Child" : ["43", "44", "45", "62", "63", "64", "65", "66", "67", "68", "69", "70", "71", "72", "73", "74", "75", "76", "77"],
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
					{"ID" : "45", "SubInstance" : "grp_generic_tanh_float_s_fu_325", "Port" : "table_exp_Z1_array_s"}]},
			{"Name" : "table_f_Z3_array_V", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "45", "SubInstance" : "grp_generic_tanh_float_s_fu_325", "Port" : "table_f_Z3_array_V"}]},
			{"Name" : "table_f_Z2_array_V", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "45", "SubInstance" : "grp_generic_tanh_float_s_fu_325", "Port" : "table_f_Z2_array_V"}]}]},
	{"ID" : "43", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_433.grp_lstm_forward_unidir_fu_600.c_slice_U", "Parent" : "42"},
	{"ID" : "44", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_433.grp_lstm_forward_unidir_fu_600.z_U", "Parent" : "42"},
	{"ID" : "45", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_433.grp_lstm_forward_unidir_fu_600.grp_generic_tanh_float_s_fu_325", "Parent" : "42", "Child" : ["46", "55", "56", "57", "58", "59", "60", "61"],
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
					{"ID" : "46", "SubInstance" : "grp_exp_generic_double_s_fu_89", "Port" : "table_exp_Z1_array_s"}]},
			{"Name" : "table_f_Z3_array_V", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "46", "SubInstance" : "grp_exp_generic_double_s_fu_89", "Port" : "table_f_Z3_array_V"}]},
			{"Name" : "table_f_Z2_array_V", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "46", "SubInstance" : "grp_exp_generic_double_s_fu_89", "Port" : "table_f_Z2_array_V"}]}]},
	{"ID" : "46", "Level" : "4", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_433.grp_lstm_forward_unidir_fu_600.grp_generic_tanh_float_s_fu_325.grp_exp_generic_double_s_fu_89", "Parent" : "45", "Child" : ["47", "48", "49", "50", "51", "52", "53", "54"],
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
	{"ID" : "47", "Level" : "5", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_433.grp_lstm_forward_unidir_fu_600.grp_generic_tanh_float_s_fu_325.grp_exp_generic_double_s_fu_89.table_exp_Z1_array_s_U", "Parent" : "46"},
	{"ID" : "48", "Level" : "5", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_433.grp_lstm_forward_unidir_fu_600.grp_generic_tanh_float_s_fu_325.grp_exp_generic_double_s_fu_89.table_f_Z3_array_V_U", "Parent" : "46"},
	{"ID" : "49", "Level" : "5", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_433.grp_lstm_forward_unidir_fu_600.grp_generic_tanh_float_s_fu_325.grp_exp_generic_double_s_fu_89.table_f_Z2_array_V_U", "Parent" : "46"},
	{"ID" : "50", "Level" : "5", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_433.grp_lstm_forward_unidir_fu_600.grp_generic_tanh_float_s_fu_325.grp_exp_generic_double_s_fu_89.main_mul_72ns_13s_84_5_1_U22", "Parent" : "46"},
	{"ID" : "51", "Level" : "5", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_433.grp_lstm_forward_unidir_fu_600.grp_generic_tanh_float_s_fu_325.grp_exp_generic_double_s_fu_89.main_mul_36ns_43ns_79_2_1_U23", "Parent" : "46"},
	{"ID" : "52", "Level" : "5", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_433.grp_lstm_forward_unidir_fu_600.grp_generic_tanh_float_s_fu_325.grp_exp_generic_double_s_fu_89.main_mul_44ns_49ns_93_2_1_U24", "Parent" : "46"},
	{"ID" : "53", "Level" : "5", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_433.grp_lstm_forward_unidir_fu_600.grp_generic_tanh_float_s_fu_325.grp_exp_generic_double_s_fu_89.main_mul_50ns_50ns_100_2_1_U25", "Parent" : "46"},
	{"ID" : "54", "Level" : "5", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_433.grp_lstm_forward_unidir_fu_600.grp_generic_tanh_float_s_fu_325.grp_exp_generic_double_s_fu_89.main_mac_muladd_16ns_16s_19s_31_1_1_U26", "Parent" : "46"},
	{"ID" : "55", "Level" : "4", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_433.grp_lstm_forward_unidir_fu_600.grp_generic_tanh_float_s_fu_325.main_faddfsub_32ns_32ns_32_5_full_dsp_1_U36", "Parent" : "45"},
	{"ID" : "56", "Level" : "4", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_433.grp_lstm_forward_unidir_fu_600.grp_generic_tanh_float_s_fu_325.main_fmul_32ns_32ns_32_4_max_dsp_1_U37", "Parent" : "45"},
	{"ID" : "57", "Level" : "4", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_433.grp_lstm_forward_unidir_fu_600.grp_generic_tanh_float_s_fu_325.main_fdiv_32ns_32ns_32_16_1_U38", "Parent" : "45"},
	{"ID" : "58", "Level" : "4", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_433.grp_lstm_forward_unidir_fu_600.grp_generic_tanh_float_s_fu_325.main_fptrunc_64ns_32_2_1_U39", "Parent" : "45"},
	{"ID" : "59", "Level" : "4", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_433.grp_lstm_forward_unidir_fu_600.grp_generic_tanh_float_s_fu_325.main_fpext_32ns_64_2_1_U40", "Parent" : "45"},
	{"ID" : "60", "Level" : "4", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_433.grp_lstm_forward_unidir_fu_600.grp_generic_tanh_float_s_fu_325.main_fcmp_32ns_32ns_1_2_1_U41", "Parent" : "45"},
	{"ID" : "61", "Level" : "4", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_433.grp_lstm_forward_unidir_fu_600.grp_generic_tanh_float_s_fu_325.main_dadd_64ns_64ns_64_5_full_dsp_1_U42", "Parent" : "45"},
	{"ID" : "62", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_433.grp_lstm_forward_unidir_fu_600.main_fadd_32ns_32ns_32_5_full_dsp_1_U50", "Parent" : "42"},
	{"ID" : "63", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_433.grp_lstm_forward_unidir_fu_600.main_fadd_32ns_32ns_32_5_full_dsp_1_U51", "Parent" : "42"},
	{"ID" : "64", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_433.grp_lstm_forward_unidir_fu_600.main_fdiv_32ns_32ns_32_16_1_U52", "Parent" : "42"},
	{"ID" : "65", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_433.grp_lstm_forward_unidir_fu_600.main_fdiv_32ns_32ns_32_16_1_U53", "Parent" : "42"},
	{"ID" : "66", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_433.grp_lstm_forward_unidir_fu_600.main_fexp_32ns_32ns_32_9_full_dsp_1_U54", "Parent" : "42"},
	{"ID" : "67", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_433.grp_lstm_forward_unidir_fu_600.main_fexp_32ns_32ns_32_9_full_dsp_1_U55", "Parent" : "42"},
	{"ID" : "68", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_433.grp_lstm_forward_unidir_fu_600.main_sptohp_32ns_16_2_1_U56", "Parent" : "42"},
	{"ID" : "69", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_433.grp_lstm_forward_unidir_fu_600.main_sptohp_32ns_16_2_1_U57", "Parent" : "42"},
	{"ID" : "70", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_433.grp_lstm_forward_unidir_fu_600.main_sptohp_32ns_16_2_1_U58", "Parent" : "42"},
	{"ID" : "71", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_433.grp_lstm_forward_unidir_fu_600.main_hptosp_16ns_32_2_1_U59", "Parent" : "42"},
	{"ID" : "72", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_433.grp_lstm_forward_unidir_fu_600.main_hptosp_16ns_32_2_1_U60", "Parent" : "42"},
	{"ID" : "73", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_433.grp_lstm_forward_unidir_fu_600.main_hadd_16ns_16ns_16_5_full_dsp_1_U61", "Parent" : "42"},
	{"ID" : "74", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_433.grp_lstm_forward_unidir_fu_600.main_hsub_16ns_16ns_16_5_full_dsp_1_U62", "Parent" : "42"},
	{"ID" : "75", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_433.grp_lstm_forward_unidir_fu_600.main_hsub_16ns_16ns_16_5_full_dsp_1_U63", "Parent" : "42"},
	{"ID" : "76", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_433.grp_lstm_forward_unidir_fu_600.main_hmul_16ns_16ns_16_4_max_dsp_1_U64", "Parent" : "42"},
	{"ID" : "77", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_433.grp_lstm_forward_unidir_fu_600.main_hmul_16ns_16ns_16_4_max_dsp_1_U65", "Parent" : "42"},
	{"ID" : "78", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_433.grp_generic_tanh_float_s_fu_634", "Parent" : "10", "Child" : ["79", "88", "89", "90", "91", "92", "93", "94"],
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
					{"ID" : "79", "SubInstance" : "grp_exp_generic_double_s_fu_89", "Port" : "table_exp_Z1_array_s"}]},
			{"Name" : "table_f_Z3_array_V", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "79", "SubInstance" : "grp_exp_generic_double_s_fu_89", "Port" : "table_f_Z3_array_V"}]},
			{"Name" : "table_f_Z2_array_V", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "79", "SubInstance" : "grp_exp_generic_double_s_fu_89", "Port" : "table_f_Z2_array_V"}]}]},
	{"ID" : "79", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_433.grp_generic_tanh_float_s_fu_634.grp_exp_generic_double_s_fu_89", "Parent" : "78", "Child" : ["80", "81", "82", "83", "84", "85", "86", "87"],
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
	{"ID" : "80", "Level" : "4", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_433.grp_generic_tanh_float_s_fu_634.grp_exp_generic_double_s_fu_89.table_exp_Z1_array_s_U", "Parent" : "79"},
	{"ID" : "81", "Level" : "4", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_433.grp_generic_tanh_float_s_fu_634.grp_exp_generic_double_s_fu_89.table_f_Z3_array_V_U", "Parent" : "79"},
	{"ID" : "82", "Level" : "4", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_433.grp_generic_tanh_float_s_fu_634.grp_exp_generic_double_s_fu_89.table_f_Z2_array_V_U", "Parent" : "79"},
	{"ID" : "83", "Level" : "4", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_433.grp_generic_tanh_float_s_fu_634.grp_exp_generic_double_s_fu_89.main_mul_72ns_13s_84_5_1_U22", "Parent" : "79"},
	{"ID" : "84", "Level" : "4", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_433.grp_generic_tanh_float_s_fu_634.grp_exp_generic_double_s_fu_89.main_mul_36ns_43ns_79_2_1_U23", "Parent" : "79"},
	{"ID" : "85", "Level" : "4", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_433.grp_generic_tanh_float_s_fu_634.grp_exp_generic_double_s_fu_89.main_mul_44ns_49ns_93_2_1_U24", "Parent" : "79"},
	{"ID" : "86", "Level" : "4", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_433.grp_generic_tanh_float_s_fu_634.grp_exp_generic_double_s_fu_89.main_mul_50ns_50ns_100_2_1_U25", "Parent" : "79"},
	{"ID" : "87", "Level" : "4", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_433.grp_generic_tanh_float_s_fu_634.grp_exp_generic_double_s_fu_89.main_mac_muladd_16ns_16s_19s_31_1_1_U26", "Parent" : "79"},
	{"ID" : "88", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_433.grp_generic_tanh_float_s_fu_634.main_faddfsub_32ns_32ns_32_5_full_dsp_1_U36", "Parent" : "78"},
	{"ID" : "89", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_433.grp_generic_tanh_float_s_fu_634.main_fmul_32ns_32ns_32_4_max_dsp_1_U37", "Parent" : "78"},
	{"ID" : "90", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_433.grp_generic_tanh_float_s_fu_634.main_fdiv_32ns_32ns_32_16_1_U38", "Parent" : "78"},
	{"ID" : "91", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_433.grp_generic_tanh_float_s_fu_634.main_fptrunc_64ns_32_2_1_U39", "Parent" : "78"},
	{"ID" : "92", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_433.grp_generic_tanh_float_s_fu_634.main_fpext_32ns_64_2_1_U40", "Parent" : "78"},
	{"ID" : "93", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_433.grp_generic_tanh_float_s_fu_634.main_fcmp_32ns_32ns_1_2_1_U41", "Parent" : "78"},
	{"ID" : "94", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_433.grp_generic_tanh_float_s_fu_634.main_dadd_64ns_64ns_64_5_full_dsp_1_U42", "Parent" : "78"},
	{"ID" : "95", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_433.grp_conv_bn_act_pool_fu_645", "Parent" : "10", "Child" : ["96", "97", "98", "99", "100", "101", "102", "103", "104", "105", "106"],
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
	{"ID" : "96", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_433.grp_conv_bn_act_pool_fu_645.ConvW4_U", "Parent" : "95"},
	{"ID" : "97", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_433.grp_conv_bn_act_pool_fu_645.BN1_var4_U", "Parent" : "95"},
	{"ID" : "98", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_433.grp_conv_bn_act_pool_fu_645.BN1_gamma4_U", "Parent" : "95"},
	{"ID" : "99", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_433.grp_conv_bn_act_pool_fu_645.main_fdiv_32ns_32ns_32_16_1_U112", "Parent" : "95"},
	{"ID" : "100", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_433.grp_conv_bn_act_pool_fu_645.main_fsqrt_32ns_32ns_32_12_1_U113", "Parent" : "95"},
	{"ID" : "101", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_433.grp_conv_bn_act_pool_fu_645.main_sptohp_32ns_16_2_1_U114", "Parent" : "95"},
	{"ID" : "102", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_433.grp_conv_bn_act_pool_fu_645.main_hptosp_16ns_32_2_1_U115", "Parent" : "95"},
	{"ID" : "103", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_433.grp_conv_bn_act_pool_fu_645.main_hadd_16ns_16ns_16_5_full_dsp_1_U116", "Parent" : "95"},
	{"ID" : "104", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_433.grp_conv_bn_act_pool_fu_645.main_hmul_16ns_16ns_16_4_max_dsp_1_U117", "Parent" : "95"},
	{"ID" : "105", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_433.grp_conv_bn_act_pool_fu_645.main_hdiv_16ns_16ns_16_7_1_U118", "Parent" : "95"},
	{"ID" : "106", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_433.grp_conv_bn_act_pool_fu_645.main_hcmp_16ns_16ns_1_2_1_U119", "Parent" : "95"},
	{"ID" : "107", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_433.grp_conv_bn_act_pool_1_fu_660", "Parent" : "10", "Child" : ["108", "109", "110", "111", "112", "113", "114", "115", "116", "117", "118", "119"],
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
	{"ID" : "108", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_433.grp_conv_bn_act_pool_1_fu_660.X_slice3_U", "Parent" : "107"},
	{"ID" : "109", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_433.grp_conv_bn_act_pool_1_fu_660.ConvW3_U", "Parent" : "107"},
	{"ID" : "110", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_433.grp_conv_bn_act_pool_1_fu_660.BN1_var3_U", "Parent" : "107"},
	{"ID" : "111", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_433.grp_conv_bn_act_pool_1_fu_660.BN1_gamma3_U", "Parent" : "107"},
	{"ID" : "112", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_433.grp_conv_bn_act_pool_1_fu_660.main_fdiv_32ns_32ns_32_16_1_U100", "Parent" : "107"},
	{"ID" : "113", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_433.grp_conv_bn_act_pool_1_fu_660.main_fsqrt_32ns_32ns_32_12_1_U101", "Parent" : "107"},
	{"ID" : "114", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_433.grp_conv_bn_act_pool_1_fu_660.main_sptohp_32ns_16_2_1_U102", "Parent" : "107"},
	{"ID" : "115", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_433.grp_conv_bn_act_pool_1_fu_660.main_hptosp_16ns_32_2_1_U103", "Parent" : "107"},
	{"ID" : "116", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_433.grp_conv_bn_act_pool_1_fu_660.main_hadd_16ns_16ns_16_5_full_dsp_1_U104", "Parent" : "107"},
	{"ID" : "117", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_433.grp_conv_bn_act_pool_1_fu_660.main_hmul_16ns_16ns_16_4_max_dsp_1_U105", "Parent" : "107"},
	{"ID" : "118", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_433.grp_conv_bn_act_pool_1_fu_660.main_hdiv_16ns_16ns_16_7_1_U106", "Parent" : "107"},
	{"ID" : "119", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_433.grp_conv_bn_act_pool_1_fu_660.main_hcmp_16ns_16ns_1_2_1_U107", "Parent" : "107"},
	{"ID" : "120", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_433.grp_conv_bn_act_pool_2_fu_676", "Parent" : "10", "Child" : ["121", "122", "123", "124", "125", "126", "127", "128", "129", "130", "131", "132"],
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
	{"ID" : "121", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_433.grp_conv_bn_act_pool_2_fu_676.X_slice2_U", "Parent" : "120"},
	{"ID" : "122", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_433.grp_conv_bn_act_pool_2_fu_676.ConvW2_U", "Parent" : "120"},
	{"ID" : "123", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_433.grp_conv_bn_act_pool_2_fu_676.BN1_var2_U", "Parent" : "120"},
	{"ID" : "124", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_433.grp_conv_bn_act_pool_2_fu_676.BN1_gamma2_U", "Parent" : "120"},
	{"ID" : "125", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_433.grp_conv_bn_act_pool_2_fu_676.main_fdiv_32ns_32ns_32_16_1_U88", "Parent" : "120"},
	{"ID" : "126", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_433.grp_conv_bn_act_pool_2_fu_676.main_fsqrt_32ns_32ns_32_12_1_U89", "Parent" : "120"},
	{"ID" : "127", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_433.grp_conv_bn_act_pool_2_fu_676.main_sptohp_32ns_16_2_1_U90", "Parent" : "120"},
	{"ID" : "128", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_433.grp_conv_bn_act_pool_2_fu_676.main_hptosp_16ns_32_2_1_U91", "Parent" : "120"},
	{"ID" : "129", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_433.grp_conv_bn_act_pool_2_fu_676.main_hadd_16ns_16ns_16_5_full_dsp_1_U92", "Parent" : "120"},
	{"ID" : "130", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_433.grp_conv_bn_act_pool_2_fu_676.main_hmul_16ns_16ns_16_4_max_dsp_1_U93", "Parent" : "120"},
	{"ID" : "131", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_433.grp_conv_bn_act_pool_2_fu_676.main_hdiv_16ns_16ns_16_7_1_U94", "Parent" : "120"},
	{"ID" : "132", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_433.grp_conv_bn_act_pool_2_fu_676.main_hcmp_16ns_16ns_1_2_1_U95", "Parent" : "120"},
	{"ID" : "133", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_433.grp_conv_bn_act_pool_3_fu_692", "Parent" : "10", "Child" : ["134", "135", "136", "137", "138", "139", "140", "141", "142", "143", "144", "145"],
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
	{"ID" : "134", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_433.grp_conv_bn_act_pool_3_fu_692.X_slice1_U", "Parent" : "133"},
	{"ID" : "135", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_433.grp_conv_bn_act_pool_3_fu_692.ConvW1_U", "Parent" : "133"},
	{"ID" : "136", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_433.grp_conv_bn_act_pool_3_fu_692.BN1_var1_U", "Parent" : "133"},
	{"ID" : "137", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_433.grp_conv_bn_act_pool_3_fu_692.BN1_gamma1_U", "Parent" : "133"},
	{"ID" : "138", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_433.grp_conv_bn_act_pool_3_fu_692.main_fdiv_32ns_32ns_32_16_1_U76", "Parent" : "133"},
	{"ID" : "139", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_433.grp_conv_bn_act_pool_3_fu_692.main_fsqrt_32ns_32ns_32_12_1_U77", "Parent" : "133"},
	{"ID" : "140", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_433.grp_conv_bn_act_pool_3_fu_692.main_sptohp_32ns_16_2_1_U78", "Parent" : "133"},
	{"ID" : "141", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_433.grp_conv_bn_act_pool_3_fu_692.main_hptosp_16ns_32_2_1_U79", "Parent" : "133"},
	{"ID" : "142", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_433.grp_conv_bn_act_pool_3_fu_692.main_hadd_16ns_16ns_16_5_full_dsp_1_U80", "Parent" : "133"},
	{"ID" : "143", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_433.grp_conv_bn_act_pool_3_fu_692.main_hmul_16ns_16ns_16_4_max_dsp_1_U81", "Parent" : "133"},
	{"ID" : "144", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_433.grp_conv_bn_act_pool_3_fu_692.main_hdiv_16ns_16ns_16_7_1_U82", "Parent" : "133"},
	{"ID" : "145", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_433.grp_conv_bn_act_pool_3_fu_692.main_hcmp_16ns_16ns_1_2_1_U83", "Parent" : "133"},
	{"ID" : "146", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_433.grp_conv_bn_act_pool_4_fu_708", "Parent" : "10", "Child" : ["147", "148", "149", "150", "151", "152", "153", "154", "155", "156", "157", "158"],
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
	{"ID" : "147", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_433.grp_conv_bn_act_pool_4_fu_708.X_slice_U", "Parent" : "146"},
	{"ID" : "148", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_433.grp_conv_bn_act_pool_4_fu_708.ConvW0_U", "Parent" : "146"},
	{"ID" : "149", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_433.grp_conv_bn_act_pool_4_fu_708.BN1_var0_U", "Parent" : "146"},
	{"ID" : "150", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_433.grp_conv_bn_act_pool_4_fu_708.BN1_gamma0_U", "Parent" : "146"},
	{"ID" : "151", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_433.grp_conv_bn_act_pool_4_fu_708.main_fdiv_32ns_32ns_32_16_1_U1", "Parent" : "146"},
	{"ID" : "152", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_433.grp_conv_bn_act_pool_4_fu_708.main_fsqrt_32ns_32ns_32_12_1_U2", "Parent" : "146"},
	{"ID" : "153", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_433.grp_conv_bn_act_pool_4_fu_708.main_sptohp_32ns_16_2_1_U3", "Parent" : "146"},
	{"ID" : "154", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_433.grp_conv_bn_act_pool_4_fu_708.main_hptosp_16ns_32_2_1_U4", "Parent" : "146"},
	{"ID" : "155", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_433.grp_conv_bn_act_pool_4_fu_708.main_hadd_16ns_16ns_16_5_full_dsp_1_U5", "Parent" : "146"},
	{"ID" : "156", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_433.grp_conv_bn_act_pool_4_fu_708.main_hmul_16ns_16ns_16_4_max_dsp_1_U6", "Parent" : "146"},
	{"ID" : "157", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_433.grp_conv_bn_act_pool_4_fu_708.main_hdiv_16ns_16ns_16_7_1_U7", "Parent" : "146"},
	{"ID" : "158", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_433.grp_conv_bn_act_pool_4_fu_708.main_hcmp_16ns_16ns_1_2_1_U8", "Parent" : "146"},
	{"ID" : "159", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_433.main_fdiv_32ns_32ns_32_16_1_U124", "Parent" : "10"},
	{"ID" : "160", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_433.main_fsqrt_32ns_32ns_32_12_1_U125", "Parent" : "10"},
	{"ID" : "161", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_433.main_sptohp_32ns_16_2_1_U126", "Parent" : "10"},
	{"ID" : "162", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_433.main_hptosp_16ns_32_2_1_U127", "Parent" : "10"},
	{"ID" : "163", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_433.main_hadd_16ns_16ns_16_5_full_dsp_1_U128", "Parent" : "10"},
	{"ID" : "164", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_433.main_hmul_16ns_16ns_16_4_max_dsp_1_U129", "Parent" : "10"},
	{"ID" : "165", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.main_fdiv_32ns_32ns_32_16_1_U147", "Parent" : "0"},
	{"ID" : "166", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.main_fsqrt_32ns_32ns_32_12_1_U148", "Parent" : "0"},
	{"ID" : "167", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.main_sptohp_32ns_16_2_1_U149", "Parent" : "0"},
	{"ID" : "168", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.main_hptosp_16ns_32_2_1_U150", "Parent" : "0"},
	{"ID" : "169", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.main_hadd_16ns_16ns_16_5_full_dsp_1_U151", "Parent" : "0"},
	{"ID" : "170", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.main_hmul_16ns_16ns_16_4_max_dsp_1_U152", "Parent" : "0"},
	{"ID" : "171", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.main_hcmp_16ns_16ns_1_2_1_U153", "Parent" : "0"}]}


set ArgLastReadFirstWriteLatency {
	main {
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
		BN2_gamma4 {Type I LastRead -1 FirstWrite -1}
		fc_0_W {Type I LastRead -1 FirstWrite -1}
		fc_0_bn_var {Type I LastRead -1 FirstWrite -1}
		fc_0_bn_gamma {Type I LastRead -1 FirstWrite -1}
		fc_1_W {Type I LastRead -1 FirstWrite -1}
		fc_1_bn_var {Type I LastRead -1 FirstWrite -1}
		fc_1_bn_gamma {Type I LastRead -1 FirstWrite -1}}
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
	{"Name" : "Latency", "Min" : "121438995", "Max" : "121678995"}
	, {"Name" : "Interval", "Min" : "121438996", "Max" : "121678996"}
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
