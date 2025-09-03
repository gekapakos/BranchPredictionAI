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
	{"ID" : "0", "Level" : "0", "Path" : "`AUTOTB_DUT_INST", "Parent" : "", "Child" : ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "148", "149", "150", "151", "152"],
		"CDFG" : "main",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "0", "ap_idle" : "1",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "120638908", "EstimateLatencyMax" : "120878908",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"WaitState" : [
			{"State" : "ap_ST_fsm_state2", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_run_all_slices_unrol_fu_439"}],
		"Port" : [
			{"Name" : "X0", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "10", "SubInstance" : "grp_run_all_slices_unrol_fu_439", "Port" : "X0"}]},
			{"Name" : "ConvW0", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "10", "SubInstance" : "grp_run_all_slices_unrol_fu_439", "Port" : "ConvW0"}]},
			{"Name" : "BN1_var0", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "10", "SubInstance" : "grp_run_all_slices_unrol_fu_439", "Port" : "BN1_var0"}]},
			{"Name" : "BN1_gamma0", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "10", "SubInstance" : "grp_run_all_slices_unrol_fu_439", "Port" : "BN1_gamma0"}]},
			{"Name" : "c_slice", "Type" : "Memory", "Direction" : "IO",
				"SubConnect" : [
					{"ID" : "10", "SubInstance" : "grp_run_all_slices_unrol_fu_439", "Port" : "c_slice"}]},
			{"Name" : "table_exp_Z1_array_s", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "10", "SubInstance" : "grp_run_all_slices_unrol_fu_439", "Port" : "table_exp_Z1_array_s"}]},
			{"Name" : "table_f_Z3_array_V", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "10", "SubInstance" : "grp_run_all_slices_unrol_fu_439", "Port" : "table_f_Z3_array_V"}]},
			{"Name" : "table_f_Z2_array_V", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "10", "SubInstance" : "grp_run_all_slices_unrol_fu_439", "Port" : "table_f_Z2_array_V"}]},
			{"Name" : "LSTM_W_ifog0", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "10", "SubInstance" : "grp_run_all_slices_unrol_fu_439", "Port" : "LSTM_W_ifog0"}]},
			{"Name" : "LSTM_R_ifog0", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "10", "SubInstance" : "grp_run_all_slices_unrol_fu_439", "Port" : "LSTM_R_ifog0"}]},
			{"Name" : "LSTM_b_ifog0", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "10", "SubInstance" : "grp_run_all_slices_unrol_fu_439", "Port" : "LSTM_b_ifog0"}]},
			{"Name" : "h_slice", "Type" : "Memory", "Direction" : "IO",
				"SubConnect" : [
					{"ID" : "10", "SubInstance" : "grp_run_all_slices_unrol_fu_439", "Port" : "h_slice"}]},
			{"Name" : "BN2_var0", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "10", "SubInstance" : "grp_run_all_slices_unrol_fu_439", "Port" : "BN2_var0"}]},
			{"Name" : "BN2_gamma0", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "10", "SubInstance" : "grp_run_all_slices_unrol_fu_439", "Port" : "BN2_gamma0"}]},
			{"Name" : "X01", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "10", "SubInstance" : "grp_run_all_slices_unrol_fu_439", "Port" : "X01"}]},
			{"Name" : "ConvW1", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "10", "SubInstance" : "grp_run_all_slices_unrol_fu_439", "Port" : "ConvW1"}]},
			{"Name" : "BN1_var1", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "10", "SubInstance" : "grp_run_all_slices_unrol_fu_439", "Port" : "BN1_var1"}]},
			{"Name" : "BN1_gamma1", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "10", "SubInstance" : "grp_run_all_slices_unrol_fu_439", "Port" : "BN1_gamma1"}]},
			{"Name" : "LSTM_W_ifog1", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "10", "SubInstance" : "grp_run_all_slices_unrol_fu_439", "Port" : "LSTM_W_ifog1"}]},
			{"Name" : "LSTM_R_ifog1", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "10", "SubInstance" : "grp_run_all_slices_unrol_fu_439", "Port" : "LSTM_R_ifog1"}]},
			{"Name" : "LSTM_b_ifog1", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "10", "SubInstance" : "grp_run_all_slices_unrol_fu_439", "Port" : "LSTM_b_ifog1"}]},
			{"Name" : "BN2_var1", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "10", "SubInstance" : "grp_run_all_slices_unrol_fu_439", "Port" : "BN2_var1"}]},
			{"Name" : "BN2_gamma1", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "10", "SubInstance" : "grp_run_all_slices_unrol_fu_439", "Port" : "BN2_gamma1"}]},
			{"Name" : "X04", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "10", "SubInstance" : "grp_run_all_slices_unrol_fu_439", "Port" : "X04"}]},
			{"Name" : "ConvW2", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "10", "SubInstance" : "grp_run_all_slices_unrol_fu_439", "Port" : "ConvW2"}]},
			{"Name" : "BN1_var2", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "10", "SubInstance" : "grp_run_all_slices_unrol_fu_439", "Port" : "BN1_var2"}]},
			{"Name" : "BN1_gamma2", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "10", "SubInstance" : "grp_run_all_slices_unrol_fu_439", "Port" : "BN1_gamma2"}]},
			{"Name" : "LSTM_W_ifog2", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "10", "SubInstance" : "grp_run_all_slices_unrol_fu_439", "Port" : "LSTM_W_ifog2"}]},
			{"Name" : "LSTM_R_ifog2", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "10", "SubInstance" : "grp_run_all_slices_unrol_fu_439", "Port" : "LSTM_R_ifog2"}]},
			{"Name" : "LSTM_b_ifog2", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "10", "SubInstance" : "grp_run_all_slices_unrol_fu_439", "Port" : "LSTM_b_ifog2"}]},
			{"Name" : "BN2_var2", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "10", "SubInstance" : "grp_run_all_slices_unrol_fu_439", "Port" : "BN2_var2"}]},
			{"Name" : "BN2_gamma2", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "10", "SubInstance" : "grp_run_all_slices_unrol_fu_439", "Port" : "BN2_gamma2"}]},
			{"Name" : "X07", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "10", "SubInstance" : "grp_run_all_slices_unrol_fu_439", "Port" : "X07"}]},
			{"Name" : "ConvW3", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "10", "SubInstance" : "grp_run_all_slices_unrol_fu_439", "Port" : "ConvW3"}]},
			{"Name" : "BN1_var3", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "10", "SubInstance" : "grp_run_all_slices_unrol_fu_439", "Port" : "BN1_var3"}]},
			{"Name" : "BN1_gamma3", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "10", "SubInstance" : "grp_run_all_slices_unrol_fu_439", "Port" : "BN1_gamma3"}]},
			{"Name" : "LSTM_W_ifog3", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "10", "SubInstance" : "grp_run_all_slices_unrol_fu_439", "Port" : "LSTM_W_ifog3"}]},
			{"Name" : "LSTM_R_ifog3", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "10", "SubInstance" : "grp_run_all_slices_unrol_fu_439", "Port" : "LSTM_R_ifog3"}]},
			{"Name" : "LSTM_b_ifog3", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "10", "SubInstance" : "grp_run_all_slices_unrol_fu_439", "Port" : "LSTM_b_ifog3"}]},
			{"Name" : "BN2_var3", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "10", "SubInstance" : "grp_run_all_slices_unrol_fu_439", "Port" : "BN2_var3"}]},
			{"Name" : "BN2_gamma3", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "10", "SubInstance" : "grp_run_all_slices_unrol_fu_439", "Port" : "BN2_gamma3"}]},
			{"Name" : "tokens4", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "10", "SubInstance" : "grp_run_all_slices_unrol_fu_439", "Port" : "tokens4"}]},
			{"Name" : "Emb4", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "10", "SubInstance" : "grp_run_all_slices_unrol_fu_439", "Port" : "Emb4"}]},
			{"Name" : "ConvW4", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "10", "SubInstance" : "grp_run_all_slices_unrol_fu_439", "Port" : "ConvW4"}]},
			{"Name" : "BN1_var4", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "10", "SubInstance" : "grp_run_all_slices_unrol_fu_439", "Port" : "BN1_var4"}]},
			{"Name" : "BN1_gamma4", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "10", "SubInstance" : "grp_run_all_slices_unrol_fu_439", "Port" : "BN1_gamma4"}]},
			{"Name" : "LSTM_W_ifog4", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "10", "SubInstance" : "grp_run_all_slices_unrol_fu_439", "Port" : "LSTM_W_ifog4"}]},
			{"Name" : "LSTM_R_ifog4", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "10", "SubInstance" : "grp_run_all_slices_unrol_fu_439", "Port" : "LSTM_R_ifog4"}]},
			{"Name" : "LSTM_b_ifog4", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "10", "SubInstance" : "grp_run_all_slices_unrol_fu_439", "Port" : "LSTM_b_ifog4"}]},
			{"Name" : "BN2_var4", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "10", "SubInstance" : "grp_run_all_slices_unrol_fu_439", "Port" : "BN2_var4"}]},
			{"Name" : "BN2_gamma4", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "10", "SubInstance" : "grp_run_all_slices_unrol_fu_439", "Port" : "BN2_gamma4"}]},
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
	{"ID" : "10", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_439", "Parent" : "0", "Child" : ["11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21", "22", "23", "24", "25", "26", "27", "28", "29", "30", "31", "32", "33", "34", "35", "36", "37", "38", "39", "40", "41", "42", "43", "44", "45", "46", "47", "48", "49", "50", "78", "95", "104", "114", "124", "134", "144", "145", "146", "147"],
		"CDFG" : "run_all_slices_unrol",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "0", "ap_idle" : "1",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "120400565", "EstimateLatencyMax" : "120640565",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"WaitState" : [
			{"State" : "ap_ST_fsm_state5", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_lstm_forward_unidir_fu_636"},
			{"State" : "ap_ST_fsm_state63", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_lstm_forward_unidir_fu_636"},
			{"State" : "ap_ST_fsm_state121", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_lstm_forward_unidir_fu_636"},
			{"State" : "ap_ST_fsm_state179", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_lstm_forward_unidir_fu_636"},
			{"State" : "ap_ST_fsm_state241", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_lstm_forward_unidir_fu_636"},
			{"State" : "ap_ST_fsm_state54", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_generic_tanh_float_s_fu_669"},
			{"State" : "ap_ST_fsm_state112", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_generic_tanh_float_s_fu_669"},
			{"State" : "ap_ST_fsm_state170", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_generic_tanh_float_s_fu_669"},
			{"State" : "ap_ST_fsm_state228", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_generic_tanh_float_s_fu_669"},
			{"State" : "ap_ST_fsm_state290", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_generic_tanh_float_s_fu_669"},
			{"State" : "ap_ST_fsm_state239", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_conv_bn_act_pool_fu_680"},
			{"State" : "ap_ST_fsm_state177", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_conv_bn_act_pool_1_fu_693"},
			{"State" : "ap_ST_fsm_state119", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_conv_bn_act_pool_2_fu_707"},
			{"State" : "ap_ST_fsm_state61", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_conv_bn_act_pool_3_fu_721"},
			{"State" : "ap_ST_fsm_state3", "FSM" : "ap_CS_fsm", "SubInstance" : "grp_conv_bn_act_pool_4_fu_735"}],
		"Port" : [
			{"Name" : "merged", "Type" : "Memory", "Direction" : "IO"},
			{"Name" : "X0", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "134", "SubInstance" : "grp_conv_bn_act_pool_4_fu_735", "Port" : "X0"}]},
			{"Name" : "ConvW0", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "134", "SubInstance" : "grp_conv_bn_act_pool_4_fu_735", "Port" : "ConvW0"}]},
			{"Name" : "BN1_var0", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "134", "SubInstance" : "grp_conv_bn_act_pool_4_fu_735", "Port" : "BN1_var0"}]},
			{"Name" : "BN1_gamma0", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "134", "SubInstance" : "grp_conv_bn_act_pool_4_fu_735", "Port" : "BN1_gamma0"}]},
			{"Name" : "c_slice", "Type" : "Memory", "Direction" : "IO",
				"SubConnect" : [
					{"ID" : "50", "SubInstance" : "grp_lstm_forward_unidir_fu_636", "Port" : "c_slice"}]},
			{"Name" : "table_exp_Z1_array_s", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "78", "SubInstance" : "grp_generic_tanh_float_s_fu_669", "Port" : "table_exp_Z1_array_s"},
					{"ID" : "50", "SubInstance" : "grp_lstm_forward_unidir_fu_636", "Port" : "table_exp_Z1_array_s"}]},
			{"Name" : "table_f_Z3_array_V", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "78", "SubInstance" : "grp_generic_tanh_float_s_fu_669", "Port" : "table_f_Z3_array_V"},
					{"ID" : "50", "SubInstance" : "grp_lstm_forward_unidir_fu_636", "Port" : "table_f_Z3_array_V"}]},
			{"Name" : "table_f_Z2_array_V", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "78", "SubInstance" : "grp_generic_tanh_float_s_fu_669", "Port" : "table_f_Z2_array_V"},
					{"ID" : "50", "SubInstance" : "grp_lstm_forward_unidir_fu_636", "Port" : "table_f_Z2_array_V"}]},
			{"Name" : "LSTM_W_ifog0", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "50", "SubInstance" : "grp_lstm_forward_unidir_fu_636", "Port" : "W_ifog"}]},
			{"Name" : "LSTM_R_ifog0", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "50", "SubInstance" : "grp_lstm_forward_unidir_fu_636", "Port" : "R_ifog"}]},
			{"Name" : "LSTM_b_ifog0", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "50", "SubInstance" : "grp_lstm_forward_unidir_fu_636", "Port" : "b_ifog"}]},
			{"Name" : "h_slice", "Type" : "Memory", "Direction" : "IO",
				"SubConnect" : [
					{"ID" : "50", "SubInstance" : "grp_lstm_forward_unidir_fu_636", "Port" : "h_last"}]},
			{"Name" : "BN2_var0", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "BN2_gamma0", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "X01", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "124", "SubInstance" : "grp_conv_bn_act_pool_3_fu_721", "Port" : "X01"}]},
			{"Name" : "ConvW1", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "124", "SubInstance" : "grp_conv_bn_act_pool_3_fu_721", "Port" : "ConvW1"}]},
			{"Name" : "BN1_var1", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "124", "SubInstance" : "grp_conv_bn_act_pool_3_fu_721", "Port" : "BN1_var1"}]},
			{"Name" : "BN1_gamma1", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "124", "SubInstance" : "grp_conv_bn_act_pool_3_fu_721", "Port" : "BN1_gamma1"}]},
			{"Name" : "LSTM_W_ifog1", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "50", "SubInstance" : "grp_lstm_forward_unidir_fu_636", "Port" : "W_ifog"}]},
			{"Name" : "LSTM_R_ifog1", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "50", "SubInstance" : "grp_lstm_forward_unidir_fu_636", "Port" : "R_ifog"}]},
			{"Name" : "LSTM_b_ifog1", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "50", "SubInstance" : "grp_lstm_forward_unidir_fu_636", "Port" : "b_ifog"}]},
			{"Name" : "BN2_var1", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "BN2_gamma1", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "X04", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "114", "SubInstance" : "grp_conv_bn_act_pool_2_fu_707", "Port" : "X04"}]},
			{"Name" : "ConvW2", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "114", "SubInstance" : "grp_conv_bn_act_pool_2_fu_707", "Port" : "ConvW2"}]},
			{"Name" : "BN1_var2", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "114", "SubInstance" : "grp_conv_bn_act_pool_2_fu_707", "Port" : "BN1_var2"}]},
			{"Name" : "BN1_gamma2", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "114", "SubInstance" : "grp_conv_bn_act_pool_2_fu_707", "Port" : "BN1_gamma2"}]},
			{"Name" : "LSTM_W_ifog2", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "50", "SubInstance" : "grp_lstm_forward_unidir_fu_636", "Port" : "W_ifog"}]},
			{"Name" : "LSTM_R_ifog2", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "50", "SubInstance" : "grp_lstm_forward_unidir_fu_636", "Port" : "R_ifog"}]},
			{"Name" : "LSTM_b_ifog2", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "50", "SubInstance" : "grp_lstm_forward_unidir_fu_636", "Port" : "b_ifog"}]},
			{"Name" : "BN2_var2", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "BN2_gamma2", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "X07", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "104", "SubInstance" : "grp_conv_bn_act_pool_1_fu_693", "Port" : "X07"}]},
			{"Name" : "ConvW3", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "104", "SubInstance" : "grp_conv_bn_act_pool_1_fu_693", "Port" : "ConvW3"}]},
			{"Name" : "BN1_var3", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "104", "SubInstance" : "grp_conv_bn_act_pool_1_fu_693", "Port" : "BN1_var3"}]},
			{"Name" : "BN1_gamma3", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "104", "SubInstance" : "grp_conv_bn_act_pool_1_fu_693", "Port" : "BN1_gamma3"}]},
			{"Name" : "LSTM_W_ifog3", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "50", "SubInstance" : "grp_lstm_forward_unidir_fu_636", "Port" : "W_ifog"}]},
			{"Name" : "LSTM_R_ifog3", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "50", "SubInstance" : "grp_lstm_forward_unidir_fu_636", "Port" : "R_ifog"}]},
			{"Name" : "LSTM_b_ifog3", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "50", "SubInstance" : "grp_lstm_forward_unidir_fu_636", "Port" : "b_ifog"}]},
			{"Name" : "BN2_var3", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "BN2_gamma3", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "tokens4", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "Emb4", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "ConvW4", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "95", "SubInstance" : "grp_conv_bn_act_pool_fu_680", "Port" : "ConvW4"}]},
			{"Name" : "BN1_var4", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "95", "SubInstance" : "grp_conv_bn_act_pool_fu_680", "Port" : "BN1_var4"}]},
			{"Name" : "BN1_gamma4", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "95", "SubInstance" : "grp_conv_bn_act_pool_fu_680", "Port" : "BN1_gamma4"}]},
			{"Name" : "LSTM_W_ifog4", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "50", "SubInstance" : "grp_lstm_forward_unidir_fu_636", "Port" : "W_ifog"}]},
			{"Name" : "LSTM_R_ifog4", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "50", "SubInstance" : "grp_lstm_forward_unidir_fu_636", "Port" : "R_ifog"}]},
			{"Name" : "LSTM_b_ifog4", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "50", "SubInstance" : "grp_lstm_forward_unidir_fu_636", "Port" : "b_ifog"}]},
			{"Name" : "BN2_var4", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "BN2_gamma4", "Type" : "Memory", "Direction" : "I"}]},
	{"ID" : "11", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_439.LSTM_W_ifog0_U", "Parent" : "10"},
	{"ID" : "12", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_439.LSTM_R_ifog0_U", "Parent" : "10"},
	{"ID" : "13", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_439.LSTM_b_ifog0_U", "Parent" : "10"},
	{"ID" : "14", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_439.h_slice_U", "Parent" : "10"},
	{"ID" : "15", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_439.BN2_var0_U", "Parent" : "10"},
	{"ID" : "16", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_439.BN2_gamma0_U", "Parent" : "10"},
	{"ID" : "17", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_439.LSTM_W_ifog1_U", "Parent" : "10"},
	{"ID" : "18", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_439.LSTM_R_ifog1_U", "Parent" : "10"},
	{"ID" : "19", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_439.LSTM_b_ifog1_U", "Parent" : "10"},
	{"ID" : "20", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_439.BN2_var1_U", "Parent" : "10"},
	{"ID" : "21", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_439.BN2_gamma1_U", "Parent" : "10"},
	{"ID" : "22", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_439.LSTM_W_ifog2_U", "Parent" : "10"},
	{"ID" : "23", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_439.LSTM_R_ifog2_U", "Parent" : "10"},
	{"ID" : "24", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_439.LSTM_b_ifog2_U", "Parent" : "10"},
	{"ID" : "25", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_439.BN2_var2_U", "Parent" : "10"},
	{"ID" : "26", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_439.BN2_gamma2_U", "Parent" : "10"},
	{"ID" : "27", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_439.LSTM_W_ifog3_U", "Parent" : "10"},
	{"ID" : "28", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_439.LSTM_R_ifog3_U", "Parent" : "10"},
	{"ID" : "29", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_439.LSTM_b_ifog3_U", "Parent" : "10"},
	{"ID" : "30", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_439.BN2_var3_U", "Parent" : "10"},
	{"ID" : "31", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_439.BN2_gamma3_U", "Parent" : "10"},
	{"ID" : "32", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_439.tokens4_U", "Parent" : "10"},
	{"ID" : "33", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_439.Emb4_U", "Parent" : "10"},
	{"ID" : "34", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_439.LSTM_W_ifog4_U", "Parent" : "10"},
	{"ID" : "35", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_439.LSTM_R_ifog4_U", "Parent" : "10"},
	{"ID" : "36", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_439.LSTM_b_ifog4_U", "Parent" : "10"},
	{"ID" : "37", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_439.BN2_var4_U", "Parent" : "10"},
	{"ID" : "38", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_439.BN2_gamma4_U", "Parent" : "10"},
	{"ID" : "39", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_439.Y_U", "Parent" : "10"},
	{"ID" : "40", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_439.U_U", "Parent" : "10"},
	{"ID" : "41", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_439.Y2_U", "Parent" : "10"},
	{"ID" : "42", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_439.U_1_U", "Parent" : "10"},
	{"ID" : "43", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_439.Y5_U", "Parent" : "10"},
	{"ID" : "44", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_439.U_2_U", "Parent" : "10"},
	{"ID" : "45", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_439.Y8_U", "Parent" : "10"},
	{"ID" : "46", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_439.U_3_U", "Parent" : "10"},
	{"ID" : "47", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_439.X0_1_U", "Parent" : "10"},
	{"ID" : "48", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_439.Y_1_U", "Parent" : "10"},
	{"ID" : "49", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_439.U_4_U", "Parent" : "10"},
	{"ID" : "50", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_439.grp_lstm_forward_unidir_fu_636", "Parent" : "10", "Child" : ["51", "52", "53", "70", "71", "72", "73", "74", "75", "76", "77"],
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
			{"Name" : "c_slice", "Type" : "Memory", "Direction" : "IO"},
			{"Name" : "table_exp_Z1_array_s", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "53", "SubInstance" : "grp_generic_tanh_float_s_fu_325", "Port" : "table_exp_Z1_array_s"}]},
			{"Name" : "table_f_Z3_array_V", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "53", "SubInstance" : "grp_generic_tanh_float_s_fu_325", "Port" : "table_f_Z3_array_V"}]},
			{"Name" : "table_f_Z2_array_V", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "53", "SubInstance" : "grp_generic_tanh_float_s_fu_325", "Port" : "table_f_Z2_array_V"}]}]},
	{"ID" : "51", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_439.grp_lstm_forward_unidir_fu_636.c_slice_U", "Parent" : "50"},
	{"ID" : "52", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_439.grp_lstm_forward_unidir_fu_636.z_U", "Parent" : "50"},
	{"ID" : "53", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_439.grp_lstm_forward_unidir_fu_636.grp_generic_tanh_float_s_fu_325", "Parent" : "50", "Child" : ["54", "63", "64", "65", "66", "67", "68", "69"],
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
					{"ID" : "54", "SubInstance" : "grp_exp_generic_double_s_fu_89", "Port" : "table_exp_Z1_array_s"}]},
			{"Name" : "table_f_Z3_array_V", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "54", "SubInstance" : "grp_exp_generic_double_s_fu_89", "Port" : "table_f_Z3_array_V"}]},
			{"Name" : "table_f_Z2_array_V", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "54", "SubInstance" : "grp_exp_generic_double_s_fu_89", "Port" : "table_f_Z2_array_V"}]}]},
	{"ID" : "54", "Level" : "4", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_439.grp_lstm_forward_unidir_fu_636.grp_generic_tanh_float_s_fu_325.grp_exp_generic_double_s_fu_89", "Parent" : "53", "Child" : ["55", "56", "57", "58", "59", "60", "61", "62"],
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
	{"ID" : "55", "Level" : "5", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_439.grp_lstm_forward_unidir_fu_636.grp_generic_tanh_float_s_fu_325.grp_exp_generic_double_s_fu_89.table_exp_Z1_array_s_U", "Parent" : "54"},
	{"ID" : "56", "Level" : "5", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_439.grp_lstm_forward_unidir_fu_636.grp_generic_tanh_float_s_fu_325.grp_exp_generic_double_s_fu_89.table_f_Z3_array_V_U", "Parent" : "54"},
	{"ID" : "57", "Level" : "5", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_439.grp_lstm_forward_unidir_fu_636.grp_generic_tanh_float_s_fu_325.grp_exp_generic_double_s_fu_89.table_f_Z2_array_V_U", "Parent" : "54"},
	{"ID" : "58", "Level" : "5", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_439.grp_lstm_forward_unidir_fu_636.grp_generic_tanh_float_s_fu_325.grp_exp_generic_double_s_fu_89.main_mul_72ns_13s_84_5_1_U16", "Parent" : "54"},
	{"ID" : "59", "Level" : "5", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_439.grp_lstm_forward_unidir_fu_636.grp_generic_tanh_float_s_fu_325.grp_exp_generic_double_s_fu_89.main_mul_36ns_43ns_79_2_1_U17", "Parent" : "54"},
	{"ID" : "60", "Level" : "5", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_439.grp_lstm_forward_unidir_fu_636.grp_generic_tanh_float_s_fu_325.grp_exp_generic_double_s_fu_89.main_mul_44ns_49ns_93_2_1_U18", "Parent" : "54"},
	{"ID" : "61", "Level" : "5", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_439.grp_lstm_forward_unidir_fu_636.grp_generic_tanh_float_s_fu_325.grp_exp_generic_double_s_fu_89.main_mul_50ns_50ns_100_2_1_U19", "Parent" : "54"},
	{"ID" : "62", "Level" : "5", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_439.grp_lstm_forward_unidir_fu_636.grp_generic_tanh_float_s_fu_325.grp_exp_generic_double_s_fu_89.main_mac_muladd_16ns_16s_19s_31_1_1_U20", "Parent" : "54"},
	{"ID" : "63", "Level" : "4", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_439.grp_lstm_forward_unidir_fu_636.grp_generic_tanh_float_s_fu_325.main_faddfsub_32ns_32ns_32_5_full_dsp_1_U30", "Parent" : "53"},
	{"ID" : "64", "Level" : "4", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_439.grp_lstm_forward_unidir_fu_636.grp_generic_tanh_float_s_fu_325.main_fmul_32ns_32ns_32_4_max_dsp_1_U31", "Parent" : "53"},
	{"ID" : "65", "Level" : "4", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_439.grp_lstm_forward_unidir_fu_636.grp_generic_tanh_float_s_fu_325.main_fdiv_32ns_32ns_32_16_1_U32", "Parent" : "53"},
	{"ID" : "66", "Level" : "4", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_439.grp_lstm_forward_unidir_fu_636.grp_generic_tanh_float_s_fu_325.main_fptrunc_64ns_32_2_1_U33", "Parent" : "53"},
	{"ID" : "67", "Level" : "4", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_439.grp_lstm_forward_unidir_fu_636.grp_generic_tanh_float_s_fu_325.main_fpext_32ns_64_2_1_U34", "Parent" : "53"},
	{"ID" : "68", "Level" : "4", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_439.grp_lstm_forward_unidir_fu_636.grp_generic_tanh_float_s_fu_325.main_fcmp_32ns_32ns_1_2_1_U35", "Parent" : "53"},
	{"ID" : "69", "Level" : "4", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_439.grp_lstm_forward_unidir_fu_636.grp_generic_tanh_float_s_fu_325.main_dadd_64ns_64ns_64_5_full_dsp_1_U36", "Parent" : "53"},
	{"ID" : "70", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_439.grp_lstm_forward_unidir_fu_636.main_fadd_32ns_32ns_32_5_full_dsp_1_U42", "Parent" : "50"},
	{"ID" : "71", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_439.grp_lstm_forward_unidir_fu_636.main_fadd_32ns_32ns_32_5_full_dsp_1_U43", "Parent" : "50"},
	{"ID" : "72", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_439.grp_lstm_forward_unidir_fu_636.main_fmul_32ns_32ns_32_4_max_dsp_1_U44", "Parent" : "50"},
	{"ID" : "73", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_439.grp_lstm_forward_unidir_fu_636.main_fmul_32ns_32ns_32_4_max_dsp_1_U45", "Parent" : "50"},
	{"ID" : "74", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_439.grp_lstm_forward_unidir_fu_636.main_fdiv_32ns_32ns_32_16_1_U46", "Parent" : "50"},
	{"ID" : "75", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_439.grp_lstm_forward_unidir_fu_636.main_fdiv_32ns_32ns_32_16_1_U47", "Parent" : "50"},
	{"ID" : "76", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_439.grp_lstm_forward_unidir_fu_636.main_fexp_32ns_32ns_32_9_full_dsp_1_U48", "Parent" : "50"},
	{"ID" : "77", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_439.grp_lstm_forward_unidir_fu_636.main_fexp_32ns_32ns_32_9_full_dsp_1_U49", "Parent" : "50"},
	{"ID" : "78", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_439.grp_generic_tanh_float_s_fu_669", "Parent" : "10", "Child" : ["79", "88", "89", "90", "91", "92", "93", "94"],
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
	{"ID" : "79", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_439.grp_generic_tanh_float_s_fu_669.grp_exp_generic_double_s_fu_89", "Parent" : "78", "Child" : ["80", "81", "82", "83", "84", "85", "86", "87"],
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
	{"ID" : "80", "Level" : "4", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_439.grp_generic_tanh_float_s_fu_669.grp_exp_generic_double_s_fu_89.table_exp_Z1_array_s_U", "Parent" : "79"},
	{"ID" : "81", "Level" : "4", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_439.grp_generic_tanh_float_s_fu_669.grp_exp_generic_double_s_fu_89.table_f_Z3_array_V_U", "Parent" : "79"},
	{"ID" : "82", "Level" : "4", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_439.grp_generic_tanh_float_s_fu_669.grp_exp_generic_double_s_fu_89.table_f_Z2_array_V_U", "Parent" : "79"},
	{"ID" : "83", "Level" : "4", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_439.grp_generic_tanh_float_s_fu_669.grp_exp_generic_double_s_fu_89.main_mul_72ns_13s_84_5_1_U16", "Parent" : "79"},
	{"ID" : "84", "Level" : "4", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_439.grp_generic_tanh_float_s_fu_669.grp_exp_generic_double_s_fu_89.main_mul_36ns_43ns_79_2_1_U17", "Parent" : "79"},
	{"ID" : "85", "Level" : "4", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_439.grp_generic_tanh_float_s_fu_669.grp_exp_generic_double_s_fu_89.main_mul_44ns_49ns_93_2_1_U18", "Parent" : "79"},
	{"ID" : "86", "Level" : "4", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_439.grp_generic_tanh_float_s_fu_669.grp_exp_generic_double_s_fu_89.main_mul_50ns_50ns_100_2_1_U19", "Parent" : "79"},
	{"ID" : "87", "Level" : "4", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_439.grp_generic_tanh_float_s_fu_669.grp_exp_generic_double_s_fu_89.main_mac_muladd_16ns_16s_19s_31_1_1_U20", "Parent" : "79"},
	{"ID" : "88", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_439.grp_generic_tanh_float_s_fu_669.main_faddfsub_32ns_32ns_32_5_full_dsp_1_U30", "Parent" : "78"},
	{"ID" : "89", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_439.grp_generic_tanh_float_s_fu_669.main_fmul_32ns_32ns_32_4_max_dsp_1_U31", "Parent" : "78"},
	{"ID" : "90", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_439.grp_generic_tanh_float_s_fu_669.main_fdiv_32ns_32ns_32_16_1_U32", "Parent" : "78"},
	{"ID" : "91", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_439.grp_generic_tanh_float_s_fu_669.main_fptrunc_64ns_32_2_1_U33", "Parent" : "78"},
	{"ID" : "92", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_439.grp_generic_tanh_float_s_fu_669.main_fpext_32ns_64_2_1_U34", "Parent" : "78"},
	{"ID" : "93", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_439.grp_generic_tanh_float_s_fu_669.main_fcmp_32ns_32ns_1_2_1_U35", "Parent" : "78"},
	{"ID" : "94", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_439.grp_generic_tanh_float_s_fu_669.main_dadd_64ns_64ns_64_5_full_dsp_1_U36", "Parent" : "78"},
	{"ID" : "95", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_439.grp_conv_bn_act_pool_fu_680", "Parent" : "10", "Child" : ["96", "97", "98", "99", "100", "101", "102", "103"],
		"CDFG" : "conv_bn_act_pool",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "0", "ap_idle" : "1",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "64919860", "EstimateLatencyMax" : "64919860",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"Port" : [
			{"Name" : "X", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "Y", "Type" : "Memory", "Direction" : "IO"},
			{"Name" : "U", "Type" : "Memory", "Direction" : "O"},
			{"Name" : "ConvW4", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "BN1_var4", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "BN1_gamma4", "Type" : "Memory", "Direction" : "I"}]},
	{"ID" : "96", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_439.grp_conv_bn_act_pool_fu_680.ConvW4_U", "Parent" : "95"},
	{"ID" : "97", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_439.grp_conv_bn_act_pool_fu_680.BN1_var4_U", "Parent" : "95"},
	{"ID" : "98", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_439.grp_conv_bn_act_pool_fu_680.BN1_gamma4_U", "Parent" : "95"},
	{"ID" : "99", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_439.grp_conv_bn_act_pool_fu_680.main_fadd_32ns_32ns_32_5_full_dsp_1_U85", "Parent" : "95"},
	{"ID" : "100", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_439.grp_conv_bn_act_pool_fu_680.main_fmul_32ns_32ns_32_4_max_dsp_1_U86", "Parent" : "95"},
	{"ID" : "101", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_439.grp_conv_bn_act_pool_fu_680.main_fdiv_32ns_32ns_32_16_1_U87", "Parent" : "95"},
	{"ID" : "102", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_439.grp_conv_bn_act_pool_fu_680.main_fcmp_32ns_32ns_1_2_1_U88", "Parent" : "95"},
	{"ID" : "103", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_439.grp_conv_bn_act_pool_fu_680.main_fsqrt_32ns_32ns_32_12_1_U89", "Parent" : "95"},
	{"ID" : "104", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_439.grp_conv_bn_act_pool_1_fu_693", "Parent" : "10", "Child" : ["105", "106", "107", "108", "109", "110", "111", "112", "113"],
		"CDFG" : "conv_bn_act_pool_1",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "0", "ap_idle" : "1",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "27732148", "EstimateLatencyMax" : "27732148",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"Port" : [
			{"Name" : "Y", "Type" : "Memory", "Direction" : "IO"},
			{"Name" : "U", "Type" : "Memory", "Direction" : "O"},
			{"Name" : "X07", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "ConvW3", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "BN1_var3", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "BN1_gamma3", "Type" : "Memory", "Direction" : "I"}]},
	{"ID" : "105", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_439.grp_conv_bn_act_pool_1_fu_693.X07_U", "Parent" : "104"},
	{"ID" : "106", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_439.grp_conv_bn_act_pool_1_fu_693.ConvW3_U", "Parent" : "104"},
	{"ID" : "107", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_439.grp_conv_bn_act_pool_1_fu_693.BN1_var3_U", "Parent" : "104"},
	{"ID" : "108", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_439.grp_conv_bn_act_pool_1_fu_693.BN1_gamma3_U", "Parent" : "104"},
	{"ID" : "109", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_439.grp_conv_bn_act_pool_1_fu_693.main_fadd_32ns_32ns_32_5_full_dsp_1_U76", "Parent" : "104"},
	{"ID" : "110", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_439.grp_conv_bn_act_pool_1_fu_693.main_fmul_32ns_32ns_32_4_max_dsp_1_U77", "Parent" : "104"},
	{"ID" : "111", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_439.grp_conv_bn_act_pool_1_fu_693.main_fdiv_32ns_32ns_32_16_1_U78", "Parent" : "104"},
	{"ID" : "112", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_439.grp_conv_bn_act_pool_1_fu_693.main_fcmp_32ns_32ns_1_2_1_U79", "Parent" : "104"},
	{"ID" : "113", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_439.grp_conv_bn_act_pool_1_fu_693.main_fsqrt_32ns_32ns_32_12_1_U80", "Parent" : "104"},
	{"ID" : "114", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_439.grp_conv_bn_act_pool_2_fu_707", "Parent" : "10", "Child" : ["115", "116", "117", "118", "119", "120", "121", "122", "123"],
		"CDFG" : "conv_bn_act_pool_2",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "0", "ap_idle" : "1",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "12684148", "EstimateLatencyMax" : "12684148",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"Port" : [
			{"Name" : "Y", "Type" : "Memory", "Direction" : "IO"},
			{"Name" : "U", "Type" : "Memory", "Direction" : "O"},
			{"Name" : "X04", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "ConvW2", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "BN1_var2", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "BN1_gamma2", "Type" : "Memory", "Direction" : "I"}]},
	{"ID" : "115", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_439.grp_conv_bn_act_pool_2_fu_707.X04_U", "Parent" : "114"},
	{"ID" : "116", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_439.grp_conv_bn_act_pool_2_fu_707.ConvW2_U", "Parent" : "114"},
	{"ID" : "117", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_439.grp_conv_bn_act_pool_2_fu_707.BN1_var2_U", "Parent" : "114"},
	{"ID" : "118", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_439.grp_conv_bn_act_pool_2_fu_707.BN1_gamma2_U", "Parent" : "114"},
	{"ID" : "119", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_439.grp_conv_bn_act_pool_2_fu_707.main_fadd_32ns_32ns_32_5_full_dsp_1_U67", "Parent" : "114"},
	{"ID" : "120", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_439.grp_conv_bn_act_pool_2_fu_707.main_fmul_32ns_32ns_32_4_max_dsp_1_U68", "Parent" : "114"},
	{"ID" : "121", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_439.grp_conv_bn_act_pool_2_fu_707.main_fdiv_32ns_32ns_32_16_1_U69", "Parent" : "114"},
	{"ID" : "122", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_439.grp_conv_bn_act_pool_2_fu_707.main_fcmp_32ns_32ns_1_2_1_U70", "Parent" : "114"},
	{"ID" : "123", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_439.grp_conv_bn_act_pool_2_fu_707.main_fsqrt_32ns_32ns_32_12_1_U71", "Parent" : "114"},
	{"ID" : "124", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_439.grp_conv_bn_act_pool_3_fu_721", "Parent" : "10", "Child" : ["125", "126", "127", "128", "129", "130", "131", "132", "133"],
		"CDFG" : "conv_bn_act_pool_3",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "0", "ap_idle" : "1",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "6045750", "EstimateLatencyMax" : "6045750",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"Port" : [
			{"Name" : "Y", "Type" : "Memory", "Direction" : "IO"},
			{"Name" : "U", "Type" : "Memory", "Direction" : "O"},
			{"Name" : "X01", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "ConvW1", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "BN1_var1", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "BN1_gamma1", "Type" : "Memory", "Direction" : "I"}]},
	{"ID" : "125", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_439.grp_conv_bn_act_pool_3_fu_721.X01_U", "Parent" : "124"},
	{"ID" : "126", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_439.grp_conv_bn_act_pool_3_fu_721.ConvW1_U", "Parent" : "124"},
	{"ID" : "127", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_439.grp_conv_bn_act_pool_3_fu_721.BN1_var1_U", "Parent" : "124"},
	{"ID" : "128", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_439.grp_conv_bn_act_pool_3_fu_721.BN1_gamma1_U", "Parent" : "124"},
	{"ID" : "129", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_439.grp_conv_bn_act_pool_3_fu_721.main_fadd_32ns_32ns_32_5_full_dsp_1_U58", "Parent" : "124"},
	{"ID" : "130", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_439.grp_conv_bn_act_pool_3_fu_721.main_fmul_32ns_32ns_32_4_max_dsp_1_U59", "Parent" : "124"},
	{"ID" : "131", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_439.grp_conv_bn_act_pool_3_fu_721.main_fdiv_32ns_32ns_32_16_1_U60", "Parent" : "124"},
	{"ID" : "132", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_439.grp_conv_bn_act_pool_3_fu_721.main_fcmp_32ns_32ns_1_2_1_U61", "Parent" : "124"},
	{"ID" : "133", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_439.grp_conv_bn_act_pool_3_fu_721.main_fsqrt_32ns_32ns_32_12_1_U62", "Parent" : "124"},
	{"ID" : "134", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_439.grp_conv_bn_act_pool_4_fu_735", "Parent" : "10", "Child" : ["135", "136", "137", "138", "139", "140", "141", "142", "143"],
		"CDFG" : "conv_bn_act_pool_4",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "0", "ap_idle" : "1",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "2950976", "EstimateLatencyMax" : "2950976",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"Port" : [
			{"Name" : "Y", "Type" : "Memory", "Direction" : "IO"},
			{"Name" : "U", "Type" : "Memory", "Direction" : "O"},
			{"Name" : "X0", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "ConvW0", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "BN1_var0", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "BN1_gamma0", "Type" : "Memory", "Direction" : "I"}]},
	{"ID" : "135", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_439.grp_conv_bn_act_pool_4_fu_735.X0_U", "Parent" : "134"},
	{"ID" : "136", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_439.grp_conv_bn_act_pool_4_fu_735.ConvW0_U", "Parent" : "134"},
	{"ID" : "137", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_439.grp_conv_bn_act_pool_4_fu_735.BN1_var0_U", "Parent" : "134"},
	{"ID" : "138", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_439.grp_conv_bn_act_pool_4_fu_735.BN1_gamma0_U", "Parent" : "134"},
	{"ID" : "139", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_439.grp_conv_bn_act_pool_4_fu_735.main_fadd_32ns_32ns_32_5_full_dsp_1_U1", "Parent" : "134"},
	{"ID" : "140", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_439.grp_conv_bn_act_pool_4_fu_735.main_fmul_32ns_32ns_32_4_max_dsp_1_U2", "Parent" : "134"},
	{"ID" : "141", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_439.grp_conv_bn_act_pool_4_fu_735.main_fdiv_32ns_32ns_32_16_1_U3", "Parent" : "134"},
	{"ID" : "142", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_439.grp_conv_bn_act_pool_4_fu_735.main_fcmp_32ns_32ns_1_2_1_U4", "Parent" : "134"},
	{"ID" : "143", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_439.grp_conv_bn_act_pool_4_fu_735.main_fsqrt_32ns_32ns_32_12_1_U5", "Parent" : "134"},
	{"ID" : "144", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_439.main_fadd_32ns_32ns_32_5_full_dsp_1_U94", "Parent" : "10"},
	{"ID" : "145", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_439.main_fmul_32ns_32ns_32_4_max_dsp_1_U95", "Parent" : "10"},
	{"ID" : "146", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_439.main_fdiv_32ns_32ns_32_16_1_U96", "Parent" : "10"},
	{"ID" : "147", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrol_fu_439.main_fsqrt_32ns_32ns_32_12_1_U97", "Parent" : "10"},
	{"ID" : "148", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.main_fadd_32ns_32ns_32_5_full_dsp_1_U119", "Parent" : "0"},
	{"ID" : "149", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.main_fmul_32ns_32ns_32_4_max_dsp_1_U120", "Parent" : "0"},
	{"ID" : "150", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.main_fdiv_32ns_32ns_32_16_1_U121", "Parent" : "0"},
	{"ID" : "151", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.main_fcmp_32ns_32ns_1_2_1_U122", "Parent" : "0"},
	{"ID" : "152", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.main_fsqrt_32ns_32ns_32_12_1_U123", "Parent" : "0"}]}


set ArgLastReadFirstWriteLatency {
	main {
		X0 {Type I LastRead -1 FirstWrite -1}
		ConvW0 {Type I LastRead -1 FirstWrite -1}
		BN1_var0 {Type I LastRead -1 FirstWrite -1}
		BN1_gamma0 {Type I LastRead -1 FirstWrite -1}
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
		X01 {Type I LastRead -1 FirstWrite -1}
		ConvW1 {Type I LastRead -1 FirstWrite -1}
		BN1_var1 {Type I LastRead -1 FirstWrite -1}
		BN1_gamma1 {Type I LastRead -1 FirstWrite -1}
		LSTM_W_ifog1 {Type I LastRead -1 FirstWrite -1}
		LSTM_R_ifog1 {Type I LastRead -1 FirstWrite -1}
		LSTM_b_ifog1 {Type I LastRead -1 FirstWrite -1}
		BN2_var1 {Type I LastRead -1 FirstWrite -1}
		BN2_gamma1 {Type I LastRead -1 FirstWrite -1}
		X04 {Type I LastRead -1 FirstWrite -1}
		ConvW2 {Type I LastRead -1 FirstWrite -1}
		BN1_var2 {Type I LastRead -1 FirstWrite -1}
		BN1_gamma2 {Type I LastRead -1 FirstWrite -1}
		LSTM_W_ifog2 {Type I LastRead -1 FirstWrite -1}
		LSTM_R_ifog2 {Type I LastRead -1 FirstWrite -1}
		LSTM_b_ifog2 {Type I LastRead -1 FirstWrite -1}
		BN2_var2 {Type I LastRead -1 FirstWrite -1}
		BN2_gamma2 {Type I LastRead -1 FirstWrite -1}
		X07 {Type I LastRead -1 FirstWrite -1}
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
		merged {Type IO LastRead 29 FirstWrite 1}
		X0 {Type I LastRead -1 FirstWrite -1}
		ConvW0 {Type I LastRead -1 FirstWrite -1}
		BN1_var0 {Type I LastRead -1 FirstWrite -1}
		BN1_gamma0 {Type I LastRead -1 FirstWrite -1}
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
		X01 {Type I LastRead -1 FirstWrite -1}
		ConvW1 {Type I LastRead -1 FirstWrite -1}
		BN1_var1 {Type I LastRead -1 FirstWrite -1}
		BN1_gamma1 {Type I LastRead -1 FirstWrite -1}
		LSTM_W_ifog1 {Type I LastRead -1 FirstWrite -1}
		LSTM_R_ifog1 {Type I LastRead -1 FirstWrite -1}
		LSTM_b_ifog1 {Type I LastRead -1 FirstWrite -1}
		BN2_var1 {Type I LastRead -1 FirstWrite -1}
		BN2_gamma1 {Type I LastRead -1 FirstWrite -1}
		X04 {Type I LastRead -1 FirstWrite -1}
		ConvW2 {Type I LastRead -1 FirstWrite -1}
		BN1_var2 {Type I LastRead -1 FirstWrite -1}
		BN1_gamma2 {Type I LastRead -1 FirstWrite -1}
		LSTM_W_ifog2 {Type I LastRead -1 FirstWrite -1}
		LSTM_R_ifog2 {Type I LastRead -1 FirstWrite -1}
		LSTM_b_ifog2 {Type I LastRead -1 FirstWrite -1}
		BN2_var2 {Type I LastRead -1 FirstWrite -1}
		BN2_gamma2 {Type I LastRead -1 FirstWrite -1}
		X07 {Type I LastRead -1 FirstWrite -1}
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
		x {Type I LastRead 4 FirstWrite -1}
		W_ifog {Type I LastRead 6 FirstWrite -1}
		R_ifog {Type I LastRead 7 FirstWrite -1}
		b_ifog {Type I LastRead 3 FirstWrite -1}
		h_last {Type IO LastRead 5 FirstWrite 1}
		c_slice {Type IO LastRead -1 FirstWrite -1}
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
		Y {Type IO LastRead 20 FirstWrite 3}
		U {Type O LastRead -1 FirstWrite 22}
		ConvW4 {Type I LastRead -1 FirstWrite -1}
		BN1_var4 {Type I LastRead -1 FirstWrite -1}
		BN1_gamma4 {Type I LastRead -1 FirstWrite -1}}
	conv_bn_act_pool_1 {
		Y {Type IO LastRead 20 FirstWrite 3}
		U {Type O LastRead -1 FirstWrite 22}
		X07 {Type I LastRead -1 FirstWrite -1}
		ConvW3 {Type I LastRead -1 FirstWrite -1}
		BN1_var3 {Type I LastRead -1 FirstWrite -1}
		BN1_gamma3 {Type I LastRead -1 FirstWrite -1}}
	conv_bn_act_pool_2 {
		Y {Type IO LastRead 20 FirstWrite 3}
		U {Type O LastRead -1 FirstWrite 22}
		X04 {Type I LastRead -1 FirstWrite -1}
		ConvW2 {Type I LastRead -1 FirstWrite -1}
		BN1_var2 {Type I LastRead -1 FirstWrite -1}
		BN1_gamma2 {Type I LastRead -1 FirstWrite -1}}
	conv_bn_act_pool_3 {
		Y {Type IO LastRead 20 FirstWrite 3}
		U {Type O LastRead -1 FirstWrite 22}
		X01 {Type I LastRead -1 FirstWrite -1}
		ConvW1 {Type I LastRead -1 FirstWrite -1}
		BN1_var1 {Type I LastRead -1 FirstWrite -1}
		BN1_gamma1 {Type I LastRead -1 FirstWrite -1}}
	conv_bn_act_pool_4 {
		Y {Type IO LastRead 20 FirstWrite 3}
		U {Type O LastRead -1 FirstWrite 22}
		X0 {Type I LastRead -1 FirstWrite -1}
		ConvW0 {Type I LastRead -1 FirstWrite -1}
		BN1_var0 {Type I LastRead -1 FirstWrite -1}
		BN1_gamma0 {Type I LastRead -1 FirstWrite -1}}}

set hasDtUnsupportedChannel 0

set PerformanceInfo {[
	{"Name" : "Latency", "Min" : "120638908", "Max" : "120878908"}
	, {"Name" : "Interval", "Min" : "120638909", "Max" : "120878909"}
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
