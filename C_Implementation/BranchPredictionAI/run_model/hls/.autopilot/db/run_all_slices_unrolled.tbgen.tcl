set moduleName run_all_slices_unrolled
set isTopModule 0
set isCombinational 0
set isDatapathOnly 0
set isPipelined 0
set pipeline_type none
set FunctionProtocol ap_ctrl_hs
set isOneStateSeq 0
set ProfileFlag 0
set StallSigGenFlag 0
set isEnableWaveformDebug 1
set hasInterrupt 0
set DLRegFirstOffset 0
set DLRegItemOffset 0
set svuvm_can_support 1
set cdfgNum 35
set C_modelName {run_all_slices_unrolled}
set C_modelType { void 0 }
set ap_memory_interface_dict [dict create]
dict set ap_memory_interface_dict merged { MEM_WIDTH 16 MEM_SIZE 64 MASTER_TYPE BRAM_CTRL MEM_ADDRESS_MODE WORD_ADDRESS PACKAGE_IO port READ_LATENCY 1 }
set C_modelArgList {
	{ merged int 16 regular {array 32 { 0 1 } 1 1 }  }
}
set hasAXIMCache 0
set l_AXIML2Cache [list]
set AXIMCacheInstDict [dict create]
set C_modelArgMapList {[ 
	{ "Name" : "merged", "interface" : "memory", "bitwidth" : 16, "direction" : "READWRITE"} ]}
# RTL Port declarations: 
set portNum 47
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
	{ merged_address1 sc_out sc_lv 5 signal 0 } 
	{ merged_ce1 sc_out sc_logic 1 signal 0 } 
	{ merged_q1 sc_in sc_lv 16 signal 0 } 
	{ grp_fu_823_p_din0 sc_out sc_lv 16 signal -1 } 
	{ grp_fu_823_p_din1 sc_out sc_lv 16 signal -1 } 
	{ grp_fu_823_p_dout0 sc_in sc_lv 16 signal -1 } 
	{ grp_fu_823_p_ce sc_out sc_logic 1 signal -1 } 
	{ grp_fu_359_p_din0 sc_out sc_lv 16 signal -1 } 
	{ grp_fu_359_p_din1 sc_out sc_lv 16 signal -1 } 
	{ grp_fu_359_p_dout0 sc_in sc_lv 16 signal -1 } 
	{ grp_fu_359_p_ce sc_out sc_logic 1 signal -1 } 
	{ grp_fu_827_p_din0 sc_out sc_lv 16 signal -1 } 
	{ grp_fu_827_p_din1 sc_out sc_lv 16 signal -1 } 
	{ grp_fu_827_p_opcode sc_out sc_lv 5 signal -1 } 
	{ grp_fu_827_p_dout0 sc_in sc_lv 1 signal -1 } 
	{ grp_fu_827_p_ce sc_out sc_logic 1 signal -1 } 
	{ grp_fu_831_p_din0 sc_out sc_lv 32 signal -1 } 
	{ grp_fu_831_p_din1 sc_out sc_lv 32 signal -1 } 
	{ grp_fu_831_p_dout0 sc_in sc_lv 32 signal -1 } 
	{ grp_fu_831_p_ce sc_out sc_logic 1 signal -1 } 
	{ grp_fu_835_p_din0 sc_out sc_lv 32 signal -1 } 
	{ grp_fu_835_p_din1 sc_out sc_lv 32 signal -1 } 
	{ grp_fu_835_p_dout0 sc_in sc_lv 32 signal -1 } 
	{ grp_fu_835_p_ce sc_out sc_logic 1 signal -1 } 
	{ grp_fu_839_p_din0 sc_out sc_lv 32 signal -1 } 
	{ grp_fu_839_p_dout0 sc_in sc_lv 16 signal -1 } 
	{ grp_fu_839_p_ce sc_out sc_logic 1 signal -1 } 
	{ grp_fu_842_p_din0 sc_out sc_lv 16 signal -1 } 
	{ grp_fu_842_p_dout0 sc_in sc_lv 32 signal -1 } 
	{ grp_fu_842_p_ce sc_out sc_logic 1 signal -1 } 
	{ grp_fu_845_p_din0 sc_out sc_lv 16 signal -1 } 
	{ grp_fu_845_p_dout0 sc_in sc_lv 32 signal -1 } 
	{ grp_fu_845_p_ce sc_out sc_logic 1 signal -1 } 
	{ grp_fu_365_p_din0 sc_out sc_lv 16 signal -1 } 
	{ grp_fu_365_p_din1 sc_out sc_lv 16 signal -1 } 
	{ grp_fu_365_p_dout0 sc_in sc_lv 16 signal -1 } 
	{ grp_fu_365_p_ce sc_out sc_logic 1 signal -1 } 
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
 	{ "name": "merged_address1", "direction": "out", "datatype": "sc_lv", "bitwidth":5, "type": "signal", "bundle":{"name": "merged", "role": "address1" }} , 
 	{ "name": "merged_ce1", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "merged", "role": "ce1" }} , 
 	{ "name": "merged_q1", "direction": "in", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "merged", "role": "q1" }} , 
 	{ "name": "grp_fu_823_p_din0", "direction": "out", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "grp_fu_823_p_din0", "role": "default" }} , 
 	{ "name": "grp_fu_823_p_din1", "direction": "out", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "grp_fu_823_p_din1", "role": "default" }} , 
 	{ "name": "grp_fu_823_p_dout0", "direction": "in", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "grp_fu_823_p_dout0", "role": "default" }} , 
 	{ "name": "grp_fu_823_p_ce", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "grp_fu_823_p_ce", "role": "default" }} , 
 	{ "name": "grp_fu_359_p_din0", "direction": "out", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "grp_fu_359_p_din0", "role": "default" }} , 
 	{ "name": "grp_fu_359_p_din1", "direction": "out", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "grp_fu_359_p_din1", "role": "default" }} , 
 	{ "name": "grp_fu_359_p_dout0", "direction": "in", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "grp_fu_359_p_dout0", "role": "default" }} , 
 	{ "name": "grp_fu_359_p_ce", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "grp_fu_359_p_ce", "role": "default" }} , 
 	{ "name": "grp_fu_827_p_din0", "direction": "out", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "grp_fu_827_p_din0", "role": "default" }} , 
 	{ "name": "grp_fu_827_p_din1", "direction": "out", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "grp_fu_827_p_din1", "role": "default" }} , 
 	{ "name": "grp_fu_827_p_opcode", "direction": "out", "datatype": "sc_lv", "bitwidth":5, "type": "signal", "bundle":{"name": "grp_fu_827_p_opcode", "role": "default" }} , 
 	{ "name": "grp_fu_827_p_dout0", "direction": "in", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "grp_fu_827_p_dout0", "role": "default" }} , 
 	{ "name": "grp_fu_827_p_ce", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "grp_fu_827_p_ce", "role": "default" }} , 
 	{ "name": "grp_fu_831_p_din0", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "grp_fu_831_p_din0", "role": "default" }} , 
 	{ "name": "grp_fu_831_p_din1", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "grp_fu_831_p_din1", "role": "default" }} , 
 	{ "name": "grp_fu_831_p_dout0", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "grp_fu_831_p_dout0", "role": "default" }} , 
 	{ "name": "grp_fu_831_p_ce", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "grp_fu_831_p_ce", "role": "default" }} , 
 	{ "name": "grp_fu_835_p_din0", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "grp_fu_835_p_din0", "role": "default" }} , 
 	{ "name": "grp_fu_835_p_din1", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "grp_fu_835_p_din1", "role": "default" }} , 
 	{ "name": "grp_fu_835_p_dout0", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "grp_fu_835_p_dout0", "role": "default" }} , 
 	{ "name": "grp_fu_835_p_ce", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "grp_fu_835_p_ce", "role": "default" }} , 
 	{ "name": "grp_fu_839_p_din0", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "grp_fu_839_p_din0", "role": "default" }} , 
 	{ "name": "grp_fu_839_p_dout0", "direction": "in", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "grp_fu_839_p_dout0", "role": "default" }} , 
 	{ "name": "grp_fu_839_p_ce", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "grp_fu_839_p_ce", "role": "default" }} , 
 	{ "name": "grp_fu_842_p_din0", "direction": "out", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "grp_fu_842_p_din0", "role": "default" }} , 
 	{ "name": "grp_fu_842_p_dout0", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "grp_fu_842_p_dout0", "role": "default" }} , 
 	{ "name": "grp_fu_842_p_ce", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "grp_fu_842_p_ce", "role": "default" }} , 
 	{ "name": "grp_fu_845_p_din0", "direction": "out", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "grp_fu_845_p_din0", "role": "default" }} , 
 	{ "name": "grp_fu_845_p_dout0", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "grp_fu_845_p_dout0", "role": "default" }} , 
 	{ "name": "grp_fu_845_p_ce", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "grp_fu_845_p_ce", "role": "default" }} , 
 	{ "name": "grp_fu_365_p_din0", "direction": "out", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "grp_fu_365_p_din0", "role": "default" }} , 
 	{ "name": "grp_fu_365_p_din1", "direction": "out", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "grp_fu_365_p_din1", "role": "default" }} , 
 	{ "name": "grp_fu_365_p_dout0", "direction": "in", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "grp_fu_365_p_dout0", "role": "default" }} , 
 	{ "name": "grp_fu_365_p_ce", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "grp_fu_365_p_ce", "role": "default" }}  ]}

set RtlHierarchyInfo {[
	{"ID" : "0", "Level" : "0", "Path" : "`AUTOTB_DUT_INST", "Parent" : "", "Child" : ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "15", "69", "73", "86", "89", "93", "106", "109", "113", "126", "129", "133", "146", "149", "153", "156", "157", "158", "159", "160", "161", "162", "163", "164", "165", "166", "167", "168", "169"],
		"CDFG" : "run_all_slices_unrolled",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "0", "ap_idle" : "1", "real_start" : "0",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "40081252", "EstimateLatencyMax" : "40125532",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"IsBlackBox" : "0",
		"Port" : [
			{"Name" : "merged", "Type" : "Memory", "Direction" : "IO",
				"SubConnect" : [
					{"ID" : "13", "SubInstance" : "grp_run_all_slices_unrolled_Pipeline_MergedLoop_fu_96", "Port" : "merged", "Inst_start_state" : "1", "Inst_end_state" : "2"},
					{"ID" : "86", "SubInstance" : "grp_run_all_slices_unrolled_Pipeline_MergedLoop0_fu_144", "Port" : "merged", "Inst_start_state" : "5", "Inst_end_state" : "6"},
					{"ID" : "106", "SubInstance" : "grp_run_all_slices_unrolled_Pipeline_MergedLoop1_fu_172", "Port" : "merged", "Inst_start_state" : "11", "Inst_end_state" : "12"},
					{"ID" : "126", "SubInstance" : "grp_run_all_slices_unrolled_Pipeline_MergedLoop2_fu_200", "Port" : "merged", "Inst_start_state" : "17", "Inst_end_state" : "18"},
					{"ID" : "146", "SubInstance" : "grp_run_all_slices_unrolled_Pipeline_MergedLoop3_fu_228", "Port" : "merged", "Inst_start_state" : "23", "Inst_end_state" : "24"},
					{"ID" : "153", "SubInstance" : "grp_run_all_slices_unrolled_Pipeline_MergedLoop4_fu_246", "Port" : "merged", "Inst_start_state" : "29", "Inst_end_state" : "30"}]},
			{"Name" : "h_slice", "Type" : "Memory", "Direction" : "IO",
				"SubConnect" : [
					{"ID" : "15", "SubInstance" : "grp_lstm_forward_unidir_fu_102", "Port" : "h_slice", "Inst_start_state" : "25", "Inst_end_state" : "26"},
					{"ID" : "69", "SubInstance" : "grp_run_all_slices_unrolled_Pipeline_Loop_BN_fu_124", "Port" : "h_slice", "Inst_start_state" : "3", "Inst_end_state" : "4"},
					{"ID" : "86", "SubInstance" : "grp_run_all_slices_unrolled_Pipeline_MergedLoop0_fu_144", "Port" : "h_slice", "Inst_start_state" : "5", "Inst_end_state" : "6"},
					{"ID" : "89", "SubInstance" : "grp_run_all_slices_unrolled_Pipeline_Loop_BN2_fu_152", "Port" : "h_slice", "Inst_start_state" : "9", "Inst_end_state" : "10"},
					{"ID" : "106", "SubInstance" : "grp_run_all_slices_unrolled_Pipeline_MergedLoop1_fu_172", "Port" : "h_slice", "Inst_start_state" : "11", "Inst_end_state" : "12"},
					{"ID" : "109", "SubInstance" : "grp_run_all_slices_unrolled_Pipeline_Loop_BN3_fu_180", "Port" : "h_slice", "Inst_start_state" : "15", "Inst_end_state" : "16"},
					{"ID" : "126", "SubInstance" : "grp_run_all_slices_unrolled_Pipeline_MergedLoop2_fu_200", "Port" : "h_slice", "Inst_start_state" : "17", "Inst_end_state" : "18"},
					{"ID" : "129", "SubInstance" : "grp_run_all_slices_unrolled_Pipeline_Loop_BN4_fu_208", "Port" : "h_slice", "Inst_start_state" : "21", "Inst_end_state" : "22"},
					{"ID" : "146", "SubInstance" : "grp_run_all_slices_unrolled_Pipeline_MergedLoop3_fu_228", "Port" : "h_slice", "Inst_start_state" : "23", "Inst_end_state" : "24"},
					{"ID" : "149", "SubInstance" : "grp_run_all_slices_unrolled_Pipeline_Loop_BN5_fu_236", "Port" : "h_slice", "Inst_start_state" : "27", "Inst_end_state" : "28"},
					{"ID" : "153", "SubInstance" : "grp_run_all_slices_unrolled_Pipeline_MergedLoop4_fu_246", "Port" : "h_slice", "Inst_start_state" : "29", "Inst_end_state" : "30"}]},
			{"Name" : "c_slice", "Type" : "Memory", "Direction" : "IO",
				"SubConnect" : [
					{"ID" : "15", "SubInstance" : "grp_lstm_forward_unidir_fu_102", "Port" : "c_slice", "Inst_start_state" : "25", "Inst_end_state" : "26"}]},
			{"Name" : "U_slice", "Type" : "Memory", "Direction" : "IO",
				"SubConnect" : [
					{"ID" : "15", "SubInstance" : "grp_lstm_forward_unidir_fu_102", "Port" : "U_slice", "Inst_start_state" : "25", "Inst_end_state" : "26"},
					{"ID" : "73", "SubInstance" : "grp_conv_bn_act_pool_fu_134", "Port" : "U_slice", "Inst_start_state" : "3", "Inst_end_state" : "4"},
					{"ID" : "93", "SubInstance" : "grp_conv_bn_act_pool_2_fu_162", "Port" : "U_slice", "Inst_start_state" : "9", "Inst_end_state" : "10"},
					{"ID" : "113", "SubInstance" : "grp_conv_bn_act_pool_3_fu_190", "Port" : "U_slice", "Inst_start_state" : "15", "Inst_end_state" : "16"},
					{"ID" : "133", "SubInstance" : "grp_conv_bn_act_pool_4_fu_218", "Port" : "U_slice", "Inst_start_state" : "21", "Inst_end_state" : "22"}]},
			{"Name" : "LSTM_W_ifog0", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "15", "SubInstance" : "grp_lstm_forward_unidir_fu_102", "Port" : "W_ifog", "Inst_start_state" : "25", "Inst_end_state" : "26"}]},
			{"Name" : "LSTM_R_ifog0", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "15", "SubInstance" : "grp_lstm_forward_unidir_fu_102", "Port" : "R_ifog", "Inst_start_state" : "25", "Inst_end_state" : "26"}]},
			{"Name" : "BN2_var0", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "69", "SubInstance" : "grp_run_all_slices_unrolled_Pipeline_Loop_BN_fu_124", "Port" : "BN2_var0", "Inst_start_state" : "3", "Inst_end_state" : "4"}]},
			{"Name" : "BN2_gamma0", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "69", "SubInstance" : "grp_run_all_slices_unrolled_Pipeline_Loop_BN_fu_124", "Port" : "BN2_gamma0", "Inst_start_state" : "3", "Inst_end_state" : "4"}]},
			{"Name" : "X_slice12", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "73", "SubInstance" : "grp_conv_bn_act_pool_fu_134", "Port" : "X_slice12", "Inst_start_state" : "3", "Inst_end_state" : "4"}]},
			{"Name" : "ConvW1", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "73", "SubInstance" : "grp_conv_bn_act_pool_fu_134", "Port" : "ConvW1", "Inst_start_state" : "3", "Inst_end_state" : "4"}]},
			{"Name" : "LSTM_W_ifog1", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "15", "SubInstance" : "grp_lstm_forward_unidir_fu_102", "Port" : "W_ifog", "Inst_start_state" : "25", "Inst_end_state" : "26"}]},
			{"Name" : "LSTM_R_ifog1", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "15", "SubInstance" : "grp_lstm_forward_unidir_fu_102", "Port" : "R_ifog", "Inst_start_state" : "25", "Inst_end_state" : "26"}]},
			{"Name" : "BN2_var1", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "89", "SubInstance" : "grp_run_all_slices_unrolled_Pipeline_Loop_BN2_fu_152", "Port" : "BN2_var1", "Inst_start_state" : "9", "Inst_end_state" : "10"}]},
			{"Name" : "BN2_gamma1", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "89", "SubInstance" : "grp_run_all_slices_unrolled_Pipeline_Loop_BN2_fu_152", "Port" : "BN2_gamma1", "Inst_start_state" : "9", "Inst_end_state" : "10"}]},
			{"Name" : "X_slice27", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "93", "SubInstance" : "grp_conv_bn_act_pool_2_fu_162", "Port" : "X_slice27", "Inst_start_state" : "9", "Inst_end_state" : "10"}]},
			{"Name" : "ConvW2", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "93", "SubInstance" : "grp_conv_bn_act_pool_2_fu_162", "Port" : "ConvW2", "Inst_start_state" : "9", "Inst_end_state" : "10"}]},
			{"Name" : "LSTM_W_ifog2", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "15", "SubInstance" : "grp_lstm_forward_unidir_fu_102", "Port" : "W_ifog", "Inst_start_state" : "25", "Inst_end_state" : "26"}]},
			{"Name" : "LSTM_R_ifog2", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "15", "SubInstance" : "grp_lstm_forward_unidir_fu_102", "Port" : "R_ifog", "Inst_start_state" : "25", "Inst_end_state" : "26"}]},
			{"Name" : "BN2_var2", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "109", "SubInstance" : "grp_run_all_slices_unrolled_Pipeline_Loop_BN3_fu_180", "Port" : "BN2_var2", "Inst_start_state" : "15", "Inst_end_state" : "16"}]},
			{"Name" : "BN2_gamma2", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "109", "SubInstance" : "grp_run_all_slices_unrolled_Pipeline_Loop_BN3_fu_180", "Port" : "BN2_gamma2", "Inst_start_state" : "15", "Inst_end_state" : "16"}]},
			{"Name" : "X_slice42", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "113", "SubInstance" : "grp_conv_bn_act_pool_3_fu_190", "Port" : "X_slice42", "Inst_start_state" : "15", "Inst_end_state" : "16"}]},
			{"Name" : "ConvW3", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "113", "SubInstance" : "grp_conv_bn_act_pool_3_fu_190", "Port" : "ConvW3", "Inst_start_state" : "15", "Inst_end_state" : "16"}]},
			{"Name" : "LSTM_W_ifog3", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "15", "SubInstance" : "grp_lstm_forward_unidir_fu_102", "Port" : "W_ifog", "Inst_start_state" : "25", "Inst_end_state" : "26"}]},
			{"Name" : "LSTM_R_ifog3", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "15", "SubInstance" : "grp_lstm_forward_unidir_fu_102", "Port" : "R_ifog", "Inst_start_state" : "25", "Inst_end_state" : "26"}]},
			{"Name" : "BN2_var3", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "129", "SubInstance" : "grp_run_all_slices_unrolled_Pipeline_Loop_BN4_fu_208", "Port" : "BN2_var3", "Inst_start_state" : "21", "Inst_end_state" : "22"}]},
			{"Name" : "BN2_gamma3", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "129", "SubInstance" : "grp_run_all_slices_unrolled_Pipeline_Loop_BN4_fu_208", "Port" : "BN2_gamma3", "Inst_start_state" : "21", "Inst_end_state" : "22"}]},
			{"Name" : "X_slice57", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "133", "SubInstance" : "grp_conv_bn_act_pool_4_fu_218", "Port" : "X_slice57", "Inst_start_state" : "21", "Inst_end_state" : "22"}]},
			{"Name" : "ConvW4", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "133", "SubInstance" : "grp_conv_bn_act_pool_4_fu_218", "Port" : "ConvW4", "Inst_start_state" : "21", "Inst_end_state" : "22"}]},
			{"Name" : "LSTM_W_ifog4", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "15", "SubInstance" : "grp_lstm_forward_unidir_fu_102", "Port" : "W_ifog", "Inst_start_state" : "25", "Inst_end_state" : "26"}]},
			{"Name" : "LSTM_R_ifog4", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "15", "SubInstance" : "grp_lstm_forward_unidir_fu_102", "Port" : "R_ifog", "Inst_start_state" : "25", "Inst_end_state" : "26"}]},
			{"Name" : "BN2_var4", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "149", "SubInstance" : "grp_run_all_slices_unrolled_Pipeline_Loop_BN5_fu_236", "Port" : "BN2_var4", "Inst_start_state" : "27", "Inst_end_state" : "28"}]},
			{"Name" : "BN2_gamma4", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "149", "SubInstance" : "grp_run_all_slices_unrolled_Pipeline_Loop_BN5_fu_236", "Port" : "BN2_gamma4", "Inst_start_state" : "27", "Inst_end_state" : "28"}]}]},
	{"ID" : "1", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.h_slice_U", "Parent" : "0"},
	{"ID" : "2", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.U_slice_U", "Parent" : "0"},
	{"ID" : "3", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.LSTM_W_ifog0_U", "Parent" : "0"},
	{"ID" : "4", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.LSTM_R_ifog0_U", "Parent" : "0"},
	{"ID" : "5", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.LSTM_W_ifog1_U", "Parent" : "0"},
	{"ID" : "6", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.LSTM_R_ifog1_U", "Parent" : "0"},
	{"ID" : "7", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.LSTM_W_ifog2_U", "Parent" : "0"},
	{"ID" : "8", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.LSTM_R_ifog2_U", "Parent" : "0"},
	{"ID" : "9", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.LSTM_W_ifog3_U", "Parent" : "0"},
	{"ID" : "10", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.LSTM_R_ifog3_U", "Parent" : "0"},
	{"ID" : "11", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.LSTM_W_ifog4_U", "Parent" : "0"},
	{"ID" : "12", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.LSTM_R_ifog4_U", "Parent" : "0"},
	{"ID" : "13", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrolled_Pipeline_MergedLoop_fu_96", "Parent" : "0", "Child" : ["14"],
		"CDFG" : "run_all_slices_unrolled_Pipeline_MergedLoop",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "0", "ap_idle" : "1", "real_start" : "0",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "34", "EstimateLatencyMax" : "34",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"IsBlackBox" : "0",
		"Port" : [
			{"Name" : "merged", "Type" : "Memory", "Direction" : "O"}],
		"Loop" : [
			{"Name" : "MergedLoop", "PipelineType" : "UPC",
				"LoopDec" : {"FSMBitwidth" : "1", "FirstState" : "ap_ST_fsm_state1", "FirstStateIter" : "", "FirstStateBlock" : "ap_ST_fsm_state1_blk", "LastState" : "ap_ST_fsm_state1", "LastStateIter" : "", "LastStateBlock" : "ap_ST_fsm_state1_blk", "QuitState" : "ap_ST_fsm_state1", "QuitStateIter" : "", "QuitStateBlock" : "ap_ST_fsm_state1_blk", "OneDepthLoop" : "1", "has_ap_ctrl" : "1", "has_continue" : "0"}}]},
	{"ID" : "14", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrolled_Pipeline_MergedLoop_fu_96.flow_control_loop_pipe_sequential_init_U", "Parent" : "13"},
	{"ID" : "15", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.grp_lstm_forward_unidir_fu_102", "Parent" : "0", "Child" : ["16", "17", "18", "19", "20", "21", "22", "24", "28", "61", "62", "63", "64", "65", "66", "67", "68"],
		"CDFG" : "lstm_forward_unidir",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "0", "ap_idle" : "1", "real_start" : "0",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "101244", "EstimateLatencyMax" : "101244",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"IsBlackBox" : "0",
		"Port" : [
			{"Name" : "W_ifog", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "R_ifog", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "h_slice", "Type" : "Memory", "Direction" : "IO",
				"SubConnect" : [
					{"ID" : "22", "SubInstance" : "grp_lstm_forward_unidir_Pipeline_VITIS_LOOP_145_1_fu_356", "Port" : "h_slice", "Inst_start_state" : "1", "Inst_end_state" : "2"},
					{"ID" : "28", "SubInstance" : "grp_lstm_forward_unidir_Pipeline_Loop2_4LSTM_fu_373", "Port" : "h_slice", "Inst_start_state" : "40", "Inst_end_state" : "41"}]},
			{"Name" : "c_slice", "Type" : "Memory", "Direction" : "IO",
				"SubConnect" : [
					{"ID" : "22", "SubInstance" : "grp_lstm_forward_unidir_Pipeline_VITIS_LOOP_145_1_fu_356", "Port" : "c_slice", "Inst_start_state" : "1", "Inst_end_state" : "2"},
					{"ID" : "28", "SubInstance" : "grp_lstm_forward_unidir_Pipeline_Loop2_4LSTM_fu_373", "Port" : "c_slice", "Inst_start_state" : "40", "Inst_end_state" : "41"}]},
			{"Name" : "U_slice", "Type" : "Memory", "Direction" : "I"}],
		"Loop" : [
			{"Name" : "Loop2_2LSTM_Loop3_2_1LSTM", "PipelineType" : "pipeline",
				"LoopDec" : {"FSMBitwidth" : "9", "FirstState" : "ap_ST_fsm_pp0_stage0", "FirstStateIter" : "ap_enable_reg_pp0_iter0", "FirstStateBlock" : "ap_block_pp0_stage0_subdone", "LastState" : "ap_ST_fsm_pp0_stage0", "LastStateIter" : "ap_enable_reg_pp0_iter16", "LastStateBlock" : "ap_block_pp0_stage0_subdone", "PreState" : ["ap_ST_fsm_state4"], "QuitState" : "ap_ST_fsm_pp0_stage0", "QuitStateIter" : "ap_enable_reg_pp0_iter16", "QuitStateBlock" : "ap_block_pp0_stage0_subdone", "PostState" : ["ap_ST_fsm_state22"]}},
			{"Name" : "Loop2_3LSTM_Loop3_3_1LSTM", "PipelineType" : "pipeline",
				"LoopDec" : {"FSMBitwidth" : "9", "FirstState" : "ap_ST_fsm_pp1_stage0", "FirstStateIter" : "ap_enable_reg_pp1_iter0", "FirstStateBlock" : "ap_block_pp1_stage0_subdone", "LastState" : "ap_ST_fsm_pp1_stage0", "LastStateIter" : "ap_enable_reg_pp1_iter16", "LastStateBlock" : "ap_block_pp1_stage0_subdone", "PreState" : ["ap_ST_fsm_state22"], "QuitState" : "ap_ST_fsm_pp1_stage0", "QuitStateIter" : "ap_enable_reg_pp1_iter16", "QuitStateBlock" : "ap_block_pp1_stage0_subdone", "PostState" : ["ap_ST_fsm_state40"]}},
			{"Name" : "Loop1LSTM", "PipelineType" : "no",
				"LoopDec" : {"FSMBitwidth" : "9", "FirstState" : "ap_ST_fsm_state3", "LastState" : ["ap_ST_fsm_state41"], "QuitState" : ["ap_ST_fsm_state3"], "PreState" : ["ap_ST_fsm_state2"], "PostState" : ["ap_ST_fsm_state1"], "OneDepthLoop" : "0", "OneStateBlock": ""}}]},
	{"ID" : "16", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_lstm_forward_unidir_fu_102.c_slice_U", "Parent" : "15"},
	{"ID" : "17", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_lstm_forward_unidir_fu_102.z_U", "Parent" : "15"},
	{"ID" : "18", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_lstm_forward_unidir_fu_102.z_1_U", "Parent" : "15"},
	{"ID" : "19", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_lstm_forward_unidir_fu_102.z_2_U", "Parent" : "15"},
	{"ID" : "20", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_lstm_forward_unidir_fu_102.z_3_U", "Parent" : "15"},
	{"ID" : "21", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_lstm_forward_unidir_fu_102.z_4_U", "Parent" : "15"},
	{"ID" : "22", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_lstm_forward_unidir_fu_102.grp_lstm_forward_unidir_Pipeline_VITIS_LOOP_145_1_fu_356", "Parent" : "15", "Child" : ["23"],
		"CDFG" : "lstm_forward_unidir_Pipeline_VITIS_LOOP_145_1",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "0", "ap_idle" : "1", "real_start" : "0",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "34", "EstimateLatencyMax" : "34",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"IsBlackBox" : "0",
		"Port" : [
			{"Name" : "h_slice", "Type" : "Memory", "Direction" : "O"},
			{"Name" : "c_slice", "Type" : "Memory", "Direction" : "O"}],
		"Loop" : [
			{"Name" : "VITIS_LOOP_145_1", "PipelineType" : "UPC",
				"LoopDec" : {"FSMBitwidth" : "1", "FirstState" : "ap_ST_fsm_state1", "FirstStateIter" : "", "FirstStateBlock" : "ap_ST_fsm_state1_blk", "LastState" : "ap_ST_fsm_state1", "LastStateIter" : "", "LastStateBlock" : "ap_ST_fsm_state1_blk", "QuitState" : "ap_ST_fsm_state1", "QuitStateIter" : "", "QuitStateBlock" : "ap_ST_fsm_state1_blk", "OneDepthLoop" : "1", "has_ap_ctrl" : "1", "has_continue" : "0"}}]},
	{"ID" : "23", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_lstm_forward_unidir_fu_102.grp_lstm_forward_unidir_Pipeline_VITIS_LOOP_145_1_fu_356.flow_control_loop_pipe_sequential_init_U", "Parent" : "22"},
	{"ID" : "24", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_lstm_forward_unidir_fu_102.grp_lstm_forward_unidir_Pipeline_Loop2_1LSTM_fu_364", "Parent" : "15", "Child" : ["25", "26", "27"],
		"CDFG" : "lstm_forward_unidir_Pipeline_Loop2_1LSTM",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "0", "ap_idle" : "1", "real_start" : "0",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "130", "EstimateLatencyMax" : "130",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"IsBlackBox" : "0",
		"Port" : [
			{"Name" : "z_4", "Type" : "Memory", "Direction" : "O"},
			{"Name" : "z_3", "Type" : "Memory", "Direction" : "O"},
			{"Name" : "z_2", "Type" : "Memory", "Direction" : "O"},
			{"Name" : "z_1", "Type" : "Memory", "Direction" : "O"},
			{"Name" : "z", "Type" : "Memory", "Direction" : "O"}],
		"Loop" : [
			{"Name" : "Loop2_1LSTM", "PipelineType" : "UPC",
				"LoopDec" : {"FSMBitwidth" : "1", "FirstState" : "ap_ST_fsm_state1", "FirstStateIter" : "", "FirstStateBlock" : "ap_ST_fsm_state1_blk", "LastState" : "ap_ST_fsm_state1", "LastStateIter" : "", "LastStateBlock" : "ap_ST_fsm_state1_blk", "QuitState" : "ap_ST_fsm_state1", "QuitStateIter" : "", "QuitStateBlock" : "ap_ST_fsm_state1_blk", "OneDepthLoop" : "1", "has_ap_ctrl" : "1", "has_continue" : "0"}}]},
	{"ID" : "25", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_lstm_forward_unidir_fu_102.grp_lstm_forward_unidir_Pipeline_Loop2_1LSTM_fu_364.mul_7ns_9ns_15_1_1_U4", "Parent" : "24"},
	{"ID" : "26", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_lstm_forward_unidir_fu_102.grp_lstm_forward_unidir_Pipeline_Loop2_1LSTM_fu_364.sparsemux_257_7_16_1_1_U5", "Parent" : "24"},
	{"ID" : "27", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_lstm_forward_unidir_fu_102.grp_lstm_forward_unidir_Pipeline_Loop2_1LSTM_fu_364.flow_control_loop_pipe_sequential_init_U", "Parent" : "24"},
	{"ID" : "28", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_lstm_forward_unidir_fu_102.grp_lstm_forward_unidir_Pipeline_Loop2_4LSTM_fu_373", "Parent" : "15", "Child" : ["29", "30", "31", "32", "33", "34", "35", "36", "37", "38", "39", "40", "41", "42", "43", "44", "45", "46", "47", "48", "49", "50", "51", "52", "53", "54", "55", "56", "57", "58", "59", "60"],
		"CDFG" : "lstm_forward_unidir_Pipeline_Loop2_4LSTM",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "0", "ap_idle" : "1", "real_start" : "0",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "75", "EstimateLatencyMax" : "75",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"IsBlackBox" : "0",
		"Port" : [
			{"Name" : "z", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "z_1", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "z_2", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "z_3", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "z_4", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "c_slice", "Type" : "Memory", "Direction" : "IO"},
			{"Name" : "h_slice", "Type" : "Memory", "Direction" : "O"}],
		"Loop" : [
			{"Name" : "Loop2_4LSTM", "PipelineType" : "UPC",
				"LoopDec" : {"FSMBitwidth" : "1", "FirstState" : "ap_ST_fsm_pp0_stage0", "FirstStateIter" : "ap_enable_reg_pp0_iter0", "FirstStateBlock" : "ap_block_pp0_stage0_subdone", "LastState" : "ap_ST_fsm_pp0_stage0", "LastStateIter" : "ap_enable_reg_pp0_iter42", "LastStateBlock" : "ap_block_pp0_stage0_subdone", "QuitState" : "ap_ST_fsm_pp0_stage0", "QuitStateIter" : "ap_enable_reg_pp0_iter42", "QuitStateBlock" : "ap_block_pp0_stage0_subdone", "OneDepthLoop" : "0", "has_ap_ctrl" : "1", "has_continue" : "0"}}]},
	{"ID" : "29", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_lstm_forward_unidir_fu_102.grp_lstm_forward_unidir_Pipeline_Loop2_4LSTM_fu_373.fadd_32ns_32ns_32_4_full_dsp_1_U13", "Parent" : "28"},
	{"ID" : "30", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_lstm_forward_unidir_fu_102.grp_lstm_forward_unidir_Pipeline_Loop2_4LSTM_fu_373.fadd_32ns_32ns_32_4_full_dsp_1_U14", "Parent" : "28"},
	{"ID" : "31", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_lstm_forward_unidir_fu_102.grp_lstm_forward_unidir_Pipeline_Loop2_4LSTM_fu_373.fadd_32ns_32ns_32_4_full_dsp_1_U15", "Parent" : "28"},
	{"ID" : "32", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_lstm_forward_unidir_fu_102.grp_lstm_forward_unidir_Pipeline_Loop2_4LSTM_fu_373.fdiv_32ns_32ns_32_10_no_dsp_1_U16", "Parent" : "28"},
	{"ID" : "33", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_lstm_forward_unidir_fu_102.grp_lstm_forward_unidir_Pipeline_Loop2_4LSTM_fu_373.fdiv_32ns_32ns_32_10_no_dsp_1_U17", "Parent" : "28"},
	{"ID" : "34", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_lstm_forward_unidir_fu_102.grp_lstm_forward_unidir_Pipeline_Loop2_4LSTM_fu_373.fdiv_32ns_32ns_32_10_no_dsp_1_U18", "Parent" : "28"},
	{"ID" : "35", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_lstm_forward_unidir_fu_102.grp_lstm_forward_unidir_Pipeline_Loop2_4LSTM_fu_373.fexp_32ns_32ns_32_8_full_dsp_1_U19", "Parent" : "28"},
	{"ID" : "36", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_lstm_forward_unidir_fu_102.grp_lstm_forward_unidir_Pipeline_Loop2_4LSTM_fu_373.fexp_32ns_32ns_32_8_full_dsp_1_U20", "Parent" : "28"},
	{"ID" : "37", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_lstm_forward_unidir_fu_102.grp_lstm_forward_unidir_Pipeline_Loop2_4LSTM_fu_373.fexp_32ns_32ns_32_8_full_dsp_1_U21", "Parent" : "28"},
	{"ID" : "38", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_lstm_forward_unidir_fu_102.grp_lstm_forward_unidir_Pipeline_Loop2_4LSTM_fu_373.sptohp_32ns_16_2_no_dsp_1_U22", "Parent" : "28"},
	{"ID" : "39", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_lstm_forward_unidir_fu_102.grp_lstm_forward_unidir_Pipeline_Loop2_4LSTM_fu_373.sptohp_32ns_16_2_no_dsp_1_U23", "Parent" : "28"},
	{"ID" : "40", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_lstm_forward_unidir_fu_102.grp_lstm_forward_unidir_Pipeline_Loop2_4LSTM_fu_373.sptohp_32ns_16_2_no_dsp_1_U24", "Parent" : "28"},
	{"ID" : "41", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_lstm_forward_unidir_fu_102.grp_lstm_forward_unidir_Pipeline_Loop2_4LSTM_fu_373.hptosp_16ns_32_2_no_dsp_1_U25", "Parent" : "28"},
	{"ID" : "42", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_lstm_forward_unidir_fu_102.grp_lstm_forward_unidir_Pipeline_Loop2_4LSTM_fu_373.hptosp_16ns_32_2_no_dsp_1_U26", "Parent" : "28"},
	{"ID" : "43", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_lstm_forward_unidir_fu_102.grp_lstm_forward_unidir_Pipeline_Loop2_4LSTM_fu_373.hptosp_16ns_32_2_no_dsp_1_U27", "Parent" : "28"},
	{"ID" : "44", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_lstm_forward_unidir_fu_102.grp_lstm_forward_unidir_Pipeline_Loop2_4LSTM_fu_373.hmul_16ns_16ns_16_4_max_dsp_1_U30", "Parent" : "28"},
	{"ID" : "45", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_lstm_forward_unidir_fu_102.grp_lstm_forward_unidir_Pipeline_Loop2_4LSTM_fu_373.hmul_16ns_16ns_16_4_max_dsp_1_U31", "Parent" : "28"},
	{"ID" : "46", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_lstm_forward_unidir_fu_102.grp_lstm_forward_unidir_Pipeline_Loop2_4LSTM_fu_373.hcmp_16ns_16ns_1_2_no_dsp_1_U32", "Parent" : "28"},
	{"ID" : "47", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_lstm_forward_unidir_fu_102.grp_lstm_forward_unidir_Pipeline_Loop2_4LSTM_fu_373.hcmp_16ns_16ns_1_2_no_dsp_1_U33", "Parent" : "28"},
	{"ID" : "48", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_lstm_forward_unidir_fu_102.grp_lstm_forward_unidir_Pipeline_Loop2_4LSTM_fu_373.hcmp_16ns_16ns_1_2_no_dsp_1_U34", "Parent" : "28"},
	{"ID" : "49", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_lstm_forward_unidir_fu_102.grp_lstm_forward_unidir_Pipeline_Loop2_4LSTM_fu_373.hcmp_16ns_16ns_1_2_no_dsp_1_U35", "Parent" : "28"},
	{"ID" : "50", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_lstm_forward_unidir_fu_102.grp_lstm_forward_unidir_Pipeline_Loop2_4LSTM_fu_373.mul_5ns_7ns_11_1_1_U36", "Parent" : "28"},
	{"ID" : "51", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_lstm_forward_unidir_fu_102.grp_lstm_forward_unidir_Pipeline_Loop2_4LSTM_fu_373.mul_6ns_8ns_13_1_1_U37", "Parent" : "28"},
	{"ID" : "52", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_lstm_forward_unidir_fu_102.grp_lstm_forward_unidir_Pipeline_Loop2_4LSTM_fu_373.mul_7ns_9ns_15_1_1_U38", "Parent" : "28"},
	{"ID" : "53", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_lstm_forward_unidir_fu_102.grp_lstm_forward_unidir_Pipeline_Loop2_4LSTM_fu_373.mul_7ns_9ns_15_1_1_U39", "Parent" : "28"},
	{"ID" : "54", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_lstm_forward_unidir_fu_102.grp_lstm_forward_unidir_Pipeline_Loop2_4LSTM_fu_373.sparsemux_11_3_16_1_1_U40", "Parent" : "28"},
	{"ID" : "55", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_lstm_forward_unidir_fu_102.grp_lstm_forward_unidir_Pipeline_Loop2_4LSTM_fu_373.sparsemux_11_3_16_1_1_U41", "Parent" : "28"},
	{"ID" : "56", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_lstm_forward_unidir_fu_102.grp_lstm_forward_unidir_Pipeline_Loop2_4LSTM_fu_373.sparsemux_11_3_16_1_1_U42", "Parent" : "28"},
	{"ID" : "57", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_lstm_forward_unidir_fu_102.grp_lstm_forward_unidir_Pipeline_Loop2_4LSTM_fu_373.sparsemux_11_3_16_1_1_U43", "Parent" : "28"},
	{"ID" : "58", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_lstm_forward_unidir_fu_102.grp_lstm_forward_unidir_Pipeline_Loop2_4LSTM_fu_373.sparsemux_7_2_16_1_1_U44", "Parent" : "28"},
	{"ID" : "59", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_lstm_forward_unidir_fu_102.grp_lstm_forward_unidir_Pipeline_Loop2_4LSTM_fu_373.sparsemux_7_2_16_1_1_U45", "Parent" : "28"},
	{"ID" : "60", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_lstm_forward_unidir_fu_102.grp_lstm_forward_unidir_Pipeline_Loop2_4LSTM_fu_373.flow_control_loop_pipe_sequential_init_U", "Parent" : "28"},
	{"ID" : "61", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_lstm_forward_unidir_fu_102.hadd_16ns_16ns_16_5_full_dsp_1_U67", "Parent" : "15"},
	{"ID" : "62", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_lstm_forward_unidir_fu_102.hmul_16ns_16ns_16_4_max_dsp_1_U68", "Parent" : "15"},
	{"ID" : "63", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_lstm_forward_unidir_fu_102.urem_8ns_4ns_3_12_1_U69", "Parent" : "15"},
	{"ID" : "64", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_lstm_forward_unidir_fu_102.mul_8ns_10ns_17_1_1_U70", "Parent" : "15"},
	{"ID" : "65", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_lstm_forward_unidir_fu_102.sparsemux_11_3_16_1_1_U71", "Parent" : "15"},
	{"ID" : "66", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_lstm_forward_unidir_fu_102.urem_8ns_4ns_3_12_1_U72", "Parent" : "15"},
	{"ID" : "67", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_lstm_forward_unidir_fu_102.mul_8ns_10ns_17_1_1_U73", "Parent" : "15"},
	{"ID" : "68", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_lstm_forward_unidir_fu_102.sparsemux_11_3_16_1_1_U74", "Parent" : "15"},
	{"ID" : "69", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrolled_Pipeline_Loop_BN_fu_124", "Parent" : "0", "Child" : ["70", "71", "72"],
		"CDFG" : "run_all_slices_unrolled_Pipeline_Loop_BN",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "0", "ap_idle" : "1", "real_start" : "0",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "71", "EstimateLatencyMax" : "71",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"IsBlackBox" : "0",
		"Port" : [
			{"Name" : "h_slice", "Type" : "Memory", "Direction" : "IO"},
			{"Name" : "BN2_var0", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "BN2_gamma0", "Type" : "Memory", "Direction" : "I"}],
		"Loop" : [
			{"Name" : "Loop_BN", "PipelineType" : "UPC",
				"LoopDec" : {"FSMBitwidth" : "1", "FirstState" : "ap_ST_fsm_pp0_stage0", "FirstStateIter" : "ap_enable_reg_pp0_iter0", "FirstStateBlock" : "ap_block_pp0_stage0_subdone", "LastState" : "ap_ST_fsm_pp0_stage0", "LastStateIter" : "ap_enable_reg_pp0_iter38", "LastStateBlock" : "ap_block_pp0_stage0_subdone", "QuitState" : "ap_ST_fsm_pp0_stage0", "QuitStateIter" : "ap_enable_reg_pp0_iter38", "QuitStateBlock" : "ap_block_pp0_stage0_subdone", "OneDepthLoop" : "0", "has_ap_ctrl" : "1", "has_continue" : "0"}}]},
	{"ID" : "70", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrolled_Pipeline_Loop_BN_fu_124.BN2_var0_U", "Parent" : "69"},
	{"ID" : "71", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrolled_Pipeline_Loop_BN_fu_124.BN2_gamma0_U", "Parent" : "69"},
	{"ID" : "72", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrolled_Pipeline_Loop_BN_fu_124.flow_control_loop_pipe_sequential_init_U", "Parent" : "69"},
	{"ID" : "73", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.grp_conv_bn_act_pool_fu_134", "Parent" : "0", "Child" : ["74", "75", "76", "80", "83", "84", "85"],
		"CDFG" : "conv_bn_act_pool",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "0", "ap_idle" : "1", "real_start" : "0",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "2638369", "EstimateLatencyMax" : "2641321",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"IsBlackBox" : "0",
		"Port" : [
			{"Name" : "X_slice12", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "ConvW1", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "U_slice", "Type" : "Memory", "Direction" : "O",
				"SubConnect" : [
					{"ID" : "80", "SubInstance" : "grp_conv_bn_act_pool_Pipeline_Loop3_2Big_fu_717", "Port" : "U_slice", "Inst_start_state" : "5", "Inst_end_state" : "40"}]}],
		"Loop" : [
			{"Name" : "Loop3_1Big_Loop4Big", "PipelineType" : "pipeline",
				"LoopDec" : {"FSMBitwidth" : "33", "FirstState" : "ap_ST_fsm_pp0_stage0", "FirstStateIter" : "ap_enable_reg_pp0_iter0", "FirstStateBlock" : "ap_block_pp0_stage0_subdone", "LastState" : "ap_ST_fsm_pp0_stage6", "LastStateIter" : "ap_enable_reg_pp0_iter1", "LastStateBlock" : "ap_block_pp0_stage6_subdone", "PreState" : ["ap_ST_fsm_state5"], "QuitState" : "ap_ST_fsm_pp0_stage6", "QuitStateIter" : "ap_enable_reg_pp0_iter1", "QuitStateBlock" : "ap_block_pp0_stage6_subdone", "PostState" : ["ap_ST_fsm_state23"]}},
			{"Name" : "Loop2Big", "PipelineType" : "no",
				"LoopDec" : {"FSMBitwidth" : "33", "FirstState" : "ap_ST_fsm_state5", "LastState" : ["ap_ST_fsm_state39"], "QuitState" : ["ap_ST_fsm_state5"], "PreState" : ["ap_ST_fsm_state4"], "PostState" : ["ap_ST_fsm_state40"], "OneDepthLoop" : "0", "OneStateBlock": ""}},
			{"Name" : "Loop1Big", "PipelineType" : "no",
				"LoopDec" : {"FSMBitwidth" : "33", "FirstState" : "ap_ST_fsm_state4", "LastState" : ["ap_ST_fsm_state40"], "QuitState" : ["ap_ST_fsm_state4"], "PreState" : ["ap_ST_fsm_state3"], "PostState" : ["ap_ST_fsm_state1"], "OneDepthLoop" : "0", "OneStateBlock": ""}}]},
	{"ID" : "74", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_bn_act_pool_fu_134.X_slice12_U", "Parent" : "73"},
	{"ID" : "75", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_bn_act_pool_fu_134.ConvW1_U", "Parent" : "73"},
	{"ID" : "76", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_bn_act_pool_fu_134.grp_conv_bn_act_pool_Pipeline_BNParamsLoop_fu_649", "Parent" : "73", "Child" : ["77", "78", "79"],
		"CDFG" : "conv_bn_act_pool_Pipeline_BNParamsLoop",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "0", "ap_idle" : "1", "real_start" : "0",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "70", "EstimateLatencyMax" : "70",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"IsBlackBox" : "0",
		"Port" : [
			{"Name" : "mux_case_31232615_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "mux_case_30230611_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "mux_case_29228607_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "mux_case_28226603_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "mux_case_27224599_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "mux_case_26222595_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "mux_case_25220591_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "mux_case_24218587_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "mux_case_23216583_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "mux_case_22214579_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "mux_case_21212575_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "mux_case_20210571_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "mux_case_19208567_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "mux_case_18206563_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "mux_case_17204559_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "mux_case_16202555_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "mux_case_15200551_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "mux_case_14198547_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "mux_case_13196543_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "mux_case_12194539_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "mux_case_11192535_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "mux_case_10190531_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "mux_case_9188527_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "mux_case_8186523_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "mux_case_7184519_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "mux_case_6182515_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "mux_case_5180511_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "mux_case_4178507_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "mux_case_3176503_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "mux_case_2174499_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "mux_case_1172495_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "mux_case_0170491_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "mux_case_31168487_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "mux_case_30166483_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "mux_case_29164479_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "mux_case_28162475_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "mux_case_27160471_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "mux_case_26158467_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "mux_case_25156463_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "mux_case_24154459_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "mux_case_23152455_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "mux_case_22150451_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "mux_case_21148447_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "mux_case_20146443_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "mux_case_19144439_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "mux_case_18142435_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "mux_case_17140431_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "mux_case_16138427_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "mux_case_15136423_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "mux_case_14134419_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "mux_case_13132415_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "mux_case_12130411_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "mux_case_11128407_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "mux_case_10126403_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "mux_case_9124399_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "mux_case_8122395_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "mux_case_7120391_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "mux_case_6118387_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "mux_case_5116383_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "mux_case_4114379_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "mux_case_3112375_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "mux_case_2110371_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "mux_case_1108367_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "mux_case_0106363_out", "Type" : "Vld", "Direction" : "O"}],
		"Loop" : [
			{"Name" : "BNParamsLoop", "PipelineType" : "UPC",
				"LoopDec" : {"FSMBitwidth" : "1", "FirstState" : "ap_ST_fsm_pp0_stage0", "FirstStateIter" : "ap_enable_reg_pp0_iter0", "FirstStateBlock" : "ap_block_pp0_stage0_subdone", "LastState" : "ap_ST_fsm_pp0_stage0", "LastStateIter" : "ap_enable_reg_pp0_iter37", "LastStateBlock" : "ap_block_pp0_stage0_subdone", "QuitState" : "ap_ST_fsm_pp0_stage0", "QuitStateIter" : "ap_enable_reg_pp0_iter37", "QuitStateBlock" : "ap_block_pp0_stage0_subdone", "OneDepthLoop" : "0", "has_ap_ctrl" : "1", "has_continue" : "0"}}]},
	{"ID" : "77", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_conv_bn_act_pool_fu_134.grp_conv_bn_act_pool_Pipeline_BNParamsLoop_fu_649.sparsemux_65_5_16_1_1_U111", "Parent" : "76"},
	{"ID" : "78", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_conv_bn_act_pool_fu_134.grp_conv_bn_act_pool_Pipeline_BNParamsLoop_fu_649.sparsemux_65_5_16_1_1_U112", "Parent" : "76"},
	{"ID" : "79", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_conv_bn_act_pool_fu_134.grp_conv_bn_act_pool_Pipeline_BNParamsLoop_fu_649.flow_control_loop_pipe_sequential_init_U", "Parent" : "76"},
	{"ID" : "80", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_bn_act_pool_fu_134.grp_conv_bn_act_pool_Pipeline_Loop3_2Big_fu_717", "Parent" : "73", "Child" : ["81", "82"],
		"CDFG" : "conv_bn_act_pool_Pipeline_Loop3_2Big",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "0", "ap_idle" : "1", "real_start" : "0",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "41", "EstimateLatencyMax" : "41",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"IsBlackBox" : "0",
		"Port" : [
			{"Name" : "pool_acc", "Type" : "OVld", "Direction" : "IO"},
			{"Name" : "pool_acc_31", "Type" : "OVld", "Direction" : "IO"},
			{"Name" : "pool_acc_30", "Type" : "OVld", "Direction" : "IO"},
			{"Name" : "pool_acc_29", "Type" : "OVld", "Direction" : "IO"},
			{"Name" : "pool_acc_28", "Type" : "OVld", "Direction" : "IO"},
			{"Name" : "pool_acc_27", "Type" : "OVld", "Direction" : "IO"},
			{"Name" : "pool_acc_26", "Type" : "OVld", "Direction" : "IO"},
			{"Name" : "pool_acc_25", "Type" : "OVld", "Direction" : "IO"},
			{"Name" : "pool_acc_24", "Type" : "OVld", "Direction" : "IO"},
			{"Name" : "pool_acc_23", "Type" : "OVld", "Direction" : "IO"},
			{"Name" : "pool_acc_22", "Type" : "OVld", "Direction" : "IO"},
			{"Name" : "pool_acc_21", "Type" : "OVld", "Direction" : "IO"},
			{"Name" : "pool_acc_20", "Type" : "OVld", "Direction" : "IO"},
			{"Name" : "pool_acc_19", "Type" : "OVld", "Direction" : "IO"},
			{"Name" : "pool_acc_18", "Type" : "OVld", "Direction" : "IO"},
			{"Name" : "pool_acc_17", "Type" : "OVld", "Direction" : "IO"},
			{"Name" : "pool_acc_16", "Type" : "OVld", "Direction" : "IO"},
			{"Name" : "pool_acc_15", "Type" : "OVld", "Direction" : "IO"},
			{"Name" : "pool_acc_14", "Type" : "OVld", "Direction" : "IO"},
			{"Name" : "pool_acc_13", "Type" : "OVld", "Direction" : "IO"},
			{"Name" : "pool_acc_12", "Type" : "OVld", "Direction" : "IO"},
			{"Name" : "pool_acc_11", "Type" : "OVld", "Direction" : "IO"},
			{"Name" : "pool_acc_10", "Type" : "OVld", "Direction" : "IO"},
			{"Name" : "pool_acc_9", "Type" : "OVld", "Direction" : "IO"},
			{"Name" : "pool_acc_8", "Type" : "OVld", "Direction" : "IO"},
			{"Name" : "pool_acc_7", "Type" : "OVld", "Direction" : "IO"},
			{"Name" : "pool_acc_6", "Type" : "OVld", "Direction" : "IO"},
			{"Name" : "pool_acc_5", "Type" : "OVld", "Direction" : "IO"},
			{"Name" : "pool_acc_4", "Type" : "OVld", "Direction" : "IO"},
			{"Name" : "pool_acc_3", "Type" : "OVld", "Direction" : "IO"},
			{"Name" : "pool_acc_2", "Type" : "OVld", "Direction" : "IO"},
			{"Name" : "pool_acc_1", "Type" : "OVld", "Direction" : "IO"},
			{"Name" : "mul4", "Type" : "None", "Direction" : "I"},
			{"Name" : "U_slice", "Type" : "Memory", "Direction" : "O"}],
		"Loop" : [
			{"Name" : "Loop3_2Big", "PipelineType" : "UPC",
				"LoopDec" : {"FSMBitwidth" : "1", "FirstState" : "ap_ST_fsm_pp0_stage0", "FirstStateIter" : "ap_enable_reg_pp0_iter0", "FirstStateBlock" : "ap_block_pp0_stage0_subdone", "LastState" : "ap_ST_fsm_pp0_stage0", "LastStateIter" : "ap_enable_reg_pp0_iter8", "LastStateBlock" : "ap_block_pp0_stage0_subdone", "QuitState" : "ap_ST_fsm_pp0_stage0", "QuitStateIter" : "ap_enable_reg_pp0_iter8", "QuitStateBlock" : "ap_block_pp0_stage0_subdone", "OneDepthLoop" : "0", "has_ap_ctrl" : "1", "has_continue" : "0"}}]},
	{"ID" : "81", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_conv_bn_act_pool_fu_134.grp_conv_bn_act_pool_Pipeline_Loop3_2Big_fu_717.sparsemux_65_5_16_1_1_U179", "Parent" : "80"},
	{"ID" : "82", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_conv_bn_act_pool_fu_134.grp_conv_bn_act_pool_Pipeline_Loop3_2Big_fu_717.flow_control_loop_pipe_sequential_init_U", "Parent" : "80"},
	{"ID" : "83", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_bn_act_pool_fu_134.sparsemux_65_5_16_1_1_U222", "Parent" : "73"},
	{"ID" : "84", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_bn_act_pool_fu_134.sparsemux_65_5_16_1_1_U223", "Parent" : "73"},
	{"ID" : "85", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_bn_act_pool_fu_134.sparsemux_65_5_16_1_1_U224", "Parent" : "73"},
	{"ID" : "86", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrolled_Pipeline_MergedLoop0_fu_144", "Parent" : "0", "Child" : ["87", "88"],
		"CDFG" : "run_all_slices_unrolled_Pipeline_MergedLoop0",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "0", "ap_idle" : "1", "real_start" : "0",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "40", "EstimateLatencyMax" : "40",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"IsBlackBox" : "0",
		"Port" : [
			{"Name" : "merged", "Type" : "Memory", "Direction" : "IO"},
			{"Name" : "h_slice", "Type" : "Memory", "Direction" : "I"}],
		"Loop" : [
			{"Name" : "MergedLoop0", "PipelineType" : "UPC",
				"LoopDec" : {"FSMBitwidth" : "1", "FirstState" : "ap_ST_fsm_pp0_stage0", "FirstStateIter" : "ap_enable_reg_pp0_iter0", "FirstStateBlock" : "ap_block_pp0_stage0_subdone", "LastState" : "ap_ST_fsm_pp0_stage0", "LastStateIter" : "ap_enable_reg_pp0_iter7", "LastStateBlock" : "ap_block_pp0_stage0_subdone", "QuitState" : "ap_ST_fsm_pp0_stage0", "QuitStateIter" : "ap_enable_reg_pp0_iter7", "QuitStateBlock" : "ap_block_pp0_stage0_subdone", "OneDepthLoop" : "0", "has_ap_ctrl" : "1", "has_continue" : "0"}}]},
	{"ID" : "87", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrolled_Pipeline_MergedLoop0_fu_144.sparsemux_7_2_16_1_1_U98", "Parent" : "86"},
	{"ID" : "88", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrolled_Pipeline_MergedLoop0_fu_144.flow_control_loop_pipe_sequential_init_U", "Parent" : "86"},
	{"ID" : "89", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrolled_Pipeline_Loop_BN2_fu_152", "Parent" : "0", "Child" : ["90", "91", "92"],
		"CDFG" : "run_all_slices_unrolled_Pipeline_Loop_BN2",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "0", "ap_idle" : "1", "real_start" : "0",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "71", "EstimateLatencyMax" : "71",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"IsBlackBox" : "0",
		"Port" : [
			{"Name" : "h_slice", "Type" : "Memory", "Direction" : "IO"},
			{"Name" : "BN2_var1", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "BN2_gamma1", "Type" : "Memory", "Direction" : "I"}],
		"Loop" : [
			{"Name" : "Loop_BN", "PipelineType" : "UPC",
				"LoopDec" : {"FSMBitwidth" : "1", "FirstState" : "ap_ST_fsm_pp0_stage0", "FirstStateIter" : "ap_enable_reg_pp0_iter0", "FirstStateBlock" : "ap_block_pp0_stage0_subdone", "LastState" : "ap_ST_fsm_pp0_stage0", "LastStateIter" : "ap_enable_reg_pp0_iter38", "LastStateBlock" : "ap_block_pp0_stage0_subdone", "QuitState" : "ap_ST_fsm_pp0_stage0", "QuitStateIter" : "ap_enable_reg_pp0_iter38", "QuitStateBlock" : "ap_block_pp0_stage0_subdone", "OneDepthLoop" : "0", "has_ap_ctrl" : "1", "has_continue" : "0"}}]},
	{"ID" : "90", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrolled_Pipeline_Loop_BN2_fu_152.BN2_var1_U", "Parent" : "89"},
	{"ID" : "91", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrolled_Pipeline_Loop_BN2_fu_152.BN2_gamma1_U", "Parent" : "89"},
	{"ID" : "92", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrolled_Pipeline_Loop_BN2_fu_152.flow_control_loop_pipe_sequential_init_U", "Parent" : "89"},
	{"ID" : "93", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.grp_conv_bn_act_pool_2_fu_162", "Parent" : "0", "Child" : ["94", "95", "96", "100", "103", "104", "105"],
		"CDFG" : "conv_bn_act_pool_2",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "0", "ap_idle" : "1", "real_start" : "0",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "5276665", "EstimateLatencyMax" : "5282569",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"IsBlackBox" : "0",
		"Port" : [
			{"Name" : "X_slice27", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "ConvW2", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "U_slice", "Type" : "Memory", "Direction" : "O",
				"SubConnect" : [
					{"ID" : "100", "SubInstance" : "grp_conv_bn_act_pool_2_Pipeline_Loop3_2Big_fu_721", "Port" : "U_slice", "Inst_start_state" : "5", "Inst_end_state" : "40"}]}],
		"Loop" : [
			{"Name" : "Loop3_1Big_Loop4Big", "PipelineType" : "pipeline",
				"LoopDec" : {"FSMBitwidth" : "33", "FirstState" : "ap_ST_fsm_pp0_stage0", "FirstStateIter" : "ap_enable_reg_pp0_iter0", "FirstStateBlock" : "ap_block_pp0_stage0_subdone", "LastState" : "ap_ST_fsm_pp0_stage6", "LastStateIter" : "ap_enable_reg_pp0_iter1", "LastStateBlock" : "ap_block_pp0_stage6_subdone", "PreState" : ["ap_ST_fsm_state5"], "QuitState" : "ap_ST_fsm_pp0_stage6", "QuitStateIter" : "ap_enable_reg_pp0_iter1", "QuitStateBlock" : "ap_block_pp0_stage6_subdone", "PostState" : ["ap_ST_fsm_state23"]}},
			{"Name" : "Loop2Big", "PipelineType" : "no",
				"LoopDec" : {"FSMBitwidth" : "33", "FirstState" : "ap_ST_fsm_state5", "LastState" : ["ap_ST_fsm_state39"], "QuitState" : ["ap_ST_fsm_state5"], "PreState" : ["ap_ST_fsm_state4"], "PostState" : ["ap_ST_fsm_state40"], "OneDepthLoop" : "0", "OneStateBlock": ""}},
			{"Name" : "Loop1Big", "PipelineType" : "no",
				"LoopDec" : {"FSMBitwidth" : "33", "FirstState" : "ap_ST_fsm_state4", "LastState" : ["ap_ST_fsm_state40"], "QuitState" : ["ap_ST_fsm_state4"], "PreState" : ["ap_ST_fsm_state3"], "PostState" : ["ap_ST_fsm_state1"], "OneDepthLoop" : "0", "OneStateBlock": ""}}]},
	{"ID" : "94", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_bn_act_pool_2_fu_162.X_slice27_U", "Parent" : "93"},
	{"ID" : "95", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_bn_act_pool_2_fu_162.ConvW2_U", "Parent" : "93"},
	{"ID" : "96", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_bn_act_pool_2_fu_162.grp_conv_bn_act_pool_2_Pipeline_BNParamsLoop_fu_653", "Parent" : "93", "Child" : ["97", "98", "99"],
		"CDFG" : "conv_bn_act_pool_2_Pipeline_BNParamsLoop",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "0", "ap_idle" : "1", "real_start" : "0",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "70", "EstimateLatencyMax" : "70",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"IsBlackBox" : "0",
		"Port" : [
			{"Name" : "mux_case_31232615_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "mux_case_30230611_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "mux_case_29228607_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "mux_case_28226603_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "mux_case_27224599_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "mux_case_26222595_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "mux_case_25220591_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "mux_case_24218587_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "mux_case_23216583_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "mux_case_22214579_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "mux_case_21212575_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "mux_case_20210571_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "mux_case_19208567_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "mux_case_18206563_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "mux_case_17204559_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "mux_case_16202555_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "mux_case_15200551_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "mux_case_14198547_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "mux_case_13196543_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "mux_case_12194539_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "mux_case_11192535_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "mux_case_10190531_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "mux_case_9188527_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "mux_case_8186523_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "mux_case_7184519_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "mux_case_6182515_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "mux_case_5180511_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "mux_case_4178507_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "mux_case_3176503_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "mux_case_2174499_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "mux_case_1172495_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "mux_case_0170491_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "mux_case_31168487_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "mux_case_30166483_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "mux_case_29164479_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "mux_case_28162475_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "mux_case_27160471_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "mux_case_26158467_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "mux_case_25156463_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "mux_case_24154459_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "mux_case_23152455_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "mux_case_22150451_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "mux_case_21148447_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "mux_case_20146443_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "mux_case_19144439_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "mux_case_18142435_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "mux_case_17140431_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "mux_case_16138427_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "mux_case_15136423_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "mux_case_14134419_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "mux_case_13132415_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "mux_case_12130411_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "mux_case_11128407_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "mux_case_10126403_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "mux_case_9124399_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "mux_case_8122395_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "mux_case_7120391_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "mux_case_6118387_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "mux_case_5116383_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "mux_case_4114379_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "mux_case_3112375_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "mux_case_2110371_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "mux_case_1108367_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "mux_case_0106363_out", "Type" : "Vld", "Direction" : "O"}],
		"Loop" : [
			{"Name" : "BNParamsLoop", "PipelineType" : "UPC",
				"LoopDec" : {"FSMBitwidth" : "1", "FirstState" : "ap_ST_fsm_pp0_stage0", "FirstStateIter" : "ap_enable_reg_pp0_iter0", "FirstStateBlock" : "ap_block_pp0_stage0_subdone", "LastState" : "ap_ST_fsm_pp0_stage0", "LastStateIter" : "ap_enable_reg_pp0_iter37", "LastStateBlock" : "ap_block_pp0_stage0_subdone", "QuitState" : "ap_ST_fsm_pp0_stage0", "QuitStateIter" : "ap_enable_reg_pp0_iter37", "QuitStateBlock" : "ap_block_pp0_stage0_subdone", "OneDepthLoop" : "0", "has_ap_ctrl" : "1", "has_continue" : "0"}}]},
	{"ID" : "97", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_conv_bn_act_pool_2_fu_162.grp_conv_bn_act_pool_2_Pipeline_BNParamsLoop_fu_653.sparsemux_65_5_16_1_1_U259", "Parent" : "96"},
	{"ID" : "98", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_conv_bn_act_pool_2_fu_162.grp_conv_bn_act_pool_2_Pipeline_BNParamsLoop_fu_653.sparsemux_65_5_16_1_1_U260", "Parent" : "96"},
	{"ID" : "99", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_conv_bn_act_pool_2_fu_162.grp_conv_bn_act_pool_2_Pipeline_BNParamsLoop_fu_653.flow_control_loop_pipe_sequential_init_U", "Parent" : "96"},
	{"ID" : "100", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_bn_act_pool_2_fu_162.grp_conv_bn_act_pool_2_Pipeline_Loop3_2Big_fu_721", "Parent" : "93", "Child" : ["101", "102"],
		"CDFG" : "conv_bn_act_pool_2_Pipeline_Loop3_2Big",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "0", "ap_idle" : "1", "real_start" : "0",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "41", "EstimateLatencyMax" : "41",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"IsBlackBox" : "0",
		"Port" : [
			{"Name" : "pool_acc", "Type" : "OVld", "Direction" : "IO"},
			{"Name" : "pool_acc_31", "Type" : "OVld", "Direction" : "IO"},
			{"Name" : "pool_acc_30", "Type" : "OVld", "Direction" : "IO"},
			{"Name" : "pool_acc_29", "Type" : "OVld", "Direction" : "IO"},
			{"Name" : "pool_acc_28", "Type" : "OVld", "Direction" : "IO"},
			{"Name" : "pool_acc_27", "Type" : "OVld", "Direction" : "IO"},
			{"Name" : "pool_acc_26", "Type" : "OVld", "Direction" : "IO"},
			{"Name" : "pool_acc_25", "Type" : "OVld", "Direction" : "IO"},
			{"Name" : "pool_acc_24", "Type" : "OVld", "Direction" : "IO"},
			{"Name" : "pool_acc_23", "Type" : "OVld", "Direction" : "IO"},
			{"Name" : "pool_acc_22", "Type" : "OVld", "Direction" : "IO"},
			{"Name" : "pool_acc_21", "Type" : "OVld", "Direction" : "IO"},
			{"Name" : "pool_acc_20", "Type" : "OVld", "Direction" : "IO"},
			{"Name" : "pool_acc_19", "Type" : "OVld", "Direction" : "IO"},
			{"Name" : "pool_acc_18", "Type" : "OVld", "Direction" : "IO"},
			{"Name" : "pool_acc_17", "Type" : "OVld", "Direction" : "IO"},
			{"Name" : "pool_acc_16", "Type" : "OVld", "Direction" : "IO"},
			{"Name" : "pool_acc_15", "Type" : "OVld", "Direction" : "IO"},
			{"Name" : "pool_acc_14", "Type" : "OVld", "Direction" : "IO"},
			{"Name" : "pool_acc_13", "Type" : "OVld", "Direction" : "IO"},
			{"Name" : "pool_acc_12", "Type" : "OVld", "Direction" : "IO"},
			{"Name" : "pool_acc_11", "Type" : "OVld", "Direction" : "IO"},
			{"Name" : "pool_acc_10", "Type" : "OVld", "Direction" : "IO"},
			{"Name" : "pool_acc_9", "Type" : "OVld", "Direction" : "IO"},
			{"Name" : "pool_acc_8", "Type" : "OVld", "Direction" : "IO"},
			{"Name" : "pool_acc_7", "Type" : "OVld", "Direction" : "IO"},
			{"Name" : "pool_acc_6", "Type" : "OVld", "Direction" : "IO"},
			{"Name" : "pool_acc_5", "Type" : "OVld", "Direction" : "IO"},
			{"Name" : "pool_acc_4", "Type" : "OVld", "Direction" : "IO"},
			{"Name" : "pool_acc_3", "Type" : "OVld", "Direction" : "IO"},
			{"Name" : "pool_acc_2", "Type" : "OVld", "Direction" : "IO"},
			{"Name" : "pool_acc_1", "Type" : "OVld", "Direction" : "IO"},
			{"Name" : "mul3", "Type" : "None", "Direction" : "I"},
			{"Name" : "U_slice", "Type" : "Memory", "Direction" : "O"}],
		"Loop" : [
			{"Name" : "Loop3_2Big", "PipelineType" : "UPC",
				"LoopDec" : {"FSMBitwidth" : "1", "FirstState" : "ap_ST_fsm_pp0_stage0", "FirstStateIter" : "ap_enable_reg_pp0_iter0", "FirstStateBlock" : "ap_block_pp0_stage0_subdone", "LastState" : "ap_ST_fsm_pp0_stage0", "LastStateIter" : "ap_enable_reg_pp0_iter8", "LastStateBlock" : "ap_block_pp0_stage0_subdone", "QuitState" : "ap_ST_fsm_pp0_stage0", "QuitStateIter" : "ap_enable_reg_pp0_iter8", "QuitStateBlock" : "ap_block_pp0_stage0_subdone", "OneDepthLoop" : "0", "has_ap_ctrl" : "1", "has_continue" : "0"}}]},
	{"ID" : "101", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_conv_bn_act_pool_2_fu_162.grp_conv_bn_act_pool_2_Pipeline_Loop3_2Big_fu_721.sparsemux_65_5_16_1_1_U326", "Parent" : "100"},
	{"ID" : "102", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_conv_bn_act_pool_2_fu_162.grp_conv_bn_act_pool_2_Pipeline_Loop3_2Big_fu_721.flow_control_loop_pipe_sequential_init_U", "Parent" : "100"},
	{"ID" : "103", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_bn_act_pool_2_fu_162.sparsemux_65_5_16_1_1_U369", "Parent" : "93"},
	{"ID" : "104", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_bn_act_pool_2_fu_162.sparsemux_65_5_16_1_1_U370", "Parent" : "93"},
	{"ID" : "105", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_bn_act_pool_2_fu_162.sparsemux_65_5_16_1_1_U371", "Parent" : "93"},
	{"ID" : "106", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrolled_Pipeline_MergedLoop1_fu_172", "Parent" : "0", "Child" : ["107", "108"],
		"CDFG" : "run_all_slices_unrolled_Pipeline_MergedLoop1",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "0", "ap_idle" : "1", "real_start" : "0",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "40", "EstimateLatencyMax" : "40",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"IsBlackBox" : "0",
		"Port" : [
			{"Name" : "merged", "Type" : "Memory", "Direction" : "IO"},
			{"Name" : "h_slice", "Type" : "Memory", "Direction" : "I"}],
		"Loop" : [
			{"Name" : "MergedLoop1", "PipelineType" : "UPC",
				"LoopDec" : {"FSMBitwidth" : "1", "FirstState" : "ap_ST_fsm_pp0_stage0", "FirstStateIter" : "ap_enable_reg_pp0_iter0", "FirstStateBlock" : "ap_block_pp0_stage0_subdone", "LastState" : "ap_ST_fsm_pp0_stage0", "LastStateIter" : "ap_enable_reg_pp0_iter7", "LastStateBlock" : "ap_block_pp0_stage0_subdone", "QuitState" : "ap_ST_fsm_pp0_stage0", "QuitStateIter" : "ap_enable_reg_pp0_iter7", "QuitStateBlock" : "ap_block_pp0_stage0_subdone", "OneDepthLoop" : "0", "has_ap_ctrl" : "1", "has_continue" : "0"}}]},
	{"ID" : "107", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrolled_Pipeline_MergedLoop1_fu_172.sparsemux_7_2_16_1_1_U246", "Parent" : "106"},
	{"ID" : "108", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrolled_Pipeline_MergedLoop1_fu_172.flow_control_loop_pipe_sequential_init_U", "Parent" : "106"},
	{"ID" : "109", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrolled_Pipeline_Loop_BN3_fu_180", "Parent" : "0", "Child" : ["110", "111", "112"],
		"CDFG" : "run_all_slices_unrolled_Pipeline_Loop_BN3",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "0", "ap_idle" : "1", "real_start" : "0",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "71", "EstimateLatencyMax" : "71",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"IsBlackBox" : "0",
		"Port" : [
			{"Name" : "h_slice", "Type" : "Memory", "Direction" : "IO"},
			{"Name" : "BN2_var2", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "BN2_gamma2", "Type" : "Memory", "Direction" : "I"}],
		"Loop" : [
			{"Name" : "Loop_BN", "PipelineType" : "UPC",
				"LoopDec" : {"FSMBitwidth" : "1", "FirstState" : "ap_ST_fsm_pp0_stage0", "FirstStateIter" : "ap_enable_reg_pp0_iter0", "FirstStateBlock" : "ap_block_pp0_stage0_subdone", "LastState" : "ap_ST_fsm_pp0_stage0", "LastStateIter" : "ap_enable_reg_pp0_iter38", "LastStateBlock" : "ap_block_pp0_stage0_subdone", "QuitState" : "ap_ST_fsm_pp0_stage0", "QuitStateIter" : "ap_enable_reg_pp0_iter38", "QuitStateBlock" : "ap_block_pp0_stage0_subdone", "OneDepthLoop" : "0", "has_ap_ctrl" : "1", "has_continue" : "0"}}]},
	{"ID" : "110", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrolled_Pipeline_Loop_BN3_fu_180.BN2_var2_U", "Parent" : "109"},
	{"ID" : "111", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrolled_Pipeline_Loop_BN3_fu_180.BN2_gamma2_U", "Parent" : "109"},
	{"ID" : "112", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrolled_Pipeline_Loop_BN3_fu_180.flow_control_loop_pipe_sequential_init_U", "Parent" : "109"},
	{"ID" : "113", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.grp_conv_bn_act_pool_3_fu_190", "Parent" : "0", "Child" : ["114", "115", "116", "120", "123", "124", "125"],
		"CDFG" : "conv_bn_act_pool_3",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "0", "ap_idle" : "1", "real_start" : "0",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "10553257", "EstimateLatencyMax" : "10565065",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"IsBlackBox" : "0",
		"Port" : [
			{"Name" : "X_slice42", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "ConvW3", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "U_slice", "Type" : "Memory", "Direction" : "O",
				"SubConnect" : [
					{"ID" : "120", "SubInstance" : "grp_conv_bn_act_pool_3_Pipeline_Loop3_2Big_fu_721", "Port" : "U_slice", "Inst_start_state" : "5", "Inst_end_state" : "40"}]}],
		"Loop" : [
			{"Name" : "Loop3_1Big_Loop4Big", "PipelineType" : "pipeline",
				"LoopDec" : {"FSMBitwidth" : "33", "FirstState" : "ap_ST_fsm_pp0_stage0", "FirstStateIter" : "ap_enable_reg_pp0_iter0", "FirstStateBlock" : "ap_block_pp0_stage0_subdone", "LastState" : "ap_ST_fsm_pp0_stage6", "LastStateIter" : "ap_enable_reg_pp0_iter1", "LastStateBlock" : "ap_block_pp0_stage6_subdone", "PreState" : ["ap_ST_fsm_state5"], "QuitState" : "ap_ST_fsm_pp0_stage6", "QuitStateIter" : "ap_enable_reg_pp0_iter1", "QuitStateBlock" : "ap_block_pp0_stage6_subdone", "PostState" : ["ap_ST_fsm_state23"]}},
			{"Name" : "Loop2Big", "PipelineType" : "no",
				"LoopDec" : {"FSMBitwidth" : "33", "FirstState" : "ap_ST_fsm_state5", "LastState" : ["ap_ST_fsm_state39"], "QuitState" : ["ap_ST_fsm_state5"], "PreState" : ["ap_ST_fsm_state4"], "PostState" : ["ap_ST_fsm_state40"], "OneDepthLoop" : "0", "OneStateBlock": ""}},
			{"Name" : "Loop1Big", "PipelineType" : "no",
				"LoopDec" : {"FSMBitwidth" : "33", "FirstState" : "ap_ST_fsm_state4", "LastState" : ["ap_ST_fsm_state40"], "QuitState" : ["ap_ST_fsm_state4"], "PreState" : ["ap_ST_fsm_state3"], "PostState" : ["ap_ST_fsm_state1"], "OneDepthLoop" : "0", "OneStateBlock": ""}}]},
	{"ID" : "114", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_bn_act_pool_3_fu_190.X_slice42_U", "Parent" : "113"},
	{"ID" : "115", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_bn_act_pool_3_fu_190.ConvW3_U", "Parent" : "113"},
	{"ID" : "116", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_bn_act_pool_3_fu_190.grp_conv_bn_act_pool_3_Pipeline_BNParamsLoop_fu_653", "Parent" : "113", "Child" : ["117", "118", "119"],
		"CDFG" : "conv_bn_act_pool_3_Pipeline_BNParamsLoop",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "0", "ap_idle" : "1", "real_start" : "0",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "70", "EstimateLatencyMax" : "70",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"IsBlackBox" : "0",
		"Port" : [
			{"Name" : "mux_case_31232615_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "mux_case_30230611_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "mux_case_29228607_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "mux_case_28226603_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "mux_case_27224599_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "mux_case_26222595_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "mux_case_25220591_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "mux_case_24218587_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "mux_case_23216583_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "mux_case_22214579_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "mux_case_21212575_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "mux_case_20210571_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "mux_case_19208567_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "mux_case_18206563_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "mux_case_17204559_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "mux_case_16202555_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "mux_case_15200551_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "mux_case_14198547_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "mux_case_13196543_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "mux_case_12194539_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "mux_case_11192535_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "mux_case_10190531_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "mux_case_9188527_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "mux_case_8186523_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "mux_case_7184519_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "mux_case_6182515_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "mux_case_5180511_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "mux_case_4178507_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "mux_case_3176503_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "mux_case_2174499_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "mux_case_1172495_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "mux_case_0170491_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "mux_case_31168487_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "mux_case_30166483_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "mux_case_29164479_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "mux_case_28162475_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "mux_case_27160471_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "mux_case_26158467_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "mux_case_25156463_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "mux_case_24154459_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "mux_case_23152455_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "mux_case_22150451_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "mux_case_21148447_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "mux_case_20146443_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "mux_case_19144439_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "mux_case_18142435_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "mux_case_17140431_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "mux_case_16138427_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "mux_case_15136423_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "mux_case_14134419_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "mux_case_13132415_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "mux_case_12130411_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "mux_case_11128407_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "mux_case_10126403_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "mux_case_9124399_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "mux_case_8122395_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "mux_case_7120391_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "mux_case_6118387_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "mux_case_5116383_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "mux_case_4114379_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "mux_case_3112375_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "mux_case_2110371_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "mux_case_1108367_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "mux_case_0106363_out", "Type" : "Vld", "Direction" : "O"}],
		"Loop" : [
			{"Name" : "BNParamsLoop", "PipelineType" : "UPC",
				"LoopDec" : {"FSMBitwidth" : "1", "FirstState" : "ap_ST_fsm_pp0_stage0", "FirstStateIter" : "ap_enable_reg_pp0_iter0", "FirstStateBlock" : "ap_block_pp0_stage0_subdone", "LastState" : "ap_ST_fsm_pp0_stage0", "LastStateIter" : "ap_enable_reg_pp0_iter37", "LastStateBlock" : "ap_block_pp0_stage0_subdone", "QuitState" : "ap_ST_fsm_pp0_stage0", "QuitStateIter" : "ap_enable_reg_pp0_iter37", "QuitStateBlock" : "ap_block_pp0_stage0_subdone", "OneDepthLoop" : "0", "has_ap_ctrl" : "1", "has_continue" : "0"}}]},
	{"ID" : "117", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_conv_bn_act_pool_3_fu_190.grp_conv_bn_act_pool_3_Pipeline_BNParamsLoop_fu_653.sparsemux_65_5_16_1_1_U406", "Parent" : "116"},
	{"ID" : "118", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_conv_bn_act_pool_3_fu_190.grp_conv_bn_act_pool_3_Pipeline_BNParamsLoop_fu_653.sparsemux_65_5_16_1_1_U407", "Parent" : "116"},
	{"ID" : "119", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_conv_bn_act_pool_3_fu_190.grp_conv_bn_act_pool_3_Pipeline_BNParamsLoop_fu_653.flow_control_loop_pipe_sequential_init_U", "Parent" : "116"},
	{"ID" : "120", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_bn_act_pool_3_fu_190.grp_conv_bn_act_pool_3_Pipeline_Loop3_2Big_fu_721", "Parent" : "113", "Child" : ["121", "122"],
		"CDFG" : "conv_bn_act_pool_3_Pipeline_Loop3_2Big",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "0", "ap_idle" : "1", "real_start" : "0",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "41", "EstimateLatencyMax" : "41",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"IsBlackBox" : "0",
		"Port" : [
			{"Name" : "pool_acc", "Type" : "OVld", "Direction" : "IO"},
			{"Name" : "pool_acc_31", "Type" : "OVld", "Direction" : "IO"},
			{"Name" : "pool_acc_30", "Type" : "OVld", "Direction" : "IO"},
			{"Name" : "pool_acc_29", "Type" : "OVld", "Direction" : "IO"},
			{"Name" : "pool_acc_28", "Type" : "OVld", "Direction" : "IO"},
			{"Name" : "pool_acc_27", "Type" : "OVld", "Direction" : "IO"},
			{"Name" : "pool_acc_26", "Type" : "OVld", "Direction" : "IO"},
			{"Name" : "pool_acc_25", "Type" : "OVld", "Direction" : "IO"},
			{"Name" : "pool_acc_24", "Type" : "OVld", "Direction" : "IO"},
			{"Name" : "pool_acc_23", "Type" : "OVld", "Direction" : "IO"},
			{"Name" : "pool_acc_22", "Type" : "OVld", "Direction" : "IO"},
			{"Name" : "pool_acc_21", "Type" : "OVld", "Direction" : "IO"},
			{"Name" : "pool_acc_20", "Type" : "OVld", "Direction" : "IO"},
			{"Name" : "pool_acc_19", "Type" : "OVld", "Direction" : "IO"},
			{"Name" : "pool_acc_18", "Type" : "OVld", "Direction" : "IO"},
			{"Name" : "pool_acc_17", "Type" : "OVld", "Direction" : "IO"},
			{"Name" : "pool_acc_16", "Type" : "OVld", "Direction" : "IO"},
			{"Name" : "pool_acc_15", "Type" : "OVld", "Direction" : "IO"},
			{"Name" : "pool_acc_14", "Type" : "OVld", "Direction" : "IO"},
			{"Name" : "pool_acc_13", "Type" : "OVld", "Direction" : "IO"},
			{"Name" : "pool_acc_12", "Type" : "OVld", "Direction" : "IO"},
			{"Name" : "pool_acc_11", "Type" : "OVld", "Direction" : "IO"},
			{"Name" : "pool_acc_10", "Type" : "OVld", "Direction" : "IO"},
			{"Name" : "pool_acc_9", "Type" : "OVld", "Direction" : "IO"},
			{"Name" : "pool_acc_8", "Type" : "OVld", "Direction" : "IO"},
			{"Name" : "pool_acc_7", "Type" : "OVld", "Direction" : "IO"},
			{"Name" : "pool_acc_6", "Type" : "OVld", "Direction" : "IO"},
			{"Name" : "pool_acc_5", "Type" : "OVld", "Direction" : "IO"},
			{"Name" : "pool_acc_4", "Type" : "OVld", "Direction" : "IO"},
			{"Name" : "pool_acc_3", "Type" : "OVld", "Direction" : "IO"},
			{"Name" : "pool_acc_2", "Type" : "OVld", "Direction" : "IO"},
			{"Name" : "pool_acc_1", "Type" : "OVld", "Direction" : "IO"},
			{"Name" : "mul2", "Type" : "None", "Direction" : "I"},
			{"Name" : "U_slice", "Type" : "Memory", "Direction" : "O"}],
		"Loop" : [
			{"Name" : "Loop3_2Big", "PipelineType" : "UPC",
				"LoopDec" : {"FSMBitwidth" : "1", "FirstState" : "ap_ST_fsm_pp0_stage0", "FirstStateIter" : "ap_enable_reg_pp0_iter0", "FirstStateBlock" : "ap_block_pp0_stage0_subdone", "LastState" : "ap_ST_fsm_pp0_stage0", "LastStateIter" : "ap_enable_reg_pp0_iter8", "LastStateBlock" : "ap_block_pp0_stage0_subdone", "QuitState" : "ap_ST_fsm_pp0_stage0", "QuitStateIter" : "ap_enable_reg_pp0_iter8", "QuitStateBlock" : "ap_block_pp0_stage0_subdone", "OneDepthLoop" : "0", "has_ap_ctrl" : "1", "has_continue" : "0"}}]},
	{"ID" : "121", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_conv_bn_act_pool_3_fu_190.grp_conv_bn_act_pool_3_Pipeline_Loop3_2Big_fu_721.sparsemux_65_5_16_1_1_U473", "Parent" : "120"},
	{"ID" : "122", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_conv_bn_act_pool_3_fu_190.grp_conv_bn_act_pool_3_Pipeline_Loop3_2Big_fu_721.flow_control_loop_pipe_sequential_init_U", "Parent" : "120"},
	{"ID" : "123", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_bn_act_pool_3_fu_190.sparsemux_65_5_16_1_1_U516", "Parent" : "113"},
	{"ID" : "124", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_bn_act_pool_3_fu_190.sparsemux_65_5_16_1_1_U517", "Parent" : "113"},
	{"ID" : "125", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_bn_act_pool_3_fu_190.sparsemux_65_5_16_1_1_U518", "Parent" : "113"},
	{"ID" : "126", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrolled_Pipeline_MergedLoop2_fu_200", "Parent" : "0", "Child" : ["127", "128"],
		"CDFG" : "run_all_slices_unrolled_Pipeline_MergedLoop2",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "0", "ap_idle" : "1", "real_start" : "0",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "40", "EstimateLatencyMax" : "40",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"IsBlackBox" : "0",
		"Port" : [
			{"Name" : "merged", "Type" : "Memory", "Direction" : "IO"},
			{"Name" : "h_slice", "Type" : "Memory", "Direction" : "I"}],
		"Loop" : [
			{"Name" : "MergedLoop2", "PipelineType" : "UPC",
				"LoopDec" : {"FSMBitwidth" : "1", "FirstState" : "ap_ST_fsm_pp0_stage0", "FirstStateIter" : "ap_enable_reg_pp0_iter0", "FirstStateBlock" : "ap_block_pp0_stage0_subdone", "LastState" : "ap_ST_fsm_pp0_stage0", "LastStateIter" : "ap_enable_reg_pp0_iter7", "LastStateBlock" : "ap_block_pp0_stage0_subdone", "QuitState" : "ap_ST_fsm_pp0_stage0", "QuitStateIter" : "ap_enable_reg_pp0_iter7", "QuitStateBlock" : "ap_block_pp0_stage0_subdone", "OneDepthLoop" : "0", "has_ap_ctrl" : "1", "has_continue" : "0"}}]},
	{"ID" : "127", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrolled_Pipeline_MergedLoop2_fu_200.sparsemux_7_2_16_1_1_U393", "Parent" : "126"},
	{"ID" : "128", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrolled_Pipeline_MergedLoop2_fu_200.flow_control_loop_pipe_sequential_init_U", "Parent" : "126"},
	{"ID" : "129", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrolled_Pipeline_Loop_BN4_fu_208", "Parent" : "0", "Child" : ["130", "131", "132"],
		"CDFG" : "run_all_slices_unrolled_Pipeline_Loop_BN4",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "0", "ap_idle" : "1", "real_start" : "0",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "71", "EstimateLatencyMax" : "71",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"IsBlackBox" : "0",
		"Port" : [
			{"Name" : "h_slice", "Type" : "Memory", "Direction" : "IO"},
			{"Name" : "BN2_var3", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "BN2_gamma3", "Type" : "Memory", "Direction" : "I"}],
		"Loop" : [
			{"Name" : "Loop_BN", "PipelineType" : "UPC",
				"LoopDec" : {"FSMBitwidth" : "1", "FirstState" : "ap_ST_fsm_pp0_stage0", "FirstStateIter" : "ap_enable_reg_pp0_iter0", "FirstStateBlock" : "ap_block_pp0_stage0_subdone", "LastState" : "ap_ST_fsm_pp0_stage0", "LastStateIter" : "ap_enable_reg_pp0_iter38", "LastStateBlock" : "ap_block_pp0_stage0_subdone", "QuitState" : "ap_ST_fsm_pp0_stage0", "QuitStateIter" : "ap_enable_reg_pp0_iter38", "QuitStateBlock" : "ap_block_pp0_stage0_subdone", "OneDepthLoop" : "0", "has_ap_ctrl" : "1", "has_continue" : "0"}}]},
	{"ID" : "130", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrolled_Pipeline_Loop_BN4_fu_208.BN2_var3_U", "Parent" : "129"},
	{"ID" : "131", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrolled_Pipeline_Loop_BN4_fu_208.BN2_gamma3_U", "Parent" : "129"},
	{"ID" : "132", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrolled_Pipeline_Loop_BN4_fu_208.flow_control_loop_pipe_sequential_init_U", "Parent" : "129"},
	{"ID" : "133", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.grp_conv_bn_act_pool_4_fu_218", "Parent" : "0", "Child" : ["134", "135", "136", "140", "143", "144", "145"],
		"CDFG" : "conv_bn_act_pool_4",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "0", "ap_idle" : "1", "real_start" : "0",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "21106441", "EstimateLatencyMax" : "21130057",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"IsBlackBox" : "0",
		"Port" : [
			{"Name" : "X_slice57", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "ConvW4", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "U_slice", "Type" : "Memory", "Direction" : "O",
				"SubConnect" : [
					{"ID" : "140", "SubInstance" : "grp_conv_bn_act_pool_4_Pipeline_Loop3_2Big_fu_721", "Port" : "U_slice", "Inst_start_state" : "5", "Inst_end_state" : "40"}]}],
		"Loop" : [
			{"Name" : "Loop3_1Big_Loop4Big", "PipelineType" : "pipeline",
				"LoopDec" : {"FSMBitwidth" : "33", "FirstState" : "ap_ST_fsm_pp0_stage0", "FirstStateIter" : "ap_enable_reg_pp0_iter0", "FirstStateBlock" : "ap_block_pp0_stage0_subdone", "LastState" : "ap_ST_fsm_pp0_stage6", "LastStateIter" : "ap_enable_reg_pp0_iter1", "LastStateBlock" : "ap_block_pp0_stage6_subdone", "PreState" : ["ap_ST_fsm_state5"], "QuitState" : "ap_ST_fsm_pp0_stage6", "QuitStateIter" : "ap_enable_reg_pp0_iter1", "QuitStateBlock" : "ap_block_pp0_stage6_subdone", "PostState" : ["ap_ST_fsm_state23"]}},
			{"Name" : "Loop2Big", "PipelineType" : "no",
				"LoopDec" : {"FSMBitwidth" : "33", "FirstState" : "ap_ST_fsm_state5", "LastState" : ["ap_ST_fsm_state39"], "QuitState" : ["ap_ST_fsm_state5"], "PreState" : ["ap_ST_fsm_state4"], "PostState" : ["ap_ST_fsm_state40"], "OneDepthLoop" : "0", "OneStateBlock": ""}},
			{"Name" : "Loop1Big", "PipelineType" : "no",
				"LoopDec" : {"FSMBitwidth" : "33", "FirstState" : "ap_ST_fsm_state4", "LastState" : ["ap_ST_fsm_state40"], "QuitState" : ["ap_ST_fsm_state4"], "PreState" : ["ap_ST_fsm_state3"], "PostState" : ["ap_ST_fsm_state1"], "OneDepthLoop" : "0", "OneStateBlock": ""}}]},
	{"ID" : "134", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_bn_act_pool_4_fu_218.X_slice57_U", "Parent" : "133"},
	{"ID" : "135", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_bn_act_pool_4_fu_218.ConvW4_U", "Parent" : "133"},
	{"ID" : "136", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_bn_act_pool_4_fu_218.grp_conv_bn_act_pool_4_Pipeline_BNParamsLoop_fu_653", "Parent" : "133", "Child" : ["137", "138", "139"],
		"CDFG" : "conv_bn_act_pool_4_Pipeline_BNParamsLoop",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "0", "ap_idle" : "1", "real_start" : "0",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "70", "EstimateLatencyMax" : "70",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"IsBlackBox" : "0",
		"Port" : [
			{"Name" : "mux_case_31232615_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "mux_case_30230611_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "mux_case_29228607_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "mux_case_28226603_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "mux_case_27224599_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "mux_case_26222595_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "mux_case_25220591_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "mux_case_24218587_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "mux_case_23216583_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "mux_case_22214579_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "mux_case_21212575_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "mux_case_20210571_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "mux_case_19208567_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "mux_case_18206563_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "mux_case_17204559_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "mux_case_16202555_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "mux_case_15200551_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "mux_case_14198547_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "mux_case_13196543_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "mux_case_12194539_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "mux_case_11192535_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "mux_case_10190531_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "mux_case_9188527_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "mux_case_8186523_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "mux_case_7184519_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "mux_case_6182515_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "mux_case_5180511_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "mux_case_4178507_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "mux_case_3176503_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "mux_case_2174499_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "mux_case_1172495_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "mux_case_0170491_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "mux_case_31168487_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "mux_case_30166483_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "mux_case_29164479_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "mux_case_28162475_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "mux_case_27160471_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "mux_case_26158467_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "mux_case_25156463_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "mux_case_24154459_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "mux_case_23152455_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "mux_case_22150451_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "mux_case_21148447_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "mux_case_20146443_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "mux_case_19144439_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "mux_case_18142435_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "mux_case_17140431_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "mux_case_16138427_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "mux_case_15136423_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "mux_case_14134419_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "mux_case_13132415_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "mux_case_12130411_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "mux_case_11128407_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "mux_case_10126403_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "mux_case_9124399_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "mux_case_8122395_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "mux_case_7120391_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "mux_case_6118387_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "mux_case_5116383_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "mux_case_4114379_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "mux_case_3112375_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "mux_case_2110371_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "mux_case_1108367_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "mux_case_0106363_out", "Type" : "Vld", "Direction" : "O"}],
		"Loop" : [
			{"Name" : "BNParamsLoop", "PipelineType" : "UPC",
				"LoopDec" : {"FSMBitwidth" : "1", "FirstState" : "ap_ST_fsm_pp0_stage0", "FirstStateIter" : "ap_enable_reg_pp0_iter0", "FirstStateBlock" : "ap_block_pp0_stage0_subdone", "LastState" : "ap_ST_fsm_pp0_stage0", "LastStateIter" : "ap_enable_reg_pp0_iter37", "LastStateBlock" : "ap_block_pp0_stage0_subdone", "QuitState" : "ap_ST_fsm_pp0_stage0", "QuitStateIter" : "ap_enable_reg_pp0_iter37", "QuitStateBlock" : "ap_block_pp0_stage0_subdone", "OneDepthLoop" : "0", "has_ap_ctrl" : "1", "has_continue" : "0"}}]},
	{"ID" : "137", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_conv_bn_act_pool_4_fu_218.grp_conv_bn_act_pool_4_Pipeline_BNParamsLoop_fu_653.sparsemux_65_5_16_1_1_U553", "Parent" : "136"},
	{"ID" : "138", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_conv_bn_act_pool_4_fu_218.grp_conv_bn_act_pool_4_Pipeline_BNParamsLoop_fu_653.sparsemux_65_5_16_1_1_U554", "Parent" : "136"},
	{"ID" : "139", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_conv_bn_act_pool_4_fu_218.grp_conv_bn_act_pool_4_Pipeline_BNParamsLoop_fu_653.flow_control_loop_pipe_sequential_init_U", "Parent" : "136"},
	{"ID" : "140", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_bn_act_pool_4_fu_218.grp_conv_bn_act_pool_4_Pipeline_Loop3_2Big_fu_721", "Parent" : "133", "Child" : ["141", "142"],
		"CDFG" : "conv_bn_act_pool_4_Pipeline_Loop3_2Big",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "0", "ap_idle" : "1", "real_start" : "0",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "41", "EstimateLatencyMax" : "41",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"IsBlackBox" : "0",
		"Port" : [
			{"Name" : "pool_acc", "Type" : "OVld", "Direction" : "IO"},
			{"Name" : "pool_acc_31", "Type" : "OVld", "Direction" : "IO"},
			{"Name" : "pool_acc_30", "Type" : "OVld", "Direction" : "IO"},
			{"Name" : "pool_acc_29", "Type" : "OVld", "Direction" : "IO"},
			{"Name" : "pool_acc_28", "Type" : "OVld", "Direction" : "IO"},
			{"Name" : "pool_acc_27", "Type" : "OVld", "Direction" : "IO"},
			{"Name" : "pool_acc_26", "Type" : "OVld", "Direction" : "IO"},
			{"Name" : "pool_acc_25", "Type" : "OVld", "Direction" : "IO"},
			{"Name" : "pool_acc_24", "Type" : "OVld", "Direction" : "IO"},
			{"Name" : "pool_acc_23", "Type" : "OVld", "Direction" : "IO"},
			{"Name" : "pool_acc_22", "Type" : "OVld", "Direction" : "IO"},
			{"Name" : "pool_acc_21", "Type" : "OVld", "Direction" : "IO"},
			{"Name" : "pool_acc_20", "Type" : "OVld", "Direction" : "IO"},
			{"Name" : "pool_acc_19", "Type" : "OVld", "Direction" : "IO"},
			{"Name" : "pool_acc_18", "Type" : "OVld", "Direction" : "IO"},
			{"Name" : "pool_acc_17", "Type" : "OVld", "Direction" : "IO"},
			{"Name" : "pool_acc_16", "Type" : "OVld", "Direction" : "IO"},
			{"Name" : "pool_acc_15", "Type" : "OVld", "Direction" : "IO"},
			{"Name" : "pool_acc_14", "Type" : "OVld", "Direction" : "IO"},
			{"Name" : "pool_acc_13", "Type" : "OVld", "Direction" : "IO"},
			{"Name" : "pool_acc_12", "Type" : "OVld", "Direction" : "IO"},
			{"Name" : "pool_acc_11", "Type" : "OVld", "Direction" : "IO"},
			{"Name" : "pool_acc_10", "Type" : "OVld", "Direction" : "IO"},
			{"Name" : "pool_acc_9", "Type" : "OVld", "Direction" : "IO"},
			{"Name" : "pool_acc_8", "Type" : "OVld", "Direction" : "IO"},
			{"Name" : "pool_acc_7", "Type" : "OVld", "Direction" : "IO"},
			{"Name" : "pool_acc_6", "Type" : "OVld", "Direction" : "IO"},
			{"Name" : "pool_acc_5", "Type" : "OVld", "Direction" : "IO"},
			{"Name" : "pool_acc_4", "Type" : "OVld", "Direction" : "IO"},
			{"Name" : "pool_acc_3", "Type" : "OVld", "Direction" : "IO"},
			{"Name" : "pool_acc_2", "Type" : "OVld", "Direction" : "IO"},
			{"Name" : "pool_acc_1", "Type" : "OVld", "Direction" : "IO"},
			{"Name" : "mul1", "Type" : "None", "Direction" : "I"},
			{"Name" : "U_slice", "Type" : "Memory", "Direction" : "O"}],
		"Loop" : [
			{"Name" : "Loop3_2Big", "PipelineType" : "UPC",
				"LoopDec" : {"FSMBitwidth" : "1", "FirstState" : "ap_ST_fsm_pp0_stage0", "FirstStateIter" : "ap_enable_reg_pp0_iter0", "FirstStateBlock" : "ap_block_pp0_stage0_subdone", "LastState" : "ap_ST_fsm_pp0_stage0", "LastStateIter" : "ap_enable_reg_pp0_iter8", "LastStateBlock" : "ap_block_pp0_stage0_subdone", "QuitState" : "ap_ST_fsm_pp0_stage0", "QuitStateIter" : "ap_enable_reg_pp0_iter8", "QuitStateBlock" : "ap_block_pp0_stage0_subdone", "OneDepthLoop" : "0", "has_ap_ctrl" : "1", "has_continue" : "0"}}]},
	{"ID" : "141", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_conv_bn_act_pool_4_fu_218.grp_conv_bn_act_pool_4_Pipeline_Loop3_2Big_fu_721.sparsemux_65_5_16_1_1_U620", "Parent" : "140"},
	{"ID" : "142", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_conv_bn_act_pool_4_fu_218.grp_conv_bn_act_pool_4_Pipeline_Loop3_2Big_fu_721.flow_control_loop_pipe_sequential_init_U", "Parent" : "140"},
	{"ID" : "143", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_bn_act_pool_4_fu_218.sparsemux_65_5_16_1_1_U663", "Parent" : "133"},
	{"ID" : "144", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_bn_act_pool_4_fu_218.sparsemux_65_5_16_1_1_U664", "Parent" : "133"},
	{"ID" : "145", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_bn_act_pool_4_fu_218.sparsemux_65_5_16_1_1_U665", "Parent" : "133"},
	{"ID" : "146", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrolled_Pipeline_MergedLoop3_fu_228", "Parent" : "0", "Child" : ["147", "148"],
		"CDFG" : "run_all_slices_unrolled_Pipeline_MergedLoop3",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "0", "ap_idle" : "1", "real_start" : "0",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "40", "EstimateLatencyMax" : "40",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"IsBlackBox" : "0",
		"Port" : [
			{"Name" : "merged", "Type" : "Memory", "Direction" : "IO"},
			{"Name" : "h_slice", "Type" : "Memory", "Direction" : "I"}],
		"Loop" : [
			{"Name" : "MergedLoop3", "PipelineType" : "UPC",
				"LoopDec" : {"FSMBitwidth" : "1", "FirstState" : "ap_ST_fsm_pp0_stage0", "FirstStateIter" : "ap_enable_reg_pp0_iter0", "FirstStateBlock" : "ap_block_pp0_stage0_subdone", "LastState" : "ap_ST_fsm_pp0_stage0", "LastStateIter" : "ap_enable_reg_pp0_iter7", "LastStateBlock" : "ap_block_pp0_stage0_subdone", "QuitState" : "ap_ST_fsm_pp0_stage0", "QuitStateIter" : "ap_enable_reg_pp0_iter7", "QuitStateBlock" : "ap_block_pp0_stage0_subdone", "OneDepthLoop" : "0", "has_ap_ctrl" : "1", "has_continue" : "0"}}]},
	{"ID" : "147", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrolled_Pipeline_MergedLoop3_fu_228.sparsemux_7_2_16_1_1_U540", "Parent" : "146"},
	{"ID" : "148", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrolled_Pipeline_MergedLoop3_fu_228.flow_control_loop_pipe_sequential_init_U", "Parent" : "146"},
	{"ID" : "149", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrolled_Pipeline_Loop_BN5_fu_236", "Parent" : "0", "Child" : ["150", "151", "152"],
		"CDFG" : "run_all_slices_unrolled_Pipeline_Loop_BN5",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "0", "ap_idle" : "1", "real_start" : "0",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "71", "EstimateLatencyMax" : "71",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"IsBlackBox" : "0",
		"Port" : [
			{"Name" : "h_slice", "Type" : "Memory", "Direction" : "IO"},
			{"Name" : "BN2_var4", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "BN2_gamma4", "Type" : "Memory", "Direction" : "I"}],
		"Loop" : [
			{"Name" : "Loop_BN", "PipelineType" : "UPC",
				"LoopDec" : {"FSMBitwidth" : "1", "FirstState" : "ap_ST_fsm_pp0_stage0", "FirstStateIter" : "ap_enable_reg_pp0_iter0", "FirstStateBlock" : "ap_block_pp0_stage0_subdone", "LastState" : "ap_ST_fsm_pp0_stage0", "LastStateIter" : "ap_enable_reg_pp0_iter38", "LastStateBlock" : "ap_block_pp0_stage0_subdone", "QuitState" : "ap_ST_fsm_pp0_stage0", "QuitStateIter" : "ap_enable_reg_pp0_iter38", "QuitStateBlock" : "ap_block_pp0_stage0_subdone", "OneDepthLoop" : "0", "has_ap_ctrl" : "1", "has_continue" : "0"}}]},
	{"ID" : "150", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrolled_Pipeline_Loop_BN5_fu_236.BN2_var4_U", "Parent" : "149"},
	{"ID" : "151", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrolled_Pipeline_Loop_BN5_fu_236.BN2_gamma4_U", "Parent" : "149"},
	{"ID" : "152", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrolled_Pipeline_Loop_BN5_fu_236.flow_control_loop_pipe_sequential_init_U", "Parent" : "149"},
	{"ID" : "153", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrolled_Pipeline_MergedLoop4_fu_246", "Parent" : "0", "Child" : ["154", "155"],
		"CDFG" : "run_all_slices_unrolled_Pipeline_MergedLoop4",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "0", "ap_idle" : "1", "real_start" : "0",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "40", "EstimateLatencyMax" : "40",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"IsBlackBox" : "0",
		"Port" : [
			{"Name" : "merged", "Type" : "Memory", "Direction" : "IO"},
			{"Name" : "h_slice", "Type" : "Memory", "Direction" : "I"}],
		"Loop" : [
			{"Name" : "MergedLoop4", "PipelineType" : "UPC",
				"LoopDec" : {"FSMBitwidth" : "1", "FirstState" : "ap_ST_fsm_pp0_stage0", "FirstStateIter" : "ap_enable_reg_pp0_iter0", "FirstStateBlock" : "ap_block_pp0_stage0_subdone", "LastState" : "ap_ST_fsm_pp0_stage0", "LastStateIter" : "ap_enable_reg_pp0_iter7", "LastStateBlock" : "ap_block_pp0_stage0_subdone", "QuitState" : "ap_ST_fsm_pp0_stage0", "QuitStateIter" : "ap_enable_reg_pp0_iter7", "QuitStateBlock" : "ap_block_pp0_stage0_subdone", "OneDepthLoop" : "0", "has_ap_ctrl" : "1", "has_continue" : "0"}}]},
	{"ID" : "154", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrolled_Pipeline_MergedLoop4_fu_246.sparsemux_7_2_16_1_1_U687", "Parent" : "153"},
	{"ID" : "155", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_run_all_slices_unrolled_Pipeline_MergedLoop4_fu_246.flow_control_loop_pipe_sequential_init_U", "Parent" : "153"},
	{"ID" : "156", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.hadd_16ns_16ns_16_5_full_dsp_1_U691", "Parent" : "0"},
	{"ID" : "157", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.hmul_16ns_16ns_16_4_max_dsp_1_U692", "Parent" : "0"},
	{"ID" : "158", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fdiv_32ns_32ns_32_10_no_dsp_1_U693", "Parent" : "0"},
	{"ID" : "159", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fsqrt_32ns_32ns_32_10_no_dsp_1_U694", "Parent" : "0"},
	{"ID" : "160", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.sptohp_32ns_16_2_no_dsp_1_U695", "Parent" : "0"},
	{"ID" : "161", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.hptosp_16ns_32_2_no_dsp_1_U696", "Parent" : "0"},
	{"ID" : "162", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.hptosp_16ns_32_2_no_dsp_1_U697", "Parent" : "0"},
	{"ID" : "163", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.hcmp_16ns_16ns_1_2_no_dsp_1_U700", "Parent" : "0"},
	{"ID" : "164", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fsub_32ns_32ns_32_4_full_dsp_1_U701", "Parent" : "0"},
	{"ID" : "165", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fmul_32ns_32ns_32_3_max_dsp_1_U702", "Parent" : "0"},
	{"ID" : "166", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.sptohp_32ns_16_2_no_dsp_1_U705", "Parent" : "0"},
	{"ID" : "167", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.hdiv_16ns_16ns_16_6_no_dsp_1_U706", "Parent" : "0"},
	{"ID" : "168", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fadd_32ns_32ns_32_4_full_dsp_1_U707", "Parent" : "0"},
	{"ID" : "169", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fmul_32ns_32ns_32_3_max_dsp_1_U708", "Parent" : "0"}]}


set ArgLastReadFirstWriteLatency {
	run_all_slices_unrolled {
		merged {Type IO LastRead 2 FirstWrite 0}
		h_slice {Type IO LastRead -1 FirstWrite -1}
		c_slice {Type IO LastRead -1 FirstWrite -1}
		U_slice {Type IO LastRead -1 FirstWrite -1}
		LSTM_W_ifog0 {Type I LastRead -1 FirstWrite -1}
		LSTM_R_ifog0 {Type I LastRead -1 FirstWrite -1}
		BN2_var0 {Type I LastRead -1 FirstWrite -1}
		BN2_gamma0 {Type I LastRead -1 FirstWrite -1}
		X_slice12 {Type I LastRead -1 FirstWrite -1}
		ConvW1 {Type I LastRead -1 FirstWrite -1}
		LSTM_W_ifog1 {Type I LastRead -1 FirstWrite -1}
		LSTM_R_ifog1 {Type I LastRead -1 FirstWrite -1}
		BN2_var1 {Type I LastRead -1 FirstWrite -1}
		BN2_gamma1 {Type I LastRead -1 FirstWrite -1}
		X_slice27 {Type I LastRead -1 FirstWrite -1}
		ConvW2 {Type I LastRead -1 FirstWrite -1}
		LSTM_W_ifog2 {Type I LastRead -1 FirstWrite -1}
		LSTM_R_ifog2 {Type I LastRead -1 FirstWrite -1}
		BN2_var2 {Type I LastRead -1 FirstWrite -1}
		BN2_gamma2 {Type I LastRead -1 FirstWrite -1}
		X_slice42 {Type I LastRead -1 FirstWrite -1}
		ConvW3 {Type I LastRead -1 FirstWrite -1}
		LSTM_W_ifog3 {Type I LastRead -1 FirstWrite -1}
		LSTM_R_ifog3 {Type I LastRead -1 FirstWrite -1}
		BN2_var3 {Type I LastRead -1 FirstWrite -1}
		BN2_gamma3 {Type I LastRead -1 FirstWrite -1}
		X_slice57 {Type I LastRead -1 FirstWrite -1}
		ConvW4 {Type I LastRead -1 FirstWrite -1}
		LSTM_W_ifog4 {Type I LastRead -1 FirstWrite -1}
		LSTM_R_ifog4 {Type I LastRead -1 FirstWrite -1}
		BN2_var4 {Type I LastRead -1 FirstWrite -1}
		BN2_gamma4 {Type I LastRead -1 FirstWrite -1}}
	run_all_slices_unrolled_Pipeline_MergedLoop {
		merged {Type O LastRead -1 FirstWrite 0}}
	lstm_forward_unidir {
		W_ifog {Type I LastRead 9 FirstWrite -1}
		R_ifog {Type I LastRead 11 FirstWrite -1}
		h_slice {Type IO LastRead 13 FirstWrite 0}
		c_slice {Type IO LastRead -1 FirstWrite -1}
		U_slice {Type I LastRead 9 FirstWrite -1}}
	lstm_forward_unidir_Pipeline_VITIS_LOOP_145_1 {
		h_slice {Type O LastRead -1 FirstWrite 0}
		c_slice {Type O LastRead -1 FirstWrite 0}}
	lstm_forward_unidir_Pipeline_Loop2_1LSTM {
		z_4 {Type O LastRead -1 FirstWrite 0}
		z_3 {Type O LastRead -1 FirstWrite 0}
		z_2 {Type O LastRead -1 FirstWrite 0}
		z_1 {Type O LastRead -1 FirstWrite 0}
		z {Type O LastRead -1 FirstWrite 0}}
	lstm_forward_unidir_Pipeline_Loop2_4LSTM {
		z {Type I LastRead 2 FirstWrite -1}
		z_1 {Type I LastRead 2 FirstWrite -1}
		z_2 {Type I LastRead 2 FirstWrite -1}
		z_3 {Type I LastRead 2 FirstWrite -1}
		z_4 {Type I LastRead 2 FirstWrite -1}
		c_slice {Type IO LastRead 0 FirstWrite 36}
		h_slice {Type O LastRead -1 FirstWrite 42}}
	run_all_slices_unrolled_Pipeline_Loop_BN {
		h_slice {Type IO LastRead 15 FirstWrite 38}
		BN2_var0 {Type I LastRead -1 FirstWrite -1}
		BN2_gamma0 {Type I LastRead -1 FirstWrite -1}}
	conv_bn_act_pool {
		X_slice12 {Type I LastRead -1 FirstWrite -1}
		ConvW1 {Type I LastRead -1 FirstWrite -1}
		U_slice {Type O LastRead -1 FirstWrite 7}}
	conv_bn_act_pool_Pipeline_BNParamsLoop {
		mux_case_31232615_out {Type O LastRead -1 FirstWrite 36}
		mux_case_30230611_out {Type O LastRead -1 FirstWrite 36}
		mux_case_29228607_out {Type O LastRead -1 FirstWrite 36}
		mux_case_28226603_out {Type O LastRead -1 FirstWrite 36}
		mux_case_27224599_out {Type O LastRead -1 FirstWrite 36}
		mux_case_26222595_out {Type O LastRead -1 FirstWrite 36}
		mux_case_25220591_out {Type O LastRead -1 FirstWrite 36}
		mux_case_24218587_out {Type O LastRead -1 FirstWrite 36}
		mux_case_23216583_out {Type O LastRead -1 FirstWrite 36}
		mux_case_22214579_out {Type O LastRead -1 FirstWrite 36}
		mux_case_21212575_out {Type O LastRead -1 FirstWrite 36}
		mux_case_20210571_out {Type O LastRead -1 FirstWrite 36}
		mux_case_19208567_out {Type O LastRead -1 FirstWrite 36}
		mux_case_18206563_out {Type O LastRead -1 FirstWrite 36}
		mux_case_17204559_out {Type O LastRead -1 FirstWrite 36}
		mux_case_16202555_out {Type O LastRead -1 FirstWrite 36}
		mux_case_15200551_out {Type O LastRead -1 FirstWrite 36}
		mux_case_14198547_out {Type O LastRead -1 FirstWrite 36}
		mux_case_13196543_out {Type O LastRead -1 FirstWrite 36}
		mux_case_12194539_out {Type O LastRead -1 FirstWrite 36}
		mux_case_11192535_out {Type O LastRead -1 FirstWrite 36}
		mux_case_10190531_out {Type O LastRead -1 FirstWrite 36}
		mux_case_9188527_out {Type O LastRead -1 FirstWrite 36}
		mux_case_8186523_out {Type O LastRead -1 FirstWrite 36}
		mux_case_7184519_out {Type O LastRead -1 FirstWrite 36}
		mux_case_6182515_out {Type O LastRead -1 FirstWrite 36}
		mux_case_5180511_out {Type O LastRead -1 FirstWrite 36}
		mux_case_4178507_out {Type O LastRead -1 FirstWrite 36}
		mux_case_3176503_out {Type O LastRead -1 FirstWrite 36}
		mux_case_2174499_out {Type O LastRead -1 FirstWrite 36}
		mux_case_1172495_out {Type O LastRead -1 FirstWrite 36}
		mux_case_0170491_out {Type O LastRead -1 FirstWrite 36}
		mux_case_31168487_out {Type O LastRead -1 FirstWrite 36}
		mux_case_30166483_out {Type O LastRead -1 FirstWrite 36}
		mux_case_29164479_out {Type O LastRead -1 FirstWrite 36}
		mux_case_28162475_out {Type O LastRead -1 FirstWrite 36}
		mux_case_27160471_out {Type O LastRead -1 FirstWrite 36}
		mux_case_26158467_out {Type O LastRead -1 FirstWrite 36}
		mux_case_25156463_out {Type O LastRead -1 FirstWrite 36}
		mux_case_24154459_out {Type O LastRead -1 FirstWrite 36}
		mux_case_23152455_out {Type O LastRead -1 FirstWrite 36}
		mux_case_22150451_out {Type O LastRead -1 FirstWrite 36}
		mux_case_21148447_out {Type O LastRead -1 FirstWrite 36}
		mux_case_20146443_out {Type O LastRead -1 FirstWrite 36}
		mux_case_19144439_out {Type O LastRead -1 FirstWrite 36}
		mux_case_18142435_out {Type O LastRead -1 FirstWrite 36}
		mux_case_17140431_out {Type O LastRead -1 FirstWrite 36}
		mux_case_16138427_out {Type O LastRead -1 FirstWrite 36}
		mux_case_15136423_out {Type O LastRead -1 FirstWrite 36}
		mux_case_14134419_out {Type O LastRead -1 FirstWrite 36}
		mux_case_13132415_out {Type O LastRead -1 FirstWrite 36}
		mux_case_12130411_out {Type O LastRead -1 FirstWrite 36}
		mux_case_11128407_out {Type O LastRead -1 FirstWrite 36}
		mux_case_10126403_out {Type O LastRead -1 FirstWrite 36}
		mux_case_9124399_out {Type O LastRead -1 FirstWrite 36}
		mux_case_8122395_out {Type O LastRead -1 FirstWrite 36}
		mux_case_7120391_out {Type O LastRead -1 FirstWrite 36}
		mux_case_6118387_out {Type O LastRead -1 FirstWrite 36}
		mux_case_5116383_out {Type O LastRead -1 FirstWrite 36}
		mux_case_4114379_out {Type O LastRead -1 FirstWrite 36}
		mux_case_3112375_out {Type O LastRead -1 FirstWrite 36}
		mux_case_2110371_out {Type O LastRead -1 FirstWrite 36}
		mux_case_1108367_out {Type O LastRead -1 FirstWrite 36}
		mux_case_0106363_out {Type O LastRead -1 FirstWrite 36}}
	conv_bn_act_pool_Pipeline_Loop3_2Big {
		pool_acc {Type IO LastRead 0 FirstWrite 0}
		pool_acc_31 {Type IO LastRead 0 FirstWrite 0}
		pool_acc_30 {Type IO LastRead 0 FirstWrite 0}
		pool_acc_29 {Type IO LastRead 0 FirstWrite 0}
		pool_acc_28 {Type IO LastRead 0 FirstWrite 0}
		pool_acc_27 {Type IO LastRead 0 FirstWrite 0}
		pool_acc_26 {Type IO LastRead 0 FirstWrite 0}
		pool_acc_25 {Type IO LastRead 0 FirstWrite 0}
		pool_acc_24 {Type IO LastRead 0 FirstWrite 0}
		pool_acc_23 {Type IO LastRead 0 FirstWrite 0}
		pool_acc_22 {Type IO LastRead 0 FirstWrite 0}
		pool_acc_21 {Type IO LastRead 0 FirstWrite 0}
		pool_acc_20 {Type IO LastRead 0 FirstWrite 0}
		pool_acc_19 {Type IO LastRead 0 FirstWrite 0}
		pool_acc_18 {Type IO LastRead 0 FirstWrite 0}
		pool_acc_17 {Type IO LastRead 0 FirstWrite 0}
		pool_acc_16 {Type IO LastRead 0 FirstWrite 0}
		pool_acc_15 {Type IO LastRead 0 FirstWrite 0}
		pool_acc_14 {Type IO LastRead 0 FirstWrite 0}
		pool_acc_13 {Type IO LastRead 0 FirstWrite 0}
		pool_acc_12 {Type IO LastRead 0 FirstWrite 0}
		pool_acc_11 {Type IO LastRead 0 FirstWrite 0}
		pool_acc_10 {Type IO LastRead 0 FirstWrite 0}
		pool_acc_9 {Type IO LastRead 0 FirstWrite 0}
		pool_acc_8 {Type IO LastRead 0 FirstWrite 0}
		pool_acc_7 {Type IO LastRead 0 FirstWrite 0}
		pool_acc_6 {Type IO LastRead 0 FirstWrite 0}
		pool_acc_5 {Type IO LastRead 0 FirstWrite 0}
		pool_acc_4 {Type IO LastRead 0 FirstWrite 0}
		pool_acc_3 {Type IO LastRead 0 FirstWrite 0}
		pool_acc_2 {Type IO LastRead 0 FirstWrite 0}
		pool_acc_1 {Type IO LastRead 0 FirstWrite 0}
		mul4 {Type I LastRead 0 FirstWrite -1}
		U_slice {Type O LastRead -1 FirstWrite 7}}
	run_all_slices_unrolled_Pipeline_MergedLoop0 {
		merged {Type IO LastRead 2 FirstWrite 7}
		h_slice {Type I LastRead 0 FirstWrite -1}}
	run_all_slices_unrolled_Pipeline_Loop_BN2 {
		h_slice {Type IO LastRead 15 FirstWrite 38}
		BN2_var1 {Type I LastRead -1 FirstWrite -1}
		BN2_gamma1 {Type I LastRead -1 FirstWrite -1}}
	conv_bn_act_pool_2 {
		X_slice27 {Type I LastRead -1 FirstWrite -1}
		ConvW2 {Type I LastRead -1 FirstWrite -1}
		U_slice {Type O LastRead -1 FirstWrite 7}}
	conv_bn_act_pool_2_Pipeline_BNParamsLoop {
		mux_case_31232615_out {Type O LastRead -1 FirstWrite 36}
		mux_case_30230611_out {Type O LastRead -1 FirstWrite 36}
		mux_case_29228607_out {Type O LastRead -1 FirstWrite 36}
		mux_case_28226603_out {Type O LastRead -1 FirstWrite 36}
		mux_case_27224599_out {Type O LastRead -1 FirstWrite 36}
		mux_case_26222595_out {Type O LastRead -1 FirstWrite 36}
		mux_case_25220591_out {Type O LastRead -1 FirstWrite 36}
		mux_case_24218587_out {Type O LastRead -1 FirstWrite 36}
		mux_case_23216583_out {Type O LastRead -1 FirstWrite 36}
		mux_case_22214579_out {Type O LastRead -1 FirstWrite 36}
		mux_case_21212575_out {Type O LastRead -1 FirstWrite 36}
		mux_case_20210571_out {Type O LastRead -1 FirstWrite 36}
		mux_case_19208567_out {Type O LastRead -1 FirstWrite 36}
		mux_case_18206563_out {Type O LastRead -1 FirstWrite 36}
		mux_case_17204559_out {Type O LastRead -1 FirstWrite 36}
		mux_case_16202555_out {Type O LastRead -1 FirstWrite 36}
		mux_case_15200551_out {Type O LastRead -1 FirstWrite 36}
		mux_case_14198547_out {Type O LastRead -1 FirstWrite 36}
		mux_case_13196543_out {Type O LastRead -1 FirstWrite 36}
		mux_case_12194539_out {Type O LastRead -1 FirstWrite 36}
		mux_case_11192535_out {Type O LastRead -1 FirstWrite 36}
		mux_case_10190531_out {Type O LastRead -1 FirstWrite 36}
		mux_case_9188527_out {Type O LastRead -1 FirstWrite 36}
		mux_case_8186523_out {Type O LastRead -1 FirstWrite 36}
		mux_case_7184519_out {Type O LastRead -1 FirstWrite 36}
		mux_case_6182515_out {Type O LastRead -1 FirstWrite 36}
		mux_case_5180511_out {Type O LastRead -1 FirstWrite 36}
		mux_case_4178507_out {Type O LastRead -1 FirstWrite 36}
		mux_case_3176503_out {Type O LastRead -1 FirstWrite 36}
		mux_case_2174499_out {Type O LastRead -1 FirstWrite 36}
		mux_case_1172495_out {Type O LastRead -1 FirstWrite 36}
		mux_case_0170491_out {Type O LastRead -1 FirstWrite 36}
		mux_case_31168487_out {Type O LastRead -1 FirstWrite 36}
		mux_case_30166483_out {Type O LastRead -1 FirstWrite 36}
		mux_case_29164479_out {Type O LastRead -1 FirstWrite 36}
		mux_case_28162475_out {Type O LastRead -1 FirstWrite 36}
		mux_case_27160471_out {Type O LastRead -1 FirstWrite 36}
		mux_case_26158467_out {Type O LastRead -1 FirstWrite 36}
		mux_case_25156463_out {Type O LastRead -1 FirstWrite 36}
		mux_case_24154459_out {Type O LastRead -1 FirstWrite 36}
		mux_case_23152455_out {Type O LastRead -1 FirstWrite 36}
		mux_case_22150451_out {Type O LastRead -1 FirstWrite 36}
		mux_case_21148447_out {Type O LastRead -1 FirstWrite 36}
		mux_case_20146443_out {Type O LastRead -1 FirstWrite 36}
		mux_case_19144439_out {Type O LastRead -1 FirstWrite 36}
		mux_case_18142435_out {Type O LastRead -1 FirstWrite 36}
		mux_case_17140431_out {Type O LastRead -1 FirstWrite 36}
		mux_case_16138427_out {Type O LastRead -1 FirstWrite 36}
		mux_case_15136423_out {Type O LastRead -1 FirstWrite 36}
		mux_case_14134419_out {Type O LastRead -1 FirstWrite 36}
		mux_case_13132415_out {Type O LastRead -1 FirstWrite 36}
		mux_case_12130411_out {Type O LastRead -1 FirstWrite 36}
		mux_case_11128407_out {Type O LastRead -1 FirstWrite 36}
		mux_case_10126403_out {Type O LastRead -1 FirstWrite 36}
		mux_case_9124399_out {Type O LastRead -1 FirstWrite 36}
		mux_case_8122395_out {Type O LastRead -1 FirstWrite 36}
		mux_case_7120391_out {Type O LastRead -1 FirstWrite 36}
		mux_case_6118387_out {Type O LastRead -1 FirstWrite 36}
		mux_case_5116383_out {Type O LastRead -1 FirstWrite 36}
		mux_case_4114379_out {Type O LastRead -1 FirstWrite 36}
		mux_case_3112375_out {Type O LastRead -1 FirstWrite 36}
		mux_case_2110371_out {Type O LastRead -1 FirstWrite 36}
		mux_case_1108367_out {Type O LastRead -1 FirstWrite 36}
		mux_case_0106363_out {Type O LastRead -1 FirstWrite 36}}
	conv_bn_act_pool_2_Pipeline_Loop3_2Big {
		pool_acc {Type IO LastRead 0 FirstWrite 0}
		pool_acc_31 {Type IO LastRead 0 FirstWrite 0}
		pool_acc_30 {Type IO LastRead 0 FirstWrite 0}
		pool_acc_29 {Type IO LastRead 0 FirstWrite 0}
		pool_acc_28 {Type IO LastRead 0 FirstWrite 0}
		pool_acc_27 {Type IO LastRead 0 FirstWrite 0}
		pool_acc_26 {Type IO LastRead 0 FirstWrite 0}
		pool_acc_25 {Type IO LastRead 0 FirstWrite 0}
		pool_acc_24 {Type IO LastRead 0 FirstWrite 0}
		pool_acc_23 {Type IO LastRead 0 FirstWrite 0}
		pool_acc_22 {Type IO LastRead 0 FirstWrite 0}
		pool_acc_21 {Type IO LastRead 0 FirstWrite 0}
		pool_acc_20 {Type IO LastRead 0 FirstWrite 0}
		pool_acc_19 {Type IO LastRead 0 FirstWrite 0}
		pool_acc_18 {Type IO LastRead 0 FirstWrite 0}
		pool_acc_17 {Type IO LastRead 0 FirstWrite 0}
		pool_acc_16 {Type IO LastRead 0 FirstWrite 0}
		pool_acc_15 {Type IO LastRead 0 FirstWrite 0}
		pool_acc_14 {Type IO LastRead 0 FirstWrite 0}
		pool_acc_13 {Type IO LastRead 0 FirstWrite 0}
		pool_acc_12 {Type IO LastRead 0 FirstWrite 0}
		pool_acc_11 {Type IO LastRead 0 FirstWrite 0}
		pool_acc_10 {Type IO LastRead 0 FirstWrite 0}
		pool_acc_9 {Type IO LastRead 0 FirstWrite 0}
		pool_acc_8 {Type IO LastRead 0 FirstWrite 0}
		pool_acc_7 {Type IO LastRead 0 FirstWrite 0}
		pool_acc_6 {Type IO LastRead 0 FirstWrite 0}
		pool_acc_5 {Type IO LastRead 0 FirstWrite 0}
		pool_acc_4 {Type IO LastRead 0 FirstWrite 0}
		pool_acc_3 {Type IO LastRead 0 FirstWrite 0}
		pool_acc_2 {Type IO LastRead 0 FirstWrite 0}
		pool_acc_1 {Type IO LastRead 0 FirstWrite 0}
		mul3 {Type I LastRead 0 FirstWrite -1}
		U_slice {Type O LastRead -1 FirstWrite 7}}
	run_all_slices_unrolled_Pipeline_MergedLoop1 {
		merged {Type IO LastRead 2 FirstWrite 7}
		h_slice {Type I LastRead 0 FirstWrite -1}}
	run_all_slices_unrolled_Pipeline_Loop_BN3 {
		h_slice {Type IO LastRead 15 FirstWrite 38}
		BN2_var2 {Type I LastRead -1 FirstWrite -1}
		BN2_gamma2 {Type I LastRead -1 FirstWrite -1}}
	conv_bn_act_pool_3 {
		X_slice42 {Type I LastRead -1 FirstWrite -1}
		ConvW3 {Type I LastRead -1 FirstWrite -1}
		U_slice {Type O LastRead -1 FirstWrite 7}}
	conv_bn_act_pool_3_Pipeline_BNParamsLoop {
		mux_case_31232615_out {Type O LastRead -1 FirstWrite 36}
		mux_case_30230611_out {Type O LastRead -1 FirstWrite 36}
		mux_case_29228607_out {Type O LastRead -1 FirstWrite 36}
		mux_case_28226603_out {Type O LastRead -1 FirstWrite 36}
		mux_case_27224599_out {Type O LastRead -1 FirstWrite 36}
		mux_case_26222595_out {Type O LastRead -1 FirstWrite 36}
		mux_case_25220591_out {Type O LastRead -1 FirstWrite 36}
		mux_case_24218587_out {Type O LastRead -1 FirstWrite 36}
		mux_case_23216583_out {Type O LastRead -1 FirstWrite 36}
		mux_case_22214579_out {Type O LastRead -1 FirstWrite 36}
		mux_case_21212575_out {Type O LastRead -1 FirstWrite 36}
		mux_case_20210571_out {Type O LastRead -1 FirstWrite 36}
		mux_case_19208567_out {Type O LastRead -1 FirstWrite 36}
		mux_case_18206563_out {Type O LastRead -1 FirstWrite 36}
		mux_case_17204559_out {Type O LastRead -1 FirstWrite 36}
		mux_case_16202555_out {Type O LastRead -1 FirstWrite 36}
		mux_case_15200551_out {Type O LastRead -1 FirstWrite 36}
		mux_case_14198547_out {Type O LastRead -1 FirstWrite 36}
		mux_case_13196543_out {Type O LastRead -1 FirstWrite 36}
		mux_case_12194539_out {Type O LastRead -1 FirstWrite 36}
		mux_case_11192535_out {Type O LastRead -1 FirstWrite 36}
		mux_case_10190531_out {Type O LastRead -1 FirstWrite 36}
		mux_case_9188527_out {Type O LastRead -1 FirstWrite 36}
		mux_case_8186523_out {Type O LastRead -1 FirstWrite 36}
		mux_case_7184519_out {Type O LastRead -1 FirstWrite 36}
		mux_case_6182515_out {Type O LastRead -1 FirstWrite 36}
		mux_case_5180511_out {Type O LastRead -1 FirstWrite 36}
		mux_case_4178507_out {Type O LastRead -1 FirstWrite 36}
		mux_case_3176503_out {Type O LastRead -1 FirstWrite 36}
		mux_case_2174499_out {Type O LastRead -1 FirstWrite 36}
		mux_case_1172495_out {Type O LastRead -1 FirstWrite 36}
		mux_case_0170491_out {Type O LastRead -1 FirstWrite 36}
		mux_case_31168487_out {Type O LastRead -1 FirstWrite 36}
		mux_case_30166483_out {Type O LastRead -1 FirstWrite 36}
		mux_case_29164479_out {Type O LastRead -1 FirstWrite 36}
		mux_case_28162475_out {Type O LastRead -1 FirstWrite 36}
		mux_case_27160471_out {Type O LastRead -1 FirstWrite 36}
		mux_case_26158467_out {Type O LastRead -1 FirstWrite 36}
		mux_case_25156463_out {Type O LastRead -1 FirstWrite 36}
		mux_case_24154459_out {Type O LastRead -1 FirstWrite 36}
		mux_case_23152455_out {Type O LastRead -1 FirstWrite 36}
		mux_case_22150451_out {Type O LastRead -1 FirstWrite 36}
		mux_case_21148447_out {Type O LastRead -1 FirstWrite 36}
		mux_case_20146443_out {Type O LastRead -1 FirstWrite 36}
		mux_case_19144439_out {Type O LastRead -1 FirstWrite 36}
		mux_case_18142435_out {Type O LastRead -1 FirstWrite 36}
		mux_case_17140431_out {Type O LastRead -1 FirstWrite 36}
		mux_case_16138427_out {Type O LastRead -1 FirstWrite 36}
		mux_case_15136423_out {Type O LastRead -1 FirstWrite 36}
		mux_case_14134419_out {Type O LastRead -1 FirstWrite 36}
		mux_case_13132415_out {Type O LastRead -1 FirstWrite 36}
		mux_case_12130411_out {Type O LastRead -1 FirstWrite 36}
		mux_case_11128407_out {Type O LastRead -1 FirstWrite 36}
		mux_case_10126403_out {Type O LastRead -1 FirstWrite 36}
		mux_case_9124399_out {Type O LastRead -1 FirstWrite 36}
		mux_case_8122395_out {Type O LastRead -1 FirstWrite 36}
		mux_case_7120391_out {Type O LastRead -1 FirstWrite 36}
		mux_case_6118387_out {Type O LastRead -1 FirstWrite 36}
		mux_case_5116383_out {Type O LastRead -1 FirstWrite 36}
		mux_case_4114379_out {Type O LastRead -1 FirstWrite 36}
		mux_case_3112375_out {Type O LastRead -1 FirstWrite 36}
		mux_case_2110371_out {Type O LastRead -1 FirstWrite 36}
		mux_case_1108367_out {Type O LastRead -1 FirstWrite 36}
		mux_case_0106363_out {Type O LastRead -1 FirstWrite 36}}
	conv_bn_act_pool_3_Pipeline_Loop3_2Big {
		pool_acc {Type IO LastRead 0 FirstWrite 0}
		pool_acc_31 {Type IO LastRead 0 FirstWrite 0}
		pool_acc_30 {Type IO LastRead 0 FirstWrite 0}
		pool_acc_29 {Type IO LastRead 0 FirstWrite 0}
		pool_acc_28 {Type IO LastRead 0 FirstWrite 0}
		pool_acc_27 {Type IO LastRead 0 FirstWrite 0}
		pool_acc_26 {Type IO LastRead 0 FirstWrite 0}
		pool_acc_25 {Type IO LastRead 0 FirstWrite 0}
		pool_acc_24 {Type IO LastRead 0 FirstWrite 0}
		pool_acc_23 {Type IO LastRead 0 FirstWrite 0}
		pool_acc_22 {Type IO LastRead 0 FirstWrite 0}
		pool_acc_21 {Type IO LastRead 0 FirstWrite 0}
		pool_acc_20 {Type IO LastRead 0 FirstWrite 0}
		pool_acc_19 {Type IO LastRead 0 FirstWrite 0}
		pool_acc_18 {Type IO LastRead 0 FirstWrite 0}
		pool_acc_17 {Type IO LastRead 0 FirstWrite 0}
		pool_acc_16 {Type IO LastRead 0 FirstWrite 0}
		pool_acc_15 {Type IO LastRead 0 FirstWrite 0}
		pool_acc_14 {Type IO LastRead 0 FirstWrite 0}
		pool_acc_13 {Type IO LastRead 0 FirstWrite 0}
		pool_acc_12 {Type IO LastRead 0 FirstWrite 0}
		pool_acc_11 {Type IO LastRead 0 FirstWrite 0}
		pool_acc_10 {Type IO LastRead 0 FirstWrite 0}
		pool_acc_9 {Type IO LastRead 0 FirstWrite 0}
		pool_acc_8 {Type IO LastRead 0 FirstWrite 0}
		pool_acc_7 {Type IO LastRead 0 FirstWrite 0}
		pool_acc_6 {Type IO LastRead 0 FirstWrite 0}
		pool_acc_5 {Type IO LastRead 0 FirstWrite 0}
		pool_acc_4 {Type IO LastRead 0 FirstWrite 0}
		pool_acc_3 {Type IO LastRead 0 FirstWrite 0}
		pool_acc_2 {Type IO LastRead 0 FirstWrite 0}
		pool_acc_1 {Type IO LastRead 0 FirstWrite 0}
		mul2 {Type I LastRead 0 FirstWrite -1}
		U_slice {Type O LastRead -1 FirstWrite 7}}
	run_all_slices_unrolled_Pipeline_MergedLoop2 {
		merged {Type IO LastRead 2 FirstWrite 7}
		h_slice {Type I LastRead 0 FirstWrite -1}}
	run_all_slices_unrolled_Pipeline_Loop_BN4 {
		h_slice {Type IO LastRead 15 FirstWrite 38}
		BN2_var3 {Type I LastRead -1 FirstWrite -1}
		BN2_gamma3 {Type I LastRead -1 FirstWrite -1}}
	conv_bn_act_pool_4 {
		X_slice57 {Type I LastRead -1 FirstWrite -1}
		ConvW4 {Type I LastRead -1 FirstWrite -1}
		U_slice {Type O LastRead -1 FirstWrite 7}}
	conv_bn_act_pool_4_Pipeline_BNParamsLoop {
		mux_case_31232615_out {Type O LastRead -1 FirstWrite 36}
		mux_case_30230611_out {Type O LastRead -1 FirstWrite 36}
		mux_case_29228607_out {Type O LastRead -1 FirstWrite 36}
		mux_case_28226603_out {Type O LastRead -1 FirstWrite 36}
		mux_case_27224599_out {Type O LastRead -1 FirstWrite 36}
		mux_case_26222595_out {Type O LastRead -1 FirstWrite 36}
		mux_case_25220591_out {Type O LastRead -1 FirstWrite 36}
		mux_case_24218587_out {Type O LastRead -1 FirstWrite 36}
		mux_case_23216583_out {Type O LastRead -1 FirstWrite 36}
		mux_case_22214579_out {Type O LastRead -1 FirstWrite 36}
		mux_case_21212575_out {Type O LastRead -1 FirstWrite 36}
		mux_case_20210571_out {Type O LastRead -1 FirstWrite 36}
		mux_case_19208567_out {Type O LastRead -1 FirstWrite 36}
		mux_case_18206563_out {Type O LastRead -1 FirstWrite 36}
		mux_case_17204559_out {Type O LastRead -1 FirstWrite 36}
		mux_case_16202555_out {Type O LastRead -1 FirstWrite 36}
		mux_case_15200551_out {Type O LastRead -1 FirstWrite 36}
		mux_case_14198547_out {Type O LastRead -1 FirstWrite 36}
		mux_case_13196543_out {Type O LastRead -1 FirstWrite 36}
		mux_case_12194539_out {Type O LastRead -1 FirstWrite 36}
		mux_case_11192535_out {Type O LastRead -1 FirstWrite 36}
		mux_case_10190531_out {Type O LastRead -1 FirstWrite 36}
		mux_case_9188527_out {Type O LastRead -1 FirstWrite 36}
		mux_case_8186523_out {Type O LastRead -1 FirstWrite 36}
		mux_case_7184519_out {Type O LastRead -1 FirstWrite 36}
		mux_case_6182515_out {Type O LastRead -1 FirstWrite 36}
		mux_case_5180511_out {Type O LastRead -1 FirstWrite 36}
		mux_case_4178507_out {Type O LastRead -1 FirstWrite 36}
		mux_case_3176503_out {Type O LastRead -1 FirstWrite 36}
		mux_case_2174499_out {Type O LastRead -1 FirstWrite 36}
		mux_case_1172495_out {Type O LastRead -1 FirstWrite 36}
		mux_case_0170491_out {Type O LastRead -1 FirstWrite 36}
		mux_case_31168487_out {Type O LastRead -1 FirstWrite 36}
		mux_case_30166483_out {Type O LastRead -1 FirstWrite 36}
		mux_case_29164479_out {Type O LastRead -1 FirstWrite 36}
		mux_case_28162475_out {Type O LastRead -1 FirstWrite 36}
		mux_case_27160471_out {Type O LastRead -1 FirstWrite 36}
		mux_case_26158467_out {Type O LastRead -1 FirstWrite 36}
		mux_case_25156463_out {Type O LastRead -1 FirstWrite 36}
		mux_case_24154459_out {Type O LastRead -1 FirstWrite 36}
		mux_case_23152455_out {Type O LastRead -1 FirstWrite 36}
		mux_case_22150451_out {Type O LastRead -1 FirstWrite 36}
		mux_case_21148447_out {Type O LastRead -1 FirstWrite 36}
		mux_case_20146443_out {Type O LastRead -1 FirstWrite 36}
		mux_case_19144439_out {Type O LastRead -1 FirstWrite 36}
		mux_case_18142435_out {Type O LastRead -1 FirstWrite 36}
		mux_case_17140431_out {Type O LastRead -1 FirstWrite 36}
		mux_case_16138427_out {Type O LastRead -1 FirstWrite 36}
		mux_case_15136423_out {Type O LastRead -1 FirstWrite 36}
		mux_case_14134419_out {Type O LastRead -1 FirstWrite 36}
		mux_case_13132415_out {Type O LastRead -1 FirstWrite 36}
		mux_case_12130411_out {Type O LastRead -1 FirstWrite 36}
		mux_case_11128407_out {Type O LastRead -1 FirstWrite 36}
		mux_case_10126403_out {Type O LastRead -1 FirstWrite 36}
		mux_case_9124399_out {Type O LastRead -1 FirstWrite 36}
		mux_case_8122395_out {Type O LastRead -1 FirstWrite 36}
		mux_case_7120391_out {Type O LastRead -1 FirstWrite 36}
		mux_case_6118387_out {Type O LastRead -1 FirstWrite 36}
		mux_case_5116383_out {Type O LastRead -1 FirstWrite 36}
		mux_case_4114379_out {Type O LastRead -1 FirstWrite 36}
		mux_case_3112375_out {Type O LastRead -1 FirstWrite 36}
		mux_case_2110371_out {Type O LastRead -1 FirstWrite 36}
		mux_case_1108367_out {Type O LastRead -1 FirstWrite 36}
		mux_case_0106363_out {Type O LastRead -1 FirstWrite 36}}
	conv_bn_act_pool_4_Pipeline_Loop3_2Big {
		pool_acc {Type IO LastRead 0 FirstWrite 0}
		pool_acc_31 {Type IO LastRead 0 FirstWrite 0}
		pool_acc_30 {Type IO LastRead 0 FirstWrite 0}
		pool_acc_29 {Type IO LastRead 0 FirstWrite 0}
		pool_acc_28 {Type IO LastRead 0 FirstWrite 0}
		pool_acc_27 {Type IO LastRead 0 FirstWrite 0}
		pool_acc_26 {Type IO LastRead 0 FirstWrite 0}
		pool_acc_25 {Type IO LastRead 0 FirstWrite 0}
		pool_acc_24 {Type IO LastRead 0 FirstWrite 0}
		pool_acc_23 {Type IO LastRead 0 FirstWrite 0}
		pool_acc_22 {Type IO LastRead 0 FirstWrite 0}
		pool_acc_21 {Type IO LastRead 0 FirstWrite 0}
		pool_acc_20 {Type IO LastRead 0 FirstWrite 0}
		pool_acc_19 {Type IO LastRead 0 FirstWrite 0}
		pool_acc_18 {Type IO LastRead 0 FirstWrite 0}
		pool_acc_17 {Type IO LastRead 0 FirstWrite 0}
		pool_acc_16 {Type IO LastRead 0 FirstWrite 0}
		pool_acc_15 {Type IO LastRead 0 FirstWrite 0}
		pool_acc_14 {Type IO LastRead 0 FirstWrite 0}
		pool_acc_13 {Type IO LastRead 0 FirstWrite 0}
		pool_acc_12 {Type IO LastRead 0 FirstWrite 0}
		pool_acc_11 {Type IO LastRead 0 FirstWrite 0}
		pool_acc_10 {Type IO LastRead 0 FirstWrite 0}
		pool_acc_9 {Type IO LastRead 0 FirstWrite 0}
		pool_acc_8 {Type IO LastRead 0 FirstWrite 0}
		pool_acc_7 {Type IO LastRead 0 FirstWrite 0}
		pool_acc_6 {Type IO LastRead 0 FirstWrite 0}
		pool_acc_5 {Type IO LastRead 0 FirstWrite 0}
		pool_acc_4 {Type IO LastRead 0 FirstWrite 0}
		pool_acc_3 {Type IO LastRead 0 FirstWrite 0}
		pool_acc_2 {Type IO LastRead 0 FirstWrite 0}
		pool_acc_1 {Type IO LastRead 0 FirstWrite 0}
		mul1 {Type I LastRead 0 FirstWrite -1}
		U_slice {Type O LastRead -1 FirstWrite 7}}
	run_all_slices_unrolled_Pipeline_MergedLoop3 {
		merged {Type IO LastRead 2 FirstWrite 7}
		h_slice {Type I LastRead 0 FirstWrite -1}}
	run_all_slices_unrolled_Pipeline_Loop_BN5 {
		h_slice {Type IO LastRead 15 FirstWrite 38}
		BN2_var4 {Type I LastRead -1 FirstWrite -1}
		BN2_gamma4 {Type I LastRead -1 FirstWrite -1}}
	run_all_slices_unrolled_Pipeline_MergedLoop4 {
		merged {Type IO LastRead 2 FirstWrite 7}
		h_slice {Type I LastRead 0 FirstWrite -1}}}

set hasDtUnsupportedChannel 0

set PerformanceInfo {[
	{"Name" : "Latency", "Min" : "40081252", "Max" : "40125532"}
	, {"Name" : "Interval", "Min" : "40081252", "Max" : "40125532"}
]}

set PipelineEnableSignalInfo {[
]}

set Spec2ImplPortList { 
	merged { ap_memory {  { merged_address0 mem_address 1 5 }  { merged_ce0 mem_ce 1 1 }  { merged_we0 mem_we 1 1 }  { merged_d0 mem_din 1 16 }  { merged_address1 MemPortADDR2 1 5 }  { merged_ce1 MemPortCE2 1 1 }  { merged_q1 MemPortDOUT2 0 16 } } }
}
