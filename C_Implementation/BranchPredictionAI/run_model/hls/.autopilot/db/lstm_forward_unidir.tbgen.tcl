set moduleName lstm_forward_unidir
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
set C_modelName {lstm_forward_unidir}
set C_modelType { void 0 }
set ap_memory_interface_dict [dict create]
dict set ap_memory_interface_dict W_ifog { MEM_WIDTH 16 MEM_SIZE 8192 MASTER_TYPE BRAM_CTRL MEM_ADDRESS_MODE WORD_ADDRESS PACKAGE_IO port READ_LATENCY 3 }
dict set ap_memory_interface_dict R_ifog { MEM_WIDTH 16 MEM_SIZE 8192 MASTER_TYPE BRAM_CTRL MEM_ADDRESS_MODE WORD_ADDRESS PACKAGE_IO port READ_LATENCY 3 }
dict set ap_memory_interface_dict h_slice { MEM_WIDTH 16 MEM_SIZE 64 MASTER_TYPE BRAM_CTRL MEM_ADDRESS_MODE WORD_ADDRESS PACKAGE_IO port READ_LATENCY 1 }
dict set ap_memory_interface_dict U_slice { MEM_WIDTH 16 MEM_SIZE 768 MASTER_TYPE BRAM_CTRL MEM_ADDRESS_MODE WORD_ADDRESS PACKAGE_IO port READ_LATENCY 3 }
set C_modelArgList {
	{ W_ifog int 16 regular {array 4096 { 1 } 3 1 }  }
	{ R_ifog int 16 regular {array 4096 { 1 } 3 1 }  }
	{ h_slice int 16 regular {array 32 { 2 3 } 1 1 } {global 2}  }
	{ U_slice int 16 regular {array 384 { 1 3 } 3 1 } {global 0}  }
}
set hasAXIMCache 0
set l_AXIML2Cache [list]
set AXIMCacheInstDict [dict create]
set C_modelArgMapList {[ 
	{ "Name" : "W_ifog", "interface" : "memory", "bitwidth" : 16, "direction" : "READONLY"} , 
 	{ "Name" : "R_ifog", "interface" : "memory", "bitwidth" : 16, "direction" : "READONLY"} , 
 	{ "Name" : "h_slice", "interface" : "memory", "bitwidth" : 16, "direction" : "READWRITE", "extern" : 0} , 
 	{ "Name" : "U_slice", "interface" : "memory", "bitwidth" : 16, "direction" : "READONLY", "extern" : 0} ]}
# RTL Port declarations: 
set portNum 20
set portList { 
	{ ap_clk sc_in sc_logic 1 clock -1 } 
	{ ap_rst sc_in sc_logic 1 reset -1 active_high_sync } 
	{ ap_start sc_in sc_logic 1 start -1 } 
	{ ap_done sc_out sc_logic 1 predone -1 } 
	{ ap_idle sc_out sc_logic 1 done -1 } 
	{ ap_ready sc_out sc_logic 1 ready -1 } 
	{ W_ifog_address0 sc_out sc_lv 12 signal 0 } 
	{ W_ifog_ce0 sc_out sc_logic 1 signal 0 } 
	{ W_ifog_q0 sc_in sc_lv 16 signal 0 } 
	{ R_ifog_address0 sc_out sc_lv 12 signal 1 } 
	{ R_ifog_ce0 sc_out sc_logic 1 signal 1 } 
	{ R_ifog_q0 sc_in sc_lv 16 signal 1 } 
	{ h_slice_address0 sc_out sc_lv 5 signal 2 } 
	{ h_slice_ce0 sc_out sc_logic 1 signal 2 } 
	{ h_slice_we0 sc_out sc_logic 1 signal 2 } 
	{ h_slice_d0 sc_out sc_lv 16 signal 2 } 
	{ h_slice_q0 sc_in sc_lv 16 signal 2 } 
	{ U_slice_address0 sc_out sc_lv 9 signal 3 } 
	{ U_slice_ce0 sc_out sc_logic 1 signal 3 } 
	{ U_slice_q0 sc_in sc_lv 16 signal 3 } 
}
set NewPortList {[ 
	{ "name": "ap_clk", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "clock", "bundle":{"name": "ap_clk", "role": "default" }} , 
 	{ "name": "ap_rst", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "reset", "bundle":{"name": "ap_rst", "role": "default" }} , 
 	{ "name": "ap_start", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "start", "bundle":{"name": "ap_start", "role": "default" }} , 
 	{ "name": "ap_done", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "predone", "bundle":{"name": "ap_done", "role": "default" }} , 
 	{ "name": "ap_idle", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "done", "bundle":{"name": "ap_idle", "role": "default" }} , 
 	{ "name": "ap_ready", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "ready", "bundle":{"name": "ap_ready", "role": "default" }} , 
 	{ "name": "W_ifog_address0", "direction": "out", "datatype": "sc_lv", "bitwidth":12, "type": "signal", "bundle":{"name": "W_ifog", "role": "address0" }} , 
 	{ "name": "W_ifog_ce0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "W_ifog", "role": "ce0" }} , 
 	{ "name": "W_ifog_q0", "direction": "in", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "W_ifog", "role": "q0" }} , 
 	{ "name": "R_ifog_address0", "direction": "out", "datatype": "sc_lv", "bitwidth":12, "type": "signal", "bundle":{"name": "R_ifog", "role": "address0" }} , 
 	{ "name": "R_ifog_ce0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "R_ifog", "role": "ce0" }} , 
 	{ "name": "R_ifog_q0", "direction": "in", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "R_ifog", "role": "q0" }} , 
 	{ "name": "h_slice_address0", "direction": "out", "datatype": "sc_lv", "bitwidth":5, "type": "signal", "bundle":{"name": "h_slice", "role": "address0" }} , 
 	{ "name": "h_slice_ce0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "h_slice", "role": "ce0" }} , 
 	{ "name": "h_slice_we0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "h_slice", "role": "we0" }} , 
 	{ "name": "h_slice_d0", "direction": "out", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "h_slice", "role": "d0" }} , 
 	{ "name": "h_slice_q0", "direction": "in", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "h_slice", "role": "q0" }} , 
 	{ "name": "U_slice_address0", "direction": "out", "datatype": "sc_lv", "bitwidth":9, "type": "signal", "bundle":{"name": "U_slice", "role": "address0" }} , 
 	{ "name": "U_slice_ce0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "U_slice", "role": "ce0" }} , 
 	{ "name": "U_slice_q0", "direction": "in", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "U_slice", "role": "q0" }}  ]}

set RtlHierarchyInfo {[
	{"ID" : "0", "Level" : "0", "Path" : "`AUTOTB_DUT_INST", "Parent" : "", "Child" : ["1", "2", "3", "4", "5", "6", "7", "9", "13", "46", "47", "48", "49", "50", "51", "52", "53"],
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
					{"ID" : "7", "SubInstance" : "grp_lstm_forward_unidir_Pipeline_VITIS_LOOP_145_1_fu_356", "Port" : "h_slice", "Inst_start_state" : "1", "Inst_end_state" : "2"},
					{"ID" : "13", "SubInstance" : "grp_lstm_forward_unidir_Pipeline_Loop2_4LSTM_fu_373", "Port" : "h_slice", "Inst_start_state" : "40", "Inst_end_state" : "41"}]},
			{"Name" : "c_slice", "Type" : "Memory", "Direction" : "IO",
				"SubConnect" : [
					{"ID" : "7", "SubInstance" : "grp_lstm_forward_unidir_Pipeline_VITIS_LOOP_145_1_fu_356", "Port" : "c_slice", "Inst_start_state" : "1", "Inst_end_state" : "2"},
					{"ID" : "13", "SubInstance" : "grp_lstm_forward_unidir_Pipeline_Loop2_4LSTM_fu_373", "Port" : "c_slice", "Inst_start_state" : "40", "Inst_end_state" : "41"}]},
			{"Name" : "U_slice", "Type" : "Memory", "Direction" : "I"}],
		"Loop" : [
			{"Name" : "Loop2_2LSTM_Loop3_2_1LSTM", "PipelineType" : "pipeline",
				"LoopDec" : {"FSMBitwidth" : "9", "FirstState" : "ap_ST_fsm_pp0_stage0", "FirstStateIter" : "ap_enable_reg_pp0_iter0", "FirstStateBlock" : "ap_block_pp0_stage0_subdone", "LastState" : "ap_ST_fsm_pp0_stage0", "LastStateIter" : "ap_enable_reg_pp0_iter16", "LastStateBlock" : "ap_block_pp0_stage0_subdone", "PreState" : ["ap_ST_fsm_state4"], "QuitState" : "ap_ST_fsm_pp0_stage0", "QuitStateIter" : "ap_enable_reg_pp0_iter16", "QuitStateBlock" : "ap_block_pp0_stage0_subdone", "PostState" : ["ap_ST_fsm_state22"]}},
			{"Name" : "Loop2_3LSTM_Loop3_3_1LSTM", "PipelineType" : "pipeline",
				"LoopDec" : {"FSMBitwidth" : "9", "FirstState" : "ap_ST_fsm_pp1_stage0", "FirstStateIter" : "ap_enable_reg_pp1_iter0", "FirstStateBlock" : "ap_block_pp1_stage0_subdone", "LastState" : "ap_ST_fsm_pp1_stage0", "LastStateIter" : "ap_enable_reg_pp1_iter16", "LastStateBlock" : "ap_block_pp1_stage0_subdone", "PreState" : ["ap_ST_fsm_state22"], "QuitState" : "ap_ST_fsm_pp1_stage0", "QuitStateIter" : "ap_enable_reg_pp1_iter16", "QuitStateBlock" : "ap_block_pp1_stage0_subdone", "PostState" : ["ap_ST_fsm_state40"]}},
			{"Name" : "Loop1LSTM", "PipelineType" : "no",
				"LoopDec" : {"FSMBitwidth" : "9", "FirstState" : "ap_ST_fsm_state3", "LastState" : ["ap_ST_fsm_state41"], "QuitState" : ["ap_ST_fsm_state3"], "PreState" : ["ap_ST_fsm_state2"], "PostState" : ["ap_ST_fsm_state1"], "OneDepthLoop" : "0", "OneStateBlock": ""}}]},
	{"ID" : "1", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.c_slice_U", "Parent" : "0"},
	{"ID" : "2", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.z_U", "Parent" : "0"},
	{"ID" : "3", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.z_1_U", "Parent" : "0"},
	{"ID" : "4", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.z_2_U", "Parent" : "0"},
	{"ID" : "5", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.z_3_U", "Parent" : "0"},
	{"ID" : "6", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.z_4_U", "Parent" : "0"},
	{"ID" : "7", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.grp_lstm_forward_unidir_Pipeline_VITIS_LOOP_145_1_fu_356", "Parent" : "0", "Child" : ["8"],
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
	{"ID" : "8", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_lstm_forward_unidir_Pipeline_VITIS_LOOP_145_1_fu_356.flow_control_loop_pipe_sequential_init_U", "Parent" : "7"},
	{"ID" : "9", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.grp_lstm_forward_unidir_Pipeline_Loop2_1LSTM_fu_364", "Parent" : "0", "Child" : ["10", "11", "12"],
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
	{"ID" : "10", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_lstm_forward_unidir_Pipeline_Loop2_1LSTM_fu_364.mul_7ns_9ns_15_1_1_U4", "Parent" : "9"},
	{"ID" : "11", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_lstm_forward_unidir_Pipeline_Loop2_1LSTM_fu_364.sparsemux_257_7_16_1_1_U5", "Parent" : "9"},
	{"ID" : "12", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_lstm_forward_unidir_Pipeline_Loop2_1LSTM_fu_364.flow_control_loop_pipe_sequential_init_U", "Parent" : "9"},
	{"ID" : "13", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.grp_lstm_forward_unidir_Pipeline_Loop2_4LSTM_fu_373", "Parent" : "0", "Child" : ["14", "15", "16", "17", "18", "19", "20", "21", "22", "23", "24", "25", "26", "27", "28", "29", "30", "31", "32", "33", "34", "35", "36", "37", "38", "39", "40", "41", "42", "43", "44", "45"],
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
	{"ID" : "14", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_lstm_forward_unidir_Pipeline_Loop2_4LSTM_fu_373.fadd_32ns_32ns_32_4_full_dsp_1_U13", "Parent" : "13"},
	{"ID" : "15", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_lstm_forward_unidir_Pipeline_Loop2_4LSTM_fu_373.fadd_32ns_32ns_32_4_full_dsp_1_U14", "Parent" : "13"},
	{"ID" : "16", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_lstm_forward_unidir_Pipeline_Loop2_4LSTM_fu_373.fadd_32ns_32ns_32_4_full_dsp_1_U15", "Parent" : "13"},
	{"ID" : "17", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_lstm_forward_unidir_Pipeline_Loop2_4LSTM_fu_373.fdiv_32ns_32ns_32_10_no_dsp_1_U16", "Parent" : "13"},
	{"ID" : "18", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_lstm_forward_unidir_Pipeline_Loop2_4LSTM_fu_373.fdiv_32ns_32ns_32_10_no_dsp_1_U17", "Parent" : "13"},
	{"ID" : "19", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_lstm_forward_unidir_Pipeline_Loop2_4LSTM_fu_373.fdiv_32ns_32ns_32_10_no_dsp_1_U18", "Parent" : "13"},
	{"ID" : "20", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_lstm_forward_unidir_Pipeline_Loop2_4LSTM_fu_373.fexp_32ns_32ns_32_8_full_dsp_1_U19", "Parent" : "13"},
	{"ID" : "21", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_lstm_forward_unidir_Pipeline_Loop2_4LSTM_fu_373.fexp_32ns_32ns_32_8_full_dsp_1_U20", "Parent" : "13"},
	{"ID" : "22", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_lstm_forward_unidir_Pipeline_Loop2_4LSTM_fu_373.fexp_32ns_32ns_32_8_full_dsp_1_U21", "Parent" : "13"},
	{"ID" : "23", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_lstm_forward_unidir_Pipeline_Loop2_4LSTM_fu_373.sptohp_32ns_16_2_no_dsp_1_U22", "Parent" : "13"},
	{"ID" : "24", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_lstm_forward_unidir_Pipeline_Loop2_4LSTM_fu_373.sptohp_32ns_16_2_no_dsp_1_U23", "Parent" : "13"},
	{"ID" : "25", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_lstm_forward_unidir_Pipeline_Loop2_4LSTM_fu_373.sptohp_32ns_16_2_no_dsp_1_U24", "Parent" : "13"},
	{"ID" : "26", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_lstm_forward_unidir_Pipeline_Loop2_4LSTM_fu_373.hptosp_16ns_32_2_no_dsp_1_U25", "Parent" : "13"},
	{"ID" : "27", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_lstm_forward_unidir_Pipeline_Loop2_4LSTM_fu_373.hptosp_16ns_32_2_no_dsp_1_U26", "Parent" : "13"},
	{"ID" : "28", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_lstm_forward_unidir_Pipeline_Loop2_4LSTM_fu_373.hptosp_16ns_32_2_no_dsp_1_U27", "Parent" : "13"},
	{"ID" : "29", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_lstm_forward_unidir_Pipeline_Loop2_4LSTM_fu_373.hmul_16ns_16ns_16_4_max_dsp_1_U30", "Parent" : "13"},
	{"ID" : "30", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_lstm_forward_unidir_Pipeline_Loop2_4LSTM_fu_373.hmul_16ns_16ns_16_4_max_dsp_1_U31", "Parent" : "13"},
	{"ID" : "31", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_lstm_forward_unidir_Pipeline_Loop2_4LSTM_fu_373.hcmp_16ns_16ns_1_2_no_dsp_1_U32", "Parent" : "13"},
	{"ID" : "32", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_lstm_forward_unidir_Pipeline_Loop2_4LSTM_fu_373.hcmp_16ns_16ns_1_2_no_dsp_1_U33", "Parent" : "13"},
	{"ID" : "33", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_lstm_forward_unidir_Pipeline_Loop2_4LSTM_fu_373.hcmp_16ns_16ns_1_2_no_dsp_1_U34", "Parent" : "13"},
	{"ID" : "34", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_lstm_forward_unidir_Pipeline_Loop2_4LSTM_fu_373.hcmp_16ns_16ns_1_2_no_dsp_1_U35", "Parent" : "13"},
	{"ID" : "35", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_lstm_forward_unidir_Pipeline_Loop2_4LSTM_fu_373.mul_5ns_7ns_11_1_1_U36", "Parent" : "13"},
	{"ID" : "36", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_lstm_forward_unidir_Pipeline_Loop2_4LSTM_fu_373.mul_6ns_8ns_13_1_1_U37", "Parent" : "13"},
	{"ID" : "37", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_lstm_forward_unidir_Pipeline_Loop2_4LSTM_fu_373.mul_7ns_9ns_15_1_1_U38", "Parent" : "13"},
	{"ID" : "38", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_lstm_forward_unidir_Pipeline_Loop2_4LSTM_fu_373.mul_7ns_9ns_15_1_1_U39", "Parent" : "13"},
	{"ID" : "39", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_lstm_forward_unidir_Pipeline_Loop2_4LSTM_fu_373.sparsemux_11_3_16_1_1_U40", "Parent" : "13"},
	{"ID" : "40", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_lstm_forward_unidir_Pipeline_Loop2_4LSTM_fu_373.sparsemux_11_3_16_1_1_U41", "Parent" : "13"},
	{"ID" : "41", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_lstm_forward_unidir_Pipeline_Loop2_4LSTM_fu_373.sparsemux_11_3_16_1_1_U42", "Parent" : "13"},
	{"ID" : "42", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_lstm_forward_unidir_Pipeline_Loop2_4LSTM_fu_373.sparsemux_11_3_16_1_1_U43", "Parent" : "13"},
	{"ID" : "43", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_lstm_forward_unidir_Pipeline_Loop2_4LSTM_fu_373.sparsemux_7_2_16_1_1_U44", "Parent" : "13"},
	{"ID" : "44", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_lstm_forward_unidir_Pipeline_Loop2_4LSTM_fu_373.sparsemux_7_2_16_1_1_U45", "Parent" : "13"},
	{"ID" : "45", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_lstm_forward_unidir_Pipeline_Loop2_4LSTM_fu_373.flow_control_loop_pipe_sequential_init_U", "Parent" : "13"},
	{"ID" : "46", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.hadd_16ns_16ns_16_5_full_dsp_1_U67", "Parent" : "0"},
	{"ID" : "47", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.hmul_16ns_16ns_16_4_max_dsp_1_U68", "Parent" : "0"},
	{"ID" : "48", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.urem_8ns_4ns_3_12_1_U69", "Parent" : "0"},
	{"ID" : "49", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.mul_8ns_10ns_17_1_1_U70", "Parent" : "0"},
	{"ID" : "50", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.sparsemux_11_3_16_1_1_U71", "Parent" : "0"},
	{"ID" : "51", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.urem_8ns_4ns_3_12_1_U72", "Parent" : "0"},
	{"ID" : "52", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.mul_8ns_10ns_17_1_1_U73", "Parent" : "0"},
	{"ID" : "53", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.sparsemux_11_3_16_1_1_U74", "Parent" : "0"}]}


set ArgLastReadFirstWriteLatency {
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
		h_slice {Type O LastRead -1 FirstWrite 42}}}

set hasDtUnsupportedChannel 0

set PerformanceInfo {[
	{"Name" : "Latency", "Min" : "101244", "Max" : "101244"}
	, {"Name" : "Interval", "Min" : "101244", "Max" : "101244"}
]}

set PipelineEnableSignalInfo {[
	{"Pipeline" : "0", "EnableSignal" : "ap_enable_pp0"}
	{"Pipeline" : "1", "EnableSignal" : "ap_enable_pp1"}
]}

set Spec2ImplPortList { 
	W_ifog { ap_memory {  { W_ifog_address0 mem_address 1 12 }  { W_ifog_ce0 mem_ce 1 1 }  { W_ifog_q0 mem_dout 0 16 } } }
	R_ifog { ap_memory {  { R_ifog_address0 mem_address 1 12 }  { R_ifog_ce0 mem_ce 1 1 }  { R_ifog_q0 mem_dout 0 16 } } }
	h_slice { ap_memory {  { h_slice_address0 mem_address 1 5 }  { h_slice_ce0 mem_ce 1 1 }  { h_slice_we0 mem_we 1 1 }  { h_slice_d0 mem_din 1 16 }  { h_slice_q0 mem_dout 0 16 } } }
	U_slice { ap_memory {  { U_slice_address0 mem_address 1 9 }  { U_slice_ce0 mem_ce 1 1 }  { U_slice_q0 mem_dout 0 16 } } }
}
