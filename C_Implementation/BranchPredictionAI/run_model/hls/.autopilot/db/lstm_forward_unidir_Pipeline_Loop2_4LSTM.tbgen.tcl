set moduleName lstm_forward_unidir_Pipeline_Loop2_4LSTM
set isTopModule 0
set isCombinational 0
set isDatapathOnly 0
set isPipelined 1
set pipeline_type loop_auto_rewind
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
set C_modelName {lstm_forward_unidir_Pipeline_Loop2_4LSTM}
set C_modelType { void 0 }
set ap_memory_interface_dict [dict create]
dict set ap_memory_interface_dict z { MEM_WIDTH 16 MEM_SIZE 52 MASTER_TYPE BRAM_CTRL MEM_ADDRESS_MODE WORD_ADDRESS PACKAGE_IO port READ_LATENCY 1 }
dict set ap_memory_interface_dict z_1 { MEM_WIDTH 16 MEM_SIZE 52 MASTER_TYPE BRAM_CTRL MEM_ADDRESS_MODE WORD_ADDRESS PACKAGE_IO port READ_LATENCY 1 }
dict set ap_memory_interface_dict z_2 { MEM_WIDTH 16 MEM_SIZE 52 MASTER_TYPE BRAM_CTRL MEM_ADDRESS_MODE WORD_ADDRESS PACKAGE_IO port READ_LATENCY 1 }
dict set ap_memory_interface_dict z_3 { MEM_WIDTH 16 MEM_SIZE 52 MASTER_TYPE BRAM_CTRL MEM_ADDRESS_MODE WORD_ADDRESS PACKAGE_IO port READ_LATENCY 1 }
dict set ap_memory_interface_dict z_4 { MEM_WIDTH 16 MEM_SIZE 52 MASTER_TYPE BRAM_CTRL MEM_ADDRESS_MODE WORD_ADDRESS PACKAGE_IO port READ_LATENCY 1 }
dict set ap_memory_interface_dict c_slice { MEM_WIDTH 16 MEM_SIZE 64 MASTER_TYPE BRAM_CTRL MEM_ADDRESS_MODE WORD_ADDRESS PACKAGE_IO port READ_LATENCY 0 }
dict set ap_memory_interface_dict h_slice { MEM_WIDTH 16 MEM_SIZE 64 MASTER_TYPE BRAM_CTRL MEM_ADDRESS_MODE WORD_ADDRESS PACKAGE_IO port READ_LATENCY 0 }
set C_modelArgList {
	{ z int 16 regular {array 26 { 1 3 } 1 1 }  }
	{ z_1 int 16 regular {array 26 { 1 3 } 1 1 }  }
	{ z_2 int 16 regular {array 26 { 1 3 } 1 1 }  }
	{ z_3 int 16 regular {array 26 { 1 3 } 1 1 }  }
	{ z_4 int 16 regular {array 26 { 1 3 } 1 1 }  }
	{ c_slice int 16 regular {array 32 { 0 1 } 1 1 } {global 2}  }
	{ h_slice int 16 regular {array 32 { 0 3 } 0 1 } {global 1}  }
}
set hasAXIMCache 0
set l_AXIML2Cache [list]
set AXIMCacheInstDict [dict create]
set C_modelArgMapList {[ 
	{ "Name" : "z", "interface" : "memory", "bitwidth" : 16, "direction" : "READONLY"} , 
 	{ "Name" : "z_1", "interface" : "memory", "bitwidth" : 16, "direction" : "READONLY"} , 
 	{ "Name" : "z_2", "interface" : "memory", "bitwidth" : 16, "direction" : "READONLY"} , 
 	{ "Name" : "z_3", "interface" : "memory", "bitwidth" : 16, "direction" : "READONLY"} , 
 	{ "Name" : "z_4", "interface" : "memory", "bitwidth" : 16, "direction" : "READONLY"} , 
 	{ "Name" : "c_slice", "interface" : "memory", "bitwidth" : 16, "direction" : "READWRITE", "extern" : 0} , 
 	{ "Name" : "h_slice", "interface" : "memory", "bitwidth" : 16, "direction" : "WRITEONLY", "extern" : 0} ]}
# RTL Port declarations: 
set portNum 40
set portList { 
	{ ap_clk sc_in sc_logic 1 clock -1 } 
	{ ap_rst sc_in sc_logic 1 reset -1 active_high_sync } 
	{ ap_start sc_in sc_logic 1 start -1 } 
	{ ap_done sc_out sc_logic 1 predone -1 } 
	{ ap_idle sc_out sc_logic 1 done -1 } 
	{ ap_ready sc_out sc_logic 1 ready -1 } 
	{ z_address0 sc_out sc_lv 5 signal 0 } 
	{ z_ce0 sc_out sc_logic 1 signal 0 } 
	{ z_q0 sc_in sc_lv 16 signal 0 } 
	{ z_1_address0 sc_out sc_lv 5 signal 1 } 
	{ z_1_ce0 sc_out sc_logic 1 signal 1 } 
	{ z_1_q0 sc_in sc_lv 16 signal 1 } 
	{ z_2_address0 sc_out sc_lv 5 signal 2 } 
	{ z_2_ce0 sc_out sc_logic 1 signal 2 } 
	{ z_2_q0 sc_in sc_lv 16 signal 2 } 
	{ z_3_address0 sc_out sc_lv 5 signal 3 } 
	{ z_3_ce0 sc_out sc_logic 1 signal 3 } 
	{ z_3_q0 sc_in sc_lv 16 signal 3 } 
	{ z_4_address0 sc_out sc_lv 5 signal 4 } 
	{ z_4_ce0 sc_out sc_logic 1 signal 4 } 
	{ z_4_q0 sc_in sc_lv 16 signal 4 } 
	{ c_slice_address0 sc_out sc_lv 5 signal 5 } 
	{ c_slice_ce0 sc_out sc_logic 1 signal 5 } 
	{ c_slice_we0 sc_out sc_logic 1 signal 5 } 
	{ c_slice_d0 sc_out sc_lv 16 signal 5 } 
	{ c_slice_address1 sc_out sc_lv 5 signal 5 } 
	{ c_slice_ce1 sc_out sc_logic 1 signal 5 } 
	{ c_slice_q1 sc_in sc_lv 16 signal 5 } 
	{ h_slice_address0 sc_out sc_lv 5 signal 6 } 
	{ h_slice_ce0 sc_out sc_logic 1 signal 6 } 
	{ h_slice_we0 sc_out sc_logic 1 signal 6 } 
	{ h_slice_d0 sc_out sc_lv 16 signal 6 } 
	{ grp_fu_386_p_din0 sc_out sc_lv 16 signal -1 } 
	{ grp_fu_386_p_din1 sc_out sc_lv 16 signal -1 } 
	{ grp_fu_386_p_dout0 sc_in sc_lv 16 signal -1 } 
	{ grp_fu_386_p_ce sc_out sc_logic 1 signal -1 } 
	{ grp_fu_395_p_din0 sc_out sc_lv 16 signal -1 } 
	{ grp_fu_395_p_din1 sc_out sc_lv 16 signal -1 } 
	{ grp_fu_395_p_dout0 sc_in sc_lv 16 signal -1 } 
	{ grp_fu_395_p_ce sc_out sc_logic 1 signal -1 } 
}
set NewPortList {[ 
	{ "name": "ap_clk", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "clock", "bundle":{"name": "ap_clk", "role": "default" }} , 
 	{ "name": "ap_rst", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "reset", "bundle":{"name": "ap_rst", "role": "default" }} , 
 	{ "name": "ap_start", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "start", "bundle":{"name": "ap_start", "role": "default" }} , 
 	{ "name": "ap_done", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "predone", "bundle":{"name": "ap_done", "role": "default" }} , 
 	{ "name": "ap_idle", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "done", "bundle":{"name": "ap_idle", "role": "default" }} , 
 	{ "name": "ap_ready", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "ready", "bundle":{"name": "ap_ready", "role": "default" }} , 
 	{ "name": "z_address0", "direction": "out", "datatype": "sc_lv", "bitwidth":5, "type": "signal", "bundle":{"name": "z", "role": "address0" }} , 
 	{ "name": "z_ce0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "z", "role": "ce0" }} , 
 	{ "name": "z_q0", "direction": "in", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "z", "role": "q0" }} , 
 	{ "name": "z_1_address0", "direction": "out", "datatype": "sc_lv", "bitwidth":5, "type": "signal", "bundle":{"name": "z_1", "role": "address0" }} , 
 	{ "name": "z_1_ce0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "z_1", "role": "ce0" }} , 
 	{ "name": "z_1_q0", "direction": "in", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "z_1", "role": "q0" }} , 
 	{ "name": "z_2_address0", "direction": "out", "datatype": "sc_lv", "bitwidth":5, "type": "signal", "bundle":{"name": "z_2", "role": "address0" }} , 
 	{ "name": "z_2_ce0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "z_2", "role": "ce0" }} , 
 	{ "name": "z_2_q0", "direction": "in", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "z_2", "role": "q0" }} , 
 	{ "name": "z_3_address0", "direction": "out", "datatype": "sc_lv", "bitwidth":5, "type": "signal", "bundle":{"name": "z_3", "role": "address0" }} , 
 	{ "name": "z_3_ce0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "z_3", "role": "ce0" }} , 
 	{ "name": "z_3_q0", "direction": "in", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "z_3", "role": "q0" }} , 
 	{ "name": "z_4_address0", "direction": "out", "datatype": "sc_lv", "bitwidth":5, "type": "signal", "bundle":{"name": "z_4", "role": "address0" }} , 
 	{ "name": "z_4_ce0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "z_4", "role": "ce0" }} , 
 	{ "name": "z_4_q0", "direction": "in", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "z_4", "role": "q0" }} , 
 	{ "name": "c_slice_address0", "direction": "out", "datatype": "sc_lv", "bitwidth":5, "type": "signal", "bundle":{"name": "c_slice", "role": "address0" }} , 
 	{ "name": "c_slice_ce0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "c_slice", "role": "ce0" }} , 
 	{ "name": "c_slice_we0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "c_slice", "role": "we0" }} , 
 	{ "name": "c_slice_d0", "direction": "out", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "c_slice", "role": "d0" }} , 
 	{ "name": "c_slice_address1", "direction": "out", "datatype": "sc_lv", "bitwidth":5, "type": "signal", "bundle":{"name": "c_slice", "role": "address1" }} , 
 	{ "name": "c_slice_ce1", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "c_slice", "role": "ce1" }} , 
 	{ "name": "c_slice_q1", "direction": "in", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "c_slice", "role": "q1" }} , 
 	{ "name": "h_slice_address0", "direction": "out", "datatype": "sc_lv", "bitwidth":5, "type": "signal", "bundle":{"name": "h_slice", "role": "address0" }} , 
 	{ "name": "h_slice_ce0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "h_slice", "role": "ce0" }} , 
 	{ "name": "h_slice_we0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "h_slice", "role": "we0" }} , 
 	{ "name": "h_slice_d0", "direction": "out", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "h_slice", "role": "d0" }} , 
 	{ "name": "grp_fu_386_p_din0", "direction": "out", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "grp_fu_386_p_din0", "role": "default" }} , 
 	{ "name": "grp_fu_386_p_din1", "direction": "out", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "grp_fu_386_p_din1", "role": "default" }} , 
 	{ "name": "grp_fu_386_p_dout0", "direction": "in", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "grp_fu_386_p_dout0", "role": "default" }} , 
 	{ "name": "grp_fu_386_p_ce", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "grp_fu_386_p_ce", "role": "default" }} , 
 	{ "name": "grp_fu_395_p_din0", "direction": "out", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "grp_fu_395_p_din0", "role": "default" }} , 
 	{ "name": "grp_fu_395_p_din1", "direction": "out", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "grp_fu_395_p_din1", "role": "default" }} , 
 	{ "name": "grp_fu_395_p_dout0", "direction": "in", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "grp_fu_395_p_dout0", "role": "default" }} , 
 	{ "name": "grp_fu_395_p_ce", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "grp_fu_395_p_ce", "role": "default" }}  ]}

set RtlHierarchyInfo {[
	{"ID" : "0", "Level" : "0", "Path" : "`AUTOTB_DUT_INST", "Parent" : "", "Child" : ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21", "22", "23", "24", "25", "26", "27", "28", "29", "30", "31", "32"],
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
	{"ID" : "1", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fadd_32ns_32ns_32_4_full_dsp_1_U13", "Parent" : "0"},
	{"ID" : "2", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fadd_32ns_32ns_32_4_full_dsp_1_U14", "Parent" : "0"},
	{"ID" : "3", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fadd_32ns_32ns_32_4_full_dsp_1_U15", "Parent" : "0"},
	{"ID" : "4", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fdiv_32ns_32ns_32_10_no_dsp_1_U16", "Parent" : "0"},
	{"ID" : "5", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fdiv_32ns_32ns_32_10_no_dsp_1_U17", "Parent" : "0"},
	{"ID" : "6", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fdiv_32ns_32ns_32_10_no_dsp_1_U18", "Parent" : "0"},
	{"ID" : "7", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fexp_32ns_32ns_32_8_full_dsp_1_U19", "Parent" : "0"},
	{"ID" : "8", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fexp_32ns_32ns_32_8_full_dsp_1_U20", "Parent" : "0"},
	{"ID" : "9", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fexp_32ns_32ns_32_8_full_dsp_1_U21", "Parent" : "0"},
	{"ID" : "10", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.sptohp_32ns_16_2_no_dsp_1_U22", "Parent" : "0"},
	{"ID" : "11", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.sptohp_32ns_16_2_no_dsp_1_U23", "Parent" : "0"},
	{"ID" : "12", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.sptohp_32ns_16_2_no_dsp_1_U24", "Parent" : "0"},
	{"ID" : "13", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.hptosp_16ns_32_2_no_dsp_1_U25", "Parent" : "0"},
	{"ID" : "14", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.hptosp_16ns_32_2_no_dsp_1_U26", "Parent" : "0"},
	{"ID" : "15", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.hptosp_16ns_32_2_no_dsp_1_U27", "Parent" : "0"},
	{"ID" : "16", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.hmul_16ns_16ns_16_4_max_dsp_1_U30", "Parent" : "0"},
	{"ID" : "17", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.hmul_16ns_16ns_16_4_max_dsp_1_U31", "Parent" : "0"},
	{"ID" : "18", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.hcmp_16ns_16ns_1_2_no_dsp_1_U32", "Parent" : "0"},
	{"ID" : "19", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.hcmp_16ns_16ns_1_2_no_dsp_1_U33", "Parent" : "0"},
	{"ID" : "20", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.hcmp_16ns_16ns_1_2_no_dsp_1_U34", "Parent" : "0"},
	{"ID" : "21", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.hcmp_16ns_16ns_1_2_no_dsp_1_U35", "Parent" : "0"},
	{"ID" : "22", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.mul_5ns_7ns_11_1_1_U36", "Parent" : "0"},
	{"ID" : "23", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.mul_6ns_8ns_13_1_1_U37", "Parent" : "0"},
	{"ID" : "24", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.mul_7ns_9ns_15_1_1_U38", "Parent" : "0"},
	{"ID" : "25", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.mul_7ns_9ns_15_1_1_U39", "Parent" : "0"},
	{"ID" : "26", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.sparsemux_11_3_16_1_1_U40", "Parent" : "0"},
	{"ID" : "27", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.sparsemux_11_3_16_1_1_U41", "Parent" : "0"},
	{"ID" : "28", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.sparsemux_11_3_16_1_1_U42", "Parent" : "0"},
	{"ID" : "29", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.sparsemux_11_3_16_1_1_U43", "Parent" : "0"},
	{"ID" : "30", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.sparsemux_7_2_16_1_1_U44", "Parent" : "0"},
	{"ID" : "31", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.sparsemux_7_2_16_1_1_U45", "Parent" : "0"},
	{"ID" : "32", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.flow_control_loop_pipe_sequential_init_U", "Parent" : "0"}]}


set ArgLastReadFirstWriteLatency {
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
	{"Name" : "Latency", "Min" : "75", "Max" : "75"}
	, {"Name" : "Interval", "Min" : "75", "Max" : "75"}
]}

set PipelineEnableSignalInfo {[
	{"Pipeline" : "0", "EnableSignal" : "ap_enable_pp0"}
]}

set Spec2ImplPortList { 
	z { ap_memory {  { z_address0 mem_address 1 5 }  { z_ce0 mem_ce 1 1 }  { z_q0 mem_dout 0 16 } } }
	z_1 { ap_memory {  { z_1_address0 mem_address 1 5 }  { z_1_ce0 mem_ce 1 1 }  { z_1_q0 mem_dout 0 16 } } }
	z_2 { ap_memory {  { z_2_address0 mem_address 1 5 }  { z_2_ce0 mem_ce 1 1 }  { z_2_q0 mem_dout 0 16 } } }
	z_3 { ap_memory {  { z_3_address0 mem_address 1 5 }  { z_3_ce0 mem_ce 1 1 }  { z_3_q0 mem_dout 0 16 } } }
	z_4 { ap_memory {  { z_4_address0 mem_address 1 5 }  { z_4_ce0 mem_ce 1 1 }  { z_4_q0 mem_dout 0 16 } } }
	c_slice { ap_memory {  { c_slice_address0 mem_address 1 5 }  { c_slice_ce0 mem_ce 1 1 }  { c_slice_we0 mem_we 1 1 }  { c_slice_d0 mem_din 1 16 }  { c_slice_address1 MemPortADDR2 1 5 }  { c_slice_ce1 MemPortCE2 1 1 }  { c_slice_q1 MemPortDOUT2 0 16 } } }
	h_slice { ap_memory {  { h_slice_address0 mem_address 1 5 }  { h_slice_ce0 mem_ce 1 1 }  { h_slice_we0 mem_we 1 1 }  { h_slice_d0 mem_din 1 16 } } }
}
