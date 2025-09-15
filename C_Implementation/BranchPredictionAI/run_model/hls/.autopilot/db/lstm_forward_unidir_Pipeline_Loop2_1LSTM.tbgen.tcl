set moduleName lstm_forward_unidir_Pipeline_Loop2_1LSTM
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
set C_modelName {lstm_forward_unidir_Pipeline_Loop2_1LSTM}
set C_modelType { void 0 }
set ap_memory_interface_dict [dict create]
dict set ap_memory_interface_dict z_4 { MEM_WIDTH 16 MEM_SIZE 52 MASTER_TYPE BRAM_CTRL MEM_ADDRESS_MODE WORD_ADDRESS PACKAGE_IO port READ_LATENCY 0 }
dict set ap_memory_interface_dict z_3 { MEM_WIDTH 16 MEM_SIZE 52 MASTER_TYPE BRAM_CTRL MEM_ADDRESS_MODE WORD_ADDRESS PACKAGE_IO port READ_LATENCY 0 }
dict set ap_memory_interface_dict z_2 { MEM_WIDTH 16 MEM_SIZE 52 MASTER_TYPE BRAM_CTRL MEM_ADDRESS_MODE WORD_ADDRESS PACKAGE_IO port READ_LATENCY 0 }
dict set ap_memory_interface_dict z_1 { MEM_WIDTH 16 MEM_SIZE 52 MASTER_TYPE BRAM_CTRL MEM_ADDRESS_MODE WORD_ADDRESS PACKAGE_IO port READ_LATENCY 0 }
dict set ap_memory_interface_dict z { MEM_WIDTH 16 MEM_SIZE 52 MASTER_TYPE BRAM_CTRL MEM_ADDRESS_MODE WORD_ADDRESS PACKAGE_IO port READ_LATENCY 0 }
set C_modelArgList {
	{ z_4 int 16 regular {array 26 { 0 3 } 0 1 }  }
	{ z_3 int 16 regular {array 26 { 0 3 } 0 1 }  }
	{ z_2 int 16 regular {array 26 { 0 3 } 0 1 }  }
	{ z_1 int 16 regular {array 26 { 0 3 } 0 1 }  }
	{ z int 16 regular {array 26 { 0 3 } 0 1 }  }
}
set hasAXIMCache 0
set l_AXIML2Cache [list]
set AXIMCacheInstDict [dict create]
set C_modelArgMapList {[ 
	{ "Name" : "z_4", "interface" : "memory", "bitwidth" : 16, "direction" : "WRITEONLY"} , 
 	{ "Name" : "z_3", "interface" : "memory", "bitwidth" : 16, "direction" : "WRITEONLY"} , 
 	{ "Name" : "z_2", "interface" : "memory", "bitwidth" : 16, "direction" : "WRITEONLY"} , 
 	{ "Name" : "z_1", "interface" : "memory", "bitwidth" : 16, "direction" : "WRITEONLY"} , 
 	{ "Name" : "z", "interface" : "memory", "bitwidth" : 16, "direction" : "WRITEONLY"} ]}
# RTL Port declarations: 
set portNum 26
set portList { 
	{ ap_clk sc_in sc_logic 1 clock -1 } 
	{ ap_rst sc_in sc_logic 1 reset -1 active_high_sync } 
	{ ap_start sc_in sc_logic 1 start -1 } 
	{ ap_done sc_out sc_logic 1 predone -1 } 
	{ ap_idle sc_out sc_logic 1 done -1 } 
	{ ap_ready sc_out sc_logic 1 ready -1 } 
	{ z_4_address0 sc_out sc_lv 5 signal 0 } 
	{ z_4_ce0 sc_out sc_logic 1 signal 0 } 
	{ z_4_we0 sc_out sc_logic 1 signal 0 } 
	{ z_4_d0 sc_out sc_lv 16 signal 0 } 
	{ z_3_address0 sc_out sc_lv 5 signal 1 } 
	{ z_3_ce0 sc_out sc_logic 1 signal 1 } 
	{ z_3_we0 sc_out sc_logic 1 signal 1 } 
	{ z_3_d0 sc_out sc_lv 16 signal 1 } 
	{ z_2_address0 sc_out sc_lv 5 signal 2 } 
	{ z_2_ce0 sc_out sc_logic 1 signal 2 } 
	{ z_2_we0 sc_out sc_logic 1 signal 2 } 
	{ z_2_d0 sc_out sc_lv 16 signal 2 } 
	{ z_1_address0 sc_out sc_lv 5 signal 3 } 
	{ z_1_ce0 sc_out sc_logic 1 signal 3 } 
	{ z_1_we0 sc_out sc_logic 1 signal 3 } 
	{ z_1_d0 sc_out sc_lv 16 signal 3 } 
	{ z_address0 sc_out sc_lv 5 signal 4 } 
	{ z_ce0 sc_out sc_logic 1 signal 4 } 
	{ z_we0 sc_out sc_logic 1 signal 4 } 
	{ z_d0 sc_out sc_lv 16 signal 4 } 
}
set NewPortList {[ 
	{ "name": "ap_clk", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "clock", "bundle":{"name": "ap_clk", "role": "default" }} , 
 	{ "name": "ap_rst", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "reset", "bundle":{"name": "ap_rst", "role": "default" }} , 
 	{ "name": "ap_start", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "start", "bundle":{"name": "ap_start", "role": "default" }} , 
 	{ "name": "ap_done", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "predone", "bundle":{"name": "ap_done", "role": "default" }} , 
 	{ "name": "ap_idle", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "done", "bundle":{"name": "ap_idle", "role": "default" }} , 
 	{ "name": "ap_ready", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "ready", "bundle":{"name": "ap_ready", "role": "default" }} , 
 	{ "name": "z_4_address0", "direction": "out", "datatype": "sc_lv", "bitwidth":5, "type": "signal", "bundle":{"name": "z_4", "role": "address0" }} , 
 	{ "name": "z_4_ce0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "z_4", "role": "ce0" }} , 
 	{ "name": "z_4_we0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "z_4", "role": "we0" }} , 
 	{ "name": "z_4_d0", "direction": "out", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "z_4", "role": "d0" }} , 
 	{ "name": "z_3_address0", "direction": "out", "datatype": "sc_lv", "bitwidth":5, "type": "signal", "bundle":{"name": "z_3", "role": "address0" }} , 
 	{ "name": "z_3_ce0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "z_3", "role": "ce0" }} , 
 	{ "name": "z_3_we0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "z_3", "role": "we0" }} , 
 	{ "name": "z_3_d0", "direction": "out", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "z_3", "role": "d0" }} , 
 	{ "name": "z_2_address0", "direction": "out", "datatype": "sc_lv", "bitwidth":5, "type": "signal", "bundle":{"name": "z_2", "role": "address0" }} , 
 	{ "name": "z_2_ce0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "z_2", "role": "ce0" }} , 
 	{ "name": "z_2_we0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "z_2", "role": "we0" }} , 
 	{ "name": "z_2_d0", "direction": "out", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "z_2", "role": "d0" }} , 
 	{ "name": "z_1_address0", "direction": "out", "datatype": "sc_lv", "bitwidth":5, "type": "signal", "bundle":{"name": "z_1", "role": "address0" }} , 
 	{ "name": "z_1_ce0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "z_1", "role": "ce0" }} , 
 	{ "name": "z_1_we0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "z_1", "role": "we0" }} , 
 	{ "name": "z_1_d0", "direction": "out", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "z_1", "role": "d0" }} , 
 	{ "name": "z_address0", "direction": "out", "datatype": "sc_lv", "bitwidth":5, "type": "signal", "bundle":{"name": "z", "role": "address0" }} , 
 	{ "name": "z_ce0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "z", "role": "ce0" }} , 
 	{ "name": "z_we0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "z", "role": "we0" }} , 
 	{ "name": "z_d0", "direction": "out", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "z", "role": "d0" }}  ]}

set RtlHierarchyInfo {[
	{"ID" : "0", "Level" : "0", "Path" : "`AUTOTB_DUT_INST", "Parent" : "", "Child" : ["1", "2", "3"],
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
	{"ID" : "1", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.mul_7ns_9ns_15_1_1_U4", "Parent" : "0"},
	{"ID" : "2", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.sparsemux_257_7_16_1_1_U5", "Parent" : "0"},
	{"ID" : "3", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.flow_control_loop_pipe_sequential_init_U", "Parent" : "0"}]}


set ArgLastReadFirstWriteLatency {
	lstm_forward_unidir_Pipeline_Loop2_1LSTM {
		z_4 {Type O LastRead -1 FirstWrite 0}
		z_3 {Type O LastRead -1 FirstWrite 0}
		z_2 {Type O LastRead -1 FirstWrite 0}
		z_1 {Type O LastRead -1 FirstWrite 0}
		z {Type O LastRead -1 FirstWrite 0}}}

set hasDtUnsupportedChannel 0

set PerformanceInfo {[
	{"Name" : "Latency", "Min" : "130", "Max" : "130"}
	, {"Name" : "Interval", "Min" : "130", "Max" : "130"}
]}

set PipelineEnableSignalInfo {[
]}

set Spec2ImplPortList { 
	z_4 { ap_memory {  { z_4_address0 mem_address 1 5 }  { z_4_ce0 mem_ce 1 1 }  { z_4_we0 mem_we 1 1 }  { z_4_d0 mem_din 1 16 } } }
	z_3 { ap_memory {  { z_3_address0 mem_address 1 5 }  { z_3_ce0 mem_ce 1 1 }  { z_3_we0 mem_we 1 1 }  { z_3_d0 mem_din 1 16 } } }
	z_2 { ap_memory {  { z_2_address0 mem_address 1 5 }  { z_2_ce0 mem_ce 1 1 }  { z_2_we0 mem_we 1 1 }  { z_2_d0 mem_din 1 16 } } }
	z_1 { ap_memory {  { z_1_address0 mem_address 1 5 }  { z_1_ce0 mem_ce 1 1 }  { z_1_we0 mem_we 1 1 }  { z_1_d0 mem_din 1 16 } } }
	z { ap_memory {  { z_address0 mem_address 1 5 }  { z_ce0 mem_ce 1 1 }  { z_we0 mem_we 1 1 }  { z_d0 mem_din 1 16 } } }
}
