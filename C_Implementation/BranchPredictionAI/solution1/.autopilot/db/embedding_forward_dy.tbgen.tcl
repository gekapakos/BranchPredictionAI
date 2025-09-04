set moduleName embedding_forward_dy
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
set C_modelName {embedding_forward_dy}
set C_modelType { void 0 }
set C_modelArgList {
	{ X int 16 regular {array 18624 { 0 3 } 0 1 }  }
}
set C_modelArgMapList {[ 
	{ "Name" : "X", "interface" : "memory", "bitwidth" : 16, "direction" : "WRITEONLY"} ]}
# RTL Port declarations: 
set portNum 10
set portList { 
	{ ap_clk sc_in sc_logic 1 clock -1 } 
	{ ap_rst sc_in sc_logic 1 reset -1 active_high_sync } 
	{ ap_start sc_in sc_logic 1 start -1 } 
	{ ap_done sc_out sc_logic 1 predone -1 } 
	{ ap_idle sc_out sc_logic 1 done -1 } 
	{ ap_ready sc_out sc_logic 1 ready -1 } 
	{ X_address0 sc_out sc_lv 15 signal 0 } 
	{ X_ce0 sc_out sc_logic 1 signal 0 } 
	{ X_we0 sc_out sc_logic 1 signal 0 } 
	{ X_d0 sc_out sc_lv 16 signal 0 } 
}
set NewPortList {[ 
	{ "name": "ap_clk", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "clock", "bundle":{"name": "ap_clk", "role": "default" }} , 
 	{ "name": "ap_rst", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "reset", "bundle":{"name": "ap_rst", "role": "default" }} , 
 	{ "name": "ap_start", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "start", "bundle":{"name": "ap_start", "role": "default" }} , 
 	{ "name": "ap_done", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "predone", "bundle":{"name": "ap_done", "role": "default" }} , 
 	{ "name": "ap_idle", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "done", "bundle":{"name": "ap_idle", "role": "default" }} , 
 	{ "name": "ap_ready", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "ready", "bundle":{"name": "ap_ready", "role": "default" }} , 
 	{ "name": "X_address0", "direction": "out", "datatype": "sc_lv", "bitwidth":15, "type": "signal", "bundle":{"name": "X", "role": "address0" }} , 
 	{ "name": "X_ce0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "X", "role": "ce0" }} , 
 	{ "name": "X_we0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "X", "role": "we0" }} , 
 	{ "name": "X_d0", "direction": "out", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "X", "role": "d0" }}  ]}

set RtlHierarchyInfo {[
	{"ID" : "0", "Level" : "0", "Path" : "`AUTOTB_DUT_INST", "Parent" : "", "Child" : ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20"],
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
	{"ID" : "1", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.tokens4_U", "Parent" : "0"},
	{"ID" : "2", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.Emb4_0_U", "Parent" : "0"},
	{"ID" : "3", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.Emb4_1_U", "Parent" : "0"},
	{"ID" : "4", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.Emb4_2_U", "Parent" : "0"},
	{"ID" : "5", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.Emb4_3_U", "Parent" : "0"},
	{"ID" : "6", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.Emb4_4_U", "Parent" : "0"},
	{"ID" : "7", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.Emb4_5_U", "Parent" : "0"},
	{"ID" : "8", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.Emb4_6_U", "Parent" : "0"},
	{"ID" : "9", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.Emb4_7_U", "Parent" : "0"},
	{"ID" : "10", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.Emb4_8_U", "Parent" : "0"},
	{"ID" : "11", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.Emb4_9_U", "Parent" : "0"},
	{"ID" : "12", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.Emb4_10_U", "Parent" : "0"},
	{"ID" : "13", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.Emb4_11_U", "Parent" : "0"},
	{"ID" : "14", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.Emb4_12_U", "Parent" : "0"},
	{"ID" : "15", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.Emb4_13_U", "Parent" : "0"},
	{"ID" : "16", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.Emb4_14_U", "Parent" : "0"},
	{"ID" : "17", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.Emb4_15_U", "Parent" : "0"},
	{"ID" : "18", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.Emb4_16_U", "Parent" : "0"},
	{"ID" : "19", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.Emb4_17_U", "Parent" : "0"},
	{"ID" : "20", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.Emb4_18_U", "Parent" : "0"}]}


set ArgLastReadFirstWriteLatency {
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
	{"Name" : "Latency", "Min" : "57619", "Max" : "57619"}
	, {"Name" : "Interval", "Min" : "57619", "Max" : "57619"}
]}

set PipelineEnableSignalInfo {[
]}

set Spec2ImplPortList { 
	X { ap_memory {  { X_address0 mem_address 1 15 }  { X_ce0 mem_ce 1 1 }  { X_we0 mem_we 1 1 }  { X_d0 mem_din 1 16 } } }
}
