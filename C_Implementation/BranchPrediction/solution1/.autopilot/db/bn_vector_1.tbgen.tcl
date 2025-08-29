set moduleName bn_vector_1
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
set C_modelName {bn_vector.1}
set C_modelType { void 0 }
set C_modelArgList {
	{ v float 32 regular {array 128 { 2 3 } 1 1 }  }
	{ gamma float 32 regular {array 128 { 1 3 } 1 1 }  }
	{ beta float 32 regular {array 128 { 1 3 } 1 1 }  }
	{ mean float 32 regular {array 128 { 1 3 } 1 1 }  }
	{ var float 32 regular {array 128 { 1 3 } 1 1 }  }
}
set C_modelArgMapList {[ 
	{ "Name" : "v", "interface" : "memory", "bitwidth" : 32, "direction" : "READWRITE"} , 
 	{ "Name" : "gamma", "interface" : "memory", "bitwidth" : 32, "direction" : "READONLY"} , 
 	{ "Name" : "beta", "interface" : "memory", "bitwidth" : 32, "direction" : "READONLY"} , 
 	{ "Name" : "mean", "interface" : "memory", "bitwidth" : 32, "direction" : "READONLY"} , 
 	{ "Name" : "var", "interface" : "memory", "bitwidth" : 32, "direction" : "READONLY"} ]}
# RTL Port declarations: 
set portNum 23
set portList { 
	{ ap_clk sc_in sc_logic 1 clock -1 } 
	{ ap_rst sc_in sc_logic 1 reset -1 active_high_sync } 
	{ ap_start sc_in sc_logic 1 start -1 } 
	{ ap_done sc_out sc_logic 1 predone -1 } 
	{ ap_idle sc_out sc_logic 1 done -1 } 
	{ ap_ready sc_out sc_logic 1 ready -1 } 
	{ v_address0 sc_out sc_lv 7 signal 0 } 
	{ v_ce0 sc_out sc_logic 1 signal 0 } 
	{ v_we0 sc_out sc_logic 1 signal 0 } 
	{ v_d0 sc_out sc_lv 32 signal 0 } 
	{ v_q0 sc_in sc_lv 32 signal 0 } 
	{ gamma_address0 sc_out sc_lv 7 signal 1 } 
	{ gamma_ce0 sc_out sc_logic 1 signal 1 } 
	{ gamma_q0 sc_in sc_lv 32 signal 1 } 
	{ beta_address0 sc_out sc_lv 7 signal 2 } 
	{ beta_ce0 sc_out sc_logic 1 signal 2 } 
	{ beta_q0 sc_in sc_lv 32 signal 2 } 
	{ mean_address0 sc_out sc_lv 7 signal 3 } 
	{ mean_ce0 sc_out sc_logic 1 signal 3 } 
	{ mean_q0 sc_in sc_lv 32 signal 3 } 
	{ var_address0 sc_out sc_lv 7 signal 4 } 
	{ var_ce0 sc_out sc_logic 1 signal 4 } 
	{ var_q0 sc_in sc_lv 32 signal 4 } 
}
set NewPortList {[ 
	{ "name": "ap_clk", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "clock", "bundle":{"name": "ap_clk", "role": "default" }} , 
 	{ "name": "ap_rst", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "reset", "bundle":{"name": "ap_rst", "role": "default" }} , 
 	{ "name": "ap_start", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "start", "bundle":{"name": "ap_start", "role": "default" }} , 
 	{ "name": "ap_done", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "predone", "bundle":{"name": "ap_done", "role": "default" }} , 
 	{ "name": "ap_idle", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "done", "bundle":{"name": "ap_idle", "role": "default" }} , 
 	{ "name": "ap_ready", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "ready", "bundle":{"name": "ap_ready", "role": "default" }} , 
 	{ "name": "v_address0", "direction": "out", "datatype": "sc_lv", "bitwidth":7, "type": "signal", "bundle":{"name": "v", "role": "address0" }} , 
 	{ "name": "v_ce0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "v", "role": "ce0" }} , 
 	{ "name": "v_we0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "v", "role": "we0" }} , 
 	{ "name": "v_d0", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "v", "role": "d0" }} , 
 	{ "name": "v_q0", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "v", "role": "q0" }} , 
 	{ "name": "gamma_address0", "direction": "out", "datatype": "sc_lv", "bitwidth":7, "type": "signal", "bundle":{"name": "gamma", "role": "address0" }} , 
 	{ "name": "gamma_ce0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gamma", "role": "ce0" }} , 
 	{ "name": "gamma_q0", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "gamma", "role": "q0" }} , 
 	{ "name": "beta_address0", "direction": "out", "datatype": "sc_lv", "bitwidth":7, "type": "signal", "bundle":{"name": "beta", "role": "address0" }} , 
 	{ "name": "beta_ce0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "beta", "role": "ce0" }} , 
 	{ "name": "beta_q0", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "beta", "role": "q0" }} , 
 	{ "name": "mean_address0", "direction": "out", "datatype": "sc_lv", "bitwidth":7, "type": "signal", "bundle":{"name": "mean", "role": "address0" }} , 
 	{ "name": "mean_ce0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "mean", "role": "ce0" }} , 
 	{ "name": "mean_q0", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "mean", "role": "q0" }} , 
 	{ "name": "var_address0", "direction": "out", "datatype": "sc_lv", "bitwidth":7, "type": "signal", "bundle":{"name": "var", "role": "address0" }} , 
 	{ "name": "var_ce0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "var", "role": "ce0" }} , 
 	{ "name": "var_q0", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "var", "role": "q0" }}  ]}

set RtlHierarchyInfo {[
	{"ID" : "0", "Level" : "0", "Path" : "`AUTOTB_DUT_INST", "Parent" : "", "Child" : ["1", "2", "3", "4"],
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
	{"ID" : "1", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.main_faddfsub_32ns_32ns_32_5_full_dsp_1_U132", "Parent" : "0"},
	{"ID" : "2", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.main_fmul_32ns_32ns_32_4_max_dsp_1_U133", "Parent" : "0"},
	{"ID" : "3", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.main_fdiv_32ns_32ns_32_16_1_U134", "Parent" : "0"},
	{"ID" : "4", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.main_fsqrt_32ns_32ns_32_12_1_U135", "Parent" : "0"}]}


set ArgLastReadFirstWriteLatency {
	bn_vector_1 {
		v {Type IO LastRead 13 FirstWrite 45}
		gamma {Type I LastRead 34 FirstWrite -1}
		beta {Type I LastRead 34 FirstWrite -1}
		mean {Type I LastRead 13 FirstWrite -1}
		var {Type I LastRead 1 FirstWrite -1}}}

set hasDtUnsupportedChannel 0

set PerformanceInfo {[
	{"Name" : "Latency", "Min" : "5761", "Max" : "5761"}
	, {"Name" : "Interval", "Min" : "5761", "Max" : "5761"}
]}

set PipelineEnableSignalInfo {[
]}

set Spec2ImplPortList { 
	v { ap_memory {  { v_address0 mem_address 1 7 }  { v_ce0 mem_ce 1 1 }  { v_we0 mem_we 1 1 }  { v_d0 mem_din 1 32 }  { v_q0 mem_dout 0 32 } } }
	gamma { ap_memory {  { gamma_address0 mem_address 1 7 }  { gamma_ce0 mem_ce 1 1 }  { gamma_q0 mem_dout 0 32 } } }
	beta { ap_memory {  { beta_address0 mem_address 1 7 }  { beta_ce0 mem_ce 1 1 }  { beta_q0 mem_dout 0 32 } } }
	mean { ap_memory {  { mean_address0 mem_address 1 7 }  { mean_ce0 mem_ce 1 1 }  { mean_q0 mem_dout 0 32 } } }
	var { ap_memory {  { var_address0 mem_address 1 7 }  { var_ce0 mem_ce 1 1 }  { var_q0 mem_dout 0 32 } } }
}
