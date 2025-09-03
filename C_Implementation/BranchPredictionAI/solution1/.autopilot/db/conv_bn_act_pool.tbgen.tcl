set moduleName conv_bn_act_pool
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
set C_modelName {conv_bn_act_pool}
set C_modelType { void 0 }
set C_modelArgList {
	{ Tlen int 11 regular  }
	{ W float 32 regular {array 7168 { 1 3 } 1 1 }  }
	{ B float 32 regular {array 32 { 1 3 } 1 1 }  }
	{ C int 11 regular  }
	{ gamma float 32 regular {array 32 { 1 3 } 1 1 }  }
	{ beta float 32 regular {array 32 { 1 3 } 1 1 }  }
	{ mean float 32 regular {array 32 { 1 3 } 1 1 }  }
	{ var float 32 regular {array 32 { 1 3 } 1 1 }  }
	{ Psz int 7 regular  }
	{ U float 32 regular {array 384 { 0 3 } 0 1 }  }
	{ X_slice float 32 regular {array 18624 { 1 3 } 1 1 } {global 0}  }
}
set C_modelArgMapList {[ 
	{ "Name" : "Tlen", "interface" : "wire", "bitwidth" : 11, "direction" : "READONLY"} , 
 	{ "Name" : "W", "interface" : "memory", "bitwidth" : 32, "direction" : "READONLY"} , 
 	{ "Name" : "B", "interface" : "memory", "bitwidth" : 32, "direction" : "READONLY"} , 
 	{ "Name" : "C", "interface" : "wire", "bitwidth" : 11, "direction" : "READONLY"} , 
 	{ "Name" : "gamma", "interface" : "memory", "bitwidth" : 32, "direction" : "READONLY"} , 
 	{ "Name" : "beta", "interface" : "memory", "bitwidth" : 32, "direction" : "READONLY"} , 
 	{ "Name" : "mean", "interface" : "memory", "bitwidth" : 32, "direction" : "READONLY"} , 
 	{ "Name" : "var", "interface" : "memory", "bitwidth" : 32, "direction" : "READONLY"} , 
 	{ "Name" : "Psz", "interface" : "wire", "bitwidth" : 7, "direction" : "READONLY"} , 
 	{ "Name" : "U", "interface" : "memory", "bitwidth" : 32, "direction" : "WRITEONLY"} , 
 	{ "Name" : "X_slice", "interface" : "memory", "bitwidth" : 32, "direction" : "READONLY", "extern" : 0} ]}
# RTL Port declarations: 
set portNum 34
set portList { 
	{ ap_clk sc_in sc_logic 1 clock -1 } 
	{ ap_rst sc_in sc_logic 1 reset -1 active_high_sync } 
	{ ap_start sc_in sc_logic 1 start -1 } 
	{ ap_done sc_out sc_logic 1 predone -1 } 
	{ ap_idle sc_out sc_logic 1 done -1 } 
	{ ap_ready sc_out sc_logic 1 ready -1 } 
	{ Tlen sc_in sc_lv 11 signal 0 } 
	{ W_address0 sc_out sc_lv 13 signal 1 } 
	{ W_ce0 sc_out sc_logic 1 signal 1 } 
	{ W_q0 sc_in sc_lv 32 signal 1 } 
	{ B_address0 sc_out sc_lv 5 signal 2 } 
	{ B_ce0 sc_out sc_logic 1 signal 2 } 
	{ B_q0 sc_in sc_lv 32 signal 2 } 
	{ C sc_in sc_lv 11 signal 3 } 
	{ gamma_address0 sc_out sc_lv 5 signal 4 } 
	{ gamma_ce0 sc_out sc_logic 1 signal 4 } 
	{ gamma_q0 sc_in sc_lv 32 signal 4 } 
	{ beta_address0 sc_out sc_lv 5 signal 5 } 
	{ beta_ce0 sc_out sc_logic 1 signal 5 } 
	{ beta_q0 sc_in sc_lv 32 signal 5 } 
	{ mean_address0 sc_out sc_lv 5 signal 6 } 
	{ mean_ce0 sc_out sc_logic 1 signal 6 } 
	{ mean_q0 sc_in sc_lv 32 signal 6 } 
	{ var_address0 sc_out sc_lv 5 signal 7 } 
	{ var_ce0 sc_out sc_logic 1 signal 7 } 
	{ var_q0 sc_in sc_lv 32 signal 7 } 
	{ Psz sc_in sc_lv 7 signal 8 } 
	{ U_address0 sc_out sc_lv 9 signal 9 } 
	{ U_ce0 sc_out sc_logic 1 signal 9 } 
	{ U_we0 sc_out sc_logic 1 signal 9 } 
	{ U_d0 sc_out sc_lv 32 signal 9 } 
	{ X_slice_address0 sc_out sc_lv 15 signal 10 } 
	{ X_slice_ce0 sc_out sc_logic 1 signal 10 } 
	{ X_slice_q0 sc_in sc_lv 32 signal 10 } 
}
set NewPortList {[ 
	{ "name": "ap_clk", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "clock", "bundle":{"name": "ap_clk", "role": "default" }} , 
 	{ "name": "ap_rst", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "reset", "bundle":{"name": "ap_rst", "role": "default" }} , 
 	{ "name": "ap_start", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "start", "bundle":{"name": "ap_start", "role": "default" }} , 
 	{ "name": "ap_done", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "predone", "bundle":{"name": "ap_done", "role": "default" }} , 
 	{ "name": "ap_idle", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "done", "bundle":{"name": "ap_idle", "role": "default" }} , 
 	{ "name": "ap_ready", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "ready", "bundle":{"name": "ap_ready", "role": "default" }} , 
 	{ "name": "Tlen", "direction": "in", "datatype": "sc_lv", "bitwidth":11, "type": "signal", "bundle":{"name": "Tlen", "role": "default" }} , 
 	{ "name": "W_address0", "direction": "out", "datatype": "sc_lv", "bitwidth":13, "type": "signal", "bundle":{"name": "W", "role": "address0" }} , 
 	{ "name": "W_ce0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "W", "role": "ce0" }} , 
 	{ "name": "W_q0", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "W", "role": "q0" }} , 
 	{ "name": "B_address0", "direction": "out", "datatype": "sc_lv", "bitwidth":5, "type": "signal", "bundle":{"name": "B", "role": "address0" }} , 
 	{ "name": "B_ce0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "B", "role": "ce0" }} , 
 	{ "name": "B_q0", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "B", "role": "q0" }} , 
 	{ "name": "C", "direction": "in", "datatype": "sc_lv", "bitwidth":11, "type": "signal", "bundle":{"name": "C", "role": "default" }} , 
 	{ "name": "gamma_address0", "direction": "out", "datatype": "sc_lv", "bitwidth":5, "type": "signal", "bundle":{"name": "gamma", "role": "address0" }} , 
 	{ "name": "gamma_ce0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gamma", "role": "ce0" }} , 
 	{ "name": "gamma_q0", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "gamma", "role": "q0" }} , 
 	{ "name": "beta_address0", "direction": "out", "datatype": "sc_lv", "bitwidth":5, "type": "signal", "bundle":{"name": "beta", "role": "address0" }} , 
 	{ "name": "beta_ce0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "beta", "role": "ce0" }} , 
 	{ "name": "beta_q0", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "beta", "role": "q0" }} , 
 	{ "name": "mean_address0", "direction": "out", "datatype": "sc_lv", "bitwidth":5, "type": "signal", "bundle":{"name": "mean", "role": "address0" }} , 
 	{ "name": "mean_ce0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "mean", "role": "ce0" }} , 
 	{ "name": "mean_q0", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "mean", "role": "q0" }} , 
 	{ "name": "var_address0", "direction": "out", "datatype": "sc_lv", "bitwidth":5, "type": "signal", "bundle":{"name": "var", "role": "address0" }} , 
 	{ "name": "var_ce0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "var", "role": "ce0" }} , 
 	{ "name": "var_q0", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "var", "role": "q0" }} , 
 	{ "name": "Psz", "direction": "in", "datatype": "sc_lv", "bitwidth":7, "type": "signal", "bundle":{"name": "Psz", "role": "default" }} , 
 	{ "name": "U_address0", "direction": "out", "datatype": "sc_lv", "bitwidth":9, "type": "signal", "bundle":{"name": "U", "role": "address0" }} , 
 	{ "name": "U_ce0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "U", "role": "ce0" }} , 
 	{ "name": "U_we0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "U", "role": "we0" }} , 
 	{ "name": "U_d0", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "U", "role": "d0" }} , 
 	{ "name": "X_slice_address0", "direction": "out", "datatype": "sc_lv", "bitwidth":15, "type": "signal", "bundle":{"name": "X_slice", "role": "address0" }} , 
 	{ "name": "X_slice_ce0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "X_slice", "role": "ce0" }} , 
 	{ "name": "X_slice_q0", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "X_slice", "role": "q0" }}  ]}

set RtlHierarchyInfo {[
	{"ID" : "0", "Level" : "0", "Path" : "`AUTOTB_DUT_INST", "Parent" : "", "Child" : ["1", "2", "3", "4", "5", "6", "7", "8", "9"],
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
	{"ID" : "1", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.Y_U", "Parent" : "0"},
	{"ID" : "2", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.main_faddfsub_32ns_32ns_32_5_full_dsp_1_U1", "Parent" : "0"},
	{"ID" : "3", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.main_fmul_32ns_32ns_32_4_max_dsp_1_U2", "Parent" : "0"},
	{"ID" : "4", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.main_fdiv_32ns_32ns_32_16_1_U3", "Parent" : "0"},
	{"ID" : "5", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.main_sitofp_32s_32_6_1_U4", "Parent" : "0"},
	{"ID" : "6", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.main_fcmp_32ns_32ns_1_2_1_U5", "Parent" : "0"},
	{"ID" : "7", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.main_fsqrt_32ns_32ns_32_12_1_U6", "Parent" : "0"},
	{"ID" : "8", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.main_udiv_10ns_7ns_9_14_seq_1_U7", "Parent" : "0"},
	{"ID" : "9", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.main_mac_muladd_11ns_14ns_10ns_23_1_1_U8", "Parent" : "0"}]}


set ArgLastReadFirstWriteLatency {
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
	{"Name" : "Latency", "Min" : "2932457", "Max" : "107081133"}
	, {"Name" : "Interval", "Min" : "2932457", "Max" : "107081133"}
]}

set PipelineEnableSignalInfo {[
]}

set Spec2ImplPortList { 
	Tlen { ap_none {  { Tlen in_data 0 11 } } }
	W { ap_memory {  { W_address0 mem_address 1 13 }  { W_ce0 mem_ce 1 1 }  { W_q0 mem_dout 0 32 } } }
	B { ap_memory {  { B_address0 mem_address 1 5 }  { B_ce0 mem_ce 1 1 }  { B_q0 mem_dout 0 32 } } }
	C { ap_none {  { C in_data 0 11 } } }
	gamma { ap_memory {  { gamma_address0 mem_address 1 5 }  { gamma_ce0 mem_ce 1 1 }  { gamma_q0 mem_dout 0 32 } } }
	beta { ap_memory {  { beta_address0 mem_address 1 5 }  { beta_ce0 mem_ce 1 1 }  { beta_q0 mem_dout 0 32 } } }
	mean { ap_memory {  { mean_address0 mem_address 1 5 }  { mean_ce0 mem_ce 1 1 }  { mean_q0 mem_dout 0 32 } } }
	var { ap_memory {  { var_address0 mem_address 1 5 }  { var_ce0 mem_ce 1 1 }  { var_q0 mem_dout 0 32 } } }
	Psz { ap_none {  { Psz in_data 0 7 } } }
	U { ap_memory {  { U_address0 mem_address 1 9 }  { U_ce0 mem_ce 1 1 }  { U_we0 mem_we 1 1 }  { U_d0 mem_din 1 32 } } }
	X_slice { ap_memory {  { X_slice_address0 mem_address 1 15 }  { X_slice_ce0 mem_ce 1 1 }  { X_slice_q0 mem_dout 0 32 } } }
}
