set moduleName conv_bn_act_pool
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
set C_modelName {conv_bn_act_pool}
set C_modelType { void 0 }
set ap_memory_interface_dict [dict create]
dict set ap_memory_interface_dict U_slice { MEM_WIDTH 16 MEM_SIZE 768 MASTER_TYPE BRAM_CTRL MEM_ADDRESS_MODE WORD_ADDRESS PACKAGE_IO port READ_LATENCY 1 }
set C_modelArgList {
	{ U_slice int 16 regular {array 384 { 0 3 } 1 1 } {global 1}  }
}
set hasAXIMCache 0
set l_AXIML2Cache [list]
set AXIMCacheInstDict [dict create]
set C_modelArgMapList {[ 
	{ "Name" : "U_slice", "interface" : "memory", "bitwidth" : 16, "direction" : "WRITEONLY", "extern" : 0} ]}
# RTL Port declarations: 
set portNum 65
set portList { 
	{ ap_clk sc_in sc_logic 1 clock -1 } 
	{ ap_rst sc_in sc_logic 1 reset -1 active_high_sync } 
	{ ap_start sc_in sc_logic 1 start -1 } 
	{ ap_done sc_out sc_logic 1 predone -1 } 
	{ ap_idle sc_out sc_logic 1 done -1 } 
	{ ap_ready sc_out sc_logic 1 ready -1 } 
	{ U_slice_address0 sc_out sc_lv 9 signal 0 } 
	{ U_slice_ce0 sc_out sc_logic 1 signal 0 } 
	{ U_slice_we0 sc_out sc_logic 1 signal 0 } 
	{ U_slice_d0 sc_out sc_lv 16 signal 0 } 
	{ grp_fu_295_p_din0 sc_out sc_lv 32 signal -1 } 
	{ grp_fu_295_p_din1 sc_out sc_lv 32 signal -1 } 
	{ grp_fu_295_p_opcode sc_out sc_lv 2 signal -1 } 
	{ grp_fu_295_p_dout0 sc_in sc_lv 32 signal -1 } 
	{ grp_fu_295_p_ce sc_out sc_logic 1 signal -1 } 
	{ grp_fu_299_p_din0 sc_out sc_lv 32 signal -1 } 
	{ grp_fu_299_p_din1 sc_out sc_lv 32 signal -1 } 
	{ grp_fu_299_p_dout0 sc_in sc_lv 32 signal -1 } 
	{ grp_fu_299_p_ce sc_out sc_logic 1 signal -1 } 
	{ grp_fu_303_p_din0 sc_out sc_lv 32 signal -1 } 
	{ grp_fu_303_p_din1 sc_out sc_lv 32 signal -1 } 
	{ grp_fu_303_p_dout0 sc_in sc_lv 32 signal -1 } 
	{ grp_fu_303_p_ce sc_out sc_logic 1 signal -1 } 
	{ grp_fu_307_p_din0 sc_out sc_lv 32 signal -1 } 
	{ grp_fu_307_p_din1 sc_out sc_lv 32 signal -1 } 
	{ grp_fu_307_p_dout0 sc_in sc_lv 32 signal -1 } 
	{ grp_fu_307_p_ce sc_out sc_logic 1 signal -1 } 
	{ grp_fu_311_p_din0 sc_out sc_lv 32 signal -1 } 
	{ grp_fu_311_p_dout0 sc_in sc_lv 16 signal -1 } 
	{ grp_fu_311_p_ce sc_out sc_logic 1 signal -1 } 
	{ grp_fu_314_p_din0 sc_out sc_lv 16 signal -1 } 
	{ grp_fu_314_p_din1 sc_out sc_lv 16 signal -1 } 
	{ grp_fu_314_p_dout0 sc_in sc_lv 16 signal -1 } 
	{ grp_fu_314_p_ce sc_out sc_logic 1 signal -1 } 
	{ grp_fu_318_p_din0 sc_out sc_lv 32 signal -1 } 
	{ grp_fu_318_p_din1 sc_out sc_lv 32 signal -1 } 
	{ grp_fu_318_p_opcode sc_out sc_lv 2 signal -1 } 
	{ grp_fu_318_p_dout0 sc_in sc_lv 32 signal -1 } 
	{ grp_fu_318_p_ce sc_out sc_logic 1 signal -1 } 
	{ grp_fu_322_p_din0 sc_out sc_lv 32 signal -1 } 
	{ grp_fu_322_p_din1 sc_out sc_lv 32 signal -1 } 
	{ grp_fu_322_p_dout0 sc_in sc_lv 32 signal -1 } 
	{ grp_fu_322_p_ce sc_out sc_logic 1 signal -1 } 
	{ grp_fu_326_p_din0 sc_out sc_lv 32 signal -1 } 
	{ grp_fu_326_p_dout0 sc_in sc_lv 16 signal -1 } 
	{ grp_fu_326_p_ce sc_out sc_logic 1 signal -1 } 
	{ grp_fu_329_p_din0 sc_out sc_lv 16 signal -1 } 
	{ grp_fu_329_p_dout0 sc_in sc_lv 32 signal -1 } 
	{ grp_fu_329_p_ce sc_out sc_logic 1 signal -1 } 
	{ grp_fu_332_p_din0 sc_out sc_lv 16 signal -1 } 
	{ grp_fu_332_p_dout0 sc_in sc_lv 32 signal -1 } 
	{ grp_fu_332_p_ce sc_out sc_logic 1 signal -1 } 
	{ grp_fu_283_p_din0 sc_out sc_lv 16 signal -1 } 
	{ grp_fu_283_p_din1 sc_out sc_lv 16 signal -1 } 
	{ grp_fu_283_p_dout0 sc_in sc_lv 16 signal -1 } 
	{ grp_fu_283_p_ce sc_out sc_logic 1 signal -1 } 
	{ grp_fu_335_p_din0 sc_out sc_lv 16 signal -1 } 
	{ grp_fu_335_p_din1 sc_out sc_lv 16 signal -1 } 
	{ grp_fu_335_p_dout0 sc_in sc_lv 16 signal -1 } 
	{ grp_fu_335_p_ce sc_out sc_logic 1 signal -1 } 
	{ grp_fu_287_p_din0 sc_out sc_lv 16 signal -1 } 
	{ grp_fu_287_p_din1 sc_out sc_lv 16 signal -1 } 
	{ grp_fu_287_p_opcode sc_out sc_lv 5 signal -1 } 
	{ grp_fu_287_p_dout0 sc_in sc_lv 1 signal -1 } 
	{ grp_fu_287_p_ce sc_out sc_logic 1 signal -1 } 
}
set NewPortList {[ 
	{ "name": "ap_clk", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "clock", "bundle":{"name": "ap_clk", "role": "default" }} , 
 	{ "name": "ap_rst", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "reset", "bundle":{"name": "ap_rst", "role": "default" }} , 
 	{ "name": "ap_start", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "start", "bundle":{"name": "ap_start", "role": "default" }} , 
 	{ "name": "ap_done", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "predone", "bundle":{"name": "ap_done", "role": "default" }} , 
 	{ "name": "ap_idle", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "done", "bundle":{"name": "ap_idle", "role": "default" }} , 
 	{ "name": "ap_ready", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "ready", "bundle":{"name": "ap_ready", "role": "default" }} , 
 	{ "name": "U_slice_address0", "direction": "out", "datatype": "sc_lv", "bitwidth":9, "type": "signal", "bundle":{"name": "U_slice", "role": "address0" }} , 
 	{ "name": "U_slice_ce0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "U_slice", "role": "ce0" }} , 
 	{ "name": "U_slice_we0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "U_slice", "role": "we0" }} , 
 	{ "name": "U_slice_d0", "direction": "out", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "U_slice", "role": "d0" }} , 
 	{ "name": "grp_fu_295_p_din0", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "grp_fu_295_p_din0", "role": "default" }} , 
 	{ "name": "grp_fu_295_p_din1", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "grp_fu_295_p_din1", "role": "default" }} , 
 	{ "name": "grp_fu_295_p_opcode", "direction": "out", "datatype": "sc_lv", "bitwidth":2, "type": "signal", "bundle":{"name": "grp_fu_295_p_opcode", "role": "default" }} , 
 	{ "name": "grp_fu_295_p_dout0", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "grp_fu_295_p_dout0", "role": "default" }} , 
 	{ "name": "grp_fu_295_p_ce", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "grp_fu_295_p_ce", "role": "default" }} , 
 	{ "name": "grp_fu_299_p_din0", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "grp_fu_299_p_din0", "role": "default" }} , 
 	{ "name": "grp_fu_299_p_din1", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "grp_fu_299_p_din1", "role": "default" }} , 
 	{ "name": "grp_fu_299_p_dout0", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "grp_fu_299_p_dout0", "role": "default" }} , 
 	{ "name": "grp_fu_299_p_ce", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "grp_fu_299_p_ce", "role": "default" }} , 
 	{ "name": "grp_fu_303_p_din0", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "grp_fu_303_p_din0", "role": "default" }} , 
 	{ "name": "grp_fu_303_p_din1", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "grp_fu_303_p_din1", "role": "default" }} , 
 	{ "name": "grp_fu_303_p_dout0", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "grp_fu_303_p_dout0", "role": "default" }} , 
 	{ "name": "grp_fu_303_p_ce", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "grp_fu_303_p_ce", "role": "default" }} , 
 	{ "name": "grp_fu_307_p_din0", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "grp_fu_307_p_din0", "role": "default" }} , 
 	{ "name": "grp_fu_307_p_din1", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "grp_fu_307_p_din1", "role": "default" }} , 
 	{ "name": "grp_fu_307_p_dout0", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "grp_fu_307_p_dout0", "role": "default" }} , 
 	{ "name": "grp_fu_307_p_ce", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "grp_fu_307_p_ce", "role": "default" }} , 
 	{ "name": "grp_fu_311_p_din0", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "grp_fu_311_p_din0", "role": "default" }} , 
 	{ "name": "grp_fu_311_p_dout0", "direction": "in", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "grp_fu_311_p_dout0", "role": "default" }} , 
 	{ "name": "grp_fu_311_p_ce", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "grp_fu_311_p_ce", "role": "default" }} , 
 	{ "name": "grp_fu_314_p_din0", "direction": "out", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "grp_fu_314_p_din0", "role": "default" }} , 
 	{ "name": "grp_fu_314_p_din1", "direction": "out", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "grp_fu_314_p_din1", "role": "default" }} , 
 	{ "name": "grp_fu_314_p_dout0", "direction": "in", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "grp_fu_314_p_dout0", "role": "default" }} , 
 	{ "name": "grp_fu_314_p_ce", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "grp_fu_314_p_ce", "role": "default" }} , 
 	{ "name": "grp_fu_318_p_din0", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "grp_fu_318_p_din0", "role": "default" }} , 
 	{ "name": "grp_fu_318_p_din1", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "grp_fu_318_p_din1", "role": "default" }} , 
 	{ "name": "grp_fu_318_p_opcode", "direction": "out", "datatype": "sc_lv", "bitwidth":2, "type": "signal", "bundle":{"name": "grp_fu_318_p_opcode", "role": "default" }} , 
 	{ "name": "grp_fu_318_p_dout0", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "grp_fu_318_p_dout0", "role": "default" }} , 
 	{ "name": "grp_fu_318_p_ce", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "grp_fu_318_p_ce", "role": "default" }} , 
 	{ "name": "grp_fu_322_p_din0", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "grp_fu_322_p_din0", "role": "default" }} , 
 	{ "name": "grp_fu_322_p_din1", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "grp_fu_322_p_din1", "role": "default" }} , 
 	{ "name": "grp_fu_322_p_dout0", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "grp_fu_322_p_dout0", "role": "default" }} , 
 	{ "name": "grp_fu_322_p_ce", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "grp_fu_322_p_ce", "role": "default" }} , 
 	{ "name": "grp_fu_326_p_din0", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "grp_fu_326_p_din0", "role": "default" }} , 
 	{ "name": "grp_fu_326_p_dout0", "direction": "in", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "grp_fu_326_p_dout0", "role": "default" }} , 
 	{ "name": "grp_fu_326_p_ce", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "grp_fu_326_p_ce", "role": "default" }} , 
 	{ "name": "grp_fu_329_p_din0", "direction": "out", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "grp_fu_329_p_din0", "role": "default" }} , 
 	{ "name": "grp_fu_329_p_dout0", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "grp_fu_329_p_dout0", "role": "default" }} , 
 	{ "name": "grp_fu_329_p_ce", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "grp_fu_329_p_ce", "role": "default" }} , 
 	{ "name": "grp_fu_332_p_din0", "direction": "out", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "grp_fu_332_p_din0", "role": "default" }} , 
 	{ "name": "grp_fu_332_p_dout0", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "grp_fu_332_p_dout0", "role": "default" }} , 
 	{ "name": "grp_fu_332_p_ce", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "grp_fu_332_p_ce", "role": "default" }} , 
 	{ "name": "grp_fu_283_p_din0", "direction": "out", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "grp_fu_283_p_din0", "role": "default" }} , 
 	{ "name": "grp_fu_283_p_din1", "direction": "out", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "grp_fu_283_p_din1", "role": "default" }} , 
 	{ "name": "grp_fu_283_p_dout0", "direction": "in", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "grp_fu_283_p_dout0", "role": "default" }} , 
 	{ "name": "grp_fu_283_p_ce", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "grp_fu_283_p_ce", "role": "default" }} , 
 	{ "name": "grp_fu_335_p_din0", "direction": "out", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "grp_fu_335_p_din0", "role": "default" }} , 
 	{ "name": "grp_fu_335_p_din1", "direction": "out", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "grp_fu_335_p_din1", "role": "default" }} , 
 	{ "name": "grp_fu_335_p_dout0", "direction": "in", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "grp_fu_335_p_dout0", "role": "default" }} , 
 	{ "name": "grp_fu_335_p_ce", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "grp_fu_335_p_ce", "role": "default" }} , 
 	{ "name": "grp_fu_287_p_din0", "direction": "out", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "grp_fu_287_p_din0", "role": "default" }} , 
 	{ "name": "grp_fu_287_p_din1", "direction": "out", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "grp_fu_287_p_din1", "role": "default" }} , 
 	{ "name": "grp_fu_287_p_opcode", "direction": "out", "datatype": "sc_lv", "bitwidth":5, "type": "signal", "bundle":{"name": "grp_fu_287_p_opcode", "role": "default" }} , 
 	{ "name": "grp_fu_287_p_dout0", "direction": "in", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "grp_fu_287_p_dout0", "role": "default" }} , 
 	{ "name": "grp_fu_287_p_ce", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "grp_fu_287_p_ce", "role": "default" }}  ]}

set RtlHierarchyInfo {[
	{"ID" : "0", "Level" : "0", "Path" : "`AUTOTB_DUT_INST", "Parent" : "", "Child" : ["1", "2", "3", "7", "10", "11", "12"],
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
					{"ID" : "7", "SubInstance" : "grp_conv_bn_act_pool_Pipeline_Loop3_2Big_fu_717", "Port" : "U_slice", "Inst_start_state" : "5", "Inst_end_state" : "40"}]}],
		"Loop" : [
			{"Name" : "Loop3_1Big_Loop4Big", "PipelineType" : "pipeline",
				"LoopDec" : {"FSMBitwidth" : "33", "FirstState" : "ap_ST_fsm_pp0_stage0", "FirstStateIter" : "ap_enable_reg_pp0_iter0", "FirstStateBlock" : "ap_block_pp0_stage0_subdone", "LastState" : "ap_ST_fsm_pp0_stage6", "LastStateIter" : "ap_enable_reg_pp0_iter1", "LastStateBlock" : "ap_block_pp0_stage6_subdone", "PreState" : ["ap_ST_fsm_state5"], "QuitState" : "ap_ST_fsm_pp0_stage6", "QuitStateIter" : "ap_enable_reg_pp0_iter1", "QuitStateBlock" : "ap_block_pp0_stage6_subdone", "PostState" : ["ap_ST_fsm_state23"]}},
			{"Name" : "Loop2Big", "PipelineType" : "no",
				"LoopDec" : {"FSMBitwidth" : "33", "FirstState" : "ap_ST_fsm_state5", "LastState" : ["ap_ST_fsm_state39"], "QuitState" : ["ap_ST_fsm_state5"], "PreState" : ["ap_ST_fsm_state4"], "PostState" : ["ap_ST_fsm_state40"], "OneDepthLoop" : "0", "OneStateBlock": ""}},
			{"Name" : "Loop1Big", "PipelineType" : "no",
				"LoopDec" : {"FSMBitwidth" : "33", "FirstState" : "ap_ST_fsm_state4", "LastState" : ["ap_ST_fsm_state40"], "QuitState" : ["ap_ST_fsm_state4"], "PreState" : ["ap_ST_fsm_state3"], "PostState" : ["ap_ST_fsm_state1"], "OneDepthLoop" : "0", "OneStateBlock": ""}}]},
	{"ID" : "1", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.X_slice12_U", "Parent" : "0"},
	{"ID" : "2", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.ConvW1_U", "Parent" : "0"},
	{"ID" : "3", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.grp_conv_bn_act_pool_Pipeline_BNParamsLoop_fu_649", "Parent" : "0", "Child" : ["4", "5", "6"],
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
	{"ID" : "4", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_bn_act_pool_Pipeline_BNParamsLoop_fu_649.sparsemux_65_5_16_1_1_U111", "Parent" : "3"},
	{"ID" : "5", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_bn_act_pool_Pipeline_BNParamsLoop_fu_649.sparsemux_65_5_16_1_1_U112", "Parent" : "3"},
	{"ID" : "6", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_bn_act_pool_Pipeline_BNParamsLoop_fu_649.flow_control_loop_pipe_sequential_init_U", "Parent" : "3"},
	{"ID" : "7", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.grp_conv_bn_act_pool_Pipeline_Loop3_2Big_fu_717", "Parent" : "0", "Child" : ["8", "9"],
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
	{"ID" : "8", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_bn_act_pool_Pipeline_Loop3_2Big_fu_717.sparsemux_65_5_16_1_1_U179", "Parent" : "7"},
	{"ID" : "9", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_bn_act_pool_Pipeline_Loop3_2Big_fu_717.flow_control_loop_pipe_sequential_init_U", "Parent" : "7"},
	{"ID" : "10", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.sparsemux_65_5_16_1_1_U222", "Parent" : "0"},
	{"ID" : "11", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.sparsemux_65_5_16_1_1_U223", "Parent" : "0"},
	{"ID" : "12", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.sparsemux_65_5_16_1_1_U224", "Parent" : "0"}]}


set ArgLastReadFirstWriteLatency {
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
		U_slice {Type O LastRead -1 FirstWrite 7}}}

set hasDtUnsupportedChannel 0

set PerformanceInfo {[
	{"Name" : "Latency", "Min" : "2638369", "Max" : "2641321"}
	, {"Name" : "Interval", "Min" : "2638369", "Max" : "2641321"}
]}

set PipelineEnableSignalInfo {[
	{"Pipeline" : "0", "EnableSignal" : "ap_enable_pp0"}
]}

set Spec2ImplPortList { 
	U_slice { ap_memory {  { U_slice_address0 mem_address 1 9 }  { U_slice_ce0 mem_ce 1 1 }  { U_slice_we0 mem_we 1 1 }  { U_slice_d0 mem_din 1 16 } } }
}
