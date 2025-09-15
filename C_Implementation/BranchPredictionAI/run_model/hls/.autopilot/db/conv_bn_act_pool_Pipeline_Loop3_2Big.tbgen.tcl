set moduleName conv_bn_act_pool_Pipeline_Loop3_2Big
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
set C_modelName {conv_bn_act_pool_Pipeline_Loop3_2Big}
set C_modelType { void 0 }
set ap_memory_interface_dict [dict create]
dict set ap_memory_interface_dict U_slice { MEM_WIDTH 16 MEM_SIZE 768 MASTER_TYPE BRAM_CTRL MEM_ADDRESS_MODE WORD_ADDRESS PACKAGE_IO port READ_LATENCY 1 }
set C_modelArgList {
	{ pool_acc int 16 regular {pointer 2}  }
	{ pool_acc_31 int 16 regular {pointer 2}  }
	{ pool_acc_30 int 16 regular {pointer 2}  }
	{ pool_acc_29 int 16 regular {pointer 2}  }
	{ pool_acc_28 int 16 regular {pointer 2}  }
	{ pool_acc_27 int 16 regular {pointer 2}  }
	{ pool_acc_26 int 16 regular {pointer 2}  }
	{ pool_acc_25 int 16 regular {pointer 2}  }
	{ pool_acc_24 int 16 regular {pointer 2}  }
	{ pool_acc_23 int 16 regular {pointer 2}  }
	{ pool_acc_22 int 16 regular {pointer 2}  }
	{ pool_acc_21 int 16 regular {pointer 2}  }
	{ pool_acc_20 int 16 regular {pointer 2}  }
	{ pool_acc_19 int 16 regular {pointer 2}  }
	{ pool_acc_18 int 16 regular {pointer 2}  }
	{ pool_acc_17 int 16 regular {pointer 2}  }
	{ pool_acc_16 int 16 regular {pointer 2}  }
	{ pool_acc_15 int 16 regular {pointer 2}  }
	{ pool_acc_14 int 16 regular {pointer 2}  }
	{ pool_acc_13 int 16 regular {pointer 2}  }
	{ pool_acc_12 int 16 regular {pointer 2}  }
	{ pool_acc_11 int 16 regular {pointer 2}  }
	{ pool_acc_10 int 16 regular {pointer 2}  }
	{ pool_acc_9 int 16 regular {pointer 2}  }
	{ pool_acc_8 int 16 regular {pointer 2}  }
	{ pool_acc_7 int 16 regular {pointer 2}  }
	{ pool_acc_6 int 16 regular {pointer 2}  }
	{ pool_acc_5 int 16 regular {pointer 2}  }
	{ pool_acc_4 int 16 regular {pointer 2}  }
	{ pool_acc_3 int 16 regular {pointer 2}  }
	{ pool_acc_2 int 16 regular {pointer 2}  }
	{ pool_acc_1 int 16 regular {pointer 2}  }
	{ mul4 int 9 regular  }
	{ U_slice int 16 regular {array 384 { 0 3 } 1 1 } {global 1}  }
}
set hasAXIMCache 0
set l_AXIML2Cache [list]
set AXIMCacheInstDict [dict create]
set C_modelArgMapList {[ 
	{ "Name" : "pool_acc", "interface" : "wire", "bitwidth" : 16, "direction" : "READWRITE"} , 
 	{ "Name" : "pool_acc_31", "interface" : "wire", "bitwidth" : 16, "direction" : "READWRITE"} , 
 	{ "Name" : "pool_acc_30", "interface" : "wire", "bitwidth" : 16, "direction" : "READWRITE"} , 
 	{ "Name" : "pool_acc_29", "interface" : "wire", "bitwidth" : 16, "direction" : "READWRITE"} , 
 	{ "Name" : "pool_acc_28", "interface" : "wire", "bitwidth" : 16, "direction" : "READWRITE"} , 
 	{ "Name" : "pool_acc_27", "interface" : "wire", "bitwidth" : 16, "direction" : "READWRITE"} , 
 	{ "Name" : "pool_acc_26", "interface" : "wire", "bitwidth" : 16, "direction" : "READWRITE"} , 
 	{ "Name" : "pool_acc_25", "interface" : "wire", "bitwidth" : 16, "direction" : "READWRITE"} , 
 	{ "Name" : "pool_acc_24", "interface" : "wire", "bitwidth" : 16, "direction" : "READWRITE"} , 
 	{ "Name" : "pool_acc_23", "interface" : "wire", "bitwidth" : 16, "direction" : "READWRITE"} , 
 	{ "Name" : "pool_acc_22", "interface" : "wire", "bitwidth" : 16, "direction" : "READWRITE"} , 
 	{ "Name" : "pool_acc_21", "interface" : "wire", "bitwidth" : 16, "direction" : "READWRITE"} , 
 	{ "Name" : "pool_acc_20", "interface" : "wire", "bitwidth" : 16, "direction" : "READWRITE"} , 
 	{ "Name" : "pool_acc_19", "interface" : "wire", "bitwidth" : 16, "direction" : "READWRITE"} , 
 	{ "Name" : "pool_acc_18", "interface" : "wire", "bitwidth" : 16, "direction" : "READWRITE"} , 
 	{ "Name" : "pool_acc_17", "interface" : "wire", "bitwidth" : 16, "direction" : "READWRITE"} , 
 	{ "Name" : "pool_acc_16", "interface" : "wire", "bitwidth" : 16, "direction" : "READWRITE"} , 
 	{ "Name" : "pool_acc_15", "interface" : "wire", "bitwidth" : 16, "direction" : "READWRITE"} , 
 	{ "Name" : "pool_acc_14", "interface" : "wire", "bitwidth" : 16, "direction" : "READWRITE"} , 
 	{ "Name" : "pool_acc_13", "interface" : "wire", "bitwidth" : 16, "direction" : "READWRITE"} , 
 	{ "Name" : "pool_acc_12", "interface" : "wire", "bitwidth" : 16, "direction" : "READWRITE"} , 
 	{ "Name" : "pool_acc_11", "interface" : "wire", "bitwidth" : 16, "direction" : "READWRITE"} , 
 	{ "Name" : "pool_acc_10", "interface" : "wire", "bitwidth" : 16, "direction" : "READWRITE"} , 
 	{ "Name" : "pool_acc_9", "interface" : "wire", "bitwidth" : 16, "direction" : "READWRITE"} , 
 	{ "Name" : "pool_acc_8", "interface" : "wire", "bitwidth" : 16, "direction" : "READWRITE"} , 
 	{ "Name" : "pool_acc_7", "interface" : "wire", "bitwidth" : 16, "direction" : "READWRITE"} , 
 	{ "Name" : "pool_acc_6", "interface" : "wire", "bitwidth" : 16, "direction" : "READWRITE"} , 
 	{ "Name" : "pool_acc_5", "interface" : "wire", "bitwidth" : 16, "direction" : "READWRITE"} , 
 	{ "Name" : "pool_acc_4", "interface" : "wire", "bitwidth" : 16, "direction" : "READWRITE"} , 
 	{ "Name" : "pool_acc_3", "interface" : "wire", "bitwidth" : 16, "direction" : "READWRITE"} , 
 	{ "Name" : "pool_acc_2", "interface" : "wire", "bitwidth" : 16, "direction" : "READWRITE"} , 
 	{ "Name" : "pool_acc_1", "interface" : "wire", "bitwidth" : 16, "direction" : "READWRITE"} , 
 	{ "Name" : "mul4", "interface" : "wire", "bitwidth" : 9, "direction" : "READONLY"} , 
 	{ "Name" : "U_slice", "interface" : "memory", "bitwidth" : 16, "direction" : "WRITEONLY", "extern" : 0} ]}
# RTL Port declarations: 
set portNum 111
set portList { 
	{ ap_clk sc_in sc_logic 1 clock -1 } 
	{ ap_rst sc_in sc_logic 1 reset -1 active_high_sync } 
	{ ap_start sc_in sc_logic 1 start -1 } 
	{ ap_done sc_out sc_logic 1 predone -1 } 
	{ ap_idle sc_out sc_logic 1 done -1 } 
	{ ap_ready sc_out sc_logic 1 ready -1 } 
	{ pool_acc_i sc_in sc_lv 16 signal 0 } 
	{ pool_acc_o sc_out sc_lv 16 signal 0 } 
	{ pool_acc_o_ap_vld sc_out sc_logic 1 outvld 0 } 
	{ pool_acc_31_i sc_in sc_lv 16 signal 1 } 
	{ pool_acc_31_o sc_out sc_lv 16 signal 1 } 
	{ pool_acc_31_o_ap_vld sc_out sc_logic 1 outvld 1 } 
	{ pool_acc_30_i sc_in sc_lv 16 signal 2 } 
	{ pool_acc_30_o sc_out sc_lv 16 signal 2 } 
	{ pool_acc_30_o_ap_vld sc_out sc_logic 1 outvld 2 } 
	{ pool_acc_29_i sc_in sc_lv 16 signal 3 } 
	{ pool_acc_29_o sc_out sc_lv 16 signal 3 } 
	{ pool_acc_29_o_ap_vld sc_out sc_logic 1 outvld 3 } 
	{ pool_acc_28_i sc_in sc_lv 16 signal 4 } 
	{ pool_acc_28_o sc_out sc_lv 16 signal 4 } 
	{ pool_acc_28_o_ap_vld sc_out sc_logic 1 outvld 4 } 
	{ pool_acc_27_i sc_in sc_lv 16 signal 5 } 
	{ pool_acc_27_o sc_out sc_lv 16 signal 5 } 
	{ pool_acc_27_o_ap_vld sc_out sc_logic 1 outvld 5 } 
	{ pool_acc_26_i sc_in sc_lv 16 signal 6 } 
	{ pool_acc_26_o sc_out sc_lv 16 signal 6 } 
	{ pool_acc_26_o_ap_vld sc_out sc_logic 1 outvld 6 } 
	{ pool_acc_25_i sc_in sc_lv 16 signal 7 } 
	{ pool_acc_25_o sc_out sc_lv 16 signal 7 } 
	{ pool_acc_25_o_ap_vld sc_out sc_logic 1 outvld 7 } 
	{ pool_acc_24_i sc_in sc_lv 16 signal 8 } 
	{ pool_acc_24_o sc_out sc_lv 16 signal 8 } 
	{ pool_acc_24_o_ap_vld sc_out sc_logic 1 outvld 8 } 
	{ pool_acc_23_i sc_in sc_lv 16 signal 9 } 
	{ pool_acc_23_o sc_out sc_lv 16 signal 9 } 
	{ pool_acc_23_o_ap_vld sc_out sc_logic 1 outvld 9 } 
	{ pool_acc_22_i sc_in sc_lv 16 signal 10 } 
	{ pool_acc_22_o sc_out sc_lv 16 signal 10 } 
	{ pool_acc_22_o_ap_vld sc_out sc_logic 1 outvld 10 } 
	{ pool_acc_21_i sc_in sc_lv 16 signal 11 } 
	{ pool_acc_21_o sc_out sc_lv 16 signal 11 } 
	{ pool_acc_21_o_ap_vld sc_out sc_logic 1 outvld 11 } 
	{ pool_acc_20_i sc_in sc_lv 16 signal 12 } 
	{ pool_acc_20_o sc_out sc_lv 16 signal 12 } 
	{ pool_acc_20_o_ap_vld sc_out sc_logic 1 outvld 12 } 
	{ pool_acc_19_i sc_in sc_lv 16 signal 13 } 
	{ pool_acc_19_o sc_out sc_lv 16 signal 13 } 
	{ pool_acc_19_o_ap_vld sc_out sc_logic 1 outvld 13 } 
	{ pool_acc_18_i sc_in sc_lv 16 signal 14 } 
	{ pool_acc_18_o sc_out sc_lv 16 signal 14 } 
	{ pool_acc_18_o_ap_vld sc_out sc_logic 1 outvld 14 } 
	{ pool_acc_17_i sc_in sc_lv 16 signal 15 } 
	{ pool_acc_17_o sc_out sc_lv 16 signal 15 } 
	{ pool_acc_17_o_ap_vld sc_out sc_logic 1 outvld 15 } 
	{ pool_acc_16_i sc_in sc_lv 16 signal 16 } 
	{ pool_acc_16_o sc_out sc_lv 16 signal 16 } 
	{ pool_acc_16_o_ap_vld sc_out sc_logic 1 outvld 16 } 
	{ pool_acc_15_i sc_in sc_lv 16 signal 17 } 
	{ pool_acc_15_o sc_out sc_lv 16 signal 17 } 
	{ pool_acc_15_o_ap_vld sc_out sc_logic 1 outvld 17 } 
	{ pool_acc_14_i sc_in sc_lv 16 signal 18 } 
	{ pool_acc_14_o sc_out sc_lv 16 signal 18 } 
	{ pool_acc_14_o_ap_vld sc_out sc_logic 1 outvld 18 } 
	{ pool_acc_13_i sc_in sc_lv 16 signal 19 } 
	{ pool_acc_13_o sc_out sc_lv 16 signal 19 } 
	{ pool_acc_13_o_ap_vld sc_out sc_logic 1 outvld 19 } 
	{ pool_acc_12_i sc_in sc_lv 16 signal 20 } 
	{ pool_acc_12_o sc_out sc_lv 16 signal 20 } 
	{ pool_acc_12_o_ap_vld sc_out sc_logic 1 outvld 20 } 
	{ pool_acc_11_i sc_in sc_lv 16 signal 21 } 
	{ pool_acc_11_o sc_out sc_lv 16 signal 21 } 
	{ pool_acc_11_o_ap_vld sc_out sc_logic 1 outvld 21 } 
	{ pool_acc_10_i sc_in sc_lv 16 signal 22 } 
	{ pool_acc_10_o sc_out sc_lv 16 signal 22 } 
	{ pool_acc_10_o_ap_vld sc_out sc_logic 1 outvld 22 } 
	{ pool_acc_9_i sc_in sc_lv 16 signal 23 } 
	{ pool_acc_9_o sc_out sc_lv 16 signal 23 } 
	{ pool_acc_9_o_ap_vld sc_out sc_logic 1 outvld 23 } 
	{ pool_acc_8_i sc_in sc_lv 16 signal 24 } 
	{ pool_acc_8_o sc_out sc_lv 16 signal 24 } 
	{ pool_acc_8_o_ap_vld sc_out sc_logic 1 outvld 24 } 
	{ pool_acc_7_i sc_in sc_lv 16 signal 25 } 
	{ pool_acc_7_o sc_out sc_lv 16 signal 25 } 
	{ pool_acc_7_o_ap_vld sc_out sc_logic 1 outvld 25 } 
	{ pool_acc_6_i sc_in sc_lv 16 signal 26 } 
	{ pool_acc_6_o sc_out sc_lv 16 signal 26 } 
	{ pool_acc_6_o_ap_vld sc_out sc_logic 1 outvld 26 } 
	{ pool_acc_5_i sc_in sc_lv 16 signal 27 } 
	{ pool_acc_5_o sc_out sc_lv 16 signal 27 } 
	{ pool_acc_5_o_ap_vld sc_out sc_logic 1 outvld 27 } 
	{ pool_acc_4_i sc_in sc_lv 16 signal 28 } 
	{ pool_acc_4_o sc_out sc_lv 16 signal 28 } 
	{ pool_acc_4_o_ap_vld sc_out sc_logic 1 outvld 28 } 
	{ pool_acc_3_i sc_in sc_lv 16 signal 29 } 
	{ pool_acc_3_o sc_out sc_lv 16 signal 29 } 
	{ pool_acc_3_o_ap_vld sc_out sc_logic 1 outvld 29 } 
	{ pool_acc_2_i sc_in sc_lv 16 signal 30 } 
	{ pool_acc_2_o sc_out sc_lv 16 signal 30 } 
	{ pool_acc_2_o_ap_vld sc_out sc_logic 1 outvld 30 } 
	{ pool_acc_1_i sc_in sc_lv 16 signal 31 } 
	{ pool_acc_1_o sc_out sc_lv 16 signal 31 } 
	{ pool_acc_1_o_ap_vld sc_out sc_logic 1 outvld 31 } 
	{ mul4 sc_in sc_lv 9 signal 32 } 
	{ U_slice_address0 sc_out sc_lv 9 signal 33 } 
	{ U_slice_ce0 sc_out sc_logic 1 signal 33 } 
	{ U_slice_we0 sc_out sc_logic 1 signal 33 } 
	{ U_slice_d0 sc_out sc_lv 16 signal 33 } 
	{ grp_fu_3051_p_din0 sc_out sc_lv 16 signal -1 } 
	{ grp_fu_3051_p_din1 sc_out sc_lv 16 signal -1 } 
	{ grp_fu_3051_p_dout0 sc_in sc_lv 16 signal -1 } 
	{ grp_fu_3051_p_ce sc_out sc_logic 1 signal -1 } 
}
set NewPortList {[ 
	{ "name": "ap_clk", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "clock", "bundle":{"name": "ap_clk", "role": "default" }} , 
 	{ "name": "ap_rst", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "reset", "bundle":{"name": "ap_rst", "role": "default" }} , 
 	{ "name": "ap_start", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "start", "bundle":{"name": "ap_start", "role": "default" }} , 
 	{ "name": "ap_done", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "predone", "bundle":{"name": "ap_done", "role": "default" }} , 
 	{ "name": "ap_idle", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "done", "bundle":{"name": "ap_idle", "role": "default" }} , 
 	{ "name": "ap_ready", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "ready", "bundle":{"name": "ap_ready", "role": "default" }} , 
 	{ "name": "pool_acc_i", "direction": "in", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "pool_acc", "role": "i" }} , 
 	{ "name": "pool_acc_o", "direction": "out", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "pool_acc", "role": "o" }} , 
 	{ "name": "pool_acc_o_ap_vld", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "outvld", "bundle":{"name": "pool_acc", "role": "o_ap_vld" }} , 
 	{ "name": "pool_acc_31_i", "direction": "in", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "pool_acc_31", "role": "i" }} , 
 	{ "name": "pool_acc_31_o", "direction": "out", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "pool_acc_31", "role": "o" }} , 
 	{ "name": "pool_acc_31_o_ap_vld", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "outvld", "bundle":{"name": "pool_acc_31", "role": "o_ap_vld" }} , 
 	{ "name": "pool_acc_30_i", "direction": "in", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "pool_acc_30", "role": "i" }} , 
 	{ "name": "pool_acc_30_o", "direction": "out", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "pool_acc_30", "role": "o" }} , 
 	{ "name": "pool_acc_30_o_ap_vld", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "outvld", "bundle":{"name": "pool_acc_30", "role": "o_ap_vld" }} , 
 	{ "name": "pool_acc_29_i", "direction": "in", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "pool_acc_29", "role": "i" }} , 
 	{ "name": "pool_acc_29_o", "direction": "out", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "pool_acc_29", "role": "o" }} , 
 	{ "name": "pool_acc_29_o_ap_vld", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "outvld", "bundle":{"name": "pool_acc_29", "role": "o_ap_vld" }} , 
 	{ "name": "pool_acc_28_i", "direction": "in", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "pool_acc_28", "role": "i" }} , 
 	{ "name": "pool_acc_28_o", "direction": "out", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "pool_acc_28", "role": "o" }} , 
 	{ "name": "pool_acc_28_o_ap_vld", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "outvld", "bundle":{"name": "pool_acc_28", "role": "o_ap_vld" }} , 
 	{ "name": "pool_acc_27_i", "direction": "in", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "pool_acc_27", "role": "i" }} , 
 	{ "name": "pool_acc_27_o", "direction": "out", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "pool_acc_27", "role": "o" }} , 
 	{ "name": "pool_acc_27_o_ap_vld", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "outvld", "bundle":{"name": "pool_acc_27", "role": "o_ap_vld" }} , 
 	{ "name": "pool_acc_26_i", "direction": "in", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "pool_acc_26", "role": "i" }} , 
 	{ "name": "pool_acc_26_o", "direction": "out", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "pool_acc_26", "role": "o" }} , 
 	{ "name": "pool_acc_26_o_ap_vld", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "outvld", "bundle":{"name": "pool_acc_26", "role": "o_ap_vld" }} , 
 	{ "name": "pool_acc_25_i", "direction": "in", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "pool_acc_25", "role": "i" }} , 
 	{ "name": "pool_acc_25_o", "direction": "out", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "pool_acc_25", "role": "o" }} , 
 	{ "name": "pool_acc_25_o_ap_vld", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "outvld", "bundle":{"name": "pool_acc_25", "role": "o_ap_vld" }} , 
 	{ "name": "pool_acc_24_i", "direction": "in", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "pool_acc_24", "role": "i" }} , 
 	{ "name": "pool_acc_24_o", "direction": "out", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "pool_acc_24", "role": "o" }} , 
 	{ "name": "pool_acc_24_o_ap_vld", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "outvld", "bundle":{"name": "pool_acc_24", "role": "o_ap_vld" }} , 
 	{ "name": "pool_acc_23_i", "direction": "in", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "pool_acc_23", "role": "i" }} , 
 	{ "name": "pool_acc_23_o", "direction": "out", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "pool_acc_23", "role": "o" }} , 
 	{ "name": "pool_acc_23_o_ap_vld", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "outvld", "bundle":{"name": "pool_acc_23", "role": "o_ap_vld" }} , 
 	{ "name": "pool_acc_22_i", "direction": "in", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "pool_acc_22", "role": "i" }} , 
 	{ "name": "pool_acc_22_o", "direction": "out", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "pool_acc_22", "role": "o" }} , 
 	{ "name": "pool_acc_22_o_ap_vld", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "outvld", "bundle":{"name": "pool_acc_22", "role": "o_ap_vld" }} , 
 	{ "name": "pool_acc_21_i", "direction": "in", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "pool_acc_21", "role": "i" }} , 
 	{ "name": "pool_acc_21_o", "direction": "out", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "pool_acc_21", "role": "o" }} , 
 	{ "name": "pool_acc_21_o_ap_vld", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "outvld", "bundle":{"name": "pool_acc_21", "role": "o_ap_vld" }} , 
 	{ "name": "pool_acc_20_i", "direction": "in", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "pool_acc_20", "role": "i" }} , 
 	{ "name": "pool_acc_20_o", "direction": "out", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "pool_acc_20", "role": "o" }} , 
 	{ "name": "pool_acc_20_o_ap_vld", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "outvld", "bundle":{"name": "pool_acc_20", "role": "o_ap_vld" }} , 
 	{ "name": "pool_acc_19_i", "direction": "in", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "pool_acc_19", "role": "i" }} , 
 	{ "name": "pool_acc_19_o", "direction": "out", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "pool_acc_19", "role": "o" }} , 
 	{ "name": "pool_acc_19_o_ap_vld", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "outvld", "bundle":{"name": "pool_acc_19", "role": "o_ap_vld" }} , 
 	{ "name": "pool_acc_18_i", "direction": "in", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "pool_acc_18", "role": "i" }} , 
 	{ "name": "pool_acc_18_o", "direction": "out", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "pool_acc_18", "role": "o" }} , 
 	{ "name": "pool_acc_18_o_ap_vld", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "outvld", "bundle":{"name": "pool_acc_18", "role": "o_ap_vld" }} , 
 	{ "name": "pool_acc_17_i", "direction": "in", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "pool_acc_17", "role": "i" }} , 
 	{ "name": "pool_acc_17_o", "direction": "out", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "pool_acc_17", "role": "o" }} , 
 	{ "name": "pool_acc_17_o_ap_vld", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "outvld", "bundle":{"name": "pool_acc_17", "role": "o_ap_vld" }} , 
 	{ "name": "pool_acc_16_i", "direction": "in", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "pool_acc_16", "role": "i" }} , 
 	{ "name": "pool_acc_16_o", "direction": "out", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "pool_acc_16", "role": "o" }} , 
 	{ "name": "pool_acc_16_o_ap_vld", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "outvld", "bundle":{"name": "pool_acc_16", "role": "o_ap_vld" }} , 
 	{ "name": "pool_acc_15_i", "direction": "in", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "pool_acc_15", "role": "i" }} , 
 	{ "name": "pool_acc_15_o", "direction": "out", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "pool_acc_15", "role": "o" }} , 
 	{ "name": "pool_acc_15_o_ap_vld", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "outvld", "bundle":{"name": "pool_acc_15", "role": "o_ap_vld" }} , 
 	{ "name": "pool_acc_14_i", "direction": "in", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "pool_acc_14", "role": "i" }} , 
 	{ "name": "pool_acc_14_o", "direction": "out", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "pool_acc_14", "role": "o" }} , 
 	{ "name": "pool_acc_14_o_ap_vld", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "outvld", "bundle":{"name": "pool_acc_14", "role": "o_ap_vld" }} , 
 	{ "name": "pool_acc_13_i", "direction": "in", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "pool_acc_13", "role": "i" }} , 
 	{ "name": "pool_acc_13_o", "direction": "out", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "pool_acc_13", "role": "o" }} , 
 	{ "name": "pool_acc_13_o_ap_vld", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "outvld", "bundle":{"name": "pool_acc_13", "role": "o_ap_vld" }} , 
 	{ "name": "pool_acc_12_i", "direction": "in", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "pool_acc_12", "role": "i" }} , 
 	{ "name": "pool_acc_12_o", "direction": "out", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "pool_acc_12", "role": "o" }} , 
 	{ "name": "pool_acc_12_o_ap_vld", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "outvld", "bundle":{"name": "pool_acc_12", "role": "o_ap_vld" }} , 
 	{ "name": "pool_acc_11_i", "direction": "in", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "pool_acc_11", "role": "i" }} , 
 	{ "name": "pool_acc_11_o", "direction": "out", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "pool_acc_11", "role": "o" }} , 
 	{ "name": "pool_acc_11_o_ap_vld", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "outvld", "bundle":{"name": "pool_acc_11", "role": "o_ap_vld" }} , 
 	{ "name": "pool_acc_10_i", "direction": "in", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "pool_acc_10", "role": "i" }} , 
 	{ "name": "pool_acc_10_o", "direction": "out", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "pool_acc_10", "role": "o" }} , 
 	{ "name": "pool_acc_10_o_ap_vld", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "outvld", "bundle":{"name": "pool_acc_10", "role": "o_ap_vld" }} , 
 	{ "name": "pool_acc_9_i", "direction": "in", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "pool_acc_9", "role": "i" }} , 
 	{ "name": "pool_acc_9_o", "direction": "out", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "pool_acc_9", "role": "o" }} , 
 	{ "name": "pool_acc_9_o_ap_vld", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "outvld", "bundle":{"name": "pool_acc_9", "role": "o_ap_vld" }} , 
 	{ "name": "pool_acc_8_i", "direction": "in", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "pool_acc_8", "role": "i" }} , 
 	{ "name": "pool_acc_8_o", "direction": "out", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "pool_acc_8", "role": "o" }} , 
 	{ "name": "pool_acc_8_o_ap_vld", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "outvld", "bundle":{"name": "pool_acc_8", "role": "o_ap_vld" }} , 
 	{ "name": "pool_acc_7_i", "direction": "in", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "pool_acc_7", "role": "i" }} , 
 	{ "name": "pool_acc_7_o", "direction": "out", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "pool_acc_7", "role": "o" }} , 
 	{ "name": "pool_acc_7_o_ap_vld", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "outvld", "bundle":{"name": "pool_acc_7", "role": "o_ap_vld" }} , 
 	{ "name": "pool_acc_6_i", "direction": "in", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "pool_acc_6", "role": "i" }} , 
 	{ "name": "pool_acc_6_o", "direction": "out", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "pool_acc_6", "role": "o" }} , 
 	{ "name": "pool_acc_6_o_ap_vld", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "outvld", "bundle":{"name": "pool_acc_6", "role": "o_ap_vld" }} , 
 	{ "name": "pool_acc_5_i", "direction": "in", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "pool_acc_5", "role": "i" }} , 
 	{ "name": "pool_acc_5_o", "direction": "out", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "pool_acc_5", "role": "o" }} , 
 	{ "name": "pool_acc_5_o_ap_vld", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "outvld", "bundle":{"name": "pool_acc_5", "role": "o_ap_vld" }} , 
 	{ "name": "pool_acc_4_i", "direction": "in", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "pool_acc_4", "role": "i" }} , 
 	{ "name": "pool_acc_4_o", "direction": "out", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "pool_acc_4", "role": "o" }} , 
 	{ "name": "pool_acc_4_o_ap_vld", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "outvld", "bundle":{"name": "pool_acc_4", "role": "o_ap_vld" }} , 
 	{ "name": "pool_acc_3_i", "direction": "in", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "pool_acc_3", "role": "i" }} , 
 	{ "name": "pool_acc_3_o", "direction": "out", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "pool_acc_3", "role": "o" }} , 
 	{ "name": "pool_acc_3_o_ap_vld", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "outvld", "bundle":{"name": "pool_acc_3", "role": "o_ap_vld" }} , 
 	{ "name": "pool_acc_2_i", "direction": "in", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "pool_acc_2", "role": "i" }} , 
 	{ "name": "pool_acc_2_o", "direction": "out", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "pool_acc_2", "role": "o" }} , 
 	{ "name": "pool_acc_2_o_ap_vld", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "outvld", "bundle":{"name": "pool_acc_2", "role": "o_ap_vld" }} , 
 	{ "name": "pool_acc_1_i", "direction": "in", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "pool_acc_1", "role": "i" }} , 
 	{ "name": "pool_acc_1_o", "direction": "out", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "pool_acc_1", "role": "o" }} , 
 	{ "name": "pool_acc_1_o_ap_vld", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "outvld", "bundle":{"name": "pool_acc_1", "role": "o_ap_vld" }} , 
 	{ "name": "mul4", "direction": "in", "datatype": "sc_lv", "bitwidth":9, "type": "signal", "bundle":{"name": "mul4", "role": "default" }} , 
 	{ "name": "U_slice_address0", "direction": "out", "datatype": "sc_lv", "bitwidth":9, "type": "signal", "bundle":{"name": "U_slice", "role": "address0" }} , 
 	{ "name": "U_slice_ce0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "U_slice", "role": "ce0" }} , 
 	{ "name": "U_slice_we0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "U_slice", "role": "we0" }} , 
 	{ "name": "U_slice_d0", "direction": "out", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "U_slice", "role": "d0" }} , 
 	{ "name": "grp_fu_3051_p_din0", "direction": "out", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "grp_fu_3051_p_din0", "role": "default" }} , 
 	{ "name": "grp_fu_3051_p_din1", "direction": "out", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "grp_fu_3051_p_din1", "role": "default" }} , 
 	{ "name": "grp_fu_3051_p_dout0", "direction": "in", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "grp_fu_3051_p_dout0", "role": "default" }} , 
 	{ "name": "grp_fu_3051_p_ce", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "grp_fu_3051_p_ce", "role": "default" }}  ]}

set RtlHierarchyInfo {[
	{"ID" : "0", "Level" : "0", "Path" : "`AUTOTB_DUT_INST", "Parent" : "", "Child" : ["1", "2"],
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
	{"ID" : "1", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.sparsemux_65_5_16_1_1_U179", "Parent" : "0"},
	{"ID" : "2", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.flow_control_loop_pipe_sequential_init_U", "Parent" : "0"}]}


set ArgLastReadFirstWriteLatency {
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
	{"Name" : "Latency", "Min" : "41", "Max" : "41"}
	, {"Name" : "Interval", "Min" : "41", "Max" : "41"}
]}

set PipelineEnableSignalInfo {[
	{"Pipeline" : "0", "EnableSignal" : "ap_enable_pp0"}
]}

set Spec2ImplPortList { 
	pool_acc { ap_ovld {  { pool_acc_i in_data 0 16 }  { pool_acc_o out_data 1 16 }  { pool_acc_o_ap_vld out_vld 1 1 } } }
	pool_acc_31 { ap_ovld {  { pool_acc_31_i in_data 0 16 }  { pool_acc_31_o out_data 1 16 }  { pool_acc_31_o_ap_vld out_vld 1 1 } } }
	pool_acc_30 { ap_ovld {  { pool_acc_30_i in_data 0 16 }  { pool_acc_30_o out_data 1 16 }  { pool_acc_30_o_ap_vld out_vld 1 1 } } }
	pool_acc_29 { ap_ovld {  { pool_acc_29_i in_data 0 16 }  { pool_acc_29_o out_data 1 16 }  { pool_acc_29_o_ap_vld out_vld 1 1 } } }
	pool_acc_28 { ap_ovld {  { pool_acc_28_i in_data 0 16 }  { pool_acc_28_o out_data 1 16 }  { pool_acc_28_o_ap_vld out_vld 1 1 } } }
	pool_acc_27 { ap_ovld {  { pool_acc_27_i in_data 0 16 }  { pool_acc_27_o out_data 1 16 }  { pool_acc_27_o_ap_vld out_vld 1 1 } } }
	pool_acc_26 { ap_ovld {  { pool_acc_26_i in_data 0 16 }  { pool_acc_26_o out_data 1 16 }  { pool_acc_26_o_ap_vld out_vld 1 1 } } }
	pool_acc_25 { ap_ovld {  { pool_acc_25_i in_data 0 16 }  { pool_acc_25_o out_data 1 16 }  { pool_acc_25_o_ap_vld out_vld 1 1 } } }
	pool_acc_24 { ap_ovld {  { pool_acc_24_i in_data 0 16 }  { pool_acc_24_o out_data 1 16 }  { pool_acc_24_o_ap_vld out_vld 1 1 } } }
	pool_acc_23 { ap_ovld {  { pool_acc_23_i in_data 0 16 }  { pool_acc_23_o out_data 1 16 }  { pool_acc_23_o_ap_vld out_vld 1 1 } } }
	pool_acc_22 { ap_ovld {  { pool_acc_22_i in_data 0 16 }  { pool_acc_22_o out_data 1 16 }  { pool_acc_22_o_ap_vld out_vld 1 1 } } }
	pool_acc_21 { ap_ovld {  { pool_acc_21_i in_data 0 16 }  { pool_acc_21_o out_data 1 16 }  { pool_acc_21_o_ap_vld out_vld 1 1 } } }
	pool_acc_20 { ap_ovld {  { pool_acc_20_i in_data 0 16 }  { pool_acc_20_o out_data 1 16 }  { pool_acc_20_o_ap_vld out_vld 1 1 } } }
	pool_acc_19 { ap_ovld {  { pool_acc_19_i in_data 0 16 }  { pool_acc_19_o out_data 1 16 }  { pool_acc_19_o_ap_vld out_vld 1 1 } } }
	pool_acc_18 { ap_ovld {  { pool_acc_18_i in_data 0 16 }  { pool_acc_18_o out_data 1 16 }  { pool_acc_18_o_ap_vld out_vld 1 1 } } }
	pool_acc_17 { ap_ovld {  { pool_acc_17_i in_data 0 16 }  { pool_acc_17_o out_data 1 16 }  { pool_acc_17_o_ap_vld out_vld 1 1 } } }
	pool_acc_16 { ap_ovld {  { pool_acc_16_i in_data 0 16 }  { pool_acc_16_o out_data 1 16 }  { pool_acc_16_o_ap_vld out_vld 1 1 } } }
	pool_acc_15 { ap_ovld {  { pool_acc_15_i in_data 0 16 }  { pool_acc_15_o out_data 1 16 }  { pool_acc_15_o_ap_vld out_vld 1 1 } } }
	pool_acc_14 { ap_ovld {  { pool_acc_14_i in_data 0 16 }  { pool_acc_14_o out_data 1 16 }  { pool_acc_14_o_ap_vld out_vld 1 1 } } }
	pool_acc_13 { ap_ovld {  { pool_acc_13_i in_data 0 16 }  { pool_acc_13_o out_data 1 16 }  { pool_acc_13_o_ap_vld out_vld 1 1 } } }
	pool_acc_12 { ap_ovld {  { pool_acc_12_i in_data 0 16 }  { pool_acc_12_o out_data 1 16 }  { pool_acc_12_o_ap_vld out_vld 1 1 } } }
	pool_acc_11 { ap_ovld {  { pool_acc_11_i in_data 0 16 }  { pool_acc_11_o out_data 1 16 }  { pool_acc_11_o_ap_vld out_vld 1 1 } } }
	pool_acc_10 { ap_ovld {  { pool_acc_10_i in_data 0 16 }  { pool_acc_10_o out_data 1 16 }  { pool_acc_10_o_ap_vld out_vld 1 1 } } }
	pool_acc_9 { ap_ovld {  { pool_acc_9_i in_data 0 16 }  { pool_acc_9_o out_data 1 16 }  { pool_acc_9_o_ap_vld out_vld 1 1 } } }
	pool_acc_8 { ap_ovld {  { pool_acc_8_i in_data 0 16 }  { pool_acc_8_o out_data 1 16 }  { pool_acc_8_o_ap_vld out_vld 1 1 } } }
	pool_acc_7 { ap_ovld {  { pool_acc_7_i in_data 0 16 }  { pool_acc_7_o out_data 1 16 }  { pool_acc_7_o_ap_vld out_vld 1 1 } } }
	pool_acc_6 { ap_ovld {  { pool_acc_6_i in_data 0 16 }  { pool_acc_6_o out_data 1 16 }  { pool_acc_6_o_ap_vld out_vld 1 1 } } }
	pool_acc_5 { ap_ovld {  { pool_acc_5_i in_data 0 16 }  { pool_acc_5_o out_data 1 16 }  { pool_acc_5_o_ap_vld out_vld 1 1 } } }
	pool_acc_4 { ap_ovld {  { pool_acc_4_i in_data 0 16 }  { pool_acc_4_o out_data 1 16 }  { pool_acc_4_o_ap_vld out_vld 1 1 } } }
	pool_acc_3 { ap_ovld {  { pool_acc_3_i in_data 0 16 }  { pool_acc_3_o out_data 1 16 }  { pool_acc_3_o_ap_vld out_vld 1 1 } } }
	pool_acc_2 { ap_ovld {  { pool_acc_2_i in_data 0 16 }  { pool_acc_2_o out_data 1 16 }  { pool_acc_2_o_ap_vld out_vld 1 1 } } }
	pool_acc_1 { ap_ovld {  { pool_acc_1_i in_data 0 16 }  { pool_acc_1_o out_data 1 16 }  { pool_acc_1_o_ap_vld out_vld 1 1 } } }
	mul4 { ap_none {  { mul4 in_data 0 9 } } }
	U_slice { ap_memory {  { U_slice_address0 mem_address 1 9 }  { U_slice_ce0 mem_ce 1 1 }  { U_slice_we0 mem_we 1 1 }  { U_slice_d0 mem_din 1 16 } } }
}
