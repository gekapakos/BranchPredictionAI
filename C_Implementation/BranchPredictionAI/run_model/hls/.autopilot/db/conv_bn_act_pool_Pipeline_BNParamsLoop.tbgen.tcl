set moduleName conv_bn_act_pool_Pipeline_BNParamsLoop
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
set C_modelName {conv_bn_act_pool_Pipeline_BNParamsLoop}
set C_modelType { void 0 }
set ap_memory_interface_dict [dict create]
set C_modelArgList {
	{ mux_case_31232615_out int 16 regular {pointer 1}  }
	{ mux_case_30230611_out int 16 regular {pointer 1}  }
	{ mux_case_29228607_out int 16 regular {pointer 1}  }
	{ mux_case_28226603_out int 16 regular {pointer 1}  }
	{ mux_case_27224599_out int 16 regular {pointer 1}  }
	{ mux_case_26222595_out int 16 regular {pointer 1}  }
	{ mux_case_25220591_out int 16 regular {pointer 1}  }
	{ mux_case_24218587_out int 16 regular {pointer 1}  }
	{ mux_case_23216583_out int 16 regular {pointer 1}  }
	{ mux_case_22214579_out int 16 regular {pointer 1}  }
	{ mux_case_21212575_out int 16 regular {pointer 1}  }
	{ mux_case_20210571_out int 16 regular {pointer 1}  }
	{ mux_case_19208567_out int 16 regular {pointer 1}  }
	{ mux_case_18206563_out int 16 regular {pointer 1}  }
	{ mux_case_17204559_out int 16 regular {pointer 1}  }
	{ mux_case_16202555_out int 16 regular {pointer 1}  }
	{ mux_case_15200551_out int 16 regular {pointer 1}  }
	{ mux_case_14198547_out int 16 regular {pointer 1}  }
	{ mux_case_13196543_out int 16 regular {pointer 1}  }
	{ mux_case_12194539_out int 16 regular {pointer 1}  }
	{ mux_case_11192535_out int 16 regular {pointer 1}  }
	{ mux_case_10190531_out int 16 regular {pointer 1}  }
	{ mux_case_9188527_out int 16 regular {pointer 1}  }
	{ mux_case_8186523_out int 16 regular {pointer 1}  }
	{ mux_case_7184519_out int 16 regular {pointer 1}  }
	{ mux_case_6182515_out int 16 regular {pointer 1}  }
	{ mux_case_5180511_out int 16 regular {pointer 1}  }
	{ mux_case_4178507_out int 16 regular {pointer 1}  }
	{ mux_case_3176503_out int 16 regular {pointer 1}  }
	{ mux_case_2174499_out int 16 regular {pointer 1}  }
	{ mux_case_1172495_out int 16 regular {pointer 1}  }
	{ mux_case_0170491_out int 16 regular {pointer 1}  }
	{ mux_case_31168487_out int 16 regular {pointer 1}  }
	{ mux_case_30166483_out int 16 regular {pointer 1}  }
	{ mux_case_29164479_out int 16 regular {pointer 1}  }
	{ mux_case_28162475_out int 16 regular {pointer 1}  }
	{ mux_case_27160471_out int 16 regular {pointer 1}  }
	{ mux_case_26158467_out int 16 regular {pointer 1}  }
	{ mux_case_25156463_out int 16 regular {pointer 1}  }
	{ mux_case_24154459_out int 16 regular {pointer 1}  }
	{ mux_case_23152455_out int 16 regular {pointer 1}  }
	{ mux_case_22150451_out int 16 regular {pointer 1}  }
	{ mux_case_21148447_out int 16 regular {pointer 1}  }
	{ mux_case_20146443_out int 16 regular {pointer 1}  }
	{ mux_case_19144439_out int 16 regular {pointer 1}  }
	{ mux_case_18142435_out int 16 regular {pointer 1}  }
	{ mux_case_17140431_out int 16 regular {pointer 1}  }
	{ mux_case_16138427_out int 16 regular {pointer 1}  }
	{ mux_case_15136423_out int 16 regular {pointer 1}  }
	{ mux_case_14134419_out int 16 regular {pointer 1}  }
	{ mux_case_13132415_out int 16 regular {pointer 1}  }
	{ mux_case_12130411_out int 16 regular {pointer 1}  }
	{ mux_case_11128407_out int 16 regular {pointer 1}  }
	{ mux_case_10126403_out int 16 regular {pointer 1}  }
	{ mux_case_9124399_out int 16 regular {pointer 1}  }
	{ mux_case_8122395_out int 16 regular {pointer 1}  }
	{ mux_case_7120391_out int 16 regular {pointer 1}  }
	{ mux_case_6118387_out int 16 regular {pointer 1}  }
	{ mux_case_5116383_out int 16 regular {pointer 1}  }
	{ mux_case_4114379_out int 16 regular {pointer 1}  }
	{ mux_case_3112375_out int 16 regular {pointer 1}  }
	{ mux_case_2110371_out int 16 regular {pointer 1}  }
	{ mux_case_1108367_out int 16 regular {pointer 1}  }
	{ mux_case_0106363_out int 16 regular {pointer 1}  }
}
set hasAXIMCache 0
set l_AXIML2Cache [list]
set AXIMCacheInstDict [dict create]
set C_modelArgMapList {[ 
	{ "Name" : "mux_case_31232615_out", "interface" : "wire", "bitwidth" : 16, "direction" : "WRITEONLY"} , 
 	{ "Name" : "mux_case_30230611_out", "interface" : "wire", "bitwidth" : 16, "direction" : "WRITEONLY"} , 
 	{ "Name" : "mux_case_29228607_out", "interface" : "wire", "bitwidth" : 16, "direction" : "WRITEONLY"} , 
 	{ "Name" : "mux_case_28226603_out", "interface" : "wire", "bitwidth" : 16, "direction" : "WRITEONLY"} , 
 	{ "Name" : "mux_case_27224599_out", "interface" : "wire", "bitwidth" : 16, "direction" : "WRITEONLY"} , 
 	{ "Name" : "mux_case_26222595_out", "interface" : "wire", "bitwidth" : 16, "direction" : "WRITEONLY"} , 
 	{ "Name" : "mux_case_25220591_out", "interface" : "wire", "bitwidth" : 16, "direction" : "WRITEONLY"} , 
 	{ "Name" : "mux_case_24218587_out", "interface" : "wire", "bitwidth" : 16, "direction" : "WRITEONLY"} , 
 	{ "Name" : "mux_case_23216583_out", "interface" : "wire", "bitwidth" : 16, "direction" : "WRITEONLY"} , 
 	{ "Name" : "mux_case_22214579_out", "interface" : "wire", "bitwidth" : 16, "direction" : "WRITEONLY"} , 
 	{ "Name" : "mux_case_21212575_out", "interface" : "wire", "bitwidth" : 16, "direction" : "WRITEONLY"} , 
 	{ "Name" : "mux_case_20210571_out", "interface" : "wire", "bitwidth" : 16, "direction" : "WRITEONLY"} , 
 	{ "Name" : "mux_case_19208567_out", "interface" : "wire", "bitwidth" : 16, "direction" : "WRITEONLY"} , 
 	{ "Name" : "mux_case_18206563_out", "interface" : "wire", "bitwidth" : 16, "direction" : "WRITEONLY"} , 
 	{ "Name" : "mux_case_17204559_out", "interface" : "wire", "bitwidth" : 16, "direction" : "WRITEONLY"} , 
 	{ "Name" : "mux_case_16202555_out", "interface" : "wire", "bitwidth" : 16, "direction" : "WRITEONLY"} , 
 	{ "Name" : "mux_case_15200551_out", "interface" : "wire", "bitwidth" : 16, "direction" : "WRITEONLY"} , 
 	{ "Name" : "mux_case_14198547_out", "interface" : "wire", "bitwidth" : 16, "direction" : "WRITEONLY"} , 
 	{ "Name" : "mux_case_13196543_out", "interface" : "wire", "bitwidth" : 16, "direction" : "WRITEONLY"} , 
 	{ "Name" : "mux_case_12194539_out", "interface" : "wire", "bitwidth" : 16, "direction" : "WRITEONLY"} , 
 	{ "Name" : "mux_case_11192535_out", "interface" : "wire", "bitwidth" : 16, "direction" : "WRITEONLY"} , 
 	{ "Name" : "mux_case_10190531_out", "interface" : "wire", "bitwidth" : 16, "direction" : "WRITEONLY"} , 
 	{ "Name" : "mux_case_9188527_out", "interface" : "wire", "bitwidth" : 16, "direction" : "WRITEONLY"} , 
 	{ "Name" : "mux_case_8186523_out", "interface" : "wire", "bitwidth" : 16, "direction" : "WRITEONLY"} , 
 	{ "Name" : "mux_case_7184519_out", "interface" : "wire", "bitwidth" : 16, "direction" : "WRITEONLY"} , 
 	{ "Name" : "mux_case_6182515_out", "interface" : "wire", "bitwidth" : 16, "direction" : "WRITEONLY"} , 
 	{ "Name" : "mux_case_5180511_out", "interface" : "wire", "bitwidth" : 16, "direction" : "WRITEONLY"} , 
 	{ "Name" : "mux_case_4178507_out", "interface" : "wire", "bitwidth" : 16, "direction" : "WRITEONLY"} , 
 	{ "Name" : "mux_case_3176503_out", "interface" : "wire", "bitwidth" : 16, "direction" : "WRITEONLY"} , 
 	{ "Name" : "mux_case_2174499_out", "interface" : "wire", "bitwidth" : 16, "direction" : "WRITEONLY"} , 
 	{ "Name" : "mux_case_1172495_out", "interface" : "wire", "bitwidth" : 16, "direction" : "WRITEONLY"} , 
 	{ "Name" : "mux_case_0170491_out", "interface" : "wire", "bitwidth" : 16, "direction" : "WRITEONLY"} , 
 	{ "Name" : "mux_case_31168487_out", "interface" : "wire", "bitwidth" : 16, "direction" : "WRITEONLY"} , 
 	{ "Name" : "mux_case_30166483_out", "interface" : "wire", "bitwidth" : 16, "direction" : "WRITEONLY"} , 
 	{ "Name" : "mux_case_29164479_out", "interface" : "wire", "bitwidth" : 16, "direction" : "WRITEONLY"} , 
 	{ "Name" : "mux_case_28162475_out", "interface" : "wire", "bitwidth" : 16, "direction" : "WRITEONLY"} , 
 	{ "Name" : "mux_case_27160471_out", "interface" : "wire", "bitwidth" : 16, "direction" : "WRITEONLY"} , 
 	{ "Name" : "mux_case_26158467_out", "interface" : "wire", "bitwidth" : 16, "direction" : "WRITEONLY"} , 
 	{ "Name" : "mux_case_25156463_out", "interface" : "wire", "bitwidth" : 16, "direction" : "WRITEONLY"} , 
 	{ "Name" : "mux_case_24154459_out", "interface" : "wire", "bitwidth" : 16, "direction" : "WRITEONLY"} , 
 	{ "Name" : "mux_case_23152455_out", "interface" : "wire", "bitwidth" : 16, "direction" : "WRITEONLY"} , 
 	{ "Name" : "mux_case_22150451_out", "interface" : "wire", "bitwidth" : 16, "direction" : "WRITEONLY"} , 
 	{ "Name" : "mux_case_21148447_out", "interface" : "wire", "bitwidth" : 16, "direction" : "WRITEONLY"} , 
 	{ "Name" : "mux_case_20146443_out", "interface" : "wire", "bitwidth" : 16, "direction" : "WRITEONLY"} , 
 	{ "Name" : "mux_case_19144439_out", "interface" : "wire", "bitwidth" : 16, "direction" : "WRITEONLY"} , 
 	{ "Name" : "mux_case_18142435_out", "interface" : "wire", "bitwidth" : 16, "direction" : "WRITEONLY"} , 
 	{ "Name" : "mux_case_17140431_out", "interface" : "wire", "bitwidth" : 16, "direction" : "WRITEONLY"} , 
 	{ "Name" : "mux_case_16138427_out", "interface" : "wire", "bitwidth" : 16, "direction" : "WRITEONLY"} , 
 	{ "Name" : "mux_case_15136423_out", "interface" : "wire", "bitwidth" : 16, "direction" : "WRITEONLY"} , 
 	{ "Name" : "mux_case_14134419_out", "interface" : "wire", "bitwidth" : 16, "direction" : "WRITEONLY"} , 
 	{ "Name" : "mux_case_13132415_out", "interface" : "wire", "bitwidth" : 16, "direction" : "WRITEONLY"} , 
 	{ "Name" : "mux_case_12130411_out", "interface" : "wire", "bitwidth" : 16, "direction" : "WRITEONLY"} , 
 	{ "Name" : "mux_case_11128407_out", "interface" : "wire", "bitwidth" : 16, "direction" : "WRITEONLY"} , 
 	{ "Name" : "mux_case_10126403_out", "interface" : "wire", "bitwidth" : 16, "direction" : "WRITEONLY"} , 
 	{ "Name" : "mux_case_9124399_out", "interface" : "wire", "bitwidth" : 16, "direction" : "WRITEONLY"} , 
 	{ "Name" : "mux_case_8122395_out", "interface" : "wire", "bitwidth" : 16, "direction" : "WRITEONLY"} , 
 	{ "Name" : "mux_case_7120391_out", "interface" : "wire", "bitwidth" : 16, "direction" : "WRITEONLY"} , 
 	{ "Name" : "mux_case_6118387_out", "interface" : "wire", "bitwidth" : 16, "direction" : "WRITEONLY"} , 
 	{ "Name" : "mux_case_5116383_out", "interface" : "wire", "bitwidth" : 16, "direction" : "WRITEONLY"} , 
 	{ "Name" : "mux_case_4114379_out", "interface" : "wire", "bitwidth" : 16, "direction" : "WRITEONLY"} , 
 	{ "Name" : "mux_case_3112375_out", "interface" : "wire", "bitwidth" : 16, "direction" : "WRITEONLY"} , 
 	{ "Name" : "mux_case_2110371_out", "interface" : "wire", "bitwidth" : 16, "direction" : "WRITEONLY"} , 
 	{ "Name" : "mux_case_1108367_out", "interface" : "wire", "bitwidth" : 16, "direction" : "WRITEONLY"} , 
 	{ "Name" : "mux_case_0106363_out", "interface" : "wire", "bitwidth" : 16, "direction" : "WRITEONLY"} ]}
# RTL Port declarations: 
set portNum 172
set portList { 
	{ ap_clk sc_in sc_logic 1 clock -1 } 
	{ ap_rst sc_in sc_logic 1 reset -1 active_high_sync } 
	{ ap_start sc_in sc_logic 1 start -1 } 
	{ ap_done sc_out sc_logic 1 predone -1 } 
	{ ap_idle sc_out sc_logic 1 done -1 } 
	{ ap_ready sc_out sc_logic 1 ready -1 } 
	{ mux_case_31232615_out sc_out sc_lv 16 signal 0 } 
	{ mux_case_31232615_out_ap_vld sc_out sc_logic 1 outvld 0 } 
	{ mux_case_30230611_out sc_out sc_lv 16 signal 1 } 
	{ mux_case_30230611_out_ap_vld sc_out sc_logic 1 outvld 1 } 
	{ mux_case_29228607_out sc_out sc_lv 16 signal 2 } 
	{ mux_case_29228607_out_ap_vld sc_out sc_logic 1 outvld 2 } 
	{ mux_case_28226603_out sc_out sc_lv 16 signal 3 } 
	{ mux_case_28226603_out_ap_vld sc_out sc_logic 1 outvld 3 } 
	{ mux_case_27224599_out sc_out sc_lv 16 signal 4 } 
	{ mux_case_27224599_out_ap_vld sc_out sc_logic 1 outvld 4 } 
	{ mux_case_26222595_out sc_out sc_lv 16 signal 5 } 
	{ mux_case_26222595_out_ap_vld sc_out sc_logic 1 outvld 5 } 
	{ mux_case_25220591_out sc_out sc_lv 16 signal 6 } 
	{ mux_case_25220591_out_ap_vld sc_out sc_logic 1 outvld 6 } 
	{ mux_case_24218587_out sc_out sc_lv 16 signal 7 } 
	{ mux_case_24218587_out_ap_vld sc_out sc_logic 1 outvld 7 } 
	{ mux_case_23216583_out sc_out sc_lv 16 signal 8 } 
	{ mux_case_23216583_out_ap_vld sc_out sc_logic 1 outvld 8 } 
	{ mux_case_22214579_out sc_out sc_lv 16 signal 9 } 
	{ mux_case_22214579_out_ap_vld sc_out sc_logic 1 outvld 9 } 
	{ mux_case_21212575_out sc_out sc_lv 16 signal 10 } 
	{ mux_case_21212575_out_ap_vld sc_out sc_logic 1 outvld 10 } 
	{ mux_case_20210571_out sc_out sc_lv 16 signal 11 } 
	{ mux_case_20210571_out_ap_vld sc_out sc_logic 1 outvld 11 } 
	{ mux_case_19208567_out sc_out sc_lv 16 signal 12 } 
	{ mux_case_19208567_out_ap_vld sc_out sc_logic 1 outvld 12 } 
	{ mux_case_18206563_out sc_out sc_lv 16 signal 13 } 
	{ mux_case_18206563_out_ap_vld sc_out sc_logic 1 outvld 13 } 
	{ mux_case_17204559_out sc_out sc_lv 16 signal 14 } 
	{ mux_case_17204559_out_ap_vld sc_out sc_logic 1 outvld 14 } 
	{ mux_case_16202555_out sc_out sc_lv 16 signal 15 } 
	{ mux_case_16202555_out_ap_vld sc_out sc_logic 1 outvld 15 } 
	{ mux_case_15200551_out sc_out sc_lv 16 signal 16 } 
	{ mux_case_15200551_out_ap_vld sc_out sc_logic 1 outvld 16 } 
	{ mux_case_14198547_out sc_out sc_lv 16 signal 17 } 
	{ mux_case_14198547_out_ap_vld sc_out sc_logic 1 outvld 17 } 
	{ mux_case_13196543_out sc_out sc_lv 16 signal 18 } 
	{ mux_case_13196543_out_ap_vld sc_out sc_logic 1 outvld 18 } 
	{ mux_case_12194539_out sc_out sc_lv 16 signal 19 } 
	{ mux_case_12194539_out_ap_vld sc_out sc_logic 1 outvld 19 } 
	{ mux_case_11192535_out sc_out sc_lv 16 signal 20 } 
	{ mux_case_11192535_out_ap_vld sc_out sc_logic 1 outvld 20 } 
	{ mux_case_10190531_out sc_out sc_lv 16 signal 21 } 
	{ mux_case_10190531_out_ap_vld sc_out sc_logic 1 outvld 21 } 
	{ mux_case_9188527_out sc_out sc_lv 16 signal 22 } 
	{ mux_case_9188527_out_ap_vld sc_out sc_logic 1 outvld 22 } 
	{ mux_case_8186523_out sc_out sc_lv 16 signal 23 } 
	{ mux_case_8186523_out_ap_vld sc_out sc_logic 1 outvld 23 } 
	{ mux_case_7184519_out sc_out sc_lv 16 signal 24 } 
	{ mux_case_7184519_out_ap_vld sc_out sc_logic 1 outvld 24 } 
	{ mux_case_6182515_out sc_out sc_lv 16 signal 25 } 
	{ mux_case_6182515_out_ap_vld sc_out sc_logic 1 outvld 25 } 
	{ mux_case_5180511_out sc_out sc_lv 16 signal 26 } 
	{ mux_case_5180511_out_ap_vld sc_out sc_logic 1 outvld 26 } 
	{ mux_case_4178507_out sc_out sc_lv 16 signal 27 } 
	{ mux_case_4178507_out_ap_vld sc_out sc_logic 1 outvld 27 } 
	{ mux_case_3176503_out sc_out sc_lv 16 signal 28 } 
	{ mux_case_3176503_out_ap_vld sc_out sc_logic 1 outvld 28 } 
	{ mux_case_2174499_out sc_out sc_lv 16 signal 29 } 
	{ mux_case_2174499_out_ap_vld sc_out sc_logic 1 outvld 29 } 
	{ mux_case_1172495_out sc_out sc_lv 16 signal 30 } 
	{ mux_case_1172495_out_ap_vld sc_out sc_logic 1 outvld 30 } 
	{ mux_case_0170491_out sc_out sc_lv 16 signal 31 } 
	{ mux_case_0170491_out_ap_vld sc_out sc_logic 1 outvld 31 } 
	{ mux_case_31168487_out sc_out sc_lv 16 signal 32 } 
	{ mux_case_31168487_out_ap_vld sc_out sc_logic 1 outvld 32 } 
	{ mux_case_30166483_out sc_out sc_lv 16 signal 33 } 
	{ mux_case_30166483_out_ap_vld sc_out sc_logic 1 outvld 33 } 
	{ mux_case_29164479_out sc_out sc_lv 16 signal 34 } 
	{ mux_case_29164479_out_ap_vld sc_out sc_logic 1 outvld 34 } 
	{ mux_case_28162475_out sc_out sc_lv 16 signal 35 } 
	{ mux_case_28162475_out_ap_vld sc_out sc_logic 1 outvld 35 } 
	{ mux_case_27160471_out sc_out sc_lv 16 signal 36 } 
	{ mux_case_27160471_out_ap_vld sc_out sc_logic 1 outvld 36 } 
	{ mux_case_26158467_out sc_out sc_lv 16 signal 37 } 
	{ mux_case_26158467_out_ap_vld sc_out sc_logic 1 outvld 37 } 
	{ mux_case_25156463_out sc_out sc_lv 16 signal 38 } 
	{ mux_case_25156463_out_ap_vld sc_out sc_logic 1 outvld 38 } 
	{ mux_case_24154459_out sc_out sc_lv 16 signal 39 } 
	{ mux_case_24154459_out_ap_vld sc_out sc_logic 1 outvld 39 } 
	{ mux_case_23152455_out sc_out sc_lv 16 signal 40 } 
	{ mux_case_23152455_out_ap_vld sc_out sc_logic 1 outvld 40 } 
	{ mux_case_22150451_out sc_out sc_lv 16 signal 41 } 
	{ mux_case_22150451_out_ap_vld sc_out sc_logic 1 outvld 41 } 
	{ mux_case_21148447_out sc_out sc_lv 16 signal 42 } 
	{ mux_case_21148447_out_ap_vld sc_out sc_logic 1 outvld 42 } 
	{ mux_case_20146443_out sc_out sc_lv 16 signal 43 } 
	{ mux_case_20146443_out_ap_vld sc_out sc_logic 1 outvld 43 } 
	{ mux_case_19144439_out sc_out sc_lv 16 signal 44 } 
	{ mux_case_19144439_out_ap_vld sc_out sc_logic 1 outvld 44 } 
	{ mux_case_18142435_out sc_out sc_lv 16 signal 45 } 
	{ mux_case_18142435_out_ap_vld sc_out sc_logic 1 outvld 45 } 
	{ mux_case_17140431_out sc_out sc_lv 16 signal 46 } 
	{ mux_case_17140431_out_ap_vld sc_out sc_logic 1 outvld 46 } 
	{ mux_case_16138427_out sc_out sc_lv 16 signal 47 } 
	{ mux_case_16138427_out_ap_vld sc_out sc_logic 1 outvld 47 } 
	{ mux_case_15136423_out sc_out sc_lv 16 signal 48 } 
	{ mux_case_15136423_out_ap_vld sc_out sc_logic 1 outvld 48 } 
	{ mux_case_14134419_out sc_out sc_lv 16 signal 49 } 
	{ mux_case_14134419_out_ap_vld sc_out sc_logic 1 outvld 49 } 
	{ mux_case_13132415_out sc_out sc_lv 16 signal 50 } 
	{ mux_case_13132415_out_ap_vld sc_out sc_logic 1 outvld 50 } 
	{ mux_case_12130411_out sc_out sc_lv 16 signal 51 } 
	{ mux_case_12130411_out_ap_vld sc_out sc_logic 1 outvld 51 } 
	{ mux_case_11128407_out sc_out sc_lv 16 signal 52 } 
	{ mux_case_11128407_out_ap_vld sc_out sc_logic 1 outvld 52 } 
	{ mux_case_10126403_out sc_out sc_lv 16 signal 53 } 
	{ mux_case_10126403_out_ap_vld sc_out sc_logic 1 outvld 53 } 
	{ mux_case_9124399_out sc_out sc_lv 16 signal 54 } 
	{ mux_case_9124399_out_ap_vld sc_out sc_logic 1 outvld 54 } 
	{ mux_case_8122395_out sc_out sc_lv 16 signal 55 } 
	{ mux_case_8122395_out_ap_vld sc_out sc_logic 1 outvld 55 } 
	{ mux_case_7120391_out sc_out sc_lv 16 signal 56 } 
	{ mux_case_7120391_out_ap_vld sc_out sc_logic 1 outvld 56 } 
	{ mux_case_6118387_out sc_out sc_lv 16 signal 57 } 
	{ mux_case_6118387_out_ap_vld sc_out sc_logic 1 outvld 57 } 
	{ mux_case_5116383_out sc_out sc_lv 16 signal 58 } 
	{ mux_case_5116383_out_ap_vld sc_out sc_logic 1 outvld 58 } 
	{ mux_case_4114379_out sc_out sc_lv 16 signal 59 } 
	{ mux_case_4114379_out_ap_vld sc_out sc_logic 1 outvld 59 } 
	{ mux_case_3112375_out sc_out sc_lv 16 signal 60 } 
	{ mux_case_3112375_out_ap_vld sc_out sc_logic 1 outvld 60 } 
	{ mux_case_2110371_out sc_out sc_lv 16 signal 61 } 
	{ mux_case_2110371_out_ap_vld sc_out sc_logic 1 outvld 61 } 
	{ mux_case_1108367_out sc_out sc_lv 16 signal 62 } 
	{ mux_case_1108367_out_ap_vld sc_out sc_logic 1 outvld 62 } 
	{ mux_case_0106363_out sc_out sc_lv 16 signal 63 } 
	{ mux_case_0106363_out_ap_vld sc_out sc_logic 1 outvld 63 } 
	{ grp_fu_756_p_din0 sc_out sc_lv 32 signal -1 } 
	{ grp_fu_756_p_din1 sc_out sc_lv 32 signal -1 } 
	{ grp_fu_756_p_opcode sc_out sc_lv 2 signal -1 } 
	{ grp_fu_756_p_dout0 sc_in sc_lv 32 signal -1 } 
	{ grp_fu_756_p_ce sc_out sc_logic 1 signal -1 } 
	{ grp_fu_3032_p_din0 sc_out sc_lv 32 signal -1 } 
	{ grp_fu_3032_p_din1 sc_out sc_lv 32 signal -1 } 
	{ grp_fu_3032_p_opcode sc_out sc_lv 2 signal -1 } 
	{ grp_fu_3032_p_dout0 sc_in sc_lv 32 signal -1 } 
	{ grp_fu_3032_p_ce sc_out sc_logic 1 signal -1 } 
	{ grp_fu_760_p_din0 sc_out sc_lv 32 signal -1 } 
	{ grp_fu_760_p_din1 sc_out sc_lv 32 signal -1 } 
	{ grp_fu_760_p_dout0 sc_in sc_lv 32 signal -1 } 
	{ grp_fu_760_p_ce sc_out sc_logic 1 signal -1 } 
	{ grp_fu_3036_p_din0 sc_out sc_lv 32 signal -1 } 
	{ grp_fu_3036_p_din1 sc_out sc_lv 32 signal -1 } 
	{ grp_fu_3036_p_dout0 sc_in sc_lv 32 signal -1 } 
	{ grp_fu_3036_p_ce sc_out sc_logic 1 signal -1 } 
	{ grp_fu_3040_p_din0 sc_out sc_lv 32 signal -1 } 
	{ grp_fu_3040_p_din1 sc_out sc_lv 32 signal -1 } 
	{ grp_fu_3040_p_dout0 sc_in sc_lv 32 signal -1 } 
	{ grp_fu_3040_p_ce sc_out sc_logic 1 signal -1 } 
	{ grp_fu_3044_p_din0 sc_out sc_lv 32 signal -1 } 
	{ grp_fu_3044_p_din1 sc_out sc_lv 32 signal -1 } 
	{ grp_fu_3044_p_dout0 sc_in sc_lv 32 signal -1 } 
	{ grp_fu_3044_p_ce sc_out sc_logic 1 signal -1 } 
	{ grp_fu_764_p_din0 sc_out sc_lv 32 signal -1 } 
	{ grp_fu_764_p_dout0 sc_in sc_lv 16 signal -1 } 
	{ grp_fu_764_p_ce sc_out sc_logic 1 signal -1 } 
	{ grp_fu_3048_p_din0 sc_out sc_lv 32 signal -1 } 
	{ grp_fu_3048_p_dout0 sc_in sc_lv 16 signal -1 } 
	{ grp_fu_3048_p_ce sc_out sc_logic 1 signal -1 } 
	{ grp_fu_767_p_din0 sc_out sc_lv 16 signal -1 } 
	{ grp_fu_767_p_dout0 sc_in sc_lv 32 signal -1 } 
	{ grp_fu_767_p_ce sc_out sc_logic 1 signal -1 } 
	{ grp_fu_770_p_din0 sc_out sc_lv 16 signal -1 } 
	{ grp_fu_770_p_dout0 sc_in sc_lv 32 signal -1 } 
	{ grp_fu_770_p_ce sc_out sc_logic 1 signal -1 } 
}
set NewPortList {[ 
	{ "name": "ap_clk", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "clock", "bundle":{"name": "ap_clk", "role": "default" }} , 
 	{ "name": "ap_rst", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "reset", "bundle":{"name": "ap_rst", "role": "default" }} , 
 	{ "name": "ap_start", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "start", "bundle":{"name": "ap_start", "role": "default" }} , 
 	{ "name": "ap_done", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "predone", "bundle":{"name": "ap_done", "role": "default" }} , 
 	{ "name": "ap_idle", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "done", "bundle":{"name": "ap_idle", "role": "default" }} , 
 	{ "name": "ap_ready", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "ready", "bundle":{"name": "ap_ready", "role": "default" }} , 
 	{ "name": "mux_case_31232615_out", "direction": "out", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "mux_case_31232615_out", "role": "default" }} , 
 	{ "name": "mux_case_31232615_out_ap_vld", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "outvld", "bundle":{"name": "mux_case_31232615_out", "role": "ap_vld" }} , 
 	{ "name": "mux_case_30230611_out", "direction": "out", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "mux_case_30230611_out", "role": "default" }} , 
 	{ "name": "mux_case_30230611_out_ap_vld", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "outvld", "bundle":{"name": "mux_case_30230611_out", "role": "ap_vld" }} , 
 	{ "name": "mux_case_29228607_out", "direction": "out", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "mux_case_29228607_out", "role": "default" }} , 
 	{ "name": "mux_case_29228607_out_ap_vld", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "outvld", "bundle":{"name": "mux_case_29228607_out", "role": "ap_vld" }} , 
 	{ "name": "mux_case_28226603_out", "direction": "out", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "mux_case_28226603_out", "role": "default" }} , 
 	{ "name": "mux_case_28226603_out_ap_vld", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "outvld", "bundle":{"name": "mux_case_28226603_out", "role": "ap_vld" }} , 
 	{ "name": "mux_case_27224599_out", "direction": "out", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "mux_case_27224599_out", "role": "default" }} , 
 	{ "name": "mux_case_27224599_out_ap_vld", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "outvld", "bundle":{"name": "mux_case_27224599_out", "role": "ap_vld" }} , 
 	{ "name": "mux_case_26222595_out", "direction": "out", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "mux_case_26222595_out", "role": "default" }} , 
 	{ "name": "mux_case_26222595_out_ap_vld", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "outvld", "bundle":{"name": "mux_case_26222595_out", "role": "ap_vld" }} , 
 	{ "name": "mux_case_25220591_out", "direction": "out", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "mux_case_25220591_out", "role": "default" }} , 
 	{ "name": "mux_case_25220591_out_ap_vld", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "outvld", "bundle":{"name": "mux_case_25220591_out", "role": "ap_vld" }} , 
 	{ "name": "mux_case_24218587_out", "direction": "out", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "mux_case_24218587_out", "role": "default" }} , 
 	{ "name": "mux_case_24218587_out_ap_vld", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "outvld", "bundle":{"name": "mux_case_24218587_out", "role": "ap_vld" }} , 
 	{ "name": "mux_case_23216583_out", "direction": "out", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "mux_case_23216583_out", "role": "default" }} , 
 	{ "name": "mux_case_23216583_out_ap_vld", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "outvld", "bundle":{"name": "mux_case_23216583_out", "role": "ap_vld" }} , 
 	{ "name": "mux_case_22214579_out", "direction": "out", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "mux_case_22214579_out", "role": "default" }} , 
 	{ "name": "mux_case_22214579_out_ap_vld", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "outvld", "bundle":{"name": "mux_case_22214579_out", "role": "ap_vld" }} , 
 	{ "name": "mux_case_21212575_out", "direction": "out", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "mux_case_21212575_out", "role": "default" }} , 
 	{ "name": "mux_case_21212575_out_ap_vld", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "outvld", "bundle":{"name": "mux_case_21212575_out", "role": "ap_vld" }} , 
 	{ "name": "mux_case_20210571_out", "direction": "out", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "mux_case_20210571_out", "role": "default" }} , 
 	{ "name": "mux_case_20210571_out_ap_vld", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "outvld", "bundle":{"name": "mux_case_20210571_out", "role": "ap_vld" }} , 
 	{ "name": "mux_case_19208567_out", "direction": "out", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "mux_case_19208567_out", "role": "default" }} , 
 	{ "name": "mux_case_19208567_out_ap_vld", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "outvld", "bundle":{"name": "mux_case_19208567_out", "role": "ap_vld" }} , 
 	{ "name": "mux_case_18206563_out", "direction": "out", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "mux_case_18206563_out", "role": "default" }} , 
 	{ "name": "mux_case_18206563_out_ap_vld", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "outvld", "bundle":{"name": "mux_case_18206563_out", "role": "ap_vld" }} , 
 	{ "name": "mux_case_17204559_out", "direction": "out", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "mux_case_17204559_out", "role": "default" }} , 
 	{ "name": "mux_case_17204559_out_ap_vld", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "outvld", "bundle":{"name": "mux_case_17204559_out", "role": "ap_vld" }} , 
 	{ "name": "mux_case_16202555_out", "direction": "out", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "mux_case_16202555_out", "role": "default" }} , 
 	{ "name": "mux_case_16202555_out_ap_vld", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "outvld", "bundle":{"name": "mux_case_16202555_out", "role": "ap_vld" }} , 
 	{ "name": "mux_case_15200551_out", "direction": "out", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "mux_case_15200551_out", "role": "default" }} , 
 	{ "name": "mux_case_15200551_out_ap_vld", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "outvld", "bundle":{"name": "mux_case_15200551_out", "role": "ap_vld" }} , 
 	{ "name": "mux_case_14198547_out", "direction": "out", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "mux_case_14198547_out", "role": "default" }} , 
 	{ "name": "mux_case_14198547_out_ap_vld", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "outvld", "bundle":{"name": "mux_case_14198547_out", "role": "ap_vld" }} , 
 	{ "name": "mux_case_13196543_out", "direction": "out", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "mux_case_13196543_out", "role": "default" }} , 
 	{ "name": "mux_case_13196543_out_ap_vld", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "outvld", "bundle":{"name": "mux_case_13196543_out", "role": "ap_vld" }} , 
 	{ "name": "mux_case_12194539_out", "direction": "out", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "mux_case_12194539_out", "role": "default" }} , 
 	{ "name": "mux_case_12194539_out_ap_vld", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "outvld", "bundle":{"name": "mux_case_12194539_out", "role": "ap_vld" }} , 
 	{ "name": "mux_case_11192535_out", "direction": "out", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "mux_case_11192535_out", "role": "default" }} , 
 	{ "name": "mux_case_11192535_out_ap_vld", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "outvld", "bundle":{"name": "mux_case_11192535_out", "role": "ap_vld" }} , 
 	{ "name": "mux_case_10190531_out", "direction": "out", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "mux_case_10190531_out", "role": "default" }} , 
 	{ "name": "mux_case_10190531_out_ap_vld", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "outvld", "bundle":{"name": "mux_case_10190531_out", "role": "ap_vld" }} , 
 	{ "name": "mux_case_9188527_out", "direction": "out", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "mux_case_9188527_out", "role": "default" }} , 
 	{ "name": "mux_case_9188527_out_ap_vld", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "outvld", "bundle":{"name": "mux_case_9188527_out", "role": "ap_vld" }} , 
 	{ "name": "mux_case_8186523_out", "direction": "out", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "mux_case_8186523_out", "role": "default" }} , 
 	{ "name": "mux_case_8186523_out_ap_vld", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "outvld", "bundle":{"name": "mux_case_8186523_out", "role": "ap_vld" }} , 
 	{ "name": "mux_case_7184519_out", "direction": "out", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "mux_case_7184519_out", "role": "default" }} , 
 	{ "name": "mux_case_7184519_out_ap_vld", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "outvld", "bundle":{"name": "mux_case_7184519_out", "role": "ap_vld" }} , 
 	{ "name": "mux_case_6182515_out", "direction": "out", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "mux_case_6182515_out", "role": "default" }} , 
 	{ "name": "mux_case_6182515_out_ap_vld", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "outvld", "bundle":{"name": "mux_case_6182515_out", "role": "ap_vld" }} , 
 	{ "name": "mux_case_5180511_out", "direction": "out", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "mux_case_5180511_out", "role": "default" }} , 
 	{ "name": "mux_case_5180511_out_ap_vld", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "outvld", "bundle":{"name": "mux_case_5180511_out", "role": "ap_vld" }} , 
 	{ "name": "mux_case_4178507_out", "direction": "out", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "mux_case_4178507_out", "role": "default" }} , 
 	{ "name": "mux_case_4178507_out_ap_vld", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "outvld", "bundle":{"name": "mux_case_4178507_out", "role": "ap_vld" }} , 
 	{ "name": "mux_case_3176503_out", "direction": "out", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "mux_case_3176503_out", "role": "default" }} , 
 	{ "name": "mux_case_3176503_out_ap_vld", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "outvld", "bundle":{"name": "mux_case_3176503_out", "role": "ap_vld" }} , 
 	{ "name": "mux_case_2174499_out", "direction": "out", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "mux_case_2174499_out", "role": "default" }} , 
 	{ "name": "mux_case_2174499_out_ap_vld", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "outvld", "bundle":{"name": "mux_case_2174499_out", "role": "ap_vld" }} , 
 	{ "name": "mux_case_1172495_out", "direction": "out", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "mux_case_1172495_out", "role": "default" }} , 
 	{ "name": "mux_case_1172495_out_ap_vld", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "outvld", "bundle":{"name": "mux_case_1172495_out", "role": "ap_vld" }} , 
 	{ "name": "mux_case_0170491_out", "direction": "out", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "mux_case_0170491_out", "role": "default" }} , 
 	{ "name": "mux_case_0170491_out_ap_vld", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "outvld", "bundle":{"name": "mux_case_0170491_out", "role": "ap_vld" }} , 
 	{ "name": "mux_case_31168487_out", "direction": "out", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "mux_case_31168487_out", "role": "default" }} , 
 	{ "name": "mux_case_31168487_out_ap_vld", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "outvld", "bundle":{"name": "mux_case_31168487_out", "role": "ap_vld" }} , 
 	{ "name": "mux_case_30166483_out", "direction": "out", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "mux_case_30166483_out", "role": "default" }} , 
 	{ "name": "mux_case_30166483_out_ap_vld", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "outvld", "bundle":{"name": "mux_case_30166483_out", "role": "ap_vld" }} , 
 	{ "name": "mux_case_29164479_out", "direction": "out", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "mux_case_29164479_out", "role": "default" }} , 
 	{ "name": "mux_case_29164479_out_ap_vld", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "outvld", "bundle":{"name": "mux_case_29164479_out", "role": "ap_vld" }} , 
 	{ "name": "mux_case_28162475_out", "direction": "out", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "mux_case_28162475_out", "role": "default" }} , 
 	{ "name": "mux_case_28162475_out_ap_vld", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "outvld", "bundle":{"name": "mux_case_28162475_out", "role": "ap_vld" }} , 
 	{ "name": "mux_case_27160471_out", "direction": "out", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "mux_case_27160471_out", "role": "default" }} , 
 	{ "name": "mux_case_27160471_out_ap_vld", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "outvld", "bundle":{"name": "mux_case_27160471_out", "role": "ap_vld" }} , 
 	{ "name": "mux_case_26158467_out", "direction": "out", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "mux_case_26158467_out", "role": "default" }} , 
 	{ "name": "mux_case_26158467_out_ap_vld", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "outvld", "bundle":{"name": "mux_case_26158467_out", "role": "ap_vld" }} , 
 	{ "name": "mux_case_25156463_out", "direction": "out", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "mux_case_25156463_out", "role": "default" }} , 
 	{ "name": "mux_case_25156463_out_ap_vld", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "outvld", "bundle":{"name": "mux_case_25156463_out", "role": "ap_vld" }} , 
 	{ "name": "mux_case_24154459_out", "direction": "out", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "mux_case_24154459_out", "role": "default" }} , 
 	{ "name": "mux_case_24154459_out_ap_vld", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "outvld", "bundle":{"name": "mux_case_24154459_out", "role": "ap_vld" }} , 
 	{ "name": "mux_case_23152455_out", "direction": "out", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "mux_case_23152455_out", "role": "default" }} , 
 	{ "name": "mux_case_23152455_out_ap_vld", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "outvld", "bundle":{"name": "mux_case_23152455_out", "role": "ap_vld" }} , 
 	{ "name": "mux_case_22150451_out", "direction": "out", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "mux_case_22150451_out", "role": "default" }} , 
 	{ "name": "mux_case_22150451_out_ap_vld", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "outvld", "bundle":{"name": "mux_case_22150451_out", "role": "ap_vld" }} , 
 	{ "name": "mux_case_21148447_out", "direction": "out", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "mux_case_21148447_out", "role": "default" }} , 
 	{ "name": "mux_case_21148447_out_ap_vld", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "outvld", "bundle":{"name": "mux_case_21148447_out", "role": "ap_vld" }} , 
 	{ "name": "mux_case_20146443_out", "direction": "out", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "mux_case_20146443_out", "role": "default" }} , 
 	{ "name": "mux_case_20146443_out_ap_vld", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "outvld", "bundle":{"name": "mux_case_20146443_out", "role": "ap_vld" }} , 
 	{ "name": "mux_case_19144439_out", "direction": "out", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "mux_case_19144439_out", "role": "default" }} , 
 	{ "name": "mux_case_19144439_out_ap_vld", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "outvld", "bundle":{"name": "mux_case_19144439_out", "role": "ap_vld" }} , 
 	{ "name": "mux_case_18142435_out", "direction": "out", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "mux_case_18142435_out", "role": "default" }} , 
 	{ "name": "mux_case_18142435_out_ap_vld", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "outvld", "bundle":{"name": "mux_case_18142435_out", "role": "ap_vld" }} , 
 	{ "name": "mux_case_17140431_out", "direction": "out", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "mux_case_17140431_out", "role": "default" }} , 
 	{ "name": "mux_case_17140431_out_ap_vld", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "outvld", "bundle":{"name": "mux_case_17140431_out", "role": "ap_vld" }} , 
 	{ "name": "mux_case_16138427_out", "direction": "out", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "mux_case_16138427_out", "role": "default" }} , 
 	{ "name": "mux_case_16138427_out_ap_vld", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "outvld", "bundle":{"name": "mux_case_16138427_out", "role": "ap_vld" }} , 
 	{ "name": "mux_case_15136423_out", "direction": "out", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "mux_case_15136423_out", "role": "default" }} , 
 	{ "name": "mux_case_15136423_out_ap_vld", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "outvld", "bundle":{"name": "mux_case_15136423_out", "role": "ap_vld" }} , 
 	{ "name": "mux_case_14134419_out", "direction": "out", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "mux_case_14134419_out", "role": "default" }} , 
 	{ "name": "mux_case_14134419_out_ap_vld", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "outvld", "bundle":{"name": "mux_case_14134419_out", "role": "ap_vld" }} , 
 	{ "name": "mux_case_13132415_out", "direction": "out", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "mux_case_13132415_out", "role": "default" }} , 
 	{ "name": "mux_case_13132415_out_ap_vld", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "outvld", "bundle":{"name": "mux_case_13132415_out", "role": "ap_vld" }} , 
 	{ "name": "mux_case_12130411_out", "direction": "out", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "mux_case_12130411_out", "role": "default" }} , 
 	{ "name": "mux_case_12130411_out_ap_vld", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "outvld", "bundle":{"name": "mux_case_12130411_out", "role": "ap_vld" }} , 
 	{ "name": "mux_case_11128407_out", "direction": "out", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "mux_case_11128407_out", "role": "default" }} , 
 	{ "name": "mux_case_11128407_out_ap_vld", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "outvld", "bundle":{"name": "mux_case_11128407_out", "role": "ap_vld" }} , 
 	{ "name": "mux_case_10126403_out", "direction": "out", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "mux_case_10126403_out", "role": "default" }} , 
 	{ "name": "mux_case_10126403_out_ap_vld", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "outvld", "bundle":{"name": "mux_case_10126403_out", "role": "ap_vld" }} , 
 	{ "name": "mux_case_9124399_out", "direction": "out", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "mux_case_9124399_out", "role": "default" }} , 
 	{ "name": "mux_case_9124399_out_ap_vld", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "outvld", "bundle":{"name": "mux_case_9124399_out", "role": "ap_vld" }} , 
 	{ "name": "mux_case_8122395_out", "direction": "out", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "mux_case_8122395_out", "role": "default" }} , 
 	{ "name": "mux_case_8122395_out_ap_vld", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "outvld", "bundle":{"name": "mux_case_8122395_out", "role": "ap_vld" }} , 
 	{ "name": "mux_case_7120391_out", "direction": "out", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "mux_case_7120391_out", "role": "default" }} , 
 	{ "name": "mux_case_7120391_out_ap_vld", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "outvld", "bundle":{"name": "mux_case_7120391_out", "role": "ap_vld" }} , 
 	{ "name": "mux_case_6118387_out", "direction": "out", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "mux_case_6118387_out", "role": "default" }} , 
 	{ "name": "mux_case_6118387_out_ap_vld", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "outvld", "bundle":{"name": "mux_case_6118387_out", "role": "ap_vld" }} , 
 	{ "name": "mux_case_5116383_out", "direction": "out", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "mux_case_5116383_out", "role": "default" }} , 
 	{ "name": "mux_case_5116383_out_ap_vld", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "outvld", "bundle":{"name": "mux_case_5116383_out", "role": "ap_vld" }} , 
 	{ "name": "mux_case_4114379_out", "direction": "out", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "mux_case_4114379_out", "role": "default" }} , 
 	{ "name": "mux_case_4114379_out_ap_vld", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "outvld", "bundle":{"name": "mux_case_4114379_out", "role": "ap_vld" }} , 
 	{ "name": "mux_case_3112375_out", "direction": "out", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "mux_case_3112375_out", "role": "default" }} , 
 	{ "name": "mux_case_3112375_out_ap_vld", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "outvld", "bundle":{"name": "mux_case_3112375_out", "role": "ap_vld" }} , 
 	{ "name": "mux_case_2110371_out", "direction": "out", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "mux_case_2110371_out", "role": "default" }} , 
 	{ "name": "mux_case_2110371_out_ap_vld", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "outvld", "bundle":{"name": "mux_case_2110371_out", "role": "ap_vld" }} , 
 	{ "name": "mux_case_1108367_out", "direction": "out", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "mux_case_1108367_out", "role": "default" }} , 
 	{ "name": "mux_case_1108367_out_ap_vld", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "outvld", "bundle":{"name": "mux_case_1108367_out", "role": "ap_vld" }} , 
 	{ "name": "mux_case_0106363_out", "direction": "out", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "mux_case_0106363_out", "role": "default" }} , 
 	{ "name": "mux_case_0106363_out_ap_vld", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "outvld", "bundle":{"name": "mux_case_0106363_out", "role": "ap_vld" }} , 
 	{ "name": "grp_fu_756_p_din0", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "grp_fu_756_p_din0", "role": "default" }} , 
 	{ "name": "grp_fu_756_p_din1", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "grp_fu_756_p_din1", "role": "default" }} , 
 	{ "name": "grp_fu_756_p_opcode", "direction": "out", "datatype": "sc_lv", "bitwidth":2, "type": "signal", "bundle":{"name": "grp_fu_756_p_opcode", "role": "default" }} , 
 	{ "name": "grp_fu_756_p_dout0", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "grp_fu_756_p_dout0", "role": "default" }} , 
 	{ "name": "grp_fu_756_p_ce", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "grp_fu_756_p_ce", "role": "default" }} , 
 	{ "name": "grp_fu_3032_p_din0", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "grp_fu_3032_p_din0", "role": "default" }} , 
 	{ "name": "grp_fu_3032_p_din1", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "grp_fu_3032_p_din1", "role": "default" }} , 
 	{ "name": "grp_fu_3032_p_opcode", "direction": "out", "datatype": "sc_lv", "bitwidth":2, "type": "signal", "bundle":{"name": "grp_fu_3032_p_opcode", "role": "default" }} , 
 	{ "name": "grp_fu_3032_p_dout0", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "grp_fu_3032_p_dout0", "role": "default" }} , 
 	{ "name": "grp_fu_3032_p_ce", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "grp_fu_3032_p_ce", "role": "default" }} , 
 	{ "name": "grp_fu_760_p_din0", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "grp_fu_760_p_din0", "role": "default" }} , 
 	{ "name": "grp_fu_760_p_din1", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "grp_fu_760_p_din1", "role": "default" }} , 
 	{ "name": "grp_fu_760_p_dout0", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "grp_fu_760_p_dout0", "role": "default" }} , 
 	{ "name": "grp_fu_760_p_ce", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "grp_fu_760_p_ce", "role": "default" }} , 
 	{ "name": "grp_fu_3036_p_din0", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "grp_fu_3036_p_din0", "role": "default" }} , 
 	{ "name": "grp_fu_3036_p_din1", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "grp_fu_3036_p_din1", "role": "default" }} , 
 	{ "name": "grp_fu_3036_p_dout0", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "grp_fu_3036_p_dout0", "role": "default" }} , 
 	{ "name": "grp_fu_3036_p_ce", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "grp_fu_3036_p_ce", "role": "default" }} , 
 	{ "name": "grp_fu_3040_p_din0", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "grp_fu_3040_p_din0", "role": "default" }} , 
 	{ "name": "grp_fu_3040_p_din1", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "grp_fu_3040_p_din1", "role": "default" }} , 
 	{ "name": "grp_fu_3040_p_dout0", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "grp_fu_3040_p_dout0", "role": "default" }} , 
 	{ "name": "grp_fu_3040_p_ce", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "grp_fu_3040_p_ce", "role": "default" }} , 
 	{ "name": "grp_fu_3044_p_din0", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "grp_fu_3044_p_din0", "role": "default" }} , 
 	{ "name": "grp_fu_3044_p_din1", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "grp_fu_3044_p_din1", "role": "default" }} , 
 	{ "name": "grp_fu_3044_p_dout0", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "grp_fu_3044_p_dout0", "role": "default" }} , 
 	{ "name": "grp_fu_3044_p_ce", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "grp_fu_3044_p_ce", "role": "default" }} , 
 	{ "name": "grp_fu_764_p_din0", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "grp_fu_764_p_din0", "role": "default" }} , 
 	{ "name": "grp_fu_764_p_dout0", "direction": "in", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "grp_fu_764_p_dout0", "role": "default" }} , 
 	{ "name": "grp_fu_764_p_ce", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "grp_fu_764_p_ce", "role": "default" }} , 
 	{ "name": "grp_fu_3048_p_din0", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "grp_fu_3048_p_din0", "role": "default" }} , 
 	{ "name": "grp_fu_3048_p_dout0", "direction": "in", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "grp_fu_3048_p_dout0", "role": "default" }} , 
 	{ "name": "grp_fu_3048_p_ce", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "grp_fu_3048_p_ce", "role": "default" }} , 
 	{ "name": "grp_fu_767_p_din0", "direction": "out", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "grp_fu_767_p_din0", "role": "default" }} , 
 	{ "name": "grp_fu_767_p_dout0", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "grp_fu_767_p_dout0", "role": "default" }} , 
 	{ "name": "grp_fu_767_p_ce", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "grp_fu_767_p_ce", "role": "default" }} , 
 	{ "name": "grp_fu_770_p_din0", "direction": "out", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "grp_fu_770_p_din0", "role": "default" }} , 
 	{ "name": "grp_fu_770_p_dout0", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "grp_fu_770_p_dout0", "role": "default" }} , 
 	{ "name": "grp_fu_770_p_ce", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "grp_fu_770_p_ce", "role": "default" }}  ]}

set RtlHierarchyInfo {[
	{"ID" : "0", "Level" : "0", "Path" : "`AUTOTB_DUT_INST", "Parent" : "", "Child" : ["1", "2", "3"],
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
	{"ID" : "1", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.sparsemux_65_5_16_1_1_U111", "Parent" : "0"},
	{"ID" : "2", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.sparsemux_65_5_16_1_1_U112", "Parent" : "0"},
	{"ID" : "3", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.flow_control_loop_pipe_sequential_init_U", "Parent" : "0"}]}


set ArgLastReadFirstWriteLatency {
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
		mux_case_0106363_out {Type O LastRead -1 FirstWrite 36}}}

set hasDtUnsupportedChannel 0

set PerformanceInfo {[
	{"Name" : "Latency", "Min" : "70", "Max" : "70"}
	, {"Name" : "Interval", "Min" : "70", "Max" : "70"}
]}

set PipelineEnableSignalInfo {[
	{"Pipeline" : "0", "EnableSignal" : "ap_enable_pp0"}
]}

set Spec2ImplPortList { 
	mux_case_31232615_out { ap_vld {  { mux_case_31232615_out out_data 1 16 }  { mux_case_31232615_out_ap_vld out_vld 1 1 } } }
	mux_case_30230611_out { ap_vld {  { mux_case_30230611_out out_data 1 16 }  { mux_case_30230611_out_ap_vld out_vld 1 1 } } }
	mux_case_29228607_out { ap_vld {  { mux_case_29228607_out out_data 1 16 }  { mux_case_29228607_out_ap_vld out_vld 1 1 } } }
	mux_case_28226603_out { ap_vld {  { mux_case_28226603_out out_data 1 16 }  { mux_case_28226603_out_ap_vld out_vld 1 1 } } }
	mux_case_27224599_out { ap_vld {  { mux_case_27224599_out out_data 1 16 }  { mux_case_27224599_out_ap_vld out_vld 1 1 } } }
	mux_case_26222595_out { ap_vld {  { mux_case_26222595_out out_data 1 16 }  { mux_case_26222595_out_ap_vld out_vld 1 1 } } }
	mux_case_25220591_out { ap_vld {  { mux_case_25220591_out out_data 1 16 }  { mux_case_25220591_out_ap_vld out_vld 1 1 } } }
	mux_case_24218587_out { ap_vld {  { mux_case_24218587_out out_data 1 16 }  { mux_case_24218587_out_ap_vld out_vld 1 1 } } }
	mux_case_23216583_out { ap_vld {  { mux_case_23216583_out out_data 1 16 }  { mux_case_23216583_out_ap_vld out_vld 1 1 } } }
	mux_case_22214579_out { ap_vld {  { mux_case_22214579_out out_data 1 16 }  { mux_case_22214579_out_ap_vld out_vld 1 1 } } }
	mux_case_21212575_out { ap_vld {  { mux_case_21212575_out out_data 1 16 }  { mux_case_21212575_out_ap_vld out_vld 1 1 } } }
	mux_case_20210571_out { ap_vld {  { mux_case_20210571_out out_data 1 16 }  { mux_case_20210571_out_ap_vld out_vld 1 1 } } }
	mux_case_19208567_out { ap_vld {  { mux_case_19208567_out out_data 1 16 }  { mux_case_19208567_out_ap_vld out_vld 1 1 } } }
	mux_case_18206563_out { ap_vld {  { mux_case_18206563_out out_data 1 16 }  { mux_case_18206563_out_ap_vld out_vld 1 1 } } }
	mux_case_17204559_out { ap_vld {  { mux_case_17204559_out out_data 1 16 }  { mux_case_17204559_out_ap_vld out_vld 1 1 } } }
	mux_case_16202555_out { ap_vld {  { mux_case_16202555_out out_data 1 16 }  { mux_case_16202555_out_ap_vld out_vld 1 1 } } }
	mux_case_15200551_out { ap_vld {  { mux_case_15200551_out out_data 1 16 }  { mux_case_15200551_out_ap_vld out_vld 1 1 } } }
	mux_case_14198547_out { ap_vld {  { mux_case_14198547_out out_data 1 16 }  { mux_case_14198547_out_ap_vld out_vld 1 1 } } }
	mux_case_13196543_out { ap_vld {  { mux_case_13196543_out out_data 1 16 }  { mux_case_13196543_out_ap_vld out_vld 1 1 } } }
	mux_case_12194539_out { ap_vld {  { mux_case_12194539_out out_data 1 16 }  { mux_case_12194539_out_ap_vld out_vld 1 1 } } }
	mux_case_11192535_out { ap_vld {  { mux_case_11192535_out out_data 1 16 }  { mux_case_11192535_out_ap_vld out_vld 1 1 } } }
	mux_case_10190531_out { ap_vld {  { mux_case_10190531_out out_data 1 16 }  { mux_case_10190531_out_ap_vld out_vld 1 1 } } }
	mux_case_9188527_out { ap_vld {  { mux_case_9188527_out out_data 1 16 }  { mux_case_9188527_out_ap_vld out_vld 1 1 } } }
	mux_case_8186523_out { ap_vld {  { mux_case_8186523_out out_data 1 16 }  { mux_case_8186523_out_ap_vld out_vld 1 1 } } }
	mux_case_7184519_out { ap_vld {  { mux_case_7184519_out out_data 1 16 }  { mux_case_7184519_out_ap_vld out_vld 1 1 } } }
	mux_case_6182515_out { ap_vld {  { mux_case_6182515_out out_data 1 16 }  { mux_case_6182515_out_ap_vld out_vld 1 1 } } }
	mux_case_5180511_out { ap_vld {  { mux_case_5180511_out out_data 1 16 }  { mux_case_5180511_out_ap_vld out_vld 1 1 } } }
	mux_case_4178507_out { ap_vld {  { mux_case_4178507_out out_data 1 16 }  { mux_case_4178507_out_ap_vld out_vld 1 1 } } }
	mux_case_3176503_out { ap_vld {  { mux_case_3176503_out out_data 1 16 }  { mux_case_3176503_out_ap_vld out_vld 1 1 } } }
	mux_case_2174499_out { ap_vld {  { mux_case_2174499_out out_data 1 16 }  { mux_case_2174499_out_ap_vld out_vld 1 1 } } }
	mux_case_1172495_out { ap_vld {  { mux_case_1172495_out out_data 1 16 }  { mux_case_1172495_out_ap_vld out_vld 1 1 } } }
	mux_case_0170491_out { ap_vld {  { mux_case_0170491_out out_data 1 16 }  { mux_case_0170491_out_ap_vld out_vld 1 1 } } }
	mux_case_31168487_out { ap_vld {  { mux_case_31168487_out out_data 1 16 }  { mux_case_31168487_out_ap_vld out_vld 1 1 } } }
	mux_case_30166483_out { ap_vld {  { mux_case_30166483_out out_data 1 16 }  { mux_case_30166483_out_ap_vld out_vld 1 1 } } }
	mux_case_29164479_out { ap_vld {  { mux_case_29164479_out out_data 1 16 }  { mux_case_29164479_out_ap_vld out_vld 1 1 } } }
	mux_case_28162475_out { ap_vld {  { mux_case_28162475_out out_data 1 16 }  { mux_case_28162475_out_ap_vld out_vld 1 1 } } }
	mux_case_27160471_out { ap_vld {  { mux_case_27160471_out out_data 1 16 }  { mux_case_27160471_out_ap_vld out_vld 1 1 } } }
	mux_case_26158467_out { ap_vld {  { mux_case_26158467_out out_data 1 16 }  { mux_case_26158467_out_ap_vld out_vld 1 1 } } }
	mux_case_25156463_out { ap_vld {  { mux_case_25156463_out out_data 1 16 }  { mux_case_25156463_out_ap_vld out_vld 1 1 } } }
	mux_case_24154459_out { ap_vld {  { mux_case_24154459_out out_data 1 16 }  { mux_case_24154459_out_ap_vld out_vld 1 1 } } }
	mux_case_23152455_out { ap_vld {  { mux_case_23152455_out out_data 1 16 }  { mux_case_23152455_out_ap_vld out_vld 1 1 } } }
	mux_case_22150451_out { ap_vld {  { mux_case_22150451_out out_data 1 16 }  { mux_case_22150451_out_ap_vld out_vld 1 1 } } }
	mux_case_21148447_out { ap_vld {  { mux_case_21148447_out out_data 1 16 }  { mux_case_21148447_out_ap_vld out_vld 1 1 } } }
	mux_case_20146443_out { ap_vld {  { mux_case_20146443_out out_data 1 16 }  { mux_case_20146443_out_ap_vld out_vld 1 1 } } }
	mux_case_19144439_out { ap_vld {  { mux_case_19144439_out out_data 1 16 }  { mux_case_19144439_out_ap_vld out_vld 1 1 } } }
	mux_case_18142435_out { ap_vld {  { mux_case_18142435_out out_data 1 16 }  { mux_case_18142435_out_ap_vld out_vld 1 1 } } }
	mux_case_17140431_out { ap_vld {  { mux_case_17140431_out out_data 1 16 }  { mux_case_17140431_out_ap_vld out_vld 1 1 } } }
	mux_case_16138427_out { ap_vld {  { mux_case_16138427_out out_data 1 16 }  { mux_case_16138427_out_ap_vld out_vld 1 1 } } }
	mux_case_15136423_out { ap_vld {  { mux_case_15136423_out out_data 1 16 }  { mux_case_15136423_out_ap_vld out_vld 1 1 } } }
	mux_case_14134419_out { ap_vld {  { mux_case_14134419_out out_data 1 16 }  { mux_case_14134419_out_ap_vld out_vld 1 1 } } }
	mux_case_13132415_out { ap_vld {  { mux_case_13132415_out out_data 1 16 }  { mux_case_13132415_out_ap_vld out_vld 1 1 } } }
	mux_case_12130411_out { ap_vld {  { mux_case_12130411_out out_data 1 16 }  { mux_case_12130411_out_ap_vld out_vld 1 1 } } }
	mux_case_11128407_out { ap_vld {  { mux_case_11128407_out out_data 1 16 }  { mux_case_11128407_out_ap_vld out_vld 1 1 } } }
	mux_case_10126403_out { ap_vld {  { mux_case_10126403_out out_data 1 16 }  { mux_case_10126403_out_ap_vld out_vld 1 1 } } }
	mux_case_9124399_out { ap_vld {  { mux_case_9124399_out out_data 1 16 }  { mux_case_9124399_out_ap_vld out_vld 1 1 } } }
	mux_case_8122395_out { ap_vld {  { mux_case_8122395_out out_data 1 16 }  { mux_case_8122395_out_ap_vld out_vld 1 1 } } }
	mux_case_7120391_out { ap_vld {  { mux_case_7120391_out out_data 1 16 }  { mux_case_7120391_out_ap_vld out_vld 1 1 } } }
	mux_case_6118387_out { ap_vld {  { mux_case_6118387_out out_data 1 16 }  { mux_case_6118387_out_ap_vld out_vld 1 1 } } }
	mux_case_5116383_out { ap_vld {  { mux_case_5116383_out out_data 1 16 }  { mux_case_5116383_out_ap_vld out_vld 1 1 } } }
	mux_case_4114379_out { ap_vld {  { mux_case_4114379_out out_data 1 16 }  { mux_case_4114379_out_ap_vld out_vld 1 1 } } }
	mux_case_3112375_out { ap_vld {  { mux_case_3112375_out out_data 1 16 }  { mux_case_3112375_out_ap_vld out_vld 1 1 } } }
	mux_case_2110371_out { ap_vld {  { mux_case_2110371_out out_data 1 16 }  { mux_case_2110371_out_ap_vld out_vld 1 1 } } }
	mux_case_1108367_out { ap_vld {  { mux_case_1108367_out out_data 1 16 }  { mux_case_1108367_out_ap_vld out_vld 1 1 } } }
	mux_case_0106363_out { ap_vld {  { mux_case_0106363_out out_data 1 16 }  { mux_case_0106363_out_ap_vld out_vld 1 1 } } }
}
