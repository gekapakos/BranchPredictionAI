set ModuleHierarchy {[{
"Name" : "run_model","ID" : "0","Type" : "sequential",
"SubInsts" : [
	{"Name" : "grp_run_all_slices_unrolled_fu_263","ID" : "1","Type" : "sequential",
		"SubInsts" : [
		{"Name" : "grp_run_all_slices_unrolled_Pipeline_MergedLoop_fu_96","ID" : "2","Type" : "sequential",
			"SubLoops" : [
			{"Name" : "MergedLoop","ID" : "3","Type" : "pipeline"},]},
		{"Name" : "grp_lstm_forward_unidir_fu_102","ID" : "4","Type" : "sequential",
			"SubInsts" : [
			{"Name" : "grp_lstm_forward_unidir_Pipeline_VITIS_LOOP_145_1_fu_356","ID" : "5","Type" : "sequential",
				"SubLoops" : [
				{"Name" : "VITIS_LOOP_145_1","ID" : "6","Type" : "pipeline"},]},],
			"SubLoops" : [
			{"Name" : "Loop1LSTM","ID" : "7","Type" : "no",
			"SubInsts" : [
			{"Name" : "grp_lstm_forward_unidir_Pipeline_Loop2_1LSTM_fu_364","ID" : "8","Type" : "sequential",
					"SubLoops" : [
					{"Name" : "Loop2_1LSTM","ID" : "9","Type" : "pipeline"},]},
			{"Name" : "grp_lstm_forward_unidir_Pipeline_Loop2_4LSTM_fu_373","ID" : "10","Type" : "sequential",
					"SubLoops" : [
					{"Name" : "Loop2_4LSTM","ID" : "11","Type" : "pipeline"},]},],
			"SubLoops" : [
			{"Name" : "Loop2_2LSTM_Loop3_2_1LSTM","ID" : "12","Type" : "pipeline"},
			{"Name" : "Loop2_3LSTM_Loop3_3_1LSTM","ID" : "13","Type" : "pipeline"},]},]},
		{"Name" : "grp_run_all_slices_unrolled_Pipeline_Loop_BN_fu_124","ID" : "14","Type" : "sequential",
			"SubLoops" : [
			{"Name" : "Loop_BN","ID" : "15","Type" : "pipeline"},]},
		{"Name" : "grp_conv_bn_act_pool_fu_134","ID" : "16","Type" : "sequential",
			"SubInsts" : [
			{"Name" : "grp_conv_bn_act_pool_Pipeline_BNParamsLoop_fu_649","ID" : "17","Type" : "sequential",
				"SubLoops" : [
				{"Name" : "BNParamsLoop","ID" : "18","Type" : "pipeline"},]},],
			"SubLoops" : [
			{"Name" : "Loop1Big","ID" : "19","Type" : "no",
			"SubInsts" : [
			{"Name" : "grp_conv_bn_act_pool_Pipeline_Loop3_2Big_fu_717","ID" : "20","Type" : "sequential",
					"SubLoops" : [
					{"Name" : "Loop3_2Big","ID" : "21","Type" : "pipeline"},]},],
			"SubLoops" : [
			{"Name" : "Loop2Big","ID" : "22","Type" : "no",
				"SubLoops" : [
				{"Name" : "Loop3_1Big_Loop4Big","ID" : "23","Type" : "pipeline"},]},]},]},
		{"Name" : "grp_run_all_slices_unrolled_Pipeline_MergedLoop0_fu_144","ID" : "24","Type" : "sequential",
			"SubLoops" : [
			{"Name" : "MergedLoop0","ID" : "25","Type" : "pipeline"},]},
		{"Name" : "grp_run_all_slices_unrolled_Pipeline_Loop_BN2_fu_152","ID" : "26","Type" : "sequential",
			"SubLoops" : [
			{"Name" : "Loop_BN","ID" : "27","Type" : "pipeline"},]},
		{"Name" : "grp_conv_bn_act_pool_2_fu_162","ID" : "28","Type" : "sequential",
			"SubInsts" : [
			{"Name" : "grp_conv_bn_act_pool_2_Pipeline_BNParamsLoop_fu_653","ID" : "29","Type" : "sequential",
				"SubLoops" : [
				{"Name" : "BNParamsLoop","ID" : "30","Type" : "pipeline"},]},],
			"SubLoops" : [
			{"Name" : "Loop1Big","ID" : "31","Type" : "no",
			"SubInsts" : [
			{"Name" : "grp_conv_bn_act_pool_2_Pipeline_Loop3_2Big_fu_721","ID" : "32","Type" : "sequential",
					"SubLoops" : [
					{"Name" : "Loop3_2Big","ID" : "33","Type" : "pipeline"},]},],
			"SubLoops" : [
			{"Name" : "Loop2Big","ID" : "34","Type" : "no",
				"SubLoops" : [
				{"Name" : "Loop3_1Big_Loop4Big","ID" : "35","Type" : "pipeline"},]},]},]},
		{"Name" : "grp_run_all_slices_unrolled_Pipeline_MergedLoop1_fu_172","ID" : "36","Type" : "sequential",
			"SubLoops" : [
			{"Name" : "MergedLoop1","ID" : "37","Type" : "pipeline"},]},
		{"Name" : "grp_run_all_slices_unrolled_Pipeline_Loop_BN3_fu_180","ID" : "38","Type" : "sequential",
			"SubLoops" : [
			{"Name" : "Loop_BN","ID" : "39","Type" : "pipeline"},]},
		{"Name" : "grp_conv_bn_act_pool_3_fu_190","ID" : "40","Type" : "sequential",
			"SubInsts" : [
			{"Name" : "grp_conv_bn_act_pool_3_Pipeline_BNParamsLoop_fu_653","ID" : "41","Type" : "sequential",
				"SubLoops" : [
				{"Name" : "BNParamsLoop","ID" : "42","Type" : "pipeline"},]},],
			"SubLoops" : [
			{"Name" : "Loop1Big","ID" : "43","Type" : "no",
			"SubInsts" : [
			{"Name" : "grp_conv_bn_act_pool_3_Pipeline_Loop3_2Big_fu_721","ID" : "44","Type" : "sequential",
					"SubLoops" : [
					{"Name" : "Loop3_2Big","ID" : "45","Type" : "pipeline"},]},],
			"SubLoops" : [
			{"Name" : "Loop2Big","ID" : "46","Type" : "no",
				"SubLoops" : [
				{"Name" : "Loop3_1Big_Loop4Big","ID" : "47","Type" : "pipeline"},]},]},]},
		{"Name" : "grp_run_all_slices_unrolled_Pipeline_MergedLoop2_fu_200","ID" : "48","Type" : "sequential",
			"SubLoops" : [
			{"Name" : "MergedLoop2","ID" : "49","Type" : "pipeline"},]},
		{"Name" : "grp_run_all_slices_unrolled_Pipeline_Loop_BN4_fu_208","ID" : "50","Type" : "sequential",
			"SubLoops" : [
			{"Name" : "Loop_BN","ID" : "51","Type" : "pipeline"},]},
		{"Name" : "grp_conv_bn_act_pool_4_fu_218","ID" : "52","Type" : "sequential",
			"SubInsts" : [
			{"Name" : "grp_conv_bn_act_pool_4_Pipeline_BNParamsLoop_fu_653","ID" : "53","Type" : "sequential",
				"SubLoops" : [
				{"Name" : "BNParamsLoop","ID" : "54","Type" : "pipeline"},]},],
			"SubLoops" : [
			{"Name" : "Loop1Big","ID" : "55","Type" : "no",
			"SubInsts" : [
			{"Name" : "grp_conv_bn_act_pool_4_Pipeline_Loop3_2Big_fu_721","ID" : "56","Type" : "sequential",
					"SubLoops" : [
					{"Name" : "Loop3_2Big","ID" : "57","Type" : "pipeline"},]},],
			"SubLoops" : [
			{"Name" : "Loop2Big","ID" : "58","Type" : "no",
				"SubLoops" : [
				{"Name" : "Loop3_1Big_Loop4Big","ID" : "59","Type" : "pipeline"},]},]},]},
		{"Name" : "grp_run_all_slices_unrolled_Pipeline_MergedLoop3_fu_228","ID" : "60","Type" : "sequential",
			"SubLoops" : [
			{"Name" : "MergedLoop3","ID" : "61","Type" : "pipeline"},]},
		{"Name" : "grp_run_all_slices_unrolled_Pipeline_Loop_BN5_fu_236","ID" : "62","Type" : "sequential",
			"SubLoops" : [
			{"Name" : "Loop_BN","ID" : "63","Type" : "pipeline"},]},
		{"Name" : "grp_run_all_slices_unrolled_Pipeline_MergedLoop4_fu_246","ID" : "64","Type" : "sequential",
			"SubLoops" : [
			{"Name" : "MergedLoop4","ID" : "65","Type" : "pipeline"},]},]},
	{"Name" : "grp_run_model_Pipeline_Loop_BN_fu_331","ID" : "66","Type" : "sequential",
		"SubLoops" : [
		{"Name" : "Loop_BN","ID" : "67","Type" : "pipeline"},]},
	{"Name" : "grp_run_model_Pipeline_ReLULoop1_fu_340","ID" : "68","Type" : "sequential",
		"SubLoops" : [
		{"Name" : "ReLULoop1","ID" : "69","Type" : "pipeline"},]},
	{"Name" : "grp_run_model_Pipeline_Loop_BN1_fu_345","ID" : "70","Type" : "sequential",
		"SubLoops" : [
		{"Name" : "Loop_BN","ID" : "71","Type" : "pipeline"},]},
	{"Name" : "grp_run_model_Pipeline_ReLULoop2_fu_354","ID" : "72","Type" : "sequential",
		"SubLoops" : [
		{"Name" : "ReLULoop2","ID" : "73","Type" : "pipeline"},]},],
"SubLoops" : [
	{"Name" : "OuterLoopDense_InnerLoopDense","ID" : "74","Type" : "pipeline"},
	{"Name" : "OuterLoopDense_InnerLoopDense","ID" : "75","Type" : "pipeline"},]
}]}