# This script segment is generated automatically by AutoPilot

# clear list
if {${::AESL::PGuard_autoexp_gen}} {
    cg_default_interface_gen_dc_begin
    cg_default_interface_gen_bundle_begin
    AESL_LIB_XILADAPTER::native_axis_begin
}

# XIL_BRAM:
if {${::AESL::PGuard_autoexp_gen}} {
if {[info proc ::AESL_LIB_XILADAPTER::xil_bram_gen] == "::AESL_LIB_XILADAPTER::xil_bram_gen"} {
eval "::AESL_LIB_XILADAPTER::xil_bram_gen { \
    id 507 \
    name U_slice \
    reset_level 1 \
    sync_rst true \
    dir O \
    corename U_slice \
    op interface \
    ports { U_slice_address0 { O 9 vector } U_slice_ce0 { O 1 bit } U_slice_we0 { O 1 bit } U_slice_d0 { O 16 vector } } \
} "
} else {
puts "@W \[IMPL-110\] Cannot find bus interface model in the library. Ignored generation of bus interface for 'U_slice'"
}
}


# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 474 \
    name pool_acc \
    type other \
    dir IO \
    reset_level 1 \
    sync_rst true \
    corename dc_pool_acc \
    op interface \
    ports { pool_acc_i { I 16 vector } pool_acc_o { O 16 vector } pool_acc_o_ap_vld { O 1 bit } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 475 \
    name pool_acc_31 \
    type other \
    dir IO \
    reset_level 1 \
    sync_rst true \
    corename dc_pool_acc_31 \
    op interface \
    ports { pool_acc_31_i { I 16 vector } pool_acc_31_o { O 16 vector } pool_acc_31_o_ap_vld { O 1 bit } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 476 \
    name pool_acc_30 \
    type other \
    dir IO \
    reset_level 1 \
    sync_rst true \
    corename dc_pool_acc_30 \
    op interface \
    ports { pool_acc_30_i { I 16 vector } pool_acc_30_o { O 16 vector } pool_acc_30_o_ap_vld { O 1 bit } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 477 \
    name pool_acc_29 \
    type other \
    dir IO \
    reset_level 1 \
    sync_rst true \
    corename dc_pool_acc_29 \
    op interface \
    ports { pool_acc_29_i { I 16 vector } pool_acc_29_o { O 16 vector } pool_acc_29_o_ap_vld { O 1 bit } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 478 \
    name pool_acc_28 \
    type other \
    dir IO \
    reset_level 1 \
    sync_rst true \
    corename dc_pool_acc_28 \
    op interface \
    ports { pool_acc_28_i { I 16 vector } pool_acc_28_o { O 16 vector } pool_acc_28_o_ap_vld { O 1 bit } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 479 \
    name pool_acc_27 \
    type other \
    dir IO \
    reset_level 1 \
    sync_rst true \
    corename dc_pool_acc_27 \
    op interface \
    ports { pool_acc_27_i { I 16 vector } pool_acc_27_o { O 16 vector } pool_acc_27_o_ap_vld { O 1 bit } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 480 \
    name pool_acc_26 \
    type other \
    dir IO \
    reset_level 1 \
    sync_rst true \
    corename dc_pool_acc_26 \
    op interface \
    ports { pool_acc_26_i { I 16 vector } pool_acc_26_o { O 16 vector } pool_acc_26_o_ap_vld { O 1 bit } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 481 \
    name pool_acc_25 \
    type other \
    dir IO \
    reset_level 1 \
    sync_rst true \
    corename dc_pool_acc_25 \
    op interface \
    ports { pool_acc_25_i { I 16 vector } pool_acc_25_o { O 16 vector } pool_acc_25_o_ap_vld { O 1 bit } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 482 \
    name pool_acc_24 \
    type other \
    dir IO \
    reset_level 1 \
    sync_rst true \
    corename dc_pool_acc_24 \
    op interface \
    ports { pool_acc_24_i { I 16 vector } pool_acc_24_o { O 16 vector } pool_acc_24_o_ap_vld { O 1 bit } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 483 \
    name pool_acc_23 \
    type other \
    dir IO \
    reset_level 1 \
    sync_rst true \
    corename dc_pool_acc_23 \
    op interface \
    ports { pool_acc_23_i { I 16 vector } pool_acc_23_o { O 16 vector } pool_acc_23_o_ap_vld { O 1 bit } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 484 \
    name pool_acc_22 \
    type other \
    dir IO \
    reset_level 1 \
    sync_rst true \
    corename dc_pool_acc_22 \
    op interface \
    ports { pool_acc_22_i { I 16 vector } pool_acc_22_o { O 16 vector } pool_acc_22_o_ap_vld { O 1 bit } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 485 \
    name pool_acc_21 \
    type other \
    dir IO \
    reset_level 1 \
    sync_rst true \
    corename dc_pool_acc_21 \
    op interface \
    ports { pool_acc_21_i { I 16 vector } pool_acc_21_o { O 16 vector } pool_acc_21_o_ap_vld { O 1 bit } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 486 \
    name pool_acc_20 \
    type other \
    dir IO \
    reset_level 1 \
    sync_rst true \
    corename dc_pool_acc_20 \
    op interface \
    ports { pool_acc_20_i { I 16 vector } pool_acc_20_o { O 16 vector } pool_acc_20_o_ap_vld { O 1 bit } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 487 \
    name pool_acc_19 \
    type other \
    dir IO \
    reset_level 1 \
    sync_rst true \
    corename dc_pool_acc_19 \
    op interface \
    ports { pool_acc_19_i { I 16 vector } pool_acc_19_o { O 16 vector } pool_acc_19_o_ap_vld { O 1 bit } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 488 \
    name pool_acc_18 \
    type other \
    dir IO \
    reset_level 1 \
    sync_rst true \
    corename dc_pool_acc_18 \
    op interface \
    ports { pool_acc_18_i { I 16 vector } pool_acc_18_o { O 16 vector } pool_acc_18_o_ap_vld { O 1 bit } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 489 \
    name pool_acc_17 \
    type other \
    dir IO \
    reset_level 1 \
    sync_rst true \
    corename dc_pool_acc_17 \
    op interface \
    ports { pool_acc_17_i { I 16 vector } pool_acc_17_o { O 16 vector } pool_acc_17_o_ap_vld { O 1 bit } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 490 \
    name pool_acc_16 \
    type other \
    dir IO \
    reset_level 1 \
    sync_rst true \
    corename dc_pool_acc_16 \
    op interface \
    ports { pool_acc_16_i { I 16 vector } pool_acc_16_o { O 16 vector } pool_acc_16_o_ap_vld { O 1 bit } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 491 \
    name pool_acc_15 \
    type other \
    dir IO \
    reset_level 1 \
    sync_rst true \
    corename dc_pool_acc_15 \
    op interface \
    ports { pool_acc_15_i { I 16 vector } pool_acc_15_o { O 16 vector } pool_acc_15_o_ap_vld { O 1 bit } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 492 \
    name pool_acc_14 \
    type other \
    dir IO \
    reset_level 1 \
    sync_rst true \
    corename dc_pool_acc_14 \
    op interface \
    ports { pool_acc_14_i { I 16 vector } pool_acc_14_o { O 16 vector } pool_acc_14_o_ap_vld { O 1 bit } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 493 \
    name pool_acc_13 \
    type other \
    dir IO \
    reset_level 1 \
    sync_rst true \
    corename dc_pool_acc_13 \
    op interface \
    ports { pool_acc_13_i { I 16 vector } pool_acc_13_o { O 16 vector } pool_acc_13_o_ap_vld { O 1 bit } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 494 \
    name pool_acc_12 \
    type other \
    dir IO \
    reset_level 1 \
    sync_rst true \
    corename dc_pool_acc_12 \
    op interface \
    ports { pool_acc_12_i { I 16 vector } pool_acc_12_o { O 16 vector } pool_acc_12_o_ap_vld { O 1 bit } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 495 \
    name pool_acc_11 \
    type other \
    dir IO \
    reset_level 1 \
    sync_rst true \
    corename dc_pool_acc_11 \
    op interface \
    ports { pool_acc_11_i { I 16 vector } pool_acc_11_o { O 16 vector } pool_acc_11_o_ap_vld { O 1 bit } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 496 \
    name pool_acc_10 \
    type other \
    dir IO \
    reset_level 1 \
    sync_rst true \
    corename dc_pool_acc_10 \
    op interface \
    ports { pool_acc_10_i { I 16 vector } pool_acc_10_o { O 16 vector } pool_acc_10_o_ap_vld { O 1 bit } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 497 \
    name pool_acc_9 \
    type other \
    dir IO \
    reset_level 1 \
    sync_rst true \
    corename dc_pool_acc_9 \
    op interface \
    ports { pool_acc_9_i { I 16 vector } pool_acc_9_o { O 16 vector } pool_acc_9_o_ap_vld { O 1 bit } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 498 \
    name pool_acc_8 \
    type other \
    dir IO \
    reset_level 1 \
    sync_rst true \
    corename dc_pool_acc_8 \
    op interface \
    ports { pool_acc_8_i { I 16 vector } pool_acc_8_o { O 16 vector } pool_acc_8_o_ap_vld { O 1 bit } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 499 \
    name pool_acc_7 \
    type other \
    dir IO \
    reset_level 1 \
    sync_rst true \
    corename dc_pool_acc_7 \
    op interface \
    ports { pool_acc_7_i { I 16 vector } pool_acc_7_o { O 16 vector } pool_acc_7_o_ap_vld { O 1 bit } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 500 \
    name pool_acc_6 \
    type other \
    dir IO \
    reset_level 1 \
    sync_rst true \
    corename dc_pool_acc_6 \
    op interface \
    ports { pool_acc_6_i { I 16 vector } pool_acc_6_o { O 16 vector } pool_acc_6_o_ap_vld { O 1 bit } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 501 \
    name pool_acc_5 \
    type other \
    dir IO \
    reset_level 1 \
    sync_rst true \
    corename dc_pool_acc_5 \
    op interface \
    ports { pool_acc_5_i { I 16 vector } pool_acc_5_o { O 16 vector } pool_acc_5_o_ap_vld { O 1 bit } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 502 \
    name pool_acc_4 \
    type other \
    dir IO \
    reset_level 1 \
    sync_rst true \
    corename dc_pool_acc_4 \
    op interface \
    ports { pool_acc_4_i { I 16 vector } pool_acc_4_o { O 16 vector } pool_acc_4_o_ap_vld { O 1 bit } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 503 \
    name pool_acc_3 \
    type other \
    dir IO \
    reset_level 1 \
    sync_rst true \
    corename dc_pool_acc_3 \
    op interface \
    ports { pool_acc_3_i { I 16 vector } pool_acc_3_o { O 16 vector } pool_acc_3_o_ap_vld { O 1 bit } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 504 \
    name pool_acc_2 \
    type other \
    dir IO \
    reset_level 1 \
    sync_rst true \
    corename dc_pool_acc_2 \
    op interface \
    ports { pool_acc_2_i { I 16 vector } pool_acc_2_o { O 16 vector } pool_acc_2_o_ap_vld { O 1 bit } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 505 \
    name pool_acc_1 \
    type other \
    dir IO \
    reset_level 1 \
    sync_rst true \
    corename dc_pool_acc_1 \
    op interface \
    ports { pool_acc_1_i { I 16 vector } pool_acc_1_o { O 16 vector } pool_acc_1_o_ap_vld { O 1 bit } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 506 \
    name mul2 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_mul2 \
    op interface \
    ports { mul2 { I 9 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id -1 \
    name ap_ctrl \
    type ap_ctrl \
    reset_level 1 \
    sync_rst true \
    corename ap_ctrl \
    op interface \
    ports { ap_start { I 1 bit } ap_ready { O 1 bit } ap_done { O 1 bit } ap_idle { O 1 bit } } \
} "
}


# Adapter definition:
set PortName ap_clk
set DataWd 1 
if {${::AESL::PGuard_autoexp_gen}} {
if {[info proc cg_default_interface_gen_clock] == "cg_default_interface_gen_clock"} {
eval "cg_default_interface_gen_clock { \
    id -2 \
    name ${PortName} \
    reset_level 1 \
    sync_rst true \
    corename apif_ap_clk \
    data_wd ${DataWd} \
    op interface \
}"
} else {
puts "@W \[IMPL-113\] Cannot find bus interface model in the library. Ignored generation of bus interface for '${PortName}'"
}
}


# Adapter definition:
set PortName ap_rst
set DataWd 1 
if {${::AESL::PGuard_autoexp_gen}} {
if {[info proc cg_default_interface_gen_reset] == "cg_default_interface_gen_reset"} {
eval "cg_default_interface_gen_reset { \
    id -3 \
    name ${PortName} \
    reset_level 1 \
    sync_rst true \
    corename apif_ap_rst \
    data_wd ${DataWd} \
    op interface \
}"
} else {
puts "@W \[IMPL-114\] Cannot find bus interface model in the library. Ignored generation of bus interface for '${PortName}'"
}
}



# merge
if {${::AESL::PGuard_autoexp_gen}} {
    cg_default_interface_gen_dc_end
    cg_default_interface_gen_bundle_end
    AESL_LIB_XILADAPTER::native_axis_end
}


# flow_control definition:
set InstName run_model_flow_control_loop_pipe_sequential_init_U
set CompName run_model_flow_control_loop_pipe_sequential_init
set name flow_control_loop_pipe_sequential_init
if {${::AESL::PGuard_autocg_gen} && ${::AESL::PGuard_autocg_ipmgen}} {
if {[info proc ::AESL_LIB_VIRTEX::xil_gen_UPC_flow_control] == "::AESL_LIB_VIRTEX::xil_gen_UPC_flow_control"} {
eval "::AESL_LIB_VIRTEX::xil_gen_UPC_flow_control { \
    name ${name} \
    prefix run_model_ \
}"
} else {
puts "@W \[IMPL-107\] Cannot find ::AESL_LIB_VIRTEX::xil_gen_UPC_flow_control, check your platform lib"
}
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler $CompName BINDTYPE interface TYPE internal_upc_flow_control INSTNAME $InstName
}


