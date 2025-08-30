############################################################
## This file is generated automatically by Vivado HLS.
## Please DO NOT edit it.
## Copyright (C) 1986-2019 Xilinx, Inc. All Rights Reserved.
############################################################
open_project BranchPredictionAI
set_top main_5_slices.cpp
add_files head_weights.h
add_files main_5_slices.cpp
add_files slice0_weights.h
add_files slice1_weights.h
add_files slice2_weights.h
add_files slice3_weights.h
add_files slice4_weights.h
open_solution "solution1"
set_part {xc7z020clg484-1} -tool vivado
create_clock -period 10 -name default
#source "./BranchPredictionAI/solution1/directives.tcl"
#csim_design
csynth_design
#cosim_design
export_design -format ip_catalog
