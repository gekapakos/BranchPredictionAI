#!/bin/sh
lli=${LLVMINTERP-lli}
exec $lli \
    /home/gkapakos/Desktop/ECE/10th_Semester/Architecture_of_Parallel_Systems/Project/BranchPredictionAI/C_Implementation/BranchPredictionAI/solution1/.autopilot/db/a.g.bc ${1+"$@"}
