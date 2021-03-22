#/bin/bash

# set today's date to use as save directory
date=$(date +"%y-%m-%d")

for h in "check_lost_load" "priority_list" "pl_plus_ll";
do qsub -l h_rt=2:00:00 test_tree_search.sh $HOME/Scratch/results/${date}/guided_astar/g5/feb4_g5_d30_v1_h4_p05_${h} $HOME/AISO_HPC/feb4/g5/g5_d30_v1.json $HOME/AISO_HPC/mar14/5_env_params.json $HOME/Scratch/results/feb4_g5_d30_v1/ac_final.pt 4 0.05 a_star $HOME/AISO_HPC/AISO/input_g5_d30.txt $h ; 
done