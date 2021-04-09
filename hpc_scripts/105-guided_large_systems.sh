#/bin/bash

# set today's date to use as save directory
date=$(date +"%y-%m-%d")

# guided search, g={10,20,30}, H=4, rho=0.05
for g in 10 20 30;
do qsub -l h_rt=24:00:00 submit_tree_search.sh $HOME/Scratch/results/${date}_105/g${g} $HOME/AISO_HPC/best_policies/g${g}/params.json $HOME/ts4uc/data/day_ahead/${g}gen/30min/env_params.json $HOME/AISO_HPC/best_policies/g${g}/ac_final.pt 4 0.05 uniform_cost_search $HOME/AISO_HPC/AISO/input_g${g}_d30.txt ;
done ;
