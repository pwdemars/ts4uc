#/bin/bash

# set today's date to use as save directory
date=$(date +"%y-%m-%d")

#
rho=02
epoch=270000
seed=100
tree_search_func_name="ida_star"
for g in 100; 
  do for t in 60;
    do for heuristic_method in "advanced_priority_list";
      do let secs=$t*48*3+500 && \
         time=$(date -d@$secs -u +%H:%M:%S) && \
         qsub -l h_rt=$time submit_anytime_tree_search.sh $HOME/Scratch/results/${date}_206/guided_${tree_search_func_name}/g${g}/g${g}_t${t}_p${rho}_${heuristic_method}_e${epoch} \
														 $HOME/AISO_HPC/best_policies/g${g}/params.json \
														 $HOME/ts4uc/data/day_ahead/${g}gen/30min/env_params.json \
														 $HOME/AISO_HPC/best_policies/g${g}/ac_${epoch}.pt \
														 ${t} \
														 0.${rho} \
														 ${tree_search_func_name} \
														 $HOME/ts4uc/data/hpc_params/input_day_ahead_g${g}.txt \
														 $heuristic_method \
														 $seed ;
	  done ;
	done ;
  done ;
