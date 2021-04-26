#/bin/bash

for g in 10 20 30 ; do python create_input_rolling_scenarios.py $HOME/ts4uc/data/day_ahead/${g}gen/30min $HOME/ts4uc/data/hpc_params/input_203_g${g}_scenarios.txt ; done

# set today's date to use as save directory
date=$(date +"%y-%m-%d")

#
rho=05
tree_search_func_name="ida_star"
for g in 10 20 30; 
  do for t in 60;
    do for heuristic_method in "advanced_priority_list";
      do let secs=$t*48*3+100 && \
         time=$(date -d@$secs -u +%H:%M:%S) && \
         qsub -l h_rt=$time submit_rolling_tree_search_scenarios.sh $HOME/Scratch/results/${date}_203_scenarios/rolling_guided_${tree_search_func_name}/g${g}/g${g}_t${t}_p${rho}_${heuristic_method} \
														 $HOME/AISO_HPC/best_policies/g${g}/params.json \
														 $HOME/ts4uc/data/day_ahead/${g}gen/30min/env_params.json \
														 $HOME/AISO_HPC/best_policies/g${g}/ac_final.pt \
														 ${t} \
														 0.${rho} \
														 ${tree_search_func_name} \
														 $HOME/ts4uc/data/hpc_params/input_203_g${g}_scenarios.txt \
														 $heuristic_method ;
	  done ;
	done ;
  done ;
