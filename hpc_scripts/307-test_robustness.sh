#/bin/bash

# set today's date to use as save directory
date=$(date +"%y-%m-%d")

#
rho=05
epoch="final"
tree_search_func_name="ida_star"
t=60
seed=1
heuristic_method="advanced_priority_list"
for g in 30; 
	for num_scenarios in 50 100 200 500 ;
  do let secs=$t*48*3+500 && \
     time=$(date -d@$secs -u +%H:%M:%S) && \
     qsub -l h_rt=$time submit_anytime_tree_search.sh $HOME/Scratch/results/${date}_307/guided_${tree_search_func_name}/g${g}_s${num_scenarios} \
												 $HOME/AISO_HPC/best_policies/robustness/g${g}/params.json \
												 $HOME/ts4uc/data/envs/robustness/${g}gen.json \
												 $HOME/AISO_HPC/best_policies/robustness/g${g}/ac_final.pt \
												 ${t} \
												 0.${rho} \
												 ${tree_search_func_name} \
												 $HOME/ts4uc/data/hpc_params/input_day_ahead_g${g}.txt \
												 $heuristic_method \
												 $seed \ 
												 $num_scenarios
done ;
