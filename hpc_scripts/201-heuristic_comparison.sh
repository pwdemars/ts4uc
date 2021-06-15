#/bin/bash

# set today's date to use as save directory
date=$(date +"%y-%m-%d")

# Different for 30 gens! 
for g in 10 20 30;
	do for tree_search_func_name in a_star;
		do for H in 4 6 8; 
			do for rho in 05;
				do for heuristic_method in "simple_priority_list" "simple_priority_list_ED" "advanced_priority_list";
					do qsub -l h_rt=24:00:00 submit_tree_search.sh $HOME/Scratch/results/${date}_201/guided_${tree_search_func_name}/g${g}/g${g}_h${H}_p${rho}_${heuristic_method} \
																 $HOME/AISO_HPC/best_policies/g${g}/params.json \
																 $HOME/ts4uc/data/day_ahead/${g}gen/30min/env_params.json \
																 $HOME/AISO_HPC/best_policies/g${g}/ac_final.pt \
																 ${H} \
																 0.${rho} \
																 ${tree_search_func_name} \
																 $HOME/ts4uc/data/hpc_params/input_day_ahead_g${g}.txt \
																 $heuristic_method ; 
				done ;
			done ;
		done ;
	done ;
done ;
