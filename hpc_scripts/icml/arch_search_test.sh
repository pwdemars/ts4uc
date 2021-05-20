#/bin/bash

# set date that policies were trained. will be used for tests too 
date=$1
save_dir_root=$HOME/Scratch/results/${date}_icml

echo "Reading policies from and writing results to: ${save_dir_root}"

for g in 10 20 30;
	for v in {1..9}:
		do for tree_search_func_name in a_star;
			do for H in 4 6; 
				do for rho in 05;
					do for heuristic_method in "advanced_priority_list";
						do qsub -l h_rt=12:00:00 ../submit_tree_search.sh ${save_dir_root}/test/g${g}_v${v}/h${H}_p${rho}_${heuristic_method} \
																	 ${save_dir_root}/train/g${g}_v${v}/params.json \
																	 $HOME/ts4uc/data/day_ahead/${g}gen/30min/env_params.json \
																	 ${save_dir_root}/train/g${g}_v${v}/ac_final.pt \
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
done ;
