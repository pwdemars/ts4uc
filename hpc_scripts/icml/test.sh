#/bin/bash

# set date that policies were trained. will be used for tests too 
date=$1

# Different for 30 gens! 
for g in 10 20 30;
	do for tree_search_func_name in a_star;
		do for H in 2 4 ; 
			do for rho in 05;
				do for heuristic_method in "advanced_priority_list" "none";
					do qsub -l h_rt=12:00:00 submit_tree_search.sh $HOME/Scratch/results/${date}_icml/icml/g${g}/h${H}_p${rho}_${heuristic_method} \
																 $HOME/Scratch/results/${date}_icml/train/g${g}/params.json \
																 $HOME/ts4uc/data/day_ahead/${g}gen/30min/env_params.json \
																 $HOME/Scratch/results/${date}_icml/train/g${g}/ac_final.pt \
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

for g in 10 20 30;
	do for tree_search_func_name in a_star;
		do for H in 6 8; 
			do for rho in 05;
				do for heuristic_method in "advanced_priority_list";
					do qsub -l h_rt=16:00:00 submit_tree_search.sh $HOME/Scratch/results/${date}_icml/test/g${g}/h${H}_p${rho}_${heuristic_method} \
																 $HOME/Scratch/results/${date}_icml/train/g${g}/params.json \
																 $HOME/ts4uc/data/day_ahead/${g}gen/30min/env_params.json \
																 $HOME/Scratch/results/${date}_icml/train/g${g}/ac_final.pt \
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
