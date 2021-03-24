#/bin/bash

# set today's date to use as save directory
date=$(date +"%y-%m-%d")

# Different for 30 gens! 
for g in 5;
	do for tree_search_func_name in a_star rta_star;
		do for H in 8 12 16 20 24; 
			do for rho in 05;
				do for heuristic_method in "priority_list" "pl_plus_ll";
					do qsub -l h_rt=00:30:00 submit_tree_search.sh $HOME/Scratch/results/${date}/guided_${tree_search_func_name}/g${g}/feb4_g${g}_d30_v1_h${H}_p${rho}_${heuristic_method} \
																 $HOME/AISO_HPC/best_policies/g${g}/params.json \
																 $HOME/AISO_HPC/mar14/${g}_env_params.json \
																 $HOME/AISO_HPC/best_policies/g${g}/ac_final.pt \
																 ${H} \
																 0.${rho} \
																 ${tree_search_func_name} \
																 $HOME/AISO_HPC/AISO/input_g${g}_d30.txt \
																 $heuristic_method ; 
				done ;
			done ;
		done ;
	done ;
done ;
