#/bin/bash

# set date that policies were trained. will be used for tests too 
root=$1
save_dir_root=$HOME/Scratch/results/${root}

echo "Reading policies from and writing results to: ${save_dir_root}"

# A* test
for g in 10 20 30;
	do for tree_search_func_name in a_star;
		do for H in 2 4 ; 
			do for rho in 05;
				do for heuristic_method in "advanced_priority_list" "none";
					do qsub -l h_rt=2:00:00 ../submit_tree_search.sh ${save_dir_root}/test/g${g}/h${H}_p${rho}_${heuristic_method} \
																 ${save_dir_root}/train/g${g}/params.json \
																 ${save_dir_root}/train/g${g}/env_params.json \
																 ${save_dir_root}/train/g${g}/ac_final.pt \
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
					do qsub -l h_rt=8:00:00 ../submit_tree_search.sh ${save_dir_root}/test/g${g}/h${H}_p${rho}_${heuristic_method} \
																 ${save_dir_root}/train/g${g}/params.json \
																 ${save_dir_root}/train/g${g}/env_params.json \
																 ${save_dir_root}/train/g${g}/ac_final.pt \
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

## Anytime test
rho=05
tree_search_func_name="ida_star"
for g in 10 20 30; 
  do for t in 2 5 10 30 60;
    do for heuristic_method in "advanced_priority_list";
      do let secs=$t*48*3+500 && \
         time=$(date -d@$secs -u +%H:%M:%S) && \
         qsub -l h_rt=$time ../submit_anytime_tree_search.sh ${save_dir_root}/test_anytime/g${g}/t${t}_p${rho}_${heuristic_method} \
														 ${save_dir_root}/train/g${g}/params.json \
														 ${save_dir_root}/train/g${g}/env_params.json \
														 ${save_dir_root}/train/g${g}/ac_final.pt \
														 ${t} \
														 0.${rho} \
														 ${tree_search_func_name} \
														 $HOME/ts4uc/data/hpc_params/input_day_ahead_g${g}.txt \
														 $heuristic_method ;
	  done ;
	done ;
  done ;