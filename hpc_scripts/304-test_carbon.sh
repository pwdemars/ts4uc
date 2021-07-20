# set today's date to use as save directory
date=$(date +"%y-%m-%d")

#
rho=05
tree_search_func_name="ida_star"
t=60
for g in 10 20 30; 
  do for c in 25 50;
    do for heuristic_method in "advanced_priority_list";
      do let secs=$t*48*3+500 && \
         time=$(date -d@$secs -u +%H:%M:%S) && \
         qsub -l h_rt=$time submit_anytime_tree_search.sh $HOME/Scratch/results/${date}_304/guided_${tree_search_func_name}/c${c}/g${g} \
														 $HOME/AISO_HPC/best_policies/carbon${c}/g${g}/params.json \
														 $HOME/ts4uc/data/envs/carbon/${g}gen_carbon${c}.json \
														 $HOME/AISO_HPC/best_policies/carbon${c}/g${g}/ac_final.pt \
														 ${t} \
														 0.${rho} \
														 ${tree_search_func_name} \
														 $HOME/ts4uc/data/hpc_params/input_day_ahead_g${g}.txt \
														 $heuristic_method ;
	  done ;
	done ;
done ;
