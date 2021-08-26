# set today's date to use as save directory
date=$(date +"%y-%m-%d")

rho=05
tree_search_func_name="ida_star"
t=60
g=30
r=50
seed=1
for c in 0 25 50;
	do for num_scenarios in 50 100 200 500 ;
	  do for heuristic_method in "advanced_priority_list";
	    do let secs=$t*48*3+500 && \
	       time=$(date -d@$secs -u +%H:%M:%S) && \
	       qsub -l h_rt=$time submit_anytime_tree_search.sh $HOME/Scratch/results/${date}_309/g${g}_c${c}_n${num_scenarios} \
														 $HOME/AISO_HPC/best_policies/curtailment/g${g}_c${c}_${r}/params.json \
														 $HOME/ts4uc/data/envs/curtailment/${g}gen_c${c}_${r}.json \
														 $HOME/AISO_HPC/best_policies/curtailment/g${g}_c${c}_${r}/ac_final.pt \
														 ${t} \
														 0.${rho} \
														 ${tree_search_func_name} \
														 $HOME/ts4uc/data/hpc_params/input_day_ahead_g${g}.txt \
														 ${heuristic_method} \
														 ${seed} \
														 ${num_scenarios} ;
	  done ;
	done ;
done ;
