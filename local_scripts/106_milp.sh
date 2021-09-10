#!/bin/bash

# set today's date to use as save directory
date=$(date +"%y-%m-%d")

for g in 10 20 30 ;
do python $HOME/pglib-uc/solve_and_test.py --save_dir ../results/${date}_106/milp_g${g}_perfect \
						 --env_params_fn $HOME/ts4uc/data/day_ahead/${g}gen/30min/env_params.json \
						 --test_data_dir $HOME/ts4uc/data/day_ahead/${g}gen/30min \
						 --num_samples 1000 \
						 --perfect_forecast true ;
done 					

for g in 10 20 30 ;
	do for s in 3 4 5 ; 
		do python $HOME/pglib-uc/solve_and_test.py --save_dir ../results/${date}_106/milp_g${g}_${s}sigma \
						 --env_params_fn $HOME/ts4uc/data/day_ahead/${g}gen/30min/env_params.json \
						 --test_data_dir $HOME/ts4uc/data/day_ahead/${g}gen/30min \
						 --num_samples 1000 \
						 --reserve_sigma $s ;
	done ;
done 					
