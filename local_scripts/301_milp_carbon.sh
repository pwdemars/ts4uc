#!/bin/bash

# set today's date to use as save directory
date=$(date +"%y-%m-%d")

for g in 10 20 30 ;
do for c in 25 50 ; 
do python $HOME/pglib-uc/solve_and_test.py --save_dir ../results/${date}_301/milp_g${g}_carbon${c}_4sigma \
						 --env_params_fn $HOME/ts4uc/data/envs/carbon/${g}gen_carbon${c}.json \
						 --test_data_dir $HOME/ts4uc/data/day_ahead/${g}gen/30min \
						 --num_samples 1000 \
						 --reserve_sigma 4 ;
done ; 
done