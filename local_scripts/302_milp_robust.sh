#!/bin/bash

# set today's date to use as save directory
date=$(date +"%y-%m-%d")

# iterate through 1..3 for n-1, n-2, n-3 and 10, 20, 30 gen problems
for g in 1 2 3 ;
do python $HOME/pglib-uc/solve_and_test.py --save_dir ../results/${date}_302/milp_g${g}0_n_minus_x \
						 --env_params_fn $HOME/ts4uc/data/envs/robustness/${g}0gen.json \
						 --test_data_dir $HOME/ts4uc/data/day_ahead/${g}0gen/30min \
						 --num_samples 1000 \
						 --reserve_sigma 4 \
						 --n_minus_x ${g} ;
done 					