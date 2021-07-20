#!/bin/bash

# set today's date to use as save directory
date=$(date +"%y-%m-%d")

for g in 10 20 30 ;
do python $HOME/pglib-uc/solve_and_test.py --save_dir ../results/${date}_302/milp_g${g}_n_minus_one \
						 --env_params_fn $HOME/ts4uc/data/envs/robustness/${g}gen.json \
						 --test_data_dir $HOME/ts4uc/data/day_ahead/${g}gen/30min \
						 --num_samples 1000 \
						 --reserve_sigma 4 \
						 --n_minus_one ;
done 					