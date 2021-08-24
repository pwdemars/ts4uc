#!/bin/bash

# set today's date to use as save directory
date=$(date +"%y-%m-%d")

g=30
python $HOME/pglib-uc/solve_and_test.py --save_dir ../results/${date}_310/milp_g${g}_custom_reserve \
						 --env_params_fn $HOME/ts4uc/data/envs/robustness/${g}gen.json \
						 --test_data_dir $HOME/ts4uc/data/day_ahead/${g}gen/30min \
						 --num_samples 1000 \
						 --custom_reserve_fn $HOME/thesis/output/chapter3/data/exp307_ida_star_reserves.csv ;