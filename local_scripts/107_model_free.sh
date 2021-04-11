#!/bin/bash

# set today's date to use as save dir
date=$(date +"%y-%m-%d")

for g in 10 20 30 ;
do python $HOME/ts4uc/ts4uc/agents/solve_model_free.py -s $HOME/ts4uc/results/${date}_107/g${g}_mf -e $HOME/ts4uc/data/day_ahead/${g}gen/30min/env_params.json -t $HOME/ts4uc/data/day_ahead/${g}gen/30min -pf $HOME/thesis/results/chapter1/new_results/policies/exp101/g${g}/ac_final.pt -pp $HOME/thesis/results/chapter1/new_results/policies/exp101/g${g}/params.json --num_samples 1000 ;
   done 
