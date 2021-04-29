#!/bin/bash

# set today's date to use as save directory
date=$(date +"%y-%m-%d")

# 5 gen: 8 workers, 25,000 epochs
g=5
workers=8
epochs=25000
hrs=4
qsub -pe smp $workers -l h_rt=${hrs}:00:00 submit_train.sh ${date}_101/g${g} $HOME/ts4uc/data/policy_params/exp101/g${g}/params.json $HOME/ts4uc/data/day_ahead/${g}gen/30min/env_params.json $workers $epochs

# 10 gen: 8 workers, 100,000 epochs
g=10
workers=8
epochs=100000
hrs=12
qsub -pe smp $workers -l h_rt=${hrs}:00:00 submit_train.sh ${date}_101/g${g} $HOME/ts4uc/data/policy_params/exp101/g${g}/params.json $HOME/ts4uc/data/day_ahead/${g}gen/30min/env_params.json $workers $epochs

# 20 gen: 8 workers, 100,000 epochs
g=20
workers=8
epochs=200000
hrs=18
qsub -pe smp $workers -l h_rt=${hrs}:00:00 submit_train.sh ${date}_101/g${g} $HOME/ts4uc/data/policy_params/exp101/g${g}/params.json $HOME/ts4uc/data/day_ahead/${g}gen/30min/env_params.json $workers $epochs

# 30 gen: 8 workers, 200,000 epochs
g=30
workers=8
epochs=300000
hrs=24
qsub -pe smp $workers -l h_rt=${hrs}:00:00 submit_train.sh ${date}_101/g${g} $HOME/ts4uc/data/policy_params/exp101/g${g}/params.json $HOME/ts4uc/data/day_ahead/${g}gen/30min/env_params.json $workers $epochs

# 6, 7, 8, 9 gens
workers=8
epochs=25000
hrs=4
for g in 6 7 8 9;
do qsub -pe smp $workers -l h_rt=${hrs}:00:00 submit_train.sh ${date}_101/g${g} $HOME/ts4uc/data/policy_params/exp101/g${g}/params.json $HOME/ts4uc/data/day_ahead/${g}gen/30min/env_params.json $workers $epochs ;
done
