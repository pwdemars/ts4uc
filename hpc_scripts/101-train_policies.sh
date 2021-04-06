#!/bin/bash

# set today's date to use as save directory
date=$(date +"%y-%m-%d")

# 5 gen: 8 workers, 25,000 epochs
g=5
workers=8
epochs=25000
hrs=4
qsub -pe smp $workers -l h_rt=${hrs}:00:00 submit_train.sh ${date}_101_g${g} $HOME/AISO_HPC/best_policies/g${g}/params.json $HOME/AISO_HPC/mar14/${g}_env_params.json $workers $epochs

# 10 gen: 8 workers, 100,000 epochs
g=10
workers=8
epochs=100000
hrs=12
qsub -pe smp $workers -l h_rt=${hrs}:00:00 submit_train.sh ${date}_101_g${g} $HOME/AISO_HPC/best_policies/g${g}/params.json $HOME/AISO_HPC/mar14/${g}_env_params.json $workers $epochs

# 20 gen: 8 workers, 100,000 epochs
g=20
workers=8
epochs=100000
hrs=12
qsub -pe smp $workers -l h_rt=${hrs}:00:00 submit_train.sh ${date}_101_g${g} $HOME/AISO_HPC/best_policies/g${g}/params.json $HOME/AISO_HPC/mar14/${g}_env_params.json $workers $epochs

# 30 gen: 8 workers, 200,000 epochs
g=30
workers=8
epochs=200000
hrs=24
qsub -pe smp $workers -l h_rt=${hrs}:00:00 submit_train.sh ${date}_101_g${g} $HOME/AISO_HPC/best_policies/g${g}/params.json $HOME/AISO_HPC/mar14/${g}_env_params.json $workers $epochs
