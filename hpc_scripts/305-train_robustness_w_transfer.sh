#!/bin/bash

# set today's date to use as save directory
date=$(date +"%y-%m-%d")

# 10 gen
num_gen=10
workers=8
epochs=100000
hrs=12
entropy_coef=0.05
clip_ratio=0.1
ac_lr=3e-05
cr_lr=3e-04
ac_weights_fn=$HOME/AISO_HPC/best_policies/g${num_gen}/ac_final.pt
ac_params_fn=$HOME/AISO_HPC/best_policies/g${num_gen}/params.json

qsub -pe smp $workers -l h_rt=${hrs}:00:00 ./submit_train_transfer.sh \
     ${date}_305/g${num_gen} $HOME/ts4uc/data/envs/robustness/${num_gen}gen.json $workers $epochs $entropy_coef $clip_ratio $ac_lr $cr_lr $ac_weights_fn $ac_params_fn 

# 20 gen
num_gen=20
workers=8
epochs=200000
hrs=18
entropy_coef=0.001
clip_ratio=0.1
ac_lr=3e-05
cr_lr=3e-04
ac_weights_fn=$HOME/AISO_HPC/best_policies/g${num_gen}/ac_final.pt
ac_params_fn=$HOME/AISO_HPC/best_policies/g${num_gen}/params.json

qsub -pe smp $workers -l h_rt=${hrs}:00:00 ./submit_train_transfer.sh \
     ${date}_305/g${num_gen} $HOME/ts4uc/data/envs/robustness/${num_gen}gen.json $workers $epochs $entropy_coef $clip_ratio $ac_lr $cr_lr $ac_weights_fn $ac_params_fn 

# 30 gen
num_gen=30
workers=8
epochs=300000
hrs=24
entropy_coef=0.0
clip_ratio=0.1
ac_lr=3e-05
cr_lr=3e-04
ac_weights_fn=$HOME/AISO_HPC/best_policies/g${num_gen}/ac_final.pt
ac_params_fn=$HOME/AISO_HPC/best_policies/g${num_gen}/params.json

qsub -pe smp $workers -l h_rt=${hrs}:00:00 ./submit_train_transfer.sh \
     ${date}_305/g${num_gen} $HOME/ts4uc/data/envs/robustness/${num_gen}gen.json $workers $epochs $entropy_coef $clip_ratio $ac_lr $cr_lr $ac_weights_fn $ac_params_fn
