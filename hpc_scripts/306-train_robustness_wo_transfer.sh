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
ac_arch="100,50,25"
cr_arch="64,64"
qsub -pe smp $workers -l h_rt=${hrs}:00:00 ./submit_train.sh \
     ${date}_305/g${num_gen} $HOME/ts4uc/data/envs/robustness/${num_gen}gen.json $workers $epochs $entropy_coef $clip_ratio $ac_lr $cr_lr $ac_arch $cr_arch

# 20 gen
num_gen=20
workers=8
epochs=200000
hrs=18
entropy_coef=0.001
clip_ratio=0.1
ac_lr=3e-05
cr_lr=3e-04
ac_arch="64,64"
cr_arch="100,50,25"

qsub -pe smp $workers -l h_rt=${hrs}:00:00 ./submit_train.sh \
     ${date}_305/g${num_gen} $HOME/ts4uc/data/envs/robustness/${num_gen}gen.json $workers $epochs $entropy_coef $clip_ratio $ac_lr $cr_lr $ac_arch $cr_arch

# 30 gen
num_gen=30
workers=8
epochs=300000
hrs=24
entropy_coef=0.0
clip_ratio=0.1
ac_lr=3e-05
cr_lr=3e-04
ac_arch="64,64"
cr_arch="400,300"

qsub -pe smp $workers -l h_rt=${hrs}:00:00 ./submit_train.sh \
     ${date}_305/g${num_gen} $HOME/ts4uc/data/envs/robustness/${num_gen}gen.json $workers $epochs $entropy_coef $clip_ratio $ac_lr $cr_lr $ac_arch $cr_arch
