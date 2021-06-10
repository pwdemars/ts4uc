#!/bin/bash

# set today's date to use as save directory
date=$(date +"%y-%m-%d")

# 100 gen
num_gen=100
workers=8
epochs=500000
hrs=48
entropy_coef=0.05
clip_ratio=0.1
ac_lr=1e-05
cr_lr=1e-04
ac_arch="128,64"
cr_arch="128,64"
buffer_size=10000
qsub -pe smp $workers -l h_rt=${hrs}:00:00 submit_train.sh \
     ${date}_101/g${num_gen} $HOME/ts4uc/data/day_ahead/${num_gen}gen/30min/env_params.json $workers $epochs $entropy_coef $clip_ratio $ac_lr $cr_lr $ac_arch $cr_arch $buffer_size

