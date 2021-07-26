#!/bin/bash

# set today's date to use as save directory
date=$(date +"%y-%m-%d")

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
     ${date}_306/g${num_gen} $HOME/ts4uc/data/envs/robustness/${num_gen}gen.json $workers $epochs $entropy_coef $clip_ratio $ac_lr $cr_lr $ac_arch $cr_arch
