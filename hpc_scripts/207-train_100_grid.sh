#!/bin/bash

# set today's date to use as save directory
date=$(date +"%y-%m-%d")

# 100 gen
num_gen=100
workers=8
hrs=48

clip_ratio=0.1
ac_lr=1e-05
cr_lr=1e-04
buffer_size=5000
epochs=300000
ac_arch="128,64"
cr_arch="128,64"

i=0
entropy_coef=0.
entropy_target=0.
qsub -pe smp $workers -l h_rt=${hrs}:00:00 submit_train.sh \
     ${date}_207/g${num_gen}_v${i} $HOME/ts4uc/data/day_ahead/${num_gen}gen/30min/env_params.json $workers $epochs $entropy_coef $clip_ratio $ac_lr $cr_lr $ac_arch $cr_arch $buffer_size $entropy_target


i=1
for entropy_coef in 0.1 1.0; 
     do for entropy_target in 0.1567 0.1919 0.2648 ; 
          do qsub -pe smp $workers -l h_rt=${hrs}:00:00 submit_train.sh \
          ${date}_207/g${num_gen}_v${i} $HOME/ts4uc/data/day_ahead/${num_gen}gen/30min/env_params.json $workers $epochs $entropy_coef $clip_ratio $ac_lr $cr_lr $ac_arch $cr_arch $buffer_size $entropy_target \
          && i=$((i+1)) ;
     done ;
done ;