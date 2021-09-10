#!/bin/bash

# set today's date to use as save directory
date=$(date +"%y-%m-%d")

num_gen=30
workers=8
epochs=500000
hrs=16
entropy_coef=0.0
clip_ratio=0.1
ac_lr=3e-05
cr_lr=3e-04
ac_arch="64,64"
cr_arch="400,300"
for c in 0 25 50 ; 
  do for r in 50 ; 
     do qsub -pe smp $workers -l h_rt=${hrs}:00:00 ./submit_train.sh \
     ${date}_308/g${num_gen}_c${c}_${r} $HOME/ts4uc/data/envs/curtailment/${num_gen}gen_c${c}_${r}.json $workers $epochs $entropy_coef $clip_ratio $ac_lr $cr_lr $ac_arch $cr_arch ;
  done ; 
done 
