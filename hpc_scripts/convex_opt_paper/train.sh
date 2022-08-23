#!/bin/bash

# set today's date to use as save directory
date=$(date +"%y-%m-%d")

workers=8
epochs=300000
hrs=48
entropy_coef=0.05
clip_ratio=0.1
ac_lr=3e-05
cr_lr=3e-04
ac_arch="64,64"
cr_arch="400,300"
buffer_size=5000

for num_gen in {40,50} ; 
  do qsub -pe smp $workers -l h_rt=${hrs}:00:00 ../submit_train.sh \
     ${date}_convex_opt_comp/g${num_gen}_$RANDOM $HOME/AISO_HPC/AISO/rl4uc/rl4uc/data/envs/${num_gen}gen.json $workers $epochs $entropy_coef $clip_ratio $ac_lr $cr_lr $ac_arch $cr_arch $buffer_size ;
done
