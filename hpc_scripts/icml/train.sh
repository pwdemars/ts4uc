#!/bin/bash

# set today's date to use as save directory
date=$(date +"%y-%m-%d")

# 10 gen
num_gen=10
workers=8
epochs=200000
hrs=24
entropy_coef=0.05
clip_ratio=0.1
ac_lr=1e-05
cr_lr=1e-04
ac_arch="100,50,25"
cr_arch="64,64"
for c in 0 1;
do qsub -pe smp $workers -l h_rt=${hrs}:00:00 ../submit_train.sh \
     ${date}_icml_carbon${c}/train/g${num_gen} $HOME/ts4uc/data/envs/${num_gen}gen/carbon${c}.json $workers $epochs $entropy_coef $clip_ratio $ac_lr $cr_lr $ac_arch $cr_arch ; 
 done ;


# 20 gen
num_gen=20
workers=8
epochs=400000
hrs=36
entropy_coef=0.001
clip_ratio=0.1
ac_lr=1e-05
cr_lr=1e-04
ac_arch="64,64"
cr_arch="100,50,25"
for c in 0 1;
 do qsub -pe smp $workers -l h_rt=${hrs}:00:00 ../submit_train.sh \
     ${date}_icml_carbon${c}/train/g${num_gen} $HOME/ts4uc/data/envs/${num_gen}gen/carbon${c}.json $workers $epochs $entropy_coef $clip_ratio $ac_lr $cr_lr $ac_arch $cr_arch ; 
 done ; 

# 30 gen
num_gen=30
workers=8
epochs=600000
hrs=48
entropy_coef=0.0
clip_ratio=0.1
ac_lr=1e-05
cr_lr=1e-04
ac_arch="64,64"
cr_arch="400,300"
for c in 0 1; 
do qsub -pe smp $workers -l h_rt=${hrs}:00:00 ../submit_train.sh \
     ${date}_icml_carbon${c}/train/g${num_gen} $HOME/ts4uc/data/envs/${num_gen}gen/carbon${c}.json $workers $epochs $entropy_coef $clip_ratio $ac_lr $cr_lr $ac_arch $cr_arch ;
 done ;

