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
num_layers=3
num_nodes=32
qsub -pe smp $workers -l h_rt=${hrs}:00:00 submit_train.sh \
     ${date}_101/g${num_gen} $num_gen $workers $epochs $entropy_coef $clip_ratio $ac_lr $cr_lr $num_layers $num_nodes

# 20 gen
num_gen=20
workers=8
epochs=200000
hrs=24
entropy_coef=0.001
clip_ratio=0.1
ac_lr=3e-05
cr_lr=3e-04
num_layers=3
num_nodes=32
qsub -pe smp $workers -l h_rt=${hrs}:00:00 submit_train.sh \
     ${date}_101/g${num_gen} $num_gen $workers $epochs $entropy_coef $clip_ratio $ac_lr $cr_lr $num_layers $num_nodes

# 30 gen
num_gen=30
workers=8
epochs=300000
hrs=24
entropy_coef=0.0
clip_ratio=0.1
ac_lr=3e-05
cr_lr=3e-04
num_layers=3
num_nodes=32
qsub -pe smp $workers -l h_rt=${hrs}:00:00 submit_train.sh \
     ${date}_101/g${num_gen} $num_gen $workers $epochs $entropy_coef $clip_ratio $ac_lr $cr_lr $num_layers $num_nodes

# 5, 6, 7, 8, 9 gens
workers=8
epochs=25000
hrs=3
entropy_coef=0.05
clip_ratio=0.1
ac_lr=3e-05
cr_lr=3e-04
num_layers=3
num_nodes=32
for num_gen in 5 6 7 8 9;
do qsub -pe smp $workers -l h_rt=${hrs}:00:00 submit_train.sh \
	${date}_101/g${num_gen} $num_gen $workers $epochs $entropy_coef $clip_ratio $ac_lr $cr_lr $num_layers $num_nodes ;
done



