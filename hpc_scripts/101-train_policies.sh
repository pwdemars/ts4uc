#!/bin/bash

# set today's date to use as save directory
date=$(date +"%y-%m-%d")

# 10 gen: 8 workers, 100,000 epochs
num_gen=10
workers=8
entropy_coef=0.01
epochs=200000
hrs=12
save_dir=${date}_101/g${num_gen}
qsub -pe smp $workers -l h_rt=${hrs}:00:00 submit_train.sh \
     $save_dir $num_gen $workers $epochs $entropy_coef

# 20 gen: 8 workers, 200,000 epochs
num_gen=20
workers=8
entropy_coef=0.001
epochs=200000
hrs=24
save_dir=${date}_101/g${num_gen}
qsub -pe smp $workers -l h_rt=${hrs}:00:00 submit_train.sh \
     $save_dir $num_gen $workers $epochs $entropy_coef

# 30 gen: 8 workers, 300,000 epochs
num_gen=30
workers=8
entropy_coef=0.001
epochs=300000
hrs=24
save_dir=${date}_101/g${num_gen}
qsub -pe smp $workers -l h_rt=${hrs}:00:00 submit_train.sh \
     $save_dir $num_gen $workers $epochs $entropy_coef

# 5, 6, 7, 8, 9 gens
workers=8
epochs=25000
hrs=4
save_dir=${date}_101/g${num_gen}
entropy_coef=0.05
for num_gen in 5 6 7 8 9;
do qsub -pe smp $workers -l h_rt=${hrs}:00:00 submit_train.sh \
	$save_dir $num_gen $workers $epochs $entropy_coef ;
done



