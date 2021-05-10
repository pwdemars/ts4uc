#!/bin/bash

save_dir=$1
num_gen=$2
workers=$3
epochs=$4
entropy_coef=$5
clip_ratio=$6
ac_lr=$7
cr_lr=$8
num_layers=$9
num_nodes=${10}

python $HOME/ts4uc/ts4uc/agents/ppo_async/train.py \
       --save_dir $save_dir \
       --num_gen $num_gen \
       --epochs $epochs \
       --workers $workers \
       --entropy_coef $entropy_coef \
       --clip_ratio $clip_ratio \
       --ac_learning_rate $ac_lr \
       --cr_learning_rate $cr_lr \
       --num_layers $num_layers \
       --num_nodes $num_nodes \
       --buffer_size 2000 \
       --seed 10
