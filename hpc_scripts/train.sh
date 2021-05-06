#!/bin/bash

save_dir=$1
num_gen=$2
workers=$3
epochs=$4
entropy_coef=$5

python $HOME/ts4uc/ts4uc/agents/ppo_async/train.py \
       --save_dir $save_dir \
       --num_gen $num_gen \
       --epochs $epochs \
       --workers $workers \
       --entropy_coef $entropy_coef \
       --ac_learning_rate 3e-05 \
       --cr_learning_rate 3e-04 \
       --num_layers 3 \
       --num_nodes 64 \
       --clip_ratio 0.1 \
       --buffer_size 2000 \
       --seed 10
