#!/bin/bash

save_dir=$1
env_fn=$2
workers=$3
epochs=$4
entropy_coef=$5
clip_ratio=$6
ac_lr=$7
cr_lr=$8
ac_arch=$9
cr_arch=${10}

python $HOME/ts4uc/ts4uc/agents/ppo_async/train.py \
       --save_dir $save_dir \
       --env_fn $env_fn \
       --epochs $epochs \
       --workers $workers \
       --entropy_coef $entropy_coef \
       --clip_ratio $clip_ratio \
       --ac_learning_rate $ac_lr \
       --cr_learning_rate $cr_lr \
       --ac_arch $ac_arch \
       --cr_arch $cr_arch \
       --buffer_size 2000 \
       --credit_assignment_1hr 0.9 \
       --forecast_horizon_hrs 12 \
       --seed 10
