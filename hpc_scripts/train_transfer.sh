#!/bin/bash

save_dir=$1
env_fn=$2
workers=$3
epochs=$4
entropy_coef=$5
clip_ratio=$6
ac_lr=$7
cr_lr=$8
ac_weights_fn=$9
ac_params_fn=${10}
buffer_size=${11:-2000}

python $HOME/ts4uc/ts4uc/agents/ppo_async/train.py \
       --save_dir $save_dir \
       --env_fn $env_fn \
       --epochs $epochs \
       --workers $workers \
       --entropy_coef $entropy_coef \
       --clip_ratio $clip_ratio \
       --ac_learning_rate $ac_lr \
       --cr_learning_rate $cr_lr \
       --buffer_size $buffer_size \
       --ac_weights_fn $ac_weights_fn \
       --ac_params_fn $ac_params_fn \
       --credit_assignment_1hr 0.9 \
       --forecast_horizon_hrs 12 \
       --seed 10