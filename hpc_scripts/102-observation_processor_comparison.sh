#/bin/bash

date=$(date +"%y-%m-%d")

qsub -pe smp 8 -l h_rt=6:00:00 ./submit_train.sh ${date}_102_v1 $HOME/AISO_HPC/mar23/mar23_g5_v1.json $HOME/ts4uc/data/day_ahead/5gen/30min/env_params.json 8 25000
qsub -pe smp 8 -l h_rt=6:00:00 ./submit_train.sh ${date}_102_v2 $HOME/AISO_HPC/mar23/mar23_g5_v2.json $HOME/ts4uc/data/day_ahead/5gen/30min/env_params.json 8 25000
