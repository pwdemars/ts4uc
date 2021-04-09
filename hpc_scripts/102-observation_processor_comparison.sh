#/bin/bash

date=$(date +"%y-%m-%d")

qsub -pe smp 8 -l h_rt=6:00:00 ./submit_train.sh ${date}_102/v1 $HOME/ts4uc/data/policy_params/exp102/params_DA.json $HOME/ts4uc/data/day_ahead/5gen/30min/env_params.json 8 25000
qsub -pe smp 8 -l h_rt=6:00:00 ./submit_train.sh ${date}_102/v2 $HOME/ts4uc/data/policy_params/exp102/params_LH.json $HOME/ts4uc/data/day_ahead/5gen/30min/env_params.json 8 25000
