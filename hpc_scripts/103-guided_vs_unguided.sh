#/bin/bash

# set today's date to use as save directory
date=$(date +"%y-%m-%d")

# guided search, g={5..10}, H=2, rho=0.05
for g in {5..10};
do qsub -l h_rt=0:30:00 submit_tree_search.sh $HOME/Scratch/results/${date}_103/guided/g${g}/g${g}_h2_p05 $HOME/AISO_HPC/best_policies/g${g}/params.json $HOME/ts4uc/data/day_ahead/${g}gen/30min/env_params.json $HOME/AISO_HPC/best_policies/g${g}/ac_final.pt 2 0.05 uniform_cost_search $HOME/AISO_HPC/AISO/input_g${g}_d30.txt;
done

# unguided search, g={5..10}, H=2
for g in {5..10};
do qsub -l h_rt=24:00:00 submit_tree_search.sh $HOME/Scratch/results/${date}_103/unguided/g${g}_h2 none $HOME/ts4uc/data/day_ahead/${g}gen/30min/env_params.json none 2 -1 uniform_cost_search $HOME/AISO_HPC/AISO/input_g${g}_d30.txt;
done
