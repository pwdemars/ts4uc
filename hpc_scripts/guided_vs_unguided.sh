#/bin/bash

# set today's date to use as save directory
date=$(date +"%m-%d-%y")

# guided search, g={5..10}, H=2, rho=0.05
for g in {5..10};
do qsub -l h_rt=1:00:00 test_tree_search.sh $HOME/Scratch/results/${date}/guided/g${g}/feb4_g${g}_d30_v1_h2_p05 $HOME/AISO_HPC/feb4/g${g}/g${g}_d30_v1.json $HOME/AISO_HPC/mar14/${g}_env_params.json $HOME/Scratch/results/feb4_g${g}_d30_v1/ac_final.pt 2 0.05 uniform_cost_search $HOME/AISO_HPC/AISO/input_g${g}_d30.txt;
done

# unguided search, g={5..10}, H=2
for g in {5..10};
do qsub -l h_rt=1:00:00 test_tree_search.sh $HOME/Scratch/results/${date}/unguided/g${g}/h2 none $HOME/AISO_HPC/mar14/${g}_env_params.json none 2 -1 uniform_cost_search $HOME/AISO_HPC/AISO/input_g${g}_d30.txt;
done

