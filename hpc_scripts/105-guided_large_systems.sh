#/bin/bash

# set today's date to use as save directory
date=$(date +"%y-%m-%d")

# guided search, g={10,20,30}, H=4, rho=0.05
# use policies *_v1 for g10 and g20, use *_v3 for g30. 
qsub -l h_rt=24:00:00 submit_tree_search.sh $HOME/Scratch/results/${date}/guided/g10/feb4_g10_d30_v1_h4_p05 $HOME/AISO_HPC/feb4/g10/g10_d30_v1.json $HOME/AISO_HPC/mar14/10_env_params.json $HOME/Scratch/results/feb4_g10_d30_v1/ac_final.pt 4 0.05 uniform_cost_search $HOME/AISO_HPC/AISO/input_g10_d30.txt
qsub -l h_rt=24:00:00 submit_tree_search.sh $HOME/Scratch/results/${date}/guided/g20/feb4_g20_d30_v1_h4_p05 $HOME/AISO_HPC/feb4/g20/g20_d30_v1.json $HOME/AISO_HPC/mar14/20_env_params.json $HOME/Scratch/results/feb4_g20_d30_v1/ac_final.pt 4 0.05 uniform_cost_search $HOME/AISO_HPC/AISO/input_g20_d30.txt
qsub -l h_rt=24:00:00 submit_tree_search.sh $HOME/Scratch/results/${date}/guided/g30/feb4_g30_d30_v3_h4_p05 $HOME/AISO_HPC/feb4/g30/g30_d30_v3.json $HOME/AISO_HPC/mar14/30_env_params.json $HOME/Scratch/results/feb4_g30_d30_v3/ac_final.pt 4 0.05 uniform_cost_search $HOME/AISO_HPC/AISO/input_g30_d30.txt