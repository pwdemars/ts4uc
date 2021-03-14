#/bin/bash

qsub -l h_rt=0:30:00 test_tree_search.sh $HOME/Scratch/results/mar12_g5_guided $HOME/AISO_HPC/feb4/g5/g5_d30_v1.json $HOME/AISO_HPC/feb4/5_arma_params.json $HOME/Scratch/results/feb4_g5_d30_v1/ac_final.pt 2 0.05 uniform_cost_search $HOME/AISO_HPC/AISO/input_g5_d30.txt

qsub -l h_rt=0:30:00 test_tree_search.sh $HOME/Scratch/results/mar12_g5_unguided $HOME/AISO_HPC/feb4/g5/g5_d30_v1.json $HOME/AISO_HPC/feb4/5_arma_params.json none 2 0.05 uniform_cost_search $HOME/AISO_HPC/AISO/input_g5_d30.txt
