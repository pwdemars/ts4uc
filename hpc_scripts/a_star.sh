#/bin/bash

# set today's date to use as save directory
date=$(date +"%y-%m-%d")

# guided a* search, g={5,10,20,30}, H=4, rho=0.05
# use policies *_v1 for g10 and g20, use *_v3 for g30. 
qsub -l h_rt=2:00:00 test_tree_search.sh $HOME/Scratch/results/${date}/guided_astar/g5/feb4_g5_d30_v1_h4_p05 $HOME/AISO_HPC/feb4/g5/g5_d30_v1.json $HOME/AISO_HPC/mar14/5_env_params.json $HOME/Scratch/results/feb4_g5_d30_v1/ac_final.pt 4 0.05 a_star $HOME/AISO_HPC/AISO/input_g5_d30.txt
qsub -l h_rt=2:00:00 test_tree_search.sh $HOME/Scratch/results/${date}/guided_astar/g10/feb4_g10_d30_v1_h4_p05 $HOME/AISO_HPC/feb4/g10/g10_d30_v1.json $HOME/AISO_HPC/mar14/10_env_params.json $HOME/Scratch/results/feb4_g10_d30_v1/ac_final.pt 4 0.05 a_star $HOME/AISO_HPC/AISO/input_g10_d30.txt
qsub -l h_rt=2:00:00 test_tree_search.sh $HOME/Scratch/results/${date}/guided_astar/g20/feb4_g20_d30_v1_h4_p05 $HOME/AISO_HPC/feb4/g20/g20_d30_v1.json $HOME/AISO_HPC/mar14/20_env_params.json $HOME/Scratch/results/feb4_g20_d30_v1/ac_final.pt 4 0.05 a_star $HOME/AISO_HPC/AISO/input_g20_d30.txt
qsub -l h_rt=2:00:00 test_tree_search.sh $HOME/Scratch/results/${date}/guided_astar/g30/feb4_g30_d30_v3_h4_p05 $HOME/AISO_HPC/feb4/g30/g30_d30_v3.json $HOME/AISO_HPC/mar14/30_env_params.json $HOME/Scratch/results/feb4_g30_d30_v3/ac_final.pt 4 0.05 a_star $HOME/AISO_HPC/AISO/input_g30_d30.txt

# unguided a* search, g={5,10,20,30}, H=2
qsub -l h_rt=2:00:00 test_tree_search.sh $HOME/Scratch/results/${date}/unguided_astar/g5/feb4_g5_d30_v1_h4_p05 none $HOME/AISO_HPC/mar14/5_env_params.json none 2 -1 a_star $HOME/AISO_HPC/AISO/input_g5_d30.txt
qsub -l h_rt=2:00:00 test_tree_search.sh $HOME/Scratch/results/${date}/unguided_astar/g10/feb4_g10_d30_v1_h4_p05 none $HOME/AISO_HPC/mar14/10_env_params.json none 2 -1 a_star $HOME/AISO_HPC/AISO/input_g10_d30.txt
qsub -l h_rt=2:00:00 test_tree_search.sh $HOME/Scratch/results/${date}/unguided_astar/g20/feb4_g20_d30_v1_h4_p05 none $HOME/AISO_HPC/mar14/20_env_params.json none 2 -1 a_star $HOME/AISO_HPC/AISO/input_g20_d30.txt
qsub -l h_rt=2:00:00 test_tree_search.sh $HOME/Scratch/results/${date}/unguided_astar/g30/feb4_g30_d30_v3_h4_p05 none $HOME/AISO_HPC/mar14/30_env_params.json none 2 -1 a_star $HOME/AISO_HPC/AISO/input_g30_d30.txt

# guided real-time a* search, g={5,10,20,30}, H=4, rho=0.05
# use policies *_v1 for g5, g10 and g20, use *_v3 for g30. 
qsub -l h_rt=2:00:00 test_tree_search.sh $HOME/Scratch/results/${date}/guided_rtastar/g5/feb4_g5_d30_v1_h4_p05 $HOME/AISO_HPC/feb4/g5/g5_d30_v1.json $HOME/AISO_HPC/mar14/5_env_params.json $HOME/Scratch/results/feb4_g5_d30_v1/ac_final.pt 4 0.05 rta_star $HOME/AISO_HPC/AISO/input_g5_d30.txt
qsub -l h_rt=2:00:00 test_tree_search.sh $HOME/Scratch/results/${date}/guided_rtastar/g10/feb4_g10_d30_v1_h4_p05 $HOME/AISO_HPC/feb4/g10/g10_d30_v1.json $HOME/AISO_HPC/mar14/10_env_params.json $HOME/Scratch/results/feb4_g10_d30_v1/ac_final.pt 4 0.05 rta_star $HOME/AISO_HPC/AISO/input_g10_d30.txt
qsub -l h_rt=2:00:00 test_tree_search.sh $HOME/Scratch/results/${date}/guided_rtastar/g20/feb4_g20_d30_v1_h4_p05 $HOME/AISO_HPC/feb4/g20/g20_d30_v1.json $HOME/AISO_HPC/mar14/20_env_params.json $HOME/Scratch/results/feb4_g20_d30_v1/ac_final.pt 4 0.05 rta_star $HOME/AISO_HPC/AISO/input_g20_d30.txt
qsub -l h_rt=2:00:00 test_tree_search.sh $HOME/Scratch/results/${date}/guided_rtastar/g30/feb4_g30_d30_v3_h4_p05 $HOME/AISO_HPC/feb4/g30/g30_d30_v3.json $HOME/AISO_HPC/mar14/30_env_params.json $HOME/Scratch/results/feb4_g30_d30_v3/ac_final.pt 4 0.05 rta_star $HOME/AISO_HPC/AISO/input_g30_d30.txt

# unguided real-time a* search, g={5,10,20,30}, H=2
qsub -l h_rt=2:00:00 test_tree_search.sh $HOME/Scratch/results/${date}/unguided_rtastar/g5/feb4_g5_d30_v1_h4_p05 none $HOME/AISO_HPC/mar14/5_env_params.json none 2 -1 rta_star $HOME/AISO_HPC/AISO/input_g5_d30.txt
qsub -l h_rt=2:00:00 test_tree_search.sh $HOME/Scratch/results/${date}/unguided_rtastar/g10/feb4_g10_d30_v1_h4_p05 none $HOME/AISO_HPC/mar14/10_env_params.json none 2 -1 rta_star $HOME/AISO_HPC/AISO/input_g10_d30.txt
qsub -l h_rt=2:00:00 test_tree_search.sh $HOME/Scratch/results/${date}/unguided_rtastar/g20/feb4_g20_d30_v1_h4_p05 none $HOME/AISO_HPC/mar14/20_env_params.json none 2 -1 rta_star $HOME/AISO_HPC/AISO/input_g20_d30.txt
qsub -l h_rt=2:00:00 test_tree_search.sh $HOME/Scratch/results/${date}/unguided_rtastar/g30/feb4_g30_d30_v3_h4_p05 none $HOME/AISO_HPC/mar14/30_env_params.json none 2 -1 rta_star $HOME/AISO_HPC/AISO/input_g30_d30.txt
