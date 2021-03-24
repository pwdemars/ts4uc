#/bin/bash

# set today's date to use as save directory
date=$(date +"%y-%m-%d")

# guided search, g=5; H={1, 2, 4, 6, 8, 12, 16, 20, 24}; rho={0.01, 0.02, 0.05, 0.1, 0.25, 0.33}
for H in 1 2 4;
do for rho in 01 05;
do qsub -l h_rt=6:00:00 submit_tree_search.sh $HOME/Scratch/results/${date}/guided/g5/feb4_g5_d30_v1_h${H}_p${rho} \
	$HOME/AISO_HPC/feb4/g5/g5_d30_v1.json \
	$HOME/AISO_HPC/mar14/5_env_params.json \
	$HOME/Scratch/results/feb4_g5_d30_v1/ac_final.pt \
	${H} \
	0.${rho} \
	uniform_cost_search \
	$HOME/AISO_HPC/AISO/input_g5_d30.txt ;
done
done

for H in 1 2 4;
do for rho in 1 25 33;
do qsub -l h_rt=1:00:00 submit_tree_search.sh $HOME/Scratch/results/${date}/guided/g5/feb4_g5_d30_v1_h${H}_p${rho} \
	$HOME/AISO_HPC/feb4/g5/g5_d30_v1.json \
	$HOME/AISO_HPC/mar14/5_env_params.json \
	$HOME/Scratch/results/feb4_g5_d30_v1/ac_final.pt \
	${H} \
	0.${rho} \
	uniform_cost_search \
	$HOME/AISO_HPC/AISO/input_g5_d30.txt ;
done
done

for H in 6 8;
do for rho in 01 05 1 25 33;
do qsub -l h_rt=24:00:00 submit_tree_search.sh $HOME/Scratch/results/${date}/guided/g5/feb4_g5_d30_v1_h${H}_p${rho} \
	$HOME/AISO_HPC/feb4/g5/g5_d30_v1.json \
	$HOME/AISO_HPC/mar14/5_env_params.json \
	$HOME/Scratch/results/feb4_g5_d30_v1/ac_final.pt \
	${H} \
	0.${rho} \
	uniform_cost_search \
	$HOME/AISO_HPC/AISO/input_g5_d30.txt ;
done
done

for H in 12 16 20 24;
do for rho in 05 1 25 33;
do qsub -l h_rt=24:00:00 submit_tree_search.sh $HOME/Scratch/results/${date}/guided/g5/feb4_g5_d30_v1_h${H}_p${rho} \
	$HOME/AISO_HPC/feb4/g5/g5_d30_v1.json \
	$HOME/AISO_HPC/mar14/5_env_params.json \
	$HOME/Scratch/results/feb4_g5_d30_v1/ac_final.pt \
	${H} \
	0.${rho} \
	uniform_cost_search \
	$HOME/AISO_HPC/AISO/input_g5_d30.txt ;
done
done