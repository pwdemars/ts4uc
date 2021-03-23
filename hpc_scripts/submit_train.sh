#!/bin/bash -l
#$ -wd /home/uclqpde/Scratch/output
#$ -M patrick.demars.14@ucl.ac.uk
#$ -m bes
#$ -l mem=4G

save_dir=$1
params_filename=$2
env_params_filename=$3
num_procs=$4
num_epochs=$5

module load python3/3.7
export OMP_NUM_THREADS=1

mkdir $HOME/Scratch/results/$save_dir

cd $TMPDIR

python $HOME/ts4uc/ts4uc/agents/train_ac.py --save_dir $save_dir --params_fn $params_filename --env_params_fn $env_params_filename --num_procs $num_procs --num_epochs $num_epochs

tar zcvf $HOME/Scratch/results/$save_dir/results.tar.gz $TMPDIR
