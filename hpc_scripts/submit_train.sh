#!/bin/bash -l
#$ -wd /home/uclqpde/Scratch/output
#$ -M patrick.demars.14@ucl.ac.uk
#$ -m bes
#$ -l mem=4G

save_dir=$1
num_gen=$2
workers=$3
epochs=$4
entropy_coef=$5
clip_ratio=$6
ac_lr=$7
cr_lr=$8
num_layers=$9
num_nodes=${10}

module load python3/3.7
export OMP_NUM_THREADS=1

mkdir -p $HOME/Scratch/results/$save_dir

cd $TMPDIR

$HOME/ts4uc/hpc_scripts/train.sh \
    $save_dir $num_gen $workers $epochs $entropy_coef $clip_ratio $ac_lr $cr_lr $num_layers $num_nodes

tar zcvf $HOME/Scratch/results/$save_dir/results.tar.gz $TMPDIR
