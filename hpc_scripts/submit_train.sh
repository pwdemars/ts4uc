#!/bin/bash -l
#$ -wd /home/uclqpde/Scratch/output
#$ -M patrick.demars.14@ucl.ac.uk
#$ -m bes
#$ -l mem=4G

save_dir=$1
env_fn=$2
workers=$3
epochs=$4
entropy_coef=$5
clip_ratio=$6
ac_lr=$7
cr_lr=$8
ac_arch=$9
cr_arch=${10}
buffer_size=${11:-2000}

module load python3/3.7
export OMP_NUM_THREADS=1

mkdir -p $HOME/Scratch/results/$save_dir

cd $TMPDIR

$HOME/ts4uc/hpc_scripts/train.sh \
    $save_dir $env_fn $workers $epochs $entropy_coef $clip_ratio $ac_lr $cr_lr $ac_arch $cr_arch

tar zcvf $HOME/Scratch/results/$save_dir/results.tar.gz $TMPDIR
