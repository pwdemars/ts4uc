#!/bin/bash
# Training with no entropy annealing (note that this is NOT a CLI argument, just changed in the source code before running)

# set today's date to use as save directory
date=$(date +"%y-%m-%d")

workers=8
epochs=300000
hrs=48
clip_ratio=0.1
ac_lr=3e-05
cr_lr=3e-04
ac_arch="64,64"
cr_arch="400,300"
buffer_size=5000

entropy_coef=0.01
for num_gen in {10,20,30,40,50} ; 
  do qsub -pe smp $workers -l h_rt=${hrs}:00:00 ../submit_train.sh \
     ${date}_convex_opt_comp_ent/g${num_gen}_e01_$RANDOM $HOME/rl-convex-opt/envs/${num_gen}gen.json $workers $epochs $entropy_coef $clip_ratio $ac_lr $cr_lr $ac_arch $cr_arch $buffer_size ;
done

entropy_coef=0.05
for num_gen in {10,20,30,40,50} ; 
  do qsub -pe smp $workers -l h_rt=${hrs}:00:00 ../submit_train.sh \
     ${date}_convex_opt_comp_ent/g${num_gen}_e05_$RANDOM $HOME/rl-convex-opt/envs/${num_gen}gen.json $workers $epochs $entropy_coef $clip_ratio $ac_lr $cr_lr $ac_arch $cr_arch $buffer_size ;
done

entropy_coef=0.1
for num_gen in {10,20,30,40,50} ; 
  do qsub -pe smp $workers -l h_rt=${hrs}:00:00 ../submit_train.sh \
     ${date}_convex_opt_comp_ent/g${num_gen}_e10_$RANDOM $HOME/rl-convex-opt/envs/${num_gen}gen.json $workers $epochs $entropy_coef $clip_ratio $ac_lr $cr_lr $ac_arch $cr_arch $buffer_size ;
done

entropy_coef=0.2
for num_gen in {10,20,30,40,50} ; 
  do qsub -pe smp $workers -l h_rt=${hrs}:00:00 ../submit_train.sh \
     ${date}_convex_opt_comp_ent/g${num_gen}_e20_$RANDOM $HOME/rl-convex-opt/envs/${num_gen}gen.json $workers $epochs $entropy_coef $clip_ratio $ac_lr $cr_lr $ac_arch $cr_arch $buffer_size ;
done

entropy_coef=0.3
for num_gen in {10,20,30,40,50} ; 
  do qsub -pe smp $workers -l h_rt=${hrs}:00:00 ../submit_train.sh \
     ${date}_convex_opt_comp_ent/g${num_gen}_e30_$RANDOM $HOME/rl-convex-opt/envs/${num_gen}gen.json $workers $epochs $entropy_coef $clip_ratio $ac_lr $cr_lr $ac_arch $cr_arch $buffer_size ;
done
