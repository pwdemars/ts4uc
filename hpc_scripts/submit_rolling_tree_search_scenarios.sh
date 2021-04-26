#!/bin/bash -l
#$ -wd /home/uclqpde/Scratch/output
#$ -M patrick.demars.14@ucl.ac.uk
#$ -l mem=8G

# number of array jobs: corresponds to number test profiles multiplied by number of scenarios
#$ -t 1-20000

save_dir=$1
params_filename=$2
env_params_filename=$3
policy_filename=$4
time_budget=$5
branching_threshold=$6
tree_search_func_name=$7
paramfile=$8
heuristic_method=${9:-none}

number=$SGE_TASK_ID

index="`sed -n ${number}p $paramfile | awk '{print $1}'`"
test_data="`sed -n ${number}p $paramfile | awk '{print $2}'`"
scenario_idx="`sed -n ${number}p $paramfile | awk '{print $3}'`"
prof_name="`sed -n ${number}p $paramfile | awk '{print $4}'`"

module load gcc-libs
module load python3/3.7
# export OMP_NUM_THREADS=1

cd $TMPDIR

python $HOME/ts4uc/ts4uc/tree_search/rolling_horizon.py --save_dir ${save_dir}/${prof_name}/${scenario_idx} \
												  --policy_params_fn $params_filename \
												  --env_params_fn $env_params_filename \
												  --policy_filename $policy_filename \
												  --test_data $test_data \
												  --branching_threshold $branching_threshold \
												  --time_budget $time_budget \
												  --num_scenarios 100 \
												  --tree_search_func_name $tree_search_func_name \
												  --heuristic_method $heuristic_method \
												  --error_scenario_idx $scenario_idx \
												  --seed 1

