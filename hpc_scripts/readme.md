# Experiments

The files in this folder are used to run the experiments.

Here are the experiments:

## Chapter 1 

**All guided tree searches in this chapter use uniform-cost search.**

- `101-train_policies.sh` (TODO): train expansion policies for 5, 6, 7, 8, 9, 10, 20 and 30 generator systems, to be used for guided tree search. 
- `102-observation_processor_comparison.sh`: relatively short training of two expansion policies, one using the DayAheadProcessor class to process observations, the other using the LimitedHorizonProcessor. 
- `103-guided_vs_unguided.sh`: guided and unguided tree search for 5--10 generators inclusive, using policies trained in experiment 101. Setting H=2 and rho=0.05 for all. 
- `104-guided_parameter_comparison.sh`: compare parameters for 5 generator guided tree search, varying H and rho.
- `105-guided_large_systems.sh`: run guided tree search with policies from 101 for the 10, 20 and 30 generator problems using H=4, rho=0.05.

## Chapter 2

- `201-heuristic_comparison.sh`: compare A* and RTA* using 3 heuristic methods `check_lost_load`, `priority_list` and `pl_plus_ll` for 5, 10, 20 and 30 generator day-ahead problems. Note: results should ultimately be combined with the uniform-cost search results from experiment 105 (for 10, 20 and 30 generator problems). H={2,4,6}, rho=0.05. 
- `202-anytime_ida_star.sh` (TODO): using the same heuristics from the previous experiments, use IDA* to solve the day ahead problem with increasing computational budget: {1, 2, 5, 10, 30, 60}s. 
