#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from ts4uc.tree_search.scenarios import get_scenarios
from rl4uc.environment import make_env
from ts4uc.agents.ac_agent import ACAgent
import ts4uc.helpers as helpers
import ts4uc.tree_search.tree_search as tree_search

import numpy as np
import argparse 
import torch
import pandas as pd
import os
import json
import gc
import time

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Solve a single day with tree search')
    parser.add_argument('--save_dir', type=str, required=True,
                        help='Directory to save results')
    parser.add_argument('--params_fn', type=str, required=True,
                        help='Filename for parameters')
    parser.add_argument('--arma_params_fn', type=str, required=True,
                        help='Filename for ARMA parameters')
    parser.add_argument('--policy_filename', type=str, required=False,
                        help="Filename for policy [.pt]. Set to 'none' or omit this argument to train from scratch", default=None)
    parser.add_argument('--test_data', type=str, required=True,
                        help='Location of problem file [.csv]')
    parser.add_argument('--num_samples', type=int, required=False, default=1000,
                        help='Number of times to sample running the schedules through the environment')
    parser.add_argument('--decision_branching_threshold', type=float, required=False, default=0.01,
                        help='Decision node branching threshold')
    parser.add_argument('--expansion_mode', type=str, required=False, default='guided',
                        help='Method to use for the expansion nodes')
    parser.add_argument('--seed', type=int, required=False, default=None,
                        help='Set random seed')
    parser.add_argument('--horizon', type=int, required=False, default=1,
                        help='Lookahead horizon')
    parser.add_argument('--num_scenarios', type=int, required=False, default=100,
                        help='Number of scenarios to use when calculating expected costs')
    parser.add_argument('--cost_to_go', type=str, required=False, default="false",
                        help='Use cost to go heuristic for state evaluation. Default is False')
    parser.add_argument('--step_size', type=int, required=False, default=1,
                        help='The resolution at which the search tree is built. Default is 1: there is a node for every timestep')

    args = parser.parse_args()

    # For HPC purposes, allow 'none' to be passed as arg to policy_filename
    if args.policy_filename == "none":
        args.policy_filename = None

    if args.cost_to_go == "true":
        args.cost_to_go = True
    elif args.cost_to_go == "false":
        args.cost_to_go = False
    print(args.cost_to_go)

    # Create results directory
    os.makedirs(args.save_dir, exist_ok=True)

    # Load parameters
    with open(args.params_fn) as f:
        params = json.load(f)

    # Update params
    params.update({'decision_branching_threshold': args.decision_branching_threshold,
                   'expansion_mode': args.expansion_mode,
                   'cost_to_go': args.cost_to_go,
                   'horizon': args.horizon,
                   'step_size': args.step_size})

    # Read the ARMA parameters. 
    arma_params = json.load(open(args.arma_params_fn))
    params.update({'arma_params': arma_params})

    # Set random seeds
    if args.seed is None or args.seed == "none":
        seed = np.random.randint(0,10000)
    else:
        seed = args.seed
    params.update({'test_seed': seed})
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Save params file to save_dir 
    with open(os.path.join(args.save_dir, 'params.json'), 'w') as fp:
        fp.write(json.dumps(params, sort_keys=True, indent=4))

    # Save arma params to save_dir
    with open(os.path.join(args.save_dir, 'arma_params.json'), 'w') as fp:
        fp.write(json.dumps(arma_params, sort_keys=True, indent=4))

    prof_name = os.path.basename(os.path.normpath(args.test_data)).split('.')[0]

    print("------------------")
    print("Profile: {}".format(prof_name))
    print("------------------")

    # Initialise environment with forecast profile and reference forecast (for scaling)
    profile_df = pd.read_csv(args.test_data)
    env = make_env(mode='test', profiles_df=profile_df, **params)

    # Generate scenarios for demand and wind errors
    demand_errors, wind_errors = get_scenarios(env, args.num_scenarios, env.episode_length)
    scenarios = (profile_df.demand.values + demand_errors) - (profile_df.wind.values + wind_errors)
    scenarios = np.clip(scenarios, env.min_demand, env.max_demand)
    scenarios = np.insert(scenarios, 0, [0]*args.num_scenarios, axis=1) # Insert the initial scenarios for the root node (can be set to 0). 

    # Load policy 
    if args.policy_filename is not None:
        policy_network = ACAgent(env, **params)
        if torch.cuda.is_available():
            policy_network.cuda()
        policy_network.load_state_dict(torch.load(args.policy_filename))        
        policy_network.eval()
        print("Using trained policy network")
    else:
        policy_network = None
        print("Using untrained policy network")

    print(f"Using cost to go: {args.cost_to_go}")
    print(f"Horizon: {args.horizon}")

    # Run the tree search
    retain_tree = False if args.step_size > 1 else True
    s = time.time()
    schedule_result, n_branches = tree_search.solve_day_ahead(env=env, 
                                                              H=args.horizon, 
                                                              scenarios=scenarios, 
                                                              policy_network=policy_network, 
                                                              expansion_mode=args.expansion_mode, 
                                                              cost_to_go=args.cost_to_go, 
                                                              retain_tree=retain_tree, 
                                                              step_size=args.step_size,
                                                              node_params=params)
    time_taken = time.time() - s

    # Get distribution of costs for solution by running multiple times through environment
    TEST_SAMPLE_SEED=999
    test_costs, lost_loads = helpers.test_schedule(env, schedule_result, TEST_SAMPLE_SEED, args.num_samples)
    helpers.save_results(prof_name, args.save_dir, env.num_gen, schedule_result, test_costs, lost_loads, time_taken)
    helpers.save_branches(prof_name, args.save_dir, n_branches)

    print("Done")
    print()
    print("Mean costs: ${:.2f}".format(np.mean(test_costs)))
    print("Lost load prob: {:.3f}%".format(100*np.sum(lost_loads)/(args.num_samples * env.episode_length)))
    print("Time taken: {:.2f}s".format(time_taken))
    print()	
