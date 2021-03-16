#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from rl4uc.environment import make_env

from ts4uc.tree_search import scenarios
from ts4uc.agents.ac_agent import ACAgent
from ts4uc import helpers
from ts4uc.tree_search.algos import Node, uniform_cost_search, a_star, rta_star, brute_force

import numpy as np
import argparse 
import torch
import pandas as pd
import os
import json
import gc
import time

def solve_day_ahead(env, 
                    horizon, 
                    net_demand_scenarios,
                    tree_search_func=uniform_cost_search, 
                    **policy_kwargs):
    """
    Solve a day rooted at env. 
    
    Return the schedule and the number of branches at the root for each time period. 
    """
    env.reset()
    final_schedule = np.zeros((env.episode_length, env.num_gen))

    node = Node(env=env,
            parent=None,
            action=None,
            step_cost=0,
            path_cost=0)

    for t in range(env.episode_length):
        terminal_timestep = min(env.episode_timestep + horizon, env.episode_length-1)
        path, cost = tree_search_func(node, 
                                      terminal_timestep, 
                                      net_demand_scenarios,
                                      **policy_kwargs)
        a_best = path[0]
        print(f"Period {env.episode_timestep+1}", np.array(a_best, dtype=int), cost)
        final_schedule[t, :] = a_best
        env.step(a_best, deterministic=True)

        node = node.children[a_best.tobytes()]
        node.parent, node.path_cost = None, 0

        gc.collect()
        
    return final_schedule

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Solve a single day with tree search')
    parser.add_argument('--save_dir', type=str, required=True,
                        help='Directory to save results')
    parser.add_argument('--policy_params_fn', type=str, required=False,
                        help='Filename for parameters')
    parser.add_argument('--env_params_fn', type=str, required=True,
                        help='Filename for environment parameters, including ARMAs, number of generators, dispatch frequency')
    parser.add_argument('--policy_filename', type=str, required=False,
                        help="Filename for policy [.pt]. Set to 'none' or omit this argument to train from scratch", default=None)
    parser.add_argument('--test_data', type=str, required=True,
                        help='Location of problem file [.csv]')
    parser.add_argument('--num_samples', type=int, required=False, default=1000,
                        help='Number of times to sample running the schedules through the environment')
    parser.add_argument('--branching_threshold', type=float, required=False, default=0.05,
                        help='Branching threshold (for guided expansion)')
    parser.add_argument('--seed', type=int, required=False, default=np.random.randint(0,10000),
                        help='Set random seed')
    parser.add_argument('--horizon', type=int, required=False, default=1,
                        help='Lookahead horizon')
    parser.add_argument('--num_scenarios', type=int, required=False, default=100,
                        help='Number of scenarios to use when calculating expected costs')
    parser.add_argument('--tree_search_func_name', type=str, required=False, default='uniform_cost_search',
                        help='Tree search algorithm to use')

    args = parser.parse_args()

    # For HPC purposes, allow 'none' to be passed as arg to policy_filename
    if args.policy_filename == "none": args.policy_filename = None
    if args.policy_params_fn == "none": args.policy_params_fn = None
    if args.branching_threshold == -1: args.branching_threshold = None

    # Create results directory
    os.makedirs(args.save_dir, exist_ok=True)

    # Update params
    params = vars(args)

    # Read the parameters
    env_params = json.load(open(args.env_params_fn))
    if args.policy_params_fn is not None: policy_params = json.load(open(args.policy_params_fn))

    # Set random seeds
    print(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Save params file to save_dir 
    with open(os.path.join(args.save_dir, 'params.json'), 'w') as fp:
        fp.write(json.dumps(params, sort_keys=True, indent=4))

    # Save env params to save_dir
    with open(os.path.join(args.save_dir, 'env_params.json'), 'w') as fp:
        fp.write(json.dumps(env_params, sort_keys=True, indent=4))

    prof_name = os.path.basename(os.path.normpath(args.test_data)).split('.')[0]

    print("------------------")
    print("Profile: {}".format(prof_name))
    print("------------------")

    # Initialise environment with forecast profile and reference forecast (for scaling)
    profile_df = pd.read_csv(args.test_data)
    env = make_env(mode='test', profiles_df=profile_df, **env_params)
    print(env.num_gen)

    # Generate scenarios for demand and wind errors
    demand_errors, wind_errors = scenarios.get_scenarios(env, args.num_scenarios, env.episode_length)
    scenarios = (profile_df.demand.values + demand_errors) - (profile_df.wind.values + wind_errors)
    scenarios = np.clip(scenarios, env.min_demand, env.max_demand)

    # Load policy 
    if args.policy_filename is not None:
        policy = ACAgent(env, test_seed=args.seed, **policy_params)
        if torch.cuda.is_available():
            policy.cuda()
        policy.load_state_dict(torch.load(args.policy_filename))        
        policy.eval()
        print("Guided search")
    else:
        policy = None
        print("Unguided search")

    # Convert the tree_search_method argument to a function:
    func_list = [uniform_cost_search, a_star, rta_star, brute_force]
    func_names = [f.__name__ for f in func_list]
    funcs_dict = dict(zip(func_names, func_list))

    # Run the tree search
    s = time.time()
    schedule_result = solve_day_ahead(env=env, 
                                      net_demand_scenarios=scenarios, 
                                      tree_search_func=funcs_dict[args.tree_search_func_name],
                                      policy=policy,
                                      **params)
    time_taken = time.time() - s

    # Get distribution of costs for solution by running multiple times through environment
    TEST_SAMPLE_SEED=999
    test_costs, lost_loads = helpers.test_schedule(env, schedule_result, TEST_SAMPLE_SEED, args.num_samples)
    helpers.save_results(prof_name, args.save_dir, env.num_gen, schedule_result, test_costs, lost_loads, time_taken)

    print("Done")
    print()
    print("Mean costs: ${:.2f}".format(np.mean(test_costs)))
    print("Lost load prob: {:.3f}%".format(100*np.sum(lost_loads)/(args.num_samples * env.episode_length)))
    print("Time taken: {:.2f}s".format(time_taken))
    print()	
