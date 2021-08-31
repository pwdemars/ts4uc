#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from rl4uc.environment import make_env

from ts4uc.tree_search import node
from ts4uc import helpers

from ts4uc.tree_search.scenarios import get_net_demand_scenarios, get_scenarios, get_global_outage_scenarios
from ts4uc.agents.ppo_async.ac_agent import ACAgent
from ts4uc.tree_search.algos import uniform_cost_search, a_star, rta_star, brute_force

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
                    demand_scenarios,
                    wind_scenarios,
                    tree_search_func=uniform_cost_search, 
                    **params):
    """
    Solve a day rooted at env. 
    
    Return the schedule and the number of branches at the root for each time period. 
    """
    env.reset()
    final_schedule = np.zeros((env.episode_length, env.action_size))

    root = node.Node(env=env,
            parent=None,
            action=None,
            step_cost=0,
            path_cost=0)

    if env.outages:
        root.availability_scenarios = np.ones(shape=(demand_scenarios.shape[0], env.num_gen))

    period_times = []
    breadths = []
    for t in range(env.episode_length):
        period_start_time = time.time()
        terminal_timestep = min(env.episode_timestep + horizon, env.episode_length-1)
        path, cost = tree_search_func(root, 
                                      terminal_timestep, 
                                      demand_scenarios,
                                      wind_scenarios,
                                      **params)
        a_best = path[0]

        breadth = len(root.children)
        breadths.append(breadth)

        final_schedule[t, :] = a_best
        env.step(a_best, deterministic=True)
        print(f"Period {env.episode_timestep+1}", np.array(a_best, dtype=int), round(cost, 2), round(time.time()-period_start_time, 2))

        root = root.children[a_best.tobytes()]
        root.parent, root.path_cost = None, 0

        gc.collect()

        period_times.append(time.time()-period_start_time)

    return final_schedule, period_times, breadths

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Solve a single day with tree search')
    parser.add_argument('--save_dir', '-s', type=str, required=True,
                        help='Directory to save results')
    parser.add_argument('--policy_params_fn', '-pp', type=str, required=False,
                        help='Filename for parameters', default='none')
    parser.add_argument('--env_params_fn', '-e', type=str, required=True,
                        help='Filename for environment parameters, including ARMAs, number of generators, dispatch frequency')
    parser.add_argument('--policy_filename', '-pf', type=str, required=False,
                        help="Filename for policy [.pt]. Set to 'none' or omit this argument to train from scratch", default='none')
    parser.add_argument('--test_data', '-t', type=str, required=True,
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
    parser.add_argument('--heuristic_method', type=str, required=False, default='check_lost_load',
                        help='Heuristic method to use (when using A* or its variants)')

    args = parser.parse_args()

    if args.branching_threshold == -1:
        args.branching_threshold = None
    if args.heuristic_method.lower() == 'none':
        args.heuristic_method = None
    if args.policy_params_fn.lower() == 'none':
        args.policy_params_fn = None
    if args.policy_filename.lower() == 'none':
        args.policy_filename = None

    # Create results directory
    os.makedirs(args.save_dir, exist_ok=True)

    # Update params
    params = vars(args)

    # Read the parameters
    env_params = json.load(open(args.env_params_fn))
    if args.policy_params_fn is not None:
        policy_params = json.load(open(args.policy_params_fn))

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

    # Generate scenarios for demand and wind errors
    # scenarios = get_net_demand_scenarios(profile_df, env, args.num_scenarios)
    demand_scenarios, wind_scenarios = get_scenarios(profile_df, env, args.num_scenarios)
    if env.outages:
        global_outage_scenarios = get_global_outage_scenarios(env, env.episode_length + env.gen_info.status.max(), args.num_scenarios)
    else:
        global_outage_scenarios = None

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
    schedule_result, period_times, breadths = solve_day_ahead(env=env, 
                                      demand_scenarios=demand_scenarios, 
                                      wind_scenarios=wind_scenarios,
                                      global_outage_scenarios=global_outage_scenarios,
                                      tree_search_func=funcs_dict[args.tree_search_func_name],
                                      policy=policy,
                                      **params)
    time_taken = time.time() - s

    # Get distribution of costs for solution by running multiple times through environment
    TEST_SAMPLE_SEED=999
    results = helpers.test_schedule(env, schedule_result, TEST_SAMPLE_SEED, args.num_samples)

    helpers.save_results(prof_name=prof_name, 
                         save_dir=args.save_dir, 
                         env=env, 
                         schedule=schedule_result,
                         test_costs=results['total_cost'], 
                         test_kgco2=results['kgco2'],
                         lost_loads=results['lost_load_events'],
                         results_df=results,
                         time_taken=time_taken,
                         breadths=breadths,
                         period_time_taken=period_times)

    print("Done")
    print()
    print("Mean costs: ${:.2f}".format(np.mean(results['total_cost'])))
    print("Lost load prob: {:.3f}%".format(100*np.sum(results['lost_load_events'])/(args.num_samples * env.episode_length)))
    print("Time taken: {:.2f}s".format(time_taken))
    print("Mean curtailed {:.2f}MWh".format(results.curtailed_mwh.mean()))
    print("Mean CO2: {:.2f}kg".format(results.kgco2.mean()))
    print() 
