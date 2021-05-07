#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import time 
import queue
import gc
import pandas as pd
import os
import torch
import argparse
import json
import random

from rl4uc.environment import make_env

from ts4uc.tree_search import node as node_mod, expansion, informed_search, scenarios
from ts4uc.tree_search.anytime import ida_star
from ts4uc import helpers
from ts4uc.agents.ppo_async.ac_agent import ACAgent

def solve_rolling_anytime(env,
                          time_budget,
                          tree_search_func,
                          real_errors=None,
                          **params):
    """
    Solve the UC problem in a rolling context, beginning with the state defined by env.
    """
    final_schedule = np.zeros((env.episode_length, env.num_gen))
    env.reset()
    operating_cost = 0
    depths = []
    real_net_demands = []

    ens_count = 0

    net_demand_scenarios = np.zeros((params.get('num_scenarios'), env.episode_length))

    # initialise env
    root = node_mod.Node(env=env,
            parent=None,
            action=None,
            step_cost=0,
            path_cost=0)
    
    for t in range(env.episode_length):
        # generate new scenarios, seeded by the ARMA processes in env. 
        remaining_periods = env.episode_length - t 
        demand_errors, wind_errors = scenarios.sample_errors(env,
                                                             params.get('num_scenarios'),
                                                             remaining_periods,
                                                             seeded=True)
        demand_forecast = env.profiles_df.demand[t:].values
        wind_forecast = env.profiles_df.wind[t:].values

        new_scenarios = (demand_forecast + demand_errors) - (wind_forecast + wind_errors)
        new_scenarios = np.clip(new_scenarios, env.min_demand, env.max_demand)

        net_demand_scenarios[:,env.episode_timestep+1:] = new_scenarios

        # find least expected cost path 
        path = tree_search_func(root,
                                time_budget,
                                net_demand_scenarios,
                                recalc_costs=True,
                                **params)

        depth = len(path)
        print(depth)
        depths.append(depth)
        if depth == 0:
            random_child_bytes = random.sample(list(root.children), 1)[0]
            a_best = root.children[random_child_bytes].action
        else:
            a_best = path[0]

        final_schedule[t, :] = a_best

        if isinstance(real_errors, pd.DataFrame):
            s_error = real_errors.iloc[t]
            errors = {'demand': (s_error.demand_x, s_error.demand_z),
                      'wind': (s_error.wind_x, s_error.wind_z)}
        else:
            errors = None
        
        # sample new state
        _, reward, _ = env.step(a_best, errors=errors)

        print(f"Period {env.episode_timestep}", np.array(a_best, dtype=int), np.round(-reward ,2))

        operating_cost -= reward
        real_net_demands.append(env.net_demand)
        if env.ens: ens_count += 1

        root = root.children[a_best.tobytes()]
        root.parent, root.path_cost = None, 0

        gc.collect()

    lolp = ens_count / env.episode_length

    return final_schedule, operating_cost, lolp, real_net_demands, depths


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Solve a single day with tree search')
    parser.add_argument('--save_dir', '-s', type=str, required=True,
                        help='Directory to save results')
    parser.add_argument('--policy_params_fn', '-pp', type=str, required=False,
                        help='Filename for parameters')
    parser.add_argument('--env_params_fn', '-e', type=str, required=True,
                        help='Filename for environment parameters, including ARMAs, number of generators, dispatch frequency')
    parser.add_argument('--policy_filename', '-pf', type=str, required=False,
                        help="Filename for policy [.pt]. Set to 'none' or omit this argument to train from scratch", default=None)
    parser.add_argument('--test_data', '-t', type=str, required=True,
                        help='Location of problem file [.csv]')
    parser.add_argument('--branching_threshold', type=float, required=False, default=0.05,
                        help='Branching threshold (for guided expansion)')
    parser.add_argument('--seed', type=int, required=False, default=np.random.randint(0,10000),
                        help='Set random seed')
    parser.add_argument('--time_budget', type=float, required=False, default=1,
                        help='Time budget in seconds for anytime algorithm')
    parser.add_argument('--tree_search_func_name', type=str, required=False, default='ida_star',
                        help='Tree search algorithm to use')
    parser.add_argument('--num_scenarios', type=int, required=False, default=100,
                        help='Number of scenarios to use when calculating expected costs')
    parser.add_argument('--heuristic_method', type=str, required=False, default='none',
                        help='Heuristic method to use (when using A* or its variants)')
    parser.add_argument('--error_scenario_idx', type=str, required=False, default=None,
                        help='Scenario index if running a specific scenario for forecast errors')

    args = parser.parse_args()

    if args.branching_threshold == -1: args.branching_threshold = None
    if args.heuristic_method.lower() == 'none': args.heuristic_method = None

    # Create results directory
    os.makedirs(args.save_dir, exist_ok=True)

    # Update params
    params = vars(args)

    # Add 'rolling'=True' to params
    params.update({'rolling': True})

    # Read the parameters
    env_params = json.load(open(args.env_params_fn))
    if args.policy_params_fn is not None: policy_params = json.load(open(args.policy_params_fn))

    # Read the error scenarios
    if args.error_scenario_idx != None:
        real_errors = helpers.retrieve_error_scenarios(env_params.get('num_gen'))
        real_errors = real_errors[real_errors.scenario==int(args.error_scenario_idx)]
        real_errors = real_errors.set_index('period')
    else:
        real_errors = None

    # Set random seeds
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
    func_list = [ida_star]
    func_names = [f.__name__ for f in func_list]
    funcs_dict = dict(zip(func_names, func_list))

    # Run the tree search
    s = time.time()
    schedule_result, cost, lolp, real_net_demands, depths = solve_rolling_anytime(env=env, 
                                                                                  tree_search_func=funcs_dict[args.tree_search_func_name],
                                                                                  policy=policy,
                                                                                  real_errors=real_errors,
                                                                                  **params)
    time_taken = time.time() - s

    helpers.save_results_rolling(prof_name=prof_name,
                                 save_dir=args.save_dir,
                                 schedule=schedule_result,
                                 real_net_demands=real_net_demands,
                                 cost=cost,
                                 time=time_taken,
                                 lolp=lolp,
                                 depths=depths
                                )

    print("Operating cost: ${:.2f}".format(cost))
    print("Time taken: {:.2f}s".format(time_taken))




