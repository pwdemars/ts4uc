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
import signal

from rl4uc.environment import make_env

from ts4uc.tree_search import node as node_mod, expansion, informed_search
from ts4uc.tree_search.algos import a_star
from ts4uc import helpers
from ts4uc.agents.ppo_async.ac_agent import ACAgent
from ts4uc.tree_search.scenarios import get_net_demand_scenarios

def solve_day_ahead_anytime(env, 
                            time_budget, 
                            net_demand_scenarios,
                            tree_search_func, 
                            **params):
    """
    """
    env.reset()
    final_schedule = np.zeros((env.episode_length, env.num_gen))

    root = node_mod.Node(env=env,
            parent=None,
            action=None,
            step_cost=0,
            path_cost=0)

    depths = []
    breadths= []
    for t in range(env.episode_length):
        s = time.time()
        path = tree_search_func(root, 
                                      time_budget,
                                      net_demand_scenarios,
                                      **params)
        depth = len(path)
        depths.append(depth)
        breadth = len(root.children)
        breadths.append(breadth)
        
        if depth == 0:
            random_child_bytes = random.sample(list(root.children), 1)[0] #FIXME 
            a_best = root.children[random_child_bytes].action
        else:
            a_best = path[0]

        final_schedule[t, :] = a_best
        env.step(a_best, deterministic=True)

        print(f"Period {env.episode_timestep+1}", np.array(a_best, dtype=int), round(time.time()-s, 2))

        root = root.children[a_best.tobytes()]
        root.parent, root.path_cost = None, 0

        gc.collect()
        
    return final_schedule, depths, breadths


def ida_star(root,
             time_budget,
             net_demand_scenarios,
             heuristic_method,
             recalc_costs=False,
             **policy_kwargs):

    start_time = time.time()

    class TimeoutException(Exception):   # Custom exception class
        pass

    def timeout_handler(signum, frame):   # Custom signal handler
        raise TimeoutException

    # Change the behavior of SIGALRM
    signal.signal(signal.SIGALRM, timeout_handler)

    horizon = 0
    best_path = []
    while (time.time() - start_time) < time_budget:
        time_remaining = time_budget - (time.time() - start_time)
        signal.setitimer(signal.ITIMER_REAL, time_remaining)

        horizon += 1
        terminal_timestep = min(root.state.episode_timestep+horizon,
                                root.state.episode_length-1)

        try:
            best_path, _ = a_star(root,
                                  terminal_timestep,
                                  net_demand_scenarios,
                                  heuristic_method,
                                  early_stopping=False,
                                  recalc_costs=recalc_costs,
                                  **policy_kwargs)
        except TimeoutException:
            break
        else:
            signal.setitimer(signal.ITIMER_REAL, 0)

        # No need to run if best path takes us to the end of the day
        if len(best_path) == (root.state.episode_length -
                              root.state.episode_timestep - 1):
            break

    return best_path


def ida_star_non_unix(root,
                      time_budget,
                      net_demand_scenarios,
                      heuristic_method,
                      **policy_kwargs):
    """
    IDA* that will run on non-Unix system (doesn't rely on signal).
    """
    start_time = time.time()
    horizon = 0
    best_path = []
    node = root
    while (time.time() - start_time) < time_budget:

        horizon += 1  # Increment the horizon (iterative deepening)
        terminal_timestep = min(root.state.episode_timestep+horizon,
                                root.state.episode_length-1)
        print("Horizon: {}".format(horizon))

        if node.state.is_terminal() or (node.state.episode_timestep ==
                                        terminal_timestep):
            best_path, _ = node_mod.get_solution(node)
            return best_path

        node = root  # Return to root

        frontier = queue.PriorityQueue()
        # include the object id in the priority queue.
        # prevents type error when path_costs are identical.
        frontier.put((0, id(node), node))

        while (time.time() - start_time) < time_budget:
            assert frontier, "Failed to find a goal state"
            node = frontier.get()[2]
            if node.state.is_terminal() or (node.state.episode_timestep ==
                                            terminal_timestep):
                best_path, _ = node_mod.get_solution(node)
                break
            actions = expansion.get_actions(node, **policy_kwargs)

            for action in actions:
                net_demand_scenarios_t = np.take(net_demand_scenarios,
                                                 node.state.episode_timestep+1,
                                                 axis=1)
                child = expansion.get_child_node(node,
                                                 action,
                                                 net_demand_scenarios_t)
                child.heuristic_cost = informed_search.heuristic(child,
                                                                 (terminal_timestep -
                                                                  child.state.episode_timestep),
                                                                 heuristic_method)
                node.children[action.tobytes()] = child
                frontier.put((child.path_cost + child.heuristic_cost,
                              id(child),
                              child))

    return best_path


def anytime_uniform_cost_search(node,
                                terminal_timestep,
                                net_demand_scenarios,
                                **policy_kwargs):
    """Uniform cost search with backup"""
    if node.state.is_terminal() or (node.state.episode_timestep ==
                                    terminal_timestep):
        return node_mod.get_solution(node)
    frontier = queue.PriorityQueue()
    # include the object id in the priority queue.
    # prevents type error when path_costs are identical.
    frontier.put((0, id(node), node))
    while True:
        assert frontier, "Failed to find a goal state"
        node = frontier.get()[2]
        if node.state.is_terminal() or (node.state.episode_timestep ==
                                        terminal_timestep):
            return node_mod.get_solution(node)
        actions = expansion.get_actions(node, **policy_kwargs)
        for action in actions:
            net_demand_scenarios_t = np.take(net_demand_scenarios,
                                             node.state.episode_timestep+1,
                                             axis=1)
            child = expansion.get_child_node(node,
                                             action,
                                             net_demand_scenarios_t)
            node.children[action.tobytes()] = child
            frontier.put((child.path_cost, id(child), child))

            # Early stopping if root has one child
            if node.parent is None and len(actions) == 1:
                return [actions[0]], 0
        backup(node)


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
    parser.add_argument('--num_samples', type=int, required=False, default=1000,
                        help='Number of times to sample running the schedules through the environment')
    parser.add_argument('--branching_threshold', type=float, required=False, default=0.05,
                        help='Branching threshold (for guided expansion)')
    parser.add_argument('--seed', type=int, required=False, default=np.random.randint(0,10000),
                        help='Set random seed')
    parser.add_argument('--time_budget', type=float, required=False, default=2,
                        help='Time budget in seconds for anytime algorithm')
    parser.add_argument('--num_scenarios', type=int, required=False, default=100,
                        help='Number of scenarios to use when calculating expected costs')
    parser.add_argument('--tree_search_func_name', type=str, required=False, default='ida_star',
                        help='Tree search algorithm to use')
    parser.add_argument('--heuristic_method', type=str, required=False, default='none',
                        help='Heuristic method to use (when using A* or its variants)')

    args = parser.parse_args()

    if args.branching_threshold == -1:
        args.branching_threshold = None
    if args.heuristic_method.lower() == 'none':
        args.heuristic_method = None

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
    scenarios = get_net_demand_scenarios(profile_df, env, args.num_scenarios)

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
    func_list = [ida_star, ida_star_non_unix]
    func_names = [f.__name__ for f in func_list]
    funcs_dict = dict(zip(func_names, func_list))

    # Run the tree search
    s = time.time()
    schedule_result, depths, breadths = solve_day_ahead_anytime(env=env, 
                                                                net_demand_scenarios=scenarios, 
                                                                tree_search_func=funcs_dict[args.tree_search_func_name],
                                                                policy=policy,
                                                                **params)
    time_taken = time.time() - s

    # Get distribution of costs for solution by running multiple times through environment
    TEST_SAMPLE_SEED=999
    test_costs, lost_loads, test_kgco2 = helpers.test_schedule(env, schedule_result, TEST_SAMPLE_SEED, args.num_samples)
    helpers.save_results(prof_name=prof_name, 
                         save_dir=args.save_dir, 
                         num_gen=env.num_gen, 
                         schedule=schedule_result,
                         test_costs=test_costs, 
                         test_kgco2=test_kgco2,
                         lost_loads=lost_loads,
                         time_taken=time_taken,
                         breadths=breadths,
                         depths=depths)

    print("Done")
    print()
    print("Mean costs: ${:.2f}".format(np.mean(test_costs)))
    print("Lost load prob: {:.3f}%".format(100*np.sum(lost_loads)/(args.num_samples * env.episode_length)))
    print("Time taken: {:.2f}s".format(time_taken))
    print() 

