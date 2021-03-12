#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from ts4uc.tree_search import scenarios
from rl4uc.environment import make_env
from ts4uc.agents.ac_agent import ACAgent
import informed_search

import numpy as np
import pandas as pd
import itertools
import copy
import queue
import gc
import time
import os

class Node(object):
    """Node class for search trees"""
    def __init__(self, env, parent, action, path_cost):
        self.state = env
        self.parent = parent
        self.action = action
        self.path_cost = path_cost
        self.is_expanded = False

def get_actions(env, policy, **policy_kwargs):
    if policy != None:
        return get_actions_with_policy(env, policy, **policy_kwargs)
    else:
        return get_all_actions(env)

def get_all_actions(env):
    """Get all actions from the `env` state"""
    constrained_gens = np.where(np.logical_or(env.must_on, env.must_off))
    unconstrained_gens = np.delete(np.arange(env.num_gen), constrained_gens)

    # All permutations of available generators
    all_perms = np.array(list(itertools.product(range(2), repeat=unconstrained_gens.size)))

    # Create action array 
    actions = np.zeros((all_perms.shape[0], env.num_gen))
    actions[:,constrained_gens] = env.commitment[constrained_gens]
    actions[:,unconstrained_gens] = all_perms
    
    return actions

def get_actions_with_policy(env, policy, num_samples=1000, branching_threshold=0.05):
    """
    Use a policy to get a state of actions for the `env` state.

    This is used to implement `guided expansion`.
    """
    action_dict, log_prob = policy.generate_multiple_actions_batched(env, env.state, num_samples, branching_threshold)
    
    action = np.where(env.status > 0, 1, 0)
    action_id = ''.join(str(int(i)) for i in action)
    action_id = int(action_id, 2)
    action_dict.update({action_id: action})

    actions = np.array(list(action_dict.values()))

    return actions

def get_child_node(node, action, net_demand_scenarios=None, deterministic=True):
    """
    Return a child node corresponding to taking `action` from the state 
    corresponding to `node`.
    
    The child node has `node` as its parent.
    """
    new_env = copy.deepcopy(node.state)
    _, reward, _ = new_env.step(action, deterministic=deterministic)

    if net_demand_scenarios is None:
        cost = -reward
    else:
        cost = scenarios.calculate_expected_costs(new_env, net_demand_scenarios)

    child = Node(env=new_env,
                parent=node,
                action=action, 
                path_cost = node.path_cost + cost)
    
    return child

def get_solution(node):
    """Return the solution path (list of actions) leading to node."""
    s = []
    path_cost = node.path_cost
    while node.parent is not None:
        s.insert(0, node.action)
        node = node.parent
    return s, path_cost

def uniform_cost_search(env, 
                        terminal_timestep, 
                        net_demand_scenarios,
                        **policy_kwargs):
    """Uniform cost search"""
    node = Node(env=env,
                parent=None,
                action=None,
                path_cost=0)
    if node.state.is_terminal() or node.state.episode_timestep == terminal_timestep:
        return get_solution(node)
    frontier = queue.PriorityQueue()
    frontier.put((0, id(node), node)) # include the object id in the priority queue. prevents type error when path_costs are identical.
    while True:
        assert frontier, "Failed to find a goal state"
        node = frontier.get()[2]
        if node.state.is_terminal() or node.state.episode_timestep == terminal_timestep:
            return get_solution(node)
        for action in get_actions(node.state, 
                                  **policy_kwargs):
            net_demand_scenarios_t = np.take(net_demand_scenarios, node.state.episode_timestep+1, axis=1)
            child = get_child_node(node, action, net_demand_scenarios_t)
            frontier.put((child.path_cost, id(child), child))

def a_star(env, 
           terminal_timestep, 
           net_demand_scenarios,
           **policy_kwargs):
    """Uniform cost search"""
    node = Node(env=env,
                parent=None,
                action=None,
                path_cost=0)
    if node.state.is_terminal() or node.state.episode_timestep == terminal_timestep:
        return get_solution(node)
    frontier = queue.PriorityQueue()
    frontier.put((0, id(node), node)) # include the object id in the priority queue. prevents type error when path_costs are identical.
    while True:
        assert frontier, "Failed to find a goal state"
        node = frontier.get()[2]
        if node.state.is_terminal() or node.state.episode_timestep == terminal_timestep:
            return get_solution(node)
        for action in get_actions(node.state, 
                                  **policy_kwargs):
            net_demand_scenarios_t = np.take(net_demand_scenarios, node.state.episode_timestep, axis=1)
            child = get_child_node(node, action, net_demand_scenarios_t)
            heuristic_cost = informed_search.heuristic(child, terminal_timestep - child.state.episode_timestep)
            frontier.put((child.path_cost + heuristic_cost, id(child), child))

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

    for t in range(env.episode_length):
        terminal_timestep = min(env.episode_timestep + horizon, env.episode_length-1)
        path, cost = tree_search_func(env, 
                                      terminal_timestep, 
                                      net_demand_scenarios,
                                      **policy_kwargs)
        a_best = path[0]
        print(f"Period {env.episode_timestep+1}", np.array(a_best, dtype=int), cost)
        final_schedule[t, :] = a_best
        env.step(a_best, deterministic=True)
        gc.collect()
        
    return final_schedule

if __name__=="__main__":

    import json
    import time
    import torch

    # User inputs:
    HORIZON=2
    NUM_SCENARIOS=100
    NUM_GEN=5
    NUM_TEST_SAMPLES=1000
    BRANCHING_THRESHOLD=0.1
    PROF_NAME = 'profile_2017-05-26'
    SAVE_DIR = 'foo'
    PROFILE_FN = '../../data/day_ahead/5gen/30min/{}.csv'.format(PROF_NAME)
    PARAMS_FN = '../../results/feb4_g5_d30_v1/params.json'
    POLICY_FN = '../../results/feb4_g5_d30_v1/ac_final.pt'
    TREE_SEARCH_FUNC = a_star

    os.makedirs(SAVE_DIR, exist_ok=True)

    profile_df = pd.read_csv(PROFILE_FN)
    params = json.load(open(PARAMS_FN))
    env = make_env(mode='test', profiles_df=profile_df, **params)
    env.reset()

    # Generate scenarios
    np.random.seed(2)
    demand_errors, wind_errors = scenarios.get_scenarios(env, NUM_SCENARIOS, env.episode_length)
    net_demand_scenarios = (profile_df.demand.values + demand_errors) - (profile_df.wind.values + wind_errors)
    net_demand_scenarios = np.clip(net_demand_scenarios, env.min_demand, env.max_demand)

    # Load the policy
    policy = ACAgent(env, **params)
    policy.load_state_dict(torch.load(POLICY_FN))

    s = time.time()
    schedule_result = solve_day_ahead(env, 
                                      HORIZON, 
                                      net_demand_scenarios, 
                                      policy=None, 
                                      branching_threshold=
                                      BRANCHING_THRESHOLD, 
                                      tree_search_func=a_star)
    time_taken = time.time() - s

    # Get distribution of costs for solution by running multiple times through environment
    TEST_SAMPLE_SEED=999
    test_costs, lost_loads = helpers.test_schedule(env, schedule_result, TEST_SAMPLE_SEED, NUM_TEST_SAMPLES)
    helpers.save_results(PROF_NAME, SAVE_DIR, env.num_gen, schedule_result, test_costs, lost_loads, time_taken)

    print("Done")
    print()
    print("Mean costs: ${:.2f}".format(np.mean(test_costs)))
    print("Lost load prob: {:.3f}%".format(100*np.sum(lost_loads)/(NUM_TEST_SAMPLES * env.episode_length)))
    print("Time taken: {:.2f}s".format(time_taken))
    print() 

    