#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from scenarios import get_scenarios, calculate_expected_costs
from rl4uc.environment import make_env
from ts4uc.tree_search.mcts.node import DecisionNode, DummyNode
import ts4uc.helpers as helpers


import numpy as np
import argparse 
import torch
import pandas as pd
import os
import json
import gc
import time

def find_best_path(node, H, scenarios, expansion_mode='guided', cost_to_go=False, step_size=1):
    """
    Find the best path from node up to a horizon H, returning the path
    and its expected cost.

    This is a depth first search that exhaustively searches a tree up to depth
    H, using an expansion policy to determine the branching factor.
    """
    if node.expected_cost is None:
        net_demands = np.take(scenarios, node.depth_from_root(), axis=1)
        # net_demands = np.take(scenarios, node.environment.episode_timestep, axis=1)
        node.expected_cost = calculate_expected_costs(node.environment, net_demands)

    if H == 0 or node.node_is_terminal():
        if cost_to_go is True:
            cost_to_go = -node.simulate_pl()
        else:
            cost_to_go = 0
        return [node.environment.commitment], node.expected_cost + cost_to_go

    if node.is_expanded is False:
        node.expansion_mode = expansion_mode # change the expansion mode: vanilla or guided
        node.expand_decision()
        node.is_expanded = True

    if (node.depth_from_root() == 0) and (len(node.children) == 1):
        random_node = list(node.children.values())[0]
        random_node.expand_deterministic(step_size=step_size)
        return [node.environment.commitment, list(node.children.values())[0].action], None

    options = []
    for random_node in list(node.children.values()):
        # get to the next decision node
        random_node.expand_deterministic(step_size=step_size)
        child = random_node.children[0]
        path, cost = find_best_path(child, H-1, scenarios, expansion_mode, cost_to_go)
        options.append((path, cost))

    path, cost = min(options, key=lambda option: option[1])
    path.insert(0, node.environment.commitment)

    return path, cost + node.expected_cost

def solve_day_ahead(env, H, scenarios, node_params, policy_network=None, expansion_mode='guided', cost_to_go=False, retain_tree=False, step_size=1):
    """
    Solve a day rooted at env. 
    
    Return the schedule and the number of branches at the root for each time period. 
    """
    env.reset()
    final_schedule = np.zeros((env.episode_length, env.num_gen))
    n_branches = np.zeros(env.episode_length, dtype=int)

    # Initialise root
    root = DecisionNode(environment=env, 
        policy_network=policy_network, 
        parent=DummyNode(env.num_gen),
       **node_params)    

    for t in range(env.episode_length):
        if retain_tree is False:
            root = DecisionNode(environment=env, 
                policy_network=policy_network, 
                parent=DummyNode(env.num_gen),
               **node_params)
        path, cost = find_best_path(node=root, 
                                    H=H, 
                                    scenarios=scenarios[:,env.episode_timestep+1:], 
                                    expansion_mode=expansion_mode, 
                                    cost_to_go=cost_to_go, 
                                    step_size=step_size)
        a_best = path[1] # take first action in least cost path (index 0 is the current commitment)
        print(f"Period {root.environment.episode_timestep + 1}", np.array(a_best, dtype=int), cost)
        final_schedule[t, :] = a_best
        n_branches[t] = len(root.children) # number of children at the root
        env.step(a_best, deterministic=True)

        # Advance root node to best child
        if retain_tree:
            best_action_id = int(''.join(str(int(i)) for i in a_best), 2)
            root = root.children[best_action_id].children[0]
            del root.parent
            root.parent = DummyNode(env.num_gen)

        gc.collect()
        
    return final_schedule, n_branches

def solve_rolling(env, H, node_params, policy_network=None, num_scenarios=100, expansion_mode='guided', cost_to_go=False, step_size=1):
    """
    Solve the UC problem in a rolling context, beginning with the state defined by env.
    """
    final_schedule = np.zeros((env.episode_length, env.num_gen))
    env.reset() 
    operating_cost = 0
    for t in range(env.episode_length):
        # initialise env
        root = DecisionNode(environment=env, 
            policy_network=policy_network, 
            parent=DummyNode(env.num_gen),
           **node_params)   
        # generate scenarios
        demand_errors, wind_errors = get_scenarios(env, num_scenarios, H)
        demand_forecast = env.state['demand_forecast'][:H]
        wind_forecast = env.state['wind_forecast'][:H]

        scenarios = (demand_forecast + demand_errors) - (wind_forecast + wind_errors)
        scenarios = np.clip(scenarios, env.min_demand, env.max_demand)
        scenarios = np.insert(scenarios, 0, [0]*num_scenarios, axis=1)

        # find least expected cost path 
        path, path_cost = find_best_path(node=root, 
                                    H=H, 
                                    scenarios=scenarios, 
                                    expansion_mode=expansion_mode, 
                                    cost_to_go=cost_to_go, 
                                    step_size=step_size)
        # choose best action
        a_best = path[1]
        final_schedule[t, :] = a_best

        # sample new state
        _, reward, _ = env.step(a_best, deterministic=False)

        print(f"Period {root.environment.episode_timestep}", np.array(a_best, dtype=int), -reward)

        operating_cost -= reward

        gc.collect()

    return final_schedule, operating_cost    
