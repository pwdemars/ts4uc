#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from ts4uc.tree_search import scenarios, expansion, informed_search
from ts4uc.tree_search import node as node_mod
from rl4uc.environment import make_env
from ts4uc.tree_search.mcts.node import DecisionNode, DummyNode


import numpy as np
import pandas as pd
import itertools
import copy
import queue
import gc
import time
import os
import heapq

DEFAULT_HEURISTIC_METHOD='check_lost_load'

def uniform_cost_search(node, 
                        terminal_timestep, 
                        demand_scenarios,
                        wind_scenarios,
                        global_outage_scenarios,
                        **policy_kwargs):
    """Uniform cost search"""
    if node.state.is_terminal() or node.state.episode_timestep == terminal_timestep:
        return node_mod.get_solution(node)
    frontier = queue.PriorityQueue()
    frontier.put((0, id(node), node)) # include the object id in the priority queue. prevents type error when path_costs are identical.
    while True:
        assert frontier, "Failed to find a goal state"
        node = frontier.get()[2]
        if node.state.is_terminal() or node.state.episode_timestep == terminal_timestep:
            return node_mod.get_solution(node)
        actions = expansion.get_actions(node, **policy_kwargs)
        for action in actions:
            demand_scenarios_t = np.take(demand_scenarios, node.state.episode_timestep+1, axis=1)
            wind_scenarios_t = np.take(wind_scenarios, node.state.episode_timestep+1, axis=1)
            child = expansion.get_child_node(node=node, 
                                             action=action, 
                                             demand_scenarios=demand_scenarios_t,
                                             wind_scenarios=wind_scenarios_t,
                                             global_outage_scenarios=global_outage_scenarios)
            node.children[action.tobytes()] = child
            frontier.put((child.path_cost, id(child), child))

            # Early stopping if root has one child
            if node.parent is None and len(actions) == 1:
                return [actions[0]], 0

def a_star(node, 
           terminal_timestep, 
           demand_scenarios,
           wind_scenarios,
           global_outage_scenarios,
           heuristic_method,
           early_stopping=True,
           recalc_costs=False,
           **policy_kwargs):
    """A*"""
    if node.state.is_terminal() or node.state.episode_timestep == terminal_timestep:
        return node_mod.get_solution(node)

    frontier = []
    heapq.heappush(frontier, (0, id(node), node))

    # frontier = queue.PriorityQueue()
    # frontier.put((0, id(node), node)) # include the object id in the priority queue. prevents type error when path_costs are identical.
    while True:
        # node = frontier.get()[2]
        node = heapq.heappop(frontier)[2]
        if node.state.is_terminal() or node.state.episode_timestep == terminal_timestep:
            return node_mod.get_solution(node)
        actions = expansion.get_actions(node, **policy_kwargs)
        for action in actions:
            demand_scenarios_t = np.take(demand_scenarios, node.state.episode_timestep+1, axis=1)
            wind_scenarios_t = np.take(wind_scenarios, node.state.episode_timestep+1, axis=1)
            child = expansion.get_child_node(node, action, demand_scenarios_t, wind_scenarios_t, global_outage_scenarios, recalc_costs=recalc_costs)
            child.heuristic_cost = informed_search.heuristic(child, terminal_timestep - child.state.episode_timestep, heuristic_method)
            node.children[action.tobytes()] = child
            # frontier.put((child.path_cost + child.heuristic_cost, id(child), child))
            heapq.heappush(frontier, (child.path_cost + child.heuristic_cost, id(child), child))

            # Early stopping if root has one child
            if early_stopping and node.parent is None and len(actions) == 1:
                return [actions[0]], 0

        node.is_expanded = True

def rta_star(node,
             terminal_timestep,
             demand_scenarios,
             wind_scenarios,
             heuristic_method,
             **policy_kwargs):
    """Real time A*"""
    if node.state.is_terminal() or node.state.episode_timestep == terminal_timestep:
        return node_mod.get_solution(node)
    frontier = queue.PriorityQueue()
    frontier.put((0, id(node), node)) # include the object id in the priority queue. prevents type error when path_costs are identical.
    while True:
        assert frontier, "Failed to find a goal state"
        node = frontier.get()[2]
        if node.state.is_terminal() or node.state.episode_timestep == terminal_timestep:
            return node_mod.get_solution(node)
        actions = expansion.get_actions(node, **policy_kwargs)
        for action in actions:
            demand_scenarios_t = np.take(demand_scenarios, node.state.episode_timestep+1, axis=1)
            wind_scenarios_t = np.take(wind_scenarios, node.state.episode_timestep+1, axis=1)
            child = expansion.get_child_node(node, action, demand_scenarios, wind_scenarios)
            if child.heuristic_cost == None:
                horizon = child.state.episode_length - child.state.episode_timestep - 1 # Run heuristic to the end of the episode
                child.heuristic_cost = informed_search.heuristic(child, horizon, heuristic_method)
            node.children[action.tobytes()] = child
            frontier.put((child.path_cost + child.heuristic_cost, id(child), child))

            # Early stopping if root has one child
            if node.parent is None and len(actions) == 1:
                return [actions[0]], 0

def brute_force(env,
                terminal_timestep,
                demand_scenarios,
                wind_scenarios,
                heuristic_method,
                **kwargs):
    policy_network = kwargs.get('policy', None)
    H = terminal_timestep - env.episode_timestep 
    expansion_mode = 'vanilla' if policy_network is None else 'guided'
    root = DecisionNode(environment=env, 
        policy_network=kwargs.get('policy', None), 
        parent=DummyNode(env.num_gen),
       **kwargs)

    path, cost = find_best_path(node=root,
                                H=H,
                                demand_scenarios=demand_scenarios,
                                wind_scenarios=wind_scenarios,
                                expansion_mode=expansion_mode)
    path = path[1:]
    return path, cost

def find_best_path(node, H, demand_scenarios, wind_scenarios, expansion_mode='guided', cost_to_go=False, step_size=1):
    """
    Brute force algorithm, used for the original IEEE Smart Grid submission
    """
    if node.expected_cost is None:
        net_demands = np.take(net_demand_scenarios, node.environment.episode_timestep, axis=1)
        node.expected_cost = scenarios.calculate_expected_costs(node.environment, net_demands)

    if H == 0 or node.node_is_terminal():
        if cost_to_go is True:
            cost_to_go = -node.simulate_pl()
        else:
            cost_to_go = 0
        return [node.environment.commitment], node.expected_cost + cost_to_go

    if node.is_expanded is False:
        node.expansion_mode = expansion_mode # change the expansion mode: vanilla or guided
        node.expand_decision()

    if (node.depth_from_root() == 0) and (len(node.children) == 1):
        random_node = list(node.children.values())[0]
        random_node.expand_deterministic(step_size=step_size)
        return [node.environment.commitment, list(node.children.values())[0].action], None

    options = []
    for random_node in list(node.children.values()):
        # get to the next decision node
        random_node.expand_deterministic(step_size=step_size)
        child = random_node.children[0]
        path, cost = find_best_path(child, H-1, net_demand_scenarios, expansion_mode, cost_to_go)
        options.append((path, cost))

    path, cost = min(options, key=lambda option: option[1])
    path.insert(0, node.environment.commitment)

    return path, cost + node.expected_cost

def backup(node):
    while node.parent is not None:
        node.num_visits += 1
        node = node.parent
