#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from ts4uc.tree_search import scenarios
from rl4uc.environment import make_env
from ts4uc.agents.ac_agent import ACAgent
from ts4uc.tree_search.mcts.node import DecisionNode, DummyNode
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
    def __init__(self, env, parent, action, step_cost, path_cost):
        self.state = env
        self.parent = parent
        self.action = action
        self.path_cost = path_cost
        self.is_expanded = False
        self.step_cost = step_cost
        self.children = {}

def get_actions(node, policy, **policy_kwargs):
    """Wrapper function for get actions with policy (guided search) or without (unguided)"""
    env = node.state
    if node.is_expanded: 
        actions = [child.action for child in list(node.children.values())]
    elif policy != None:
        actions = get_actions_with_policy(env, policy, **policy_kwargs)
    else:
        actions = get_all_actions(env)
    node.is_expanded = True
    return actions

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

def get_actions_with_policy(env, policy, **policy_kwargs):
    """
    Use a policy to get a state of actions for the `env` state.

    This is used to implement `guided expansion`.
    """
    branching_threshold = policy_kwargs.get('branching_threshold', 0.05)
    num_samples = policy_kwargs.get('num_samples', 1000)

    action_dict, log_prob = policy.generate_multiple_actions_batched(env, env.state, num_samples, branching_threshold)
    
    # Add the do nothing action
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
    if action.tobytes() in node.children:
        child = node.children[action.tobytes()]
        child.path_cost = node.path_cost + child.step_cost
        return node.children[action.tobytes()]

    new_env = copy.deepcopy(node.state)
    _, reward, _ = new_env.step(action, deterministic=deterministic)

    if net_demand_scenarios is None:
        cost = -reward
    else:
        cost = scenarios.calculate_expected_costs(new_env, net_demand_scenarios)

    # TODO: add step cost
    child = Node(env=new_env,
                parent=node,
                action=action,
                step_cost=cost, 
                path_cost=node.path_cost + cost)
    
    return child

def get_solution(node):
    """Return the solution path (list of actions) leading to node."""
    s = []
    path_cost = node.path_cost
    while node.parent is not None:
        s.insert(0, node.action)
        node = node.parent
    return s, path_cost

def uniform_cost_search(node, 
                        terminal_timestep, 
                        net_demand_scenarios,
                        **policy_kwargs):
    """Uniform cost search"""
    if node.state.is_terminal() or node.state.episode_timestep == terminal_timestep:
        return get_solution(node)
    frontier = queue.PriorityQueue()
    frontier.put((0, id(node), node)) # include the object id in the priority queue. prevents type error when path_costs are identical.
    while True:
        assert frontier, "Failed to find a goal state"
        node = frontier.get()[2]
        if node.state.is_terminal() or node.state.episode_timestep == terminal_timestep:
            return get_solution(node)
        actions = get_actions(node, **policy_kwargs)
        # Early stopping if root node has only one child.
        if node.parent is None and len(actions)==1:
            return [actions[0]], 0
        for action in actions:
            net_demand_scenarios_t = np.take(net_demand_scenarios, node.state.episode_timestep+1, axis=1)
            child = get_child_node(node, action, net_demand_scenarios_t)
            node.children[action.tobytes()] = child
            frontier.put((child.path_cost, id(child), child))

def a_star(node, 
           terminal_timestep, 
           net_demand_scenarios,
           **policy_kwargs):
    """A*"""
    if node.state.is_terminal() or node.state.episode_timestep == terminal_timestep:
        return get_solution(node)
    frontier = queue.PriorityQueue()
    frontier.put((0, id(node), node)) # include the object id in the priority queue. prevents type error when path_costs are identical.
    while True:
        assert frontier, "Failed to find a goal state"
        node = frontier.get()[2]
        if node.state.is_terminal() or node.state.episode_timestep == terminal_timestep:
            return get_solution(node)
        actions = get_actions(node.state, **policy_kwargs)
        # Early stopping if root node has only one child.
        if node.parent is None and len(actions)==1:
            return [actions[0]], 0
        for action in actions:
            net_demand_scenarios_t = np.take(net_demand_scenarios, node.state.episode_timestep+1, axis=1)
            child = get_child_node(node, action, net_demand_scenarios_t)
            heuristic_cost = informed_search.heuristic(child, terminal_timestep - child.state.episode_timestep)
            frontier.put((child.path_cost + heuristic_cost, id(child), child))

def rta_star(node,
             terminal_timestep,
             net_demand_scenarios,
             **policy_kwargs):
    """Real time A*"""
    if node.state.is_terminal() or node.state.episode_timestep == terminal_timestep:
        return get_solution(node)
    frontier = queue.PriorityQueue()
    frontier.put((0, id(node), node)) # include the object id in the priority queue. prevents type error when path_costs are identical.
    while True:
        assert frontier, "Failed to find a goal state"
        node = frontier.get()[2]
        if node.state.is_terminal() or node.state.episode_timestep == terminal_timestep:
            return get_solution(node)
        actions = get_actions(node.state, **policy_kwargs)
        # Early stopping if root node has only one child.
        if node.parent is None and len(actions)==1:
            return [actions[0]], 0
        for action in actions:
            net_demand_scenarios_t = np.take(net_demand_scenarios, node.state.episode_timestep+1, axis=1)
            child = get_child_node(node, action, net_demand_scenarios_t)
            if child.heuristic_cost is None:
                horizon = child.state.episode_length - child.state.episode_timestep - 1
                child.heuristic_cost = informed_search.heuristic(child, horizon)
            frontier.put((child.path_cost + child.heuristic_cost, id(child), child))

def brute_force(env,
                terminal_timestep,
                net_demand_scenarios,
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
                                net_demand_scenarios=net_demand_scenarios,
                                expansion_mode=expansion_mode)
    path = path[1:]
    return path, cost

def find_best_path(node, H, net_demand_scenarios, expansion_mode='guided', cost_to_go=False, step_size=1):
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