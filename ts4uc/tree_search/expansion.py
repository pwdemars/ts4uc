#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import copy
import itertools

from ts4uc.tree_search import scenarios
from ts4uc.tree_search.node import Node

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
