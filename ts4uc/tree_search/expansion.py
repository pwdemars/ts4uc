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
    return actions

def get_all_actions(env):
    """Get all actions from the `env` state"""
    constrained_gens = np.where(np.logical_or(env.must_on, env.must_off))
    unconstrained_gens = np.delete(np.arange(env.action_size), constrained_gens)

    # All permutations of available generators
    all_perms = np.array(list(itertools.product(range(2), repeat=unconstrained_gens.size)))

    # Create action array 
    actions = np.zeros((all_perms.shape[0], env.action_size))
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
    if env.curtailment:
        action = np.append(action, 0)
    action_id = ''.join(str(int(i)) for i in action)
    action_id = int(action_id, 2)
    action_dict.update({action_id: action})

    actions = np.array(list(action_dict.values()))

    return actions

def get_child_node(node, action, demand_scenarios=None, wind_scenarios=None,
                   global_outage_scenarios=None, deterministic=True, recalc_costs=False):
    """
    Return a child node corresponding to taking `action` from the state 
    corresponding to `node`.
    
    The child node has `node` as its parent.
    """
    if action.tobytes() in node.children:
        child = node.children[action.tobytes()]
        if recalc_costs == True: # If rolling horizon, then always recalculate costs
            child.step_cost = scenarios.calculate_expected_costs(child.state, action, demand_scenarios, wind_scenarios) # FIXME: account for availability
        child.path_cost = node.path_cost + child.step_cost
        return node.children[action.tobytes()]

    new_env = copy.deepcopy(node.state)
    _, reward, _ = new_env.step(action, deterministic=deterministic)

    # If modelling outages, sample possible generator availabilities for this node
    if new_env.outages:
        availability_scenarios = scenarios.sample_availability_scenarios(global_outage_scenarios, node.availability_scenarios, node.state.status, action) # sample outage scenarios (using OLD env)
        # outage_scenarios = scenarios.sample_outage_scenarios(global_outage_scenarios, node.state.status) 
        # availability_scenarios = np.clip(node.availability_scenarios - outage_scenarios, 0, 1)
        # availability_scenarios = scenarios.sample_availability_single(new_env, action, node.availability_scenarios)
    else:
        availability_scenarios = None

    if demand_scenarios is None:
        cost = -reward
    else:
        cost = scenarios.calculate_expected_costs(new_env, action, demand_scenarios, wind_scenarios, availability_scenarios)

    child = Node(env=new_env,
                parent=node,
                action=action,
                step_cost=cost, 
                path_cost=node.path_cost + cost)

    child.availability_scenarios = availability_scenarios
    
    return child
