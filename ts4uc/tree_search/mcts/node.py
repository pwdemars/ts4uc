#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 11:03:25 2019

@author: patrickdemars
"""

import torch
import numpy as np
import copy
import collections
import time
import os
import itertools
from scipy.spatial import ConvexHull, distance_matrix  

from ts4uc.tree_search.mcts.priority_list import run_PL_new, run_PL_new2, run_PL_new3

EXPANSION_SEED=2

DEFAULT_EXPLORATION_SCALING = 10.
DEFAULT_PLANNING_HORIZON_HRS = 24
DEFAULT_DECISION_BRANCHING_THRESHOLD = 0.01
DEFAULT_BRANCHING_RANDOM = 10
DEFAULT_PRUNE_THRESHOLD = 1.0
DEFAULT_DECISION_EXPANSION_MODE = 'guided'
DEFAULT_RANDOM_EXPANSION_MODE = 'reduced'
DEFAULT_SIMULATION_MODE = 'priority'

class DecisionNode(object):
    """
    DecisionNode object for tree search.

    Each node has an environment attribute that is an instance of the
   Environment class.
    Also holds the parent state; the number of visits n; the total action
    value w; the mean action value q and a prior p.
    """
    def __init__(self, environment, policy_network, state_id=0, cum_reward=0, ens=False, parent=None,
                 **kwargs):
        
        self.kwargs = kwargs
        self.parent = parent
        self.environment = environment
        self.policy_network = policy_network

        # Modes for expansion and simulation 
        self.expansion_mode = kwargs.get('decision_expansion_mode', DEFAULT_DECISION_EXPANSION_MODE)
        self.simulation_mode = kwargs.get('simulation_mode', DEFAULT_SIMULATION_MODE)                

        # Set the exploration coefficient in UCB. 
        self.global_exploration = (environment.num_gen * 
                                   kwargs.get('exploration_scale',DEFAULT_EXPLORATION_SCALING))
                
        # Branching threshold for decision node, required for guided expansion 
        self.branching_threshold = float(kwargs.get('branching_threshold', DEFAULT_DECISION_BRANCHING_THRESHOLD))

        # Threshold of lost load probability at which a node should be pruned 
        self.prune_threshold = kwargs.get('prune_threshold', DEFAULT_PRUNE_THRESHOLD)

        # Planning horizon in periods: how many steps ahead to search
        self.planning_horizon = int((kwargs.get('planning_horizon_hrs', DEFAULT_PLANNING_HORIZON_HRS) *
                                    60/self.environment.dispatch_freq_mins))

        # Step sizes determining the decision periods for MCTS. Actions are 'do nothing' in between 
        self.step_sizes = kwargs.get('step_sizes', [1]*self.planning_horizon)
        assert sum(self.step_sizes) == self.planning_horizon # Step sizes must equal the planning horizon 
        
        # General node variables
        self.node_type = 'decision'
        
        self.actions = {}
        self.state_id = state_id
        self.children = {} # Dictionary of children. Keys are actions converted to int. 
        self.child_value = {} 
        self.child_number_visits = {} 
        self.child_best_value = {}
        self.child_number_lost_load = {}    
        self.is_expanded = False
        self.expected_cost = None
        
        # The sum of rewards from the previous decision node to self
        self.cum_reward = cum_reward
        
        # Did any of the states between here and the preceding decision node have lost load? 
        self.ens = ens
        
    @property
    def number_visits(self):
        return self.parent.child_number_visits[self.state_id]
    
    @number_visits.setter
    def number_visits(self, value):            
        self.parent.child_number_visits[self.state_id] = value

    @property
    def total_value(self):
        return self.parent.child_value[self.state_id]
    
    @total_value.setter
    def total_value(self, value):
        """Sets the total value for the node. 
        
        Set by changing the value in the parent's child_value array, at the
        action_idx stored by the node.
        """
        self.parent.child_value[self.state_id] = value

    @property
    def number_lost_load(self):
        return self.parent.child_number_lost_load[self.state_id]

    @number_lost_load.setter
    def number_lost_load(self, value):            
        self.parent.child_number_lost_load[self.state_id] = value
        
    @property
    def best_value(self):
        return self.parent.child_best_value[self.state_id]
    
    @best_value.setter
    def best_value(self, value):
        self.parent.child_best_value[self.state_id] = value
        
    def child_q(self):
        """
        Get Q-values for child nodes 
        """
        return (np.array(list(self.child_value.values())) / 
                (0.01 + np.array(list(self.child_number_visits.values()))))

    def child_u(self):
        """
        Get U-values for child nodes
        """
        cnv = np.array(list(self.child_number_visits.values()))
        num = (self.global_exploration *
               np.sqrt(np.sum(cnv)))
        denom = 1 + cnv
        return num/denom
    
    def child_UCB(self):
        """
        Calculate UCB valuation of children
        """
        return self.child_q() + self.child_u()
    
    def node_is_terminal(self):
        """
        Determine whether the node is terminal. 
        
        Node is terminal if either the environment is terminal (end of episode)
        or the node is forecast_length from the root (end of forecast).
        """
        end_of_forecast = (self.periods_from_root() == self.planning_horizon)
        end_of_episode = self.environment.is_terminal()
        return end_of_episode or end_of_forecast
    
    def generate_action_random(self):
        """
        Random action selection. 
        
        1. Fix constrained generators. 
        
        2. For each unconstrained generator, choose status randomly.
        
        3. Convert and action to action_id. 
        """
        # Init action with constraints
        action = np.zeros(self.environment.num_gen, dtype=int)
        action[np.where(self.environment.must_on)[0]] = 1
        
        # Determine constrained gens
        constrained_gens = np.where(np.logical_or(self.environment.must_on, self.environment.must_off))
        unconstrained_gens = np.delete(np.arange(self.environment.num_gen), constrained_gens)

        for g in unconstrained_gens:
            action[g] = np.random.choice(np.arange(2))

        # Convert action to bit string
        action_id = ''.join(str(int(i)) for i in action)
        # Convert bit string to int
        action_id = int(action_id, 2)
        
        return action, action_id
        
    def choose_global(self):
        """
        Global action selection. 
        
        Argmax on UCB over existing children.
        
        Returns:
            - action_id (int): key for children dictionary (integer encoding)
            of binary action string.
        """        
        idx = np.argmax(self.child_UCB())
        action = list(self.children.values())[idx].action
        action_id = list(self.children.keys())[idx]
        return action, action_id

    def add_random_node(self, action, action_id):
        """
        Add a RandomNode as child for action and action_id.
        """
        node = RandomNode(action=action, action_id=action_id, parent=self, **self.kwargs)
        self.children[action_id] = node
        self.child_value[action_id] = 0
        self.child_number_visits[action_id] = 0
        self.child_number_lost_load[action_id] = 0
        self.child_best_value[action_id] = 0

    def should_prune(self):
        """
        Determine if node should be pruned.

        A number of criteria may be used to determine whether a node should be pruned. 
        """
        if isinstance(self.parent, DummyNode):
            return False # never prune a root node

        ens = self.ens
        no_children = self.is_expanded and (len(self.children) == 0)
        infeasible = self.environment.is_feasible() is False
        prune_thresh = self.number_lost_load/self.number_visits > self.prune_threshold

        if any([ens, no_children, infeasible, prune_thresh]):
            if self.depth_from_root() == 1:
                print(self.parent.action_id)
            print(ens, no_children, infeasible, prune_thresh)

        return ens or no_children or infeasible or prune_thresh
            
    def select_and_expand(self):
        """
        Select and expand routine that also includes random nodes (corresponding
        to different realisations of a (state,action) transition). 
        """
        current = self
        value = 0
        lost_load = 0 
        while True:
            if current.node_type == 'decision':
                if current.node_is_terminal():
                    return current, value, lost_load
                # elif current.should_prune():
                    # print("pruning", current.periods_from_root())
                    # current.parent.prune()
                    # return None, None
                elif current.is_expanded:
                    action, action_id = current.choose_global()
                    current = current.children[action_id]
                    assert current.node_type == 'random'
                else:
                    current.expand_decision()
                    current.is_expanded = True
                    lost_load = current.ens
                    return current, value, lost_load
            else:
                if current.is_expanded is False:
                    current.expand_random()
                    current.is_expanded = True
                child_id = current.choose_child()
                current = current.children[child_id]
                assert current.node_type == 'decision'
                value += current.cum_reward
    
    def prune(self):
        """
        Prune node, removing it from parent's children. 
        """
        if isinstance(self.parent, DummyNode):
            return # cannot prune root ndoe
        del self.parent.children[self.action_id]
        del self.parent.child_value[self.action_id]
        del self.parent.child_number_visits[self.action_id]
        del self.parent.child_best_value[self.action_id]

    def expand_decision(self):
        """
        Wrapper for the expansion methods in MCTS: vanilla or Guided Expansion. 

        The expansion mode must be set in the params or reverts to default.  
        """
        if self.expansion_mode == 'vanilla':
            self.expand_vanilla()
        elif self.expansion_mode == 'guided':
            self.expand_guided()
        else:
            raise ValueError("Invalid expansion mode. Must be either `vanilla` or `guided`")

    def expand_vanilla(self):
        """
        Expand with actions, equivalent to a naive expansion used in Vanilla MCTS.
        """
        env = self.environment

        constrained_gens = np.where(np.logical_or(env.must_on, env.must_off))
        unconstrained_gens = np.delete(np.arange(env.num_gen), constrained_gens)

        # All permutations of available generators
        all_perms = np.array(list(itertools.product(range(2), repeat=unconstrained_gens.size)))

        # Create action array 
        actions = np.zeros((all_perms.shape[0], env.num_gen))
        actions[:,constrained_gens] = env.commitment[constrained_gens]
        actions[:,unconstrained_gens] = all_perms

        for action in actions:
            action_id = ''.join(str(int(i)) for i in action)
            action_id = int(action_id, 2)
            self.add_random_node(action, action_id)
            
    def expand_guided(self):
        """
        Add up to N RandomNodes to children.

        Sample N-1 children using the policy network and always the do nothing 
        action. 
        """
        N_SAMPLES = 1000 # How many actions to generate? 
        if self.policy_network is not None:
            action_dict, log_prob = self.policy_network.generate_multiple_actions_batched(self.environment,
                                                                                  self.environment.state,
                                                                                  N_SAMPLES, self.branching_threshold)
            # Record joint log probability of children
            self.branch_log_prob = log_prob

        else:
            action_dict = {}
            action, action_id = self.generate_action_random()
            action_dict[action_id] = action
        
        # Add do nothing
        if self.environment.mode == 'test':
            action = np.where(self.environment.status > 0, 1, 0)
            action_id = ''.join(str(int(i)) for i in action)
            action_id = int(action_id, 2)
            action_dict.update({action_id: action})
        
        # Add random nodes
        for action_id in action_dict:
            self.add_random_node(action_dict[action_id], action_id)

    def simulate(self):
        """
        Wrapper function for the simulate methods

        Simulate mode should be set in params or reverts to default
        """
        if self.simulation_mode == 'priority':
            value_to_go = self.simulate_pl()
        elif self.simulation_mode == 'rollout':
            value_to_go = self.simulate_rollout()
        else:
            raise ValueError('Invalid simulation mode')

        return value_to_go

    def simulate_rollout(self):
        """
        Vanilla rollout approach: take random actions until the end of the day. 
        
        The actions are 'legalised' so that illegal actions can never be taken.

        NOTE: this is much slower than the PL method. 
        """
        NUM_ROLLOUTS = 100

        if self.node_is_terminal():
            value_to_go = 0
        else: 
            horizon = min(self.planning_horizon - self.periods_from_root(),
                          self.environment.episode_length - self.environment.episode_timestep)
            rollout_values = np.zeros(NUM_ROLLOUTS)
            for i in range(NUM_ROLLOUTS):
                rollout_env = copy.deepcopy(self.environment)
                value = 0 
                done = False
                while not done:
                    action = np.random.choice(2, size=self.environment.num_gen)
                    action = rollout_env.legalise_action(action)
                    _, reward, done = rollout_env.step(action)
                    value += reward
                rollout_values[i] = value
            value_to_go = np.mean(rollout_values)
        return value_to_go

    def simulate_pl(self):
        """"
        Simulation method using priority list. Called in the wrapper function simulate()
        """
        if self.node_is_terminal():
            value_to_go = 0
        else:
            horizon = min(self.planning_horizon - self.periods_from_root(),
                          self.environment.episode_length - self.environment.episode_timestep)
            value_to_go = -run_PL_new2(self.environment, int(horizon))
            # print("Value to go: {}".format(value_to_go/horizon))
        
        return value_to_go 
            
    def backup(self, value, lost_load):
        """
        Backup the value, which is the sum of the trajectory value (from selection
        phase) and value-to-go (from simulate phase).
        """
        current = self
        while current.parent is not None:
            current.total_value += value
            current.number_visits += 1 
            current.number_lost_load += lost_load
            if current.node_type == "decision" and current.should_prune():
                print("pruning")
                current.parent.prune() # Prune random node 
                current = current.parent.parent
            else:
                current = current.parent

    def depth_from_root(self):
        """
        Root node has depth 0, its child (a RandomNode) has depth 0.5, whose child
        (a DecisionNode) has depth 1 etc.
        """
        current = self
        depth = 0 
        while current.parent.parent is not None: 
            depth += 1
            current = current.parent
        depth = float(depth)/2 # RandomNodes have 1/2 integer depth
        return depth
    
    def periods_from_root(self):
        """
        Determine the number of settlement periods of self from the root. This
        may differ from self.depth_from_root() if the step_sizes are non-uniform.
        """
        depth = int(self.depth_from_root())
        periods = sum(self.step_sizes[:depth])
        return periods

    def sample_arma_convex_hull(self, N_samples, target_selected):
        """
        This function is useful for sampling a diverse and extreme set of errors 
        from the ARMA processes. We sample N times from each ARMA, then calculate the convex
        hull of the set. We then reduce the number of points to target_selected
        using the remove_one_point function (maximising the shortest nearest neighbour distance). 

        The function returns two (N_samples, 2) dimensional arrays, giving selected xs and zs for 
        demand and wind processes.
        """

        def remove_one_point(matrix, idx):
            """
            Remove a point while maximising the shortest nearest neighbour distance (needs proof)
            """
            dist_matrix = distance_matrix(matrix, matrix)
            dist_matrix[dist_matrix == 0] = np.inf # set diagonal to inf
            dist_matrix[np.tril_indices(dist_matrix.shape[0], -1)] = np.nan # set lower left half of matrix to nan 
            x, y = np.where(dist_matrix == np.nanmin(dist_matrix)) # closest pair indexes 
            
            x_dists = np.append(dist_matrix[:,x].reshape(1,-1), dist_matrix[x,:])
            y_dists = np.append(dist_matrix[:,y].reshape(1,-1), dist_matrix[y,:])
            
            x_dists = x_dists[~np.isnan(x_dists)]
            y_dists = y_dists[~np.isnan(y_dists)]

            x_dists = np.sort(x_dists)
            y_dists = np.sort(y_dists)

            if np.min(y_dists[1]) < np.min(x_dists[1]):
                to_remove = y
            else:
                to_remove = x
            
            matrix = np.delete(matrix, to_remove, axis=0)
            idx = np.delete(idx, to_remove)
            
            return matrix, idx

        xs = np.zeros(shape=(N_samples,2)) # for forecast error samples (xs)
        zs = np.zeros(shape=(N_samples,2)) # white noise samples

        for i in range(N_samples):
            xs[i,0], zs[i,0] = self.environment.arma_demand.sample_error()
            xs[i,1], zs[i,1] = self.environment.arma_wind.sample_error()

        # Find the convex hull and retrieve the vertices. 
        hull = ConvexHull(xs)
        selected_xs = xs[hull.vertices] 
        idx = np.arange(selected_xs.shape[0]) # index of selected errors from the convex hull. starts off complete.

        # Remove points from the selected errors while maximising the shortest nearest neighbour distance (needs proof)
        while len(idx) > target_selected: 
            selected_xs, idx = remove_one_point(selected_xs, idx)

        # Finally filter the zs to include only the convex hull 
        selected_zs = zs[hull.vertices[idx]]

        return selected_xs, selected_zs
    

class RandomNode(object):
    """
    A RandomNode refers to an action. Its children are DecisionNodes, 
    corresponding to realisations of the state-action transition. 
    """
    def __init__(self, action, action_id, parent, **kwargs):
        self.kwargs = kwargs
        self.parent = parent
        self.action = action 
        self.action_id = action_id 
        self.children = {}
        self.child_number_visits = {}
        self.child_number_lost_load = {}
        self.child_value = {}
        
        self.step_sizes = self.parent.step_sizes

        self.node_type = 'random'

        self.expansion_mode = kwargs.get('random_expansion_mode', DEFAULT_RANDOM_EXPANSION_MODE)

        self.is_expanded = False

        self.ens = False

        self.max_branching = int(kwargs.get('random_branching_factor', DEFAULT_BRANCHING_RANDOM))
    
    @property
    def number_visits(self):
        return self.parent.child_number_visits[self.action_id]
    
    @number_visits.setter
    def number_visits(self, value):            
        self.parent.child_number_visits[self.action_id] = value

    @property
    def total_value(self):
        return self.parent.child_value[self.action_id]
    
    @total_value.setter
    def total_value(self, value):
        self.parent.child_value[self.action_id] = value

    @property
    def number_lost_load(self):
        return self.parent.child_number_lost_load[self.action_id]

    @number_lost_load.setter
    def number_lost_load(self, value):
        self.parent.child_number_lost_load[self.action_id] = value

    def choose_child(self):
        """
        Choose the least visited child, returning its key (state_id)
        """
        return min(self.child_number_visits, key=self.child_number_visits.get)
    
    def add_decision_node(self, env, key, cum_reward=None, ens=False):
        """
        Add a decision node corresponding to the state in env, with name key.
        """
        new_node = DecisionNode(parent = self,
                        state_id=key,
                        environment = env,
                        policy_network = self.parent.policy_network,
                        cum_reward=cum_reward,
                        ens=ens,
                        **self.parent.kwargs)
        self.children[key] = new_node
        self.child_number_visits[key] = 0
        self.child_number_lost_load[key] = 0
        self.child_value[key] = 0

        # if new_node.depth_from_root() == 1:
        #     new_node.environment.arma_demand.xs = new_node.

        # For decision nodes with depth 1, need to hot start the ARMAs based on the episode timestep.
        # Used for day-ahead problem only.
        if new_node.depth_from_root() == 1:
            for i in range(new_node.environment.episode_timestep):
                new_node.environment.arma_demand.step()
                new_node.environment.arma_wind.step()

                # np.random.seed(self.kwargs.get('test_seed'))
                # N = 200
                # selected_xs, selected_zs = self.parent.sample_arma_convex_hull(N, self.max_branching)
                # selected_xs[]

    def expand_random(self):
        """
        Wrapper function that calls one of the expansion methods for random nodes
        """
        # if self.expansion_mode == "reduced" and self.parent.depth_from_root() == 0:
            # self.expand_reduced()
        # else:
            # self.expand_naive()

        if self.expansion_mode == 'naive':
            self.expand_naive()
        elif self.expansion_mode == 'reduced':
            self.expand_reduced()
        else:
            ValueError("Invalid random expansion mode")

    def expand_reduced(self):
        """
        Expand a random node with a reduced subset of actions, chosen by sampling forecast
        errors from the environment. 
        """
        np.random.seed(self.kwargs.get('test_seed'))
        N = 200
        selected_xs, selected_zs = self.parent.sample_arma_convex_hull(N, self.max_branching)

        depth = int(self.parent.depth_from_root())
        step_size = self.step_sizes[depth]
        for i, (xs, zs) in enumerate(zip(selected_xs, selected_zs)):
            new_env = copy.deepcopy(self.parent.environment)
            cum_reward = 0
            ens = False
            for j in range(step_size):
                # TODO: Seed ARMA processes for first generation decision nodes...? 
                if j == 0:
                    obs, reward, done = new_env.step(self.action, errors={'demand': (xs[0], zs[0]),
                                                                          'wind': (xs[1], zs[1])})
                else:
                    obs, reward, done = new_env.step(self.action)

                if self.kwargs.get('reward_version') == 1:
                    reward = reward
                elif self.kwargs.get('reward_version') == 2:
                    reward = reward - new_env.ens_cost
                elif self.kwargs.get('reward_version') == 3:
                    reward = reward - new_env.ens_cost/2
                else:
                    raise ValueError('must pass correct reward version')

                reward = reward/new_env.net_demand
                reward = reward * (new_env.forecast - new_env.wind_forecast)
                cum_reward += reward
                if new_env.ens: 
                    # print("yes------------------ENS: {}, REWARD: {}, DEMAND: {}, COMMITMENT: {}".format(new_env.ens_cost, reward, new_env.net_demand, new_env.commitment))
                    ens = True
                if new_env.episode_timestep == new_env.episode_length-1:
                    break
            self.add_decision_node(new_env, i, cum_reward=cum_reward, ens=ens)

    def expand_naive(self):
        """
        Expand with a list of step_sizes, determining how many steps to take at 
        each random node depth. 
        
        In this case we don't add the forecast
        """
        depth = int(self.parent.depth_from_root())
        step_size = self.step_sizes[depth]
        np.random.seed(seed=self.kwargs.get('test_seed') + self.parent.state_id)
        for i in range(self.max_branching):
            new_env = copy.deepcopy(self.parent.environment)
            cum_reward = 0
            ens = False
            for j in range(step_size):
                obs, reward, done = new_env.step(self.action)

                if self.kwargs.get('reward_version') == 1:
                    reward = reward
                elif self.kwargs.get('reward_version') == 2:
                    reward = reward - new_env.ens_cost
                elif self.kwargs.get('reward_version') == 3:
                    reward = reward - new_env.ens_cost/2
                else:
                    raise ValueError('must pass correct reward version')

                reward = reward/new_env.net_demand
                reward = reward * (new_env.forecast - new_env.wind_forecast)
                cum_reward += reward
                if new_env.ens: 
                    # print("yes------------------ENS: {}, REWARD: {}, DEMAND: {}, COMMITMENT: {}".format(new_env.ens_cost, reward, new_env.net_demand, new_env.commitment))
                    ens = True
                if new_env.episode_timestep == new_env.episode_length-1:
                    break 
            self.add_decision_node(new_env, i, cum_reward=cum_reward, ens=ens)
        np.random.seed(self.kwargs.get('test_seed')) # revert random seed
    
    def expand_deterministic(self, step_size=1):
        """
        Function for adding a single decision node that carries the determinsitic realisation
        of the state-action transition. 

        This is useful when random nodes are not needed.
        """
        new_env = copy.deepcopy(self.parent.environment)
        ens=False
        for i in range(step_size):
            cum_reward = 0
            obs, reward, done = new_env.step(self.action, deterministic=True)
            cum_reward += reward            
            if new_env.episode_timestep == new_env.episode_length-1:
                break
            if new_env.ens: 
                ens=True
        self.add_decision_node(new_env, 0, cum_reward=cum_reward, ens=ens)


    def prune(self):
        """
        Prune node, removing it from parent's children. 
        """
        del self.parent.children[self.action_id]
        del self.parent.child_value[self.action_id]
        del self.parent.child_number_visits[self.action_id]
        del self.parent.child_best_value[self.action_id]
                    

class DummyNode(object):
    """
    Functions as parent of root node. 
    """
    def __init__(self, num_gen):
        self.parent = None
        self.action = None    
        self.child_value = collections.defaultdict(float)
        self.child_number_visits = collections.defaultdict(float)
        self.child_number_lost_load = collections.defaultdict(float)
        self.child_best_value = collections.defaultdict(float)

