#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from rl4uc import dispatch

import numpy as np

def check_lost_load(state, horizon):
        """Check if forecast demand can be met for the next H periods from state"""
        horizon = min(horizon, max(state.gen_info.t_min_down.max(), abs(state.gen_info.t_min_down.min())))
        for t in range(1,horizon+1):
                net_demand = state.episode_forecast[state.episode_timestep+t] - state.episode_wind_forecast[state.episode_timestep+t] # Nominal demand for t+1th period ahead
                future_status = state.status + (t)*np.where(state.status >0, 1, -1) # Assume all generators are kept on where possible

                available_generators = np.logical_or((-future_status >= state.t_min_down), state.commitment) # Determines the availability of generators as binary array

                available_cap = np.dot(available_generators, state.max_output)

                if available_cap < net_demand:
                        return np.inf
        return 0

def simple_priority_list(state, horizon): 
        """
        A simple priority list algorithm for estimating operating costs.

        This does not obey any operating constraints and only approximates the economic dispatch.

        TODO: add lost load check
        """
        time_interval = state.dispatch_freq_mins/60

        # Sort to priority list order 
        gen_info = state.gen_info.sort_values('min_fuel_cost')

        # Net demand forecast
        demand = state.episode_forecast[state.episode_timestep+1:state.episode_timestep+horizon+1]
        wind = state.episode_wind_forecast[state.episode_timestep+1:state.episode_timestep+horizon+1]
        net_demand = demand - wind
        # Get the cumulative sum of outputs, in PL order
        maxs = gen_info.max_output.values
        cumsum = np.cumsum(maxs)
        cumsum = np.insert(0,0,cumsum)
        # Initialise the commitment schedule
        dispatch = np.zeros((demand.size, maxs.size))
        # Calculate the economic dispatch
        for i, d in enumerate(net_demand):
                m = np.where(cumsum > d)[0][0] # marginal generator
                marginal_mwh = d - cumsum[m-1] # output of marginal generator
                dispatch[i, :m+1] = maxs[:m+1] # full output for non-marginal generators
                dispatch[i, m] = marginal_mwh # marginal_mwh for the marginal generator

        # Calculate operating costs
        costs = np.dot(dispatch, gen_info.min_fuel_cost).sum() * time_interval

        return costs

def simple_priority_list_ED(state, horizon):
    """Simple priority list with economic dispatch"""

    time_interval = state.dispatch_freq_mins/60

    # Sort to priority list order 
    gen_info = state.gen_info.sort_values('min_fuel_cost')

    # Net demand forecast
    demand = state.episode_forecast[state.episode_timestep+1:state.episode_timestep+horizon+1]
    wind = state.episode_wind_forecast[state.episode_timestep+1:state.episode_timestep+horizon+1]
    net_demand = demand - wind

    # Get the cumulative sum of outputs, in PL order
    maxs = gen_info.max_output.values
    cumsum = np.cumsum(maxs)
    cumsum = np.insert(0,0,cumsum)

    uc_schedule = np.zeros((len(net_demand), len(maxs)))
    for t, d in enumerate(net_demand):
            m = np.where(cumsum > d)[0][0] # marginal generator
            uc_schedule[t,:m+1] = 1
            
    cost = economic_fuel_cost(state, uc_schedule.T, net_demand, gen_info, time_interval)
    
    return cost

def constrained_commitment(gen_info_sorted, state, net_demand):
        """Determine a UC schedule that satisfies initial generator constraints"""
        # Initialise the schedule
        uc_schedule = np.zeros([state.num_gen, net_demand.size])
        
        # Fix constrained gens
        for g in range(state.num_gen):
                if state.status[g] < 0:
                        n = max(gen_info_sorted.t_min_down.values[g] + state.status[g], 0)
                        uc_schedule[g,:n] = -1
                else:
                        n = max(gen_info_sorted.t_min_up.values[g] - state.status[g], 0)
                        uc_schedule[g,:n] = 1
                        
        on_schedule = np.where(uc_schedule==-1, 0, uc_schedule)
        committed_cap = np.dot(on_schedule.T, gen_info_sorted.max_output.values) 
        residual_load = net_demand - committed_cap

        for t in range(net_demand.size):
                res = residual_load[t]
                free_gens = np.where(uc_schedule[:,t] == 0)[0]
                for g in free_gens:
                        if res <= 0:
                                break
                        # Commit the next generator
                        uc_schedule[g,t] = 1
                        res -= gen_info_sorted.max_output.values[g]

        uc_schedule[uc_schedule == -1 ] =  0 

        return uc_schedule

def weighted_fuel_cost(uc_schedule, net_demand, gen_info_sorted, time_interval):
        """Calculate the average fuel cost per period as a weighted average of min fuel costs of online generators."""
        uc_schedule = uc_schedule.T
        weighted_avg_fc = np.zeros(uc_schedule.shape[0])
        
        for t in range(uc_schedule.shape[0]):
                online_gens = np.where(uc_schedule[t])[0]
                weighted_avg_fc[t] = (np.dot(gen_info_sorted.min_fuel_cost.values[online_gens], 
                                                                  gen_info_sorted.min_output.values[online_gens])/
                                                          np.sum(gen_info_sorted.min_output.values[online_gens]))
                
        fuel_cost = np.dot(weighted_avg_fc, net_demand) * time_interval

        return fuel_cost

def economic_fuel_cost(state, uc_schedule, net_demand, gen_info_sorted, time_interval):
        fc = 0 
        # TODO: Use rl4uc.environment to solve the dispatch (for consistency)
        for commitment, nd in zip(uc_schedule.T, net_demand):
                c, d = state.calculate_fuel_cost_and_dispatch(nd, commitment)
                # disp = economic_dispatch(commitment, nd, gen_info_sorted)
                # c = (np.multiply(np.square(disp), gen_info_sorted.a.values) + 
                #                          np.multiply(disp, gen_info_sorted.b.values) + 
                #                          gen_info_sorted.c.values) * time_interval
                # c = np.sum(commitment * c)
                fc += np.sum(c) 
        return fc

def economic_dispatch(commitment, net_demand, gen_info_sorted, lambda_low=0., lambda_high=100., epsilon=1.):

        disp = np.zeros(commitment.size)
        idx = np.where(commitment==1)[0]

        on_max = gen_info_sorted.max_output.values[idx]
        on_min = gen_info_sorted.min_output.values[idx]
        on_a = gen_info_sorted.a.values[idx]
        on_b = gen_info_sorted.b.values[idx]

        if np.sum(on_max) < net_demand:
                econ = on_max
        elif np.sum(on_min) > net_demand:
                econ = on_min
        else:
                econ = dispatch.lambda_iteration(net_demand, 
                                                                                  lambda_low, 
                                                                                  lambda_high, 
                                                                                  on_a,
                                                                                  on_b,
                                                                                  on_min, 
                                                                                  on_max,
                                                                                  epsilon)
        disp[idx] = econ
        return disp

def start_cost(uc_schedule, gen_info_sorted, state):
        """Calculate start costs for the UC schedule"""
        extended_schedule = np.hstack((np.where(state.status > 0, 1, 0).reshape(-1,1),
                                                                   uc_schedule))
        starts = np.sum(np.diff(extended_schedule) == 1, axis=1)
        start_cost = np.dot(gen_info_sorted.hot_cost.values, starts)
        return start_cost 

def lost_load_cost(uc_schedule, gen_info_sorted, net_demand, voll, time_interval):
        committed = np.dot(uc_schedule.T, gen_info_sorted.max_output.values)
        ens = np.maximum(net_demand - committed, 0)
        llc = np.sum(ens * voll * time_interval) # lost load cost
        return llc

def advanced_priority_list(state, horizon, fuel_cost_method='economic', start_costs=False, lost_load_costs=False):
        """
        Priority list cost estimation for environment for subsequent periods.
        
        In this version, generator constraints must be obeyed
        """
        # Sort by min fuel cost (priority list order)
        gen_info_sorted = state.gen_info.sort_values('min_fuel_cost')
        time_interval = state.dispatch_freq_mins/60

        # Net demand forecast
        demand = state.episode_forecast[state.episode_timestep+1:state.episode_timestep+horizon+1]
        wind = state.episode_wind_forecast[state.episode_timestep+1:state.episode_timestep+horizon+1]
        net_demand = demand - wind

        uc_schedule = constrained_commitment(gen_info_sorted, state, net_demand)

        if fuel_cost_method=='weighted':
                fc = weighted_fuel_cost(uc_schedule, net_demand, gen_info_sorted, time_interval)
        elif fuel_cost_method=='economic':
                fc = economic_fuel_cost(state, uc_schedule, net_demand, gen_info_sorted, time_interval)
        else:
                raise ValueError('{} is not a valid fuel cost calculation method'.format(fuel_cost_method))

        if start_costs:
                sc = start_cost(uc_schedule, gen_info_sorted, state)
        else: 
                sc = 0

        if lost_load_costs:
                llc=lost_load_cost(uc_schedule, gen_info_sorted, net_demand, state.voll, time_interval)
        else: 
                llc = 0

        return fc + sc + llc

def pl_plus_ll(state, horizon):
        if check_lost_load(state, horizon) == np.inf:
                return np.inf
        else:
                return simple_priority_list(state, horizon)

def heuristic(node, horizon, method=None):
        """Simple heuristic that givees np.inf if a node's state is infeasible, else 0"""
        if not method:
                return 0
        elif horizon == 0:
                return 0
        elif method=='check_lost_load':
                heuristic_cost = check_lost_load(node.state, horizon)
        elif method=='simple_priority_list':
                heuristic_cost = simple_priority_list(node.state, horizon)
        elif method=='simple_priority_list_ED':
                heuristic_cost = simple_priority_list_ED(node.state, horizon)
        elif method=='pl_plus_ll':
                heuristic_cost = pl_plus_ll(node.state, horizon)
        elif method=='advanced_priority_list':
                heuristic_cost = advanced_priority_list(node.state, horizon)
        else:
                raise ValueError('{} is not a valid heuristic'.format(method))
        return heuristic_cost
