#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 30 11:10:29 2019

@author: patrickdemars
"""

import numpy as np
import rl4uc.dispatch as dispatch_module

def get_pl(gen_info):  
    """Create a priority list.
    The priority list is created according to the output cost
    in £/MWh when at max output. 
    """ 
    num_gen = len(gen_info)
    a = gen_info['a'].to_numpy()
    b = gen_info['b'].to_numpy()
    c = gen_info['c'].to_numpy()
    outputs = np.array(gen_info['max_output'][:num_gen])
    pl = (a*(outputs**2) + b*outputs + c)/outputs
    return pl

def increment_status(rle):
    """
    Add ON status to Run Length Encoding (RLE)

    First check whether last status is ON. 
    If yes, then increment the length of the last item in the RLE. 
    Otherwise, add [1,1] to the RLE. 

    Args:
        - rle: run length encoding object (list of lists) # TODO: can this be list of tuples?

    Returns:
        - rle: updated rle

    # TODO: can this also return the cold/hot starts?
    """
    if rle[-1][0] == 1:
        rle[-1][1] += 1
    else:
        rle.append([1,1])
    return rle

def decrement_status(rle):
    """
    Add OFF status to RLE. 

    Check whether last status is OFF.
    If yes, increment the length of the last item in the RLE.
    Otherwise, add [0,1] to the RLE. 

    Args:
        - rle: run length encoding object

    Returns:
        - rle: updated RLE
    """
    if rle[-1][0] == 0:
        rle[-1][1] += 1
    else:
        rle.append([0,1])
    return rle

def convert_rle(rle, num_gen):
    """
    Convert an RLE commitment schedule to binary commitment schedule. 
    """
    status = [[] for i in range(num_gen)]
    for i, r in enumerate(rle):
        for item in r:
            status[i].append([item[0]]*item[1])
        status[i] =  sum(status[i], []) 
    return status

def pl_senjyu(env, lookahead):
    
    # Get demand profile
    demand = env.all_forecast[env.episode_timestep+1: env.episode_timestep+1+lookahead]
    # Sort gen_info by heat rate
    gen_info = env.gen_info
    gen_info['status'] = env.status
    gen_info_sorted = gen_info.sort_values(by='min_fuel_cost')
    num_gen = len(gen_info_sorted)
    
    # Create current run length encoding for current status
    rle = [[[int(gen_info_sorted['status'][i] > 0), 
                abs(gen_info_sorted['status'][i])]] for i in range(num_gen)]    
    
    # Generate optimistic RLE for demand profile (no constraints)
    for t, d in enumerate(demand):
        cap = 0
        j = 0
        for j in range(num_gen):
            if cap < d:
                cap += gen_info_sorted['max_output'][j]
                increment_status(rle[j])
            else:
                decrement_status(rle[j])
                
    # Satisfy min up/down time constraints
    # TODO: check this works when there are consecutive min up/down time violations
    for i, g in enumerate(rle):
        for j, item in enumerate(g[:-1]):
            if item[1] < gen_info_sorted['t_min_down'][i]: # if min down time is not satisfied --> change status
                if item[0] == 0:
                    item[0] = 1
                else:
                    item[0] = 0
    
    # Remove initial status to convert RLE to schedule
    for i in range(num_gen):
        rle[i][0][1] += -abs(gen_info_sorted['status'][i])
        
    # Convert RLE to schedule:
    status = convert_rle(rle, num_gen)

    return np.array(status)
            
def calculate_outputs(lm, a, b, mins, maxs, num_gen):
    """Calculate outputs for all generators as a function of lambda.
    lm: lambda
    a, b: coefficients for quadratic curves of the form cost = a^2p + bp + c
    num_gen: number of generators
    
    Returns a list of individual generator outputs. 
    """
    outputs = np.zeros(num_gen)
    i = 0
    while i < num_gen:
        p = (lm - b[i])/a[i]
        if p < mins[i]:
            p = mins[i]
        elif p > maxs[i]:
            p = maxs[i]
        outputs[i] = p
        i += 1
    return outputs 


def lambda_iteration(load, lambda_low, lambda_high, num_gen, a, b, mins, maxs, epsilon):
    """Calculate economic dispatch using lambda iteration. 
    
    lambda_low, lambda_high: initial lower and upper values for lambda
    a: coefficients for quadratic load curves
    b: constants for quadratic load curves
    epsilon: error as a function 
    
    Returns a list of outputs for the generators.
    """
    num_gen = len(a)
    lambda_low = np.float(lambda_low)
    lambda_high = np.float(lambda_high)
    lambda_mid = 0
    total_output = sum(calculate_outputs(lambda_high, a, b, mins, maxs, num_gen))
    while abs(total_output - load) > epsilon:
        lambda_mid = (lambda_high + lambda_low)/2
        total_output = sum(calculate_outputs(lambda_mid, a, b, mins, maxs, num_gen))
        if total_output - load > 0:
            lambda_high = lambda_mid
        else:
            lambda_low = lambda_mid
    return calculate_outputs(lambda_mid, a, b, mins, maxs, num_gen)

def economic_dispatch(env, action, demand, lambda_lo, lambda_hi):
    """Calcuate economic dispatch using lambda iteration.
    
    Args:
        action (numpy array): on/off commands
        lambda_lo (float): initial lower lambda value
        lambda_hi (float): initial upper lambda value
        
    """
    
    idx = np.where(np.array(action) == 1)[0]
    on_a = np.array(env.a[idx])
    on_b = np.array(env.b[idx])
    on_min = np.array(env.min_output[idx])
    on_max = np.array(env.max_output[idx])        
    disp = np.zeros(env.num_gen)
    ens = 0
    if np.sum(on_max) < demand:
        econ = on_max
        ens = demand - np.sum(on_max)
    elif np.sum(on_min) > demand:
        econ = on_min
        ens = np.sum(on_min) - demand
    else:
        econ = dispatch_module.lambda_iteration(demand, lambda_lo,
                                         lambda_hi, on_a, on_b,
                                         on_min, on_max, 0.1)
    for (i, e) in zip(idx, econ):
        disp[i] = e
    return disp, ens

def calculate_fuel_costs(env, outputs, action):
    """Calculate fuel costs.
    
    Outputs an array of production costs for each unit. 
    """
    a = env.a
    b = env.b
    c = env.c    
    costs_all = a*(outputs**2) + b*outputs + c
    fuel_costs = np.multiply(action, costs_all)
    return fuel_costs

def calculate_ens_costs(ens):
    """
    Calculate ENS costs for a vector of ENS. 
    
    Positive is generation deficit, negative is generation surplus. 
    Generation deficit is penalised at a different cost to surplus.
    """
    cost = []
    deficit_cost = 40
    surplus_cost = 40
    for v in ens:
        if v >= 0:
            cost.append(deficit_cost*v)
        else:
            cost.append(surplus_cost*v)
    return cost
        
def run_PL(node):
    """
    Run priority list simulation to the end of the day. 
    
    Steps: 
        1. Get priority list solution (this is approximate and is NOT a feasible
        solution.)
        2. Get economic dispatch for this, including ENS. 
        3. Calculate fuel costs and ENS costs: 
            - ENS costs are penalised at prices of £25/MWh for deficit and 
            £10/MWh for surplus, to account for the fact that the PL solution 
            is not optimal (or usually feasible).
    """
    env = node.environment
    gen_info = env.generator_mix
    len_day = 24 - env.episode_timestep
    uc = pl_senjyu(env, 
                   env.all_forecast[env.episode_timestep+1:],
                   node.environment.min_fuel_cost)
    uc_h = np.array(uc).transpose()
    dispatch = []
    ens = []
    for i, h in enumerate(uc_h):
        d, e = get_dispatch(h, env.all_forecast[i + 1 + env.episode_timestep], 20, gen_info)
        dispatch.append(d)
        ens.append(e)
    fuel_costs = [sum(calculate_fuel_costs(dispatch[i], gen_info)) for i in range(len_day)]
    ens_costs = calculate_ens_costs(ens)
    total_costs = [fuel_costs[i] + ens_costs[i] for i in range(len_day)]
    return sum(total_costs)
        
def run_PL_new(env, lookahead):
    """
    New version of priority list that determines feasibility of the current status
    before running the PL.
    
    This PL does not consider start costs. 
    """
    demand = env.episode_forecast[env.episode_timestep+1:env.episode_timestep+lookahead+1]
    wind = env.episode_wind_forecast[env.episode_timestep+1:env.episode_timestep+lookahead+1]
    net_demand = demand-wind # net demand is demand-wind
    demand=net_demand
    
    if env.is_feasible() is False:
        return -env.min_reward * demand.size
    
    gen_info_sorted = env.gen_info.sort_values(by='min_fuel_cost')
    caps = np.cumsum(gen_info_sorted.max_output.values)[:-1]
    
    uc_schedule = np.array(np.array([demand < c for c in caps]))
    uc_schedule = np.where(uc_schedule, 0, 1)
    uc_schedule = np.vstack((np.ones(demand.size), uc_schedule))
    uc_schedule = uc_schedule.T
            
    # Calculate fuel costs   
    marginal_gens_idx = np.array([np.max(np.where(u == 1)) for u in uc_schedule])
    non_marginal_gens = np.zeros((demand.size, env.num_gen), dtype=bool)
    marginal_gens = np.zeros((demand.size, env.num_gen), dtype=bool)
    for t, mg in enumerate(marginal_gens_idx):
        marginal_gens[t, mg] = True
        non_marginal_gens[t][np.delete(np.arange(env.num_gen), mg)] = True

    economic_dispatch = np.zeros((demand.size, env.num_gen), dtype=float)
    
    loaded_gens = np.array(non_marginal_gens * uc_schedule, dtype=bool)
    economic_dispatch = np.array(loaded_gens * gen_info_sorted.max_output.values, dtype=float)
    
    residual_load = demand - np.sum(economic_dispatch, axis=1)
    economic_dispatch[marginal_gens] = residual_load
    
    fc = np.sum(economic_dispatch * gen_info_sorted.min_fuel_cost.values * env.dispatch_resolution)
    
    return fc
    
def run_PL_senjyu(env, lookahead):
    """
    Get PL cost estimate using the pl_senjyu function above. Based on the 2003
    paper from Senjyu et al. 
    """
    if env.is_feasible() is False:
        print("PL: infeasible")
        return -env.min_reward
    
    else:
        demand = env.all_forecast[env.episode_timestep+1: env.episode_timestep+1+lookahead]
        gen_info_sorted = env.gen_info.sort_values(by='min_fuel_cost')
        
        # Run Senjyu to get binary schedule
        uc_schedule = pl_senjyu(env, lookahead).T
                
        # Calculate fuel costs   
        marginal_gens_idx = np.array([np.max(np.where(u == 1)) for u in uc_schedule])
        non_marginal_gens = np.zeros((lookahead, env.num_gen), dtype=bool)
        marginal_gens = np.zeros((lookahead, env.num_gen), dtype=bool)
        for t, mg in enumerate(marginal_gens_idx):
            marginal_gens[t, mg] = True
            non_marginal_gens[t][np.delete(np.arange(env.num_gen), mg)] = True
    
        economic_dispatch = np.zeros((lookahead, env.num_gen), dtype=float)
        
        loaded_gens = np.array(non_marginal_gens * uc_schedule, dtype=bool)
        economic_dispatch = np.array(loaded_gens * gen_info_sorted.max_output.values, dtype=float)
        
        residual_load = demand - np.sum(economic_dispatch, axis=1)
        economic_dispatch[marginal_gens] = residual_load
        
        fc = np.sum(economic_dispatch * gen_info_sorted.min_fuel_cost.values * env.dispatch_resolution)
        
        return fc
        
def run_PL_endless(env, lookahead):
    """
    Run priority list simulation for 24 hours ahead. 
    
    Steps: 
        1. Get priority list solution (this is approximate and is not necessarily a feasible
        solution.)
        2. Get economic dispatch for this, including ENS. 
        3. Calculate fuel costs and ENS costs: 
            - ENS costs are penalised at prices of £25/MWh for deficit and 
            £10/MWh for surplus, to account for the fact that the PL solution 
            is not optimal (or usually feasible).
    """
    uc = pl_senjyu(env, 
                   env.all_forecast[env.episode_timestep+1:env.episode_timestep+lookahead+1],
                   env.min_fuel_cost)
    
    uc_h = np.array(uc).transpose()
    dispatch = []
    ens = []
    for i, h in enumerate(uc_h):
        d, e = economic_dispatch(env=env, action=h, 
                                 demand=env.all_forecast[env.episode_timestep+i+1],
                                 lambda_lo=0, lambda_hi=30)
        dispatch.append(d)
        ens.append(e)

    fuel_costs = [sum(calculate_fuel_costs(env, dispatch[i], uc_h[i])) for i in range(lookahead)]
    
    ens_costs = calculate_ens_costs(ens)
    
    total_costs = [fuel_costs[i] + ens_costs[i] for i in range(lookahead)]
    return sum(total_costs)

def run_PL_new2(env, lookahead):
    """
    Priority list cost estimation for environment for subsequent periods.
    
    In this version, generator constraints must be obeyed
    """
    demand = env.episode_forecast[env.episode_timestep+1:env.episode_timestep+lookahead+1]
    wind = env.episode_wind_forecast[env.episode_timestep+1:env.episode_timestep+lookahead+1]
    net_demand = demand-wind # net demand is demand-wind
    demand=net_demand

    # Demand with reserve
    MARGIN = 0.2
    reserve = demand * (1+MARGIN)
    
    if env.is_feasible() is False:
        return -env.min_reward*demand.size
        
    gen_info_sorted = env.gen_info.sort_values(by='min_fuel_cost')
 
    uc_schedule = np.zeros([env.num_gen, demand.size])
    free_generators = np.zeros([env.num_gen, demand.size])
    # Fix constrained gens
    for g in range(env.num_gen):
        if env.status[g] < 0:
            n = max(gen_info_sorted.t_min_down.values[g] + env.status[g], 0)
            uc_schedule[g,:n] = 0
        else:
            n = max(gen_info_sorted.t_min_up.values[g] - env.status[g], 0)
            uc_schedule[g,:n] = 1
        free_generators[g,n:] = 1
            
    committed_cap = np.dot(uc_schedule.T, gen_info_sorted.max_output.values)
    residual_load = reserve - committed_cap

    for t in range(demand.size):
        res = residual_load[t]
        free_gens = np.where(free_generators[:,t] == 1)[0]
        for g in free_gens:
            if res <= 0:
                break
            # Commit the next generator
            uc_schedule[g,t] = 1
            res -= gen_info_sorted.max_output.values[g]

    # Estimate start costs (assume all starts are hot)
    extended_schedule = np.hstack((np.where(env.status > 0, 1, 0).reshape(-1,1),
                              uc_schedule))
    starts = np.sum(np.diff(extended_schedule) == 1, axis=1)
    start_cost = np.dot(gen_info_sorted.hot_cost.values, starts)

    # Estimate fuel costs using average heat rate of online generators, weighted by capacity
    uc_schedule = uc_schedule.T
    weighted_avg_hr = np.zeros(uc_schedule.shape[0])
    
    for t in range(uc_schedule.shape[0]):
        online_gens = np.where(uc_schedule[t])[0]
        weighted_avg_hr[t] = (np.dot(gen_info_sorted.max_cost_per_mwh.values[online_gens], 
                                  gen_info_sorted.min_output.values[online_gens])/
                              np.sum(gen_info_sorted.min_output.values[online_gens]))
        
    fuel_cost = np.dot(weighted_avg_hr, demand)*env.dispatch_resolution
    
    return fuel_cost + start_cost

def run_PL_new3(env, lookahead):    
    """
    This is like run_PL_new, except it reduces the resolution of the demand profile to 
    one hour resolution. This is simply done by downsampling by a factor of 1/env.dispatch_resolution
    """
    demand = env.episode_forecast[env.episode_timestep+1:env.episode_timestep+lookahead+1]
    wind = env.episode_wind_forecast[env.episode_timestep+1:env.episode_timestep+lookahead+1]
    net_demand = demand-wind # net demand is demand-wind
    demand=net_demand*1.1

    if env.is_feasible() is False:
        return -env.min_reward * demand.size

    gen_info_sorted = env.gen_info.sort_values(by='min_fuel_cost')
    
    # total number of periods: 
    num_periods = demand.size
    
    # Truncate the demand to hourly resolution,
    step_size = int(1/env.dispatch_resolution)
    remaining_periods = num_periods % step_size 
    demand = demand[0:demand.size:step_size]
    
    # Run PL as normal on the new demand 
    caps = np.cumsum(gen_info_sorted.max_output.values)[:-1]
        
    uc_schedule = np.array(np.array([demand < c for c in caps]))
    uc_schedule = np.where(uc_schedule, 0, 1)
    uc_schedule = np.vstack((np.ones(demand.size), uc_schedule))
    uc_schedule = uc_schedule.T
            
    # Calculate fuel costs   
    marginal_gens_idx = np.array([np.max(np.where(u == 1)) for u in uc_schedule])
    non_marginal_gens = np.zeros((demand.size, env.num_gen), dtype=bool)
    marginal_gens = np.zeros((demand.size, env.num_gen), dtype=bool)
    for t, mg in enumerate(marginal_gens_idx):
        marginal_gens[t, mg] = True
        non_marginal_gens[t][np.delete(np.arange(env.num_gen), mg)] = True

    economic_dispatch = np.zeros((demand.size, env.num_gen), dtype=float)
    
    loaded_gens = np.array(non_marginal_gens * uc_schedule, dtype=bool)
    economic_dispatch = np.array(loaded_gens * gen_info_sorted.max_output.values, dtype=float)
    
    residual_load = demand - np.sum(economic_dispatch, axis=1)
    economic_dispatch[marginal_gens] = residual_load
    
    # Fuel cost up to last period
    fuel_cost = np.sum(economic_dispatch[:-1] * gen_info_sorted.min_fuel_cost.values * 1)
    
    # Last period fuel cost: is a normal period if total periods divides exactly
    # into step_size, else multiply by remaining_periods/step_size. 
    if remaining_periods == 0:
        last_fuel_cost = np.sum(economic_dispatch[-1] * gen_info_sorted.min_fuel_cost.values * 1)
    else:
        last_fuel_cost = np.sum(economic_dispatch[-1] * gen_info_sorted.min_fuel_cost.values * (remaining_periods / step_size))
    
    return fuel_cost + last_fuel_cost

def run_PL_all(env):
    """
    Run run simple Priority List heuristic for each time period in a demand profile.

    Args:
        - root_env (Env): Env instance for the root environment 
    """
    demand = env.episode_forecast[env.episode_timestep+1:env.episode_length]
    wind = env.episode_wind_forecast[env.episode_timestep+1:env.episode_length]
    net_demand = demand-wind # net demand is demand-wind
    demand=net_demand

    # Demand with reserve
    MARGIN = 0.2
    reserve = demand * (1+MARGIN)
    
    gen_info_sorted = env.gen_info.sort_values(by='min_fuel_cost')
    caps = np.cumsum(gen_info_sorted.max_output.values)[:-1]
    
    uc_schedule = np.array(np.array([demand < c for c in caps]))
    uc_schedule = np.where(uc_schedule, 0, 1)
    uc_schedule = np.vstack((np.ones(demand.size), uc_schedule))
    uc_schedule = uc_schedule.T
            
    # Calculate fuel costs   
    marginal_gens_idx = np.array([np.max(np.where(u == 1)) for u in uc_schedule])
    non_marginal_gens = np.zeros((demand.size, env.num_gen), dtype=bool)
    marginal_gens = np.zeros((demand.size, env.num_gen), dtype=bool)
    for t, mg in enumerate(marginal_gens_idx):
        marginal_gens[t, mg] = True
        non_marginal_gens[t][np.delete(np.arange(env.num_gen), mg)] = True

    economic_dispatch = np.zeros((demand.size, env.num_gen), dtype=float)
    
    loaded_gens = np.array(non_marginal_gens * uc_schedule, dtype=bool)
    economic_dispatch = np.array(loaded_gens * gen_info_sorted.max_output.values, dtype=float)
    
    residual_load = demand - np.sum(economic_dispatch, axis=1)
    economic_dispatch[marginal_gens] = residual_load
    
    fc = np.sum(economic_dispatch * gen_info_sorted.min_fuel_cost.values * env.dispatch_resolution, axis=1)
    
    return fc


    
