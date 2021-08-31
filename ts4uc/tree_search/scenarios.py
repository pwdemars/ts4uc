#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import copy

def sample_errors(env, N, horizon, seeded=False):
    """
    Calculate N realisations of net demand for demand and wind forecast errors
    using ARMA processes that may be at t>=0. 
    """
    demand_errors = np.zeros((N, horizon))
    wind_errors = np.zeros((N, horizon))
    for i in range(N):
        arma_demand = copy.deepcopy(env.arma_demand)
        arma_wind = copy.deepcopy(env.arma_wind)

        if seeded==False:
            arma_demand.reset()
            arma_wind.reset()

        for j in range(horizon):
            demand_errors[i,j] = arma_demand.step()
            wind_errors[i,j] = arma_wind.step()

    return demand_errors, wind_errors

def calculate_expected_costs(env, action, demand_scenarios, wind_scenarios, availability_scenarios=None):
    """
    Calculate the expected fuel costs over a set of possible 
    net demands.

    This involves calculating the fuel costs and lost load costs 
    for each net demand realisation, and then adding on the startup 
    cost (which is invariant with the net demand).
    """
    total = 0

    if availability_scenarios is None:
        availability_scenarios = np.ones(shape=(demand_scenarios.size, env.num_gen))

    if env.curtailment:
        curtail = bool(action[-1])
        commitment_action = np.copy(action)[:-1]
    else:
        curtail = False
        commitment_action = action

    if curtail: 
        wind_scenarios = wind_scenarios * env.curtailment_factor
        
    net_demand_scenarios = demand_scenarios - wind_scenarios

    for net_demand, availability in zip(net_demand_scenarios, availability_scenarios):

        fuel_costs, disp = env.calculate_fuel_cost_and_dispatch(net_demand, commitment_action, availability)
        fuel_cost = np.sum(fuel_costs)
        lost_load_cost = env.calculate_lost_load_cost(net_demand, disp)
        # if lost_load_cost > 0:
            # print("Lost load at period {}. Demand {:.2f}, disp {:.2f}".format(env.episode_timestep, net_demand, np.sum(disp)))
        total += fuel_cost + lost_load_cost

    exp_cost = total/demand_scenarios.shape[0]
    exp_cost += env.start_cost

    return exp_cost

def get_net_demand_scenarios(profile_df, env, num_scenarios):

    demand_errors, wind_errors = sample_errors(env, num_scenarios, env.episode_length)
    scenarios = (profile_df.demand.values + demand_errors) - (profile_df.wind.values + wind_errors)
    scenarios = np.clip(scenarios, env.min_demand, env.max_demand)

    return scenarios

def get_scenarios(profile_df, env, num_scenarios):

    demand_errors, wind_errors = sample_errors(env, num_scenarios, env.episode_length)
    demand_scenarios = profile_df.demand.values + demand_errors
    wind_scenarios =  profile_df.wind.values + wind_errors
    
    return demand_scenarios, wind_scenarios

def sample_availability_multiple(env, schedule, initial_availability):
    """
    Sample possible realisations of generator availability over multiple time periods
    """
    time_periods = schedule.shape[0]
    num_scenarios = initial_availability.shape[0]
    availability = np.zeros(shape=(num_scenarios, time_periods, env.num_gen))
    for i, avail in enumerate(initial_availability):
        env.availability = avail
        for t in range(time_periods):
            env.commitment = schedule[t]
            outage = env._sample_outage()
            env._update_availability(outage)
            availability[i,t] = env.availability

    return availability

def sample_availability_single(env, action, initial_availability):
    """
    Sample possible realisations of generator availability for a single timestep

    In order to sample availability correctly, we have to set initial 'seed' availabilities. 
    This function returns a new set of availabilities, with dimensions (n_scenarios, num_gen)
    """
    original_availability = np.copy(env.availability)
    original_commitment = np.copy(env.commitment)
    
    num_scenarios = initial_availability.shape[0]
    availabilities = np.zeros(shape=(num_scenarios, env.num_gen))
    
    for i, avail in enumerate(initial_availability):
        outage = env._sample_outage(avail, action)
        new_availability = np.clip(avail - outage, 0, 1)
        availabilities[i] = new_availability
        
    return availabilities

def get_global_outage_scenarios(env, T, num_scenarios):
    """
    Generate a (env.num_gen, env.episode_length, num_scenarios) array of availability 
    scenarios. 

    The idea is that for a node with 
    """

    outage_scenarios = np.zeros((env.num_gen, T, num_scenarios))
    avail = np.ones(env.num_gen)
    action = np.ones(env.num_gen)
    for t in range(T):
        status = np.ones(env.num_gen)*(t+1)
        for n in range(num_scenarios):
            outage = env._sample_outage(avail, action, status)
            outage_scenarios[:,t,n] = outage

    return outage_scenarios

def sample_outage_scenarios(global_outage_scenarios, status, action): 
    outage_scenarios = np.zeros((global_outage_scenarios.shape[0], global_outage_scenarios.shape[-1]))
    # on_idx = np.logical_and((np.where(status > 0)[0], action))
    on_idx = np.logical_and(status, action)
    print(on_idx)
    outage_scenarios[on_idx] = global_outage_scenarios[on_idx, status[on_idx]] 
    print(outage_scenarios.mean())
    return outage_scenarios.T

def sample_availability_scenarios(global_outage_scenarios, previous_availability_scenarios, status, action):
    outage_scenarios = sample_outage_scenarios(global_outage_scenarios, status, action)
    availability_scenarios = np.clip(previous_availability_scenarios - outage_scenarios, 0, 1)
    return availability_scenarios


