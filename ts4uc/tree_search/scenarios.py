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

def calculate_expected_costs(env, net_demands):
    """
    Calculate the expected fuel costs over a set of possible 
    net demands.

    This involves calculating the fuel costs and lost load costs 
    for each net demand realisation, and then adding on the startup 
    cost (which is invariant with the net demand).
    """
    total = 0
    for net_demand in net_demands:
        fuel_cost, disp = env.calculate_fuel_cost_and_dispatch(net_demand)
        lost_load_cost = env.calculate_lost_load_cost(net_demand, disp)
        total += fuel_cost + lost_load_cost

    exp_cost = total/net_demands.shape[0]
    exp_cost += env.start_cost

    return exp_cost

def get_net_demand_scenarios(profile_df, env, num_scenarios):
    demand_errors, wind_errors = sample_errors(env, num_scenarios, env.episode_length)
    scenarios = (profile_df.demand.values + demand_errors) - (profile_df.wind.values + wind_errors)
    return np.clip(scenarios, env.min_demand, env.max_demand)
