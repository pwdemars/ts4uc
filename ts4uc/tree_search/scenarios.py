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

def calculate_expected_costs(env, action, net_demands):
    """
    Calculate the expected fuel costs over a set of possible 
    net demands.

    This involves calculating the fuel costs and lost load costs 
    for each net demand realisation, and then adding on the startup 
    cost (which is invariant with the net demand).
    """
    total = 0
    for net_demand in net_demands:
        fuel_costs, disp = env.calculate_fuel_cost_and_dispatch(net_demand, action)
        fuel_cost = np.sum(fuel_costs)
        lost_load_cost = env.calculate_lost_load_cost(net_demand, disp)
        carbon_cost, _ = env._calculate_carbon_cost(fuel_costs)
        # if lost_load_cost > 0:
            # print("Lost load at period {}. Demand {:.2f}, disp {:.2f}".format(env.episode_timestep, net_demand, np.sum(disp)))
        total += fuel_cost + lost_load_cost + carbon_cost

    exp_cost = total/net_demands.shape[0]
    exp_cost += env.start_cost

    return exp_cost

def get_net_demand_scenarios(profile_df, env, num_scenarios):

    demand_errors, wind_errors = sample_errors(env, num_scenarios, env.episode_length)
    scenarios = (profile_df.demand.values + demand_errors) - (profile_df.wind.values + wind_errors)
    scenarios = np.clip(scenarios, env.min_demand, env.max_demand)

    import pandas as pd 

    df = pd.DataFrame({'max_demand_errors': np.max(demand_errors, axis=0),
                       'min_wind_errors': np.min(wind_errors, axis=0),
                       'std_demand_errors': np.std(demand_errors, axis=0),
                       'std_wind_errors': np.std(wind_errors, axis=0),
                       'max_demand_scenarios': np.max(scenarios, axis=0)})
    return scenarios
