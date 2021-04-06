#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 16:55:21 2020

@author: patrickdemars
"""

import numpy as np
import pandas as pd
import os

import rl4uc.helpers as rl4uc_helpers

def cap_states(states, gen_info):
    """
    Transforms states such that on/off times are capped to min up/down times
    respectively. Helps training by reducing the state space. Also improves
    robustness to unseen values.
    """
    new_states = np.copy(states)
    num_gen = len(gen_info)

    for i in range(num_gen):
        on_idx = np.where(states[:,i] > 0)
        off_idx = np.where(states[:,i] < 0)
        new_states[on_idx,i] = np.minimum(states[on_idx,i], gen_info.t_min_up.values[i])
        new_states[off_idx,i] = np.maximum(states[off_idx,i], -gen_info.t_min_down.values[i])

    return new_states

def scale_states(states, gen_info):
    """
    Scales states linearly so that generator status = 1 when on & available to turn off,
    and 0 when off and available to turn on. 
    
    The demand forecast vector remains unchanged.
    """
    num_gen = gen_info.shape[0]
    x_min = -gen_info['t_min_down'].values
    x_max = gen_info['t_min_up'].values
    x = states[:,:num_gen]
    states[:,:num_gen] = 2*(x[:,:num_gen]-x_min)/(x_max-x_min) - 1
    return states

def get_paths(node, paths=None, current_path=None):
    """
    Determine the realisations of demand from the tree rooted at 
    node. Returns a list of demand profiles, corresponding to the unique traversals
    from root to leaf.
    """
    if paths is None:
        paths = []
    if current_path is None:
        current_path = []

    if node.node_type == 'decision':
        current_path.append(node.environment.net_demand)
    
    if len(node.children) == 0:
        paths.append(current_path)
    else:
        for child in node.children.values():
            get_paths(child, paths, list(current_path))
    
    return paths

def process_observation(obs, env, forecast_horizon, forecast_errors=False):
    timestep = obs['timestep']
    status_norm = rl4uc_helpers.cap_and_normalise_status(obs['status'], env)
    demand_forecast = obs['demand_forecast'][timestep+1:]
    wind_forecast = obs['wind_forecast'][timestep+1:]

    # Repeat the final value if forecast does not reach the horizon
    if len(demand_forecast) < forecast_horizon:
        demand_forecast = np.append(demand_forecast,
                                    np.repeat(demand_forecast[-1],
                                              forecast_horizon-len(demand_forecast)))
    demand_forecast = demand_forecast[:forecast_horizon]

    # Repeat the final value if forecast does not reach the horizon
    if len(wind_forecast) < forecast_horizon:
        wind_forecast = np.append(wind_forecast,
                                    np.repeat(wind_forecast[-1],
                                              forecast_horizon-len(wind_forecast)))
    wind_forecast = wind_forecast[:forecast_horizon]

    # Scale the demand and wind 
    demand_norm = demand_forecast / env.max_demand
    wind_norm = wind_forecast / env.max_demand

    # Scale the timestep
    timestep_norm = np.array(timestep / env.episode_length).reshape(1,)

    processed_obs = np.concatenate((status_norm,
                                    demand_norm,
                                    wind_norm,
                                    timestep_norm))

    return processed_obs

def calculate_gamma(credit_assignment_1hr, dispatch_freq_mins):
    """
    Calculate gamma based on a desired credit assignment at 1hr and dispatch frequency
    in minutes,
    
    This function is used to control for the fact that gamma should vary with 
    varying dispatch frequency (higher frequency dispatch should have a higher gamma,
    corresponding for longer time dependencies in terms of number of decision periods.
    """
    gamma = credit_assignment_1hr ** (dispatch_freq_mins / (60))
    return gamma


def plot_demand_realisations(env, num_samples=10):
    import matplotlib.pyplot as plt 
    
    seed = np.random.randint(10000)
    
    for i in range(num_samples):
        np.random.seed(seed)
        env.reset()
        np.random.seed()
        
        demands = []
        winds = []
        net_demands = []
        net_forecasts = []
        
        for j in range(env.episode_length):
            env.step(np.ones(env.num_gen))
            demands.append(env.demand_real)
            winds.append(env.wind_real)
            net_forecasts.append(env.forecast - env.wind_forecast)
            net_demands.append(env.net_demand)
        # plt.plot(demands)
        # plt.plot(winds)
        plt.plot(net_demands, color='black', alpha=0.3)

    plt.plot(net_forecasts, color='red')

    plt.show()

def test_schedule(env, schedule, seed=999, num_samples=1000, deterministic=False):
    test_costs = []
    lost_loads = []
    print("Testing schedule...")
    np.random.seed(seed)
    for i in range(num_samples):
        env.reset()
        total_reward = 0 
        ll = 0
        for action in schedule:
            action = np.where(np.array(action)>0, 1, 0)
            obs,reward,done = env.step(action, deterministic)
            total_reward += reward
            if env.ens:
                ll += 1
                print("ENS at period {}; forecast: {:.2f}; real: {:.2f}".format(env.episode_timestep, env.forecast - env.wind_forecast, env.net_demand))
        test_costs.append(-total_reward)
        lost_loads.append(ll)

    return test_costs, lost_loads

def save_results(prof_name, 
                 save_dir, 
                 num_gen, 
                 schedule,
                 test_costs, 
                 lost_loads, 
                 time_taken,
                 period_time_taken=None,
                 depths=None):
    # save test costs
    all_test_costs = pd.DataFrame({prof_name: test_costs})
    all_test_costs.to_csv(os.path.join(save_dir, '{}_costs.csv'.format(prof_name)), index=False, compression=None)

    # save lost load events
    all_lost_loads = pd.DataFrame({prof_name: lost_loads})
    all_lost_loads.to_csv(os.path.join(save_dir, '{}_lost_load.csv'.format(prof_name)), index=False, compression=None)

    # save schedule
    columns =  ['schedule_' + str(i) for i in range(num_gen)]
    schedule_df = pd.DataFrame(schedule, columns=columns)
    schedule_df.to_csv(os.path.join(save_dir, '{}_solution.csv'.format(prof_name)), index=False)

    if period_time_taken is not None:
        period_tt_df = pd.DataFrame({'period': np.arange(len(period_time_taken)),
                                     'time': period_time_taken})
        period_tt_df.to_csv(os.path.join(save_dir, '{}_period_times.csv'.format(prof_name)), index=False)

    if depths is not None:
        depths_df = pd.DataFrame({'period': np.arange(len(depths)),
                                  'time': depths})
        depths_df.to_csv(os.path.join(save_dir, '{}_depths.csv'.format(prof_name)), index=False)

    # save time taken
    with open(os.path.join(save_dir, '{}_time.txt'.format(prof_name)), 'w') as f:
        f.write(str(round(time_taken, 2)))
        f.write('\n')
        f.close()

def get_scenarios(env, N):
    """
    Calculate N realisations of net demand for demand and wind forecast errors
    using ARMA processes defined in env. 
    """
    demand_errors = np.zeros((N, env.episode_length))
    wind_errors = np.zeros((N, env.episode_length))
    for i in range(N):
        env.arma_demand.reset()
        env.arma_wind.reset()

        for j in range(env.episode_length):
            demand_errors[i,j] = env.arma_demand.step()
            wind_errors[i,j] = env.arma_wind.step()

    return demand_errors, wind_errors

def save_branches(prof_name, save_dir, n_branches):
    """
    Save the number of branches from the root at each timestep
    """
    fn = os.path.join(save_dir, '{}_branches.txt'.format(prof_name))
    np.savetxt(fn, n_branches, fmt='%d')
