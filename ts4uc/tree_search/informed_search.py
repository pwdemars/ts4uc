#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

def check_lost_load(state, horizon):
	"""Check if forecast demand can be met for the next H periods from state"""
	if horizon == 0:
		return 0
	horizon = min(horizon, max(state.gen_info.t_min_down.max(), abs(state.gen_info.t_min_down.min())))
	for t in range(1,horizon+1):
		net_demand = state.episode_forecast[state.episode_timestep+t] - state.episode_wind_forecast[state.episode_timestep+t] # Nominal demand for t+1th period ahead
		future_status = state.status + (t)*np.where(state.status >0, 1, -1) # Assume all generators are kept on where possible

		available_generators = np.logical_or((-future_status >= state.t_min_down), state.commitment) # Determines the availability of generators as binary array

		available_cap = np.dot(available_generators, state.max_output)

		if available_cap < net_demand:
			return np.inf

	return 0

def priority_list(state, horizon): 
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
	# Get the cumulative sum of outputs, sorted by heat rates
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

def advanced_priority_list(state, horizon):
	"""
	Priority list cost estimation for environment for subsequent periods.
	
	In this version, generator constraints must be obeyed
	"""
	if horizon == 0:
		return 0

	time_interval = state.dispatch_freq_mins/60

	# Sort to priority list order 
	gen_info_sorted = state.gen_info.sort_values('min_fuel_cost')

	# Net demand forecast
	demand = state.episode_forecast[state.episode_timestep+1:state.episode_timestep+horizon+1]
	wind = state.episode_wind_forecast[state.episode_timestep+1:state.episode_timestep+horizon+1]
	net_demand = demand - wind
		 
	uc_schedule = np.zeros([state.num_gen, demand.size])
	free_generators = np.zeros([state.num_gen, demand.size])
	# Fix constrained gens
	for g in range(state.num_gen):
		if state.status[g] < 0:
			n = max(gen_info_sorted.t_min_down.values[g] + state.status[g], 0)
			uc_schedule[g,:n] = 0
		else:
			n = max(gen_info_sorted.t_min_up.values[g] - state.status[g], 0)
			uc_schedule[g,:n] = 1
		free_generators[g,n:] = 1
			
	committed_cap = np.dot(uc_schedule.T, gen_info_sorted.max_output.values)
	residual_load = net_demand - committed_cap

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
	# extended_schedule = np.hstack((np.where(state.status > 0, 1, 0).reshape(-1,1),
	# 						  uc_schedule))
	# starts = np.sum(np.diff(extended_schedule) == 1, axis=1)
	# start_cost = np.dot(gen_info_sorted.hot_cost.values, starts)
	start_cost = 0

	# Estimate fuel costs using average heat rate of online generators, weighted by capacity
	uc_schedule = uc_schedule.T
	weighted_avg_hr = np.zeros(uc_schedule.shape[0])
	
	for t in range(uc_schedule.shape[0]):
		online_gens = np.where(uc_schedule[t])[0]
		weighted_avg_hr[t] = (np.dot(gen_info_sorted.min_fuel_cost.values[online_gens], 
								  gen_info_sorted.min_output.values[online_gens])/
							  np.sum(gen_info_sorted.min_output.values[online_gens]))
		
	fuel_cost = np.dot(weighted_avg_hr, demand) * time_interval

	# Add a lost load cost if res > 0
	if res > 0: 
		lost_load_cost = res * state.voll * time_interval
	else:
		lost_load_cost = 0
	
	return fuel_cost + start_cost + lost_load_cost

def pl_plus_ll(state, horizon):
	if check_lost_load(state, horizon) == np.inf:
		return np.inf
	else:
		return priority_list(state, horizon)

def heuristic(node, horizon, method='pl_plus_ll'):
	"""Simple heuristic that givees np.inf if a node's state is infeasible, else 0"""
	if method=='check_lost_load':
		heuristic_cost = check_lost_load(node.state, horizon)
	elif method=='priority_list':
		heuristic_cost = priority_list(node.state, horizon)
	elif method=='pl_plus_ll':
		heuristic_cost = pl_plus_ll(node.state, horizon)
	elif method=='advanced_priority_list':
		heuristic_cost = advanced_priority_list(node.state, horizon)
	else:
		raise ValueError('{} is not a valid heuristic'.format(method))
	return heuristic_cost