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

def heuristic(node, horizon, method='check_lost_load'):
	"""Simple heuristic that gives np.inf if a node's state is infeasible, else 0"""
	if method=='check_lost_load':
		return check_lost_load(node.state, horizon)
	else:
		raise ValueError('{} is not a valid heuristic'.format(method))