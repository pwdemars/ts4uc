#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from rl4uc.environment import make_env

from ts4uc.tree_search.scenarios import get_net_demand_scenarios
from ts4uc.tree_search.day_ahead import solve_day_ahead
from ts4uc.tree_search.algos import uniform_cost_search
from ts4uc.agents.ac_agent import ACAgent
from ts4uc import helpers

import numpy as np 
import pandas as pd 
import torch
import json

POLICY_FILENAME = '../data/dummy_policies/g5/ac_final.pt' 
POLICY_PARAMS_FN = '../data/dummy_policies/g5/params.json'
ENV_PARAMS_FN = '../data/dummy_policies/g5/env_params.json'
TEST_DATA_FN = '../data/day_ahead/5gen/30min/profile_2019-11-09.csv'
HORIZON = 2
BRANCHING_THRESHOLD = 0.05
TREE_SEARCH_FUNC_NAME = 'uniform_cost_search'
SEED = 1 
NUM_SCENARIOS = 100
TEST_SAMPLE_SEED = 999
TIME_PERIODS = 4
NUM_SAMPLES = 1000

def test_uniform_cost_search():

        np.random.seed(SEED)
        torch.manual_seed(SEED)

        # Load parameters
        env_params = json.load(open(ENV_PARAMS_FN))
        policy_params = json.load(open(POLICY_PARAMS_FN))

        # Load profile 
        profile_df = pd.read_csv(TEST_DATA_FN)[:TIME_PERIODS]

        params = {'horizon': HORIZON,
                          'branching_threshold': BRANCHING_THRESHOLD}

        # Init env
        env = make_env(mode='test', profiles_df=profile_df, **env_params)

        # Load policy
        policy = ACAgent(env, test_seed=SEED, **policy_params)
        policy.load_state_dict(torch.load(POLICY_FILENAME))
        policy.eval()

        # Generate scenarios for demand and wind errors
        scenarios = get_net_demand_scenarios(profile_df, env, NUM_SCENARIOS)

        solve_returns = solve_day_ahead(env=env, 
                                          net_demand_scenarios=scenarios, 
                                          tree_search_func=uniform_cost_search,
                                          policy=policy,
                                          **params)
        schedule_result = solve_returns[0]

        # Get distribution of costs for solution by running multiple times through environment
        test_costs, lost_loads = helpers.test_schedule(env, schedule_result, TEST_SAMPLE_SEED, NUM_SAMPLES)
        mean_cost = np.mean(test_costs)

        assert np.isclose(mean_cost, 22608.283119377622), "Costs were: {}".format(mean_cost)
