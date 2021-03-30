from rl4uc.environment import make_env

from ts4uc.tree_search.scenarios import get_net_demand_scenarios
from ts4uc.tree_search.anytime import solve_day_ahead_anytime
from ts4uc.tree_search.anytime import ida_star
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
SEED = 1 
NUM_SCENARIOS = 100
TEST_SAMPLE_SEED = 999
NUM_SAMPLES = 1000
TIME_BUDGET = 1
TIME_PERIODS = 5
HEURISTIC_METHOD = 'advanced_priority_list'

def test_ida_star():

    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # Load parameters
    env_params = json.load(open(ENV_PARAMS_FN))
    policy_params = json.load(open(POLICY_PARAMS_FN))

    # Load profile 
    profile_df = pd.read_csv(TEST_DATA_FN)
    profile_df = profile_df[:TIME_PERIODS]

    params = {'horizon': HORIZON,
              'branching_threshold': BRANCHING_THRESHOLD,
              'heuristic_method': HEURISTIC_METHOD}

    # Init env
    env = make_env(mode='test', profiles_df=profile_df, **env_params)

    # Load policy
    policy = None

    # Generate scenarios for demand and wind errors
    scenarios = get_net_demand_scenarios(profile_df, env, NUM_SCENARIOS)

    schedule_result = solve_day_ahead_anytime(env=env, 
                                              time_budget=TIME_BUDGET,
                                              net_demand_scenarios=scenarios, 
                                              tree_search_func=ida_star,
                                              policy=policy,
                                              **params)
    # Get distribution of costs for solution by running multiple times through environment
    test_costs, lost_loads = helpers.test_schedule(env, schedule_result, TEST_SAMPLE_SEED, NUM_SAMPLES)

    assert np.isclose(np.mean(test_costs), 27395.61572936192)
