from ts4uc.tree_search import rolling_horizon
from ts4uc.tree_search import anytime
from rl4uc.environment import make_env
import pandas as pd
import numpy as np


def test_rolling():

    np.random.seed(2)

    test_df = pd.DataFrame({'demand': [600, 700, 800],
                            'wind': [50, 30, 10]})
    env = make_env(mode='test', profiles_df=test_df, num_gen=5)
    returns = rolling_horizon.solve_rolling_anytime(env,
                                                    tree_search_func=anytime.ida_star,
                                                    time_budget=1.5,
                                                    num_scenarios=100,
                                                    heuristic_method=None,
                                                    policy=None)
    cost = returns[1]

    assert np.isclose(cost, 22178.303084543917), "Costs were {}".format(cost)
