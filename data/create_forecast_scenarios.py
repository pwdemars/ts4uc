#!/usr/bin/env python3
# Generate forecast error scenarios for evaluating schedules

from rl4uc.environment import make_env
import numpy as np
import pandas as pd
import json
import os


def sample_errors(seed, env, n_scenarios, n_periods=48):

    np.random.seed(seed)

    all_dfs = []
    for n in range(n_scenarios):
        env.reset()
        demand_xs, demand_zs = [], []
        wind_xs, wind_zs = [], []
        for i in range(n_periods):
            env.arma_demand.step()
            env.arma_wind.step()

            demand_xs.append(env.arma_demand.xs[0])
            demand_zs.append(env.arma_demand.zs[0])

            wind_xs.append(env.arma_wind.xs[0])
            wind_zs.append(env.arma_wind.zs[0])

        scenario_df = pd.DataFrame({'demand_x': demand_xs,
                                    'demand_z': demand_zs,
                                    'wind_x': wind_xs,
                                    'wind_z': wind_zs,
                                    'period': np.arange(n_periods)})

        scenario_df['scenario'] = n
        all_dfs.append(scenario_df)

    scenario_df = pd.concat(all_dfs)

    return scenario_df


if __name__ == '__main__':

    SEED = 999
    NUM_SCENARIOS = 1000
    os.makedirs('error_scenarios', exist_ok=True)

    for NUM_GEN in [10, 20, 30]:
        # Initialise the environment
        env_params = json.load(open('day_ahead/{}gen/30min/env_params.json'.format(NUM_GEN)))
        env = make_env(**env_params)
        _, prof_df = env.sample_day()
        env = make_env(mode='test', profiles_df=prof_df, **env_params)

        scenario_df = sample_errors(SEED, env, NUM_SCENARIOS)

        scenario_df.to_csv('error_scenarios/{}gen_scenarios.csv'.format(NUM_GEN),
                           index=False)
