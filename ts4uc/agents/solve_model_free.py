#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from rl4uc.environment import make_env

from ts4uc import helpers

from ts4uc.agents.ppo_async.ac_agent import ACAgent

import numpy as np
import argparse 
import torch
import pandas as pd
import os
import json
import gc
import time

def solve_model_free_day_ahead(env,
                               policy):
    obs = env.reset()
    final_schedule = np.zeros((env.episode_length, env.num_gen))
    for t in range(env.episode_length):
        a, sub_obs, sub_acts, log_probs = policy.generate_action(env, obs)
        obs, reward, done = env.step(a, deterministic=True)
        final_schedule[t,:] = a
    return final_schedule
        

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Solve a single day with tree search')
    parser.add_argument('--save_dir', '-s', type=str, required=True,
                        help='Directory to save results')
    parser.add_argument('--policy_params_fn', '-pp', type=str, required=True,
                        help='Filename for parameters')
    parser.add_argument('--env_params_fn', '-e', type=str, required=True,
                        help='Filename for environment parameters, including ARMAs, number of generators, dispatch frequency')
    parser.add_argument('--policy_filename', '-pf', type=str, required=True,
                        help="Filename for policy [.pt]. Set to 'none' or omit this argument to train from scratch")
    parser.add_argument('--test_data_dir', '-t', type=str, required=True,
                        help='Directory with problem files [.csv]')
    parser.add_argument('--num_samples', type=int, required=False, default=1000,
                        help='Number of times to sample running the schedules through the environment')
    parser.add_argument('--seed', type=int, required=False, default=np.random.randint(0,10000),
                        help='Set random seed')

    args = parser.parse_args()

    # Create results directory
    os.makedirs(args.save_dir, exist_ok=True)

    # Update params
    params = vars(args)

    # Update params to note model_free
    params.update({'expansion_mode': 'model_free'})

    # Read the parameters
    env_params = json.load(open(args.env_params_fn))
    policy_params = json.load(open(args.policy_params_fn))

    # Set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Save params file to save_dir 
    with open(os.path.join(args.save_dir, 'params.json'), 'w') as fp:
        fp.write(json.dumps(params, sort_keys=True, indent=4))

    # Save env params to save_dir
    with open(os.path.join(args.save_dir, 'env_params.json'), 'w') as fp:
        fp.write(json.dumps(env_params, sort_keys=True, indent=4))

    for test_data in [f for f in os.listdir(args.test_data_dir) if '.csv' in f]:

        prof_name = os.path.basename(os.path.normpath(test_data)).split('.')[0]

        print("------------------")
        print("Profile: {}".format(prof_name))
        print("------------------")

        # Initialise environment with forecast profile and reference forecast (for scaling)
        profile_df = pd.read_csv(os.path.join(args.test_data_dir, test_data))
        env = make_env(mode='test', profiles_df=profile_df, **env_params)

        # Load policy
        policy = ACAgent(env, test_seed=args.seed, **policy_params)
        policy.load_state_dict(torch.load(args.policy_filename))
        policy.eval()

        # Solve
        s = time.time()
        schedule_result = solve_model_free_day_ahead(env, policy)
        time_taken = time.time() - s

        # # Get distribution of costs for solution by running multiple times through environment
        # TEST_SAMPLE_SEED=999
        # test_costs, lost_loads = helpers.test_schedule(env, schedule_result, TEST_SAMPLE_SEED, args.num_samples)
        # helpers.save_results(prof_name=prof_name, 
        #                      save_dir=args.save_dir, 
        #                      num_gen=env.num_gen, 
        #                      schedule=schedule_result,
        #                      test_costs=test_costs, 
        #                      lost_loads=lost_loads,
        #                      time_taken=time_taken)

        TEST_SAMPLE_SEED=999
        results = helpers.test_schedule(env, schedule_result, TEST_SAMPLE_SEED, args.num_samples)
        helpers.save_results(prof_name=prof_name, 
                             save_dir=args.save_dir, 
                             num_gen=env.num_gen, 
                             schedule=schedule_result,
                             test_costs=results['total_cost'], 
                             test_kgco2=results['kgco2'],
                             lost_loads=results['lost_load_events'],
                             results_df=results,
                             time_taken=time_taken)

        print("Done")
        print()
        print("Mean costs: ${:.2f}".format(np.mean(results['total_cost'])))
        print("Lost load prob: {:.3f}%".format(100*np.sum(results['lost_load_events'])/(args.num_samples * env.episode_length)))
        print("Time taken: {:.2f}s".format(time_taken))
        print() 
