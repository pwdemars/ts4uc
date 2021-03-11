#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  4 17:12:08 2020

@author: patrickdemars
"""

DEFAULT_SAVE_INTERVAL = 5000
EPOCH_SAVE_INTERVAL = 1000

from rl4uc.rl4uc.environment import make_env
import torch
from torch.optim.lr_scheduler import LambdaLR
import torch.optim as optim
import torch.multiprocessing as mp
mp.set_start_method('spawn', force=True)

from ac_agent import ACAgent

import numpy as np
import os
import time

def log(save_dir, rewards, timesteps, mean_entropy, loss_v, explained_variance):

    rpt = np.array(rewards)/np.array(timesteps)

    with open(os.path.join(save_dir, 'epoch_mean_timesteps.txt'), 'a') as f:
        f.write(str(np.mean(timesteps)) + '\n')
    with open(os.path.join(save_dir, 'epoch_mean_rpt.txt'), 'a') as f:
            f.write(str(np.mean(rpt)) + '\n')
    with open(os.path.join(save_dir, 'epoch_q25_timesteps.txt'), 'a') as f:
        f.write(str(np.quantile(timesteps, 0.25)) + '\n')
    with open(os.path.join(save_dir, 'epoch_q25_rpt.txt'), 'a') as f:
            f.write(str(np.quantile(rpt, 0.25)) + '\n')
    with open(os.path.join(save_dir, 'epoch_q75_timesteps.txt'), 'a') as f:
        f.write(str(np.quantile(timesteps, 0.75)) + '\n')
    with open(os.path.join(save_dir, 'epoch_q75_rpt.txt'), 'a') as f:
            f.write(str(np.quantile(rpt, 0.75)) + '\n')
    with open(os.path.join(save_dir, 'epoch_mean_entropy.txt'), 'a') as f:
            f.write(str(mean_entropy) + '\n')
    with open(os.path.join(save_dir, 'epoch_loss_v.txt'), 'a') as f:
            f.write(str(loss_v) + '\n')
    with open(os.path.join(save_dir, 'epoch_explained_variance.txt'), 'a') as f:
            f.write(str(explained_variance) + '\n')

def save_ac(save_dir, ac, epoch):
    torch.save(ac.state_dict(), os.path.join(save_dir, 'ac_' + str(epoch.item()) + '.pt'))

def run_epoch(save_dir, env, local_ac, shared_ac, pi_optimizer, v_optimizer, epoch_counter):
    obs = env.reset()
    epoch_done = False
    done = False
    ep_reward, ep_timesteps = 0, 0

    rewards = []
    timesteps = []

    while epoch_done is False:

        # Choose action
        action, sub_obs, sub_acts, log_probs = local_ac.generate_action(env, obs)
        
        # Get state-value pair
        value, obs_processed = local_ac.get_value(obs)
        
        # Advance environment
        new_obs, reward, done = env.step(action)

        # Simple transformation of reward
        reward = 1+reward/-env.min_reward
        reward = reward.clip(-10, 10)

        # Update episode rewards and timesteps survived
        ep_reward += reward
        ep_timesteps += 1

        local_ac.critic_buffer.store(obs_processed, reward)
        for idx in range(len(sub_obs)):
            local_ac.actor_buffer.store(sub_obs[idx], sub_acts[idx], log_probs[idx], reward, value)

        obs = new_obs

        if done:
            local_ac.actor_buffer.finish_ep(last_val=0)
            local_ac.critic_buffer.finish_ep(last_val=0)
            
            rewards.append(ep_reward)
            timesteps.append(ep_timesteps)

            obs = env.reset()
            ep_reward, ep_timesteps = 0,0

        if local_ac.actor_buffer.is_full():
            if not done: 
                local_ac.actor_buffer.finish_ep(last_val=local_ac.get_value(obs)[0].detach().numpy())
                local_ac.critic_buffer.finish_ep(last_val=local_ac.get_value(obs)[0].detach().numpy())

            entropy, loss_v, explained_variance = shared_ac.update(local_ac, pi_optimizer, v_optimizer)
            mean_entropy, loss_v, explained_variance = torch.mean(entropy).item(), loss_v.item(), explained_variance.item()

            epoch_done = True

        done = False
            
    log(save_dir, rewards, timesteps, mean_entropy, loss_v, explained_variance)
    if epoch_counter % EPOCH_SAVE_INTERVAL == 0:
        print("---------------------------")
        print("saving actor critic weights")
        print("---------------------------")
        save_ac(save_dir, shared_ac, epoch_counter) 

def run_worker(save_dir, rank, num_epochs, shared_ac, epoch_counter, params):
    """
    Training with a single worker. 
    
    Each worker initialises its own optimiser. Parameters for the policy network
    are shared between workers.
        
    Results are written to .txt files which are shared between workers.
    """
    start_time = time.time()
    
    pi_optimizer = optim.Adam(shared_ac.parameters(), lr=params.get('ac_learning_rate'))
    v_optimizer = optim.Adam(shared_ac.parameters(), lr=params.get('cr_learning_rate'))
    
    # Scheduler for learning rates
#     lambda_lr = lambda epoch: (num_epochs - epoch_counter).item()/num_epochs
#     pi_scheduler = LambdaLR(pi_optimizer, lr_lambda=lambda_lr)
#     v_scheduler = LambdaLR(v_optimizer, lr_lambda=lambda_lr)


    np.random.seed(params.get('seed') + rank)
    env = make_env(**params)
    
    local_ac = ACAgent(env, **params)
        
    while epoch_counter < num_epochs:
        
        epoch_counter += 1 
        print("Epoch: {}".format(epoch_counter.item()))
        
        local_ac.load_state_dict(shared_ac.state_dict())
        
        # Update entropy coefficient (beta)
#         factor = (num_epochs - epoch_counter + 1).item()/num_epochs
        factor = local_ac.entropy_decay_rate ** (epoch_counter.item())
        local_ac.entropy_coef = local_ac.entropy_coef_init * factor
                
        # Run an epoch, including updating the shared network
        run_epoch(save_dir, env, local_ac, shared_ac, pi_optimizer, v_optimizer, epoch_counter)
        
        # Step LRs
#         pi_scheduler.step()
#         v_scheduler.step()
    
    # Record time taken
    time_taken = time.time() - start_time
    with open(os.path.join(save_dir, 'time_taken.txt'), 'w') as f:
        f.write(str(time_taken) + '\n')
        
if __name__ == "__main__":
    
    import argparse
    import json
    
    parser = argparse.ArgumentParser(description='Model-free actor-critic for UC')
    parser.add_argument('--save_dir', type=str, required=True,
                        help='Directory to save models and results')
    parser.add_argument('--params_fn', type=str, required=True,
                        help='Filename for parameters')
    parser.add_argument('--ac_filename', type=str, required=False,
                        help="Filename for actor critic [.pt]", default=None)
    parser.add_argument('--num_procs', type=int, required=False,
                        help="Number of workers", default=1)
    parser.add_argument('--arma_params_fn', type=str, required=True,
                        help='Filename for ARMA parameters')
    parser.add_argument('--num_epochs', type=int, required=True, help='Number of training epochs')

    # TODO: Allow num_procs to be set either at command line OR in params.json

    args = parser.parse_args()
    
    # Allow 'none' to be passed as arg to state value or policy filenames
    if args.ac_filename == "none": args.ac_filename = None
    
    # Make results directory and files
    os.makedirs(args.save_dir, exist_ok=True)
    open(os.path.join(args.save_dir, 'epoch_mean_timesteps.txt'), 'w').close()
    open(os.path.join(args.save_dir, 'epoch_mean_rpt.txt'), 'w').close()
    open(os.path.join(args.save_dir, 'epoch_q25_timesteps.txt'), 'w').close()
    open(os.path.join(args.save_dir, 'epoch_q25_rpt.txt'), 'w').close()
    open(os.path.join(args.save_dir, 'epoch_q75_timesteps.txt'), 'w').close()
    open(os.path.join(args.save_dir, 'epoch_q75_rpt.txt'), 'w').close()
    open(os.path.join(args.save_dir, 'epoch_mean_entropy.txt'), 'w').close()
    open(os.path.join(args.save_dir, 'epoch_loss_v.txt'), 'w').close()
    open(os.path.join(args.save_dir, 'epoch_explained_variance.txt'), 'w').close()

    epoch_counter = torch.tensor([0])
    epoch_counter.share_memory_()
       
    # Read params
    with open(args.params_fn) as f:
        params = json.load(f)

    # Set random seeds
    params.update({'seed': params.get('seed', np.random.randint(0,10000)), # Choose a random seed and add to params if none exists
                   'num_epochs': args.num_epochs}) 
    np.random.seed(params.get('seed'))
    torch.manual_seed(params.get('seed'))
        
    # Read the ARMA parameters. 
    arma_params = json.load(open(args.arma_params_fn))
    params.update({'arma_params':arma_params})

    # Check if cuda is available:
    if torch.cuda.is_available():
         torch.set_default_tensor_type('torch.cuda.FloatTensor')
    
    # initialise environment and the shared networks 
    env = make_env(**params)
    shared_ac = ACAgent(env, **params)
    
    # update params with ARMA sigmas 
    params.update({'demand_sigma': env.arma_demand.sigma,
                   'wind_sigma': env.arma_wind.sigma})
    
    # Save params file to save_dir 
    with open(os.path.join(args.save_dir, 'params.json'), 'w') as fp:
        fp.write(json.dumps(params, sort_keys=True, indent=4))

    # Save arma params to save_dir
    with open(os.path.join(args.save_dir, 'arma_params.json'), 'w') as fp:
        fp.write(json.dumps(arma_params, sort_keys=True, indent=4))

    # Load weights if necessary
    if args.ac_filename is not None:
        shared_ac.load_state_dict(torch.load(args.ac_filename))
        print("Using trained actor critic network")
    else:
        print("Using untrained actor critic network")
    shared_ac.train()
    shared_ac.share_memory()
        
    processes = []
    for rank in range(args.num_procs):
        p = mp.Process(target=run_worker, args=(args.save_dir, rank, args.num_epochs, shared_ac, epoch_counter, params))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
        
    # Save policy network
    torch.save(shared_ac.state_dict(), os.path.join(args.save_dir, 'ac_final.pt'))
