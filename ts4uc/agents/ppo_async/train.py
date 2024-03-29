#!/usr/bin/env python3
# -*- coding: utf-8 -*-
EPOCH_SAVE_INTERVAL = 10000

from rl4uc.environment import make_env
import torch
from torch.optim.lr_scheduler import LambdaLR
import torch.optim as optim
import torch.multiprocessing as mp
from ts4uc import helpers
mp.set_start_method('spawn', force=True)

from ts4uc.agents.ppo_async.ac_agent import ACAgent

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
    ep_rews, ep_vals, ep_sub_acts = [], [], []

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
        # reward = 1+reward/-env.min_reward
        reward = reward / -env.min_reward
        reward = np.log(-1/reward)

        # Clip to between (-10, 10)
        reward = reward.clip(-10, 10)

        # Update episode rewards and timesteps survived
        ep_reward += reward
        ep_rews.append(reward)
        ep_vals.append(value.detach().item())
        ep_sub_acts.append(len(sub_acts))
        ep_timesteps += 1

        local_ac.critic_buffer.store(obs_processed, reward)
        for idx in range(len(sub_obs)):
            local_ac.actor_buffer.store(sub_obs[idx], sub_acts[idx], log_probs[idx], reward, value)

        obs = new_obs

        if done:
            local_ac.actor_buffer.finish_ep_new(ts=ep_sub_acts, 
                                                ep_rews=ep_rews,
                                                ep_vals=ep_vals,
                                                last_val=0)
            local_ac.critic_buffer.finish_ep(last_val=0)
            
            rewards.append(ep_reward)
            timesteps.append(ep_timesteps)

            obs = env.reset()
            ep_reward, ep_timesteps = 0,0
            ep_rews, ep_vals, ep_sub_acts = [], [], []

        if local_ac.actor_buffer.is_full():
            if not done: 
                local_ac.actor_buffer.finish_ep_new(ts=ep_sub_acts, 
                                                    ep_rews=ep_rews,
                                                    ep_vals=ep_vals,
                                                    last_val=local_ac.get_value(obs)[0].detach().numpy())
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

def run_worker(save_dir, rank, num_epochs, shared_ac, epoch_counter, env_params, params):
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
    env = make_env(**env_params)
    
    local_ac = ACAgent(env, **params)
        
    while epoch_counter < num_epochs:
        
        epoch_counter += 1 
        print("Epoch: {}".format(epoch_counter.item()))

        # If using target entropy, then schedule
        if params.get('entropy_target', 0) != 0:
            shared_ac.entropy_coef = params.get('entropy_coef') * (3 * epoch_counter.item() / num_epochs )

        # Anneal entropy
        ANNEAL_ENTROPY = False
        if ANNEAL_ENTROPY: 
            new_coef = (1 - epoch_counter / num_epochs) * params['entropy_coef']
            print(new_coef)
            local_ac.entropy_coef = shared_ac.entropy_coef = new_coef

        
        local_ac.load_state_dict(shared_ac.state_dict())
                
        # Run an epoch, including updating the shared network
        run_epoch(save_dir, env, local_ac, shared_ac, pi_optimizer, v_optimizer, epoch_counter)
    
    # Record time taken
    time_taken = time.time() - start_time
    with open(os.path.join(save_dir, 'time_taken.txt'), 'w') as f:
        f.write(str(time_taken) + '\n')
        
if __name__ == "__main__":
    
    import argparse
    import json

    parser = argparse.ArgumentParser(description='Train PPO agent')
    parser.add_argument('--save_dir', type=str, required=True)
    parser.add_argument('--workers', type=int, required=False, default=1)
    parser.add_argument('--env_fn', type=str, required=True)
    parser.add_argument('--epochs', type=int, required=True)
    parser.add_argument('--buffer_size', type=int, required=False, default=2000)
    parser.add_argument('--seed', type=int, required=False, default=np.random.randint(1000000))

    # The following params will be used to setup the PPO agent
    parser.add_argument('--ac_learning_rate', type=float, required=False, default=3e-05)
    parser.add_argument('--cr_learning_rate', type=float, required=False, default=3e-04)
    parser.add_argument('--ac_arch', type=str, required=False, default="32,32,32")
    parser.add_argument('--cr_arch', type=str, required=False, default="32,32,32")
    parser.add_argument('--entropy_coef', type=float, required=False, default=0.01)
    parser.add_argument('--clip_ratio', type=float, required=False, default=0.1)
    parser.add_argument('--forecast_horizon_hrs', type=int, required=False, default=12)
    parser.add_argument('--credit_assignment_1hr', type=float, required=False, default=0.9)
    parser.add_argument('--minibatch_size', type=int, required=False, default=None)
    parser.add_argument('--update_epochs', type=int, required=False, default=4)
    parser.add_argument('--observation_processor', type=str, required=False, default='LimitedHorizonProcessor')
    parser.add_argument('--gradient_steps', type=int, required=False, default=10)
    parser.add_argument('--entropy_target', type=float, required=False, default=None)

    # Alternatively, pass a filename for trained weights and parameters, used to set network architectures.
    parser.add_argument('--ac_weights_fn', type=str, required=False, default=None)
    parser.add_argument('--ac_params_fn', type=str, required=False, default=None)
    
    args = parser.parse_args()

    # Make save dir
    os.makedirs(args.save_dir, exist_ok=True)

    # Load policy params and save them to the local directory. 
    params = vars(args)
    with open(os.path.join(args.save_dir, 'params.json'), 'w') as fp:
        fp.write(json.dumps(params, sort_keys=True, indent=4))

    # If training using a pre-defined AC networks --> overwrite archs
    if args.ac_params_fn is not None:
        ac_params = json.load(open(args.ac_params_fn))
        params.update({'ac_arch': ac_params['ac_arch'], 'cr_arch': ac_params['cr_arch']})
    
    # Load the env params and save them to save_dir
    env_params = json.load(open(args.env_fn))
    with open(os.path.join(args.save_dir, 'env_params.json'), 'w') as fp:
        fp.write(json.dumps(env_params, sort_keys=True, indent=4))
    
    # Make log files
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
        
    # Check if cuda is available:
    if torch.cuda.is_available():
         torch.set_default_tensor_type('torch.cuda.FloatTensor')
    
    # initialise environment and the shared networks 
    env = make_env(**env_params)
    shared_ac = ACAgent(env, **params)

    if args.ac_weights_fn is not None:
        print("********************Using pre-trained AC weights***********************")
        shared_ac.load_state_dict(torch.load(args.ac_weights_fn))

    shared_ac.train()
    shared_ac.share_memory()

        
    processes = []
    for rank in range(args.workers):
        p = mp.Process(target=run_worker, args=(args.save_dir, rank, args.epochs, shared_ac, epoch_counter, env_params, params))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
        
    # Save policy network
    torch.save(shared_ac.state_dict(), os.path.join(args.save_dir, 'ac_final.pt'))
