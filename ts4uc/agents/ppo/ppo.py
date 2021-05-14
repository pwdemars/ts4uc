#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  4 16:19:08 2020

@author: patrickdemars
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

import time
import numpy as np
import scipy.signal as signal

from ts4uc.helpers import process_observation

from rl4uc import processor

class PPOAgent(nn.Module):
    """
    Actor critic with dual input and output layers.
    
    The input dimensions are different for actor and critic. 
    """
    def __init__(self, env, **kwargs):
        super(PPOAgent, self).__init__()
        self.dispatch_freq_mins = env.dispatch_freq_mins
        self.forecast_horizon = int(kwargs.get('forecast_horizon_hrs') * 60 / self.dispatch_freq_mins)
        self.env = env
        
        if kwargs.get('observation_processor') == 'LimitedHorizonProcessor':
            self.obs_processor = processor.LimitedHorizonProcessor(env, forecast_horizon=self.forecast_horizon)
        elif kwargs.get('observation_processor') == 'DayAheadProcessor':
            self.obs_processor = processor.DayAheadProcessor(env, forecast_errors=kwargs.get('observe_forecast_errors', False))
        else:
            raise ValueError(f"{kwargs.get('observation_processor')} is not a valid observation processor")
        
        self.n_in_ac = 2*env.num_gen + self.obs_processor.obs_size
        self.n_in_cr = self.obs_processor.obs_size
        
        self.num_nodes = int(kwargs.get('num_nodes'))
        self.num_layers = int(kwargs.get('num_layers'))
        self.max_demand = env.max_demand # used for normalisation
    
        self.num_epochs = int(kwargs.get('update_epochs'))
        self.clip_ratio = float(kwargs.get('clip_ratio'))
        self.entropy_coef = self.entropy_coef_init = float(kwargs.get('entropy_coef'))
        self.minibatch_size = kwargs.get('minibatch_size')
        
        self.in_ac = nn.Linear(self.n_in_ac, self.num_nodes)
        self.in_cr = nn.Linear(self.n_in_cr, self.num_nodes)
        
        self.ac_layers = nn.ModuleList([nn.Linear(self.num_nodes, self.num_nodes) for i in range(self.num_layers)])
        self.cr_layers = nn.ModuleList([nn.Linear(self.num_nodes, self.num_nodes) for i in range(self.num_layers)])

        
        self.output_ac = nn.Linear(self.num_nodes, 2)
        self.output_cr = nn.Linear(self.num_nodes, 1)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.test_seed = kwargs.get('test_seed')

        print("Entropy coef: {}".format(self.entropy_coef),
              "Clip ratio: {}".format(self.clip_ratio),
              "Layers: {}".format(self.num_layers),
              "Nodes: {}".format(self.num_nodes),
              "Minibatch size: {}".format(self.minibatch_size),
              "Update epochs: {}".format(self.num_epochs))

        # self.mean_reward, self.std_reward = self.mean_std_reward()

    def mean_std_reward(self, N=10000):
        """
        Get estimates for the mean and std. of the rewards, so they can be normalised
        """
        self.env.reset()
        rewards = []
        for i in range(N):
            o, r, d = self.env.step(np.random.randint(2, size=5))
            rewards.append(r)
            if d: 
                self.env.reset()
        self.env.reset()
        return np.mean(rewards), np.std(rewards)

    def get_action_scores(self, x):
        x = self.in_ac(x)
        x = F.relu(x)
        for l in self.ac_layers:
            x = l(x)
            x = F.relu(x)
        action_scores = self.output_ac(x)
        return action_scores

    def forward_ac(self, x):
        action_scores = self.get_action_scores(x)
        pi = Categorical(probs=F.softmax(action_scores, dim=action_scores.ndim-1))
        return pi
    
    def forward_cr(self, x):
        x = self.in_cr(x)
        x = F.relu(x)
        for l in self.cr_layers: 
            x = l(x)
            x = F.relu(x)
        return self.output_cr(x)
    
    def get_value(self, obs):
        x = self.obs_processor.process(obs)
        x = torch.as_tensor(x).float().to(self.device)
        return self.forward_cr(x), x
        
    def generate_action(self, env, obs):
        """
        1. Determine constrained generators, init action and concatenate onto state
        2. Concatenate one-hot encoding onto state
        3. For all unconstrained generators:
            Forward pass with x[i]=1 (one-hot encoding).
            Sample action from softmax, change action[i]
            Change part of action part of state 
        """
        x = self.obs_processor.process(obs)
 
        # Init action with constraints
        action = np.zeros(env.num_gen, dtype=int)
        action[np.where(env.must_on)[0]] = 1
        
        # Determine constrained gens
        constrained_gens = np.where(np.logical_or(env.must_on, env.must_off))
        unconstrained_gens = np.delete(np.arange(env.num_gen), constrained_gens)
        
        # Append action
        x = np.append(action, x)
        
        # Append one-hot encoding
        x = np.append(np.zeros(env.num_gen), x)

        # Convert state to tensor        
        x = torch.as_tensor(x).float().to(self.device)
        
        # Init log_probs
        log_probs = []
        sub_obs = []
        sub_acts = []
        
        # Init entropys
        entropys = []
        
        for idx in unconstrained_gens:
            
            x_g = x.clone()
            
            # Set one hot encoding
            x_g[idx] = 1
            
            # Forward pass
            pi = self.forward_ac(x_g)
            
            # Sample action
            a = pi.sample()
            
            # Update log_prob
            log_prob = pi.log_prob(a)
            log_probs.append(log_prob)
            entropys.append(pi.entropy())
            
            # Add to data buffer 
            sub_obs.append(x_g)
            sub_acts.append(a)

            # Change action
            action[idx] = a
            
            # Change state tensor x 
            x[env.num_gen+idx] = a
        
        return action, sub_obs, sub_acts, log_probs
    
    def generate_multiple_actions_batched(self, env, obs, N_samples, threshold, lower_threshold=True):
        """
        Function that generates N actions sequentially. Replaces and combines
        the generate_multiple_actions and generate_action functions.
        """
        # Process observation, either extending or truncating the forecasts to correct length
        x = self.obs_processor.process(obs)

        # Init action with constraints
        action = np.zeros(env.num_gen, dtype=int)
        action[np.where(env.must_on)[0]] = 1
        
        # Determine constrained gens
        constrained_gens = np.where(np.logical_or(env.must_on, env.must_off))
        unconstrained_gens = np.delete(np.arange(env.num_gen), constrained_gens)
        
        # Append action
        x = np.append(action, x)
        
        # Append one-hot encoding
        x = np.append(np.zeros(env.num_gen), x)

        # Convert state to tensor        
        x = torch.as_tensor(x).float().to(self.device)

        # Repeat x N times
        xs = x.repeat(N_samples, 1)

        torch.manual_seed(self.test_seed) # Ensures that same set of actions are generated for a given observation
        
        for idx in unconstrained_gens:
            # Set one-hot encoding
            xs[:,idx] = 1

            # Forward pass
            pi = self.forward_ac(xs)

            # sample actions
            a = pi.sample()
                        
            # Change state tensors x 
            xs[:,env.num_gen+idx] = a
            
            # Reset one-hot encoding
            xs[:,idx] = 0
        
        # Retrieve actions
        actions = xs[:,env.num_gen:2*env.num_gen].cpu().detach().numpy()

        # Get unique actions
        uniq, counts = np.unique(actions, axis=0, return_counts=True)

        # Delete for memory saving  
        del xs
        del actions

        # Determine actions meeting threshold
        action_freqs = counts/N_samples # frequency of actions
        threshold_mask = action_freqs >= threshold # actions whose frequency exceeds threshold        

        if lower_threshold: # Lower threshold if no actions meet the threshold
            best_action_mask = action_freqs.max() == action_freqs # actions sharing most counts 
            best_idx = np.logical_or(best_action_mask, threshold_mask) 
        else: 
            best_idx = threshold_idx
            
        best_actions = uniq[best_idx] 
        
        # Take only 1/threshold actions 
        max_actions = int(1/threshold)
        best_actions = best_actions[:max_actions]
        # print("Actions meeting threshold: {}; total actions: {}".format(threshold_mask.sum(), len(best_actions)))

        # Create action dictionary
        action_dict = {}
        for a in best_actions:
            # Convert action to bit string
            action_id = ''.join(str(int(i)) for i in a)
            # Convert bit string to int
            action_id = int(action_id, 2)
            action_dict[action_id] = a

        return action_dict, 0

    def compute_loss_pi(self, data):
        """
        Function from spinningup implementation to calcualte the PPO loss. 
        """
        obs, act, adv, logp_old = data['obs'], data['act'], data['adv'], data['logp']

        # Policy loss
        pi = self.forward_ac(obs)
        # print(act)
        logp = pi.log_prob(act)
        
        # PPO
        ratio = torch.exp(logp - logp_old)
        clip_adv = torch.clamp(ratio, 1-self.clip_ratio, 1+self.clip_ratio) * adv
        loss_pi = -(torch.min(ratio * adv, clip_adv)).mean() - self.entropy_coef*pi.entropy().mean()
        
        # compute entropy
        entropy = pi.entropy()
        
        return loss_pi, entropy
                    
    def compute_loss_v(self, data):
        # TODO: make sure that the correct network (worker or global) is computing value 
        obs, ret = data['obs'], data['ret']
        pred = self.forward_cr(obs)
        explained_variance = 1 - (torch.var(ret - pred))/torch.var(ret)
        return ((pred - ret)**2).mean(), explained_variance

    def get_minibatch(self, data, max_idx):
        idx = np.random.choice(np.arange(max_idx), size=self.minibatch_size, replace=False)
        batch = {}
        for k in data.keys():
            batch.update({k: data[k][idx]})
        return batch

    def update(self, actor_buf, critic_buf, pi_optimizer, v_optimizer):
        
        if self.minibatch_size is not None:
            TRAIN_PI_ITERS = int((self.num_epochs * actor_buf.num_used) / self.minibatch_size)
        else:
            TRAIN_PI_ITERS = self.num_epochs
        
        data = actor_buf.get()
        print("Actor buffer num_used: {}".format(actor_buf.num_used))

        for i in range(TRAIN_PI_ITERS):
            if self.minibatch_size is not None:
                minibatch = self.get_minibatch(data, actor_buf.num_used)
            else:
                minibatch=data

            pi_optimizer.zero_grad()
            loss_pi, entropy = self.compute_loss_pi(minibatch)
            loss_pi.backward()
            pi_optimizer.step()

        data = critic_buf.get()
        
        if self.minibatch_size is not None:
            TRAIN_V_ITERS = int((self.num_epochs * critic_buf.num_used) / self.minibatch_size)
        else:
            TRAIN_V_ITERS = self.num_epochs
            
        for i in range(TRAIN_V_ITERS):
            
            if self.minibatch_size is not None:
                minibatch = self.get_minibatch(data, critic_buf.num_used)
            else:
                minibatch=data

            v_optimizer.zero_grad()
            loss_v, explained_variance = self.compute_loss_v(minibatch)
            loss_v.backward()            
            v_optimizer.step()

        return entropy, loss_v, explained_variance

        
