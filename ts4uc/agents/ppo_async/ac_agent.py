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

from ts4uc.helpers import calculate_gamma
from ts4uc.helpers import process_observation

from rl4uc import processor

DEFAULT_ENTROPY_COEF = 0
DEFAULT_GAMMA = 0.95
DEFAULT_MINIBATCH_SIZE = 256
DEFAULT_NUM_EPOCHS = 10
DEFAULT_OBSERVE_FORECAST_ERRORS = False
DEFAULT_OBSERVATION_PROCESSOR = 'LimitedHorizonProcessor'

def discount_cumsum(x, discount):
    """
    Calculate discounted cumulative sum, from SpinningUp repo.
    """
    return signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]

class ActorBuffer: 
    def __init__(self, sub_obs_dim, act_dim, size, gamma, min_reward, lam=0.95):
        self.sub_obs_buf = np.zeros((size, sub_obs_dim), dtype=np.float32)
        self.act_buf = np.zeros((size, act_dim), dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.gamma = gamma
        self.lam = lam
        self.num_used, self.ep_start_idx, self.max_size = 0, 0, size
        
        self.min_reward = min_reward
        
    def store(self, sub_obs, act, logp, rew, val):

        self.num_used = min(self.num_used, self.max_size-1)
        
        self.sub_obs_buf[self.num_used] = sub_obs
        self.act_buf[self.num_used] = act
        self.logp_buf[self.num_used] = logp
        self.rew_buf[self.num_used] = rew
        self.val_buf[self.num_used] = val
        
        self.num_used +=1 
        
    def finish_ep(self, last_val=0):
        path_slice = slice(self.ep_start_idx, self.num_used)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)
        
        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = discount_cumsum(deltas, self.gamma * self.lam)
        
        self.ep_start_idx = self.num_used

    def finish_ep_new(self, ts, ep_rews, ep_vals, last_val=0):
        path_slice = slice(self.ep_start_idx, self.num_used)
        rews = np.append(ep_rews, last_val)
        vals = np.append(ep_vals, last_val)

        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        advs = discount_cumsum(deltas, self.gamma * self.lam)
        advs_rep = np.repeat(advs, ts)
        
        # Fix to avoid the actor buffer overflowing
        advs_rep = advs_rep[:self.num_used - self.ep_start_idx]

        self.adv_buf[path_slice] = advs_rep
        
        self.ep_start_idx = self.num_used
        
    def get(self):
        assert self.num_used == self.max_size    # buffer has to be full before you can get
        self.num_used, self.ep_start_idx = 0, 0
        # the next two lines implement the advantage normalization trick
        adv_mean, adv_std = np.mean(self.adv_buf), np.std(self.adv_buf)
        
        self.adv_buf = (self.adv_buf - adv_mean) / adv_std
        
        data = dict(sub_obs=self.sub_obs_buf, act=self.act_buf,
                    adv=self.adv_buf, logp=self.logp_buf)
        return {k: torch.as_tensor(v, dtype=torch.float32) for k,v in data.items()}
    
    def is_full(self):
        return self.num_used == self.max_size
    
class CriticBuffer: 
    def __init__(self, obs_dim, size, gamma, min_reward, lam=0.95):
        self.obs_buf = np.zeros((size, obs_dim), dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.gamma = gamma
        self.lam = lam
        self.num_used, self.ep_start_idx, self.max_size = 0, 0, size
        
        self.min_reward = min_reward
        
    def store(self, obs, rew):
        
        self.num_used = min(self.num_used, self.max_size-1)
        
        self.obs_buf[self.num_used] = obs
        self.rew_buf[self.num_used] = rew
        
        self.num_used +=1 
        
    def finish_ep(self, last_val=0):
        path_slice = slice(self.ep_start_idx, self.num_used)
        rews = np.append(self.rew_buf[path_slice], last_val)
        
        # the next line computes rewards-to-go, to be targets for the value function
        self.ret_buf[path_slice] = discount_cumsum(rews, self.gamma)[:-1]
        
        self.ep_start_idx = self.num_used
        
    def get(self):
        obs_buf = self.obs_buf[:self.num_used]
        ret_buf = self.ret_buf[:self.num_used]
        
        self.num_used, self.ep_start_idx = 0, 0
        
        data = dict(obs=obs_buf, ret=ret_buf)
        return {k: torch.as_tensor(v, dtype=torch.float32) for k,v in data.items()}
    
    def is_full(self):
        return self.num_used == self.max_size


class ACAgent(nn.Module):
    """
    Actor critic with dual input and output layers.
    
    The input dimensions are different for actor and critic. 
    """
    def __init__(self, env, **kwargs):
        super(ACAgent, self).__init__()
        self.entropy_target = None
        self.__dict__.update(kwargs)
        self.dispatch_freq_mins = env.dispatch_freq_mins
        self.forecast_horizon = int(kwargs.get('forecast_horizon_hrs', 12) * 60 / self.dispatch_freq_mins)
        self.env = env
        
        if kwargs.get('observation_processor', DEFAULT_OBSERVATION_PROCESSOR) == 'LimitedHorizonProcessor':
            self.obs_processor = processor.LimitedHorizonProcessor(env, forecast_horizon=self.forecast_horizon)
        elif kwargs.get('observation_processor', DEFAULT_OBSERVATION_PROCESSOR) == 'DayAheadProcessor':
            self.obs_processor = processor.DayAheadProcessor(env, forecast_errors=kwargs.get('observe_forecast_errors', DEFAULT_OBSERVE_FORECAST_ERRORS))
        else:
            raise ValueError(f"{kwargs.get('observation_processor')} is not a valid observation processor")
        
        self.n_in_ac = 2*env.action_size + self.obs_processor.obs_size
        self.n_in_cr = self.obs_processor.obs_size
        
        self.max_demand = env.max_demand # used for normalisation

        # Allowing for backwards compatibility with old policies
        if kwargs.get('num_nodes', None) != None:
            self.ac_arch = [self.num_nodes] * (self.num_layers + 1) 
            self.cr_arch = [self.num_nodes] * (self.num_layers + 1)

        else:
            self.ac_arch = [int(v) for v in self.ac_arch.split(',')]
            self.cr_arch = [int(v) for v in self.cr_arch.split(',')]
        
        if kwargs.get('credit_assignment_1hr') is None:
            self.gamma = DEFAULT_GAMMA
        else:
            self.gamma = calculate_gamma(kwargs.get('credit_assignment_1hr'), env.dispatch_freq_mins)
        print("Gamma: {}".format(self.gamma))

        # attrs = vars(self)
        # for item in attrs.items():
        #     print("%s: %s" % item)
        
        self.actor_buffer = ActorBuffer(self.n_in_ac, 1, self.buffer_size, self.gamma, env.min_reward)
        self.critic_buffer = CriticBuffer(self.n_in_cr, self.buffer_size, self.gamma, env.min_reward)

        # Create actor
        self.in_ac = nn.Linear(self.n_in_ac, self.ac_arch[0])
        self.ac_layers = nn.ModuleList([nn.Linear(self.ac_arch[i], self.ac_arch[i+1]) for i in range(len(self.ac_arch)-1)])
        self.output_ac = nn.Linear(self.ac_arch[-1], 2)

        # Create critic
        self.in_cr = nn.Linear(self.n_in_cr, self.cr_arch[0])
        self.cr_layers = nn.ModuleList([nn.Linear(self.cr_arch[i], self.cr_arch[i+1]) for i in range(len(self.cr_arch)-1)])
        self.output_cr = nn.Linear(self.cr_arch[-1], 1)

        self.activation = torch.relu

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    def get_action_scores(self, x):
        x = self.in_ac(x)
        x = self.activation(x)
        for l in self.ac_layers:
            x = l(x)
            x = self.activation(x)
        action_scores = self.output_ac(x)
        return action_scores

    def forward_ac(self, x):
        action_scores = self.get_action_scores(x)
        pi = Categorical(probs=F.softmax(action_scores, dim=action_scores.ndim-1))
        return pi
    
    def forward_cr(self, x):
        x = self.in_cr(x)
        x = self.activation(x)
        for l in self.cr_layers: 
            x = l(x)
            x = self.activation(x)
        return self.output_cr(x)
    
    def get_value(self, obs):
        x = self.obs_processor.process(obs)
        x = torch.as_tensor(x).float().to(self.device)
        return self.forward_cr(x), x

    def obs_to_input(self, env, obs):
        """
        Preprocessing of observation to create an input array x

        This includes pre-processing the states (clipping up/down times etc.) using self.obs_processor
        as well as adding the one-hot encoding and action draft to the beginning of the input vector

        The final array includes: [one hot encoding, draft action, state vector]
        """
        # Process observation, either extending or truncating the forecasts to correct length
        processed_obs = self.obs_processor.process(obs)

        # Init action with constraints
        draft_action = np.zeros(env.action_size, dtype=int)
        draft_action[np.where(env.must_on)[0]] = 1
        
        # Determine constrained gens
        constrained_gens = np.where(np.logical_or(env.must_on, env.must_off))
        unconstrained_gens = np.delete(np.arange(env.action_size), constrained_gens)

        # If using curtailment action, then predict curtailment action (last in action vector) first
        if env.curtailment: 
            unconstrained_gens = np.roll(unconstrained_gens, 1)
        
        one_hot_encoding = np.zeros(env.action_size)
        
        x = np.concatenate([one_hot_encoding, draft_action, processed_obs])

        # Convert state to tensor        
        x = torch.as_tensor(x).float().to(self.device)

        return x, unconstrained_gens, draft_action
        
    def generate_action(self, env, obs):
        """
        1. Determine constrained generators, init action and concatenate onto state
        2. Concatenate one-hot encoding onto state
        3. For all unconstrained generators:
            Forward pass with x[i]=1 (one-hot encoding).
            Sample action from softmax, change action[i]
            Change part of action part of state 
        """
        x, unconstrained_gens, action = self.obs_to_input(env, obs)
        
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
            x[env.action_size+idx] = a
        
        return action, sub_obs, sub_acts, log_probs
    
    def generate_multiple_actions_batched(self, env, obs, N_samples, threshold, lower_threshold=True):
        """
        Function that generates N actions sequentially. Replaces and combines
        the generate_multiple_actions and generate_action functions.
        """
        x, unconstrained_gens, action = self.obs_to_input(env, obs)

        # Repeat x N times
        xs = x.repeat(N_samples, 1)

        torch.manual_seed(self.test_seed) # Ensures that same set of actions are generated for a given observation
        
        for idx in unconstrained_gens:
            # Set one-hot encoding
            xs[:,idx] = 1

            # Forward pass
            pi = self.forward_ac(xs)

            # sample actions
            a = pi.sample()
                        
            # Change state tensors x 
            xs[:,env.action_size+idx] = a
            
            # Reset one-hot encoding
            xs[:,idx] = 0
        
        # Retrieve actions
        actions = xs[:,env.action_size:2*env.action_size].cpu().detach().numpy()

        # Get unique actions
        uniq, counts = np.unique(actions, axis=0, return_counts=True)

        # Delete for memory saving  
        del xs
        del actions

        # Determine actions meeting threshold
        action_freqs = counts/N_samples # frequency of actions
        threshold_mask = action_freqs >= threshold # actions whose frequency exceeds threshold      

        if lower_threshold: # Lower threshold if no actions meet the threshold
            best_action_mask = action_freqs.max() == action_freqs # actions sharing most counts 
            best_idx = np.logical_or(best_action_mask, threshold_mask) 
        else: 
            best_idx = threshold_idx
            
        best_actions = uniq[best_idx] 
        
        # Take only 1/threshold actions 
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
        obs, act, adv, logp_old = data['sub_obs'], data['act'], data['adv'], data['logp']

        # Policy loss
        pi = self.forward_ac(obs)
        logp = pi.log_prob(act[:,0])
        
        # Useful comparison: VPG
        # entropy = pi.entropy()
        # entropy_coef = 0
        # loss_pi = -(logp * (adv + entropy_coef * entropy )).mean() # useful comparison: VPG
        
        if self.entropy_target is not None:
            # Entropy penalty!
            # target_p = (branching_threshold)**(1./float(self.env.num_gen)) # target probability for single generator
            # entropy_target = -target_p * np.log2(target_p) - (1-target_p) * np.log2(1-target_p) # conversion to entropy 

            # Entropy penalty
            entropy_penalty = (pi.entropy().mean() - self.entropy_target)**2
            entropy_bonus = -entropy_penalty * self.entropy_coef

        else:
            # Vanilla entropy bonus
            entropy_bonus = self.entropy_coef * pi.entropy().mean()

        # PPO
        ratio = torch.exp(logp - logp_old)
        clip_adv = torch.clamp(ratio, 1-self.clip_ratio, 1+self.clip_ratio) * adv

        loss_pi = -(torch.min(ratio * adv, clip_adv)).mean() - entropy_bonus
        
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
        
    def update(self, worker_net, pi_optimizer, v_optimizer):
        
        
        if self.minibatch_size is not None:
            TRAIN_PI_ITERS = int((self.gradient_steps * self.buffer_size) / self.minibatch_size)
        else:
            TRAIN_PI_ITERS = self.gradient_steps
        
        data = worker_net.actor_buffer.get()
        for i in range(TRAIN_PI_ITERS):
            if self.minibatch_size is not None:
                minibatch = self.get_minibatch(data, self.buffer_size)
            else:
                minibatch=data

            pi_optimizer.zero_grad()
            loss_pi, entropy = self.compute_loss_pi(minibatch)
            loss_pi.backward()
            
            for param, shared_param in zip(worker_net.parameters(),
                                   self.parameters()):
                if shared_param.grad is not None:
                    break
                shared_param._grad = param.grad
                        
            pi_optimizer.step()

        data = worker_net.critic_buffer.get()
        
        if self.minibatch_size is not None:
            max_idx = len(list(data.items())[0][1]) # Number of entries in the critic buffer
            TRAIN_V_ITERS = int((self.gradient_steps * max_idx) / self.minibatch_size)
        else:
            TRAIN_V_ITERS = self.gradient_steps
            
        for i in range(TRAIN_V_ITERS):
            
            if self.minibatch_size is not None:
                minibatch = self.get_minibatch(data, max_idx)
            else:
                minibatch=data

            v_optimizer.zero_grad()
            loss_v, explained_variance = self.compute_loss_v(minibatch)
            loss_v.backward()
            for param, shared_param in zip(worker_net.parameters(),
                       self.parameters()):
                if shared_param.grad is not None:
                    break
                shared_param._grad = param.grad
            
            v_optimizer.step()

        return entropy, loss_v, explained_variance
