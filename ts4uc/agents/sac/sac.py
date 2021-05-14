import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

import time
import numpy as np
import scipy.signal as signal

from ts4uc.helpers import process_observation, calculate_gamma, mean_std_reward

from rl4uc import processor

def polyak_update(params, target_params, tau):
    """
    Perform a Polyak average update on ``target_params`` using ``params``:
    target parameters are slowly updated towards the main parameters.
    ``tau``, the soft update coefficient controls the interpolation:
    ``tau=1`` corresponds to copying the parameters to the target ones whereas nothing happens when ``tau=0``.
    The Polyak update is done in place, with ``no_grad``, and therefore does not create intermediate tensors,
    or a computation graph, reducing memory cost and improving performance.  We scale the target params
    by ``1-tau`` (in-place), add the new weights, scaled by ``tau`` and store the result of the sum in the target
    params (in place).
    See https://github.com/DLR-RM/stable-baselines3/issues/93

    :param params: parameters to use to update the target params
    :param target_params: parameters to update
    :param tau: the soft update coefficient ("Polyak update", between 0 and 1)
    """
    with torch.no_grad():
        # zip does not raise an exception if length of parameters does not match.
        for param, target_param in zip(params, target_params):
            target_param.data.mul_(1 - tau)
            torch.add(target_param.data, param.data, alpha=tau, out=target_param.data)

class Critic(nn.Module):
    def __init__(self, env, **kwargs):
        super(Critic, self).__init__()

        self.n_in = kwargs.get('obs_size') + 1 
        self.num_nodes = int(kwargs.get('num_nodes'))
        self.num_layers = int(kwargs.get('num_layers'))

        self.in_layer = nn.Linear(self.n_in, self.num_nodes)
        self.layers = nn.ModuleList([nn.Linear(self.num_nodes, self.num_nodes) for i in range(self.num_layers)])
        self.out_layer = nn.Linear(self.num_nodes, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=kwargs.get('cr_learning_rate'))

    def forward(self, obs, act):
        x = torch.cat([obs, act], dim=1)
        x = self.in_layer(x)
        x = F.relu(x)
        for l in self.layers: 
            x = l(x)
            x = F.relu(x)
        return self.out_layer(x)

class SACAgent(nn.Module):

    def __init__(self, env, **kwargs):

        super(SACAgent, self).__init__()

        self.dispatch_freq_mins = env.dispatch_freq_mins
        self.forecast_horizon = int(kwargs.get('forecast_horizon_hrs') * 60 / self.dispatch_freq_mins)
        self.env = env
        
        if kwargs.get('observation_processor') == 'LimitedHorizonProcessor':
            self.obs_processor = processor.LimitedHorizonProcessor(env, forecast_horizon=self.forecast_horizon)
        elif kwargs.get('observation_processor') == 'DayAheadProcessor':
            self.obs_processor = processor.DayAheadProcessor(env, forecast_errors=kwargs.get('observe_forecast_errors', False))
        else:
            raise ValueError(f"{kwargs.get('observation_processor')} is not a valid observation processor")
        
        self.n_in = 2*env.num_gen + self.obs_processor.obs_size
        
        self.num_nodes = int(kwargs.get('num_nodes'))
        self.num_layers = int(kwargs.get('num_layers'))

        n_critics = 2
        self.critics = [Critic(env, obs_size=self.n_in, **kwargs) for i in range(n_critics)]
        self.critics_target = []
        for cr in self.critics:
            cr_target = Critic(env, obs_size=self.n_in, **kwargs)
            cr_target.load_state_dict(cr.state_dict())
            self.critics_target.append(cr_target)

        self.in_ac = nn.Linear(self.n_in, self.num_nodes)
        self.ac_layers = nn.ModuleList([nn.Linear(self.num_nodes, self.num_nodes) for i in range(self.num_layers)])
        self.output_ac = nn.Linear(self.num_nodes, 2)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.test_seed = kwargs.get('test_seed')

        self.gradient_steps = kwargs.get('gradient_steps')
        self.minibatch_size = kwargs.get('minibatch_size', None)
        self.ent_coef = "auto"
        self.target_entropy = kwargs.get('target_entropy')

        self.pi_optimizer = optim.Adam(self.parameters(), lr=kwargs.get('ac_learning_rate'))


        # Default initial value of ent_coef when learned
        init_value = 1.0
        if "_" in self.ent_coef:
            init_value = float(self.ent_coef.split("_")[1])
            assert init_value > 0.0, "The initial value of ent_coef must be greater than 0"

        # Note: we optimize the log of the entropy coeff which is slightly different from the paper
        # as discussed in https://github.com/rail-berkeley/softlearning/issues/37
        self.log_ent_coef = torch.log(torch.ones(1, device=self.device) * init_value).requires_grad_(True)
        self.ent_coef_optimizer = torch.optim.Adam([self.log_ent_coef], lr=kwargs.get('ent_learning_rate'))
        self.target_update_interval = 1
        self.gamma = calculate_gamma(kwargs.get('credit_assignment_1hr'), env.dispatch_freq_mins)
        self.tau = 0.005

        # self.mean_reward, self.std_reward = mean_std_reward(self.env)

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


    def obs_with_constraints(self, env, obs):

        if env.episode_timestep == (env.episode_length - 1):
            return torch.zeros(self.n_in), None

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

        return x, action


    def generate_action(self, env, obs):
        """
        1. Determine constrained generators, init action and concatenate onto state
        2. Concatenate one-hot encoding onto state
        3. For all unconstrained generators:
            Forward pass with x[i]=1 (one-hot encoding).
            Sample action from softmax, change action[i]
            Change part of action part of state 
        """
        x, action = self.obs_with_constraints(env, obs)
        
        # Init log_probs
        log_probs = []
        sub_obs = []
        sub_acts = []
        
        # Init entropys
        entropys = []

        constrained_gens = np.where(np.logical_or(env.must_on, env.must_off))
        unconstrained_gens = np.delete(np.arange(env.num_gen), constrained_gens)
        
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

    def update(self, buf):

        ent_coef_losses, ent_coefs = [], []
        actor_losses, critic_losses = [], []

        for gradient_step in range(self.gradient_steps):

            minibatch = buf.get(minibatch_size=self.minibatch_size)

            obs, act, rew, next_obs, dones = minibatch['obs'], minibatch['act'], minibatch['rew'], minibatch['next_obs'], minibatch['done']

            print("Mean reward: {}, std reward: {}".format(rew.mean(), rew.std()))

            pi = self.forward_ac(obs)
            log_prob = pi.log_prob(act)

            # Important: detach the variable from the graph
            # so we don't change it with other losses
            # see https://github.com/rail-berkeley/softlearning/issues/60
            ent_coef = torch.exp(self.log_ent_coef.detach())
            print("Entropy coef: ", ent_coef)
            ent_coef_loss = -(self.log_ent_coef * (log_prob + self.target_entropy).detach()).mean()
            ent_coef_losses.append(ent_coef_loss.item())

            ent_coefs.append(ent_coef.item())

            if ent_coef_loss is not None:
                self.ent_coef_optimizer.zero_grad()
                ent_coef_loss.backward()
                self.ent_coef_optimizer.step()

            with torch.no_grad():
                # Select action according to policy
                next_pi = self.forward_ac(next_obs)
                next_actions = next_pi.sample()
                print("Entropy: {}".format(next_pi.entropy().mean()))
                next_log_prob = next_pi.log_prob(next_actions)
                # Compute the next Q values: min over all critics targets
                next_q_values = [critic.forward(obs, act) for critic in self.critics_target]
                next_q_values = torch.cat(next_q_values, dim=1)
                next_q_values, _ = torch.min(next_q_values, dim=1, keepdim=True)
                # add entropy term
                next_q_values = next_q_values - ent_coef * next_log_prob.reshape(-1, 1)
                # td error + entropy term
                # TODO: this is not correct for my problem. Sub-actions should have the same Q-values 
                # Discounting should take this into account
                target_q_values = rew + (1 - dones) * self.gamma * next_q_values 

            # Get current Q-values estimates for each critic network
            # using action from the replay buffer
            current_q_values = [critic.forward(obs, act) for critic in self.critics]

            # Compute critic loss
            critic_loss = 0.5 * sum([F.mse_loss(current_q, target_q_values) for current_q in current_q_values])
            critic_losses.append(critic_loss.item())


            # Optimize the critic
            [cr.optimizer.zero_grad() for cr in self.critics]
            critic_loss.backward()
            [cr.optimizer.step() for cr in self.critics]

            # Compute actor loss
            # Alternative: actor_loss = th.mean(log_prob - qf1_pi)
            # Mean over all critic networks
            q_values_pi = [critic.forward(obs, act) for critic in self.critics]
            q_values_pi = torch.cat(q_values_pi, dim=1)
            min_qf_pi, _ = torch.min(q_values_pi, dim=1, keepdim=True)
            actor_loss = (ent_coef * log_prob - min_qf_pi).mean()
            actor_losses.append(actor_loss.item())
            print("Actor loss: {}".format(actor_loss))

            # Optimize the actor
            self.pi_optimizer.zero_grad()
            actor_loss.backward()
            self.pi_optimizer.step()

            # Update target networks
            if gradient_step % self.target_update_interval == 0:
                for cr, target_cr in zip(self.critics, self.critics_target):
                    polyak_update(cr.parameters(), target_cr.parameters(), self.tau)

