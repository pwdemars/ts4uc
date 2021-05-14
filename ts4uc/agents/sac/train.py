import torch
import torch.optim as optim
import torch.multiprocessing as mp
import numpy as np
import time
import scipy.signal as signal
import copy
from collections import namedtuple

from rl4uc.environment import make_env
from ts4uc.agents.sac import sac
from ts4uc.agents.ppo.train import Logger
from ts4uc import helpers
import pandas as pd 
import os

device = "cpu"
mp.set_start_method('spawn', True)

MsgUpdateRequest = namedtuple('MsgUpdateRequest', ['agent', 'update'])

class ReplayBuffer:

    def __init__(self, max_size, obs_dim=None):

        self.max_size = max_size
        self.obs_buf = torch.zeros(
            (max_size, obs_dim)).to(device).share_memory_()
        self.act_buf = torch.zeros(
            max_size, 1).to(device).share_memory_()
        self.rew_buf = torch.zeros(
            max_size, 1).to(device).share_memory_()
        self.next_obs_buf = torch.zeros(
            max_size, obs_dim).to(device).share_memory_()
        self.done_buf = torch.zeros(
            max_size, 1).to(device).share_memory_() 
        self.discount_buf = torch.zeros(
            max_size, 1).to(device).share_memory_() 
        self.num_used = torch.zeros(1, dtype=int).share_memory_()

    def reset(self):
        self.num_used.zero_()

    def store(self, obs=None, act=None, rew=None, next_obs=None, done=None, discount=None):

        n = self.num_used.item() % self.max_size

        if obs != None:
            self.obs_buf[n] = obs
        if act != None:
            self.act_buf[n] = act
        if rew != None:
            self.rew_buf[n] = rew
        if next_obs != None:
            self.next_obs_buf[n] = next_obs
        if done != None:
            self.done_buf[n] = done
        if discount != None:
            self.dsicount_buf[n] = discount

        self.num_used += 1

    def get(self, minibatch_size=None):

        if minibatch_size != None:
            idx = np.random.choice(np.arange(self.max_size), size=minibatch_size, replace=False)
        else:
            idx = np.arange(self.max_size)

        data = dict(obs=self.obs_buf[idx], act=self.act_buf[idx],
                    rew=self.rew_buf[idx], next_obs=self.next_obs_buf[idx],
                    done=self.done_buf[idx], discount=self.dsicount_buf[idx])

        return {k: torch.as_tensor(v, dtype=torch.float32) for k,v in data.items()}

class Worker(mp.Process):
    """
    Worker class
    """
    def __init__(self, worker_id, env, pipe, policy, gamma, lam, 
                num_epochs, steps_per_epoch, buf, logger):

        mp.Process.__init__(self, name=worker_id)


        self.worker_id = worker_id
        self.policy = policy
        self.env = copy.deepcopy(env)
        self.gamma = gamma
        self.lam = lam
        self.pipe = pipe
        self.num_epochs = num_epochs
        self.steps_per_epoch = steps_per_epoch
        self.buf = buf
        self.logger = logger

    def run(self):

        for epoch in range(self.num_epochs):

            all_ep_rewards, all_ep_timesteps = self.run_epoch()

            self.logger.log(reward=np.mean(all_ep_rewards),
                            std_reward=np.std(all_ep_rewards),
                            q25_reward=np.quantile(all_ep_rewards, 0.25),
                            q75_reward=np.quantile(all_ep_rewards, 0.75),
                            timesteps=np.mean(all_ep_timesteps),
                            std_timesteps=np.std(all_ep_timesteps),
                            q25_timesteps=np.quantile(all_ep_timesteps, 0.25),
                            q75_timesteps=np.quantile(all_ep_timesteps, 0.75),
                            worker_id=int(self.worker_id),
                            epoch=epoch)

            msg = MsgUpdateRequest(int(self.worker_id), True)
            self.pipe.send(msg)
            msg = self.pipe.recv()

    def run_epoch(self):

        all_ep_timesteps = []
        all_ep_rewards = []

        acts = []
        obss = []
        next_obss = []
        rews = []
        dones = []

        while sum(all_ep_timesteps) <= self.steps_per_epoch: 

            obs = self.env.reset() 

            ep_timesteps = 0
            ep_reward = 0
            unscaled_ep_rewards = []
            ep_rewards = []
            ep_values = []

            done = False

            while not done:
                # Choose action with actor
                a, sub_obs, sub_acts, _ = self.policy.generate_action(self.env, obs)

                # Step environment 
                obs, reward, done = self.env.step(a)

                # Record unscaled reward
                unscaled_ep_rewards.append(reward)

                # Transform the reward
                reward = 1+reward/-self.env.min_reward
                # reward = (reward - self.policy.mean_reward) / self.policy.std_reward
                reward = reward.clip(-10, 10)

                # Update episode rewards and timesteps
                ep_timesteps += 1

                for i in range(len(sub_acts)):

                    if ((i+1) == len(sub_acts)):
                        next_obs, _ = self.policy.obs_with_constraints(self.env, obs)
                        discount = self.gamma
                    else:
                        next_obs = sub_obs[i+1]
                        discount = 1
                        

                    self.buf.store(obs=sub_obs[i],
                                   act=sub_acts[i],
                                   rew=reward,
                                   next_obs=next_obs,
                                   done=done,
                                   discount=discount)

            all_ep_rewards.append(sum(unscaled_ep_rewards))
            all_ep_timesteps.append(ep_timesteps)

        return all_ep_rewards, all_ep_timesteps


def train(save_dir,
          timesteps, 
          num_workers, 
          steps_per_epoch, 
          buffer_size,
          env_params, 
          policy_params,
          gamma,
          lam):

    # Total number of epochs (updates)
    num_epochs = int(timesteps / steps_per_epoch)

    # Number of timesteps each worker should gather per epoch
    worker_steps_per_epoch = int(steps_per_epoch / num_workers)


    env = make_env(**env_params)
    policy = sac.SACAgent(env, **policy_params).share_memory()

    # The actor buffer will typically take more entries than the critic buffer,
    # because it records sub-actions. Hence there is usually more than one entry
    # per timestep. Here we set the size to be the max possible.
    buf = ReplayBuffer(max_size=buffer_size, obs_dim=policy.n_in)

    logger = Logger(num_epochs, num_workers, steps_per_epoch)


    # Worker update requests
    update_request = [False]*num_workers
    epoch_counter = 0

    workers = []
    pipes = []

    for worker_id in range(num_workers):
        p_start, p_end = mp.Pipe()
        worker = Worker(worker_id=str(worker_id),
                        env=env,
                        policy=policy,
                        pipe=p_end,
                        buf=buf,
                        logger=logger,
                        num_epochs=num_epochs,
                        steps_per_epoch=worker_steps_per_epoch,
                        gamma=gamma,
                        lam=lam)
        worker.start()
        workers.append(worker)
        pipes.append(p_start)

    start_time = time.time()

    # starting training loop
    while epoch_counter < num_epochs:
        for i, conn in enumerate(pipes):
            if conn.poll():
                msg = conn.recv()

                # if agent is waiting for network update
                if type(msg).__name__ == "MsgUpdateRequest":
                    update_request[i] = True
                    if False not in update_request:

                        print()
                        print("Epoch: {}".format(epoch_counter))
                        print("Updating")
                        print("--------")


                        policy.update(buf)

                        epoch_counter += 1
                        update_request = [False]*num_workers
                        msg = epoch_counter

                        # periodically save the logs
                        if (epoch_counter + 1) % 10 == 0:
                            logger.save_to_csv(os.path.join(save_dir, 'logs.csv'))

                        # send to signal subprocesses to continue
                        for pipe in pipes:
                            pipe.send(msg)

    time_taken = time.time() - start_time

    logger.save_to_csv(os.path.join(save_dir, 'logs.csv'))
    torch.save(policy.state_dict(), os.path.join(save_dir, 'ac_final.pt'))

    # Record training time
    with open(os.path.join(save_dir, 'time_taken.txt'), 'w') as f:
        f.write(str(time_taken) + '\n')


if __name__ == "__main__":
    
    import argparse
    import json

    
    parser = argparse.ArgumentParser(description='Train SAC agent')
    parser.add_argument('--save_dir', type=str, required=True)
    parser.add_argument('--workers', type=int, required=False, default=1)
    parser.add_argument('--num_gen', type=int, required=True)
    parser.add_argument('--timesteps', type=int, required=True)
    parser.add_argument('--steps_per_epoch', type=int, required=True)
    parser.add_argument('--buffer_size', type=int, required=True)

    # The following params will be used to setup the agent
    parser.add_argument('--ac_learning_rate', type=float, required=False, default=3e-05)
    parser.add_argument('--cr_learning_rate', type=float, required=False, default=3e-04)
    parser.add_argument('--ent_learning_rate', type=float, required=False, default=1e-02)
    parser.add_argument('--num_layers', type=int, required=False, default=3)
    parser.add_argument('--num_nodes', type=int, required=False, default=32)
    parser.add_argument('--target_entropy', type=float, required=False, default=0.2)
    parser.add_argument('--forecast_horizon_hrs', type=int, required=False, default=12)
    parser.add_argument('--credit_assignment_1hr', type=float, required=False, default=0.9)
    parser.add_argument('--minibatch_size', type=int, required=False, default=None)
    parser.add_argument('--gradient_steps', type=int, required=False, default=4)
    parser.add_argument('--observation_processor', type=str, required=False, default='LimitedHorizonProcessor')


    args = parser.parse_args()

    # Make save dir
    os.makedirs(args.save_dir, exist_ok=True)

    # Load policy params and save them to the local directory. 
    policy_params = vars(args)
    with open(os.path.join(args.save_dir, 'params.json'), 'w') as fp:
        fp.write(json.dumps(policy_params, sort_keys=True, indent=4))
    
    # Load the env params and save them to save_dir
    env_params = helpers.retrieve_env_params(args.num_gen)
    with open(os.path.join(args.save_dir, 'env_params.json'), 'w') as fp:
        fp.write(json.dumps(env_params, sort_keys=True, indent=4))

    # Calculate gamma 
    gamma = helpers.calculate_gamma(policy_params['credit_assignment_1hr'], env_params['dispatch_freq_mins'])

    # Lambda for GAE 
    lam = 0.95

    train(save_dir=args.save_dir,
          timesteps=args.timesteps,
          num_workers=args.workers,
          steps_per_epoch=args.steps_per_epoch,
          buffer_size=args.buffer_size,
          env_params=env_params,
          policy_params=policy_params,
          gamma=gamma,
          lam=lam)



