import torch
import torch.optim as optim
import torch.multiprocessing as mp
import numpy as np
import time
import scipy.signal as signal
import copy
from collections import namedtuple

from rl4uc.environment import make_env
from ts4uc.agents.ppo.ppo import PPOAgent
from ts4uc import helpers
import pandas as pd 
import os

device = "cpu"
mp.set_start_method('spawn', True)

MsgUpdateRequest = namedtuple('MsgUpdateRequest', ['agent', 'update'])

class NewLogger:

	def __init__(self, num_epochs, num_workers, steps_per_epoch, *args):

		self.num_epochs = num_epochs
		self.steps_per_epoch = steps_per_epoch
		self.num_workers = num_workers
		self.log = {}
		for key in args:
			self.log[key] = torch.zeros((num_epochs, num_workers)).share_memory_()

	def store(self, key, value, epoch, worker_id):

		self.log[key][epoch, worker_id] = value

	def save_to_csv(self, fn):

		df = pd.DataFrame()
		for key in self.log:
			df[key] = self.log[key].numpy().mean(axis=1)
		df['epoch'] = np.arange(self.num_epochs)
		df['timestep'] = df['epoch'] * self.steps_per_epoch
		df.to_csv(fn, index=False)

class Logger:

	def __init__(self, num_epochs, num_workers, steps_per_epoch):

		self.num_epochs = num_epochs
		self.steps_per_epoch = steps_per_epoch
		self.num_workers = num_workers
		self.reward = torch.zeros((num_epochs, num_workers)).share_memory_()
		self.std_reward = torch.zeros((num_epochs, num_workers)).share_memory_()
		self.q25_reward = torch.zeros((num_epochs, num_workers)).share_memory_()
		self.q75_reward = torch.zeros((num_epochs, num_workers)).share_memory_()
		
		self.timesteps = torch.zeros((num_epochs, num_workers)).share_memory_()
		self.std_timesteps = torch.zeros((num_epochs, num_workers)).share_memory_()
		self.q25_timesteps = torch.zeros((num_epochs, num_workers)).share_memory_()
		self.q75_timesteps = torch.zeros((num_epochs, num_workers)).share_memory_()



	def log(self, reward, std_reward, q25_reward, q75_reward, 
			timesteps, std_timesteps, q25_timesteps, q75_timesteps,
			epoch, worker_id):

		self.reward[epoch, worker_id] = reward
		self.std_reward[epoch, worker_id] = std_reward
		self.q25_reward[epoch, worker_id] = q25_reward
		self.q75_reward[epoch, worker_id] = q75_reward

		self.timesteps[epoch, worker_id] = timesteps
		self.std_timesteps[epoch, worker_id] = std_timesteps
		self.q25_timesteps[epoch, worker_id] = q25_timesteps
		self.q75_timesteps[epoch, worker_id] = q75_timesteps


	def save_to_csv(self, fn):
		
		# Training epoch and timesteps observed
		epoch = np.arange(self.num_epochs)
		timestep = epoch * self.steps_per_epoch

		avg_reward = self.reward.numpy().mean(axis=1)
		std_reward = self.std_reward.numpy().mean(axis=1)
		q25_reward = self.q25_reward.numpy().mean(axis=1)
		q75_reward = self.q75_reward.numpy().mean(axis=1)
		
		avg_timesteps = self.timesteps.numpy().mean(axis=1)
		std_timesteps = self.std_timesteps.numpy().mean(axis=1)
		q25_timesteps = self.q25_timesteps.numpy().mean(axis=1)
		q75_timesteps = self.q75_timesteps.numpy().mean(axis=1)


		df = pd.DataFrame({'epoch': epoch,
						   'timestep': timestep, 
						   'avg_reward': avg_reward,
						   'std_reward': std_reward,
						   'q25_reward': q25_reward,
						   'q75_reward': q75_reward,
						   'avg_timesteps': avg_timesteps,
						   'std_timesteps': std_timesteps,
						   'q25_timesteps': q25_timesteps,
						   'q75_timesteps': q75_timesteps})

		df.to_csv(fn, index=False)



class SharedBuffer:

	def __init__(self, max_size, obs_dim=None):

		self.obs_buf = torch.zeros(
			(max_size, obs_dim)).to(device).share_memory_()
		self.act_buf = torch.zeros(
			max_size, 1).to(device).share_memory_()
		self.logp_buf = torch.zeros(
			max_size, 1).to(device).share_memory_()
		self.adv_buf = torch.zeros(
			max_size, 1).to(device).share_memory_()
		self.ret_buf = torch.zeros(
			max_size, 1).to(device).share_memory_() 
		self.num_used = torch.zeros(1, dtype=int).share_memory_()

	def reset(self):
		self.num_used.zero_()

	def store(self, obs=None, act=None, logp=None, adv=None, ret=None):

		n = self.num_used.item()

		if obs != None:
			self.obs_buf[n] = obs
		if act != None:
			self.act_buf[n] = act
		if logp != None:
			self.logp_buf[n] = logp
		if adv != None:
			self.adv_buf[n] = adv
		if ret != None:
			self.ret_buf[n] = ret

		self.num_used += 1

	def get(self):

		n = self.num_used

		obs_buf = self.obs_buf[:n]
		act_buf = self.act_buf[:n]
		logp_buf = self.logp_buf[:n]
		adv_buf = self.adv_buf[:n]
		ret_buf = self.ret_buf[:n]

		# advantage normalisation
		adv_mean, adv_std = torch.mean(adv_buf), torch.std(adv_buf)
		adv_buf = (adv_buf - adv_mean) / adv_std

		data = dict(obs=obs_buf, act=act_buf,
					adv=adv_buf, logp=logp_buf,
					ret=ret_buf)

		return {k: torch.as_tensor(v, dtype=torch.float32) for k,v in data.items()}

def discount_cumsum(x, discount):
	"""
	Calculate discounted cumulative sum, from SpinningUp repo.
	"""
	return signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]

class Worker(mp.Process):
	"""
	Worker class
	"""
	def __init__(self, worker_id, env, pipe, policy, gamma, lam, 
				num_epochs, steps_per_epoch, actor_buf, critic_buf, logger):

		mp.Process.__init__(self, name=worker_id)


		self.worker_id = worker_id
		self.policy = policy
		self.env = copy.deepcopy(env)
		self.gamma = gamma
		self.lam = lam
		self.pipe = pipe
		self.num_epochs = num_epochs
		self.steps_per_epoch = steps_per_epoch
		self.actor_buf = actor_buf
		self.critic_buf = critic_buf
		self.logger = logger

	def run(self):

		for epoch in range(self.num_epochs):

			all_ep_rewards, all_ep_timesteps = self.run_epoch()

			# self.logger.log(reward=np.mean(all_ep_rewards),
			# 				std_reward=np.std(all_ep_rewards),
			# 				q25_reward=np.quantile(all_ep_rewards, 0.25),
			# 				q75_reward=np.quantile(all_ep_rewards, 0.75),
			# 				timesteps=np.mean(all_ep_timesteps),
			# 				std_timesteps=np.std(all_ep_timesteps),
			# 				q25_timesteps=np.quantile(all_ep_timesteps, 0.25),
			# 				q75_timesteps=np.quantile(all_ep_timesteps, 0.75),
			# 				worker_id=int(self.worker_id),
			# 				epoch=epoch)
			self.logger.store('mean_reward', np.mean(all_ep_rewards), epoch, int(self.worker_id))
			self.logger.store('std_reward', np.std(all_ep_rewards), epoch, int(self.worker_id))
			self.logger.store('q25_reward', np.quantile(all_ep_rewards, 0.25), epoch, int(self.worker_id))
			self.logger.store('q75_reward', np.quantile(all_ep_rewards, 0.75), epoch, int(self.worker_id))
			self.logger.store('mean_timesteps', np.mean(all_ep_timesteps), epoch, int(self.worker_id))
			self.logger.store('std_timesteps', np.std(all_ep_timesteps), epoch, int(self.worker_id))
			self.logger.store('q25_timesteps', np.quantile(all_ep_timesteps, 0.25), epoch, int(self.worker_id))
			self.logger.store('q75_timesteps', np.quantile(all_ep_timesteps, 0.75), epoch, int(self.worker_id))

			msg = MsgUpdateRequest(int(self.worker_id), True)
			self.pipe.send(msg)
			msg = self.pipe.recv()

			self.actor_buf.reset()
			self.critic_buf.reset()

	def run_epoch(self):

		all_ep_timesteps = []
		all_ep_rewards = []
		sub_acts = []
		sub_obss = []
		obss = []
		advs = []
		rets = []
		logps = []
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
				a, sub_obs, sub_act, log_probs = self.policy.generate_action(self.env, obs)

				# print(a)

				# Record value from critic
				value, obs_processed = self.policy.get_value(obs)
				obss.append(obs_processed)
				ep_values.append(value.item())

				# Step environment 
				new_obs, reward, done = self.env.step(a)

				# Record unscaled reward
				unscaled_ep_rewards.append(reward)

				# Transform the reward
				reward = 1+reward/-self.env.min_reward
				# reward = (reward - self.policy.mean_reward) / self.policy.std_reward
				# print(reward)
				reward = reward.clip(-10, 10)

				# Update episode rewards and timesteps
				ep_timesteps += 1

				sub_acts.append(sub_act)
				sub_obss.append(sub_obs)
				logps.append(log_probs)
				ep_rewards.append(reward)

			all_ep_rewards.append(sum(unscaled_ep_rewards))
			all_ep_timesteps.append(ep_timesteps)

			# the next lines implement GAE-Lambda advantage calculation
			ep_rewards.append(0)
			ep_values.append(0)
			ep_rewards = np.array(ep_rewards)
			ep_values = np.array(ep_values)
			ep_returns = discount_cumsum(ep_rewards, self.gamma)[:-1]
			deltas = ep_rewards[:-1] + self.gamma * ep_values[1:] - ep_values[:-1]
			ep_advantages = discount_cumsum(deltas, self.gamma * self.lam)
			for adv in ep_advantages:
				advs.append(adv)
			for ret in ep_returns:
				rets.append(ret)

		# TODO: consider using torch.cat to remove these for loops
		for i in range(self.steps_per_epoch):
			for j in range(len(sub_acts[i])):
				self.actor_buf.store(obs = sub_obss[i][j],
									 act = sub_acts[i][j], 
									 logp = logps[i][j],
									 adv = advs[i])

		for i in range(self.steps_per_epoch):
			self.critic_buf.store(obs = obss[i],
								  ret = rets[i])


		return all_ep_rewards, all_ep_timesteps


def train(save_dir,
		  timesteps, 
		  num_workers, 
		  steps_per_epoch, 
		  env_params, 
		  policy_params,
		  gamma,
		  lam):

	# Total number of epochs (updates)
	num_epochs = int(timesteps / steps_per_epoch)

	# Number of timesteps each worker should gather per epoch
	worker_steps_per_epoch = int(steps_per_epoch / num_workers)

	env = make_env(**env_params)
	policy = PPOAgent(env, **policy_params).share_memory()

	pi_optimizer = optim.Adam(policy.parameters(), lr=policy_params.get('ac_learning_rate'))
	v_optimizer = optim.Adam(policy.parameters(), lr=policy_params.get('cr_learning_rate'))

	# The actor buffer will typically take more entries than the critic buffer,
	# because it records sub-actions. Hence there is usually more than one entry
	# per timestep. Here we set the size to be the max possible.
	actor_buf = SharedBuffer(max_size=num_workers*steps_per_epoch*env.num_gen, obs_dim=policy.n_in_ac)
	critic_buf = SharedBuffer(max_size=num_workers*steps_per_epoch, obs_dim=policy.n_in_cr)

	log_keys = ('mean_reward', 'std_reward', 'q25_reward', 'q75_reward',
			    'mean_timesteps', 'std_timesteps', 'q25_timesteps', 'q75_timesteps', 'entropy')
	logger = NewLogger(num_epochs, num_workers, steps_per_epoch, *log_keys)


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
						actor_buf=actor_buf, 
						critic_buf=critic_buf,
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

						print("Epoch: {}".format(epoch_counter))
						print("Updating")


						entropy, loss_v, explained_variance = policy.update(actor_buf, critic_buf, pi_optimizer, v_optimizer)
						print("Entropy: {}".format(entropy.mean()))
						for w in range(num_workers):
							logger.store('entropy', entropy.mean().detach(), epoch_counter, w)

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

	
	parser = argparse.ArgumentParser(description='Train PPO agent')
	parser.add_argument('--save_dir', type=str, required=True)
	parser.add_argument('--workers', type=int, required=False, default=1)
	parser.add_argument('--num_gen', type=int, required=True)
	parser.add_argument('--timesteps', type=int, required=True)
	parser.add_argument('--steps_per_epoch', type=int, required=True)

	# The following params will be used to setup the PPO agent
	parser.add_argument('--ac_learning_rate', type=float, required=False, default=3e-05)
	parser.add_argument('--cr_learning_rate', type=float, required=False, default=3e-04)
	parser.add_argument('--num_layers', type=int, required=False, default=3)
	parser.add_argument('--num_nodes', type=int, required=False, default=32)
	parser.add_argument('--entropy_coef', type=float, required=False, default=0.01)
	parser.add_argument('--update_epochs', type=int, required=False, default=4)
	parser.add_argument('--clip_ratio', type=float, required=False, default=0.1)
	parser.add_argument('--forecast_horizon_hrs', type=int, required=False, default=12)
	parser.add_argument('--credit_assignment_1hr', type=float, required=False, default=0.9)
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
		  env_params=env_params,
		  policy_params=policy_params,
		  gamma=gamma,
		  lam=lam)



