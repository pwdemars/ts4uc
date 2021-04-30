import torch
import torch.optim as optim
import numpy as np
import torch.multiprocessing as mp
import scipy.signal as signal
import copy
from collections import namedtuple

from rl4uc.environment import make_env
from ts4uc.agents.ppo.ppo import PPOAgent

device = "cpu"
mp.set_start_method('spawn', True)

MsgUpdateRequest = namedtuple('MsgUpdateRequest', ['agent', 'update'])

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
		print("Getting data... number in buffer: {}".format(n))

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
				num_epochs, steps_per_epoch, actor_buf, critic_buf):

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

	def run(self):

		for epoch in range(self.num_epochs):

			self.run_epoch()

			msg = MsgUpdateRequest(int(self.worker_id), True)
			self.pipe.send(msg)
			msg = self.pipe.recv()

			self.actor_buf.reset()
			self.critic_buf.reset()

	def run_epoch(self):

		all_ep_timesteps = []
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
			ep_rewards = []
			ep_values = []

			done = False

			while not done:
				# Choose action with actor
				a, sub_obs, sub_act, log_probs = self.policy.generate_action(self.env, obs)

				# Record value from critic
				value, obs_processed = self.policy.get_value(obs)
				obss.append(obs_processed)
				ep_values.append(value.item())

				# Step environment 
				new_obs, reward, done = self.env.step(a)

				# Transform the reward
				reward = 1+reward/-self.env.min_reward
				reward = reward.clip(-10, 10)

				# Update episode rewards and timesteps
				ep_reward += reward
				ep_timesteps += 1

				sub_acts.append(sub_act)
				sub_obss.append(sub_obs)
				logps.append(log_probs)
				ep_rewards.append(reward)

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

		print(np.mean(all_ep_timesteps))

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


def train(timesteps, 
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

	pi_optimizer = optim.Adam(policy.parameters(), lr=3e-3)
	v_optimizer = optim.Adam(policy.parameters(), lr=3e-2)

	# The actor buffer will typically take more entries than the critic buffer,
	# because it records sub-actions. Hence there is usually more than one entry
	# per timestep. Here we set the size to be the max possible.
	actor_buf = SharedBuffer(max_size=num_workers*steps_per_epoch*env.num_gen, obs_dim=policy.n_in_ac)
	critic_buf = SharedBuffer(max_size=num_workers*steps_per_epoch, obs_dim=policy.n_in_cr)


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
						num_epochs=num_epochs,
						steps_per_epoch=worker_steps_per_epoch,
						gamma=gamma,
						lam=lam)
		worker.start()
		workers.append(worker)
		pipes.append(p_start)

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

						epoch_counter += 1
						update_request = [False]*num_workers
						msg = epoch_counter

						print(actor_buf.adv_buf[:actor_buf.num_used])

						# send to signal subprocesses to continue
						for pipe in pipes:
							pipe.send(msg)


if __name__ == "__main__":

	import json

	env_params = {'num_gen': 5}
	policy_params = {'ac_learning_rate': 3e-5,
					 'cr_learning_rate': 3e-4,
					 'clip_ratio': 0.1,
					 'entropy_coef': 0.05,
					 'minibatch_size': 50,
					 'num_layers': 3,
					 'num_nodes': 32,
					 'buffer_size'
					 'forecast_horizon_hrs': 12}


	timesteps = 1000
	num_workers = 2
	steps_per_epoch = 100

	assert timesteps > steps_per_epoch

	train(timesteps=timesteps,
		  num_workers=num_workers,
		  steps_per_epoch=steps_per_epoch,
		  env_params=env_params,
		  policy_params=policy_params,
		  gamma=0.99,
		  lam=0.95)



