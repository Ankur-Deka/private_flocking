import numpy as np
import gym
from airsim_envs import gym_airsim


env = gym.make('simpleFlockingAirsim-v0')
directions = [1, 4, 2, 3]
for direc in directions:
	dummy_actions = direc*np.ones(9)
	for i in range(20):
		obs_nxt, rew, done, info = env.step(dummy_actions)
		print('obs_nxt', obs_nxt)