# import numpy as np
# import gym
# from airsim_envs import gym_airsim
# import time

# env = gym.make('simpleFlockingAirsim-v0')
# env.world.max_steps_episode = 100
# print(env.observation_space)
# directions = [4]#, 2, 3, 4]
# for direc in directions:
# 	print('\n\n\n\ndirection {}'.format(direc))
# 	time.sleep(0.1)
# 	dummy_actions = direc*np.ones(5)#.array([1,1,0,0,0,0,0,0,0])
# 	for i in range(100):
# 		obs_nxt, rew, done, info = env.step(dummy_actions)
# 		# print('obs_nxt', obs_nxt)

import json
init_config = json.load(open('/home/ankur/Documents/AirSim/settings.json'))
print(init_config['ClockSpeed'])