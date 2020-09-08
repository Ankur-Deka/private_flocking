import gym
import sys

gym.envs.register(id='simpleFlockingAirsim-v0',
				  entry_point='airsim_envs.gym_airsim:simpleFlockingAirsim')