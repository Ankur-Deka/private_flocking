import os
import argparse
import time
import sys
sys.path.append('/home/ankur/MSR_Research_Home/private_flocking')
import utils
from classes import Flock
import pygame
from pygame.locals import *
import gym
from gym import spaces
import numpy as np
import json
import copy
from types import SimpleNamespace

args = SimpleNamespace()
# initial settings
args.init_settings_path='/home/ankur/Documents/AirSim/settings.json'
# Flock
args.flock_number='5'
args.drivetrain_type='ForwardOnly'
args.base_dist=4
args.spread=1
args.frontness=0
args.sideness=0
args.add_observation_error=False
# Error and limitation settings
args.add_actuation_error=False
args.gaussian_error_mean=0
args.gaussian_error_std=0.1
args.add_v_threshold=False
args.v_thres=5
args.flocking_method='reynolds'
args.leader_id=-1
# Flocking methods
args.trajectory='Sinusoidal'
args.lookahead=0.5
# Reynolds flocking
# Leaders
args.line_len=10000
args.zigzag_len=60
args.zigzag_width=5
args.sine_period_ratio=10
args.sine_width=5
args.v_leader=1
args.leader_sep_weight=0.3
args.leader_ali_weight=0.3
args.leader_coh_weight=0.3
args.leader_sep_max_cutoff=3
args.leader_ali_radius=200
args.leader_coh_radius=200
args.pos2v_scale=0.5
args.sep_weight=1.25
args.ali_weight=1.0
args.coh_weight=0.75
# Followers
args.sep_max_cutoff=3
args.ali_radius=200
args.coh_radius=200
# Misc
args.single_sim_duration=3.0
args.log_step=25
args.log_level='INFO'
args.optim_path=''
args.log_dir='logs'
args.log_time='000'
args.num_gpu=1
args.random_seed=12345
args.clock_speed=json.load(open('/home/ankur/Documents/AirSim/settings.json'))['ClockSpeed'] # must be same as set in settings.json

class simpleFlockingAirsim(gym.Env):
    def __init__(self):
        from controllers.reynolds import Controller
        from controllers.leader_controller import Leader_Controller

        # path = os.path.join(args.optim_path, args.log_dir)
        # timestamp = utils.log_args(path, args)
        # logger = utils.get_logger(path, timestamp)

        self.velocity_gain_step = 1   # target velocity can change by this much
        self.step_duration = 0.4/args.clock_speed#0.04          # allow this much time for each step (in real time, so should adjust according to airsim clockspeed)
        self.damping = 0.8
        self.controller_list = []
        self.leader_controller_list = []
        self.flock = Flock(args)
        self.n = int(args.flock_number)
        self.goal_at_top = True
        self.size = 25 # size of floor/2 in unreal in meters (unreal shows in cm, so 10000/100 = 100)
        self.action_space = [spaces.Discrete(7)]*self.n
        self.observation_space = [spaces.Box(low=-np.inf, high=+np.inf, shape=(9,), dtype=np.float32)]*self.n

        for drone in self.flock.drones:
            if drone.vehicle_name in self.flock.leader_list:
                controller = Leader_Controller(drone, self.flock.flock_list, args)
                self.leader_controller_list.append(controller)
            else:
                controller = Controller(drone, self.flock.flock_list, args)
            self.controller_list.append(controller)
        self.world = SimpleNamespace()
        self.world.leader_name = 0
        self.world.dim_p = 3
        self.world.steps = 0
        self.reset()

    def seed(self, seed=None):
        np.random.seed(seed)

    def get_obs(self):
        # get new observations
        cur_positions = np.zeros((self.n, 3))
        cur_velocities = np.zeros((self.n, 3))
        for i,drone in enumerate(self.flock.drones):
            cur_positions[i] = drone.get_position()/self.size
            cur_velocities[i] = drone.get_velocity()


        # interchange x and y axes
        cur_positions = cur_positions[:,[1,0,2]]
        cur_velocities = cur_velocities[:,[1,0,2]]

        # negate z axis and scale
        cur_positions[:,-1] = -cur_positions[:,-1]
        cur_velocities[:,-1] = -cur_velocities[:,-1]
        
        # now positions are in MAPE like coordinate (velocities may not be to scale)!
        # goal location in relative coordinate for first agent
        goal_loc = np.zeros((self.n, 3))
        goal_loc[0] = self.goal_loc - cur_positions[0]
        # print('goal_loc', goal_loc)
        

        # form the obs_vector: [n, 9]
        obs = np.concatenate((cur_velocities, cur_positions, goal_loc), axis = 1)
        # print('\ngym airsim') 
        # print('goal_loc')
        # print(self.goal_loc)
        # print('cur_positions')
        # print(cur_positions)
        # print('obs', obs.round(4))
        return obs

    def reset_goal(self):
        self.goal_loc = np.random.uniform([-1, 0, 0.2], [1,0.9,0.3]) # in mape like coordinate
        airsim_goal_pose = copy.copy(self.goal_loc)[[1,0,2]]*self.size   # convert to arisim coordinate
        airsim_goal_pose[-1] = -airsim_goal_pose[-1]
        pose = self.flock.client.simGetVehiclePose(vehicle_name='Drone{}'.format(self.n)) # goal drone
        pose.position.x_val, pose.position.y_val, pose.position.z_val = airsim_goal_pose

            # pose.position.x_val = (pos[drone.index][0] - drone.origin[0])
            # pose.position.y_val = (pos[drone.index][1] - drone.origin[1])
            # pose.position.z_val = z_val
        self.flock.client.simSetVehiclePose(pose, True, 'Drone{}'.format(self.n))
        time.sleep(0.1)
        print('goal', self.goal_loc, 'airsim goal pose', pose)

    def reset(self):
        self.flock.reset()  # initial state of drones and re-establish communication
        time.sleep(0.1)
        
        # self.goal_loc = np.array([-25/25,0/25,0.2])
        # obs = self.get_obs()
        # print('before take off state')
        # print(obs)
        self.flock.client.simPause(False)
        self.reset_goal()
        self.flock.take_off()
        # print('afer take off state')
        # print(obs)
        self.flock.initial_altitudes()
        self.flock.initial_speeds() # since there's speed, might want to capture latest observation
        self.flock.client.simPause(True)
        self.world.steps = 0
        self.prev_dists = None
        # reset the goal loc
        # if self.goal_at_top:
        #     self.goal_loc = np.random.uniform([-self.size, 0.9*self.size, 7.5], [self.size,0.9*self.size, 12.5])
        # else:
        #     self.goal_loc = np.random.uniform([-self.size, 0, 7.5], [size,0.9*self.size, 12.5])
        obs = self.get_obs()

        return obs

    def step(self, actions): 
    # actions: MultiDiscrete([7,7,7...,7]), one of 7 actions for each agent
    #          [n]
        self.world.steps += 1
        # process discrete actions
        for i,a in enumerate(actions):
            if a == 0:
                self.flock.drones[i].hover()
            else:
                velocity_gain = np.zeros(3)
                if a == 1: velocity_gain[1] = -self.velocity_gain_step   # interchange x and y
                elif a == 2: velocity_gain[1] = +self.velocity_gain_step
                elif a == 3: velocity_gain[0] = -self.velocity_gain_step
                elif a == 4: velocity_gain[0] = +self.velocity_gain_step
                elif a == 5: velocity_gain[2] = +self.velocity_gain_step   # negate z axis
                elif a == 6: velocity_gain[2] = -self.velocity_gain_step
                
                # get current velocity
                cur_velocity = self.flock.drones[i].get_velocity()

                # set to target velocity
                target_velocity = cur_velocity*self.damping + velocity_gain
                self.flock.drones[i].move_by_velocity(target_velocity)

        self.flock.client.simPause(False)
        # let the env run for some time
        time.sleep(self.step_duration)
        self.flock.client.simPause(True)
        # get new observations
        obs_nxt = self.get_obs()

        # compute rewards
        agent_poses = obs_nxt[:,3:6]
        dists = np.linalg.norm(agent_poses-self.goal_loc, axis = 1)
        if self.prev_dists is None:
            rewards = np.zeros(self.n)
        else:
            rewards = 1e2*(self.prev_dists-dists)
        # rewards = -dists
        # print('dists', dists)
        self.prev_dists = dists.copy()
    
        done = [self.world.steps >= self.world.max_steps_episode]*self.n
        info = {'is_success': False, 'env_done': done[0]}
                
        return obs_nxt, rewards, done, info 













# def main():

    

    #airsim.wait_key('Press any key to go to different altitudes')
    # print("Going to different alti tudes")
    # flock.initial_altitudes()

    # #airsim.wait_key('Press any key to start initial motion')
    # print("Starting random motion")
    # flock.initial_speeds()

    # #airsim.wait_key('Press any key to start flocking')
    # print("Now flocking")
    # count = 0

    # first_drone_name = flock.drones[0].vehicle_name
    # init_sim_time = flock.client.getMultirotorState(vehicle_name=first_drone_name).timestamp

    # while True:
    #         for controller in controller_list:
    #             controller.step()
    #         if count % 1 == 0:
    #             flock.log_flock_kinematics(logger, count)

    #         count += 1
    #         pygame.display.set_mode((1,1))
    #         pygame.event.pump()
    #         keys = pygame.key.get_pressed()
    #         if keys[K_ESCAPE]:
    #             flock.reset()
    #             break
    #         curr_sim_time = flock.client.getMultirotorState(vehicle_name=first_drone_name).timestamp
    #         if (curr_sim_time-init_sim_time)/1e9/60 > args.single_sim_duration:
    #             tf = time.time()
    #             print("Real world time, ", (tf-ti)/60)
    #             flock.reset()
    #             break


if __name__ == '__main__':
    env = gym_airsim()
    # for i in range(1):
    #     print('i')
    #     env.reset()
