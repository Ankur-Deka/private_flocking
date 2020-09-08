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
import numpy as np

parser = argparse.ArgumentParser('flocking')

# initial settings
parser.add_argument('--init_settings_path', type=str, default='/home/ankur/MSR_Research_Home/private_flocking/settings.json', help='initial settings of drones')

# Flock
parser.add_argument('--flock_number', type=str, default='9', choices=['4', '5', '9', '12', '13', '16', '19', '21'])
parser.add_argument('--drivetrain_type', type=str, default='ForwardOnly', choices=['ForwardOnly', 'MaxDegreeOfFreedom'])
parser.add_argument('--base_dist', type=float, default=4)
parser.add_argument('--spread', type=float, default=1)
parser.add_argument('--frontness', type=float, default=0, help="relative position of leader wrt flock (-1 to +1)")
parser.add_argument('--sideness', type=float, default=0)

# Error and limitation settings
parser.add_argument('--add_observation_error', action='store_true', default=False)
parser.add_argument('--add_actuation_error', action='store_true', default=False)
parser.add_argument('--gaussian_error_mean', type=float, default=0)
parser.add_argument('--gaussian_error_std', type=float, default=0.1)
parser.add_argument('--add_v_threshold', action='store_true', default=False)
parser.add_argument('--v_thres', type=float, default=5)

# Flocking methods
parser.add_argument('--flocking_method', type=str, default='reynolds', choices=['reynolds'])

# Reynolds flocking
# Leaders
parser.add_argument('--leader_id', type=int, default=-1)
parser.add_argument('--trajectory', type=str, default='Sinusoidal', choices=['Line', 'Zigzag', 'Sinusoidal'])
parser.add_argument('--lookahead', type=float, default=0.5, help='lookahead x distance')
parser.add_argument('--line_len', type=float, default=10000, help='total length (x_direction) of the line, zigzag and sinusoidal trajectory')
parser.add_argument('--zigzag_len', type=float, default=60, help='period length of the zigzag trajectory')
parser.add_argument('--zigzag_width', type=float, default=5, help='one-sided width of the zigzag trajectory')
parser.add_argument('--sine_period_ratio', type=float, default=10, help='period ratio of the sine wave trajectory')
parser.add_argument('--sine_width', type=float, default=5, help='amplitude of the sine wave trajectory')
parser.add_argument('--v_leader', type=float, default=1)
parser.add_argument('--leader_sep_weight', type=float, default=0.3)
parser.add_argument('--leader_ali_weight', type=float, default=0.3)
parser.add_argument('--leader_coh_weight', type=float, default=0.3)
parser.add_argument('--leader_sep_max_cutoff', type=float, default=3)
parser.add_argument('--leader_ali_radius', type=float, default=200)
parser.add_argument('--leader_coh_radius', type=float, default=200)

# Followers
parser.add_argument('--pos2v_scale', type=float, default=0.5)
parser.add_argument('--sep_weight', type=float, default=1.25)
parser.add_argument('--ali_weight', type=float, default=1.0)
parser.add_argument('--coh_weight', type=float, default=0.75)
parser.add_argument('--sep_max_cutoff', type=float, default=3)
parser.add_argument('--ali_radius', type=float, default=200)
parser.add_argument('--coh_radius', type=float, default=200)

# Misc
parser.add_argument('--single_sim_duration', type=float, default=3.0, help='simulation duration for single experiment in minutes')
parser.add_argument('--log_step', type=int, default=25)
parser.add_argument('--log_level', type=str, default='INFO', choices=['INFO', 'DEBUG', 'WARN'])
parser.add_argument('--optim_path', type=str, default='')
parser.add_argument('--log_dir', type=str, default='logs')
parser.add_argument('--log_time', type=str, default='000')
parser.add_argument('--num_gpu', type=int, default=1)
parser.add_argument('--random_seed', type=int, default=12345)


class simpleFlockingAirsim(gym.Env):
    def __init__(self):
        # ti = time.time()

        args = parser.parse_args()
        if args.flocking_method == 'reynolds':
            from controllers.reynolds import Controller
        else:
            print("Wrong flocking controller specified.")
            sys.exit(1)
        from controllers.leader_controller import Leader_Controller

        # path = os.path.join(args.optim_path, args.log_dir)
        # timestamp = utils.log_args(path, args)
        # logger = utils.get_logger(path, timestamp)

        self.flock = Flock(args)
        self.velocity_gain_step = 0.5   # target velocity can change by this much
        self.step_duration = 0.5          # allow this much time for each step
        # utils.log_init_state(logger, self.flock)

        self.controller_list = []
        self.leader_controller_list = []
        self.num_agents = 9

        for drone in self.flock.drones:
            if drone.vehicle_name in self.flock.leader_list:
                controller = Leader_Controller(drone, self.flock.flock_list, args)
                self.leader_controller_list.append(controller)
            else:
                controller = Controller(drone, self.flock.flock_list, args)
            self.controller_list.append(controller)
        self.reset()

    def reset(self):
        self.flock.reset()  # initial state of drones and re-establish communication
        time.sleep(1)
        self.flock.take_off()
        # self.flock.initial_altitudes()
        # self.flock.initial_speeds() # since there's speed, might want to capture latest observation

    def step(self, actions): 
    # actions: MultiDiscrete([7,7,7...,7]), one of 7 actions for each agent
    #          [num_agents]
        
        # process discrete actions
        velocity_gains = np.zeros((self.num_agents, 3))
        for i,a in enumerate(actions):
            if a == 1: velocity_gains[i,0] = -self.velocity_gain_step
            if a == 2: velocity_gains[i,0] = +self.velocity_gain_step
            if a == 3: velocity_gains[i,1] = -self.velocity_gain_step
            if a == 4: velocity_gains[i,1] = +self.velocity_gain_step
            if a == 5: velocity_gains[i,2] = -self.velocity_gain_step
            if a == 6: velocity_gains[i,2] = +self.velocity_gain_step
        
        cur_velocities = np.zeros((self.num_agents, 3))
        
        # get current velocities
        for i,drone in enumerate(self.flock.drones):
            cur_velocities[i] = drone.get_velocity()

        target_velocities = cur_velocities+velocity_gains
        for i,drone in enumerate(self.flock.drones):
            drone.move_by_velocity(target_velocities[i])

        # let the env run for some time
        time.sleep(self.step_duration)

        # get new observations
        cur_positions = np.zeros((self.num_agents, 3))
        cur_velocities = np.zeros((self.num_agents, 3))
        for i,drone in enumerate(self.flock.drones):
            cur_positions[i] = drone.get_position()
            cur_velocities[i] = drone.get_velocity()

        # form the obs_vector: [num_agents, 4]
        obs_nxt = np.concatenate((cur_velocities, cur_positions), axis = 1)

        return obs_nxt, 0, False, None 













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
