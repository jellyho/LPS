import h5py
import json
import os
import numpy as np
# import torch
from collections import OrderedDict
from copy import deepcopy
import time
import jax
from .policy_wrapper import preprocess_observation

from droid.misc.time import time_ms

class OnlineRobotEnv:
    def __init__(self, env, controller, sparse=False):
        self.env = env
        self.controller = controller
        self.sparse=sparse

    def reset(self):
        self.wait_for_controller("Press A to Reset the environment")
        print('Resetting...')
        self.env._robot.establish_connection()
        self.controller.reset_state()
        self.env.reset(randomize=False)
        self.control_timestamps = {"step_start": time_ms()}
        obs = self.env.get_observation()
        print("Reset Done!")
        return preprocess_observation(obs)
    
    def get_observation(self):
        obs = self.env.get_observation()
        return preprocess_observation(obs)
    
    def step(self, action):
        controller_info = {} if (self.controller is None) else self.controller.get_info()
        skip_action = not controller_info["movement_enabled"]
        done = False
        if self.sparse:
            reward = 0.0
        else:
            reward = -1.0
        
        skipped = False

        self.control_timestamps["step_start"] = time_ms()
        action_info = self.env.step(action)

        obs = self.env.get_observation()
        obs = preprocess_observation(obs)

        done = controller_info["success"] or controller_info["failure"]

        if done:
            if controller_info["success"]:
                if self.sparse:
                    reward = 1
                else:
                    reward = 0
                
            else:
                if self.sparse:
                    reward = 0.0
                else:
                    reward = -100.0
                

        comp_time = time_ms() - self.control_timestamps["step_start"]
        sleep_left = (1 / self.env.control_hz) - (comp_time / 1000)
        if sleep_left > 0:
            time.sleep(sleep_left)

        return obs, reward, done  #, skipped
    
    def wait_for_controller(self, msg="Press A to continue"):
        controller_info = self.controller.get_info()
        while controller_info["success"] or controller_info["failure"]:
            controller_info = self.controller.get_info()
        waiting = False
        if msg is not None:
            print(msg)
        while not waiting:
            controller_info = self.controller.get_info()
            if controller_info["success"]:
                waiting = True
                signal = 'A'
            elif controller_info["failure"]:
                waiting = True
                signal = 'B'
        return signal
    
    def wait_for_noskip(self):
        waiting = False
        while not waiting:
            controller_info = self.controller.get_info()
            waiting = controller_info["movement_enabled"]