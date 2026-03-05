import h5py
import json
import os
import numpy as np
# import torch
from collections import OrderedDict
from copy import deepcopy
import time
import jax

def supply_rng(f, rng=jax.random.PRNGKey(0)):
    """Helper function to split the random number generator key before each call to the function."""

    def wrapped(*args, **kwargs):
        nonlocal rng
        rng, key = jax.random.split(rng)
        return f(*args, rng=key, **kwargs)

    return wrapped

def preprocess_observation(observation):
    output_state = {}
    robot_state = observation['robot_state']
    cart_pos = robot_state['cartesian_position']
    grp_pos = robot_state['gripper_position']
    concat_state = np.concatenate([cart_pos, [grp_pos]])

    imgs = observation['image']
    cam_keys = sorted([k for k in imgs.keys() if k.endswith('_left')])
    imgs_cam1 = imgs[cam_keys[0]][:][..., :3][..., ::-1]
    imgs_cam2 = imgs[cam_keys[1]][:][..., :3][..., ::-1]
    images = np.concatenate([imgs_cam1, imgs_cam2], axis=-1)

    output_state['state'] = concat_state
    output_state['image'] = images

    return output_state


class ReplayWrapper:
    def __init__(self, file_dir):
        self.idx = 0
        self.file = h5py.File(file_dir)
        self.max_idx = len(self.file['action']['cartesian_velocity'])

    def forward(self, observation):
        if self.idx < self.max_idx:
            cart = self.file['action']['cartesian_velocity'][self.idx]
            grp = self.file['action']['gripper_position'][self.idx]
            action = np.concatenate([cart, [grp]])
            print(action)
            self.idx += 1
        else:
            return np.zeros(7,)
        return action
    
class JAXWrapper:
    def __init__(self, agent):
        self.agent = agent
        self.horizon_length = self.agent.config['horizon_length']
        self.horizon_counter = 0
        self.action = 0
        rng = jax.random.PRNGKey(np.random.randint(0, 2**32))
        self.actor_fn = supply_rng(agent.sample_actions, rng=rng)

    def reset(self):
        self.action = 0
        self.horizon_counter = 0

    def forward(self, observation):
        if self.horizon_counter == 0:
            p_obs = preprocess_observation(observation)
            self.action = self.actor_fn(observations=p_obs)
        output_action = self.action[self.horizon_counter]
        self.horizon_counter += 1

        if self.horizon_counter == self.horizon_length:
            self.horizon_counter = 0

        print(output_action.shape, output_action)

        return output_action

class JAXWrapperOnline:
    def __init__(self, agent):
        self.agent = agent
        self.horizon_length = self.agent.config['horizon_length']
        self.horizon_counter = 0
        self.action = 0
        rng = jax.random.PRNGKey(np.random.randint(0, 2**32))
        self.actor_fn = supply_rng(agent.sample_actions, rng=rng)

    def reset(self):
        self.action = 0
        self.horizon_counter = 0

    def forward(self, observation):
        if self.horizon_counter == 0:
            self.action = self.actor_fn(observations=observation)
        output_action = self.action[self.horizon_counter]
        self.horizon_counter += 1

        if self.horizon_counter == self.horizon_length:
            self.horizon_counter = 0

        return output_action