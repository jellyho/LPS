from utils.flax_utils import restore_agent, load_example_batch, print_batch_shapes, print_param_stats
import glob
import tqdm
import wandb
import os
import json
import random
import time
import jax
import numpy as np
import ml_collections
from absl import app, flags
from ml_collections import config_flags
from agents import agents
from droid_utils.policy_wrapper import ReplayWrapper, JAXWrapper
from droid.controllers.oculus_controller import VRPolicy
from droid.robot_env import RobotEnv
from droid.user_interface.data_collector import DataCollecter
from droid.user_interface.gui import RobotGUI  

if 'CUDA_VISIBLE_DEVICES' in os.environ:
    os.environ['EGL_DEVICE_ID'] = os.environ['CUDA_VISIBLE_DEVICES']
    os.environ['MUJOCO_EGL_DEVICE_ID'] = os.environ['CUDA_VISIBLE_DEVICES']

def load_config_from_json(json_path):
    """Load configuration from a JSON file and convert to ConfigDict."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    # Convert dictionary to ConfigDict (allows dot notation and type safety)
    config = ml_collections.ConfigDict(data)
    return config

FLAGS = flags.FLAGS

flags.DEFINE_string('checkpoint_path', '', 'Checkpoint path')
flags.DEFINE_integer('checkpoint_step', 1000000, 'Checkpoint step')
flags.DEFINE_integer('seed', 100, 'seed')

def main(_):
    path = FLAGS.checkpoint_path
    step = FLAGS.checkpoint_step
    flag_config = load_config_from_json(f'{path}/flags.json')
    config = flag_config.agent
    config["horizon_length"] = flag_config.horizon_length
    ckpt_name = path.split("/")[-1]
    log_dir = "eval_logs/" + ckpt_name

    example_batch = load_example_batch(path)
    print_batch_shapes(example_batch)
    
    agent_class = agents[config['agent_name']]
    agent = agent_class.create(
        FLAGS.seed,
        example_batch['observations'],
        example_batch['actions'],
        config,
    )
    agent = restore_agent(agent, path, step)
    
    ## policy wrapper
    np.random.seed(FLAGS.seed)

    policy = JAXWrapper(agent)

    ## robo Env
    imsize = 224
    action_space = "cartesian_velocity"
    gripper_action_space = "position"

    data_processing_kwargs = dict(
        timestep_filtering_kwargs=dict(
            action_space=action_space,
            gripper_action_space=gripper_action_space,
            robot_state_keys=["cartesian_position", "gripper_position", "joint_positions"],
        ),
        image_transform_kwargs=dict(
            remove_alpha=True,
            bgr_to_rgb=True,
            to_tensor=False,
            augment=False,
        ),
    )
    timestep_filtering_kwargs = data_processing_kwargs.get("timestep_filtering_kwargs", {})
    image_transform_kwargs = data_processing_kwargs.get("image_transform_kwargs", {})

    policy_timestep_filtering_kwargs = {}
    policy_image_transform_kwargs = {}

    policy_timestep_filtering_kwargs.update(timestep_filtering_kwargs)
    policy_image_transform_kwargs.update(image_transform_kwargs)

    camera_kwargs = dict(
        hand_camera=dict(image=True, concatenate_images=False, resolution=(imsize, imsize), resize_func="cv2"),
        varied_camera=dict(image=True, concatenate_images=False, resolution=(imsize, imsize), resize_func="cv2"),
    )
    
    policy_camera_kwargs = {}
    policy_camera_kwargs.update(camera_kwargs)

    env = RobotEnv(
        action_space=policy_timestep_filtering_kwargs["action_space"],
        gripper_action_space=policy_timestep_filtering_kwargs["gripper_action_space"],
        camera_kwargs=policy_camera_kwargs
    )
    controller = VRPolicy()

    # Launch GUI #
    data_collector = DataCollecter(
        env=env,
        controller=controller,
        policy=policy,
        save_traj_dir=log_dir,
        save_data=True,
    )
    RobotGUI(robot=data_collector)


if __name__ == '__main__':
    try:
        app.run(main)
    finally:
        pass