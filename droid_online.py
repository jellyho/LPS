from utils.flax_utils import restore_agent
import glob, tqdm, wandb, os, json, random, time, jax
from absl import app, flags
from ml_collections import config_flags
from utils.flax_utils import save_agent, load_example_batch
from agents import agents
import numpy as np
from envs.env_utils import make_env_and_datasets
from droid_utils.policy_wrapper import ReplayWrapper, JAXWrapperOnline
from droid.controllers.oculus_controller import VRPolicy
from droid.robot_env import RobotEnv
from droid.user_interface.data_collector import DataCollecter
from droid.user_interface.gui import RobotGUI  
import json
import ml_collections
from utils.log_utils import setup_wandb, get_exp_name, get_flag_dict, CsvLogger
from utils.datasets import Dataset, ReplayBuffer, save_compact_buffer
from droid_utils.online_utils import OnlineRobotEnv
import signal
import sys
import atexit


class LoggingHelper:
    def __init__(self, wandb_logger):
        self.wandb_logger = wandb_logger
        self.first_time = time.time()
        self.last_time = time.time()

    def iterate(self, key, value):
        if 'hist' in key:
            return wandb.Histogram(value)
        else:
            return value

    def log(self, data, step, prefix=None):
        if prefix is None:
            self.wandb_logger.log({f'{k}': self.iterate(k, v) for k, v in data.items()}, step=step)
        else:
            self.wandb_logger.log({f'{prefix}/{k}': self.iterate(k, v) for k, v in data.items()}, step=step)


if 'CUDA_VISIBLE_DEVICES' in os.environ:
    os.environ['EGL_DEVICE_ID'] = os.environ['CUDA_VISIBLE_DEVICES']
    os.environ['MUJOCO_EGL_DEVICE_ID'] = os.environ['CUDA_VISIBLE_DEVICES']

def load_config_from_json(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    # Convert dictionary to ConfigDict
    config = ml_collections.ConfigDict(data)
    return config

FLAGS = flags.FLAGS

flags.DEFINE_string('checkpoint_path', '', 'Checkpoint path')
flags.DEFINE_string('run_group', 'Debug', 'Run group.')
flags.DEFINE_string('project', 'MFQ', 'Run group.')
flags.DEFINE_integer('checkpoint_step', 1000000, 'Checkpoint step')
flags.DEFINE_string('save_dir', 'online_ckpt', 'save directory')
flags.DEFINE_integer('seed', 100, 'seed')
flags.DEFINE_integer('online_steps', 10000, 'oneline steps')
flags.DEFINE_string('env_name', '', 'Run group.')
flags.DEFINE_integer('utd', 10, "update to data ratio")
flags.DEFINE_integer('gradient_steps', 100, "gradient steps")
flags.DEFINE_integer('batch_size', 128, "batch_size")
flags.DEFINE_float('discount', 0.99, 'discount factor')
flags.DEFINE_float('p_aug', 0.5, 'aug prob')
flags.DEFINE_string('droid_dataset_dir', None, 'DROID dataset directory')
flags.DEFINE_integer('horizon_length', 5, 'action chunking length.')
flags.DEFINE_bool('droid_use_failure', False, 'Use failure DROID dataset or not')
flags.DEFINE_bool('sparse', False, 'use sparse reward?')


def get_param_count(agent):
    """Calculate and return the number of parameters in the network."""
    params = agent.network.params
    if hasattr(params, 'unfreeze'):
        params = params.unfreeze()
    
    param_counts = {}
    
    # Calculate module-wise parameter counts
    for module_name, module_params in params.items():
        module_leaves = jax.tree_util.tree_leaves(module_params)
        param_counts[module_name] = sum(param.size for param in module_leaves)
    
    # Calculate total parameters
    all_leaves = jax.tree_util.tree_leaves(params)
    param_counts['total'] = sum(param.size for param in all_leaves)
    
    return param_counts

def print_param_stats(agent):
    """Print network parameter statistics."""
    param_counts = get_param_count(agent)
    
    print("Network Parameter Statistics:")
    print("-" * 50)
    
    # Print module-wise parameter counts
    for module_name, count in param_counts.items():
        if module_name != 'total':
            print(f"{module_name}: {count:,} parameters ({count * 4 / (1024**2):.2f} MB)")
    
    # Print total parameter count
    total = param_counts['total']
    print("-" * 50)
    print(f"Total parameters: {total:,} ({total * 4 / (1024**2):.2f} MB)")

def main(_):
    path = FLAGS.checkpoint_path
    step = FLAGS.checkpoint_step
    flag_config = load_config_from_json(f'{path}/flags.json')

    exp_name = get_exp_name(FLAGS.seed)
    run = setup_wandb(project=FLAGS.project, group=FLAGS.run_group, name=exp_name, mode='offline')
    run.tags = run.tags + (FLAGS.env_name,)


    config = flag_config.agent
    config["horizon_length"] = flag_config.horizon_length
    config["batch_size"] = FLAGS.batch_size
    config['online_mode'] = True
    config['training_steps'] = int(FLAGS.online_steps * FLAGS.utd)
    ckpt_name = path.split("/")[-1]
    log_dir = "online_eval_logs/" + ckpt_name

    def print_batch_shapes(batch, prefix=""):
        for k, v in batch.items():
            try:
                print(f"{prefix}{k}: {v.shape}")
            except (AttributeError, TypeError):
                if isinstance(v, dict):
                    print_batch_shapes(v, prefix=f"{prefix}{k}.")
                else:
                    pass

    def get_initialization_sample(batch, index=0):
        sample = {}
        for k, v in batch.items():
            if hasattr(v, 'items'):
                sample[k] = get_initialization_sample(v, index)
            else:
                sample[k] = v[index]
                
        if 'terminals' in sample:
            sample['terminals'] = np.ones_like(sample['terminals'])
            
        return sample

    env, eval_env, train_dataset, val_dataset = make_env_and_datasets(
        FLAGS.env_name, 
        droid_dir=FLAGS.droid_dataset_dir,
        droid_use_failure=FLAGS.droid_use_failure,
        sparse=FLAGS.sparse,
        horizon_length=FLAGS.horizon_length,
    )
    def process_train_dataset(dataset, is_dataset=True):
        if is_dataset:
            dataset = Dataset.create(**dataset)
        dataset.actor_action_sequence = ( FLAGS.horizon_length )
        dataset.critic_action_sequence = ( FLAGS.horizon_length )
        dataset.nstep = 1 # Actually N step
        dataset.discount = FLAGS.discount
        dataset.discount2 = FLAGS.discount
        dataset.p_aug = FLAGS.p_aug
        return dataset
    
    # Usage
    example_batch = load_example_batch(path)
    print_batch_shapes(example_batch)   

    train_dataset = process_train_dataset(train_dataset, True)
    example_transition = get_initialization_sample(train_dataset)
    replay_buffer = ReplayBuffer.create(example_transition, size=10000)
    replay_buffer = process_train_dataset(replay_buffer, False)
    replay_buffer.update_locs()
    
    agent_class = agents[config['agent_name']]
    agent = agent_class.create(
        FLAGS.seed,
        example_batch['observations'],
        example_batch['actions'],
        config,
    )
    agent = restore_agent(agent, path, step)
    
    new_network = agent.network.replace(step=0)
    agent = agent.replace(network=new_network)

    ## policy wrapper
    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)

    policy = JAXWrapperOnline(agent)
    discount = FLAGS.discount
    config["horizon_length"] = FLAGS.horizon_length
    
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

    logger = LoggingHelper(
        wandb_logger=wandb,
    )

    def fast_merge(ds_batch, rb_batch):
        return jax.tree_util.tree_map(
            lambda x, y: np.concatenate([x, y], axis=0),
            ds_batch, 
            rb_batch
        )
    
    online_env = OnlineRobotEnv(env, controller, sparse=FLAGS.sparse)
    online_step = 0
    total_success = 0
    total_failure = 0
    success_list = []
    
    # Save initial agent
    save_agent(agent, FLAGS.save_dir, online_step)

    try:
        with tqdm.tqdm(total=FLAGS.online_steps, initial=online_step) as pbar:
            while online_step < FLAGS.online_steps:
                obs = online_env.reset()
                policy.reset()
                print('You can start rolling out!!!!!!')
                done = False
                # skipped = True
                local_steps = 0
                online_env.wait_for_noskip()
                obs = online_env.get_observation()
                
                while not done:
                    online_env.wait_for_noskip()
                    action = policy.forward(obs)
                    next_obs, reward, done = online_env.step(action)
                    # if skipped:
                    #     pass
                    # else:
                    online_step += 1
                    local_steps += 1
                    pbar.update(1)
                    # next_obs = online_env.get_observation()
                    success = False
                    if done:
                        if FLAGS.sparse:
                            if reward == 1:
                                total_success += 1
                                success_list.append(1)
                                success = True
                            else:
                                total_failure += 1
                                success_list.append(0)
                        else:
                            if reward == 0:
                                total_success += 1
                                success_list.append(1)
                                success = True
                            else:
                                total_failure += 1
                                success_list.append(0)                

                    # Giving enough failure signals
                    repeat = 5 if done else 1
                    for r in range(repeat):
                        replay_buffer.add_transition(
                            dict(
                                observations=obs,
                                actions=action,
                                rewards=reward if r == 0 else (0 if FLAGS.sparse else -1),
                                terminals=1.0 if done and r == repeat - 1 else 0.0,
                                masks=1.0 if not done else 0.0,
                                next_observations=next_obs,
                            )
                        )
                    obs = next_obs
                
                # Logging per episode
                log_dict = {}
                log_dict['eval/success'] = 1.0 if success else 0.0
                log_dict['eval/execution_steps'] = local_steps
                logger.log(log_dict, step=online_step)
                replay_buffer.update_locs()
                
                if len(success_list) > 0:
                    ma_sr = sum(success_list[-20:]) / len(success_list[-20:]) * 100.0
                    log_dict['eval/ma_sr'] = ma_sr
                    print('MA 20 SR : ', ma_sr, '%')

                online_env.reset()
                
                # Training
                if (total_failure + total_success) > 3:
                    for gradient_step in tqdm.tqdm(range(int(FLAGS.utd*local_steps)), desc="Training"):
                        rb_batch = replay_buffer.sample(config['batch_size'] // 2)
                        ds_batch = train_dataset.sample(config['batch_size'] // 2)

                        batch = fast_merge(ds_batch, rb_batch)
                        agent, info = agent.update(batch)
                        logger.log(info, step=online_step)

                signal_input = online_env.wait_for_controller("Press A to continue next rollout, B for finish.")

                if (online_step + 1) % 1000 == 0:
                    save_compact_buffer(replay_buffer, f'replay_buffer.npz')
                    save_agent(agent, FLAGS.save_dir, online_step)

                if signal_input == 'A':
                    continue
                elif signal_input == 'B':
                    break
    
    except KeyboardInterrupt:
        print("\n\n[!] KeyboardInterrupt detected (Ctrl+C). Saving current state before exiting...")
    
    # --- SAVE LOGIC ---
    # This runs whether the loop finished naturally or was interrupted
    print("Saving replay buffer and agent...")
    try:
        save_compact_buffer(replay_buffer, f'replay_buffer.npz')
        save_agent(agent, FLAGS.save_dir, online_step)
        print("Save complete.")
    except Exception as e:
        print(f"Error while saving: {e}")

    # --- CLEANUP ---
    try:
        print("Running cleanup...")
        online_env.reset()
        os.system('nmcli connection up "Internet"')
        
        if wandb.run is not None:
            wandb_dir = os.path.dirname(logger.wandb_logger.run.dir)
            logger.wandb_logger.run.finish()
            os.system(f'wandb sync {wandb_dir}')
            
        print("Cleanup complete.")
    except Exception as e:
        print(f"Error during cleanup: {e}")

    return 0

if __name__ == '__main__':
    try:
        app.run(main)
    finally:
        pass