import json
import os
import random

import jax
import ml_collections
import numpy as np
import tqdm
import wandb
from absl import app, flags

import robots
from agents import agents
from robots.online_env import OnlineRobotEnv
from robots.policy_wrapper import JAXWrapperOnline
from envs import normalization
from envs.env_utils import make_env_and_datasets
from utils.datasets import Dataset, ReplayBuffer, save_compact_buffer
from utils.flax_utils import restore_agent, save_agent, load_example_batch, print_batch_shapes
from utils.log_utils import setup_wandb, get_exp_name


class LoggingHelper:
    def __init__(self, wandb_logger):
        self.wandb_logger = wandb_logger

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
    return ml_collections.ConfigDict(data)

FLAGS = flags.FLAGS

flags.DEFINE_string('checkpoint_path', '', 'Checkpoint path')
flags.DEFINE_integer('image_size', 224, 'Camera resolution')
flags.DEFINE_integer('buffer_size', 10000, 'Online replay buffer capacity; one 224x224x6 transition costs ~0.6MB')
flags.DEFINE_string('robot', 'droid', f'Robot adapter: {robots.available()}')
flags.DEFINE_string('run_group', 'Debug', 'Run group.')
flags.DEFINE_string('project', 'LPS', 'Wandb project.')
flags.DEFINE_integer('checkpoint_step', 1000000, 'Checkpoint step')
flags.DEFINE_string('save_dir', 'online_ckpt', 'Save directory')
flags.DEFINE_integer('seed', 100, 'Random seed')
flags.DEFINE_integer('online_steps', 10000, 'Number of online environment steps')
flags.DEFINE_string('env_name', '', 'Dataset name under the dataset directory')
flags.DEFINE_integer('utd', 10, 'Gradient steps per environment step, paid when an episode ends')
flags.DEFINE_integer('batch_size', 128, 'Batch size, half offline data and half online')
flags.DEFINE_float('discount', 0.99, 'Discount factor')
flags.DEFINE_float('p_aug', 0.5, 'Probability of random-crop augmentation per batch')
flags.DEFINE_string('droid_dataset_dir', None, 'DROID dataset directory')
flags.DEFINE_string('lerobot_dataset_dir', None, 'LeRobot dataset directory')
flags.DEFINE_integer('horizon_length', 5, 'Action chunk length')
flags.DEFINE_bool('droid_use_failure', False, 'Use failure DROID dataset or not')
flags.DEFINE_bool('sparse', False, 'Use the sparse reward convention')


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

    # The offline half goes on the policy's own scale, read back from the checkpoint -- not
    # recomputed, which would drift from what it was trained under.
    norm_stats = normalization.load(path)
    env, eval_env, train_dataset, val_dataset = make_env_and_datasets(
        FLAGS.env_name,
        droid_dir=FLAGS.droid_dataset_dir,
        lerobot_dir=FLAGS.lerobot_dataset_dir,
        droid_use_failure=FLAGS.droid_use_failure,
        sparse=FLAGS.sparse, discount=FLAGS.discount,
        horizon_length=FLAGS.horizon_length,
        normalize_stats=norm_stats,
    )
    def process_train_dataset(dataset, is_dataset=True):
        if is_dataset:
            dataset = Dataset.create(**dataset)
        dataset.action_sequence = FLAGS.horizon_length
        dataset.discount = FLAGS.discount
        dataset.p_aug = FLAGS.p_aug
        return dataset
    
    example_batch = load_example_batch(path)
    print_batch_shapes(example_batch)

    train_dataset = process_train_dataset(train_dataset, True)
    example_transition = get_initialization_sample(train_dataset)
    replay_buffer = ReplayBuffer.create(example_transition, size=FLAGS.buffer_size)
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

    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)

    policy = JAXWrapperOnline(agent, norm_stats)

    robot = robots.get(FLAGS.robot)(image_size=FLAGS.image_size)

    logger = LoggingHelper(
        wandb_logger=wandb,
    )

    def fast_merge(ds_batch, rb_batch):
        return jax.tree_util.tree_map(
            lambda x, y: np.concatenate([x, y], axis=0),
            ds_batch, 
            rb_batch
        )
    
    online_env = OnlineRobotEnv(robot, sparse=FLAGS.sparse, discount=FLAGS.discount, norm_stats=norm_stats)
    online_step = 0
    last_save_step = 0
    os.makedirs(FLAGS.save_dir, exist_ok=True)
    buffer_path = os.path.join(FLAGS.save_dir, "replay_buffer.npz")
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
                print('Ready to roll out.')
                done = False
                local_steps = 0
                online_env.wait_for_noskip()
                obs = online_env.get_observation()
                
                while not done:
                    online_env.wait_for_noskip()
                    action = policy.forward(obs)
                    next_obs, reward, done = online_env.step(action)
                    online_step += 1
                    local_steps += 1
                    pbar.update(1)
                    success = False
                    if done:
                        # The operator's verdict, not the reward: a failure also ends on reward 0.
                        success = online_env.last_outcome == 'success'
                        total_success += int(success)
                        total_failure += int(not success)
                        success_list.append(int(success))

                    # Repeat the final transition so the episode end is sampled more often. Only the
                    # first copy carries the outcome reward; the rest carry the per-step reward.
                    repeat = 5 if done else 1
                    for r in range(repeat):
                        replay_buffer.add_transition(
                            dict(
                                observations=obs,
                                actions=online_env.normalize_action(action),
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
                if len(success_list) > 0:
                    ma_sr = sum(success_list[-20:]) / len(success_list[-20:]) * 100.0
                    log_dict['eval/ma_sr'] = ma_sr
                    print('MA 20 SR : ', ma_sr, '%')
                logger.log(log_dict, step=online_step)

                online_env.reset()
                
                # Training
                if (total_failure + total_success) > 3:
                    for _ in tqdm.tqdm(range(int(FLAGS.utd*local_steps)), desc="Training"):
                        rb_batch = replay_buffer.sample(config['batch_size'] // 2)
                        ds_batch = train_dataset.sample(config['batch_size'] // 2)

                        batch = fast_merge(ds_batch, rb_batch)
                        agent, info = agent.update(batch)
                        logger.log(info, step=online_step)

                signal_input = online_env.wait_for_controller("Press A to continue next rollout, B for finish.")

                # Episodes end on arbitrary steps, so save once at least 1000 steps have passed.
                if online_step - last_save_step >= 1000:
                    save_compact_buffer(replay_buffer, buffer_path, norm_stats)
                    save_agent(agent, FLAGS.save_dir, online_step)
                    last_save_step = online_step

                if signal_input == 'A':
                    continue
                elif signal_input == 'B':
                    break
    
    except KeyboardInterrupt:
        print("\n\n[!] KeyboardInterrupt detected (Ctrl+C). Saving current state before exiting...")
    
    # Runs whether the loop finished or was interrupted.
    print("Saving replay buffer and agent...")
    try:
        save_compact_buffer(replay_buffer, buffer_path, norm_stats)
        save_agent(agent, FLAGS.save_dir, online_step)
        print("Save complete.")
    except Exception as e:
        print(f"Error while saving: {e}")

    try:
        print("Running cleanup...")
        online_env.reset()
        
        if wandb.run is not None:
            wandb_dir = os.path.dirname(logger.wandb_logger.run.dir)
            logger.wandb_logger.run.finish()
            os.system(f'wandb sync {wandb_dir}')
            
        print("Cleanup complete.")
    except Exception as e:
        print(f"Error during cleanup: {e}")

    return 0

if __name__ == '__main__':
    app.run(main)
