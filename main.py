import glob
import tqdm
import wandb
import os
import json
import random
import time
import jax
import numpy as np
from absl import app, flags
from ml_collections import config_flags

from utils.log_utils import setup_wandb, get_exp_name, get_flag_dict
from envs.env_utils import make_env_and_datasets
from utils.flax_utils import save_agent, save_example_batch, print_param_stats, print_batch_shapes
from utils.datasets import Dataset, ReplayBuffer
from agents import agents

if 'CUDA_VISIBLE_DEVICES' in os.environ:
    os.environ['EGL_DEVICE_ID'] = os.environ['CUDA_VISIBLE_DEVICES']
    os.environ['MUJOCO_EGL_DEVICE_ID'] = os.environ['CUDA_VISIBLE_DEVICES']

FLAGS = flags.FLAGS

flags.DEFINE_string('run_group', 'Debug', 'Run group.')
flags.DEFINE_string('project', 'MFQ', 'Run group.')
flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_string('task_config', 'NO', 'suite:task_name:alpha:task_num')
flags.DEFINE_string('task_name', 'cube-triple-play', 'Task Name')
flags.DEFINE_integer('task_num', 0, 'Task Num')
flags.DEFINE_string('env_name', 'cube-triple-play-singletask-task2-v0', 'Environment (dataset) name.')
flags.DEFINE_string('save_dir', 'exp/', 'Save directory.')

flags.DEFINE_integer('offline_steps', 1000000, 'Number of offline steps.')
flags.DEFINE_integer('online_steps', 0, 'Number of online steps.')
flags.DEFINE_integer('buffer_size', 2000000, 'Replay buffer size.')
flags.DEFINE_integer('log_interval', 5000, 'Logging interval.')
flags.DEFINE_integer('eval_interval', 100000, 'Evaluation interval.')
flags.DEFINE_integer('save_interval', 1000000, 'Save interval.')
flags.DEFINE_integer('start_training', 5000, 'when does training start')

flags.DEFINE_integer('utd_ratio', 1, "update to data ratio")

flags.DEFINE_float('discount', 0.99, 'discount factor')
flags.DEFINE_float('p_aug', 0.5, 'aug prob')

flags.DEFINE_integer('eval_episodes', 50, 'Number of evaluation episodes.')
flags.DEFINE_integer('video_episodes', 0, 'Number of video episodes for each task.')
flags.DEFINE_integer('video_frame_skip', 3, 'Frame skip for videos.')

config_flags.DEFINE_config_file('agent', 'agents/lps.py', lock_config=False)

flags.DEFINE_float('dataset_proportion', 1.0, "Proportion of the dataset to use")
flags.DEFINE_integer('dataset_replace_interval', 1000, 'Dataset replace interval, used for large datasets because of memory constraints')
flags.DEFINE_string('ogbench_dataset_dir', None, 'OGBench dataset directory')

flags.DEFINE_string('droid_dataset_dir', None, 'DROID dataset directory')
flags.DEFINE_bool('droid_use_failure', False, 'Use failure DROID dataset or not')

flags.DEFINE_integer('horizon_length', 5, 'action chunking length.')
flags.DEFINE_bool('sparse', False, "make the task sparse reward")
flags.DEFINE_bool('save_all_online_states', False, "save all trajectories to npy")
flags.DEFINE_bool('record_time', False, "time_rocording")

class LoggingHelper:
    """Helper class to handle logging to WandB."""
    def __init__(self, wandb_logger):
        self.wandb_logger = wandb_logger
        self.first_time = time.time()
        self.last_time = time.time()

    def iterate(self, key, value):
        if 'hist' in key:
            return wandb.Histogram(value)
        else:
            return value

    def log(self, data, step, prefix=None,):
        if prefix is None:
            self.wandb_logger.log({f'{k}': self.iterate(k, v) for k, v in data.items()}, step=step)
        else:
            self.wandb_logger.log({f'{prefix}/{k}': self.iterate(k, v) for k, v in data.items()}, step=step)

def main(_):
    # Parse task configuration if provided
    if FLAGS.task_config != 'NO':
        suite, task_name, alpha, task_num = FLAGS.task_config.split(':')
        FLAGS.task_name = str(task_name)
        FLAGS.agent.alpha = float(alpha)
        FLAGS.task_num = int(task_num)
        if suite == 'OG':
            FLAGS.env_name = f"{task_name}-singletask-task{task_num}-v0"

    exp_name = get_exp_name(FLAGS.seed)
    run = setup_wandb(project=FLAGS.project, group=FLAGS.run_group, name=exp_name)
    run.tags = run.tags + (FLAGS.env_name,)
    
    FLAGS.save_dir = os.path.join(FLAGS.save_dir, wandb.run.project, FLAGS.run_group, FLAGS.env_name, exp_name)
    os.makedirs(FLAGS.save_dir, exist_ok=True)
    flag_dict = get_flag_dict()

    with open(os.path.join(FLAGS.save_dir, 'flags.json'), 'w') as f:
        json.dump(flag_dict, f)

    config = FLAGS.agent
    config.training_steps=FLAGS.offline_steps
    
    # Data loading
    env, eval_env, train_dataset, val_dataset = make_env_and_datasets(
        FLAGS.env_name, 
        droid_dir=FLAGS.droid_dataset_dir,
        droid_use_failure=FLAGS.droid_use_failure,
        sparse=FLAGS.sparse,
        horizon_length=FLAGS.horizon_length,
    )

    # Set seeds
    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)

    log_step = 0
    
    discount = FLAGS.discount
    config["horizon_length"] = FLAGS.horizon_length

    # Handle dataset processing
    def process_train_dataset(dataset, is_dataset=True):
        """
        Process the train dataset to:
            - handle dataset proportion
            - handle sparse reward
            - convert to action chunked dataset
        """

        if is_dataset:
            dataset = Dataset.create(**dataset)
        if FLAGS.dataset_proportion < 1.0:
            new_size = int(len(dataset['masks']) * FLAGS.dataset_proportion)
            dataset = Dataset.create(
                **{k: v[:new_size] for k, v in dataset.items()}
            )
        
        if "puzzle-3x3" in FLAGS.task_name or "scene" in FLAGS.task_name:
            # Create a new dataset with modified rewards instead of trying to modify the frozen one
            sparse_rewards = (dataset["rewards"] != 0.0) * -1.0
            ds_dict = {k: v for k, v in dataset.items()}
            ds_dict["rewards"] = sparse_rewards
            dataset = Dataset.create(**ds_dict)

        dataset.actor_action_sequence = ( FLAGS.horizon_length )
        dataset.critic_action_sequence = ( FLAGS.horizon_length )
        dataset.nstep = 1 # Actually N step
        dataset.discount = FLAGS.discount
        dataset.discount2 = FLAGS.discount
        dataset.p_aug = FLAGS.p_aug
        return dataset
    
    train_dataset = process_train_dataset(train_dataset)
    example_batch = train_dataset.sample(config['batch_size'])

    if config.get('use_DiT', False):
        save_example_batch(example_batch, FLAGS.save_dir)

    print_batch_shapes(example_batch)

    is_droid = True if FLAGS.droid_dataset_dir is not None else False

    agent_class = agents[config['agent_name']]

    agent = agent_class.create(
        FLAGS.seed,
        example_batch['observations'],
        example_batch['actions'],
        config,
    )

    print_param_stats(agent)

    # Setup logging.
    prefixes = ["eval", "env"]
    if FLAGS.offline_steps > 0:
        prefixes.append("offline_agent")

    logger = LoggingHelper(
        wandb_logger=wandb,
    )

    # Offline RL
    print('Offline RL Started', FLAGS.offline_steps)
    for i in tqdm.tqdm(range(1, FLAGS.offline_steps + 1)):
        log_step += 1
        batch = train_dataset.sample(config['batch_size'])
        agent, info = agent.update(batch)

        if i % FLAGS.log_interval == 0:
            logger.log(info, step=log_step)
        
        # Periodic saving and evaluation
        if FLAGS.save_interval > 0 and i % FLAGS.eval_interval == 0:
            if not is_droid and (FLAGS.eval_interval != 0 and i % FLAGS.eval_interval == 0):
                # Using eval, the action chunk is executed fully
                if "bandit" in FLAGS.env_name:
                    from envs.bandit_utils import evaluate
                    eval_info, _, _ = evaluate(
                        agent=agent,
                        env=eval_env,
                        action_dim=example_batch["actions"].shape[-1],
                        num_eval_episodes=FLAGS.eval_episodes,
                        num_video_episodes=FLAGS.video_episodes,
                        video_frame_skip=FLAGS.video_frame_skip,
                    )
                    logger.log(eval_info, log_step, "eval")
                else:
                    from utils.evaluation import evaluate
                    eval_info, _, video = evaluate(
                        agent=agent,
                        env=eval_env,
                        action_dim=example_batch["actions"].shape[-1],
                        num_eval_episodes=FLAGS.eval_episodes,
                        num_video_episodes=FLAGS.video_episodes,
                        video_frame_skip=FLAGS.video_frame_skip,
                    )
                    logger.log(eval_info, log_step, "eval")
                    if len(video) > 0:
                        wandb.log({
                            f"eval_video": wandb.Video(np.vstack(video).transpose(0, 3, 1, 2), fps=20, format="mp4")
                        }, step=log_step)
            if config.get('use_DiT', False):
                save_agent(agent, FLAGS.save_dir, log_step)
    
    if FLAGS.online_steps <= 0:
        return None
    
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
    
    replay_buffer = ReplayBuffer.create_from_initial_dataset(dict(train_dataset), size=FLAGS.buffer_size)
    replay_buffer = process_train_dataset(replay_buffer, False)
    replay_buffer.update_locs()

    ob, _ = env.reset()

    online_rng, rng = jax.random.split(jax.random.PRNGKey(FLAGS.seed), 2)

    action_queue = []
    action_dim = example_batch["actions"].shape[-1]

    update_info = {}
    
    print('Online RL Started', FLAGS.online_steps)

    for j in tqdm.tqdm(range(1, FLAGS.online_steps + 1)):
        log_step += 1
        online_rng, key = jax.random.split(online_rng)

        if len(action_queue) == 0:
            action = agent.sample_actions(observations=ob, rng=key)
            action_chunk = np.array(action).reshape(-1, action_dim)
            for action in action_chunk:
                action_queue.append(action)

        action = action_queue.pop(0)

        next_ob, int_reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        env_info = {}
        for key, value in info.items():
            if key.startswith("distance"):
                env_info[key] = value

        if "puzzle-3x3" in FLAGS.task_name or "scene" in FLAGS.task_name:
            assert int_reward <= 0.0
            int_reward = (int_reward != 0.0) * -1.0

        transition = dict(
            observations=ob,
            actions=action,
            rewards=int_reward,
            terminals=float(done),
            masks=1.0 - terminated,
            next_observations=next_ob,
        )
        replay_buffer.add_transition(transition)

        if done:
            ob, _ = env.reset()
            action_queue = []  # Reset the action queue
        else:
            ob = next_ob

        batch = replay_buffer.sample(FLAGS.utd_ratio * config['batch_size'])
        
        agent, update_info["online_agent"] = agent.update(batch)

        if j % FLAGS.log_interval == 0:
            logger.log(update_info, step=log_step)

        update_info = {}
        
        # Periodic evaluation and saving
        if FLAGS.save_interval > 0 and j % FLAGS.eval_interval == 0:
            if not is_droid and (FLAGS.eval_interval != 0 and j % FLAGS.eval_interval == 0):
                # During eval, the action chunk is executed fully
                if "bandit" in FLAGS.env_name:
                    from envs.bandit_utils import evaluate
                    eval_info, _, _ = evaluate(
                        agent=agent,
                        env=eval_env,
                        action_dim=example_batch["actions"].shape[-1],
                        num_eval_episodes=FLAGS.eval_episodes,
                        num_video_episodes=FLAGS.video_episodes,
                        video_frame_skip=FLAGS.video_frame_skip,
                    )
                    logger.log(eval_info, log_step, "eval")
                else:
                    from utils.evaluation import evaluate
                    eval_info, _, video = evaluate(
                        agent=agent,
                        env=eval_env,
                        action_dim=example_batch["actions"].shape[-1],
                        num_eval_episodes=FLAGS.eval_episodes,
                        num_video_episodes=FLAGS.video_episodes,
                        video_frame_skip=FLAGS.video_frame_skip,
                    )
                    logger.log(eval_info, log_step, "eval")
                    if len(video) > 0:
                        wandb.log({
                            f"eval_video": wandb.Video(np.vstack(video).transpose(0, 3, 1, 2), fps=20, format="mp4")
                        }, step=log_step)
        if FLAGS.save_interval > 0 and j % FLAGS.save_interval == 0:
            save_agent(agent, FLAGS.save_dir, log_step)

if __name__ == '__main__':
    app.run(main)
