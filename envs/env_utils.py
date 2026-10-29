import collections
import re
import time

import gymnasium
import numpy as np
from gymnasium.spaces import Box
from utils.datasets import Dataset


class EpisodeMonitor(gymnasium.Wrapper):
    """Environment wrapper to monitor episode statistics."""

    def __init__(self, env, filter_regexes=None):
        super().__init__(env)
        self._reset_stats()
        self.total_timesteps = 0
        self.filter_regexes = filter_regexes if filter_regexes is not None else []

    def _reset_stats(self):
        self.reward_sum = 0.0
        self.episode_length = 0
        self.start_time = time.time()

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)

        # Remove keys that are not needed for logging.
        for filter_regex in self.filter_regexes:
            for key in list(info.keys()):
                if re.match(filter_regex, key) is not None:
                    del info[key]

        self.reward_sum += reward
        self.episode_length += 1
        self.total_timesteps += 1
        info['total'] = {'timesteps': self.total_timesteps}

        if terminated or truncated:
            info['episode'] = {}
            info['episode']['final_reward'] = reward
            info['episode']['return'] = self.reward_sum
            info['episode']['length'] = self.episode_length
            info['episode']['duration'] = time.time() - self.start_time

            if hasattr(self.unwrapped, 'get_normalized_score'):
                info['episode']['normalized_return'] = (
                    self.unwrapped.get_normalized_score(info['episode']['return']) * 100.0
                )

        return observation, reward, terminated, truncated, info

    def reset(self, *args, **kwargs):
        self._reset_stats()
        return self.env.reset(*args, **kwargs)


class FrameStackWrapper(gymnasium.Wrapper):
    """Environment wrapper to stack observations."""

    def __init__(self, env, num_stack):
        super().__init__(env)

        self.num_stack = num_stack
        self.frames = collections.deque(maxlen=num_stack)

        low = np.concatenate([self.observation_space.low] * num_stack, axis=-1)
        high = np.concatenate([self.observation_space.high] * num_stack, axis=-1)
        self.observation_space = Box(low=low, high=high, dtype=self.observation_space.dtype)

    def get_observation(self):
        assert len(self.frames) == self.num_stack
        return np.concatenate(list(self.frames), axis=-1)

    def reset(self, **kwargs):
        ob, info = self.env.reset(**kwargs)
        for _ in range(self.num_stack):
            self.frames.append(ob)
        if 'goal' in info:
            info['goal'] = np.concatenate([info['goal']] * self.num_stack, axis=-1)
        return self.get_observation(), info

    def step(self, action):
        ob, reward, terminated, truncated, info = self.env.step(action)
        self.frames.append(ob)
        return self.get_observation(), reward, terminated, truncated, info

def _normalize(dataset, normalize_dir, stats=None):
    """Normalise a robot dataset.

    With ``stats`` the dataset is put on an existing policy's scale -- offline-to-online reads them
    back from the checkpoint, because recomputing them here would drift from what the policy was
    trained under. Otherwise they are computed and written to ``normalize_dir``.
    """
    meta = dataset.pop('_meta', None)
    from envs import normalization
    if stats is None:
        if normalize_dir is None:
            return dataset
        stats = normalization.compute_stats(dataset, observation=meta)
        normalization.save(stats, normalize_dir)
    return normalization.apply(dataset, stats)


def make_env_and_datasets(env_name, frame_stack=None, action_clip_eps=1e-5, droid_dir=None, droid_use_failure=False, sparse=False, discount=0.99, horizon_length=5, lerobot_dir=None, normalize_dir=None, normalize_stats=None, task=None, image_size=None):
    """Make offline RL environment and datasets.

    Args:
        env_name: Name of the environment or dataset.
        frame_stack: Number of frames to stack.
        action_clip_eps: Epsilon for action clipping.
        normalize_dir: Robot datasets only. When set, state and action are normalised and the
            statistics are written to this directory for evaluation to read back.
        normalize_stats: Normalise with these statistics instead of computing them -- the way
            offline-to-online puts the offline half on the policy's own scale.

    Returns:
        A tuple of the environment, evaluation environment, training dataset, and validation dataset.
    """

    if lerobot_dir is not None:
        # LeRobot dataset (parquet + mp4) -- same observation/reward contract as the DROID path.
        from envs import robot_datasets
        import os
        lerobot_path = os.path.join(lerobot_dir, env_name)
        dataset = robot_datasets.load_lerobot_dataset(lerobot_path, sparse=sparse, discount=discount, horizon_length=horizon_length,
                                                   task=task, image_size=image_size)
        dataset = _normalize(dataset, normalize_dir, normalize_stats)
        train_dataset = Dataset.create(**dataset)
        env, eval_env, val_dataset = None, None, None
    elif droid_dir is not None:
        from envs import robot_datasets
        import os
        droid_path = os.path.join(droid_dir, env_name)
        dataset = robot_datasets.load_droid_dataset(droid_path, include_failure=droid_use_failure, sparse=sparse, discount=discount, horizon_length=horizon_length)
        dataset = _normalize(dataset, normalize_dir, normalize_stats)
        train_dataset = Dataset.create(**dataset)
        env, eval_env, val_dataset = None, None, None
    elif 'singletask' in env_name:
        # OGBench.
        import ogbench
        env, train_dataset, val_dataset = ogbench.make_env_and_datasets(env_name)
        eval_env = ogbench.make_env_and_datasets(env_name, env_only=True)
        env = EpisodeMonitor(env, filter_regexes=['.*privileged.*', '.*proprio.*'])
        eval_env = EpisodeMonitor(eval_env, filter_regexes=['.*privileged.*', '.*proprio.*'])
        train_dataset = Dataset.create(**train_dataset)
        val_dataset = Dataset.create(**val_dataset)
    else:
        raise ValueError(f'Unsupported environment: {env_name}')

    if frame_stack is not None:
        env = FrameStackWrapper(env, frame_stack)
        eval_env = FrameStackWrapper(eval_env, frame_stack)

    if env is not None:
        env.reset()
    if eval_env is not None:
        eval_env.reset()

    # Clip dataset actions.
    if action_clip_eps is not None:
        train_dataset = train_dataset.copy(
            add_or_replace=dict(actions=np.clip(train_dataset['actions'], -1 + action_clip_eps, 1 - action_clip_eps))
        )
        if val_dataset is not None:
            val_dataset = val_dataset.copy(
                add_or_replace=dict(actions=np.clip(val_dataset['actions'], -1 + action_clip_eps, 1 - action_clip_eps))
            )

    return env, eval_env, train_dataset, val_dataset
