import dataclasses
from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from flax.core.frozen_dict import FrozenDict
from tqdm import tqdm
import os
import pickle

## Dataset code from DEAS
def get_size(data):
    """Return the size of the dataset."""
    sizes = jax.tree_util.tree_map(lambda arr: len(arr), data)
    return max(jax.tree_util.tree_leaves(sizes))


@partial(jax.jit, static_argnames=('padding',))
def random_crop(img, crop_from, padding):
    """Randomly crop an image.

    Args:
        img: Image to crop.
        crop_from: Coordinates to crop from.
        padding: Padding size.
    """
    padded_img = jnp.pad(img, ((padding, padding), (padding, padding), (0, 0)), mode='edge')
    return jax.lax.dynamic_slice(padded_img, crop_from, img.shape)


@partial(jax.jit, static_argnames=('padding',))
def batched_random_crop(imgs, crop_froms, padding):
    """Batched version of random_crop."""
    return jax.vmap(random_crop, (0, 0, None))(imgs, crop_froms, padding)

class Dataset(FrozenDict):
    """Dataset class."""

    @classmethod
    def create(cls, freeze=True, **fields):
        """Create a dataset from the fields.

        Args:
            freeze: Whether to freeze the arrays.
            **fields: Keys and values of the dataset.
        """
        data = fields
        assert 'observations' in data
        if freeze:
            jax.tree_util.tree_map(lambda arr: arr.setflags(write=False), data)
        return cls(data)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.size = get_size(self._dict)
        self.frame_stack = None  # Number of frames to stack; set outside the class.
        self.p_aug = None  # Image augmentation probability; set outside the class.
        self.return_next_actions = False  # Whether to additionally return next actions; set outside the class.
        self.action_sequence = 1  # Action sequence length; set outside the class.
        self.discount = 1.0  # Discount factor; set outside the class.

        self.terminal_locs = np.nonzero(self['terminals'] > 0)[0]
        if len(self.terminal_locs) == 0:
            self.terminal_locs = np.array([self.size - 1])
        self.initial_locs = np.concatenate([[0], self.terminal_locs[:-1] + 1])

        self._valid_indices = None

    def update_locs(self):
        self._valid_indices = None
        self.terminal_locs = np.nonzero(self['terminals'] > 0)[0]
        if len(self.terminal_locs) == 0:
            self.terminal_locs = np.array([self.size - 1])
        self.initial_locs = np.concatenate([[0], self.terminal_locs[:-1] + 1])

    @property
    def valid_indices(self):
        """Get valid indices that won't cross episode boundaries."""
        if self._valid_indices is None:
            valid_indices = []
            for start, end in zip(self.initial_locs, np.append(self.terminal_locs, self.size)):
                traj_len = end - start + 1
                valid_end = start + max(0, traj_len - (self.action_sequence - 1))
                if valid_end > start:
                    valid_indices.extend(range(start, valid_end))
            self._valid_indices = np.array(valid_indices)
        return self._valid_indices

    def get_random_idxs(self, num_idxs):
        """Return `num_idxs` random indices."""
        valid_indices = self.valid_indices
        assert len(valid_indices) > 0, 'No valid indices found'

        # Sample from valid indices
        rand_idx = np.random.randint(len(valid_indices), size=num_idxs)
        return valid_indices[rand_idx]

    def sample(self, batch_size: int, idxs=None):
        """Sample a batch of transitions."""
        if idxs is None:
            idxs = self.get_random_idxs(batch_size)
        batch = self.get_subset(idxs)

        if self.action_sequence > 1 and self._dict.get('rewards') is not None:
            # Find next episode start for each idx to avoid crossing episode boundaries
            search_idxs = np.searchsorted(self.terminal_locs, idxs)
            clipped_search_idxs = np.minimum(search_idxs, len(self.terminal_locs) - 1)
            next_episode_starts = np.where(
                search_idxs < len(self.terminal_locs), self.terminal_locs[clipped_search_idxs], self.size - 1
            )

            # Create sequence indices [batch, action_sequence]
            seq_offsets = np.arange(self.action_sequence)[None, :]  # [1, action_sequence]
            seq_indices = idxs[:, None] + seq_offsets  # [batch, action_sequence]
            seq_indices = np.minimum(seq_indices, next_episode_starts[:, None])

            # Get rewards and masks for all steps
            rewards = self._dict['rewards'][seq_indices]  # [batch, action_sequence]
            masks = self._dict['masks'][seq_indices]  # [batch, action_sequence]

            # Compute discounted return: r₀ + γr₁ + γ²r₂ + ...
            discount_powers = self.discount ** np.arange(self.action_sequence)[None, :]
            batch['rewards'] = np.sum(rewards * discount_powers, axis=1)  # [batch]

            # Mask: 0 if any step hits terminal
            batch['masks'] = np.prod(masks, axis=1)  # [batch]

            # Get next observations at end of action sequence
            next_obs_idxs = np.minimum(idxs + self.action_sequence - 1, next_episode_starts)
            batch['next_observations'] = jax.tree_util.tree_map(
                lambda arr: arr[next_obs_idxs], 
                self._dict['next_observations']
            )

        batch['valid'] = np.ones_like(batch['masks'])

        if self.frame_stack is not None:
            # Stack frames.
            initial_state_idxs = self.initial_locs[np.searchsorted(self.initial_locs, idxs, side='right') - 1]
            obs = []  # Will be [ob[t - frame_stack + 1], ..., ob[t]].
            next_obs = []  # Will be [ob[t - frame_stack + 1 + N], ..., ob[t + N]].

            # Get current observation stack
            for i in reversed(range(self.frame_stack)):
                # Use the initial state if the index is out of bounds.
                cur_idxs = np.maximum(idxs - i, initial_state_idxs)
                obs.append(jax.tree_util.tree_map(lambda arr: arr[cur_idxs], self['observations']))

            # Get next observation stack
            for i in reversed(range(self.frame_stack)):
                next_idxs = np.maximum(idxs + (self.action_sequence - 1) - i, initial_state_idxs)
                next_obs.append(jax.tree_util.tree_map(lambda arr: arr[next_idxs], self['observations']))

            batch['observations'] = jax.tree_util.tree_map(lambda *args: np.concatenate(args, axis=-1), *obs)
            batch['next_observations'] = jax.tree_util.tree_map(lambda *args: np.concatenate(args, axis=-1), *next_obs)
        if self.p_aug is not None:
            # Apply random-crop image augmentation.
            if np.random.rand() < self.p_aug:
                self.augment(batch, ['observations', 'next_observations'])
        return batch

    def _get_action_sequences(self, idxs):
        """Get action sequences starting from each idx, clipped at episode boundaries."""
        # Find next episode start for each idx to avoid crossing episode boundaries
        search_idxs = np.searchsorted(self.terminal_locs, idxs)
        clipped_search_idxs = np.minimum(search_idxs, len(self.terminal_locs) - 1)
        next_episode_starts = np.where(
            search_idxs < len(self.terminal_locs), self.terminal_locs[clipped_search_idxs], self.size - 1
        )

        # Create sequence indices for each idx
        seq_indices = np.expand_dims(idxs, 1) + np.arange(self.action_sequence)[None, :]

        # Clip sequence indices to stay within episodes
        seq_indices = np.minimum(seq_indices, np.expand_dims(next_episode_starts, 1))

        # Get action sequences
        action_sequences = self._dict['actions'][seq_indices]
        return action_sequences

    def get_subset(self, idxs):
        """Return a subset of the dataset given the indices."""
        result = jax.tree_util.tree_map(lambda arr: arr[idxs], self._dict)
        if self.return_next_actions:
            # WARNING: This is incorrect at the end of the trajectory. Use with caution.
            result['next_actions'] = self._dict['actions'][np.minimum(idxs + 1, self.size - 1)]

        result['actions'] = self._get_action_sequences(idxs)
        if self.return_next_actions:
            result['next_actions'] = self._get_action_sequences(np.minimum(idxs + 1, self.size - 1))
        return result

    def augment(self, batch, keys):
        """Apply image augmentation to the given keys."""
        padding = 3
        batch_size = get_size(batch[keys[0]])
        
        crop_froms = np.random.randint(0, 2 * padding + 1, (batch_size, 2))
        crop_froms = np.concatenate([crop_froms, np.zeros((batch_size, 1), dtype=np.int64)], axis=1)
        for key in keys:
            batch[key] = jax.tree_util.tree_map(
                lambda arr: np.array(batched_random_crop(arr, crop_froms, padding)) if len(arr.shape) == 4 else arr,
                batch[key],
            )


class ReplayBuffer(Dataset):
    """Replay buffer class.

    This class extends Dataset to support adding transitions.
    """

    @classmethod
    def create(cls, transition, size):
        """Create a replay buffer from the example transition.

        Args:
            transition: Example transition (dict).
            size: Size of the replay buffer.
        """

        def create_buffer(example):
            example = np.array(example)
            return np.zeros((size, *example.shape), dtype=example.dtype)

        buffer_dict = jax.tree_util.tree_map(create_buffer, transition)
        return cls(buffer_dict)

    @classmethod
    def create_from_initial_dataset(cls, init_dataset, size):
        """Create a replay buffer from the initial dataset.

        Args:
            init_dataset: Initial dataset.
            size: Size of the replay buffer.
        """

        def create_buffer(init_buffer):
            buffer = np.zeros((size, *init_buffer.shape[1:]), dtype=init_buffer.dtype)
            buffer[: len(init_buffer)] = init_buffer
            return buffer

        buffer_dict = jax.tree_util.tree_map(create_buffer, init_dataset)
        dataset = cls(buffer_dict)
        dataset.size = dataset.pointer = get_size(init_dataset)
        return dataset

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.max_size = get_size(self._dict)
        self.size = 0
        self.pointer = 0

    def add_transition(self, transition):
        """Add a transition to the replay buffer."""

        def set_idx(buffer, new_element):
            buffer[self.pointer] = new_element

        jax.tree_util.tree_map(set_idx, self._dict, transition)
        self.pointer = (self.pointer + 1) % self.max_size
        self.size = max(self.pointer, self.size)

    def clear(self):
        """Clear the replay buffer."""
        self.size = self.pointer = 0

def save_compact_buffer(buffer, path):
    """Saves only the data that has actually been added."""
    compact_data = jax.tree_util.tree_map(lambda x: x[:buffer.size], buffer._dict)
    with open(path, 'wb') as f:
        pickle.dump(compact_data, f)

def load_compact_buffer(path, max_size):
    """Loads a saved dataset and expands it into a ReplayBuffer of max_size."""
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return ReplayBuffer.create_from_initial_dataset(data, max_size)

@dataclasses.dataclass
class GCDataset:
    """Dataset class for goal-conditioned RL.

    This class provides a method to sample a batch of transitions with goals (value_goals and actor_goals) from the
    dataset. The goals are sampled from the current state, future states in the same trajectory, and random states.

    It reads the following keys from the config:
    - discount: Discount factor for geometric sampling.
    - value_p_curgoal: Probability of using the current state as the value goal.
    - value_p_trajgoal: Probability of using a future state in the same trajectory as the value goal.
    - value_p_randomgoal: Probability of using a random state as the value goal.
    - value_geom_sample: Whether to use geometric sampling for future value goals.
    - actor_p_curgoal: Probability of using the current state as the actor goal.
    - actor_p_trajgoal: Probability of using a future state in the same trajectory as the actor goal.
    - actor_p_randomgoal: Probability of using a random state as the actor goal.
    - actor_geom_sample: Whether to use geometric sampling for future actor goals.
    - gc_negative: Whether to use '0 if s == g else -1' (True) or '1 if s == g else 0' (False) as the reward.

    Attributes:
        dataset: Dataset object.
        config: Configuration dictionary.
    """

    dataset: Dataset
    config: Any

    def __post_init__(self):
        self.size = self.dataset.size

        # Pre-compute trajectory boundaries.
        (self.terminal_locs,) = np.nonzero(self.dataset['terminals'] > 0)
        self.initial_locs = np.concatenate([[0], self.terminal_locs[:-1] + 1])
        assert self.terminal_locs[-1] == self.size - 1

        # Assert probabilities sum to 1.
        assert np.isclose(
            self.config['value_p_curgoal'] + self.config['value_p_trajgoal'] + self.config['value_p_randomgoal'], 1.0
        )
        assert np.isclose(
            self.config['actor_p_curgoal'] + self.config['actor_p_trajgoal'] + self.config['actor_p_randomgoal'], 1.0
        )

    def sample(self, batch_size: int, idxs=None, evaluation=False):
        """Sample a batch of transitions with goals.

        This method samples a batch of transitions with goals (value_goals and actor_goals) from the dataset. They are
        stored in the keys 'value_goals' and 'actor_goals', respectively. It also computes the 'rewards' and 'masks'
        based on the indices of the goals.

        Args:
            batch_size: Batch size.
            idxs: Indices of the transitions to sample. If None, random indices are sampled.
            evaluation: Whether to sample for evaluation.
        """
        if idxs is None:
            idxs = self.dataset.get_random_idxs(batch_size)

        batch = self.dataset.sample(batch_size, idxs)

        value_goal_idxs = self.sample_goals(
            idxs,
            self.config['value_p_curgoal'],
            self.config['value_p_trajgoal'],
            self.config['value_p_randomgoal'],
            self.config['value_geom_sample'],
        )
        actor_goal_idxs = self.sample_goals(
            idxs,
            self.config['actor_p_curgoal'],
            self.config['actor_p_trajgoal'],
            self.config['actor_p_randomgoal'],
            self.config['actor_geom_sample'],
        )

        if 'oracle_reps' in self.dataset:
            batch['value_goals'] = self.dataset['oracle_reps'][value_goal_idxs]
            batch['actor_goals'] = self.dataset['oracle_reps'][actor_goal_idxs]
        else:
            batch['value_goals'] = self.get_observations(value_goal_idxs)
            batch['actor_goals'] = self.get_observations(actor_goal_idxs)
        successes = (idxs == value_goal_idxs).astype(float)
        batch['masks'] = 1.0 - successes
        batch['rewards'] = successes - (1.0 if self.config['gc_negative'] else 0.0)

        return batch

    def sample_goals(self, idxs, p_curgoal, p_trajgoal, p_randomgoal, geom_sample, discount=None):
        """Sample goals for the given indices."""
        batch_size = len(idxs)
        if discount is None:
            discount = self.config['discount']

        # Random goals.
        random_goal_idxs = self.dataset.get_random_idxs(batch_size)

        # Goals from the same trajectory (excluding the current state, unless it is the final state).
        final_state_idxs = self.terminal_locs[np.searchsorted(self.terminal_locs, idxs)]
        if geom_sample:
            # Geometric sampling.
            offsets = np.random.geometric(p=1 - discount, size=batch_size)  # in [1, inf)
            traj_goal_idxs = np.minimum(idxs + offsets, final_state_idxs)
        else:
            # Uniform sampling.
            distances = np.random.rand(batch_size)  # in [0, 1)
            traj_goal_idxs = np.round(
                (
                    np.minimum(idxs + self.dataset.action_sequence, final_state_idxs) * distances
                    + final_state_idxs * (1 - distances)
                )
            ).astype(int)
        if p_curgoal == 1.0:
            goal_idxs = idxs
        else:
            goal_idxs = np.where(
                np.random.rand(batch_size) < p_trajgoal / (1.0 - p_curgoal), traj_goal_idxs, random_goal_idxs
            )

            # Goals at the current state.
            goal_idxs = np.where(np.random.rand(batch_size) < p_curgoal, idxs, goal_idxs)

        return goal_idxs

    def get_observations(self, idxs):
        """Return the observations for the given indices."""
        return jax.tree_util.tree_map(lambda arr: arr[idxs], self.dataset['observations'])


@dataclasses.dataclass
class HGCDataset(GCDataset):
    """Dataset class for hierarchical goal-conditioned RL.

    This class extends GCDataset to support hierarchical goal-conditioned RL. It reads the following additional key from
    the config:
    - subgoal_steps (optional: value_subgoal_steps, actor_subgoal_steps): Subgoal steps. It is also possible to specify
        `value_subgoal_steps` and `actor_subgoal_steps` separately.
    - low_discount: If specified, return low-level value goals as well.
    """

    def compute_high_next_idxs(self, idxs, final_state_idxs, high_goal_idxs, subgoal_steps):
        """Compute the next indices for high-level goals."""
        batch_size = len(idxs)
        subgoal_steps = np.full(batch_size, subgoal_steps)

        # Ensure minimum distance for action sequences
        min_distance = np.maximum(self.dataset.action_sequence, 1)
        subgoal_steps = np.maximum(subgoal_steps, min_distance)

        # Clip to the end of the trajectory.
        subgoal_steps = np.minimum(subgoal_steps, final_state_idxs - idxs)

        # Clip to the high-level goal.
        diff_idxs = high_goal_idxs - idxs
        should_clip = (0 <= diff_idxs) & (diff_idxs < subgoal_steps)
        subgoal_steps = np.where(should_clip, diff_idxs, subgoal_steps)

        return idxs + subgoal_steps, subgoal_steps

    def get_high_actions(self, target_idxs, cur_idxs):
        if 'oracle_reps' in self.dataset:
            return self.dataset['oracle_reps'][target_idxs]
        else:
            return self.get_observations(target_idxs)

    def sample(self, batch_size: int, idxs=None, evaluation=False):
        if idxs is None:
            idxs = self.dataset.get_random_idxs(batch_size)

        batch = self.dataset.sample(batch_size, idxs)

        final_state_idxs = self.terminal_locs[np.searchsorted(self.terminal_locs, idxs)]

        # Sample high-level value goals.
        high_value_goal_idxs = self.sample_goals(
            idxs,
            self.config['value_p_curgoal'],
            self.config['value_p_trajgoal'],
            self.config['value_p_randomgoal'],
            self.config['value_geom_sample'],
        )
        value_subgoal_steps = (
            self.config['subgoal_steps']
            if self.config.get('value_subgoal_steps') is None
            else self.config['value_subgoal_steps']
        )
        high_value_next_idxs, high_value_subgoal_steps = self.compute_high_next_idxs(
            idxs,
            final_state_idxs,
            high_value_goal_idxs,
            value_subgoal_steps,
        )

        if 'oracle_reps' in self.dataset:
            batch['high_value_reps'] = self.dataset['oracle_reps'][idxs]
            batch['high_value_goals'] = self.dataset['oracle_reps'][high_value_goal_idxs]
            batch['high_value_actions'] = self.get_high_actions(high_value_next_idxs, idxs)
            batch['high_value_next_observations'] = self.get_observations(high_value_next_idxs)
        else:
            batch['high_value_reps'] = batch['observations']
            batch['high_value_goals'] = self.get_observations(high_value_goal_idxs)
            batch['high_value_actions'] = self.get_high_actions(high_value_next_idxs, idxs)
            batch['high_value_next_observations'] = self.get_observations(high_value_next_idxs)
        batch['high_value_offsets'] = high_value_goal_idxs - idxs

        high_value_successes = (high_value_subgoal_steps < value_subgoal_steps).astype(float)
        batch['high_value_subgoal_steps'] = high_value_subgoal_steps
        batch['high_value_masks'] = 1.0 - high_value_successes
        if self.config['gc_negative']:
            batch['high_value_rewards'] = -(1 - self.config['discount'] ** high_value_subgoal_steps) / (
                1 - self.config['discount']
            )
        else:
            batch['high_value_rewards'] = (self.config['discount'] ** high_value_subgoal_steps) * high_value_successes

        # Sample low-level value goals (if requested).
        if 'low_discount' in self.config:
            low_value_goal_idxs = self.sample_goals(
                idxs,
                self.config['value_p_curgoal'],
                self.config['value_p_trajgoal'],
                self.config['value_p_randomgoal'],
                geom_sample=True,
                discount=self.config['low_discount'],
            )

            if 'oracle_reps' in self.dataset:
                batch['low_value_goals'] = self.dataset['oracle_reps'][low_value_goal_idxs]
            else:
                batch['low_value_goals'] = self.get_observations(low_value_goal_idxs)
            successes = (idxs == low_value_goal_idxs).astype(float)
            batch['low_value_masks'] = 1.0 - successes
            batch['low_value_rewards'] = successes - (1.0 if self.config['gc_negative'] else 0.0)

        # One-step information.
        successes = (idxs == high_value_goal_idxs).astype(float)
        batch['masks'] = 1.0 - successes
        batch['rewards'] = successes - (1.0 if self.config['gc_negative'] else 0.0)

        # Sample high-level actor goals.
        high_actor_goal_idxs = self.sample_goals(
            idxs,
            self.config['actor_p_curgoal'],
            self.config['actor_p_trajgoal'],
            self.config['actor_p_randomgoal'],
            self.config['actor_geom_sample'],
        )
        actor_subgoal_steps = (
            self.config['subgoal_steps']
            if self.config.get('actor_subgoal_steps') is None
            else self.config['actor_subgoal_steps']
        )
        high_actor_next_idxs, high_actor_subgoal_steps = self.compute_high_next_idxs(
            idxs,
            final_state_idxs,
            high_actor_goal_idxs,
            actor_subgoal_steps,
        )

        if 'oracle_reps' in self.dataset:
            batch['high_actor_goals'] = self.dataset['oracle_reps'][high_actor_goal_idxs]
            batch['high_actor_actions'] = self.get_high_actions(high_actor_next_idxs, idxs)
            batch['high_actor_next_observations'] = self.get_observations(high_actor_next_idxs)
        else:
            batch['high_actor_goals'] = self.get_observations(high_actor_goal_idxs)
            batch['high_actor_actions'] = self.get_high_actions(high_actor_next_idxs, idxs)
            batch['high_actor_next_observations'] = self.get_observations(high_actor_next_idxs)

        # Compute low-level actor goals.
        min_low_actor_distance = np.maximum(self.dataset.action_sequence, 1)
        low_actor_goal_idxs = np.minimum(
            np.maximum(idxs + actor_subgoal_steps, idxs + min_low_actor_distance), final_state_idxs
        )

        batch['low_actor_goals'] = self.get_high_actions(low_actor_goal_idxs, idxs)

        return batch