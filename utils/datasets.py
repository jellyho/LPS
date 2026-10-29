from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from flax.core.frozen_dict import FrozenDict
import pickle

# Dataset code adapted from DEAS.
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
            self._valid_indices = np.array(valid_indices, dtype=np.int64)
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
        if self._valid_indices is None:
            _ = self.valid_indices

        idx = self.pointer
        overwriting = self.size == self.max_size or idx < self.size

        def set_idx(buffer, new_element):
            buffer[idx] = new_element

        jax.tree_util.tree_map(set_idx, self._dict, transition)
        self.pointer = (self.pointer + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

        if overwriting:
            self.update_locs()
            return

        terminal = bool(np.asarray(transition['terminals']) > 0)
        if terminal:
            self.terminal_locs = np.append(self.terminal_locs, idx)
            self.initial_locs = np.append(self.initial_locs, idx + 1)

        prev_terminals = self.terminal_locs[self.terminal_locs < idx]
        current_ep_start = prev_terminals[-1] + 1 if len(prev_terminals) > 0 else 0
        current_ep_len = idx - current_ep_start + 1
        if current_ep_len >= self.action_sequence:
            self._valid_indices = np.append(self._valid_indices, idx - self.action_sequence + 1)

    def clear(self):
        """Clear the replay buffer."""
        self.size = self.pointer = 0
        self._valid_indices = None

def save_compact_buffer(buffer, path, norm_stats=None):
    """Saves only the data that has actually been added.

    ``norm_stats`` is stored with it: the transitions are on that scale, and reloading them against
    a policy normalised differently would be silently wrong.
    """
    compact_data = jax.tree_util.tree_map(lambda x: x[:buffer.size], buffer._dict)
    with open(path, 'wb') as f:
        pickle.dump({'data': compact_data, 'norm_stats': norm_stats}, f)

def load_compact_buffer(path, max_size, norm_stats=None):
    """Loads a saved dataset and expands it into a ReplayBuffer of max_size."""
    with open(path, 'rb') as f:
        saved = pickle.load(f)
    if isinstance(saved, dict) and set(saved) == {'data', 'norm_stats'}:
        if norm_stats is not None and saved['norm_stats'] != norm_stats:
            raise ValueError(f'{path} was recorded under different normalisation statistics')
        saved = saved['data']
    return ReplayBuffer.create_from_initial_dataset(saved, max_size)

