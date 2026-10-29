"""Input/output normalisation for the robot datasets.

Both the state and the action are mapped onto [-1, 1] per dimension with their minimum and
maximum -- the range the policy head can express, since it clips its own output there.
Statistics are computed from the training set and written beside the checkpoints, so evaluation
reproduces the mapping the policy was trained under.
"""

import json
import os

import numpy as np

STATS_FILE = 'norm_stats.json'

# Below this the dimension is constant and is left alone; dividing by it would amplify noise.
_EPS = 1e-6


def _centre_scale(x):
    lo, hi = x.min(axis=0), x.max(axis=0)
    scale = (hi - lo) / 2.0
    return (hi + lo) / 2.0, np.where(scale < _EPS, 1.0, scale)


def compute_stats(dataset, observation=None):
    """Per-dimension statistics of one offline dataset.

    Args:
        dataset: dict as returned by the DROID / LeRobot loaders.
        observation: the dataset's camera keys and image shape, recorded so that evaluation can
            check the robot hands over the same observation the policy was trained on.

    Returns:
        A JSON-serialisable dict of lists.
    """
    state_centre, state_scale = _centre_scale(
        np.asarray(dataset['observations']['state'], dtype=np.float64))
    action_centre, action_scale = _centre_scale(
        np.asarray(dataset['actions'], dtype=np.float64))

    stats = dict(
        state_centre=state_centre.tolist(),
        state_scale=state_scale.tolist(),
        action_centre=action_centre.tolist(),
        action_scale=action_scale.tolist(),
    )
    if observation is not None:
        stats['observation'] = observation
    return stats


def _apply(x, stats, key):
    centre = np.asarray(stats[f'{key}_centre'], dtype=np.float32)
    scale = np.asarray(stats[f'{key}_scale'], dtype=np.float32)
    return ((np.asarray(x, dtype=np.float32) - centre) / scale).astype(np.float32)


def normalize_state(state, stats):
    return _apply(state, stats, 'state')


def normalize_action(action, stats):
    return _apply(action, stats, 'action')


def unnormalize_action(action, stats):
    """Map a policy output back to the robot's units."""
    centre = np.asarray(stats['action_centre'], dtype=np.float32)
    scale = np.asarray(stats['action_scale'], dtype=np.float32)
    return (np.asarray(action, dtype=np.float32) * scale + centre).astype(np.float32)


def apply(dataset, stats):
    """Normalise a loaded dataset in place and return it."""
    dataset['observations']['state'] = normalize_state(dataset['observations']['state'], stats)
    dataset['next_observations']['state'] = normalize_state(
        dataset['next_observations']['state'], stats)
    dataset['actions'] = normalize_action(dataset['actions'], stats)
    return dataset


def save(stats, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, STATS_FILE), 'w') as f:
        json.dump(stats, f, indent=2)


def load(save_dir):
    """Read the statistics written beside a checkpoint; None when the run was not normalised."""
    path = os.path.join(save_dir, STATS_FILE)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)
