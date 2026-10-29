"""Robot adapters -- one per embodiment.

An adapter is the robot. Constructing one opens it, and ``eval.py`` calls ``reset``, ``gate``,
``observe``, ``act`` and ``close`` on it. ``observe`` and ``act`` also convert between the robot's
conventions and the dataset's, which ``check_contract`` verifies on the first tick.

Import the SDK inside the methods, not at module level, so one robot's adapter stays readable
without its SDK installed. Copy ``template.py`` to add one.
"""

import numpy as np

ADAPTERS = {}


def register(name):
    """Register an adapter class under ``name``."""
    def wrap(cls):
        ADAPTERS[name] = cls
        return cls
    return wrap


def get(name):
    if name not in ADAPTERS:
        raise KeyError(f'unknown robot {name!r}; available: {sorted(ADAPTERS)}')
    return ADAPTERS[name]


def available():
    return sorted(ADAPTERS)


class Adapter:
    """What an embodiment has to provide."""

    def __init__(self, image_size=224, **kwargs):
        """Open the robot."""
        raise NotImplementedError

    def reset(self):
        """Start a new trajectory. A robot that needs no homing can leave this alone."""

    def gate(self):
        """Block while the operator has motion disabled.

        This pauses the loop, it does not end the trajectory -- paused time is simply not driven
        and not recorded. A platform with no such control leaves it alone and never pauses.
        """

    def observe(self):
        """Read the robot: ``{'state': (S,), 'image': (H, W, 3N)}``, RGB, in the dataset's order."""
        raise NotImplementedError

    def act(self, action):
        """Send the policy's action vector, converting it to the robot's command first."""
        raise NotImplementedError

    def close(self):
        """Release the robot. Called even when the session raises."""

    # -- online collection ------------------------------------------------------------------------
    control_hz = 15.0

    def wait_to_start(self, msg=None):
        """Block until the operator says to begin. Returns their signal, if the robot has one."""
        raise NotImplementedError

    def outcome(self):
        """``'success'``, ``'failure'``, or None while the trajectory is still running."""
        raise NotImplementedError


def check_contract(observation, action, stats, strict=True):
    """Compare one adapted observation and action against what the policy was trained on.

    Args:
        observation: the adapter's output, ``{'state': ..., 'image': ...}``.
        action: one action, before the adapter maps it to the robot's command format.
        stats: the contents of ``norm_stats.json``.
        strict: raise on a mismatch instead of only reporting it.

    Returns:
        The list of mismatches, empty when everything matches.
    """
    problems = []
    state = np.asarray(observation['state'])
    image = np.asarray(observation['image'])

    trained_state = len(stats['state_centre'])
    if state.shape[-1] != trained_state:
        problems.append(f'state is {state.shape[-1]}-D, the policy was trained on {trained_state}-D')

    trained_action = len(stats['action_centre'])
    if np.asarray(action).shape[-1] != trained_action:
        problems.append(f'action is {np.asarray(action).shape[-1]}-D, '
                        f'the policy was trained on {trained_action}-D')

    spec = stats.get('observation')
    if spec is not None:
        if tuple(image.shape[-3:]) != tuple(spec['image_shape']):
            problems.append(f'image is {tuple(image.shape[-3:])}, '
                            f'the policy was trained on {tuple(spec["image_shape"])}')
        n_trained = len(spec['camera_keys'])
        n_given = image.shape[-1] // 3
        if n_given != n_trained:
            problems.append(f'{n_given} cameras, the policy was trained on {n_trained} '
                            f'in the order {spec["camera_keys"]}')

    if problems and strict:
        raise ValueError('this robot does not match the checkpoint:\n  - ' + '\n  - '.join(problems))
    return problems


from . import droid   # noqa: E402,F401  -- registers 'droid'
