"""The online half of offline-to-online: one trajectory at a time, on a robot, with an operator.

Nothing here is embodiment-specific -- the robot is a ``robots`` adapter, and the three hooks the
operator needs (``wait_to_start``, ``gate``, ``outcome``) come from it.

Transitions are stored the way the offline dataset was stored, normalised. The two halves of a
mixed batch have to live on the same scale, and the offline half is normalised at load time.
"""

import time

from envs import normalization
from envs.robot_datasets import outcome_reward


class OnlineRobotEnv:
    def __init__(self, robot, sparse=False, discount=0.99, norm_stats=None):
        self.robot = robot
        self.sparse = sparse
        self.discount = discount
        self.norm_stats = norm_stats
        self._next_step = 0.0

    def _observation(self):
        obs = self.robot.observe()
        if self.norm_stats is None:
            return obs
        return dict(obs, state=normalization.normalize_state(obs['state'], self.norm_stats))

    def normalize_action(self, action):
        """The command as the buffer stores it -- the scale the offline actions are on."""
        if self.norm_stats is None:
            return action
        return normalization.normalize_action(action, self.norm_stats)

    def reset(self):
        self.robot.wait_to_start('Press A to reset the environment')
        print('Resetting...')
        self.robot.reset()
        self._next_step = time.time()
        print('Reset done.')
        return self._observation()

    def get_observation(self):
        return self._observation()

    def wait_for_noskip(self):
        self.robot.gate()

    def wait_for_controller(self, msg='Press A to continue'):
        return self.robot.wait_to_start(msg)

    def step(self, action):
        """One control tick. Returns the next observation, its reward and whether it ended.

        The reward is the one the datasets are built with (``envs.robot_datasets.outcome_reward``),
        from the operator's verdict.
        """
        self._next_step = time.time()
        self.robot.act(action)
        obs = self._observation()

        verdict = self.robot.outcome()
        self.last_outcome = verdict  # 'success' | 'failure' | None, from the operator's controller
        done = verdict is not None
        reward = outcome_reward(verdict, self.sparse, self.discount)

        sleep_left = 1.0 / self.robot.control_hz - (time.time() - self._next_step)
        if sleep_left > 0:
            time.sleep(sleep_left)

        return obs, reward, done
