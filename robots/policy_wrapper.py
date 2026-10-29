import numpy as np
import jax

import robots
from envs import normalization
from utils.evaluation import supply_rng


def preprocess_observation(observation, norm_stats=None):
    """Normalise a policy observation, as the dataset was normalised at training time."""
    if norm_stats is None:
        return observation
    return dict(observation, state=normalization.normalize_state(observation['state'], norm_stats))


class JAXWrapper:
    """Run a chunked policy one control tick at a time, on the robot's own action scale.

    Observations arrive raw and are normalised here; the first action is checked against the
    robot's contract.
    """

    normalize_observations = True
    verify_contract = True

    def __init__(self, agent, norm_stats=None):
        self.agent = agent
        self.horizon_length = self.agent.config['horizon_length']
        self.horizon_counter = 0
        self.action = 0
        self.norm_stats = norm_stats
        self._checked = False
        # Set to score the chunk the policy just chose; one entry per decision in `values`.
        self.value_fn = None
        self.values = []
        rng = jax.random.PRNGKey(np.random.randint(0, 2**32))
        self.actor_fn = supply_rng(agent.sample_actions, rng=rng)

    def reset(self):
        self.action = 0
        self.horizon_counter = 0
        self.values = []

    def forward(self, observation):
        if self.horizon_counter == 0:
            if self.normalize_observations:
                p_obs = preprocess_observation(observation, self.norm_stats)
            else:
                p_obs = observation
            self.action = self.actor_fn(observations=p_obs)
            self._obs = p_obs
            if self.value_fn is not None:
                self.values.append(self.value_fn(p_obs, self.action))
        output_action = self.action[self.horizon_counter]
        self.horizon_counter += 1

        if self.horizon_counter == self.horizon_length:
            self.horizon_counter = 0

        if self.norm_stats is not None:
            output_action = normalization.unnormalize_action(output_action, self.norm_stats)
            if self.verify_contract and not self._checked:
                # The first tick decides whether this robot and this checkpoint agree.
                robots.check_contract(self._obs, output_action, self.norm_stats)
                self._checked = True

        return output_action


class JAXWrapperOnline(JAXWrapper):
    """The same, for observations that arrive already normalised, as ``OnlineRobotEnv`` returns them."""

    normalize_observations = False
    verify_contract = False
