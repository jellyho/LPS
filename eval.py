"""Evaluate a checkpoint on a robot.

Nothing here is embodiment-specific: the checkpoint is restored, wrapped with the normalisation it
was trained under, and driven against the adapter named by ``--robot``. See ``robots/``.

The robot is driven only while the operator allows motion (``gate``); the trajectory itself ends
when they press Enter, or at ``--max_steps``.

An adapter that judges its own trajectories (one that implements ``outcome``, e.g. a simulator)
needs no operator: ``--episodes N`` runs N of them and reports the success rate. ``--video N``
writes the first N trajectories out as mp4, which is the only way to see what a failure actually
looked like.
"""

import functools
import json
import os
import select
import sys

import ml_collections
import numpy as np
from absl import app, flags

import robots
from agents import agents
from robots.policy_wrapper import JAXWrapper
from utils.evaluation import supply_rng
from envs import normalization
from utils.flax_utils import restore_agent, load_example_batch, print_batch_shapes

if 'CUDA_VISIBLE_DEVICES' in os.environ:
    os.environ['EGL_DEVICE_ID'] = os.environ['CUDA_VISIBLE_DEVICES']
    os.environ['MUJOCO_EGL_DEVICE_ID'] = os.environ['CUDA_VISIBLE_DEVICES']

FLAGS = flags.FLAGS

flags.DEFINE_string('checkpoint_path', '', 'Checkpoint path')
flags.DEFINE_integer('checkpoint_step', 1000000, 'Checkpoint step')
flags.DEFINE_string('robot', 'droid', 'Robot adapter')
flags.DEFINE_integer('image_size', 224, 'Camera resolution')
flags.DEFINE_integer('max_steps', 0, 'Cap a trajectory at this many steps; 0 for no cap')
flags.DEFINE_integer('episodes', 0, 'Run this many trajectories and report the success rate; '
                                    '0 waits on the operator between them')
flags.DEFINE_integer('video', 0, 'Write the first this many trajectories out as mp4')
flags.DEFINE_string('video_dir', '', 'Where the videos go; defaults to beside the checkpoint')
flags.DEFINE_integer('control_freq', 0, 'Drive the robot at this rate instead of the adapter\'s '
                                        'default; match the rate the dataset was recorded at')
flags.DEFINE_integer('nfe', 1, 'Number of decoding steps for the action (LPS only)')
flags.DEFINE_bool('base_policy', False, 'Decode the BC flow from its own prior, with no latent '
                                        'actor -- what the policy is before it is steered')
flags.DEFINE_integer('seed', 100, 'Random seed')
flags.DEFINE_integer('value_trace', 0, 'Record what the critic thinks of every chunk the policy '
                                       'chooses, along the first this many trajectories, and '
                                       'write it beside the videos')


def interrupted():
    """True once the operator has pressed Enter. Never blocks the control loop."""
    if not select.select([sys.stdin], [], [], 0)[0]:
        return False
    sys.stdin.readline()
    return True


def base_policy_fn(agent, config, steps=1):
    """The BC flow decoded from its own prior -- the base policy, with the steering removed.

    Separates the two halves of the method: a base policy that cannot imitate is a different
    problem from a latent actor that steers it somewhere worse.

    The noise comes from the distribution the agent sampled x_0 from in training, Gaussian for flow
    matching and the sphere for LPS; a fixed Gaussian here would feed LPS's decoder a prior it never
    saw.
    """
    import jax
    from agents.meanflow_utils import sample_latent_dist
    from utils.flax_utils import get_batch_shape

    dim = config['horizon_length'] * config['action_dim']
    rng = jax.random.PRNGKey(np.random.randint(0, 2 ** 32))

    def act(observations):
        nonlocal rng
        rng, noise_rng = jax.random.split(rng)
        shape = get_batch_shape(observations, config['leaf_ndims'])
        noise = sample_latent_dist(noise_rng, (*shape, dim), config['latent_dist'])
        actions = agent.compute_flow_actions(observations, noise, steps=steps)
        return np.asarray(actions).reshape(*shape, config['horizon_length'], config['action_dim'])

    return act


def nfe_policy_fn(agent, steps):
    """The steered policy, decoded in `steps` steps."""
    import jax

    return supply_rng(functools.partial(agent.sample_actions, steps=steps),
                      rng=jax.random.PRNGKey(np.random.randint(0, 2 ** 32)))


def value_fn(agent):
    """What the critic thinks of the chunk the policy just chose.

    Every ensemble member is kept: the latent actor climbs their mean, and the TD target subtracts
    their spread.
    """
    def value(observations, actions):
        qs = agent.network.select('critic')(observations, np.asarray(actions).reshape(-1))
        return np.asarray(qs).reshape(-1)

    return value


def write_video(path, frames, fps=20):
    """The episode's camera views, side by side, as one mp4."""
    import cv2
    stacked = [np.concatenate([f[..., i:i + 3] for i in range(0, f.shape[-1], 3)], axis=1)
               for f in frames]
    h, w = stacked[0].shape[:2]
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    for frame in stacked:
        writer.write(frame[..., ::-1])
    writer.release()
    print(f'    wrote {path} ({len(stacked)} frames)')


def rollout(robot, policy, judges, frames=None):
    """One trajectory. Ends on the adapter's own verdict when it has one, else on the operator."""
    robot.reset()
    policy.reset()
    step = 0
    while judges or not interrupted():
        if FLAGS.max_steps and step >= FLAGS.max_steps:
            return step, 'max_steps'
        observation = robot.observe()
        if frames is not None:
            frames.append(observation['image'])
        robot.gate()
        robot.act(policy.forward(observation))
        step += 1
        if judges:
            outcome = robot.outcome()
            if outcome is not None:
                return step, outcome
    return step, 'operator'


def main(_):
    path = FLAGS.checkpoint_path
    flag_config = ml_collections.ConfigDict(json.load(open(f'{path}/flags.json')))
    config = flag_config.agent
    config['horizon_length'] = flag_config.horizon_length

    example_batch = load_example_batch(path)
    print_batch_shapes(example_batch)

    agent = agents[config['agent_name']].create(
        FLAGS.seed,
        example_batch['observations'],
        example_batch['actions'],
        config,
    )
    agent = restore_agent(agent, path, FLAGS.checkpoint_step, params_only=True)

    np.random.seed(FLAGS.seed)
    kwargs = dict(image_size=FLAGS.image_size)
    if FLAGS.max_steps:
        kwargs['max_steps'] = FLAGS.max_steps
    if FLAGS.control_freq:
        kwargs['control_freq'] = FLAGS.control_freq
    robot = robots.get(FLAGS.robot)(**kwargs)
    policy = JAXWrapper(agent, normalization.load(path))
    if FLAGS.base_policy:
        policy.actor_fn = base_policy_fn(agent, config, FLAGS.nfe)
    elif FLAGS.nfe != 1:
        policy.actor_fn = nfe_policy_fn(agent, FLAGS.nfe)
    # A behaviour-cloning agent has no critic to trace.
    if FLAGS.value_trace and 'modules_critic' in agent.network.params:
        policy.value_fn = value_fn(agent)

    video_dir = FLAGS.video_dir or os.path.join(path, 'video')
    if FLAGS.video:
        os.makedirs(video_dir, exist_ok=True)

    # An adapter that reports outcomes can run unattended; one that does not needs the operator.
    judges = type(robot).outcome is not robots.Adapter.outcome

    outcomes = []
    try:
        while True:
            episode = len(outcomes)
            frames = [] if episode < FLAGS.video else None
            print(f'--- rollout {episode}' + ('' if judges else ': Enter ends it early'))
            steps, why = rollout(robot, policy, judges, frames)
            print(f'    {steps} steps, ended on {why}')
            outcomes.append(why)
            if frames:
                write_video(os.path.join(video_dir, f'ep{episode}_{why}.mp4'), frames)
            if episode < FLAGS.value_trace and policy.values:
                os.makedirs(video_dir, exist_ok=True)
                np.save(os.path.join(video_dir, f'ep{episode}_{why}_value.npy'),
                        np.stack(policy.values))
            if FLAGS.episodes:
                if len(outcomes) >= FLAGS.episodes:
                    break
            elif input('Enter for another, q to quit: ').strip().lower() == 'q':
                break
    finally:
        robot.close()

    if judges and outcomes:
        wins = outcomes.count('success')
        print(f'=== {wins}/{len(outcomes)} = {100 * wins / len(outcomes):.1f}% ===')


if __name__ == '__main__':
    app.run(main)
