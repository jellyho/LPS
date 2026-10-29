"""DROID: a single Franka arm with a wrist and an exterior camera."""

import numpy as np

from . import Adapter, register


def to_policy_observation(obs):
    """A DROID observation -> the policy's.

    State is the 6-D end-effector pose plus the gripper width. Every ``*_left`` camera is used, in
    serial order, stacked on the channel axis.
    """
    robot_state = obs['robot_state']
    state = np.concatenate([robot_state['cartesian_position'],
                            [robot_state['gripper_position']]])

    images = obs['image']
    cam_keys = sorted([k for k in images.keys() if k.endswith('_left')])
    # The cameras hand over BGRA: drop the alpha, then reverse to RGB.
    views = [images[k][:][..., :3][..., ::-1] for k in cam_keys]

    return {'state': state, 'image': np.concatenate(views, axis=-1)}


@register('droid')
class DROID(Adapter):
    def __init__(self, image_size=224, **kwargs):
        from droid.controllers.oculus_controller import VRPolicy
        from droid.robot_env import RobotEnv

        camera = dict(image=True, concatenate_images=False,
                      resolution=(image_size, image_size), resize_func='cv2')
        self.env = RobotEnv(
            action_space='cartesian_velocity',
            gripper_action_space='position',
            camera_kwargs=dict(hand_camera=camera, varied_camera=camera),
        )
        self.controller = VRPolicy()
        self.control_hz = self.env.control_hz

    def reset(self):
        self.env._robot.establish_connection()
        self.controller.reset_state()
        self.env.reset(randomize=False)

    def observe(self):
        return to_policy_observation(self.env.get_observation())

    def act(self, action):
        """The dataset's action is DROID's own command -- 6-D cartesian velocity plus the gripper
        position -- which is what ``RobotEnv`` is configured to take.
        """
        self.env.step(action)

    def gate(self):
        while not self.controller.get_info()['movement_enabled']:
            pass

    # -- online collection: the operator also judges the trajectory -------------------------------
    def wait_to_start(self, msg=None):
        """A ends the wait with 'success', B with 'failure'."""
        while self.outcome() is not None:
            pass
        if msg is not None:
            print(msg)
        while True:
            verdict = self.outcome()
            if verdict is not None:
                return verdict

    def outcome(self):
        info = self.controller.get_info()
        if info['success']:
            return 'success'
        if info['failure']:
            return 'failure'
        return None
