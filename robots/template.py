"""Template for a new robot. Copy this file, fill it in, import it in __init__.py.

`observe` returns the state and cameras in the dataset's order -- both arms concatenated for a
bimanual robot, cameras stacked on the channel axis -- and `act` sends the dataset's action vector
however the SDK wants it. Both are checked against `norm_stats.json` on the first tick.

`gate` is optional and pauses the rollout while an operator holds the robot.
"""

import numpy as np

from . import Adapter, register


@register('template')
class Template(Adapter):
    def __init__(self, image_size=224, **kwargs):
        # Import the SDK here, not at module level, so this file can be read without it installed.
        from your_sdk import Robot

        self.robot = Robot(camera_resolution=(image_size, image_size))

    def reset(self):
        self.robot.home()

    def observe(self):
        obs = self.robot.get_observation()
        state = np.concatenate([
            obs['arm_pose'],      # e.g. (6,)  -- or both arms for a bimanual robot
            [obs['gripper']],     # e.g. (1,)
        ])
        views = [
            obs['cam_wrist'][..., :3],      # RGB, in the dataset's camera order
            obs['cam_exterior'][..., :3],
        ]
        return {'state': state, 'image': np.concatenate(views, axis=-1)}

    def act(self, action):
        self.robot.send(action)   # e.g. send(left=action[:7], right=action[7:]) if bimanual

    def close(self):
        self.robot.disconnect()
