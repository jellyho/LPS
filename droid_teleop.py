from droid.controllers.oculus_controller import VRPolicy
from droid.robot_env import RobotEnv
from droid.user_interface.data_collector import DataCollecter
from droid.user_interface.gui import RobotGUI
import argparse

parser = argparse.ArgumentParser(description='Collect DROID demonstrations with the Oculus controller.')

parser.add_argument('--left_controller', action='store_true', help='Use left oculus controller')
parser.add_argument('--right_controller', action='store_true', help='Use right oculus controller')


args = parser.parse_args()

# Make the robot env
imsize = 224

camera_kwargs = dict(
    hand_camera=dict(image=True, concatenate_images=False, resolution=(imsize, imsize), resize_func="cv2"),
    varied_camera=dict(image=True, concatenate_images=False, resolution=(imsize, imsize), resize_func="cv2"),
)

env = RobotEnv(camera_kwargs=camera_kwargs)

if args.left_controller:
    controller = VRPolicy(right_controller=False)
    data_collector = DataCollecter(env=env, controller=controller)
    user_interface = RobotGUI(robot=data_collector, right_controller=False)
else:
    controller = VRPolicy(right_controller=True)
    data_collector = DataCollecter(env=env, controller=controller)
    user_interface = RobotGUI(robot=data_collector, right_controller=True)
