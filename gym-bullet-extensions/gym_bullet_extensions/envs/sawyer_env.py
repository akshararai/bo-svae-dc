#
# PyBullet gym env for Sawyer robot.
#
# @akshararai
#

import os
import numpy as np
np.set_printoptions(precision=4, linewidth=150, threshold=np.inf, suppress=True)

from gym_bullet_extensions.bullet_manipulator import BulletManipulator
from gym_bullet_extensions.envs.manipulator_env import ManipulatorEnv


class SawyerEnv(ManipulatorEnv):

    def __init__(self, num_objects, max_episode_steps,
                 control_mode='velocity', visualize=False, debug_level=0):
        self.debug_level = debug_level
        rest_qpos = [-0., -1.18, 0., 2.18, -0., 0.57, 3.14]
        robot = BulletManipulator(
            os.path.join('sawyer_robot', 'urdf', 'sawyer_bullet.urdf'),
            control_mode=control_mode,
            ee_joint_name='right_hand', ee_link_name='right_hand',
            base_pos=[0, 0, 0.9], dt=1.0 / 240.0,
            min_z=0.85,
            visualize=visualize,
            rest_arm_qpos=rest_qpos)
        assert(num_objects<=2)
        table_minmax_x_minmax_y = np.array([0.1, 0.67, -0.45, 0.50])
        super(SawyerEnv, self).__init__(
            robot, num_objects, table_minmax_x_minmax_y,
            'cylinder_block.urdf', max_episode_steps, visualize,
            debug_level)

    def set_join_limits_for_forward_workspace(self):
        # Set reasonable joint limits for operating the space mainly in front
        # of the robot. Our main workspace is the table in front of the robot,
        # so we are not interested in exploratory motions outside of the main
        # workspace.
        minpos = np.copy(self.robot.get_minpos())
        maxpos = np.copy(self.robot.get_maxpos())
        # operate in the workspace in front of the robot
        # minpos[0] = -0.5; maxpos[0] = 0.5
        # minpos[1] = -0.5; maxpos[1] = 0.5
        # minpos[2] = -0.5; maxpos[2] = 0.5
        # minpos[3] = -3.0; maxpos[3] = -1.0  # don't stretch out the elbo
        self.robot.set_joint_limits(minpos, maxpos)

    def get_all_init_object_poses(self, num_objects):
        # 1st object starts at x=0.25 2nd ends at x=0.48
        all_init_object_poses = np.array([
            [0.3,0.35, 0.8], [0.45,0.35,0.8]])
        init_object_quats = [[0,0,0,1]]*num_objects
        return all_init_object_poses[0:num_objects], init_object_quats

    def get_init_pos(self):
        ee_pos = np.array([0.4496, 0.1603, 1.1158])
        ee_quat = np.array([0.6422, 0.7666, 0.0003, 0.0003])
        fing_dist = 0.0
        # init_qpos = self.robot.ee_pos_to_qpos(
        #     ee_pos, ee_quat, fing_dist=fing_dist)
        # assert(init_qpos is not None)
        # Use a manual qpos (to avoid instabilities of IK solutions).
        init_qpos = np.array([-0., -1.18, 0., 2.18, -0., 0.57, 3.14])
        return init_qpos, ee_pos, ee_quat, fing_dist
