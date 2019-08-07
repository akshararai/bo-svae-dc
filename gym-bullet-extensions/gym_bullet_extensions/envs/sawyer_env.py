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
        rest_qpos = [0.0, 0.0, 0.0, 1.57079633, 0.0 ,1.03672558, 0.0]
        robot = BulletManipulator(
            os.path.join('sawyer_robot', 'urdf', 'sawyer_bullet.urdf'),
            control_mode=control_mode,
            ee_joint_name='right_hand', ee_link_name='right_hand',
            base_pos=[0, 0, 0.0], dt=1.0 / 240.0,
            kp=([200.0] * 7 + [1.0] * 2),
            kd=([2.0] * 7 + [0.1] * 2),
            min_z=0.00,
            visualize=visualize,
            rest_arm_qpos=rest_qpos)
        assert(num_objects<=2)
        table_minmax_x_minmax_y = np.array([0.15, 0.77, -0.475, 0.475])
        super(SawyerEnv, self).__init__(
            robot, num_objects, table_minmax_x_minmax_y,
            'cylinder_block.urdf', max_episode_steps, visualize,
            debug_level)

    def get_all_init_object_poses(self, num_objects):
        # 1st object starts at x=0.25 2nd ends at x=0.48
        all_init_object_poses = np.array([
            [0.31,-0.30,0.11], [0.43,-0.30,0.11]])
        init_object_quats = [[0,0,0,1]]*num_objects
        return all_init_object_poses[0:num_objects], init_object_quats

    def get_init_pos(self):
        ee_pos = np.array([0.3, -0.5, 0.25])
        ee_quat = np.array(self.robot.sim.getQuaternionFromEuler(
            [np.pi,-np.pi/2,0]))
        fing_dist = 0.0
        # init_qpos = self.robot.ee_pos_to_qpos(
        #     ee_pos, ee_quat, fing_dist=fing_dist)
        # assert(init_qpos is not None)
        # Use a manual qpos (to avoid instabilities of IK solutions).
        init_qpos = np.array([0.0, 0.0, 0.0, 1.57079633, 0.0 ,1.03672558, 0.0])
        return init_qpos, ee_pos, ee_quat, fing_dist
