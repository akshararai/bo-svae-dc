#
# Initial PyBullet gym env for Sawyer robot.
#
from copy import copy
import os
import time

import numpy as np

import gym
from gym import spaces

from gym_bullet_extensions.bullet_manipulator import BulletManipulator


class SawyerEnv(gym.Env):
    OBJECT_INIT_POSES = np.array([
        [0.05,0.3,0.15], [0.05,0.3,0.45], [0.05,0.3,0.75], [0.05,0.3,1.05]])
    OBJECT_QUATS = np.array([[0,0,0,1]]*len(OBJECT_INIT_POSES))
    OBJECT_MIN = np.array([-1,-1,0]+[-1]*3+[-1]*3)  # pos, euler sin, euler cos
    OBJECT_MAX = np.array([1,1,1]+[1]*3+[1]*3)      # pos, euler sin, euler cos

    def __init__(self, num_objects=1, max_episode_steps=300,
                 torque_control=False, visualize=False, debug_level=0):
        super(SawyerEnv, self).__init__()
        # Init bullet simulator env.
        urdf_file = os.path.join(
            'sawyer_robot', 'sawyer_description', 'urdf', 'sawyer.urdf')
        self.robot = BulletManipulator(
            urdf_file, ee_joint_name='right_j6', ee_link_name='right_hand',
            visualize=visualize)
        # TODO: add object positions to states.
        self.observation_space = spaces.Box(
            np.hstack([self.robot.get_minpos(), -1.0*self.robot.get_maxvel()]),
            np.hstack([self.robot.get_maxpos(), self.robot.get_maxvel()]))
        self.action_space = spaces.Box(
            -1.0*self.robot.get_maxforce(), self.robot.get_maxforce())
        print('observation_space', self.observation_space.low,
              self.observation_space.high)
        print('action_space', self.action_space.low, self.action_space.high)
        self.debug_level = debug_level
        self.max_episode_steps = max_episode_steps
        self.visualize = visualize
        self.object_init_poses = SawyerEnv.OBJECT_INIT_POSES[0:num_objects]
        self.object_quats = SawyerEnv.OBJECT_QUATS[0:num_objects]
        self.object_ids = self.robot.load_objects_from_file(
            'cylinder.urdf', self.object_init_poses)
        # Create a list of names for obs for easier debugging.
        joint_names = self.robot.info.joint_names
        obs_names = [nm+'_pos' for nm in joint_names]
        obs_names.extend([nm+'_vel' for nm in joint_names])
        obs_names.extend(
            ['obj'+str(i) for i in range(len(self.object_ids))])
        self.obs_names = ['sawyer_'+nm for nm in obs_names]

    def reset(self):
        if self.visualize: input('Pres Enter to continue reset')
        self.stepnum = 0
        self.badlen = 0
        self.collided = False
        self.robot.reset()
        self.robot.reset_objects(
            self.object_ids, self.object_init_poses, self.object_quats)
        return self.get_obs()

    def step(self, action):  # assume unscaled action
        torque = np.clip(action, self.action_space.low, self.action_space.high)
        # Apply torque action to reacher joints.
        self.robot.apply_joint_torque(torque)
        #if self.visualize: time.sleep(0.05)
        next_state = np.hstack([self.robot.get_qpos(), self.robot.get_qvel()])
        # Update internal counters.
        self.stepnum += 1
        # Report reward starts and other info.
        reward = 0.0
        done = (self.stepnum == self.max_episode_steps)
        # TODO: write code for collision checking
        info = {'is_bad':False}
        # This env treats objects as fixed.
        self.robot.reset_objects(
            self.object_ids, self.object_init_poses, self.object_quats)
        if done:
            final_rwd = self.final_rwd()
            info['done_obs'] = copy(next_state) # vector envs lose last frame
            info['done_reward'] = final_rwd
            if self.debug_level>=1:
                print('rwd {:0.4f} badlen {:d}'.format(final_rwd, self.badlen))
        return next_state, reward, done, info

    def render(self, mode="rgb_array", close=False):
        pass  # implemented in pybullet

    def render_debug(self, width=600):
        return self.robot.render_debug(width=width)

    def get_obs(self):
        obs = np.hstack([self.robot.get_qpos(), self.robot.get_qvel()])
        eps = 1e-6
        if (obs<self.observation_space.low-eps).any():
            print('   obs', obs, '\nvs low', self.observation_space.low)
            assert(False)
        if (obs>self.observation_space.high+eps).any():
            print('    obs', obs, '\nvs high', self.observation_space.high)
            print(obs[np.where(obs>self.observation_space.high)])
            print(self.observation_space.high[np.where(obs>self.observation_space.high)])
            assert(False)
        return obs

    def override_state(self, state):
        assert(False)  # TODO: implement

    def final_rwd(self):
        rwd = 0.0 # TODO: add reward fxn
        return rwd

    def waypt_regions_and_init(self):
        # TODO: make this code customizable for new tasks.
        obj0 = SawyerEnv.OBJECT_INIT_POSES[0]
        obj1 = SawyerEnv.OBJECT_INIT_POSES[1]
        ee_poses = np.array([[0,0,0.25],
                             [obj0[0],obj0[1]*2,obj0[2]],
                             [obj1[0],obj1[1]*2,obj1[2]],
                             [0,0,0.25]])
        waypt_lows = np.zeros_like(ee_poses)
        waypt_highs = np.zeros_like(ee_poses)
        for wpt in range(ee_poses.shape[0]):
            waypt_lows[wpt] = ee_poses[wpt] - np.array([0.1,0.1,0.0])
            waypt_highs[wpt] = ee_poses[wpt] + np.array([0.1,0.1,0.1])
        return waypt_lows, waypt_highs, self.robot.get_ee_pos()