#!/usr/bin/env python3
# by frank tian on 7.9.2020

import math
from envs.build_envs.utilities.quaternion import bullet_quaternion as bq
import random
class LaikagoTask(object):
    def __init__(self,
                 weight=1.0,
                 pose_weight=0.5,
                 velocity_weight=0.05,
                 end_effector_weight=0.2,
                 root_pose_weight=0.15,
                 root_velocity_weight=0.1,
                 mode='train'):
        self._env = None
        self._weight = weight

        # reward function parameters
        self._pose_weight = pose_weight
        self._velocity_weight = velocity_weight
        self._end_effector_weight = end_effector_weight
        self._root_pose_weight = root_pose_weight
        self._root_velocity_weight = root_velocity_weight
        self.mode = mode

        self.body_pos = None
        self.body_ori = None
        self.body_lin_vel = None
        self.body_ang_vel = None
        self.joint_pos = None
        self.joint_vel = None
        self.joint_tor = None
        return

    def __call__(self, env):
        return self.reward(env)

    def reset(self,env):
        self._env = env
        self._get_pos_vel_info()
        self.quadruped = self._env.robot.quadruped
        return

    # 此方向不考虑旋转，为狗头朝向的方向
    def _cal_current_face_ori(self):
        self._get_pos_vel_info()
        return bq(self.body_ori).ori([0, 0, 1])

    # 此方向考虑旋转，为背部朝向的方向
    def _cal_current_back_ori(self):
        self._get_pos_vel_info()
        return bq(self.body_ori).ori([0, 1, 0])

    def reward(self, env):
        del env
        return

    def done(self, env):
        del env
        return

    def update(self, env):
        pass


    def _bad_end(self):
        return False

    def _get_pybullet_client(self):
        """Get bullet client from the environment"""
        return self._env._pybullet_client

    def _get_num_joints(self):
        """Get the number of joints in the character's body."""
        pyb = self._get_pybullet_client()
        return pyb.getNumJoints(self._env.robot.quadruped)

    def _get_pos_vel_info(self):
        pyb = self._get_pybullet_client()
        quadruped = self._env.robot.quadruped
        self.body_pos = pyb.getBasePositionAndOrientation(quadruped)[0]  # 3 list: position list of 3 floats
        self.body_ori = pyb.getBasePositionAndOrientation(quadruped)[1]  # 4 list: orientation as list of 4 floats in [x,y,z,w] order
        self.body_lin_vel = pyb.getBaseVelocity(quadruped)[0]  # 3 list: linear velocity [x,y,z]
        self.body_ang_vel = pyb.getBaseVelocity(quadruped)[1]  # 3 list: angular velocity [wx,wy,wz]
        self.joint_pos = []  # float: the position value of this joint
        self.joint_vel = []  # float: the velocity value of this joint
        self.joint_tor = []  # float: the torque value of this joint
        for i in range(12):
            self.joint_pos.append(pyb.getJointState(quadruped, i)[0])
            self.joint_vel.append(pyb.getJointState(quadruped, i)[1])
            self.joint_tor.append(pyb.getJointState(quadruped, i)[3])