#!/usr/bin/env python3
# by frank tian on 7.13.2020

import math
import sys
from envs.build_envs.utilities.quaternion import bullet_quaternion as bq

class RunstraightTask(object):
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

        self.body_pos = None
        self.body_ori = None
        self.body_lin_vel = None
        self.body_ang_vel = None
        self.joint_pos = None
        self.joint_vel = None
        self.joint_tor = None

        self.body_pos_list = []
        self.mode = mode
        self.pos_list_length = 50
        return

    def __call__(self, env):
        return self.reward(env)

    def reset(self,env):
        self._env = env
        self.body_pos_list = []
        return

    def _update_pos_list(self):
        self._get_pos_vel_info()
        self.body_pos_list.append(self.body_pos)
        if len(self.body_pos_list) > self.pos_list_length:
            self.body_pos_list.pop(0)

    def _cal_average_vel(self):
        #print(self.body_pos_list[-1][0], self.body_pos_list[0][0])
        return (self.body_pos_list[-1][0] - self.body_pos_list[0][0]) / len(self.body_pos_list) / 0.01

    # 此方向不考虑旋转，为狗头朝向的方向
    def _cal_current_face_ori(self):
        self._get_pos_vel_info()
        return bq(self.body_ori).ori([0, 0, 1])

    # 此方向考虑旋转，为背部朝向的方向
    def _cal_current_back_ori(self):
        self._get_pos_vel_info()
        return bq(self.body_ori).ori([0, 1, 0])

    def _reward_of_ori(self):
        # 理想的face_ori为[1,0,0]
        face_ori = self._cal_current_face_ori()
        # 理想的back_ori为[0,0,1]
        back_ori = self._cal_current_back_ori()
        face_reward = - (1 - face_ori[0]) ** 2
        back_reward = - (1 - back_ori[2]) ** 2 * 3
        return face_reward + back_reward

    def _reward_of_body_vel(self):
        max_vel = 3
        self._update_pos_list()
        average_vel = self._cal_average_vel()
        instantaneous_vel = self.body_lin_vel[0]
        average_reward = math.exp(1 + min([average_vel, max_vel])) - math.e
        instantaneous_reward = math.exp(1 + min([instantaneous_vel, max_vel])) - math.e
        #print(average_vel,instantaneous_vel)
        #print(average_reward, instantaneous_reward)
        return average_reward + instantaneous_reward

    def _reward_of_pos(self):
        self._get_pos_vel_info()
        reward = math.exp(1 + self.body_pos[2]) - math.e
        return reward

    def _reward_of_energy(self):
        self._get_pos_vel_info()
        E = sum([abs(p[0]*p[1]) for p in zip(self.joint_tor,self.joint_vel)])
        reward = -E
        return reward

    def reward(self, env):
        """Get the reward without side effects."""
        del env
        body_vel_r = self._reward_of_body_vel() * 0.5
        energy_r = self._reward_of_energy() * 10
        ori_r = self._reward_of_ori() * 1
        pos_r = self._reward_of_pos() * 0.2
        reward = body_vel_r + ori_r + pos_r + energy_r
        if self._bad_end():
            reward = reward - 100
        if self.mode != 'train' and self._env.env_step_counter % 25 == 0:
            print('ori_r', round(ori_r,2),'pos:',round(pos_r,2),'body_vel:',round(body_vel_r,2),'energy',round(energy_r,2))
        return reward

    def _bad_end(self):
        back_ori = self._cal_current_back_ori()
        self._update_pos_list()
        average_vel = self._cal_average_vel()
        instantaneous_vel = self.body_lin_vel[0]
        if back_ori[2] < 0.7:
            if self.mode != 'train' and sys._getframe(1).f_code.co_name == 'reward':
                print('die because wrong ori')
            return True
        if self.body_pos[2] < 0.2:
            if self.mode != 'train' and sys._getframe(1).f_code.co_name == 'reward':
                print('die because wrong pos')
            return True
        if (average_vel < 0.1 and self._env.env_step_counter > self.pos_list_length) and instantaneous_vel < 0.1:
            if self.mode != 'train' and sys._getframe(1).f_code.co_name == 'reward':
                print('die because too slow')
            return True
        else:
            return False
    def done(self, env):
        """Checks if the episode is over."""
        del env
        if self.mode == 'never_done':
            return False
        if self.mode == 'train' and self._env.env_step_counter > 300:
            return True
        return self._bad_end()

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
        self.body_pos = pyb.getBasePositionAndOrientation(quadruped)[0]#3 list: position list of 3 floats
        self.body_ori = pyb.getBasePositionAndOrientation(quadruped)[1]#4 list: orientation as list of 4 floats in [x,y,z,w] order
        self.body_lin_vel = pyb.getBaseVelocity(quadruped)[0]#3 list: linear velocity [x,y,z]
        self.body_ang_vel = pyb.getBaseVelocity(quadruped)[1]#3 list: angular velocity [wx,wy,wz]
        self.joint_pos = []#float: the position value of this joint
        self.joint_vel = []#float: the velocity value of this joint
        self.joint_tor = []  # float: the torque value of this joint

        for i in range(12):
            self.joint_pos.append(pyb.getJointState(quadruped,i)[0])
            self.joint_vel.append(pyb.getJointState(quadruped,i)[1])
            self.joint_tor.append(pyb.getJointState(quadruped, i)[3])