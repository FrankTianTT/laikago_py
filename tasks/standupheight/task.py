#!/usr/bin/env python3
# by frank tian on 7.9.2020

import math
from envs.build_envs.utilities.quaternion import bullet_quaternion as bq
from envs.build_envs.laikago_task import LaikagoTask
import random

class StandupheightTaskV0(LaikagoTask):
    def __init__(self, mode='train'):
        super(StandupheightTaskV0, self).__init__(mode)
        self.fit_timer = 0
        return

    def _reward_of_collision(self):
        reward = 0
        collision_info = self._get_collision_info()
        num = [0, 0, 0, 0]
        for i in range(16):
            if collision_info[i]:
                num[i % 4] += 1
            if i % 4 == 3:
                if collision_info[i]:
                    reward += 1
            elif i % 4 == 1:
                if collision_info[i]:
                    reward -= 2
        return reward

    def _reward_of_expect_height(self):
        height_sensor = self._env.all_sensors()[-1]
        expect_height = height_sensor.get_observation()[0]
        height_reward = - abs(expect_height - self.body_pos[2])
        fit_reward = 0
        if abs(expect_height - self.body_pos[2]) < 0.03:
            self.fit_timer += 1
            if self.fit_timer > 10:
                fit_reward = 1
                self.fit_timer = 0
        else:
            self.fit_timer = 0

        return height_reward, fit_reward
    def reward(self, env):
        del env
        height_r, fit_r = self._reward_of_expect_height()
        height_r = height_r * 30
        fit_r = fit_r * 10
        joins_vel_r = - sum([abs(v) for v in self.joint_vel]) * 0.05
        collision_r = self._reward_of_collision()
        alive_r = 10
        reward = joins_vel_r + collision_r + alive_r + height_r + fit_r
        return reward

    def done(self, env):
        """Checks if the episode is over."""
        del env
        if self.mode == 'never_done': #only use in test mode
            return False
        if self.mode == 'train' and self._env.env_step_counter > 500:
            return True
        return self._bad_end()

    def _bad_end(self):
        back_ori = self._get_toward_ori()
        if back_ori[2] < 0.7:
            return True
        if self.body_pos[2] < 0.2:
            return True
        else:
            return False

class StandupheightTaskV1(StandupheightTaskV0):
    def __init__(self, mode='train'):
        super(StandupheightTaskV1, self).__init__(mode)

    def _reward_of_upward_ori(self):
        # 理想的face_ori为[1,0,0]
        face_ori = self._get_current_face_ori()
        # 理想的back_ori为[0,0,1]
        back_ori = self._get_toward_ori()
        back_reward = - (1 - back_ori[2]) ** 2
        return back_reward

    def _reward_of_stand_height(self):
        self._get_body_pos_vel_info()
        reward = self.body_pos[2]
        return reward

    def _reward_of_energy(self):
        self._get_body_pos_vel_info()
        E = sum([abs(p[0] * p[1]) for p in zip(self.joint_tor, self.joint_vel)])
        reward = -E
        return reward

    def reward(self, env):
        del env
        collision_r = self._reward_of_collision()
        ori_r = self._reward_of_upward_ori() * 20
        pos_r = self._reward_of_stand_height() * 5
        energy_r = self._reward_of_energy() * 10
        alive_r = 10
        reward = collision_r + ori_r + pos_r + energy_r + alive_r
        return reward

    def _bad_end(self):
        back_ori = self._get_toward_ori()
        if back_ori[2] < 0.7:
            return True
        if self.body_pos[2] < 0.2:
            return True
        else:
            return False