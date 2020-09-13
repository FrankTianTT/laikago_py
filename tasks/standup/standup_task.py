#!/usr/bin/env python3
# by frank tian on 7.9.2020

import math
from envs.build_envs.laikago_task import LaikagoTask
'''
Stand Up Task Version 0
reward: motor velocities penalty add up alive bonus
done: body height too low or body having deviation
'''
class StandupTaskV0(LaikagoTask):
    def __init__(self,
                 weight=1.0,
                 pose_weight=0.5,
                 velocity_weight=0.05,
                 end_effector_weight=0.2,
                 root_pose_weight=0.15,
                 root_velocity_weight=0.1,
                 mode='train',
                 max_episode_steps=300):
        super(StandupTaskV0, self).__init__(weight,
                                          pose_weight,
                                          velocity_weight,
                                          end_effector_weight,
                                          root_pose_weight,
                                          root_velocity_weight,
                                          mode,
                                          max_episode_steps)
        return

    def reward(self, env):
        del env
        reward = - sum([abs(v) for v in self.joint_vel]) * 0.1 + 5
        return reward

    def done(self, env):
        """Checks if the episode is over."""
        del env
        if self.mode == 'never_done': #only use in test mode
            return False
        if self.mode == 'train' and self._env.env_step_counter > self.max_episode_steps:
            return True
        return self._bad_end()

    def _bad_end(self):
        back_ori = self._cal_current_back_ori()
        if back_ori[2] < 0.6:
            return True
        if self.body_pos[2] < 0.2:
            return True
        else:
            return False

'''
Stand Up Task Version 1

reward: motor velocities penalty + alive bonus + collision detection penalty
done: body height too low or body having deviation or no collision detection time too long
'''
class StandupTaskV1(StandupTaskV0):
    def __init__(self,
                 weight=1.0,
                 pose_weight=0.5,
                 velocity_weight=0.05,
                 end_effector_weight=0.2,
                 root_pose_weight=0.15,
                 root_velocity_weight=0.1,
                 mode='train',
                 max_episode_steps=300):
        super(StandupTaskV0, self).__init__(weight,
                                          pose_weight,
                                          velocity_weight,
                                          end_effector_weight,
                                          root_pose_weight,
                                          root_velocity_weight,
                                          mode,
                                          max_episode_steps)
        return

    '''
    There are 16 joints in the laikago.
    No.0, 4, 8, 12 are joints between hip-motor and chassis
    No.1, 5, 9, 13 are joints between upper-leg and hip-motor
    No.2, 6, 10, 14 are joints between lower_leg and upper-leg
    No,3, 7, 11, 15 are joints of toes
    '''
    def _reward_of_collision(self):
        reward = 0
        collision_info = self._get_collision_info()
        num = [0,0,0,0]
        for i in range(16):
            if collision_info[i]:
                num[i%4] += 1
            if i % 4 == 3:
                if collision_info[i]:
                    reward += 1
            elif i % 4 == 1:
                if collision_info[i]:
                    reward -= 2
        return reward

    def reward(self, env):
        del env
        collision_r = self._reward_of_collision()
        alive_r = 10
        reward = - sum([abs(v) for v in self.joint_vel]) * 0.1 + collision_r * 0.3 + alive_r
        return reward

class StandupTaskV2(StandupTaskV1):
    def __init__(self,
                 weight=1.0,
                 pose_weight=0.5,
                 velocity_weight=0.05,
                 end_effector_weight=0.2,
                 root_pose_weight=0.15,
                 root_velocity_weight=0.1,
                 mode='train',
                 max_episode_steps=300):
        super(StandupTaskV0, self).__init__(weight,
                                          pose_weight,
                                          velocity_weight,
                                          end_effector_weight,
                                          root_pose_weight,
                                          root_velocity_weight,
                                          mode,
                                          max_episode_steps)
        return

    def _reward_of_ori(self):
        # 理想的face_ori为[1,0,0]
        face_ori = self._cal_current_face_ori()
        # 理想的back_ori为[0,0,1]
        back_ori = self._cal_current_back_ori()
        back_reward = - (1 - back_ori[2]) ** 2
        return back_reward

    def _reward_of_pos(self):
        self._get_pos_vel_info()
        reward = self.body_pos[2]
        return reward

    def _reward_of_energy(self):
        self._get_pos_vel_info()
        E = sum([abs(p[0] * p[1]) for p in zip(self.joint_tor, self.joint_vel)])
        reward = -E
        return reward

    def reward(self, env):
        del env
        collision_r = self._reward_of_collision() * 0.3
        ori_r = self._reward_of_ori() * 20
        pos_r = self._reward_of_pos() * 5
        energy_r = self._reward_of_energy() * 10
        alive_r = 10
        reward = collision_r + ori_r + pos_r + energy_r + alive_r
        return reward