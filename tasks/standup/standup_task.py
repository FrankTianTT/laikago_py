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
                 mode='train'):
        super(StandupTaskV0, self).__init__(weight,
                                          pose_weight,
                                          velocity_weight,
                                          end_effector_weight,
                                          root_pose_weight,
                                          root_velocity_weight,
                                          mode)
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
        if self.mode == 'train' and self._env.env_step_counter > 300:
            return True
        return self._bad_end()

    def _bad_end(self):
        back_ori = self._cal_current_back_ori()
        if back_ori[2] < 0.7:
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
class StandupTaskV1(LaikagoTask):
    def __init__(self,
                 weight=1.0,
                 pose_weight=0.5,
                 velocity_weight=0.05,
                 end_effector_weight=0.2,
                 root_pose_weight=0.15,
                 root_velocity_weight=0.1,
                 mode='train'):
        super(StandupTaskV1, self).__init__(weight,
                                          pose_weight,
                                          velocity_weight,
                                          end_effector_weight,
                                          root_pose_weight,
                                          root_velocity_weight,
                                          mode)
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
        for i in range(16):
            if i % 4 == 3:
                if collision_info[i]:
                    reward += 1
            else:
                if not collision_info[i]:
                    reward += 0.3
        return reward

    def reward(self, env):
        del env

        collision_r = self._reward_of_collision()
        reward = - sum([abs(v) for v in self.joint_vel]) * 0.1 + collision_r + 3
        return reward

    def done(self, env):
        """Checks if the episode is over."""
        del env
        if self.mode == 'never_done': #only use in test mode
            return False
        if self.mode == 'train' and self._env.env_step_counter > 300:
            return True
        return self._bad_end()

    def _bad_end(self):
        back_ori = self._cal_current_back_ori()
        if back_ori[2] < 0.7:
            return True
        if self.body_pos[2] < 0.2:
            return True
        else:
            return False

class StandupTaskV2(LaikagoTask):
    def __init__(self,
                 weight=1.0,
                 pose_weight=0.5,
                 velocity_weight=0.05,
                 end_effector_weight=0.2,
                 root_pose_weight=0.15,
                 root_velocity_weight=0.1,
                 mode='train'):
        super(StandupTaskV2, self).__init__(weight,
                                            pose_weight,
                                            velocity_weight,
                                            end_effector_weight,
                                            root_pose_weight,
                                            root_velocity_weight,
                                            mode)
        return

    def _reward_of_ori(self):
        # 理想的face_ori为[1,0,0]
        face_ori = self._cal_current_face_ori()
        # 理想的back_ori为[0,0,1]
        back_ori = self._cal_current_back_ori()
        face_reward = - (1 - face_ori[0]) ** 2
        back_reward = - (1 - back_ori[2]) ** 2 * 3
        return face_reward + back_reward

    def _reward_of_pos(self):
        self._get_pos_vel_info()
        reward = math.exp(1 + self.body_pos[2]) - math.e
        return reward

    def _reward_of_energy(self):
        self._get_pos_vel_info()
        E = sum([abs(p[0] * p[1]) for p in zip(self.joint_tor, self.joint_vel)])
        reward = -E
        return reward

    def _reward_of_collision(self):
        pyb = self._get_pybullet_client()
        ContactPoints = pyb.getContactPoints()
        print(ContactPoints)
        return 0

    def reward(self, env):
        del env
        ori_r = self._reward_of_ori()
        pos_r = self._reward_of_pos() * 5
        energy_r = self._reward_of_energy()
        reward = ori_r + pos_r + energy_r
        if self._bad_end():
            reward = reward - 100
        if self.mode == 'test' and self._env.env_step_counter % 50 == 0:
            print('ori_r', round(ori_r), 'pos:', round(pos_r), 'energy:', round(energy_r))
        return reward

    def done(self, env):
        """Checks if the episode is over."""
        del env
        if self.mode == 'never_done':  # only use in test mode
            return False
        if self.mode == 'train' and self._env.env_step_counter > 300:
            return True
        return self._bad_end()

    def _bad_end(self):
        back_ori = self._cal_current_back_ori()
        if back_ori[2] < 0.7:
            return True
        if self.body_pos[2] < 0.2:
            return True
        else:
            return False