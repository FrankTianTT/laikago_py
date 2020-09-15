#!/usr/bin/env python3
# by frank tian on 7.9.2020

import time
from envs.build_envs.laikago_task import LaikagoTask
'''
Stand Up Task Version 0
reward: motor velocities penalty
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

    def update(self, env):
        del env

    def reward(self, env):
        del env
        self._add_reward(self._reward_of_energy(), 1)

        return self._get_sum_reward()

    def done(self, env):
        del env
        if self._not_done_of_too_short() or self._not_done_of_mode(self.mode):
            return False
        else:
            return self._done_of_wrong_toward_ori() or self._done_of_low_height() or self._done_of_too_long()

'''
Stand Up Task Version 1

reward: motor velocities penalty + collision detection penalty
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

    def reward(self, env):
        del env
        self._add_reward(self._reward_of_sum_vel(), 1)
        self._add_reward(self._reward_of_toe_collision(), 1)
        self._add_reward(self._reward_of_stand_height(), 1)
        self._add_reward(self._reward_of_toe_upper_distance(), 1)

        return self._get_sum_reward()

    def done(self, env):
        del env
        if self._not_done_of_too_short() or self._not_done_of_mode(self.mode):
            return False
        else:
            return self._done_of_wrong_toward_ori() or self._done_of_low_height() or self._done_of_too_long()

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

    def reward(self, env):
        del env
        self._add_reward(self._reward_of_energy(), 1)
        self._add_reward(self._reward_of_upward_ori(), 1)
        self._add_reward(self._reward_of_toe_collision(), 1)
        self._add_reward(self._reward_of_stand_height(), 1)
        self._add_reward(self._reward_of_toe_upper_distance(), 1)

        return self._get_sum_reward()

    def done(self, env):
        del env
        if self._not_done_of_too_short() or self._not_done_of_mode(self.mode):
            return False
        else:
            return self._done_of_wrong_toward_ori() or self._done_of_low_height() or self._done_of_too_long()
