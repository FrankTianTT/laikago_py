import time
from envs.build_envs.laikago_task import LaikagoTask
import numpy as np
'''
Stand Up Task Version 0
reward: motor velocities penalty
done: body height too low or body having deviation
'''
class StandfromlieTaskV0(LaikagoTask):
    def __init__(self, mode='train'):
        super(StandfromlieTaskV0, self).__init__(mode)
        return

    def update(self, env):
        del env

    def reward(self, env):
        del env
        self._add_reward(self._reward_of_stand_height(), 3)
        self._add_reward(self._reward_of_sum_vel(), 1)
        self._add_reward(self._reward_of_toes_height(), 1)

        return self._get_sum_reward()

    def done(self, env):
        del env
        if self._not_done_of_too_short() or self._not_done_of_mode(self.mode):
            return False
        else:
            return self._done_of_wrong_toward_ori() or self._done_of_too_long()

class StandfromlieTaskV1(LaikagoTask):
    def __init__(self, mode='train'):
        super(StandfromlieTaskV1, self).__init__(mode)
        return

    def reward(self, env):
        del env
        self._add_reward(self._reward_of_energy(), 1)
        self._add_reward(self._reward_of_toe_collision(), 1)
        self._add_reward(self._reward_of_stand_height(), 3)
        self._add_reward(self._reward_of_toe_upper_distance(), 1)

        return self._get_sum_reward()

    def done(self, env):
        del env
        if self._not_done_of_too_short() or self._not_done_of_mode(self.mode):
            return False
        else:
            return self._done_of_wrong_toward_ori() or self._done_of_too_long()

class StandfromlieTaskV2(LaikagoTask):
    def __init__(self, mode='train'):
        super(StandfromlieTaskV2, self).__init__(mode)
        return

    def reward(self, env):
        del env
        self._add_reward(self._reward_of_energy(), 1)
        self._add_reward(self._reward_of_upward_ori(), 1)
        self._add_reward(self._reward_of_toe_collision(), 1)
        self._add_reward(self._reward_of_stand_height(), 3)
        self._add_reward(self._reward_of_toe_upper_distance(), 1)

        return self._get_sum_reward()

    def done(self, env):
        del env
        if self._not_done_of_too_short() or self._not_done_of_mode(self.mode):
            return False
        else:
            return self._done_of_wrong_toward_ori() or self._done_of_too_long()