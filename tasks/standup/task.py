import time
from envs.build_envs.laikago_task import LaikagoTask

class StandupTaskV0(LaikagoTask):
    def __init__(self, mode='train'):
        super(StandupTaskV0, self).__init__(mode)
        return

    def update(self, env):
        del env

    def reward(self, env):
        del env
        self._add_reward(self._reward_of_stand_height(), 1)
        self._add_reward(self._reward_of_toe_collision(), 1)

        return self._get_sum_reward()

    def done(self, env):
        del env
        if self._not_done_of_too_short() or self._not_done_of_mode(self.mode):
            return False
        else:
            return self._done_of_wrong_toward_ori() or self._done_of_low_height() or self._done_of_too_long()

class StandupTaskV1(LaikagoTask):
    def __init__(self, mode='train'):
        super(StandupTaskV1, self).__init__(mode)
        return

    def reward(self, env):
        del env
        self._add_reward(self._reward_of_base_move(), 1)
        self._add_reward(self._reward_of_toe_collision(), 1)
        self._add_reward(self._reward_of_stand_height(), 1)

        return self._get_sum_reward()

    def done(self, env):
        del env
        if self._not_done_of_too_short() or self._not_done_of_mode(self.mode):
            return False
        else:
            return self._done_of_wrong_toward_ori() or self._done_of_low_height() or self._done_of_too_long()

class StandupTaskV2(LaikagoTask):
    def __init__(self, mode='train'):
        super(StandupTaskV2, self).__init__(mode)
        return

    def reward(self, env):
        del env
        self._add_reward(self._reward_of_base_move(), 1)
        self._add_reward(self._reward_of_toes_height(), 1)
        self._add_reward(self._reward_of_stand_height(), 1)

        return self._get_sum_reward()

    def done(self, env):
        del env
        if self._not_done_of_too_short() or self._not_done_of_mode(self.mode):
            return False
        else:
            return self._done_of_wrong_toward_ori() or self._done_of_low_height() or self._done_of_too_long()
