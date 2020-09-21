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
        self._add_reward(self._reward_of_toe_collision(), 1)
        self._add_reward(self._reward_of_stand_height(), 1)

        return self._get_sum_reward()

    def done(self, env):
        del env
        if self._not_done_of_too_short() or self._not_done_of_mode(self.mode):
            return False
        else:
            return self._done_of_wrong_toward_ori() or self._done_of_low_height() or self._done_of_too_long()

class StandupTaskV2_1(LaikagoTask):
    def __init__(self, mode='train'):
        super(StandupTaskV2_1, self).__init__(mode)
        return

    def reward(self, env):
        del env
        self._add_reward(self._reward_of_base_move(), 1)
        self._add_reward(self._reward_of_toes_height(), 1)
        self._add_reward(self._reward_of_toe_collision(), 3)
        self._add_reward(self._reward_of_stand_height(), 1)

        return self._get_sum_reward()

    def done(self, env):
        del env
        if self._not_done_of_too_short() or self._not_done_of_mode(self.mode):
            return False
        else:
            return self._done_of_wrong_toward_ori() or self._done_of_low_height() or self._done_of_too_long()

class StandupTaskV2_2(LaikagoTask):
    def __init__(self, mode='train'):
        super(StandupTaskV2_2, self).__init__(mode)
        return

    def reward(self, env):
        del env
        self._add_reward(self._reward_of_base_move(), 1)
        self._add_reward(self._reward_of_toes_height(), 2)
        self._add_reward(self._reward_of_toe_collision(), 2)
        self._add_reward(self._reward_of_stand_height(), 1)

        return self._get_sum_reward()

    def done(self, env):
        del env
        if self._not_done_of_too_short() or self._not_done_of_mode(self.mode):
            return False
        else:
            return self._done_of_wrong_toward_ori() or self._done_of_low_height() or self._done_of_too_long()

class StandupTaskV2_3(LaikagoTask):
    def __init__(self, mode='train'):
        super(StandupTaskV2_3, self).__init__(mode)
        return

    def reward(self, env):
        del env
        self._add_reward(self._reward_of_base_move(), 3)
        self._add_reward(self._reward_of_toes_height(), 1)
        self._add_reward(self._reward_of_toe_collision(), 1)
        self._add_reward(self._reward_of_stand_height(), 1)

        return self._get_sum_reward()

    def done(self, env):
        del env
        if self._not_done_of_too_short() or self._not_done_of_mode(self.mode):
            return False
        else:
            return self._done_of_wrong_toward_ori() or self._done_of_low_height() or self._done_of_too_long()

class StandupTaskV2_4(LaikagoTask):
    def __init__(self, mode='train'):
        super(StandupTaskV2_4, self).__init__(mode)
        return

    def reward(self, env):
        del env
        self._add_reward(self._reward_of_base_move(), 1)
        self._add_reward(self._reward_of_toes_height(), 1)
        self._add_reward(self._reward_of_toe_collision(), 5)
        self._add_reward(self._reward_of_stand_height(), 1)

        return self._get_sum_reward()

    def done(self, env):
        del env
        if self._not_done_of_too_short() or self._not_done_of_mode(self.mode):
            return False
        else:
            return self._done_of_wrong_toward_ori() or self._done_of_low_height() or self._done_of_too_long()

class StandupTaskV2_5(LaikagoTask):
    def __init__(self, mode='train'):
        super(StandupTaskV2_5, self).__init__(mode)
        return

    def reward(self, env):
        del env
        self._add_reward(self._reward_of_base_move(), 1)
        self._add_reward(self._reward_of_toes_height(), 1)
        self._add_reward(self._reward_of_toe_collision(), 10)
        self._add_reward(self._reward_of_stand_height(), 1)

        return self._get_sum_reward()

    def done(self, env):
        del env
        if self._not_done_of_too_short() or self._not_done_of_mode(self.mode):
            return False
        else:
            return self._done_of_wrong_toward_ori() or self._done_of_low_height() or self._done_of_too_long()