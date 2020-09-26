from envs.build_envs.laikago_task import PushTask
import random

class StanduppushTask(PushTask):
    def __init__(self,
                 mode='train',
                 force=True,
                 max_force=1000,
                 force_delay_steps=3):
        super(StanduppushTask, self).__init__(mode=mode,
                                              force=force,
                                              max_force=max_force,
                                              force_delay_steps=force_delay_steps)
        return

    def done(self, env):
        del env
        if self._not_done_of_too_short() or self._not_done_of_mode(self.mode):
            return False
        else:
            return self._done_of_wrong_stand_ori() or self._done_of_low_height() or self._done_of_too_long()

# from StandupTaskv2
class StanduppushTaskV0(StanduppushTask):
    def __init__(self,
                 mode='train',
                 force=True,
                 max_force=1000,
                 force_delay_steps=3):
        super(StanduppushTaskV0, self).__init__(mode=mode,
                                              force=force,
                                              max_force=max_force,
                                              force_delay_steps=force_delay_steps)
        return

    def reward(self, env):
        del env
        self._add_reward(self._reward_of_base_move(), 1)
        self._add_reward(self._reward_of_toes_height(), 1)
        self._add_reward(self._reward_of_toe_collision(), 1)
        self._add_reward(self._reward_of_stand_height(), 1)

        return self._get_sum_reward()