from envs.build_envs.laikago_task import PushTask
import random

class StanduppushTaskV0(PushTask):
    def __init__(self,
                 mode='train',
                 force=True,
                 max_force=3000,
                 force_delay_steps=3):
        super(StanduppushTaskV0, self).__init__(mode=mode,
                                                force=force,
                                                max_force=max_force,
                                                force_delay_steps=force_delay_steps)
        return

    def reward(self, env):
        del env
        self._add_reward(self._reward_of_stand_height(), 1)

        return self._get_sum_reward()

    def done(self, env):
        del env
        if self._not_done_of_too_short() or self._not_done_of_mode(self.mode):
            return False
        else:
            return self._done_of_wrong_toward_ori() or self._done_of_low_height(0.2) or self._done_of_too_long()