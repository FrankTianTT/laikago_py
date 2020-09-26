from envs.build_envs.laikago_task import LaikagoTask

class RunstraightTask(LaikagoTask):
    def __init__(self, mode='train'):
        super(RunstraightTask, self).__init__(mode)
        return

    def update(self, env):
        del env

    def done(self, env):
        del env
        if self._not_done_of_too_short() or self._not_done_of_mode(self.mode):
            return False
        else:
            return (self._done_of_wrong_stand_ori() or self._done_of_low_height() or self._done_of_too_long() or
                   self._done_of_too_low() or self._done_of_wrong_run_ori())

class RunstraightTaskV0(RunstraightTask):
    def __init__(self, mode='train'):
        super(RunstraightTaskV0, self).__init__(mode)
        return

    def reward(self, env):
        self._add_reward(self._reward_of_run_vel(), 1)

        return self._get_sum_reward()

class RunstraightTaskV1(RunstraightTask):
    def __init__(self, mode='train'):
        super(RunstraightTaskV1, self).__init__(mode)
        return

    def reward(self, env):
        self._add_reward(self._reward_of_run_vel(), 1)
        self._add_reward(self._reward_of_run_ori(), 1)

        return self._get_sum_reward()

class RunstraightTaskV2(RunstraightTask):
    def __init__(self, mode='train'):
        super(RunstraightTaskV2, self).__init__(mode)
        return

    def reward(self, env):
        self._add_reward(self._reward_of_run_vel(), 1)
        self._add_reward(self._reward_of_run_ori(), 1)
        self._add_reward(self._reward_of_energy(), 1)


        return self._get_sum_reward()