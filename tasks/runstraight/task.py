#!/usr/bin/env python3
# by frank tian on 7.13.2020


from envs.build_envs.laikago_task import LaikagoTask

class RunstraightTaskV0(LaikagoTask):
    def __init__(self, mode='train'):
        super(RunstraightTaskV0, self).__init__(mode)
        return