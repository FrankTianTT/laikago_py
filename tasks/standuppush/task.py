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
