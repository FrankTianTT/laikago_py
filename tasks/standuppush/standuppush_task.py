#!/usr/bin/env python3
# by frank tian on 7.9.2020

import math
from envs.build_envs.utilities.quaternion import bullet_quaternion as bq
from envs.build_envs.laikago_task import LaikagoTask
import random

class StanduppushTask(LaikagoTask):
    def __init__(self,
                 weight=1.0,
                 pose_weight=0.5,
                 velocity_weight=0.05,
                 end_effector_weight=0.2,
                 root_pose_weight=0.15,
                 root_velocity_weight=0.1,
                 mode='train',
                 force=True):
        super(StanduppushTask, self).__init__(weight,
                                          pose_weight,
                                          velocity_weight,
                                          end_effector_weight,
                                          root_pose_weight,
                                          root_velocity_weight,
                                          mode)

        self.force = force
        self.force_id = 0
        self._get_force_ori()
        self.max_force = 3000
        self.force_delay_steps = 3
        return

    def _get_force_ori(self):
        self.force_ori = []
        f_ori = [[1, 0, 0], [0, 1, 0], [-1, 0, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]]
        for i in f_ori:
            for j in f_ori:
                ori = [o[0]+o[1] for o in zip(i,j)]
                self.force_ori.append(ori)

    def _reward_of_ori(self):
        # 理想的face_ori为[1,0,0]
        face_ori = self._cal_current_face_ori()
        # 理想的back_ori为[0,0,1]
        back_ori = self._cal_current_back_ori()
        back_reward = - (1 - back_ori[2]) ** 2 * 3
        return back_reward

    def _reward_of_pos(self):
        self._get_pos_vel_info()
        reward = math.exp(1 + self.body_pos[2]) - math.e
        return reward

    def _reward_of_energy(self):
        self._get_pos_vel_info()
        E = sum([abs(p[0]*p[1]) for p in zip(self.joint_tor,self.joint_vel)])
        reward = -E
        return reward
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

    def _bad_end(self):
        back_ori = self._cal_current_back_ori()
        if back_ori[2] < 0.5:
            return True
        if self.body_pos[2] < 0.2:
            return True
        else:
            return False

    def done(self, env):
        """Checks if the episode is over."""
        del env
        if self.mode == 'never_done': #only use in test mode
            return False
        if self.mode == 'train' and self._env.env_step_counter > 300:
            return True
        return self._bad_end()

    def _give_force(self):
        if self._env.env_step_counter % self.force_delay_steps == 0:
            self.force_id = random.randint(0,len(self.force_ori)-1)
        ori = self.force_ori[self.force_id]
        return [f*random.random() * self.max_force for f in ori]

    def update(self, env):
        if not self.force:
            return
        force = self._give_force()
        self.body_pos = env._pybullet_client.getBasePositionAndOrientation(self.quadruped)[0]
        env._pybullet_client.applyExternalForce(objectUniqueId=self.quadruped, linkIndex=-1,
                             forceObj=force, posObj=self.body_pos, flags=env._pybullet_client.WORLD_FRAME)