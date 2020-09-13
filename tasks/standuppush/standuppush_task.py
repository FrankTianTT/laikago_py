#!/usr/bin/env python3
# by frank tian on 7.9.2020

import math
from envs.build_envs.utilities.quaternion import bullet_quaternion as bq
from envs.build_envs.laikago_task import LaikagoTask
import random

class StanduppushTaskV0(LaikagoTask):
    def __init__(self,
                 weight=1.0,
                 pose_weight=0.5,
                 velocity_weight=0.05,
                 end_effector_weight=0.2,
                 root_pose_weight=0.15,
                 root_velocity_weight=0.1,
                 mode='train',
                 force=True):
        super(StanduppushTaskV0, self).__init__(weight,
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

    def _reward_of_collision(self):
        reward = 0
        collision_info = self._get_collision_info()
        num = [0, 0, 0, 0]
        for i in range(16):
            if collision_info[i]:
                num[i % 4] += 1
            if i % 4 == 3:
                if collision_info[i]:
                    reward += 1
            elif i % 4 == 1:
                if collision_info[i]:
                    reward -= 2
        return reward

    def reward(self, env):
        del env
        collision_r = self._reward_of_collision()
        alive_r = 10
        reward = - sum([abs(v) for v in self.joint_vel]) * 0.1 + collision_r + alive_r

        return reward

    def done(self, env):
        """Checks if the episode is over."""
        del env
        if self.mode == 'never_done': #only use in test mode
            return False
        if self.mode == 'train' and self._env.env_step_counter > 300:
            return True
        return self._bad_end()

    def _bad_end(self):
        back_ori = self._get_current_back_ori()
        if back_ori[2] < 0.7:
            return True
        if self.body_pos[2] < 0.2:
            return True
        else:
            return False

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

class StanduppushTaskV1(StanduppushTaskV0):
    def __init__(self,
                 weight=1.0,
                 pose_weight=0.5,
                 velocity_weight=0.05,
                 end_effector_weight=0.2,
                 root_pose_weight=0.15,
                 root_velocity_weight=0.1,
                 mode='train',
                 force=True):
        super(StanduppushTaskV1, self).__init__(weight,
                                          pose_weight,
                                          velocity_weight,
                                          end_effector_weight,
                                          root_pose_weight,
                                          root_velocity_weight,
                                          mode,
                                          force)

    def _reward_of_upward_ori(self):
        # 理想的face_ori为[1,0,0]
        face_ori = self._get_current_face_ori()
        # 理想的back_ori为[0,0,1]
        back_ori = self._get_current_back_ori()
        back_reward = - (1 - back_ori[2]) ** 2
        return back_reward

    def _reward_of_stand_height(self):
        self._get_body_pos_vel_info()
        reward = self.body_pos[2]
        return reward

    def _reward_of_energy(self):
        self._get_body_pos_vel_info()
        E = sum([abs(p[0] * p[1]) for p in zip(self.joint_tor, self.joint_vel)])
        reward = -E
        return reward

    def reward(self, env):
        del env
        collision_r = self._reward_of_collision()
        ori_r = self._reward_of_upward_ori() * 20
        pos_r = self._reward_of_stand_height() * 5
        energy_r = self._reward_of_energy() * 10
        alive_r = 10
        reward = collision_r + ori_r + pos_r + energy_r + alive_r
        return reward

    def _bad_end(self):
        back_ori = self._get_current_back_ori()
        if back_ori[2] < 0.7:
            return True
        if self.body_pos[2] < 0.2:
            return True
        else:
            return False