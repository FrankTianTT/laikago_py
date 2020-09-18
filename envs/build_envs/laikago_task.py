import math
from envs.build_envs.utilities.quaternion import bullet_quaternion as bq
import random
import numpy as np
from envs.robots.robot_config import MotorControlMode


class LaikagoTask(object):
    def __init__(self, mode='train'):
        self._env = None
        self.mode = mode

        self.body_pos = None
        self.body_ori = None
        self.body_lin_vel = None
        self.body_ang_vel = None
        self.joint_pos = None
        self.joint_vel = None
        self.joint_tor = None

        self.sum_reward = 0
        self.sum_p = 0

        self.max_r = 0
        return

    def __call__(self, env):
        return self.reward(env)

    def reset(self,env):
        self._env = env
        self.quadruped = self._env.robot.quadruped
        self._get_body_pos_vel_info()
        return

    def _add_reward(self, reward, p=1):
        self.sum_reward += reward
        self.sum_p += p

    def _get_sum_reward(self):
        reward = self.sum_reward / self.sum_p
        self.sum_reward = 0
        self.sum_p = 0
        return reward

    def normalize_reward(self, reward, min_reward, max_reward):
        return (reward - min_reward)/(max_reward - min_reward)

    def _reward_of_toe_collision(self):
        collision_info = self._get_collision_info()
        reward = 0
        for i in range(16):
            if i % 4 == 3 and collision_info[i]:
                reward += 1
        return self.normalize_reward(reward, 0, 4)

    def _reward_of_leg_collision(self):
        collision_info = self._get_collision_info()
        reward = 0
        for i in range(16):
            if i % 4 == 1 and collision_info[i]:
                reward -= 1
        return self.normalize_reward(reward, -4, 0)

    def _reward_of_upward_ori(self):
        # the expect value of toward_ori is [0,0,1].
        toward_ori = self._get_toward_ori()
        reward = - (1 - toward_ori[2]) ** 2
        return self.normalize_reward(reward, -4, 0)

    def _reward_of_stand_height(self, max_height=0.4):
        self._get_body_pos_vel_info()
        reward = self.body_pos[2]
        return self.normalize_reward(reward, 0, 0.5)  # the initial height of laikago is 0.5.

    def _reward_of_energy(self):
        if self._env._motor_control_mode == MotorControlMode.TORQUE:
            motor_torques = self._env._last_action
        else:
            motor_torques = self._env.robot.GetTrueMotorTorques()
        motor_velocities = self._env.robot.GetTrueMotorVelocities()
        reward = - float(np.abs(motor_torques * motor_velocities).mean())
        return self.normalize_reward(reward, -1000, 0)

    def _reward_of_sum_vel(self):
        self._get_joint_pos_vel_info()
        reward = - sum([abs(v) for v in self.joint_vel])
        return self.normalize_reward(reward, -300, 0)  # the min reward by sample action is -275.

    # use it when try to stand up.
    def _reward_of_toes_height(self):
        toes_pos = self._get_pos_of_toes()
        reward = - sum([pos[2] for pos in toes_pos])
        return self.normalize_reward(reward, -1, 0)  # the min reward by sample action is -0.8.

    def _reward_of_toe_upper_distance(self):
        distances = self._get_toe_upper_distance()
        reward = np.sum(distances)
        return self.normalize_reward(reward, 0, 2)

    def _reward_of_set_height(self, height=0.3):
        assert height > 0 and height < 0.5
        self._get_body_pos_vel_info()
        reward = - abs(self.body_pos[2] - height)
        return self.normalize_reward(reward, - max([0.5 - height, height - 0]), 0)

    def _not_done_of_too_short(self, threshold=20):
        return self._env.env_step_counter < threshold  # if in this case, return True to prevent to die.

    def _not_done_of_mode(self, mode):
        if mode == 'never_done':
            return True
        else:
            return False

    def _done_of_wrong_toward_ori(self, threshold=0.6):
        toward_ori = self._get_toward_ori()
        return toward_ori[2] < threshold

    def _done_of_low_height(self, threshold=0.25):
        self._get_body_pos_vel_info()
        return self.body_pos[2] < threshold

    def _done_of_too_long(self, threshold=300):
        return self._env.env_step_counter > threshold

    def reward(self, env):
        del env
        return

    def done(self, env):
        del env
        if self.mode == 'never_done':  # only use in test mode
            return False

    def update(self, env):
        pass

    def _bad_end(self):
        return False

    def _get_pybullet_client(self):
       #Get bullet client from the environmen.
        return self._env._pybullet_client

    def _get_num_joints(self):
        #Get the number of joints in the character's body.
        pyb = self._get_pybullet_client()
        return pyb.getNumJoints(self._env.robot.quadruped)

    def _get_body_pos_vel_info(self):
        pyb = self._get_pybullet_client()
        self.body_pos = pyb.getBasePositionAndOrientation(self.quadruped)[0]  # 3 list: position list of 3 floats
        self.body_ori = pyb.getBasePositionAndOrientation(self.quadruped)[1]  # 4 list: orientation as list of 4 floats in [x,y,z,w] order
        self.body_lin_vel = pyb.getBaseVelocity(self.quadruped)[0]  # 3 list: linear velocity [x,y,z]
        self.body_ang_vel = pyb.getBaseVelocity(self.quadruped)[1]  # 3 list: angular velocity [wx,wy,wz]

    def _get_joint_pos_vel_info(self):
        pyb = self._get_pybullet_client()
        self.joint_pos = []  # float: the position value of this joint
        self.joint_vel = []  # float: the velocity value of this joint
        self.joint_tor = []  # float: the torque value of this joint
        # No.3, 7, 11, 15 are all joints between foot and toe, which is no motor in it.
        for i in range(16):
            if i % 4 != 3:
                self.joint_pos.append(pyb.getJointState(self.quadruped, i)[0])
                self.joint_vel.append(pyb.getJointState(self.quadruped, i)[1])
                self.joint_tor.append(pyb.getJointState(self.quadruped, i)[3])

    # There are 16 links in the laikago.
    # No.0, 4, 8, 12 are upper legs
    # No.1, 5, 9, 13 are lower legs
    # No.2, 6, 10, 14 are feet
    # No,3, 7, 11, 15 are toes
    def _get_collision_info(self):
        pyb = self._get_pybullet_client()
        ground = self._env._world_dict["ground"]
        contact_points = pyb.getContactPoints(bodyA=self.quadruped, bodyB=ground)
        contact_ids = [point[3] for point in contact_points]
        collision_info = []
        num_joints = pyb.getNumJoints(self.quadruped)
        for i in range(num_joints):
            if i in contact_ids:
                collision_info.append(True)
            else:
                collision_info.append(False)
        return collision_info

    def _get_pos_of_upper(self):
        pyb = self._get_pybullet_client()
        upper_indexes = [0, 4, 8, 12]
        pos = [pyb.getLinkState(self.quadruped, i)[0] for i in upper_indexes]
        return pos

    def _get_pos_of_lower(self):
        pyb = self._get_pybullet_client()
        lower_indexes = [1, 5, 9, 13]
        pos = [pyb.getLinkState(self.quadruped, i)[0] for i in lower_indexes]
        return pos

    def _get_pos_of_feet(self):
        pyb = self._get_pybullet_client()
        foot_indexes = [2, 6, 10, 14]
        pos = [pyb.getLinkState(self.quadruped, i)[0] for i in foot_indexes]
        return pos

    def _get_pos_of_toes(self):
        pyb = self._get_pybullet_client()
        toe_indexes = [3, 7, 11, 15]
        pos = [pyb.getLinkState(self.quadruped, i)[0] for i in toe_indexes]
        return pos

    def _get_toe_upper_distance(self):
        pos_toe = np.array(self._get_pos_of_toes())
        pos_upper = np.array(self._get_pos_of_upper())
        distances = np.sqrt(np.sum((pos_toe - pos_upper) ** 2, axis=1))
        return distances

    # This direction is concerning rotation, which is the direction the head faces
    def _get_current_face_ori(self):
        self._get_body_pos_vel_info()
        return bq(self.body_ori).ori([0, 0, 1])

    # This direction is not concerning rotation, which is the direction the back faces
    def _get_toward_ori(self):
        self._get_body_pos_vel_info()
        return bq(self.body_ori).ori([0, 1, 0])

class PushTask(LaikagoTask):
    def __init__(self,
                 mode='train',
                 force=True,
                 max_force=3000,
                 force_delay_steps=3):
        super(PushTask, self).__init__(mode)
        self.force = force
        self._get_force_ori()
        self.max_force = max_force
        self.force_delay_steps = force_delay_steps

        self.now_force = None
        return


    def _get_force_ori(self):
        self.force_ori = []
        f_ori = [[1, 0, 0], [0, 1, 0], [-1, 0, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]]
        for i in f_ori:
            for j in f_ori:
                ori = [o[0]+o[1] for o in zip(i,j)]
                self.force_ori.append(ori)

    def _give_force(self):
        if self._env.env_step_counter % self.force_delay_steps == 0:
            force_id = random.randint(0, len(self.force_ori) - 1)
            ori = self.force_ori[force_id]
            self.now_force = [f * random.random() * self.max_force for f in ori]
        return self.now_force

    def update(self, env):
        if not self.force:
            return
        force = self._give_force()
        self.body_pos = env._pybullet_client.getBasePositionAndOrientation(self.quadruped)[0]
        env._pybullet_client.applyExternalForce(objectUniqueId=self.quadruped, linkIndex=-1,
                             forceObj=force, posObj=self.body_pos, flags=env._pybullet_client.WORLD_FRAME)