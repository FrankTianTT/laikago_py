import math
from envs.build_envs.utilities.quaternion import bullet_quaternion as bq
import random
class LaikagoTask(object):
    def __init__(self,
                 weight=1.0,
                 pose_weight=0.5,
                 velocity_weight=0.05,
                 end_effector_weight=0.2,
                 root_pose_weight=0.15,
                 root_velocity_weight=0.1,
                 mode='train'):
        self._env = None
        self._weight = weight

        # reward function parameters
        self._pose_weight = pose_weight
        self._velocity_weight = velocity_weight
        self._end_effector_weight = end_effector_weight
        self._root_pose_weight = root_pose_weight
        self._root_velocity_weight = root_velocity_weight
        self.mode = mode

        self.body_pos = None
        self.body_ori = None
        self.body_lin_vel = None
        self.body_ang_vel = None
        self.joint_pos = None
        self.joint_vel = None
        self.joint_tor = None
        return

    def __call__(self, env):
        return self.reward(env)

    def reset(self,env):
        self._env = env
        self.quadruped = self._env.robot.quadruped
        self._get_body_pos_vel_info()
        return

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
        self._get_joint_pos_vel_info()
        reward = - sum([abs(p[0] * p[1]) for p in zip(self.joint_tor, self.joint_vel)])
        return self.normalize_reward(reward, -5, 0)  # the min reward by sample action is -4.

    def _reward_of_sum_vel(self):
        self._get_joint_pos_vel_info()
        reward = - sum([abs(v) for v in self.joint_vel])
        return self.normalize_reward(reward, -300, 0)  # the min reward by sample action is -275.

    # use it when try to stand up.
    def _reward_of_toes_height(self):
        toes_height = self._get_height_of_toes()
        reward = - sum(h**2 for h in toes_height)
        return self.normalize_reward(reward, -1, 0)  # the min reward by sample action is -0.8.

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

    def _get_height_of_toes(self):
        pyb = self._get_pybullet_client()
        toe_indexes = [3, 7, 11, 15]
        heights = [pyb.getLinkState(self.quadruped, i)[0][2] for i in toe_indexes]
        return heights

    # This direction is concerning rotation, which is the direction the head faces
    def _get_current_face_ori(self):
        self._get_body_pos_vel_info()
        return bq(self.body_ori).ori([0, 0, 1])

    # This direction is not concerning rotation, which is the direction the back faces
    def _get_toward_ori(self):
        self._get_body_pos_vel_info()
        return bq(self.body_ori).ori([0, 1, 0])