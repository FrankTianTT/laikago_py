import math

class RunstraightTask(object):
    def __init__(self,
                 weight=1.0,
                 pose_weight=0.5,
                 velocity_weight=0.05,
                 end_effector_weight=0.2,
                 root_pose_weight=0.15,
                 root_velocity_weight=0.1):
        self._env = None
        self._weight = weight

        # reward function parameters
        self._pose_weight = pose_weight
        self._velocity_weight = velocity_weight
        self._end_effector_weight = end_effector_weight
        self._root_pose_weight = root_pose_weight
        self._root_velocity_weight = root_velocity_weight

        self.body_pos = None
        self.body_ori = None
        self.body_lin_vel = None
        self.body_ang_vel = None
        self.joint_pos = None
        self.joint_vel = None


        return

    def __call__(self, env):
        return self.reward(env)

    def reset(self,env):
        self._env = env
        return

    def reward(self, env):
        """Get the reward without side effects."""
        del env
        self._get_pos_vel_info()

        reward = self.body_lin_vel[0] + self.body_pos[2]*0.3
        return reward

    def done(self, env):
        """Checks if the episode is over."""
        del env
        if self._env.env_step_counter > 3000:
            return True
        self._get_pos_vel_info()
        print(self._env.env_step_counter,self.body_lin_vel[0])
        done = self.body_pos[2] < 0.15 or (self._env.env_step_counter > 100 and self.body_lin_vel[0] < 0.15)
        return done

    def _get_pybullet_client(self):
        """Get bullet client from the environment"""
        return self._env._pybullet_client

    def _get_num_joints(self):
        """Get the number of joints in the character's body."""
        pyb = self._get_pybullet_client()
        return pyb.getNumJoints(self._env.robot.quadruped)

    def _get_pos_vel_info(self):
        pyb = self._get_pybullet_client()
        quadruped = self._env.robot.quadruped
        self.body_pos = pyb.getBasePositionAndOrientation(quadruped)[0]#3 list: position list of 3 floats
        self.body_ori = pyb.getBasePositionAndOrientation(quadruped)[1]#4 list: orientation as list of 4 floats in [x,y,z,w] order
        self.body_lin_vel = pyb.getBaseVelocity(quadruped)[0]#3 list: linear velocity [x,y,z]
        self.body_ang_vel = pyb.getBaseVelocity(quadruped)[1]#3 list: angular velocity [wx,wy,wz]
        self.joint_pos = []#float: the position value of this joint
        self.joint_vel = []#float: the velocity value of this joint
        for i in range(12):
            self.joint_pos.append(pyb.getJointState(quadruped,i)[0])
            self.joint_vel.append(pyb.getJointState(quadruped,i)[1])