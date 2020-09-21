# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Pybullet simulation of a Laikago robot."""
import math
import os
import re
import numpy as np
import pybullet as pyb

from robots import laikago_pose_utils
from robots import laikago_constants
from robots import laikago_motor
from robots import minitaur
from robots import robot_config
from build_envs import locomotion_gym_config
from envs.robots.robot_config import MotorControlMode


NUM_MOTORS = 12
NUM_LEGS = 4
MOTOR_NAMES = [
    "FR_hip_motor_2_chassis_joint",
    "FR_upper_leg_2_hip_motor_joint",
    "FR_lower_leg_2_upper_leg_joint",
    "FL_hip_motor_2_chassis_joint",
    "FL_upper_leg_2_hip_motor_joint",
    "FL_lower_leg_2_upper_leg_joint",
    "RR_hip_motor_2_chassis_joint",
    "RR_upper_leg_2_hip_motor_joint",
    "RR_lower_leg_2_upper_leg_joint",
    "RL_hip_motor_2_chassis_joint",
    "RL_upper_leg_2_hip_motor_joint",
    "RL_lower_leg_2_upper_leg_joint",
]
INIT_RACK_POSITION = [0, 0, 1]
STAND_INIT_POSITION = [0, 0, 0.48]
LIE_INIT_POSITION = [0, 0, 0.15]
JOINT_DIRECTIONS = np.array([-1, 1, 1, 1, 1, 1, -1, 1, 1, 1, 1, 1])
HIP_JOINT_OFFSET = 0.0
UPPER_LEG_JOINT_OFFSET = -0.6
KNEE_JOINT_OFFSET = 0.66
DOFS_PER_LEG = 3
JOINT_OFFSETS = np.array(
    [HIP_JOINT_OFFSET, UPPER_LEG_JOINT_OFFSET, KNEE_JOINT_OFFSET] * 4)
PI = math.pi

MAX_MOTOR_ANGLE_CHANGE_PER_STEP = 0.2
_DEFAULT_HIP_POSITIONS = (
    (0.21, -0.1157, 0),
    (0.21, 0.1157, 0),
    (-0.21, -0.1157, 0),
    (-0.21, 0.1157, 0),
)

ABDUCTION_P_GAIN = 220.0
ABDUCTION_D_GAIN = 0.3
HIP_P_GAIN = 220.0
HIP_D_GAIN = 2.0
KNEE_P_GAIN = 220.0
KNEE_D_GAIN = 2.0

# Bases on the readings from Laikago's default pose.
INIT_MOTOR_ANGLES = np.array([laikago_pose_utils.LAIKAGO_DEFAULT_ABDUCTION_ANGLE,
                              laikago_pose_utils.LAIKAGO_DEFAULT_HIP_ANGLE,
                              laikago_pose_utils.LAIKAGO_DEFAULT_KNEE_ANGLE] * NUM_LEGS)

_UPPER_NAME_PATTERN = re.compile(r"\w{2}_hip_motor_2_chassis_joint")
_LOWER_NAME_PATTERN = re.compile(r"\w{2}_upper_leg_2_hip_motor_joint")
_FOOT_NAME_PATTERN = re.compile(r"\w{2}_lower_leg_2_upper_leg_joint")
_TOE_NAME_PATTERN = re.compile(r"jtoe\w{2}")

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
URDF_FILENAME = ROOT_DIR + "/laikago_model/laikago_toes_limits.urdf"

_BODY_B_FIELD_NUMBER = 2
_LINK_A_FIELD_NUMBER = 3

L_HIP_UPPER_BOUND = 60*math.pi /180
L_HIP_LOWER_BOUND = -50*math.pi /180
R_HIP_UPPER_BOUND = 50*math.pi /180
R_HIP_LOWER_BOUND = -60*math.pi /180
UPPER_LEG_UPPER_BOUND = 225*math.pi /180
UPPER_LEG_LOWER_BOUND = -30*math.pi /180
LOWER_LEG_UPPER_BOUND = -35*math.pi /180
LOWER_LEG_LOWER_BOUND = -159*math.pi /180

LIE_MOTOR_ANGLES = np.array([0, 90, -155,
                   0, 90, -155,
                   0, 180, -155,
                   0, 180, -155]) * np.pi / 180 + JOINT_OFFSETS
STAND_MOTOR_ANGLES = np.array([0, 40, -75,
                   0, 40, -75,
                   0, 40, -75,
                   0, 40, -75]) * np.pi / 180 + JOINT_OFFSETS

UPPER_TORQUE = 40
LOWER_TORQUE = - 40

class Laikago(minitaur.Minitaur):
    """A simulation for the Laikago robot."""

    POSITION_ACTION_CONFIG = [
        locomotion_gym_config.ScalarField(name="motor_angle_0", upper_bound=R_HIP_UPPER_BOUND, lower_bound=R_HIP_LOWER_BOUND),
        locomotion_gym_config.ScalarField(name="motor_angle_1", upper_bound=UPPER_LEG_UPPER_BOUND, lower_bound=UPPER_LEG_LOWER_BOUND),
        locomotion_gym_config.ScalarField(name="motor_angle_2", upper_bound=LOWER_LEG_UPPER_BOUND, lower_bound=LOWER_LEG_LOWER_BOUND),
        locomotion_gym_config.ScalarField(name="motor_angle_3", upper_bound=L_HIP_UPPER_BOUND, lower_bound=L_HIP_LOWER_BOUND),
        locomotion_gym_config.ScalarField(name="motor_angle_4", upper_bound=UPPER_LEG_UPPER_BOUND, lower_bound=UPPER_LEG_LOWER_BOUND),
        locomotion_gym_config.ScalarField(name="motor_angle_5", upper_bound=LOWER_LEG_UPPER_BOUND, lower_bound=LOWER_LEG_LOWER_BOUND),
        locomotion_gym_config.ScalarField(name="motor_angle_6", upper_bound=R_HIP_UPPER_BOUND, lower_bound=R_HIP_LOWER_BOUND),
        locomotion_gym_config.ScalarField(name="motor_angle_7", upper_bound=UPPER_LEG_UPPER_BOUND, lower_bound=UPPER_LEG_LOWER_BOUND),
        locomotion_gym_config.ScalarField(name="motor_angle_8", upper_bound=LOWER_LEG_UPPER_BOUND, lower_bound=LOWER_LEG_LOWER_BOUND),
        locomotion_gym_config.ScalarField(name="motor_angle_9", upper_bound=L_HIP_UPPER_BOUND, lower_bound=L_HIP_LOWER_BOUND),
        locomotion_gym_config.ScalarField(name="motor_angle_10", upper_bound=UPPER_LEG_UPPER_BOUND, lower_bound=UPPER_LEG_LOWER_BOUND),
        locomotion_gym_config.ScalarField(name="motor_angle_11", upper_bound=LOWER_LEG_UPPER_BOUND, lower_bound=LOWER_LEG_LOWER_BOUND)
    ]
    TORQUE_ACTION_CONFIG = [
        locomotion_gym_config.ScalarField(name="motor_angle_0", upper_bound=UPPER_TORQUE, lower_bound=LOWER_TORQUE),
        locomotion_gym_config.ScalarField(name="motor_angle_1", upper_bound=UPPER_TORQUE, lower_bound=LOWER_TORQUE),
        locomotion_gym_config.ScalarField(name="motor_angle_2", upper_bound=UPPER_TORQUE, lower_bound=LOWER_TORQUE),
        locomotion_gym_config.ScalarField(name="motor_angle_3", upper_bound=UPPER_TORQUE, lower_bound=LOWER_TORQUE),
        locomotion_gym_config.ScalarField(name="motor_angle_4", upper_bound=UPPER_TORQUE, lower_bound=LOWER_TORQUE),
        locomotion_gym_config.ScalarField(name="motor_angle_5", upper_bound=UPPER_TORQUE, lower_bound=LOWER_TORQUE),
        locomotion_gym_config.ScalarField(name="motor_angle_6", upper_bound=UPPER_TORQUE, lower_bound=LOWER_TORQUE),
        locomotion_gym_config.ScalarField(name="motor_angle_7", upper_bound=UPPER_TORQUE, lower_bound=LOWER_TORQUE),
        locomotion_gym_config.ScalarField(name="motor_angle_8", upper_bound=UPPER_TORQUE, lower_bound=LOWER_TORQUE),
        locomotion_gym_config.ScalarField(name="motor_angle_9", upper_bound=UPPER_TORQUE, lower_bound=LOWER_TORQUE),
        locomotion_gym_config.ScalarField(name="motor_angle_10", upper_bound=UPPER_TORQUE, lower_bound=LOWER_TORQUE),
        locomotion_gym_config.ScalarField(name="motor_angle_11", upper_bound=UPPER_TORQUE, lower_bound=LOWER_TORQUE)
    ]

    def __init__(self,
                 pybullet_client,
                 urdf_filename=URDF_FILENAME,
                 enable_clip_motor_commands=True,
                 time_step=0.001,
                 action_repeat=33,
                 sensors=None,
                 control_latency=0.002,
                 on_rack=False,
                 enable_action_interpolation=True,
                 enable_action_filter=True,
                 motor_control_mode=robot_config.MotorControlMode.POSITION,
                 init_pose='stand'):
        self._urdf_filename = urdf_filename

        self._enable_clip_motor_commands = enable_clip_motor_commands

        motor_kp = [ABDUCTION_P_GAIN, HIP_P_GAIN, KNEE_P_GAIN,
                    ABDUCTION_P_GAIN, HIP_P_GAIN, KNEE_P_GAIN,
                    ABDUCTION_P_GAIN, HIP_P_GAIN, KNEE_P_GAIN,
                    ABDUCTION_P_GAIN, HIP_P_GAIN, KNEE_P_GAIN]
        motor_kd = [ABDUCTION_D_GAIN, HIP_D_GAIN, KNEE_D_GAIN,
                    ABDUCTION_D_GAIN, HIP_D_GAIN, KNEE_D_GAIN,
                    ABDUCTION_D_GAIN, HIP_D_GAIN, KNEE_D_GAIN,
                    ABDUCTION_D_GAIN, HIP_D_GAIN, KNEE_D_GAIN]

        motor_torque_limits = None # jp hack
        if init_pose == 'lie':
            self.initial_motor_angles = LIE_MOTOR_ANGLES
        else:
            self.initial_motor_angles = STAND_MOTOR_ANGLES
        self.init_pose = init_pose

        super(Laikago, self).__init__(
            pybullet_client=pybullet_client,
            time_step=time_step,
            action_repeat=action_repeat,
            num_motors=NUM_MOTORS,
            dofs_per_leg=DOFS_PER_LEG,
            motor_direction=JOINT_DIRECTIONS,
            motor_control_mode=motor_control_mode,
            motor_offset=JOINT_OFFSETS,
            motor_overheat_protection=False,
            motor_model_class=laikago_motor.LaikagoMotorModel,
            motor_torque_limits=40,
            sensors=sensors,
            motor_kp=motor_kp,
            motor_kd=motor_kd,
            control_latency=control_latency,
            on_rack=on_rack,
            enable_action_interpolation=enable_action_interpolation,
            enable_action_filter=enable_action_filter)

        return

    def _SettleDownForReset(self, default_motor_angles, reset_time):
        self.ReceiveObservation()

        if reset_time <= 0:
            return
        for _ in range(500):
            self._StepInternal(
                INIT_MOTOR_ANGLES,
                motor_control_mode=robot_config.MotorControlMode.POSITION)
        if default_motor_angles is not None:
            num_steps_to_reset = int(reset_time / self.time_step)
            for _ in range(num_steps_to_reset):
                self._StepInternal(
                    default_motor_angles,
                    motor_control_mode=robot_config.MotorControlMode.POSITION)

    def GetHipPositionsInBaseFrame(self):
        return _DEFAULT_HIP_POSITIONS

    def GetToeContacts(self):
        all_contacts = self._pybullet_client.getContactPoints(bodyA=self.quadruped)
        contacts = [False, False, False, False]
        for contact in all_contacts:
            # Ignore self contacts
            if contact[_BODY_B_FIELD_NUMBER] == self.quadruped:
                continue
            try:
                toe_link_index = self._toe_link_ids.index(
                    contact[_LINK_A_FIELD_NUMBER])
                contacts[toe_link_index] = True
            except ValueError:
                continue
        return contacts

    def ComputeJacobian(self, leg_id):
        """Compute the Jacobian for a given leg."""
        # Because of the default rotation in the Laikago URDF, we need to reorder
        # the rows in the Jacobian matrix.
        return super(Laikago, self).ComputeJacobian(leg_id)[(2, 0, 1), :]

    def ResetPose(self, add_constraint):
        del add_constraint
        for name in self._joint_name_to_id:
            joint_id = self._joint_name_to_id[name]
            self._pybullet_client.setJointMotorControl2(
                bodyIndex=self.quadruped,
                jointIndex=(joint_id),
                controlMode=self._pybullet_client.VELOCITY_CONTROL,
                targetVelocity=0,
                force=0)
        for name, i in zip(MOTOR_NAMES, range(len(MOTOR_NAMES))):
            angle = self.initial_motor_angles[i]
            self._pybullet_client.resetJointState(
                self.quadruped, self._joint_name_to_id[name], angle, targetVelocity=0)

    def GetURDFFile(self):
        return self._urdf_filename

    def _BuildUrdfIds(self):
        """Build the link Ids from its name in the URDF file.

        Raises:
          ValueError: Unknown category of the joint name.
        """
        num_joints = self._pybullet_client.getNumJoints(self.quadruped)
        self._chassis_link_ids = [-1]
        self._upper_link_ids = []
        self._lower_link_ids = []
        self._foot_link_ids = []
        self._toe_link_ids = []

        for i in range(num_joints):
            joint_info = self._pybullet_client.getJointInfo(self.quadruped, i)
            joint_name = joint_info[1].decode("UTF-8")
            joint_id = self._joint_name_to_id[joint_name]
            if _UPPER_NAME_PATTERN.match(joint_name):
                self._upper_link_ids.append(joint_id)
            elif _LOWER_NAME_PATTERN.match(joint_name):
                self._lower_link_ids.append(joint_id)
            elif _FOOT_NAME_PATTERN.match(joint_name):
                self._foot_link_ids.append(joint_id)
            elif _TOE_NAME_PATTERN.match(joint_name):
                self._toe_link_ids.append(joint_id)
            else:
                raise ValueError("Unknown category of joint %s" % joint_name)
        self._chassis_link_ids.sort()
        self._motor_link_ids.sort()
        self._foot_link_ids.sort()
        self._toe_link_ids.sort()
        return

    def _GetMotorNames(self):
        return MOTOR_NAMES

    def _GetDefaultInitPosition(self):
        if self._on_rack:
            return INIT_RACK_POSITION
        elif self.init_pose == 'lie':
            return LIE_INIT_POSITION
        else:
            return STAND_INIT_POSITION

    def _GetDefaultInitOrientation(self):
        # The Laikago URDF assumes the initial pose of heading towards z axis,
        # and belly towards y axis. The following transformation is to transform
        # the Laikago initial orientation to our commonly used orientation: heading
        # towards -x direction, and z axis is the up direction.
        init_orientation = pyb.getQuaternionFromEuler([math.pi / 2.0, 0, math.pi / 2.0])
        return init_orientation

    def GetDefaultInitPosition(self):
        """Get default initial base position."""
        return self._GetDefaultInitPosition()

    def GetDefaultInitOrientation(self):
        """Get default initial base orientation."""
        return self._GetDefaultInitOrientation()

    def GetDefaultInitJointPose(self):
        """Get default initial joint pose."""
        joint_pose = (INIT_MOTOR_ANGLES + JOINT_OFFSETS) * JOINT_DIRECTIONS
        return joint_pose

    def ApplyAction(self, motor_commands, motor_control_mode=None):
        """Clips and then apply the motor commands using the motor model.

        Args:
          motor_commands: np.array. Can be motor angles, torques, hybrid commands,
            or motor pwms (for Minitaur only).N
          motor_control_mode: A MotorControlMode enum.
        """
        if self._motor_control_mode == MotorControlMode.POSITION and self._enable_clip_motor_commands:
            motor_commands = self._ClipMotorCommands(motor_commands)
        super(Laikago, self).ApplyAction(motor_commands, motor_control_mode)
        return

    def _ClipMotorCommands(self, motor_commands):
        """Clips motor commands.

        Args:
          motor_commands: np.array. Can be motor angles, torques, hybrid commands,
            or motor pwms (for Minitaur only).

        Returns:
          Clipped motor commands.
        """

        # clamp the motor command by the joint limit, in case weired things happens
        max_angle_change = MAX_MOTOR_ANGLE_CHANGE_PER_STEP
        current_motor_angles = self.GetMotorAngles()
        motor_commands = np.clip(motor_commands,
                                 current_motor_angles - max_angle_change,
                                 current_motor_angles + max_angle_change)
        return motor_commands

    @classmethod
    def GetConstants(cls):
        del cls
        return laikago_constants

if __name__ == "__main__":
    import pybullet_utils.bullet_client as bc
    import pybullet as p
    pybullet_client = bc.BulletClient(p.GUI)
    robot = Laikago(pybullet_client)