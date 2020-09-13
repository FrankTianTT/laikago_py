from envs.robots.laikago import Laikago
from robots import laikago
import pybullet
import pybullet_utils.bullet_client as bullet_client
import pybullet_data as pd
from build_envs.sensors import sensor_wrappers
from build_envs.sensors import robot_sensors
import numpy as np
import time

sensors = [
        sensor_wrappers.HistoricSensorWrapper(
            wrapped_sensor=robot_sensors.MotorAngleSensor(num_motors=laikago.NUM_MOTORS), num_history=3),
        sensor_wrappers.HistoricSensorWrapper(
            wrapped_sensor=robot_sensors.MotorVelocitiySensor(num_motors=laikago.NUM_MOTORS), num_history=3),
        sensor_wrappers.HistoricSensorWrapper(wrapped_sensor=robot_sensors.IMUSensor(), num_history=3)
    ]

pyb = bullet_client.BulletClient(connection_mode=pybullet.GUI)
pyb.setGravity(0, 0, -10)
pyb.setAdditionalSearchPath(pd.getDataPath())
ground = pyb.loadURDF("plane_implicit.urdf")
robot = Laikago(pybullet_client=pyb, sensors=sensors, on_rack=False)
quadruped = robot.quadruped
num_joints = pyb.getNumJoints(quadruped)
chassis_link_ids = [-1]
while True:
    robot.Step(np.array([0,0,0,0,0,0,0,0,0,0,0,0]))
    quadruped = robot.quadruped

    joint_pos = []  # float: the position value of this joint
    joint_vel = []  # float: the velocity value of this joint
    joint_tor = []  # float: the torque value of this joint
    for i in range(16):
        joint_pos.append(pyb.getJointState(quadruped, i)[0])
        joint_vel.append(pyb.getJointState(quadruped, i)[1])
        joint_tor.append(pyb.getJointState(quadruped, i)[3])
    joint_pos = np.array(joint_pos)/np.pi * 180

    '''
    There are 16 joints in the laikago.
    No.0, 4, 8, 12 are joints between hip-motor and chassis
    No.1, 5, 9, 13 are joints between upper-leg and hip-motor
    No.2, 6, 10, 14 are joints between lower_leg and upper-leg
    No,3, 7, 11, 15 are joints of toes
    '''

    toes_contact = robot.GetFootContacts()
    time.sleep(0.1)
    print(toes_contact)