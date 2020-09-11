from envs.robots.laikago import Laikago
from robots import laikago
import pybullet
import pybullet_utils.bullet_client as bullet_client
import pybullet_data as pd
from build_envs.sensors import sensor_wrappers
from build_envs.sensors import robot_sensors
import numpy as np

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
pyb.loadURDF("plane_implicit.urdf")
robot = Laikago(pybullet_client=pyb, sensors=sensors, on_rack=True)

while True:
    robot.Step(np.array([-60*np.pi/180,0,0,0,0,0,0,0,0,0,0,0]))
    quadruped = robot.quadruped

    joint_pos = []  # float: the position value of this joint
    joint_vel = []  # float: the velocity value of this joint
    joint_tor = []  # float: the torque value of this joint
    for i in range(16):
        joint_pos.append(pyb.getJointState(quadruped, i)[0])
        joint_vel.append(pyb.getJointState(quadruped, i)[1])
        joint_tor.append(pyb.getJointState(quadruped, i)[3])
    joint_pos = np.array(joint_pos)/np.pi * 180
    print(joint_pos[0],joint_pos[4])