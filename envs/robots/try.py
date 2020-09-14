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
            wrapped_sensor=robot_sensors.ToeTouchSensor(4), num_history=3)
    ]

pyb = bullet_client.BulletClient(connection_mode=pybullet.GUI)
pyb.setGravity(0, 0, -10)
pyb.setAdditionalSearchPath(pd.getDataPath())
ground = pyb.loadURDF("plane_implicit.urdf")
robot = Laikago(pybullet_client=pyb, sensors=sensors, on_rack=False)
quadruped = robot.quadruped
num_joints = pyb.getNumJoints(quadruped)
chassis_link_ids = [-1]
action = [0,0,0,0,0,0,0,0,0,0,0,0]


def _get_pos_of_upper(  ):
    global pyb
    upper_indexes = [0, 4, 8, 12]
    pos = [pyb.getLinkState( quadruped, i)[0] for i in upper_indexes]
    return pos


def _get_pos_of_lower(  ):
    global pyb
    lower_indexes = [1, 5, 9, 13]
    pos = [pyb.getLinkState( quadruped, i)[0] for i in lower_indexes]
    return pos


def _get_pos_of_feet(  ):
    global pyb
    foot_indexes = [2, 6, 10, 14]
    pos = [pyb.getLinkState( quadruped, i)[0] for i in foot_indexes]
    return pos


def _get_pos_of_toes(  ):
    global pyb
    toe_indexes = [3, 7, 11, 15]
    pos = [pyb.getLinkState( quadruped, i)[0] for i in toe_indexes]
    return pos


def _get_toe_upper_distance(  ):
    pos_toe = np.array( _get_pos_of_toes())
    pos_upper = np.array( _get_pos_of_upper())
    distances = np.sqrt(np.sum((pos_toe - pos_upper) ** 2, axis=1))
    return distances

while True:
    robot.Step(np.array(action))
    distances = _get_toe_upper_distance()
    print(distances)
