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
action = [-1,0,0,1,0,0,-1,0,0,1,0,0]


while True:
    robot.Step(np.array(action))
    body_pos = pyb.getBasePositionAndOrientation(quadruped)[0]
    aver = sum([pyb.getLinkState(quadruped, i)[0][2] for i in [0,3,6,9]])/4
    body_state = pyb.getLinkState(quadruped, 0)[0]
    height = body_pos[2]
    body_h = body_state[2]
    action[0] += 0.001
    action[3] -= 0.001
    action[6] += 0.001
    action[9] -= 0.001
    print(height, aver)





    # contact_points = pyb.getContactPoints(bodyA=quadruped, bodyB=ground)
    # contact_ids = [point[3] for point in contact_points]
    # collision_info = []
    # num_joints = pyb.getNumJoints(quadruped)
    # for i in range(num_joints):
    #     if i in contact_ids:
    #         collision_info.append(True)
    #     else:
    #         collision_info.append(False)
    #
    # info = [[],[],[],[]]
    # for i, c in enumerate(collision_info):
    #     info[i % 4].append(c)
    # print(info)