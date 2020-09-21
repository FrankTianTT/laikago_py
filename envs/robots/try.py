from envs.robots.laikago import Laikago
from robots import laikago
import pybullet
import pybullet_utils.bullet_client as bullet_client
import pybullet_data as pd
from build_envs.sensors import sensor_wrappers
from build_envs.sensors import robot_sensors
import numpy as np
import time
from envs.robots.robot_config import MotorControlMode
sensors = [
        sensor_wrappers.HistoricSensorWrapper(
            wrapped_sensor=robot_sensors.ToeTouchSensor(4), num_history=3)
    ]

pyb = bullet_client.BulletClient(connection_mode=pybullet.GUI)
pyb.setGravity(0, 0, -10)
pyb.setAdditionalSearchPath(pd.getDataPath())
ground = pyb.loadURDF("plane_implicit.urdf")
robot = Laikago(pybullet_client=pyb, sensors=sensors, on_rack=False, init_pose='lie', motor_control_mode=MotorControlMode.POSITION)
quadruped = robot.quadruped
num_joints = pyb.getNumJoints(quadruped)
chassis_link_ids = [-1]
action = np.array([0, 90, -155,
                   0, 90, -155,
                   0, 180, -155,
                   0, 180, -155]) * np.pi / 180
# action = np.zeros(12)


while True:
    time.sleep(1)
    print(robot.GetTrueMotorTorques())
    robot.Step(np.array(action))

