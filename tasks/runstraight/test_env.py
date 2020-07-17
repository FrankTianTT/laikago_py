import runstraight.runstraight_env_builder as env_builder
import numpy as np
import math
import time
env = env_builder.build_env(enable_randomizer=False, enable_rendering=True)

ini_ori = [0,math.sin(0.1),0,math.cos(0.1)]
revserse_ori = [0,math.sin(math.pi/2),0,math.cos(math.pi/2)]

theta = 0
while(1):
    theta += 0.01
    ori = [ 0,0, -math.sin(theta/2),math.cos(theta/2)]
    quadruped = env.robot.quadruped
    env._pybullet_client.resetBasePositionAndOrientation(quadruped,[0,0,1],ori)
    action = np.array([0,0,0,0,0,0,0,0,0,0,0,0])
    env.step(action)

    print(theta/math.pi *180)
    time.sleep(0.1)