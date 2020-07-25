import runwithcoord.runwithcoord_env_builder as env_builder
import numpy as np
import math
import time
from envs.build_envs.utilities.quaternion import bullet_quaternion as bq
env = env_builder.build_env(enable_randomizer=False, enable_rendering=True)
for i in range(100):
    quadruped = env.robot.quadruped
    action = np.array([0,0,0,0,0,0,0,0,0,0,0,0])
    o, _, _, _ = env.step(action)
    print(len(o))
    print(o[-5:-1])
    time.sleep(1)