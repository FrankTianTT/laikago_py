from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback
import runstraight.runstraight_env_builder as env_builder

ENV_NAME = 'HopperBulletEnv-v0'
TIME_STEPS = 100000

env = env_builder.build_env(enable_randomizer=True, enable_rendering=False)
max_v = 0
min_v = 0

obs = env.reset()
for i in range(1000):

    pyb = env._pybullet_client
    quadruped = env.robot.quadruped
    joint_vel = []#float: the velocity value of this joint
    for i in range(12):
        joint_vel.append(pyb.getJointState(quadruped, i)[1])

    if max(joint_vel) > max_v:
        max_v = max(joint_vel)
    if min(joint_vel) < min_v:
        min_v = min(joint_vel)
    action= env.action_space.sample()
    obs, reward, done, info = env.step(action)

env.close()
print(min_v, max_v)