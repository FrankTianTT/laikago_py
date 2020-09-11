from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback
import standupheight.standupheight_env_builder as env_builder

VERSION = 0
FORCE = True

env = env_builder.build_env(enable_randomizer=True, version=VERSION, enable_rendering=False, force=FORCE)

model = SAC.load("logs/v{}/best_model".format(VERSION))

total_reward = 0
obs = env.reset()
for i in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    total_reward += reward
    if done:
      obs = env.reset()
      print('Test reward is {:.3f}.'.format(total_reward))
      total_reward = 0
env.close()
