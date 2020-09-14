from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback
import standup.standup_env_builder as env_builder

VERSION = 0
TUNING = 0

env = env_builder.build_env(enable_randomizer=True, version=VERSION, enable_rendering=True)

model = SAC.load("logs/t{}/best_model".format(TUNING))

total_reward = 0
obs = env.reset()
for i in range(100000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    total_reward += reward
    if done:
      obs = env.reset()
      print('Test reward is {:.3f}.'.format(total_reward))
      total_reward = 0
env.close()