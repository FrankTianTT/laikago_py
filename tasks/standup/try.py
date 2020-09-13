from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback
import standup.standup_env_builder as env_builder

ENV_NAME = "Stand-Up"
TIME_STEPS = 100000
VERSION = 2

env = env_builder.build_env(enable_randomizer=True, version=VERSION, enable_rendering=True)

model = SAC.load("logs/v0/best_model")

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