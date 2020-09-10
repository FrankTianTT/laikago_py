from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback
import standupnomove.standupnomove_env_builder as env_builder

TASK_NAME = "Stand-Up-NoMove"
TIME_STEPS = 100000

env = env_builder.build_env(enable_randomizer=True, enable_rendering=True)

model = SAC.load("logs/best_model")

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