from stable_baselines3 import SAC
import pybullet_envs
import gym
from stable_baselines3.common.callbacks import EvalCallback

ENV_NAME = 'HopperBulletEnv-v0'
TIME_STEPS = 100000

env = gym.make(ENV_NAME)
eval_env = gym.make(ENV_NAME)

eval_callback = EvalCallback(eval_env, best_model_save_path='./logs/',
                             log_path='./logs/', eval_freq=500,
                             deterministic=True, render=False)
model = SAC('MlpPolicy', env, verbose=1, tensorboard_log="./log/")
model.learn(total_timesteps=TIME_STEPS, callback=eval_callback)

env.render()
obs = env.reset()
for i in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()
    if done:
      obs = env.reset()
env.close()