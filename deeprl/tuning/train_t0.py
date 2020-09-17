from stable_baselines3 import SAC
import pybullet_envs
import gym
from stable_baselines3.common.callbacks import EvalCallback

BUFFER_SIZE = int(1e6)
LEARNING_STARTS = int(1e4)
BATCH_SIZE = 64
ENT_COEF = 0.05

ENV_NAME = 'Walker2DBulletEnv-v0'
TIME_STEPS = 100000

env = gym.make(ENV_NAME)
eval_env = gym.make(ENV_NAME)

eval_callback = EvalCallback(eval_env, best_model_save_path='./logs/',
                             log_path='./logs/', eval_freq=500,
                             deterministic=True, render=False)
model = SAC('MlpPolicy', env, verbose=1, tensorboard_log="./log/",
            buffer_size=BUFFER_SIZE,
            batch_size=BATCH_SIZE,
            learning_starts=LEARNING_STARTS,
            ent_coef=ENT_COEF)

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