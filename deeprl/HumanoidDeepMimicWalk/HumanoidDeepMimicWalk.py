from stable_baselines3 import SAC
import pybullet_envs
import gym
from stable_baselines3.common.callbacks import EvalCallback

ENV_NAME = 'HumanoidDeepMimicWalkBulletEnv-v1'

buffer_size = 1000000
batch_size = 64
learning_starts = 10000
ent_coef = 0.1
time_steps = 5000000

env = gym.make(ENV_NAME)
eval_env = gym.make(ENV_NAME)

eval_callback = EvalCallback(eval_env, best_model_save_path='./logs/',
                             log_path='./logs/', eval_freq=500,
                             deterministic=True, render=False)
model = SAC('MlpPolicy',
            env,
            verbose=1,
            tensorboard_log="./log/",
            buffer_size=buffer_size,
            batch_size=batch_size,
            learning_starts=learning_starts,
            ent_coef=ent_coef)
model.learn(total_timesteps=time_steps, callback=eval_callback)