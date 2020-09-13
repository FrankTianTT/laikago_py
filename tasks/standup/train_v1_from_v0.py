from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback
import standup.standup_env_builder as env_builder
import torch

TASK_NAME = "Stand-Up"
TIME_STEPS = 5000000
VERSION = 1
MODEL_VERSION = 0

env = env_builder.build_env(enable_randomizer=True, version=VERSION, enable_rendering=False)
eval_env = env_builder.build_env(enable_randomizer=True, version=VERSION, enable_rendering=False)

eval_callback = EvalCallback(eval_env, best_model_save_path='./logs/v{}_from_v{}/'.format(VERSION, MODEL_VERSION),
                             log_path='./logs/v{}_from_v{}/'.format(VERSION, MODEL_VERSION), eval_freq=1000,
                             deterministic=True, render=False)
policy_kwargs = dict(activation_fn=torch.nn.ReLU, net_arch=[256, 256])
model = SAC.load("logs/v{}/best_model".format(MODEL_VERSION))
model.set_env(env)
# model.learn(total_timesteps=TIME_STEPS, callback=eval_callback, tb_log_name='v{}_from_v{}/'.format(VERSION, MODEL_VERSION))
#
# env.render()
# obs = env.reset()
# for i in range(1000):
#     action, _states = model.predict(obs, deterministic=True)
#     obs, reward, done, info = env.step(action)
#     env.render()
#     if done:
#       obs = env.reset()
# env.close()