from stable_baselines3 import SAC
import pybullet_envs
import gym
import torch
from stable_baselines3.common.callbacks import EvalCallback

ENV_NAME = 'HalfCheetahBulletEnv-v0'
TIME_STEPS = 100000

env = gym.make(ENV_NAME)
eval_env = gym.make(ENV_NAME)

policy_kwargs = dict(activation_fn=torch.nn.ReLU, net_arch=[256, 256, 256])

model = SAC("MlpPolicy", env, policy_kwargs=policy_kwargs, verbose=1)

total_reward = 0
# env.render()
# obs = env.reset()
# for i in range(100000):
#     action, _states = model.predict(obs, deterministic=True)
#     obs, reward, done, info = env.step(action)
#     total_reward += reward
#     env.render()
#     if done:
#       obs = env.reset()
#       print('Test reward is {:.3f}.'.format(total_reward))
#       total_reward = 0
# env.close()

print(model.policy)