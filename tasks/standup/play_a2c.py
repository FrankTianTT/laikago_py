#!/usr/bin/env python3
import argparse
import gym
import pybullet_envs
import time
import math
from algorithms import a2c_model as model
import numpy as np
import torch
import os
import envs.build_envs.standup_env_builder as env_builder

TASK_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(TASK_DIR, 'saves')
A2C_DIR = os.path.join(MODEL_DIR, "a2c-standup")
LOAD_FILE = os.path.join(A2C_DIR, "best_+1318.009_31000.dat")

if __name__ == "__main__":
    env =env_builder.build_standup_env(enable_randomizer=True,enable_rendering=True)

    net = model.A2C(env.observation_space.shape[0], env.action_space.shape[0])
    net.load_state_dict(torch.load(LOAD_FILE))
    for i in range(100):
        obs = env.reset()
        total_reward = 0.0
        total_steps = 0
        while True:
            obs_v = torch.FloatTensor([obs])
            mu_v, var_v, val_v = net(obs_v)
            action = mu_v.squeeze(dim=0).data.numpy()
            action = np.clip(action, -1, 1)
            obs, reward, done, _ = env.step(action)
            total_reward += reward
            total_steps += 1
            if done:
                break
        print("In %d steps we got %.3f reward" % (total_steps, total_reward))
