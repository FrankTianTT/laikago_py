#!/usr/bin/env python3
# by frank tian on 7.13.2020
################################
#change these when changing task
import runstraight.runstraight_env_builder as env_builder
TASK_NAME = "runstraight"
FILE_NAME = "best_+477.537_11000.dat"
################################

from network_model import trpo_model as model
import numpy as np
import torch
import os

TASK_DIR = os.path.dirname(os.path.abspath(__file__))
LOAD_FILE = os.path.join(TASK_DIR, 'saves', "trpo-"+TASK_NAME, FILE_NAME)


if __name__ == "__main__":
    env = env_builder.build_env(enable_randomizer=True, enable_rendering=True)

    net = model.TRPOActor(env.observation_space.shape[0], env.action_space.shape[0])
    net.load_state_dict(torch.load(LOAD_FILE))
    for i in range(100):
        obs = env.reset()
        total_reward = 0.0
        total_steps = 0
        while True:
            obs_v = torch.FloatTensor([obs])
            mu_v = net(obs_v)
            action = mu_v.squeeze(dim=0).data.numpy()
            action = np.clip(action, -1, 1)
            obs, reward, done, _ = env.step(action)
            total_reward += reward
            total_steps += 1
            if done:
                break
        print("In %d steps we got %.3f reward" % (total_steps, total_reward))