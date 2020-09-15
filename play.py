import sys
from os.path import abspath, join, dirname
sys.path.insert(0, join(abspath(dirname(__file__)), 'envs'))
sys.path.insert(0, join(abspath(dirname(__file__)), 'tasks'))
import importlib
import argparse
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback
import tasks
import torch


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--name", required=True, help="Name of task")
    parser.add_argument("-v", "--version", required=True,  help="Version of task")
    args = parser.parse_args()

    tasks.check_name(args.name)

    env_builder = importlib.import_module('{}.env_builder'.format(args.name))

    env = env_builder.build_env(enable_randomizer=True, version=args.version, enable_rendering=False)
    model = SAC.load('./tasks/{}/best_model/v{}/'.format(args.name, args.version))

    total_reward = 0
    obs = env.reset()
    for i in range(1000):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        total_reward += reward
        if done:
          obs = env.reset()
          print('Test reward is {:.3f}.'.format(total_reward))
          total_reward = 0
    env.close()