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

def print_sensor(sensors):
    for s in sensors:
        print(s.get_name())
        print(s.get_observation())
    print()
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--name", required=True, help="Name of task")
    parser.add_argument("-v", "--version", required=True,  help="Version of task")
    parser.add_argument("-m", "--mode", default='train', type=str)
    args = parser.parse_args()

    name = args.name
    version = args.version
    mode = args.mode

    env_builder = importlib.import_module('{}.env_builder'.format(name))

    env = env_builder.build_env(enable_randomizer=True, version=version, enable_rendering=True, mode=mode)
    model = SAC.load('./tasks/{}/best_model/v{}/best_model.zip'.format(name, version))
    total_reward = 0
    obs = env.reset()

    sensors = env.all_sensors()


    for i in range(10000):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        # print_sensor(sensors)

        total_reward += reward
        if done:
          obs = env.reset()
          print('Test reward is {:.3f}.'.format(total_reward))
          total_reward = 0
    env.close()