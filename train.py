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
    parser.add_argument("--time_steps", default=5000000)
    parser.add_argument("--buffer_size", default=1000000)
    parser.add_argument("--learning_starts", default=10000)
    parser.add_argument("--batch_size", default=64)
    parser.add_argument("--ent_coef", default=0.1)
    args = parser.parse_args()

    tasks.check_name(args.name)

    env_builder = importlib.import_module('{}.env_builder'.format(args.name))

    env = env_builder.build_env(enable_randomizer=True, version=args.version, enable_rendering=False)
    eval_env = env_builder.build_env(enable_randomizer=True, version=args.version, enable_rendering=False)

    eval_callback = EvalCallback(eval_env,
                                 best_model_save_path='./tasks/{}/best_model/v{}/'.format(args.name, args.version),
                                 log_path='./tasks/{}/best_model/v{}/'.format(args.name, args.version),
                                 eval_freq=1000,
                                 deterministic=True,
                                 render=False)
    policy_kwargs = dict(activation_fn=torch.nn.ReLU, net_arch=[256, 256])
    model = SAC('MlpPolicy',
                env,
                verbose=1,
                tensorboard_log="./tasks/{}/log/v{}/".format(args.name, args.version),
                policy_kwargs=policy_kwargs,
                buffer_size=args.buffer_size,
                batch_size=args.batch_size,
                learning_starts=args.learning_starts,
                ent_coef=args.ent_coef)
    model.learn(total_timesteps=args.time_steps, callback=eval_callback)