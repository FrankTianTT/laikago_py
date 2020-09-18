import sys
import os
import json
from os.path import abspath, join, dirname
sys.path.insert(0, join(abspath(dirname(__file__)), 'envs'))
sys.path.insert(0, join(abspath(dirname(__file__)), 'tasks'))
import importlib
import argparse
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback
import tasks
import torch


def get_file_no(file_path, algo_name='SAC'):
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    files = os.listdir(file_path)
    nums = []
    for f in files:
        if os.path.isdir(file_path + f):
            name, no = f.split('_')
            if name == algo_name:
                nums.append(int(no))
    if len(nums) == 0:
        file_no = '{}_{}'.format(algo_name, 1)
        os.makedirs(file_path+ file_no)
        return file_no
    else:
        file_no = '{}_{}'.format(algo_name, str(max(nums) + 1))
        os.makedirs(file_path + file_no)
        return file_no

def save_parameter(file_path, args):
    assert os.path.isdir(file_path)
    jsObj = json.dumps(args.__dict__)
    fileObject = open(join(file_path, 'parameter.json'), 'w')
    fileObject.write(jsObj)
    fileObject.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--name", required=True, help="Name of task")
    parser.add_argument("-v", "--version", required=True,  help="Version of task")
    parser.add_argument("-l", "--load_from_best", default=False, type=bool)
    parser.add_argument("--time_steps", default=5000000)
    parser.add_argument("--buffer_size", default=1000000)
    parser.add_argument("--learning_starts", default=10000)
    parser.add_argument("--batch_size", default=64)
    parser.add_argument("--ent_coef", default=0.1)
    parser.add_argument("--net_arch", default=[256, 256], nargs='+', type=int)

    args = parser.parse_args()

    name = args.name
    version = args.version
    net_arch = args.net_arch
    save_model_path = './tasks/{}/log_model/v{}/'.format(name, version)
    save_model_path = save_model_path + get_file_no(save_model_path)
    load_model_path = './tasks/{}/log_model/v{}'.format(name, version)
    load_model_path = load_model_path + get_file_no(load_model_path) + '/best_model.zip'

    buffer_size = args.buffer_size
    batch_size = args.batch_size
    learning_starts = args.learning_starts
    ent_coef = args.ent_coef
    time_steps = args.time_steps

    tasks.check_name(name)
    save_parameter(save_model_path, args)

    env_builder = importlib.import_module('{}.env_builder'.format(name))
    env = env_builder.build_env(enable_randomizer=True, version=version, enable_rendering=False)
    eval_env = env_builder.build_env(enable_randomizer=True, version=version, enable_rendering=False)

    eval_callback = EvalCallback(eval_env,
                                 best_model_save_path=save_model_path,
                                 log_path=save_model_path,
                                 eval_freq=1000,
                                 deterministic=True,
                                 render=False)
    policy_kwargs = dict(activation_fn=torch.nn.ReLU, net_arch=net_arch)
    model = SAC('MlpPolicy',
                env,
                verbose=1,
                tensorboard_log=save_model_path,
                policy_kwargs=policy_kwargs,
                buffer_size=buffer_size,
                batch_size=batch_size,
                learning_starts=learning_starts,
                ent_coef=ent_coef)
    if args.load_from_best:
        model = SAC.load(load_model_path)
        model.set_env(env)
    model.learn(total_timesteps=time_steps, callback=eval_callback)