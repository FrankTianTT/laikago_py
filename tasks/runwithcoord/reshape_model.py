import os
from model import ppo_model as model
import runwithcoord.runwithcoord_env_builder as env_builder
import torch

TASK_NAME = "runwithcoord"
FILE_NAME = 'run_ppo_256.dat'
HID_SIZE = 256

DEVICE = 'cuda'
#
device = torch.device(DEVICE)
#
env = env_builder.build_env(enable_randomizer=True, enable_rendering=False)
TASK_DIR = os.path.dirname(os.path.abspath(__file__))
LOAD_FILE = os.path.join(TASK_DIR, 'saves', "ppo-"+TASK_NAME, FILE_NAME)
#
act_net = model.PPOActor(env.observation_space.shape[0], env.action_space.shape[0], hid_size=HID_SIZE).to(device)
#
#
# if FILE_NAME is not '':
#     act_net.load_state_dict(torch.load(LOAD_FILE))

pretrain_model = torch.load(LOAD_FILE)
x = torch.randn(256, 15).to(device)
pretrain_model['mu.0.weight'] = torch.cat((pretrain_model['mu.0.weight'], x), 1)

act_net.load_state_dict(pretrain_model)