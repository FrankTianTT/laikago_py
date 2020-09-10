from deeprl.sac import SAC
import gym
import os

TASK_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(TASK_DIR, 'log')
SAVE_DIR = os.path.join(TASK_DIR, 'save')

env = gym.make('Hopper-v3')
sac = SAC(env, save_name='Hopper-v3', log_path=LOG_DIR, save_path=SAVE_DIR)
sac.train(200)
sac.play()

env.close()
