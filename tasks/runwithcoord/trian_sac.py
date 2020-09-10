from deeprl.sac import SAC
import runwithcoord.runwithcoord_env_builder as env_builder
import os

TASK_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(TASK_DIR, 'log')
SAVE_DIR = os.path.join(TASK_DIR, 'save')
TASK_NAME = "runwithcoord"
HID_SIZE = 256

env = env_builder.build_env(enable_randomizer=True, enable_rendering=False)
sac = SAC(env, save_name=TASK_NAME, hidden_size=HID_SIZE, log_path=LOG_DIR, save_path=SAVE_DIR)
sac.train(200)
sac.play()

env.close()