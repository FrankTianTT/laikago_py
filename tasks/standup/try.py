from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback
import standup.standup_env_builder as env_builder
import numpy as np
VERSION = 1

env = env_builder.build_env(enable_randomizer=True, version=VERSION, enable_rendering=True)


target_entropy = -np.prod(env.action_space.shape).astype(np.float32)

print(target_entropy)

print(env.action_space.shape)