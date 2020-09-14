from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback
import standup.standup_env_builder as env_builder
import numpy as np

a=np.array([[1,2],[3,4]])
print(np.sum(a**2))

print(a**2)

print(np.sum(a**2, axis=1))

print(np.sqrt(np.sum(a**2, axis=1)))