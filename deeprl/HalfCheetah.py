from deeprl.sac import SAC
import gym

env = gym.make('HalfCheetah-v3')
sac = SAC(env)
sac.train(200)
env.close()
