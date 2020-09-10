from deeprl.sac import SAC
import gym

env = gym.make('Walker2d-v3')

sac = SAC(env)
sac.train(200)
env.close()

def show_env():
    env = gym.make('Walker2d-v3')
    env.reset()
    for i in range(10000):
        action = env.action_space.sample()
        env.step(action)
        env.render()