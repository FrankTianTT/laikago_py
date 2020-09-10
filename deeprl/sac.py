import torch
from torch import nn, optim, Tensor
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import gym
import matplotlib.pyplot as plt
import numpy as np
import os
from collections import namedtuple, deque
import random
import shutil


class ReplayBuffer():

    def __init__(self, buf_sz, batch_sz=1):
        self.memory = deque(maxlen=buf_sz)
        self.batch_sz = batch_sz
        self.experience = namedtuple("Experience", field_names=['state', 'action', 'reward', 'next_state'])

    def push(self, s, a, r, s_):
        experience = self.experience(s, a, r, s_)
        self.memory.append(experience)

    def sample(self, n=None):
        k = n if n != None else self.batch_sz
        return random.sample(self.memory, k=k)

    def __len__(self):
        return len(self.memory)


class QNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim=256):
        super(QNetwork, self).__init__()

        self.linear1 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        self.linear4 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.linear5 = nn.Linear(hidden_dim, hidden_dim)
        self.linear6 = nn.Linear(hidden_dim, 1)

    def forward(self, state, action):
        # cliped double Q
        xu = torch.cat([state, action], 1)

        x1 = F.relu(self.linear1(xu))
        x1 = F.relu(self.linear2(x1))
        x1 = self.linear3(x1)

        x2 = F.relu(self.linear4(xu))
        x2 = F.relu(self.linear5(x2))
        x2 = self.linear6(x2)

        return x1, x2


class GaussianPolicy(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim=256, log_std_min=-20, log_std_max=-2):
        super(GaussianPolicy, self).__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean_linear = nn.Linear(hidden_dim, num_actions)
        self.log_std_linear = nn.Linear(hidden_dim, num_actions)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=self.log_std_min, max=self.log_std_max)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        # Reparameterization trick
        eps = torch.randn_like(mean)
        u = eps * std + mean  # u ~ N(mean, std^2)
        action = torch.tanh(u).to(self.device)
        # Enforcing action bounds
        log_prob = normal.log_prob(u) - torch.log((1 - action.pow(2)) + 1e-8)
        log_prob = log_prob.sum(1, keepdim=True)

        return action, log_prob


class SAC:

    def __init__(
            self,
            env,
            batch_sz=256,
            start_step=10000,
            target_update_interval=1,
            evaluate_interval=1,
            n_updates=1,
            gamma=0.99,
            tau=0.005,
            lr=0.0003,
            alpha=0.2,
            auto_entropy_tuning=True,
            save_model_interval=100,
            is_load_model=False
    ):
        self.env = env
        self.n_state = env.observation_space.shape[0]
        self.n_action = env.action_space.shape[0]
        self.batch_sz = batch_sz
        self.start_step = start_step
        self.target_update_interval = target_update_interval
        self.evaluate_interval = evaluate_interval
        self.n_updates = n_updates
        self.gamma = gamma
        self.tau = tau
        self.lr = lr
        self.alpha = alpha
        self.auto_entropy_tuning = auto_entropy_tuning
        self.save_model_interval = save_model_interval

        self.global_epoch = 0
        self.global_step = 0
        self.reward_list = []
        self.memory = ReplayBuffer(1000000, self.batch_sz)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        LOG_PATH = './log'
        shutil.rmtree(LOG_PATH, ignore_errors=True)
        self.writer = SummaryWriter(LOG_PATH)

        self.critic = QNetwork(self.n_state, self.n_action).to(self.device)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=self.lr)
        self.critic_target = QNetwork(self.n_state, self.n_action).to(self.device)
        self.update_target(1.0)

        self.actor = GaussianPolicy(self.n_state, self.n_action, 256).to(self.device)
        self.actor_opt = optim.Adam(self.actor.parameters(), lr=self.lr)

        if is_load_model:
            self.load_model()
            print('load model')

        if self.auto_entropy_tuning:
            self.target_entropy = -torch.prod(Tensor(self.env.action_space.shape).to(self.device)).item()
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_opt = optim.Adam([self.log_alpha], lr=self.lr)

    def select_action(self, state):
        state = Tensor(state).to(self.device).unsqueeze(0)
        action, log_prob = self.actor.sample(state)
        return action.detach().cpu().numpy()[0]

    def sample_batch(self):
        '''
        sample a batch from replay buffer
        '''
        batch_data = self.memory.sample()
        batch_state = Tensor([i.state for i in batch_data]).to(self.device)
        batch_action = Tensor([i.action for i in batch_data]).to(self.device)
        batch_reward = Tensor([i.reward for i in batch_data]).to(self.device).unsqueeze(1)
        batch_next_state = Tensor([i.next_state for i in batch_data]).to(self.device)
        return batch_state, batch_action, batch_reward, batch_next_state

    def update_critic(self, batch):
        '''
        update critic network
        J = E[0.5 * (Q(s_t, a_t) - (r(s_t, a_t) + gamma * E[V(s_{t+1})]))^2]
        '''
        batch_state, batch_action, batch_reward, batch_next_state = batch
        with torch.no_grad():
            next_action, next_log_pi = self.actor.sample(batch_next_state)
            target_Q1, target_Q2 = self.critic_target(batch_next_state, next_action)
            # using Q to estimate V: V(s_t) = E[Q(s_t, a_t) - log(pi(a_t | s_t))]
            state_value = torch.min(target_Q1, target_Q2) - self.alpha * next_log_pi
            target_value = batch_reward + self.gamma * (state_value)
        current_Q1, current_Q2 = self.critic(batch_state, batch_action)
        Q1_loss = F.mse_loss(current_Q1, target_value)
        Q2_loss = F.mse_loss(current_Q2, target_value)
        Q_loss = Q1_loss + Q2_loss
        # self.writer.add_scalar('Q loss', Q_loss, self.global_epoch)

        self.critic_opt.zero_grad()
        Q_loss.backward()
        self.critic_opt.step()

    def update_actor(self, batch):
        '''
        update actor network
        J = E[alpha * log(pi(a_t | s_t)) - Q(s_t, a_t)]
        '''
        batch_state, batch_action, batch_reward, batch_next_state = batch
        action, log_pi = self.actor.sample(batch_state)
        target_Q1, target_Q2 = self.critic(batch_state, action)
        target_Q = torch.min(target_Q1, target_Q2)
        policy_loss = ((self.alpha * log_pi) - target_Q).mean()
        # self.writer.add_scalar('Policy loss', policy_loss, self.global_epoch)

        self.actor_opt.zero_grad()
        policy_loss.backward()
        self.actor_opt.step()

    def update_alpha(self, batch):
        '''
        update alpha
        J = - alpha * log(pi(a_t | s_t) - alpha * hat(H))
        '''
        batch_state, batch_action, batch_reward, batch_next_state = batch
        action, log_pi = self.actor.sample(batch_state)
        alpha = self.log_alpha.exp()
        alpha_loss = (- alpha * (log_pi + self.target_entropy).detach()).mean()
        # self.writer.add_scalar('Alpha loss', alpha_loss, self.global_epoch)
        self.alpha_opt.zero_grad()
        alpha_loss.backward()
        self.alpha_opt.step()

        self.alpha = self.log_alpha.exp().detach().cpu().numpy()[0]
        self.writer.add_scalar('Alpha', self.alpha, self.global_epoch)

    def update_target(self, tau):
        '''
        update target critic network
        target = (1 - tau) * target + tau * current_network
        '''
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(target_param.data * (1 - self.tau) + param.data * self.tau)

    def do_update(self):
        batch = self.sample_batch()

        self.update_critic(batch)
        self.update_actor(batch)
        if self.auto_entropy_tuning:
            self.update_alpha(batch)
        if self.global_step % self.target_update_interval == 0:
            self.update_target(self.tau)

    def train(self, epochs=100):
        for epoch in range(epochs):
            s = self.env.reset()

            while True:
                if self.global_step < self.start_step:
                    a = self.env.action_space.sample()
                else:
                    a = self.select_action(s)

                if len(self.memory) > self.batch_sz:
                    for _ in range(self.n_updates):
                        self.do_update()

                s_, r, done, _ = self.env.step(a)
                self.memory.push(s, a, r, s_)
                self.global_step += 1
                if done:
                    break
                s = s_

            self.global_epoch += 1

            if self.global_epoch % self.evaluate_interval == 0:
                eval_r = self.evaluate()
                self.writer.add_scalar('Total reward', eval_r, self.global_epoch)

            if self.global_epoch % self.save_model_interval == 0:
                self.save_model()

            print('Finish epoch %d' % self.global_epoch)

    def load_model(self, epoch=100):
        self.critic.load_state_dict(torch.load('./model/critic_' + str(epoch) + '.pth'))
        self.critic_target.load_state_dict(torch.load('./model/critic_target_' + str(epoch) + '.pth'))
        self.actor.load_state_dict(torch.load('./model/actor_' + str(epoch) + '.pth'))

    def save_model(self):
        if not os.path.exists('model'):
            os.mkdir('model')
        torch.save(self.critic.state_dict(), './model/critic_' + str(self.global_epoch) + '.pth')
        torch.save(self.critic_target.state_dict(), './model/critic_target_' + str(self.global_epoch) + '.pth')
        torch.save(self.actor.state_dict(), './model/actor_' + str(self.global_epoch) + '.pth')

    def evaluate(self, n=1):
        tot_reward = 0
        for _ in range(n):
            s = self.env.reset()

            while True:
                a = self.select_action(s)
                s_, r, done, _ = self.env.step(a)
                tot_reward += r
                if done:
                    break
                s = s_

        tot_reward /= n
        self.reward_list.append(tot_reward)
        return tot_reward

    def plot_reward(self, is_show=True):
        plt.plot(self.reward_list)
        plt.title('SAC')
        plt.xlabel('epoch')
        plt.ylabel('reward')
        plt.savefig('SAC.png')
        if is_show:
            plt.show()


if __name__ == '__main__':
    env = gym.make('HalfCheetah-v2')
    # env = gym.make('Hopper-v2')
    sac = SAC(env)
    sac.train(200)
    env.close()
