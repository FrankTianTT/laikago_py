import numpy as np
import ptan
import torch.nn as nn
import torch

HID_SIZE = 128


class PPOActor(nn.Module):
    def __init__(self, obs_size, act_size, hid_size=HID_SIZE):
        super(PPOActor, self).__init__()

        self.base = nn.Sequential(
            nn.Linear(obs_size, hid_size),
            nn.ReLU(),
            nn.Linear(hid_size, hid_size),
            nn.ReLU(),
        )
        self.mu = nn.Sequential(
            nn.Linear(hid_size, act_size),
        )
        self.var = nn.Sequential(
            nn.Linear(hid_size, act_size),
            nn.Softplus(),
        )

    def forward(self, x):
        base_out = self.base(x)
        return self.mu(base_out), self.var(base_out)

class PPOCritic(nn.Module):
    def __init__(self, obs_size, hid_size=128):
        super(PPOCritic, self).__init__()

        self.value = nn.Sequential(
            nn.Linear(obs_size, hid_size),
            nn.ReLU(),
            nn.Linear(hid_size, hid_size),
            nn.ReLU(),
            nn.Linear(hid_size, 1),
        )

    def forward(self, x):
        return self.value(x)

class AgentPPO(ptan.agent.BaseAgent):
    def __init__(self, net, device="cpu"):
        self.net = net
        self.device = device

    def __call__(self, states, agent_states):
        states_v = ptan.agent.float32_preprocessor(states)
        states_v = states_v.to(self.device)

        mu_v, var_v = self.net(states_v)
        mu = mu_v.data.cpu().numpy()
        sigma = torch.sqrt(var_v).data.cpu().numpy()
        actions = np.random.normal(mu, sigma)
        actions = np.clip(actions, -1, 1)
        return actions, agent_states