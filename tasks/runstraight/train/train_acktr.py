#!/usr/bin/env python3
# by frank tian on 7.13.2020
################################
#change these when changing task
import runstraight.runstraight_env_builder as env_builder
TASK_NAME = "runstraight"
FILE_NAME = ''
################################

import os
import math
import ptan
import time
from tensorboardX import SummaryWriter
from algorithms.utilities import common,kfac
from network_model import acktr_model as model
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F

GAMMA = 0.99
REWARD_STEPS = 5
BATCH_SIZE = 32
LEARNING_RATE_ACTOR = 1e-3
LEARNING_RATE_CRITIC = 1e-3
ENTROPY_BETA = 1e-3
ENVS_COUNT = 16

TEST_ITERS = 100000
DEVICE = 'cuda'

TASK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOAD_FILE = os.path.join(TASK_DIR, 'saves', "acktr-"+TASK_NAME, FILE_NAME)

def test_net(net, env, count=10, device="cpu"):
    rewards = 0.0
    steps = 0
    for _ in range(count):
        obs = env.reset()
        while True:
            obs_v = ptan.agent.float32_preprocessor([obs]).to(device)
            mu_v = net(obs_v)[0]
            action = mu_v.squeeze(dim=0).data.cpu().numpy()
            action = np.clip(action, -1, 1)
            obs, reward, done, _ = env.step(action)
            rewards += reward
            steps += 1
            if done:
                break
    return rewards / count, steps / count


def calc_logprob(mu_v, logstd_v, actions_v):
    p1 = - ((mu_v - actions_v) ** 2) / (2*torch.exp(logstd_v).clamp(min=1e-3))
    p2 = - torch.log(torch.sqrt(2 * math.pi * torch.exp(logstd_v)))
    return p1 + p2


if __name__ == "__main__":
    device = torch.device(DEVICE)

    save_path = os.path.join(TASK_DIR, "saves", "acktr-" + TASK_NAME)
    os.makedirs(save_path, exist_ok=True)


    envs = [env_builder.build_env(enable_randomizer=True, enable_rendering=False) for _ in range(ENVS_COUNT)]
    test_env = env_builder.build_env(enable_randomizer=False, enable_rendering=False)

    act_net = model.ACKTRActor(envs[0].observation_space.shape[0], envs[0].action_space.shape[0]).to(device)
    crt_net = model.ACKTRCritic(envs[0].observation_space.shape[0]).to(device)
    print(act_net)
    print(crt_net)

    if LOAD_FILE is not '':
        act_net.load_state_dict(torch.load(LOAD_FILE))

    writer = SummaryWriter(comment="-acktr-" + TASK_NAME)
    agent = model.AgentACKTR(act_net, device=device)
    exp_source = ptan.experience.ExperienceSourceFirstLast(envs, agent, GAMMA, steps_count=REWARD_STEPS)

    opt_act = kfac.KFACOptimizer(act_net, lr=LEARNING_RATE_ACTOR)
    opt_crt = optim.Adam(crt_net.parameters(), lr=LEARNING_RATE_CRITIC)

    batch = []
    best_reward = None
    with ptan.common.utils.RewardTracker(writer) as tracker:
        with ptan.common.utils.TBMeanTracker(writer, batch_size=100) as tb_tracker:
            for step_idx, exp in enumerate(exp_source):
                rewards_steps = exp_source.pop_rewards_steps()
                if rewards_steps:
                    rewards, steps = zip(*rewards_steps)
                    tb_tracker.track("episode_steps", np.mean(steps), step_idx)
                    tracker.reward(np.mean(rewards), step_idx)

                if step_idx % TEST_ITERS == 0:
                    ts = time.time()
                    rewards, steps = test_net(act_net, test_env, device=device)
                    print("Test done in %.2f sec, reward %.3f, steps %d" % (
                        time.time() - ts, rewards, steps))
                    writer.add_scalar("test_reward", rewards, step_idx)
                    writer.add_scalar("test_steps", steps, step_idx)
                    if best_reward is None or best_reward < rewards:
                        if best_reward is not None:
                            print("Best reward updated: %.3f -> %.3f" % (best_reward, rewards))
                            name = "best_%+.3f_%d.dat" % (rewards, step_idx)
                            fname = os.path.join(save_path, name)
                            torch.save(act_net.state_dict(), fname)
                        best_reward = rewards

                batch.append(exp)
                if len(batch) < BATCH_SIZE:
                    continue

                states_v, actions_v, vals_ref_v = \
                    common.unpack_batch_acktr(batch, crt_net, last_val_gamma=GAMMA ** REWARD_STEPS, device=device)
                batch.clear()

                opt_crt.zero_grad()
                value_v = crt_net(states_v)
                loss_value_v = F.mse_loss(value_v.squeeze(-1), vals_ref_v)
                loss_value_v.backward()
                opt_crt.step()

                mu_v = act_net(states_v)
                log_prob_v = calc_logprob(mu_v, act_net.logstd, actions_v)
                if opt_act.steps % opt_act.Ts == 0:
                    opt_act.zero_grad()
                    pg_fisher_loss = -log_prob_v.mean()
                    opt_act.acc_stats = True
                    pg_fisher_loss.backward(retain_graph=True)
                    opt_act.acc_stats = False

                opt_act.zero_grad()
                adv_v = vals_ref_v.unsqueeze(dim=-1) - value_v.detach()
                loss_policy_v = -(adv_v * log_prob_v).mean()
                entropy_loss_v = ENTROPY_BETA * (-(torch.log(2 * math.pi * torch.exp(act_net.logstd)) + 1) / 2).mean()
                loss_v = loss_policy_v + entropy_loss_v
                loss_v.backward()
                opt_act.step()

                tb_tracker.track("advantage", adv_v, step_idx)
                tb_tracker.track("values", value_v, step_idx)
                tb_tracker.track("batch_rewards", vals_ref_v, step_idx)
                tb_tracker.track("loss_entropy", entropy_loss_v, step_idx)
                tb_tracker.track("loss_policy", loss_policy_v, step_idx)
                tb_tracker.track("loss_value", loss_value_v, step_idx)
                tb_tracker.track("loss_total", loss_v, step_idx)
