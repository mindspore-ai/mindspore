# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Train DQN and get checkpoint files."""

import os
import argparse
import timeit
import gym
import numpy as np
from mindspore import context
from mindspore.common import set_seed
from mindspore.train.serialization import save_checkpoint
from src.config import config_dqn as cfg
from src.config_gpu import config_dqn as cfg_gpu
from src.agent import Agent

parser = argparse.ArgumentParser(description='MindSpore dqn Example')
parser.add_argument('--device_target', type=str, default='Ascend', choices=['Ascend', 'GPU'],
                    help='device where the code will be implemented (default: Ascend)')
parser.add_argument('--ckpt_path', type=str, default="./ckpt", help='if is test, must provide\
                    path where the trained ckpt file')
args = parser.parse_args()
set_seed(1)

def save_ckpt(path, model, ckpt_name):
    """
    save ckpt file
    """
    if not os.path.exists(path):
        os.makedirs(path)

    ckpt_name = path + ckpt_name
    save_checkpoint(model, ckpt_name)


if __name__ == "__main__":
    context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target)
    if args.device_target == 'GPU':
        cfg = cfg_gpu

    env = gym.make(cfg.game)
    env = env.unwrapped
    cfg.state_space_dim = env.observation_space.shape[0]
    cfg.action_space_dim = env.action_space.n
    cfg.env_a_shape = 0 if isinstance(env.action_space.sample(),
                                      int) else env.action_space.sample().shape
    agent = Agent(**cfg)

    rewards = []
    count = 0
    times = []

    print('\nCollecting experience...')
    for episode in range(400):
        s = env.reset()
        total_reward = 1
        ep_r = 0
        while True:
            start = timeit.default_timer()
            a, flag = agent.act(s)
            s_, r, done_, _ = env.step(a)

            # modify the reward
            x, x_dot, theta, theta_dot = s_
            r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
            r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
            r = r1 + r2

            if flag:
                end = timeit.default_timer()
                differences = end - start
                times.append(differences)
                count += 1
                    # pass

            agent.store_transition(s, a, r, s_)
            ep_r += r
            if agent.memory_counter > cfg.memory_capacity:
                _ = agent.learn()
                if done_:
                    print("episode", episode, "total_reward", round(ep_r, 2))
                    rewards.append(round(ep_r, 2))
            if done_:
                break
            s = s_
    env.close()
    save_ckpt(os.path.realpath(args.ckpt_path), agent.policy_net, "/dqn.ckpt")
    rewards_numpy = np.array(rewards)

    times.remove(min(times))
    times.remove(max(times))
    times_numpy = np.array(times)

    print(rewards_numpy.mean(), times_numpy.mean())
