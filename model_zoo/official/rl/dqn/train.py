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
import gym
from mindspore import context
from mindspore.common import set_seed
from mindspore.train.serialization import save_checkpoint
from src.config import config_dqn as cfg
from src.agent import Agent

parser = argparse.ArgumentParser(description='MindSpore dqn Example')
parser.add_argument('--device_target', type=str, default="Ascend", choices=['Ascend', 'GPU'],
                    help='device where the code will be implemented (default: Ascend)')
parser.add_argument('--ckpt_path', type=str, default="./ckpt", help='if is test, must provide\
                    path where the trained ckpt file')
args = parser.parse_args()
set_seed(1)


if __name__ == "__main__":
    context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target)
    env = gym.make('CartPole-v1')
    cfg.state_space_dim = env.observation_space.shape[0]
    cfg.action_space_dim = env.action_space.n
    agent = Agent(**cfg)
    agent.load_dict()

    for episode in range(300):
        s0 = env.reset()
        total_reward = 1
        while True:
            a0 = agent.act(s0)
            s1, r1, done, _ = env.step(a0)

            if done:
                r1 = -1

            agent.put(s0, a0, r1, s1)

            if done:
                break

            total_reward += r1
            s0 = s1
            agent.learn()
        agent.load_dict()
        print("episode", episode, "total_reward", total_reward)

    path = os.path.realpath(args.ckpt_path)
    if not os.path.exists(path):
        os.makedirs(path)

    ckpt_name = path + "/dqn.ckpt"
    save_checkpoint(agent.policy_net, ckpt_name)
