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
"""Evaluation for DQN"""

import argparse
import gym
from mindspore import context
from mindspore.common import set_seed
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from src.config import config_dqn as cfg
from src.agent import Agent

parser = argparse.ArgumentParser(description='MindSpore dqn Example')
parser.add_argument('--device_target', type=str, default="Ascend", choices=['Ascend', 'GPU'],
                    help='device where the code will be implemented (default: Ascend)')
parser.add_argument('--ckpt_path', type=str, default=None, help='if is test, must provide\
                    path where the trained ckpt file')
args = parser.parse_args()
set_seed(1)


if __name__ == "__main__":
    context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target)
    env = gym.make('CartPole-v1')
    cfg.state_space_dim = env.observation_space.shape[0]
    cfg.action_space_dim = env.action_space.n
    agent = Agent(**cfg)

    # load checkpoint
    if args.ckpt_path:
        param_dict = load_checkpoint(args.ckpt_path)
        not_load_param = load_param_into_net(agent.policy_net, param_dict)
        if not_load_param:
            raise ValueError("Load param into net fail!")

    score = 0
    agent.load_dict()
    for episode in range(50):
        s0 = env.reset()
        total_reward = 1
        while True:
            a0 = agent.eval_act(s0)
            s1, r1, done, _ = env.step(a0)

            if done:
                r1 = -1

            if done:
                break

            total_reward += r1
            s0 = s1
        score += total_reward
        print("episode", episode, "total_reward", total_reward)
    print("mean_reward", score/50)
