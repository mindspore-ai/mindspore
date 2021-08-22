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

from src.config_gpu import config_dqn as cfg_gpu
from src.config import config_dqn as cfg
from src.agent import Agent

parser = argparse.ArgumentParser(description='MindSpore dqn Example')
parser.add_argument('--device_target', type=str, default='Ascend', choices=['Ascend', 'GPU'],
                    help='device where the code will be implemented (default: Ascend)')
parser.add_argument('--ckpt_path', type=str, default=None, help='if is test, must provide\
                    path where the trained ckpt file')
args = parser.parse_args()
set_seed(1)

if __name__ == "__main__":
    context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target)
    if args.device_target == 'GPU':
        cfg = cfg_gpu

    env = gym.make(cfg.game)
    env = env.unwrapped
    cfg.state_space_dim = env.observation_space.shape[0]
    cfg.action_space_dim = env.action_space.n
    cfg.env_a_shape = 0 if isinstance(env.action_space.sample(),
                                      int) else env.action_space.sample().shape  # to confirm the shape
    agent = Agent(**cfg)

    # load checkpoint
    if args.ckpt_path:
        param_dict = load_checkpoint(args.ckpt_path)
        not_load_param = load_param_into_net(agent.policy_net, param_dict)
        if not_load_param:
            raise ValueError("Load param into net fail!")

    score = 0
    for episode in range(cfg.EPOCH):
        s = env.reset()
        ep_r = 0
        while True:
            a, flag = agent.act(s)
            s_, r, done, _ = env.step(a)

            # modify the reward
            x, x_dot, theta, theta_dot = s_
            r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
            r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
            r = r1 + r2

            ep_r += r
            if done:
                break
            s = s_

        score += ep_r
        print("episode", episode, "total_reward", ep_r)
    print("mean_reward", score / cfg.EPOCH)
    