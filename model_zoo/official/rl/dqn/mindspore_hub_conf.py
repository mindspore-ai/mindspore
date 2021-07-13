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
"""hub config."""
import gym
from src.agent import Agent
from src.config import config_dqn as cfg

def dqn_net(*args, **kwargs):
    agent = Agent(*args, **kwargs)
    return agent.policy_net


def create_network(name, *args, **kwargs):
    """
    create dqn network
    """
    if name == "dqn":
        env = gym.make('CartPole-v1')
        cfg.state_space_dim = env.observation_space.shape[0]
        cfg.action_space_dim = env.action_space.n
        return dqn_net(**cfg)
    raise NotImplementedError(f"{name} is not implemented in the repo")
