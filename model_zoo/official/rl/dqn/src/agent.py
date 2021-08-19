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
"""Agent of reinforcement learning network"""

import math
import numpy as np
import mindspore.nn as nn
import mindspore.common.dtype as mstype

from mindspore import Tensor, load_param_into_net
from mindspore.ops import operations as P
from src.dqn import DQN, WithLossCell

class Agent:
    """
    DQN Agent
    """
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
        self.policy_net = DQN(self.state_space_dim, self.hidden_size, self.action_space_dim)
        self.target_net = DQN(self.state_space_dim, self.hidden_size, self.action_space_dim)
        self.policy_net.training = True
        self.policy_net.requires_grad = True
        self.learn_step_counter = 0  # for target updating
        self.memory_counter = 0  # for storing memory
        self.memory = np.zeros((self.memory_capacity, self.state_space_dim * 2 + 2))  # initialize memory
        if self.dev == 'Ascend':
            self.optimizer = nn.RMSProp(self.policy_net.trainable_params(), learning_rate=self.lr)
        else:
            self.optimizer = nn.Adam(self.policy_net.trainable_params(), learning_rate=self.lr)
        self.loss_func = nn.MSELoss()
        self.loss_net = WithLossCell(self.policy_net, self.loss_func)
        self.train_net = nn.TrainOneStepCell(self.loss_net, self.optimizer)
        self.train_net.set_train()

        self.steps = 0

        self.cast = P.Cast()
        self.expand = P.ExpandDims()
        self.reshape = P.Reshape()
        self.argmax = P.ArgMaxWithValue(axis=1, keep_dims=True)
        self.gather = P.GatherD()

    def act(self, x):
        """
        get action
        """
        self.steps += 1
        if self.dev == 'GPU':
            epsilon = self.epsi_high
        else:
            epsilon = self.epsi_low + (self.epsi_high - self.epsi_low) * (math.exp(-1.0 * self.steps / self.decay))
        flag_com = False
        if np.random.uniform() < epsilon:
            x = Tensor(x, mstype.float32)
            x = self.expand(x, 0)
            actions_value = self.policy_net.construct(x)
            action = actions_value.asnumpy()
            action = np.argmax(action)
            flag_com = True
        else:  # random
            action = np.random.randint(0, self.action_space_dim)
            action = action if self.env_a_shape == 0 else self.reshape(action, self.env_a_shape)
        return action, flag_com

    def eval_act(self, x):
        """
        choose action in eval
        """
        x = Tensor(x, mstype.float32)
        x = self.expand(x, 0)
        actions_value = self.policy_net.construct(x)
        action = actions_value.asnumpy()
        action = np.argmax(action)
        return action

    def store_transition(self, s, a, r, s_):
        """
        store transition
        """
        transition = np.hstack((s, [a, r], s_))
        index = self.memory_counter % self.memory_capacity
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):
        """
        Agent learn from experience data.
        """

        if self.learn_step_counter % self.target_replace_iter == 0:
            load_param_into_net(self.target_net, self.policy_net.parameters_dict())

        self.learn_step_counter += 1

        sample_index = np.random.choice(self.memory_capacity, self.batch_size)

        b_memory = self.memory[sample_index, :]
        b_s = Tensor(b_memory[:, :self.state_space_dim], mstype.float32)
        b_a = Tensor(b_memory[:, self.state_space_dim:self.state_space_dim + 1].astype(int), mstype.int32)
        b_r = Tensor(b_memory[:, self.state_space_dim + 1:self.state_space_dim + 2], mstype.float32)
        b_s_ = Tensor(b_memory[:, -self.state_space_dim:], mstype.float32)

        q_next = self.target_net(b_s_)
        q_next_numpy = q_next.asnumpy()
        tem_ = Tensor(np.max(q_next_numpy, axis=1).reshape(-1, 1))
        q_target = b_r + self.gamma * tem_
        loss = self.train_net(b_s, q_target, b_a)
        return loss
