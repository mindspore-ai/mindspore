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

import random
import math
import numpy as np
import mindspore.nn as nn
from mindspore import Tensor
import mindspore.common.dtype as mstype
from src.dqn import DQN, WithLossCell


class Agent:
    """
    DQN Agent
    """
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
        self.policy_net = DQN(self.state_space_dim, 256, self.action_space_dim)
        self.target_net = DQN(self.state_space_dim, 256, self.action_space_dim)
        self.optimizer = nn.RMSProp(self.policy_net.trainable_params(), learning_rate=self.lr)
        loss_fn = nn.MSELoss()
        loss_q_net = WithLossCell(self.policy_net, loss_fn)
        self.policy_net_train = nn.TrainOneStepCell(loss_q_net, self.optimizer)
        self.policy_net_train.set_train(mode=True)
        self.buffer = []
        self.steps = 0

    def act(self, s0):
        """
        Agent choose action.
        """
        self.steps += 1
        epsi = self.epsi_low + (self.epsi_high - self.epsi_low) * (math.exp(-1.0 * self.steps / self.decay))
        if random.random() < epsi:
            a0 = random.randrange(self.action_space_dim)
        else:
            s0 = np.expand_dims(s0, axis=0)
            s0 = Tensor(s0, mstype.float32)
            a0 = self.policy_net(s0).asnumpy()
            a0 = np.argmax(a0)
        return a0

    def eval_act(self, s0):
        self.steps += 1
        s0 = np.expand_dims(s0, axis=0)
        s0 = Tensor(s0, mstype.float32)
        a0 = self.policy_net(s0).asnumpy()
        a0 = np.argmax(a0)
        return a0

    def put(self, *transition):
        if len(self.buffer) == self.capacity:
            self.buffer.pop(0)
        self.buffer.append(transition)

    def load_dict(self):
        for target_item, source_item in zip(self.target_net.parameters_dict(), self.policy_net.parameters_dict()):
            target_param = self.target_net.parameters_dict()[target_item]
            source_param = self.policy_net.parameters_dict()[source_item]
            target_param.set_data(source_param.data)

    def learn(self):
        """
        Agent learn from experience data.
        """
        if (len(self.buffer)) < self.batch_size:
            return

        samples = random.sample(self.buffer, self.batch_size)
        s0, a0, r1, s1 = zip(*samples)
        s1 = Tensor(s1, mstype.float32)
        s0 = Tensor(s0, mstype.float32)
        a0 = Tensor(np.expand_dims(a0, axis=1))
        next_state_values = self.target_net(s1).asnumpy()
        next_state_values = np.max(next_state_values, axis=1)

        y_true = r1 + self.gamma * next_state_values
        y_true = Tensor(np.expand_dims(y_true, axis=1), mstype.float32)
        self.policy_net_train(s0, a0, y_true)
