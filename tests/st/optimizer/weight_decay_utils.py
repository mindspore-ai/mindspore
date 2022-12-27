# Copyright 2022 Huawei Technologies Co., Ltd
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

import numpy as np
import mindspore.nn as nn
import mindspore.common.dtype as mstype
from mindspore.nn import TrainOneStepCell, WithLossCell
from mindspore import Tensor
from mindspore.nn import Dense, ReLU
from mindspore.ops import operations as P


class WeightDecaySchdule(nn.Cell):
    def __init__(self):
        super(WeightDecaySchdule, self).__init__()
        self.weight_decay_list = Tensor([0.001, 0.001, 0.1], mstype.float32)

    def construct(self, global_step):
        return self.weight_decay_list[global_step]


class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.batch_size = 1
        self.reshape = P.Reshape()
        self.relu = ReLU()
        weight1 = Tensor(np.ones([10, 16]).astype(np.float32) * 0.01)
        weight2 = Tensor(np.ones([10, 10]).astype(np.float32) * 0.02)
        weight3 = Tensor(np.ones([10, 10]).astype(np.float32) * 0.03)
        bias1 = Tensor(np.zeros(10).astype(np.float32))
        bias2 = Tensor(np.ones(10).astype(np.float32))
        bias3 = Tensor(np.ones(10).astype(np.float32))
        self.fc1 = Dense(16, 10, weight_init=weight1, bias_init=bias1)
        self.fc2 = Dense(10, 10, weight_init=weight2, bias_init=bias2)
        self.fc3 = Dense(10, 10, weight_init=weight3, bias_init=bias3)

    def construct(self, input_x):
        output = self.reshape(input_x, (self.batch_size, -1))
        output = self.fc1(output)
        output = self.relu(output)
        output = self.fc2(output)
        output = self.relu(output)
        output = self.fc3(output)
        return output


def dynamic_weight_decay_cmp(net1, net2, optimizer1, optimizer2):
    epoch = 3
    criterion = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
    net_with_criterion1 = WithLossCell(net1, criterion)
    net_with_criterion2 = WithLossCell(net2, criterion)
    train_network1 = TrainOneStepCell(net_with_criterion1, optimizer1)
    train_network2 = TrainOneStepCell(net_with_criterion2, optimizer2)
    train_network1.set_train()
    train_network2.set_train()

    for _ in range(epoch):
        data = Tensor(np.arange(0, 16).reshape(1, 1, 4, 4).astype(np.float32) * 0.01)
        label = Tensor(np.array([0]).astype(np.int32))
        loss1 = train_network1(data, label)
        loss2 = train_network2(data, label)
        assert abs(loss1.asnumpy() - loss2.asnumpy()) < 1.e-3

    data = Tensor(np.arange(0, 16).reshape(1, 1, 4, 4).astype(np.float32) * 0.01)
    label = Tensor(np.array([0]).astype(np.int32))
    loss1 = net_with_criterion1(data, label)
    loss2 = net_with_criterion2(data, label)
    assert abs(loss1.asnumpy() - loss2.asnumpy()) < 1.e-3
