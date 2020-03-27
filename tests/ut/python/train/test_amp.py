# Copyright 2020 Huawei Technologies Co., Ltd
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
""" auto mixed precision """
import numpy as np
from mindspore import amp
from mindspore import nn
from mindspore import Tensor
from mindspore.common import dtype as mstype
import mindspore.context as context
from mindspore.model_zoo.resnet import resnet50


def setup_module(module):
    context.set_context(mode=context.GRAPH_MODE)


class Net(nn.Cell):
    def __init__(self, in_features, out_features):
        super(Net, self).__init__()
        self.dense = nn.Dense(in_features, out_features)
        self.loss = nn.MSELoss()

    def construct(self, input_x, label):
        output = self.dense(input_x)
        loss = self.loss(output, label)
        return loss


class NetNoLoss(nn.Cell):
    def __init__(self, in_features, out_features):
        super(NetNoLoss, self).__init__()
        self.dense = nn.Dense(in_features, out_features)

    def construct(self, input_x):
        return self.dense(input_x)


def test_amp_o0():
    inputs = Tensor(np.ones([16, 16]).astype(np.float32))
    label = Tensor(np.zeros([16, 16]).astype(np.float32))
    net = Net(16, 16)

    optimizer = nn.Momentum(net.trainable_params(), learning_rate=0.1, momentum=0.9)
    train_network = amp.build_train_network(net, optimizer, level="O0")
    output = train_network(inputs, label)


def test_amp_o2():
    inputs = Tensor(np.ones([16, 16]).astype(np.float32))
    label = Tensor(np.zeros([16, 16]).astype(np.float32))
    net = Net(16, 16)

    optimizer = nn.Momentum(net.trainable_params(), learning_rate=0.1, momentum=0.9)
    train_network = amp.build_train_network(net, optimizer, level="O2")
    output = train_network(inputs, label)

def test_amp_o2_loss():
    inputs = Tensor(np.ones([16, 16]).astype(np.float32))
    label = Tensor(np.zeros([16, 16]).astype(np.float32))
    net = NetNoLoss(16, 16)
    loss = nn.MSELoss()
    optimizer = nn.Momentum(net.trainable_params(), learning_rate=0.1, momentum=0.9)
    train_network = amp.build_train_network(net, optimizer, loss, level="O2")
    output = train_network(inputs, label)

def test_amp_resnet50_loss():
    inputs = Tensor(np.ones([2, 3, 224, 224]).astype(np.float32))
    label = Tensor(np.zeros([2, 10]).astype(np.float32))
    net = resnet50()
    loss = nn.SoftmaxCrossEntropyWithLogits(reduction='mean')
    optimizer = nn.Momentum(net.trainable_params(), learning_rate=0.1, momentum=0.9)
    train_network = amp.build_train_network(net, optimizer, loss, level="O2")
    train_network(inputs, label)

def test_amp_o0_loss():
    inputs = Tensor(np.ones([16, 16]).astype(np.float32))
    label = Tensor(np.zeros([16, 16]).astype(np.float32))
    net = NetNoLoss(16, 16)
    loss = nn.MSELoss()
    optimizer = nn.Momentum(net.trainable_params(), learning_rate=0.1, momentum=0.9)
    train_network = amp.build_train_network(net, optimizer, loss)
    output = train_network(inputs, label)
