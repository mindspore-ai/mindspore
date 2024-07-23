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
""" test_training """
import numpy as np

import mindspore.nn as nn
from mindspore import context
from mindspore.common.tensor import Tensor
from mindspore.nn import WithGradCell, WithLossCell
from mindspore.ops import operations as P
from tests.mark_utils import arg_mark


def setup_module(module):
    context.set_context(mode=context.PYNATIVE_MODE)


class LeNet5(nn.Cell):
    """ LeNet5 definition """

    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, pad_mode='valid')
        self.conv2 = nn.Conv2d(6, 16, 5, pad_mode='valid')
        self.fc1 = nn.Dense(16 * 5 * 5, 120)
        self.fc2 = nn.Dense(120, 84)
        self.fc3 = nn.Dense(84, 10)
        self.relu = nn.ReLU()
        self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = P.Flatten()

    def construct(self, x):
        x = self.max_pool2d(self.relu(self.conv1(x)))
        x = self.max_pool2d(self.relu(self.conv2(x)))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_loss_cell_wrapper():
    """ test_loss_cell_wrapper """
    data = Tensor(np.ones([1, 1, 32, 32]).astype(np.float32) * 0.01)
    label = Tensor(np.ones([1, 10]).astype(np.float32))
    net = LeNet5()
    loss_fn = nn.SoftmaxCrossEntropyWithLogits()
    loss_net = WithLossCell(net, loss_fn)
    loss_out = loss_net(data, label)
    assert loss_out.asnumpy().dtype == 'float32' or loss_out.asnumpy().dtype == 'float64'


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_grad_cell_wrapper():
    """ test_grad_cell_wrapper """
    data = Tensor(np.ones([1, 1, 32, 32]).astype(np.float32) * 0.01)
    label = Tensor(np.ones([1, 10]).astype(np.float32))
    dout = Tensor(np.ones([1]).astype(np.float32))
    net = LeNet5()
    loss_fn = nn.SoftmaxCrossEntropyWithLogits()
    grad_net = WithGradCell(net, loss_fn, dout)
    gradients = grad_net(data, label)
    assert isinstance(gradients[0].asnumpy()[0][0][0][0], (np.float32, np.float64))
