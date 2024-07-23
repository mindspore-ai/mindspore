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
"""
This test is used to monitor some features of MindArmour.
"""
import numpy as np

import mindspore.nn as nn
from mindspore import context, Tensor
from mindspore.nn import Cell, WithLossCell, TrainOneStepCell
from mindspore.nn.optim.momentum import Momentum
from mindspore.common.initializer import TruncatedNormal
from mindspore.ops.composite import GradOperation

from tests.mark_utils import arg_mark


def weight_variable():
    """weight initial"""
    return TruncatedNormal(0.02)


def conv(in_channels, out_channels, kernel_size, stride=1, padding=0):
    """weight initial for conv layer"""
    weight = weight_variable()
    return nn.Conv2d(in_channels, out_channels,
                     kernel_size=kernel_size, stride=stride, padding=padding,
                     weight_init=weight, has_bias=False, pad_mode="valid")


def fc_with_initialize(input_channels, out_channels):
    """weight initial for fc layer"""
    weight = weight_variable()
    bias = weight_variable()
    return nn.Dense(input_channels, out_channels, weight, bias)


class LeNet(nn.Cell):
    """
    Lenet network
    Args:
        num_class (int): Num classes, Default: 10.
    Returns:
        Tensor, output tensor
    Examples:
        >>> LeNet(num_class=10)
    """

    def __init__(self, num_class=10):
        super(LeNet, self).__init__()
        self.conv1 = conv(1, 6, 5)
        self.conv2 = conv(6, 16, 5)
        self.fc1 = fc_with_initialize(16 * 5 * 5, 120)
        self.fc2 = fc_with_initialize(120, 84)
        self.fc3 = fc_with_initialize(84, 10)
        self.relu = nn.ReLU()
        self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()

    def construct(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.max_pool2d(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.max_pool2d(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x


class GradWithSens(Cell):
    def __init__(self, network):
        super(GradWithSens, self).__init__()
        self.grad = GradOperation(get_all=False,
                                  sens_param=True)
        self.network = network

    def construct(self, inputs, weight):
        gout = self.grad(self.network)(inputs, weight)
        return gout


class GradWrapWithLoss(Cell):
    def __init__(self, network):
        super(GradWrapWithLoss, self).__init__()
        self._grad_all = GradOperation(get_all=True,
                                       sens_param=False)
        self._network = network

    def construct(self, inputs, labels):
        gout = self._grad_all(self._network)(inputs, labels)
        return gout[0]


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_grad_values_and_infer_shape():
    context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")
    inputs_np = np.random.rand(32, 1, 32, 32).astype(np.float32)
    sens = np.ones((inputs_np.shape[0], 10)).astype(np.float32)
    inputs_np_2 = np.random.rand(64, 1, 32, 32).astype(np.float32)

    net = LeNet()
    grad_all = GradWithSens(net)

    grad_out = grad_all(Tensor(inputs_np), Tensor(sens)).asnumpy()
    out_shape = net(Tensor(inputs_np_2)).asnumpy().shape
    assert np.any(grad_out != 0), 'grad result can not be all zeros'
    assert out_shape == (64, 10), 'output shape should be (64, 10)'


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_multi_grads():
    context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")
    sparse = False
    inputs_np = np.random.rand(32, 1, 32, 32).astype(np.float32)
    labels_np = np.random.randint(10, size=32).astype(np.int32)
    inputs_np_2 = np.random.rand(64, 1, 32, 32).astype(np.float32)
    labels_np_2 = np.random.randint(10, size=64).astype(np.int32)
    if not sparse:
        labels_np = np.eye(10)[labels_np].astype(np.float32)
        labels_np_2 = np.eye(10)[labels_np_2].astype(np.float32)

    net = LeNet()

    # grad operation
    loss_fn = nn.SoftmaxCrossEntropyWithLogits(sparse=sparse)
    with_loss_cell = WithLossCell(net, loss_fn)
    grad_all = GradWrapWithLoss(with_loss_cell)
    grad_out = grad_all(Tensor(inputs_np), Tensor(labels_np)).asnumpy()
    assert np.any(grad_out != 0), 'grad result can not be all zeros'

    # train-one-step operation
    loss_fn = nn.SoftmaxCrossEntropyWithLogits(sparse=sparse)
    optimizer = Momentum(filter(lambda x: x.requires_grad, net.get_parameters()),
                         0.01, 0.9)
    loss_net = WithLossCell(net, loss_fn)
    train_net = TrainOneStepCell(loss_net, optimizer)
    train_net.set_train()
    train_net(Tensor(inputs_np_2), Tensor(labels_np_2))
