# Copyright 2020-2022 Huawei Technologies Co., Ltd
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

import math
import pytest
import numpy as np

import mindspore.context as context
import mindspore.nn as nn
import mindspore as ms
from mindspore import Tensor, Parameter
from mindspore.nn import Dense
from mindspore.nn import TrainOneStepCell, WithLossCell
from mindspore.nn.optim import Adam
from mindspore.ops import operations as P
from mindspore.ops.functional import vmap

context.set_context(mode=context.GRAPH_MODE, device_target="CPU")


class NetAdam(nn.Cell):
    def __init__(self):
        super(NetAdam, self).__init__()
        self.batch_size = 1
        self.reshape = P.Reshape()
        weight = Tensor(np.ones([10, 16]).astype(np.float32) * 0.01)
        self.fc1 = Dense(16, 10, weight_init=weight)

    def construct(self, input_x):
        output = self.reshape(input_x, (self.batch_size, -1))
        output = self.fc1(output)
        return output


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_arm_cpu
@pytest.mark.env_onecard
def test_adam():
    """
    Feature: Adam optimizer
    Description: Verify if the loss is converged
    Expectation: success
    """
    epoch = 3
    net = NetAdam()
    optimizer = Adam(filter(lambda x: x.requires_grad, net.get_parameters()), learning_rate=0.01)
    criterion = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
    net_with_criterion = WithLossCell(net, criterion)
    train_network = TrainOneStepCell(net_with_criterion, optimizer)
    train_network.set_train()

    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    losses1 = []
    for _ in range(epoch):
        data = Tensor(np.arange(0, 16).reshape((1, 1, 4, 4)).astype(np.float32) * 0.01)
        label = Tensor(np.array([0]).astype(np.int32))
        loss = train_network(data, label)
        losses1.append(loss.asnumpy())
    assert losses1[0] > losses1[1]
    assert losses1[1] > losses1[2]


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_arm_cpu
@pytest.mark.env_onecard
def test_lazy_adam():
    """
    Feature: LazyAdam optimizer
    Description: Verify if the loss is converged
    Expectation: success
    """
    epoch = 3
    net = NetAdam()
    optimizer = Adam(filter(lambda x: x.requires_grad, net.get_parameters()), learning_rate=0.01, use_lazy=True)
    criterion = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
    net_with_criterion = WithLossCell(net, criterion)
    train_network = TrainOneStepCell(net_with_criterion, optimizer)
    train_network.set_train()

    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    losses1 = []
    for _ in range(epoch):
        data = Tensor(np.arange(0, 16).reshape((1, 1, 4, 4)).astype(np.float32) * 0.01)
        label = Tensor(np.array([0]).astype(np.int32))
        loss = train_network(data, label)
        losses1.append(loss.asnumpy())
    assert losses1[0] > losses1[1]
    assert losses1[1] > losses1[2]


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_arm_cpu
@pytest.mark.env_onecard
def test_lazy2():
    """
    Feature: Adam optimizer
    Description: Verify if the loss is correct
    Expectation: success
    """
    epoch = 3
    net = NetAdam()
    optimizer = Adam(filter(lambda x: x.requires_grad, net.get_parameters()), learning_rate=0.01, use_lazy=True)
    criterion = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
    net_with_criterion = WithLossCell(net, criterion)
    train_network = TrainOneStepCell(net_with_criterion, optimizer)
    train_network.set_train()

    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    losses1 = []
    for _ in range(epoch):
        data = Tensor(np.arange(0, 16).reshape((1, 1, 4, 4)).astype(np.float32) * 0.01)
        label = Tensor(np.array([0]).astype(np.int32))
        loss = train_network(data, label)
        losses1.append(loss.asnumpy())

    assert np.array_equal(losses1[-1], np.array(2.2237475, np.float32))


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_arm_cpu
@pytest.mark.env_onecard
def test_adam_offload():
    """
    Feature: LazyAdam optimizer
    Description: Verify if the loss is converged
    Expectation: success
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    epoch = 3
    net = NetAdam()
    optimizer = Adam(filter(lambda x: x.requires_grad, net.get_parameters()), learning_rate=0.01, use_offload=True)
    criterion = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
    net_with_criterion = WithLossCell(net, criterion)
    train_network = TrainOneStepCell(net_with_criterion, optimizer)
    train_network.set_train()

    losses1 = []
    for _ in range(epoch):
        data = Tensor(np.arange(0, 16).reshape((1, 1, 4, 4)).astype(np.float32) * 0.01)
        label = Tensor(np.array([0]).astype(np.int32))
        loss = train_network(data, label)
        losses1.append(loss.asnumpy())
    assert losses1[0] > losses1[1]
    assert losses1[1] > losses1[2]


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_arm_cpu
@pytest.mark.env_onecard
def test_adam_offload_acc():
    """
    Feature: AdamOffload optimizer
    Description: Verify if the loss is the same as the original AdamOffload
    Expectation: success
    """
    epoch = 3
    net = NetAdam()
    optimizer = Adam(filter(lambda x: x.requires_grad, net.get_parameters()), learning_rate=0.01, use_offload=True)
    criterion = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
    net_with_criterion = WithLossCell(net, criterion)
    train_network = TrainOneStepCell(net_with_criterion, optimizer)
    train_network.set_train()

    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    losses1 = []
    for _ in range(epoch):
        data = Tensor(np.arange(0, 16).reshape((1, 1, 4, 4)).astype(np.float32) * 0.01)
        label = Tensor(np.array([0]).astype(np.int32))
        loss = train_network(data, label)
        losses1.append(loss.asnumpy())

    assert np.array_equal(losses1[-1], np.array(2.2237475, np.float32))


def numpy_apply_adam(var, m, v, grad, beta1=0.9, beta2=0.999, eps=1e-8, lr=0.01):
    new_lr = lr * math.sqrt(1 - beta2) / (1 - beta1)
    m = m * beta1 + grad * (1 - beta1)
    v = v * beta2 + grad * grad * (1 - beta2)
    var = var - new_lr * m / (np.sqrt(v) + eps)
    return var


class AdamNetVmap(nn.Cell):
    def __init__(self, net):
        super(AdamNetVmap, self).__init__()
        shape = (8, 9, 6, 10, 5)
        self.net = net
        self.var_np = np.random.randn(*shape).astype(np.float32)
        self.m_np = np.random.randn(*shape).astype(np.float32)
        self.v_np = np.random.randn(*shape).astype(np.float32)
        self.var = Parameter(Tensor(self.var_np), name="var")
        self.m = Parameter(Tensor(self.m_np), name="m")
        self.v = Parameter(Tensor(self.v_np), name="v")
        self.vmap_adam = vmap(self.net, in_axes=(
            0, 0, 0, None, None, None, None, None, None, 0), out_axes=0)

    def construct(self, beta1_power, beta2_power, lr, beta1, beta2, epsilon, grad):
        return self.vmap_adam(self.var, self.m, self.v, beta1_power, beta2_power, lr, beta1, beta2, epsilon, grad)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_arm_cpu
@pytest.mark.env_onecard
def test_apply_adam_witm_adam_op_vmap():
    """
    Feature: Adam cpu kernel
    Description: test the Adam vmap.
    Expectation: match to np benchmark.
    """
    shape = (8, 9, 6, 10, 5)

    def cal_amsgrad(var, m, v, beta1_power, beta2_power, lr, beta1, beta2, epsilon, grad):
        return P.Adam()(var, m, v, beta1_power, beta2_power, lr, beta1, beta2, epsilon, grad)

    error = 1e-4
    grad_np = np.random.randn(*shape).astype(np.float32)
    grad = Tensor(grad_np)

    vmap_adam = AdamNetVmap(cal_amsgrad)
    _ = vmap_adam(Tensor(0.9, ms.float32), Tensor(0.999, ms.float32), Tensor(0.01, ms.float32), Tensor(
        0.9, ms.float32), Tensor(0.999, ms.float32), Tensor(1e-8, ms.float32), grad)
    ms_var = vmap_adam.var.asnumpy()
    np_var = numpy_apply_adam(vmap_adam.var_np, vmap_adam.m_np,
                              vmap_adam.v_np, grad_np)

    np.testing.assert_allclose(ms_var, np_var, rtol=error, atol=error)
