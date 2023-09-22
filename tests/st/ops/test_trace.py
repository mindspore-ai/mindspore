# Copyright 2023 Huawei Technologies Co., Ltd
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
import pytest
import random

from mindspore.common import set_seed
import mindspore as ms
from mindspore import nn, Tensor
from mindspore.ops._tracefunc import trace
from mindspore.ops import functional as F
from mindspore.ops import operations as P
from mindspore._extends import cell_attr_register
from mindspore.nn import TrainOneStepCell, WithLossCell
from mindspore.nn.optim import Momentum
from mindspore.nn.optim import Adam


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_arm_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_trace_basic_cell(mode):
    """
    Feature: trace of cell
    Description: Verify the result of trace
    Expectation: success
    """
    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.arg_max = P.ArgMaxWithValue()

        @trace
        def construct(self, x, y):
            if F.is_sequence_shape_unknown(x.shape):
                z = x + y
                return z * z
            z = x + y + 1
            _, b = self.arg_max(z)
            return z * z, b

    ms.set_context(mode=mode)
    net = Net()
    x = Tensor([1, 2, 3, 4])
    y = Tensor([4, 5, 6, 7], ms.float64)
    output, max_ = net(x, y)
    expect = np.array([36., 64., 100., 144.])
    expect_max = np.array([12])
    assert np.allclose(output.asnumpy(), expect)
    assert np.allclose(max_.asnumpy(), expect_max)
    assert output.dtype == ms.float64


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_trace_python_infer_cell():
    """
    Feature: trace of cell
    Description: Verify the result of trace
    Expectation: success
    """
    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.choleskytrsm = P.CholeskyTrsm()

        @trace
        def construct(self, x, y):
            z = self.choleskytrsm(x) + y
            return z * z
    @trace
    def func(x, k):
        z = x[0] * x[0]
        z[2] = z[1]
        z = P.Concat()((z, x[0] + k))
        return z
    ms.set_context(mode=ms.PYNATIVE_MODE)
    x = Tensor([1, 2, 3, 4])
    func_output = func((x,), 10)
    func_expect = np.array([1, 4, 4, 16, 11, 12, 13, 14])
    assert np.allclose(func_output.asnumpy(), func_expect)

    net = Net()
    x = Tensor(np.array([[0.25, 0], [0, 0.25]]), ms.float32)
    y = Tensor(np.array([[1, 2], [2, 1]]), ms.float32)
    output = net(x, y)
    expect = np.array([[9, 4], [4, 9]])
    assert np.allclose(output.asnumpy(), expect)


class CellDense(nn.Cell):
    @cell_attr_register
    def __init__(self, enable_trace):
        super(CellDense, self).__init__()
        self.fc = nn.Dense(10, 10)
        self.enable_trace = enable_trace

    @trace
    def t1(self, input_x):
        out = self.fc(input_x)
        return out

    def t2(self, input_x):
        out = self.fc(input_x)
        return out

    def construct(self, input_x):
        if self.enable_trace:
            out = self.t1(input_x)
        else:
            out = self.t2(input_x)
        return out


class MLP(nn.Cell):
    def __init__(self, enable_trace):
        super(MLP, self).__init__()
        self.batch_size = 1
        self.fc = nn.Dense(20, 10)

        layers = []
        for _ in range(2):
            layer = CellDense(enable_trace)
            layers.append(layer)

        self.layers = nn.CellList(layers)

    def construct(self, out):
        out = self.fc(out)
        for layer_module in self.layers:
            out = layer_module(out)
        return out


def train(net, data, label):
    learning_rate = 0.05
    momentum = 0.9

    optimizer = Momentum(filter(lambda x: x.requires_grad, net.get_parameters()), learning_rate, momentum)
    criterion = nn.SoftmaxCrossEntropyWithLogits(sparse=True)
    net_with_criterion = WithLossCell(net, criterion)
    train_network = TrainOneStepCell(net_with_criterion, optimizer)  # optimizer
    train_network.set_train()
    res_list = []
    for _ in range(3):
        res = train_network(data, label)
        res_list.append(res[0].asnumpy())
    return res_list


def seed_set():
    set_seed(1)
    np.random.seed(1)
    random.seed(1)


def get_mlp_cell_reuse_loss(enable_trace):
    ms.set_context(mode=ms.GRAPH_MODE)

    # gen data
    seed_set()
    data = Tensor(np.random.random([1, 20]).astype(np.float32) * 0.01)
    label = Tensor(np.array(np.random.randint(10, size=[1]), dtype=np.int32))

    # cell reuse
    net = MLP(enable_trace)
    loss_list = train(net, data, label)

    return loss_list


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_mlp_cell_reuse_trace():
    """
    Feature: cell reuse.
    Description: MLP with cell reuse.
    Expectation: No exception.
    """
    loss_trace = get_mlp_cell_reuse_loss(True)
    loss_no_trace = get_mlp_cell_reuse_loss(False)
    assert np.allclose(loss_trace, loss_no_trace, 0.001, 0.001)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_adam_trace(mode):
    """
    Feature: trace grad and mixed_precision
    Description: Verify if the loss is converged with trace
    Expectation: success
    """
    ms.set_context(mode=mode)

    class NetAdam(nn.Cell):
        def __init__(self):
            super(NetAdam, self).__init__()
            self.batch_size = 1
            self.reshape = P.Reshape()
            weight = Tensor(np.ones([10, 16]).astype(np.float32) * 0.01)
            self.fc1 = nn.Dense(16, 10, weight_init=weight, bias_init="zeros", activation="relu").to_float(ms.float16)
            self.add = P.Add()
            self.cast = P.Cast()

        def construct(self, input_x):
            output = self.reshape(input_x, (self.batch_size, -1))
            output = self.fc1(output)
            output = self.add(output, 0.1)
            output = self.cast(output, ms.float32)
            return output

    class NetAdamTrace(NetAdam):
        @trace
        def construct(self, input_x):
            return super().construct(input_x)

    def get_loss(net):
        epoch = 3
        optimizer = Adam(filter(lambda x: x.requires_grad, net.get_parameters()), learning_rate=0.01)
        criterion = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
        net_with_criterion = WithLossCell(net, criterion)
        train_network = TrainOneStepCell(net_with_criterion, optimizer)
        train_network.set_train()

        losses2 = []
        for _ in range(epoch):
            data = Tensor(np.arange(0, 16).reshape((1, 1, 4, 4)).astype(np.float32) * 0.01)
            label = Tensor(np.array([0]).astype(np.int32))
            loss = train_network(data, label)
            losses2.append(loss.asnumpy())
        return losses2

    loss = get_loss(NetAdam())
    trace_loss = get_loss(NetAdamTrace())
    assert np.allclose(loss, trace_loss, 1e-4, 1e-4)
