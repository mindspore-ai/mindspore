# Copyright 2022-2022 Huawei Technologies Co., Ltd
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

"""train mlp network with cell_attr_register"""

import os
import random
import pytest
import numpy as np
import mindspore.context as context
import mindspore.nn as nn
import mindspore as ms

from mindspore.common import mutable
from mindspore.common import set_seed
from mindspore import Tensor
from mindspore.nn import TrainOneStepCell, WithLossCell
from mindspore.nn.optim import Momentum
from mindspore._extends import cell_attr_register


def seed_set():
    set_seed(1)
    np.random.seed(1)
    random.seed(1)


def train(net, data, label):
    learning_rate = 0.05
    momentum = 0.9

    optimizer = Momentum(filter(lambda x: x.requires_grad, net.get_parameters()), learning_rate, momentum)
    criterion = nn.SoftmaxCrossEntropyWithLogits(sparse=True)
    net_with_criterion = WithLossCell(net, criterion)
    train_network = TrainOneStepCell(net_with_criterion, optimizer)  # optimizer
    train_network.set_train()
    res_list = []
    for _ in range(20):
        res = train_network(data, label)
        res_list.append(res[0].asnumpy())
    return res_list


class CellDense(nn.Cell):
    @cell_attr_register
    def __init__(self):
        super(CellDense, self).__init__()
        self.fc = nn.Dense(100, 100)

    def construct(self, input_x):
        out = self.fc(input_x)
        return out


class MLP(nn.Cell):
    def __init__(self):
        super(MLP, self).__init__()
        self.batch_size = 1
        self.fc = nn.Dense(200, 100)

        layers = []
        for _ in range(12):
            layer = CellDense()
            layers.append(layer)

        self.layers = nn.CellList(layers)

    def construct(self, out):
        out = self.fc(out)
        for layer_module in self.layers:
            out = layer_module(out)
        return out


def get_pynative_mlp_cell_reuse_loss():
    context.set_context(mode=context.PYNATIVE_MODE)

    # gen data
    seed_set()
    data = Tensor(np.random.random([1, 200]).astype(np.float32) * 0.01)
    label = Tensor(np.array(np.random.randint(100, size=[1]), dtype=np.int32))

    # cell reuse
    net = MLP()
    loss_list = train(net, data, label)
    return loss_list


def get_mlp_cell_reuse_loss(reuse):
    context.set_context(mode=context.GRAPH_MODE)
    os.environ['MS_DEV_GRAPH_REUSE'] = reuse

    # gen data
    seed_set()
    data = Tensor(np.random.random([1, 200]).astype(np.float32) * 0.01)
    label = Tensor(np.array(np.random.randint(100, size=[1]), dtype=np.int32))

    # cell reuse
    net = MLP()
    loss_list = train(net, data, label)
    del os.environ['MS_DEV_GRAPH_REUSE']

    return loss_list


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_mlp_cell_reuse_0():
    """
    Feature: cell reuse.
    Description: MLP with cell reuse.
    Expectation: No exception.
    """
    loss_graph = get_mlp_cell_reuse_loss(str(0))
    loss_pynative = get_pynative_mlp_cell_reuse_loss()
    assert np.allclose(loss_pynative, loss_graph, 0.001, 0.001)


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_mlp_cell_reuse_1():
    """
    Feature: cell reuse.
    Description: MLP with cell reuse.
    Expectation: No exception.
    """
    loss_graph = get_mlp_cell_reuse_loss(str(1))
    loss_pynative = get_pynative_mlp_cell_reuse_loss()
    assert np.allclose(loss_pynative, loss_graph, 0.001, 0.001)


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_mlp_cell_reuse_2():
    """
    Feature: cell reuse.
    Description: MLP with cell reuse.
    Expectation: No exception.
    """
    loss_graph = get_mlp_cell_reuse_loss(str(2))
    loss_pynative = get_pynative_mlp_cell_reuse_loss()
    assert np.allclose(loss_pynative, loss_graph, 0.001, 0.001)


class CellDense2(nn.Cell):
    @cell_attr_register
    def __init__(self):
        super(CellDense2, self).__init__()
        self.fc = nn.Dense(100, 100)

    def construct(self, input_x):
        out = self.fc(input_x)
        return input_x, out


class MLP2(nn.Cell):
    def __init__(self):
        super(MLP2, self).__init__()
        self.batch_size = 1
        self.fc = nn.Dense(200, 100)

        layers = []
        for _ in range(12):
            layer = CellDense2()
            layers.append(layer)

        self.layers = nn.CellList(layers)

    def construct(self, out):
        out = self.fc(out)
        for layer_module in self.layers:
            tmp, out = layer_module(out)
            out += tmp
        return out


def get_pynative_mlp_cell_reuse_loss_2():
    context.set_context(mode=context.PYNATIVE_MODE)

    # gen data
    seed_set()
    data = Tensor(np.random.random([1, 200]).astype(np.float32) * 0.01)
    label = Tensor(np.array(np.random.randint(100, size=[1]), dtype=np.int32))

    # cell reuse
    net = MLP2()
    loss_list = train(net, data, label)
    return loss_list


def get_mlp_cell_reuse_loss_2(reuse):
    context.set_context(mode=context.GRAPH_MODE)
    os.environ['MS_DEV_GRAPH_REUSE'] = reuse

    # gen data
    seed_set()
    data = Tensor(np.random.random([1, 200]).astype(np.float32) * 0.01)
    label = Tensor(np.array(np.random.randint(100, size=[1]), dtype=np.int32))

    # cell reuse
    net = MLP2()
    loss_list = train(net, data, label)
    del os.environ['MS_DEV_GRAPH_REUSE']
    return loss_list


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_mlp_cell_2_reuse_0():
    """
    Feature: cell reuse.
    Description: MLP (need flatten maketuple) with cell reuse.
    Expectation: No exception.
    """
    loss_graph = get_mlp_cell_reuse_loss_2(str(0))
    loss_pynative = get_pynative_mlp_cell_reuse_loss_2()
    assert np.allclose(loss_pynative, loss_graph, 0.001, 0.001)


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_mlp_cell_2_reuse_1():
    """
    Feature: cell reuse.
    Description: MLP (need flatten maketuple) with cell reuse.
    Expectation: No exception.
    """
    loss_graph = get_mlp_cell_reuse_loss_2(str(1))
    loss_pynative = get_pynative_mlp_cell_reuse_loss_2()
    assert np.allclose(loss_pynative, loss_graph, 0.001, 0.001)


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_mlp_cell_2_reuse_2():
    """
    Feature: cell reuse.
    Description: MLP (need flatten maketuple) with cell reuse.
    Expectation: No exception.
    """
    loss_graph = get_mlp_cell_reuse_loss_2(str(2))
    loss_pynative = get_pynative_mlp_cell_reuse_loss_2()
    assert np.allclose(loss_pynative, loss_graph, 0.001, 0.001)


class CellDenseWithControlFlow(nn.Cell):
    @cell_attr_register
    def __init__(self):
        super(CellDenseWithControlFlow, self).__init__()
        self.fc = nn.Dense(100, 100)

    def construct(self, input_x, x):
        out = self.fc(input_x)
        if x > 0:
            out = self.fc(out)
        out = self.fc(out)
        return out


class MLPWithControlFlow(nn.Cell):
    def __init__(self):
        super(MLPWithControlFlow, self).__init__()
        self.batch_size = 1
        self.fc = nn.Dense(200, 100)

        layers = []
        for _ in range(12):
            layer = CellDenseWithControlFlow()
            layers.append(layer)

        self.layers = nn.CellList(layers)

    def construct(self, out):
        out = self.fc(out)
        for layer_module in self.layers:
            x = mutable(ms.Tensor(np.array(1), dtype=ms.int32))
            out = layer_module(out, x)
        return out


def get_pynative_mlp_cell_reuse_infer():
    context.set_context(mode=context.PYNATIVE_MODE)

    # gen data
    seed_set()
    data = Tensor(np.random.random([1, 200]).astype(np.float32) * 0.01)

    # cell reuse
    net = MLPWithControlFlow()
    ret = net(data)
    return ret.asnumpy()


def get_mlp_cell_reuse_infer(reuse):
    context.set_context(mode=context.GRAPH_MODE)
    os.environ['MS_DEV_GRAPH_REUSE'] = reuse

    # gen data
    seed_set()
    data = Tensor(np.random.random([1, 200]).astype(np.float32) * 0.01)

    # cell reuse
    net = MLPWithControlFlow()
    ret = net(data)
    del os.environ['MS_DEV_GRAPH_REUSE']
    return ret.asnumpy()


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_mlp_cell_with_control_flow_reuse_0():
    """
    Feature: cell reuse.
    Description: MLP with cell reuse.
    Expectation: No exception.
    """
    loss_graph = get_mlp_cell_reuse_infer(str(0))
    loss_pynative = get_pynative_mlp_cell_reuse_infer()
    assert np.allclose(loss_pynative, loss_graph, 0.001, 0.001)


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_mlp_cell_with_control_flow_reuse_1():
    """
    Feature: cell reuse.
    Description: MLP with cell reuse.
    Expectation: No exception.
    """
    loss_graph = get_mlp_cell_reuse_infer(str(1))
    loss_pynative = get_pynative_mlp_cell_reuse_infer()
    assert np.allclose(loss_pynative, loss_graph, 0.001, 0.001)


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_mlp_cell_with_control_flow_reuse_2():
    """
    Feature: cell reuse.
    Description: MLP with cell reuse.
    Expectation: No exception.
    """
    loss_graph = get_mlp_cell_reuse_infer(str(2))
    loss_pynative = get_pynative_mlp_cell_reuse_infer()
    assert np.allclose(loss_pynative, loss_graph, 0.001, 0.001)


class CellDropDense(nn.Cell):
    @cell_attr_register
    def __init__(self):
        super(CellDropDense, self).__init__()
        self.fc = nn.Dense(100, 100)
        self.drop = nn.Dropout(p=0.1)

    def construct(self, input_x):
        out = self.fc(input_x)
        out = self.drop(out)
        return out


class DropMLP(nn.Cell):
    def __init__(self):
        super(DropMLP, self).__init__()
        self.batch_size = 1
        self.fc = nn.Dense(200, 100)

        layers = []
        for _ in range(12):
            layer = CellDropDense()
            layers.append(layer)

        self.layers = nn.CellList(layers)

    def construct(self, out):
        out = self.fc(out)
        for layer_module in self.layers:
            out = layer_module(out)
        return out


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_mlp_drop_cell_reuse():
    """
    Feature: cell reuse.
    Description: MLP with cell reuse and dropout, need update caller abstract.
    Expectation: No exception.
    """
    context.set_context(mode=context.GRAPH_MODE)
    os.environ['MS_DEV_GRAPH_REUSE'] = str(1)

    # gen data
    seed_set()
    data = Tensor(np.random.random([1, 200]).astype(np.float32) * 0.01)
    label = Tensor(np.array(np.random.randint(100, size=[1]), dtype=np.int32))

    # cell reuse
    net = DropMLP()
    train(net, data, label)
    del os.environ['MS_DEV_GRAPH_REUSE']
