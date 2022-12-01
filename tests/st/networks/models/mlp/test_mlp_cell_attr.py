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

from mindspore.common import set_seed
from mindspore import Tensor
from mindspore.nn import TrainOneStepCell, WithLossCell
from mindspore.nn.optim import Momentum
from mindspore._extends import cell_attr_register


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


class CellDropDense(nn.Cell):
    @cell_attr_register
    def __init__(self):
        super(CellDropDense, self).__init__()
        self.fc = nn.Dense(100, 100)
        self.drop = nn.Dropout(1.0 - 0.1)

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


expect_value = [4.6052, 4.5553, 4.4607, 4.3261, 4.1556, 3.9532, 3.7227,
                3.4675, 3.1912, 2.8974, 2.5900, 2.2736, 1.9538, 1.6376,
                1.3335, 1.0511, 0.8002, 0.5884, 0.4195, 0.2920]


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_mlp_cell_reuse():
    """
    Feature: cell reuse.
    Description: MLP with cell reuse.
    Expectation: No exception.
    """
    context.set_context(mode=context.GRAPH_MODE)
    os.environ['MS_DEV_GRAPH_REUSE'] = str(1)

    # gen data
    seed_set()
    data = Tensor(np.random.random([1, 200]).astype(np.float32) * 0.01)
    label = Tensor(np.array(np.random.randint(100, size=[1]), dtype=np.int32))

    # cell reuse
    net = MLP()
    loss_list = train(net, data, label)
    del os.environ['MS_DEV_GRAPH_REUSE']

    assert np.allclose(loss_list, expect_value, 0.001, 0.001)


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_mlp_cell():
    """
    Feature: cell reuse.
    Description: MLP without cell reuse.
    Expectation: No exception.
    """
    context.set_context(mode=context.GRAPH_MODE)
    os.environ['MS_DEV_GRAPH_REUSE'] = str(0)

    # gen data
    seed_set()
    data = Tensor(np.random.random([1, 200]).astype(np.float32) * 0.01)
    label = Tensor(np.array(np.random.randint(100, size=[1]), dtype=np.int32))

    # cell not reuse
    net = MLP()
    loss_list = train(net, data, label)
    del os.environ['MS_DEV_GRAPH_REUSE']

    assert np.allclose(loss_list, expect_value, 0.001, 0.001)


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
