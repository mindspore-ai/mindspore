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

import numpy as np
import pytest
import mindspore as ms
import mindspore.nn as nn
from mindspore import Tensor, context
from mindspore.common.api import _cell_graph_executor
from mindspore.nn import TrainOneStepCell, WithLossCell
from mindspore.nn.loss import SoftmaxCrossEntropyWithLogits
from mindspore.nn.optim import Lamb
from mindspore.nn.optim.momentum import Momentum
from mindspore.ops import operations as P
from mindspore.parallel import _cost_model_context as cost_model_context
from mindspore.parallel._auto_parallel_context import auto_parallel_context
from mindspore.train import Model
from mindspore.context import ParallelMode
from tests.dataset_mock import MindData

context.set_context(mode=context.PYNATIVE_MODE)

class Net(nn.Cell):
    """Net definition"""
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Dense(128, 768, activation='relu')
        self.fc2 = nn.Dense(128, 768, activation='relu')
        self.fc3 = nn.Dense(128, 768, activation='relu')
        self.fc4 = nn.Dense(768, 768, activation='relu')
        self.relu4 = nn.ReLU()
        self.relu5 = nn.ReLU()
        self.transpose = P.Transpose()
        self.matmul1 = P.MatMul()
        self.matmul2 = P.MatMul()

    def construct(self, x):
        q = self.fc1(x)
        k = self.fc2(x)
        v = self.fc3(x)
        k = self.transpose(k, (1, 0))
        c = self.relu4(self.matmul1(q, k))
        s = self.relu5(self.matmul2(c, v))
        s = self.fc4(s)
        return s

class Dataset(MindData):
    def __init__(self, predict, label, length=3):
        super(Dataset, self).__init__(size=length)
        self.predict = predict
        self.label = label
        self.index = 0
        self.length = length

    def __iter__(self):
        return self

    def __next__(self):
        if self.index >= self.length:
            raise StopIteration
        self.index += 1
        return self.predict, self.label

    def reset(self):
        self.index = 0

class DenseNet1(nn.Cell):
    def __init__(self, has_bias=True, activation='relu'):
        super(DenseNet1, self).__init__()
        self.fc1 = nn.Dense(128, 128, has_bias=has_bias, activation=activation)
        self.fc2 = nn.Dense(128, 128, has_bias=has_bias, activation=activation)
        self.fc3 = nn.Dense(128, 128, has_bias=has_bias, activation=activation)
        self.fc4 = nn.Dense(128, 128, has_bias=has_bias, activation=activation)

    def construct(self, x):
        q = self.fc1(x)
        k = self.fc2(q)
        v = self.fc3(k)
        s = self.fc4(v)
        return s

class DenseNet2(nn.Cell):
    def __init__(self, has_bias=True, activation='relu'):
        super(DenseNet2, self).__init__()
        self.fc1 = nn.Dense(128, 128, has_bias=has_bias, activation=activation)
        self.fc2 = nn.Dense(128, 128, has_bias=has_bias, activation=activation)
        self.fc3 = nn.Dense(128, 128, has_bias=has_bias, activation=activation)
        self.fc4 = nn.Dense(128, 128, has_bias=has_bias, activation=activation)
        self.fc5 = nn.Dense(128, 128, has_bias=has_bias, activation=activation)
        self.fc6 = nn.Dense(128, 128, has_bias=has_bias, activation=activation)
        self.fc7 = nn.Dense(128, 128, has_bias=has_bias, activation=activation)
        self.fc8 = nn.Dense(128, 128, has_bias=has_bias, activation=activation)

    def construct(self, x):
        q = self.fc1(x)
        k = self.fc2(q)
        v = self.fc3(k)
        s = self.fc4(v)
        t = self.fc5(s)
        u = self.fc6(t)
        w = self.fc7(u)
        z = self.fc8(w)
        return z

class DenseNet3(nn.Cell):
    def __init__(self, has_bias=True, activation='relu'):
        super(DenseNet3, self).__init__()
        self.fc1 = nn.Dense(128, 128, has_bias=has_bias, activation=activation)

    def construct(self, x):
        q = self.fc1(x)
        return q

class SimpleDMLNet(nn.Cell):
    def __init__(self, net1, net2):
        super(SimpleDMLNet, self).__init__()
        self.backbone1 = net1
        self.backbone2 = net2

    def construct(self, x):
        x1 = self.backbone1(x)
        x2 = self.backbone2(x)
        return x1 + x2

def train_common(net):
    batch_size = 32
    learning_rate = 0.1
    momentum = 0.9
    epoch_size = 2
    device_num = 4
    context.set_auto_parallel_context(device_num=device_num, parameter_broadcast=False)
    context.set_context(mode=context.GRAPH_MODE)

    predict = Tensor(np.ones([batch_size, 128]), dtype=ms.float32)
    label = Tensor(np.ones([batch_size]), dtype=ms.int32)
    dataset = Dataset(predict, label, 2)

    loss = SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
    opt = Momentum(net.trainable_params(), learning_rate, momentum)
    model = Model(net, loss, opt)

    model.train(epoch_size, dataset, dataset_sink_mode=False)
    allreduce_fusion_dict = _cell_graph_executor._get_allreduce_fusion(model._train_network)
    print(allreduce_fusion_dict)
    return allreduce_fusion_dict

def test_allreduce_fusion_auto():
    """
    Feature: test_allreduce_fusion in auto mode
    Description: allreduce fusion in auto mode
    Expectation: success
    """
    comm_fusion_dict = {"allreduce": {"mode": "auto", "config": None}}
    context.set_auto_parallel_context(parallel_mode=ParallelMode.SEMI_AUTO_PARALLEL, comm_fusion=comm_fusion_dict)
    net = SimpleDMLNet(DenseNet1(has_bias=False, activation=None), DenseNet2(has_bias=False, activation=None))
    allreduce_fusion_dict = train_common(net)
    expect_dict = {'backbone2.fc8.weight': 1,
                   'backbone2.fc7.weight': 1,
                   'backbone2.fc6.weight': 1,
                   'backbone1.fc4.weight': 1,
                   'backbone1.fc3.weight': 1,
                   'backbone1.fc2.weight': 1,
                   'backbone2.fc5.weight': 1,
                   'backbone2.fc4.weight': 1,
                   'backbone2.fc3.weight': 1,
                   'backbone2.fc2.weight': 1,
                   'backbone2.fc1.weight': 1,
                   'backbone1.fc1.weight': 1}
    assert allreduce_fusion_dict == expect_dict

def test_allreduce_fusion_size():
    """
    Feature: test_allreduce_fusion in size mode
    Description: allreduce fusion in size mode
    Expectation: success
    """
    comm_fusion_dict = {"allreduce": {"mode": "size", "config": 32}}
    context.set_auto_parallel_context(parallel_mode=ParallelMode.SEMI_AUTO_PARALLEL, comm_fusion=comm_fusion_dict)
    net = SimpleDMLNet(DenseNet1(has_bias=False, activation=None), DenseNet2(has_bias=False, activation=None))
    allreduce_fusion_dict = train_common(net)
    expect_dict = {'backbone2.fc8.weight': 1,
                   'backbone2.fc7.weight': 1,
                   'backbone2.fc6.weight': 1,
                   'backbone1.fc4.weight': 1,
                   'backbone1.fc3.weight': 1,
                   'backbone1.fc2.weight': 1,
                   'backbone2.fc5.weight': 1,
                   'backbone2.fc4.weight': 1,
                   'backbone2.fc3.weight': 1,
                   'backbone2.fc2.weight': 1,
                   'backbone2.fc1.weight': 1,
                   'backbone1.fc1.weight': 1}
    assert allreduce_fusion_dict == expect_dict
    cost_model_context.reset_cost_model_context()
    comm_fusion = auto_parallel_context().get_comm_fusion()
    assert comm_fusion_dict == comm_fusion

def test_lamb_split_fusion_in_index():
    """
    Feature: test_allreduce_fusion in index mode
    Description: allreduce fusion in index mode
    Expectation: success
    """
    comm_fusion_dict = {"allreduce": {"mode": "index", "config": [2, 4, 6, 8]}}
    context.set_auto_parallel_context(parallel_mode="data_parallel", device_num=2, enable_parallel_optimizer=True,
                                      comm_fusion=comm_fusion_dict, dataset_strategy="data_parallel")
    inputs = Tensor(np.ones([32, 128]).astype(np.float32))
    label = Tensor(np.zeros([32, 768]).astype(np.float32))
    net = Net()
    net.set_train()
    loss = nn.SoftmaxCrossEntropyWithLogits()
    optimizer = Lamb(net.trainable_params(), learning_rate=0.1)

    net_with_loss = WithLossCell(net, loss)
    train_network = TrainOneStepCell(net_with_loss, optimizer)
    _cell_graph_executor.compile(train_network, inputs, label)
    context.reset_auto_parallel_context()

def test_allreduce_fusion_size_priority():
    """
    Feature: test priority of "enable_all_reduce_fusion" and "comm_fusion"
    Description: test priority of "enable_all_reduce_fusion" and "comm_fusion"
    Expectation: success
    """
    auto_parallel_context().set_enable_all_reduce_fusion(enable_all_reduce_fusion=False)
    comm_fusion_dict = {"allreduce": {"mode": "size", "config": 32}}
    context.set_auto_parallel_context(parallel_mode=ParallelMode.SEMI_AUTO_PARALLEL, comm_fusion=comm_fusion_dict)
    net = SimpleDMLNet(DenseNet1(has_bias=False, activation=None), DenseNet2(has_bias=False, activation=None))
    allreduce_fusion_dict = train_common(net)
    expect_dict = {}
    assert allreduce_fusion_dict == expect_dict
    auto_parallel_context().set_enable_all_reduce_fusion(enable_all_reduce_fusion=True)
    allreduce_fusion_dict = train_common(net)
    expect_dict = {'backbone2.fc8.weight': 1,
                   'backbone2.fc7.weight': 1,
                   'backbone2.fc6.weight': 1,
                   'backbone1.fc4.weight': 1,
                   'backbone1.fc3.weight': 1,
                   'backbone1.fc2.weight': 1,
                   'backbone2.fc5.weight': 1,
                   'backbone2.fc4.weight': 1,
                   'backbone2.fc3.weight': 1,
                   'backbone2.fc2.weight': 1,
                   'backbone2.fc1.weight': 1,
                   'backbone1.fc1.weight': 1}
    assert allreduce_fusion_dict == expect_dict

def test_allreduce_fusion_size_one_tensor():
    """
    Feature: test_allreduce_fusion in size mode with one tensor
    Description: test_allreduce_fusion in size mode with one tensor
    Expectation: success
    """
    comm_fusion_dict = {"allreduce": {"mode": "size", "config": 32}}
    context.set_auto_parallel_context(parallel_mode=ParallelMode.SEMI_AUTO_PARALLEL, comm_fusion=comm_fusion_dict)
    net = DenseNet3(has_bias=False, activation=None)
    allreduce_fusion_dict = train_common(net)
    expect_dict = {'fc1.weight': 1}
    assert allreduce_fusion_dict == expect_dict

def test_fusion_invalid_value_failed():
    """
    Feature: test_allreduce_fusion with invalid value
    Description: test_allreduce_fusion with invalid value
    Expectation: throw TypeError
    """
    with pytest.raises(TypeError):
        comm_fusion_dict = {"allreduce": {"mode": "size", "config": "30.12"}}
        context.set_auto_parallel_context(parallel_mode=ParallelMode.SEMI_AUTO_PARALLEL, comm_fusion=comm_fusion_dict)


def test_openstate_invalid_value_failed():
    """
    Feature: test_openstate with invalid value
    Description: test_openstate with invalid value
    Expectation: throw TypeError
    """
    with pytest.raises(TypeError):
        comm_fusion_dict = {"openstate": "True"}
        context.set_auto_parallel_context(parallel_mode=ParallelMode.SEMI_AUTO_PARALLEL, comm_fusion=comm_fusion_dict)


def test_enable_invalid_value_failed():
    """
    Feature: enable_all_reduce_fusion with invalid value
    Description: enable_all_reduce_fusion with invalid value
    Expectation: throw TypeError
    """
    with pytest.raises(TypeError):
        auto_parallel_context().set_enable_all_reduce_fusion(enable_all_reduce_fusion="fusion")


def test_allreduce_fusion_openstate():
    """
    Feature: test priority of "openstate" and "comm_fusion"
    Description: test priority of "openstate" and "comm_fusion"
    Expectation: success
    """
    comm_fusion_dict = {"openstate": False, "allreduce": {"mode": "size", "config": 32}}
    context.set_auto_parallel_context(parallel_mode=ParallelMode.SEMI_AUTO_PARALLEL, comm_fusion=comm_fusion_dict)
    net = SimpleDMLNet(DenseNet1(has_bias=False, activation=None), DenseNet2(has_bias=False, activation=None))
    allreduce_fusion_dict = train_common(net)
    expect_dict = {}
    assert allreduce_fusion_dict == expect_dict


def test_allreduce_fusion_auto_with_openstate():
    """
    Feature: test_allreduce_fusion in auto mode with openstate
    Description: allreduce fusion in auto mode with openstate
    Expectation: success
    """
    comm_fusion_dict = {"openstate": True, "allreduce": {"mode": "auto", "config": None}}
    context.set_auto_parallel_context(parallel_mode=ParallelMode.SEMI_AUTO_PARALLEL, comm_fusion=comm_fusion_dict)
    net = SimpleDMLNet(DenseNet1(has_bias=False, activation=None), DenseNet2(has_bias=False, activation=None))
    allreduce_fusion_dict = train_common(net)
    expect_dict = {'backbone2.fc8.weight': 1,
                   'backbone2.fc7.weight': 1,
                   'backbone2.fc6.weight': 1,
                   'backbone1.fc4.weight': 1,
                   'backbone1.fc3.weight': 1,
                   'backbone1.fc2.weight': 1,
                   'backbone2.fc5.weight': 1,
                   'backbone2.fc4.weight': 1,
                   'backbone2.fc3.weight': 1,
                   'backbone2.fc2.weight': 1,
                   'backbone2.fc1.weight': 1,
                   'backbone1.fc1.weight': 1}
    assert allreduce_fusion_dict == expect_dict


def test_allreduce_fusion_with_openstate_reset():
    """
    Feature: test_allreduce_fusion in auto mode with openstate and reset
    Description: allreduce fusion in auto mode with openstate and reset
    Expectation: success
    """
    comm_fusion_dict = {"openstate": False, "allreduce": {"mode": "auto", "config": None}}
    context.set_auto_parallel_context(parallel_mode=ParallelMode.SEMI_AUTO_PARALLEL, comm_fusion=comm_fusion_dict)
    net = SimpleDMLNet(DenseNet1(has_bias=False, activation=None), DenseNet2(has_bias=False, activation=None))
    allreduce_fusion_dict = train_common(net)
    expect_dict = {'backbone2.fc8.weight': 1,
                   'backbone2.fc7.weight': 1,
                   'backbone2.fc6.weight': 1,
                   'backbone1.fc4.weight': 1,
                   'backbone1.fc3.weight': 1,
                   'backbone1.fc2.weight': 1,
                   'backbone2.fc5.weight': 1,
                   'backbone2.fc4.weight': 1,
                   'backbone2.fc3.weight': 1,
                   'backbone2.fc2.weight': 1,
                   'backbone2.fc1.weight': 1,
                   'backbone1.fc1.weight': 1}
    assert allreduce_fusion_dict != expect_dict
    context.reset_auto_parallel_context()
    context.set_auto_parallel_context(parallel_mode=ParallelMode.SEMI_AUTO_PARALLEL)
    allreduce_fusion_dict = train_common(net)
    assert allreduce_fusion_dict == expect_dict


def test_get_comm_fusion():
    """
    Feature: test_get_comm_fusion in auto mode with openstate and reset
    Description:  get comm fusion and reset
    Expectation: success
    """
    comm_fusion_dict = {"openstate": False, "allreduce": {"mode": "auto", "config": None}}
    context.set_auto_parallel_context(parallel_mode=ParallelMode.SEMI_AUTO_PARALLEL, comm_fusion=comm_fusion_dict)
    fusion_dict = context.get_auto_parallel_context("comm_fusion")
    assert fusion_dict == comm_fusion_dict
    context.reset_auto_parallel_context()
    fusion_dict = context.get_auto_parallel_context("comm_fusion")
    expect_dict = {"openstate": True,
                   "allreduce": {"mode": "auto", "config": None},
                   "allgather": {"mode": "auto", "config": None},
                   "reducescatter": {"mode": "auto", "config": None}}
    assert fusion_dict == expect_dict
