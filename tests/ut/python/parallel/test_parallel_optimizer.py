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
""" test adam """
import numpy as np
import pytest

import mindspore.nn as nn
from mindspore import Tensor, Parameter
from mindspore.common.api import _cell_graph_executor
from mindspore.nn import TrainOneStepCell, WithLossCell
from mindspore.nn.wrap.cell_wrapper import _VirtualDatasetCell
from mindspore.nn.optim import Adam, AdamWeightDecay, Lamb, Momentum
from mindspore.ops import operations as P
from mindspore import context


def setup_function():
    context.set_auto_parallel_context(dataset_strategy="full_batch")


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


class Net2(nn.Cell):
    """Net definition"""
    def __init__(self, strategy1, strategy2):
        super(Net2, self).__init__()
        self.fc1 = P.MatMul().shard(strategy1)
        self.fc2 = P.MatMul().shard(strategy2)
        self.p1 = Parameter(Tensor(np.ones([48, 64]).astype(np.float32)), name="weight1")
        self.p2 = Parameter(Tensor(np.ones([64, 16]).astype(np.float32)), name="weight2")

    def construct(self, x, y):
        x = self.fc1(x, self.p1)
        x = self.fc2(x, self.p2)
        return x - y


class Net3(nn.Cell):
    """Net definition"""
    def __init__(self, strategy1, strategy2):
        super(Net3, self).__init__()
        self.fc1 = P.MatMul().shard(strategy1)
        self.fc2 = P.MatMul().shard(strategy2)
        self.p1 = Parameter(Tensor(np.ones([48, 64]).astype(np.float32)), name="weight1")
        self.p2 = Parameter(Tensor(np.ones([64, 16]).astype(np.float32)), name="weight2", parallel_optimizer=False)

    def construct(self, x, y):
        x = self.fc1(x, self.p1)
        x = self.fc2(x, self.p2)
        return x - y


class Net4(nn.Cell):
    """Net definition"""
    def __init__(self, strategy1, strategy2):
        super(Net4, self).__init__()
        self.fc1 = P.MatMul().shard(strategy1)
        self.fc2 = P.MatMul().shard(strategy2)
        self.p1 = Parameter(Tensor(np.ones([48, 1152]).astype(np.float32)), name="weight1")
        self.p2 = Parameter(Tensor(np.ones([1152, 16]).astype(np.float32)), name="weight2")

    def construct(self, x, y):
        x = self.fc1(x, self.p1)
        x = self.fc2(x, self.p2)
        return x - y


def auto_parallel_compile_net(mode, dev_num, net, strategy1=None, strategy2=None):
    context.set_context(mode=context.GRAPH_MODE)
    context.set_auto_parallel_context(parallel_mode=mode, device_num=dev_num, enable_parallel_optimizer=True)
    inputs = Tensor(np.ones([32, 48]).astype(np.float32))
    label = Tensor(np.zeros([32, 16]).astype(np.float32))
    net = net(strategy1, strategy2)
    net = _VirtualDatasetCell(net)
    optimizer = Momentum(net.trainable_params(), learning_rate=0.1, momentum=0.9)
    train_network = TrainOneStepCell(net, optimizer).set_comm_fusion(4)
    train_network.set_train()
    _cell_graph_executor.compile(train_network, inputs, label, phase="train")
    context.reset_auto_parallel_context()
    return train_network


def test_auto_parallel_momentum_1():
    auto_parallel_compile_net("auto_parallel", 8, Net2)


def test_auto_parallel_momentum_2():
    # data parallel case
    auto_parallel_compile_net("auto_parallel", 8, Net2, ((8, 1), (1, 1)), ((8, 1), (1, 1)))


def test_auto_parallel_momentum_3():
    # hybrid parallel case
    # weight1 could not be shard and weight2 is repeated
    dp = 4
    context.set_auto_parallel_context(parallel_optimizer_config={"parallel_optimizer_threshold": 1})
    train_network = auto_parallel_compile_net("semi_auto_parallel", 32, Net2, ((dp, 8), (8, 1)), ((dp, 4), (4, 2)))
    param_dict = train_network.parameter_layout_dict
    # validate opt_shard_group
    assert not param_dict["weight1"][5]
    assert param_dict["weight2"][5].startswith(str(dp))


def test_auto_parallel_momentum_4():
    # hybrid parallel cases
    # devices are repeatedly used
    auto_parallel_compile_net("semi_auto_parallel", 32, Net2, ((4, 4), (4, 1)), ((4, 4), (4, 2)))


def test_auto_parallel_momentum_5():
    # test parallel optimizer filter
    context.set_auto_parallel_context(parallel_optimizer_config={"parallel_optimizer_threshold": 1})
    train_network = auto_parallel_compile_net("semi_auto_parallel", 32, Net3, ((4, 8), (8, 1)), ((4, 4), (4, 2)))
    param_dict = train_network.parameter_layout_dict
    # validate opt_shard_group
    assert not param_dict["weight1"][5]
    assert not param_dict["weight2"][5]


def test_auto_parallel_momentum_6():
    # test not fully use parallel optimizer with optimizer_weight_shard_size
    # weight1 could not be shard and weight2 is repeated
    param_shard_group_size = 2
    context.set_auto_parallel_context(optimizer_weight_shard_size=param_shard_group_size)
    context.set_auto_parallel_context(parallel_optimizer_config={"parallel_optimizer_threshold": 1})
    train_network = auto_parallel_compile_net("semi_auto_parallel", 32, Net2, ((4, 8), (8, 1)), ((4, 4), (4, 2)))
    param_dict = train_network.parameter_layout_dict
    # validate opt_shard_group
    assert param_dict["weight1"][5].startswith(str(param_shard_group_size))
    assert param_dict["weight2"][5].startswith(str(param_shard_group_size))

def test_default_threshold():
    """
    Feature: auto-parallel-optimizer(I4S85V)
    Description: the memory size of weight2(72KB) is higher than the threshold(64KB).
    Expectation: weight2 being sharded with sharding group size equal to dp.
    """
    dp = 4
    train_network = auto_parallel_compile_net("semi_auto_parallel", 32, Net4, ((dp, 8), (8, 1)), ((dp, 4), (4, 2)))
    param_dict = train_network.parameter_layout_dict
    # validate opt_shard_group
    assert param_dict["weight2"][5]

def test_user_define_threshold():
    """
    Feature: auto-parallel-optimizer(I4S85V)
    Description: the memory size of weight2(72KB) is lower than the threshold(100KB).
    Expectation: weight2 being not sharded.
    """
    dp = 4
    context.set_auto_parallel_context(parallel_optimizer_config={"parallel_optimizer_threshold": 100})
    train_network = auto_parallel_compile_net("semi_auto_parallel", 32, Net4, ((dp, 8), (8, 1)), ((dp, 4), (4, 2)))
    param_dict = train_network.parameter_layout_dict
    # validate opt_shard_group
    assert not param_dict["weight2"][5]


def test_AdamWeightDecay():
    """ test_AdamWeightDecay """
    context.set_auto_parallel_context(parallel_mode="data_parallel", device_num=2, enable_parallel_optimizer=True,
                                      parallel_optimizer_config={"parallel_optimizer_threshold": 1},
                                      dataset_strategy="data_parallel")
    inputs = Tensor(np.ones([32, 128]).astype(np.float32))
    label = Tensor(np.zeros([32, 768]).astype(np.float32))
    net = Net()
    net.set_train()
    loss = nn.SoftmaxCrossEntropyWithLogits()
    optimizer = AdamWeightDecay(net.trainable_params(), learning_rate=0.1)

    net_with_loss = WithLossCell(net, loss)
    train_network = TrainOneStepCell(net_with_loss, optimizer)
    _cell_graph_executor.compile(train_network, inputs, label)
    context.reset_auto_parallel_context()


def test_lamb_compile():
    """ test_Lamb_compile """
    context.set_auto_parallel_context(parallel_mode="data_parallel", device_num=2, enable_parallel_optimizer=True,
                                      parallel_optimizer_config={"parallel_optimizer_threshold": 2},
                                      dataset_strategy="data_parallel")
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


def test_lamb_split_fusion():
    """ test_Lamb_split_fusion """
    context.set_auto_parallel_context(parallel_mode="data_parallel", device_num=2, enable_parallel_optimizer=True,
                                      all_reduce_fusion_config=[2, 4, 6, 8], dataset_strategy="data_parallel",
                                      parallel_optimizer_config={"parallel_optimizer_threshold": 1})
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


def test_edge_case():
    """ test_edge_case """
    context.set_auto_parallel_context(enable_parallel_optimizer=True)
    net = Net()
    with pytest.raises(RuntimeError):
        context.set_auto_parallel_context(parallel_mode="stand_alone")
        Lamb(net.trainable_params(), learning_rate=0.1)
    with pytest.raises(RuntimeError):
        context.set_context(device_target="GPU")
        context.set_auto_parallel_context(parallel_mode="data_parallel", dataset_strategy="data_parallel")
        Lamb(net.trainable_params(), learning_rate=0.1)
    with pytest.raises(RuntimeError):
        context.set_context(device_target="Ascend")
        context.set_auto_parallel_context(parallel_mode="data_parallel", dataset_strategy="data_parallel")
        Adam(net.trainable_params(), learning_rate=0.1)
    with pytest.raises(RuntimeError):
        context.set_auto_parallel_context(device_num=16)
        Lamb(net.trainable_params(), learning_rate=0.1)
    with pytest.raises(ValueError):
        context.set_auto_parallel_context(parallel_optimizer_config={"parallel_optimizer_threshold": -1})
        Lamb(net.trainable_params(), learning_rate=0.1)
    context.reset_auto_parallel_context()
