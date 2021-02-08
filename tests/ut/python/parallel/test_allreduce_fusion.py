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
from mindspore.common.api import _executor
from mindspore.nn.loss import SoftmaxCrossEntropyWithLogits
from mindspore.nn.optim.momentum import Momentum
from mindspore.parallel import _cost_model_context as cost_model_context
from mindspore.parallel._auto_parallel_context import auto_parallel_context
from mindspore.train import Model
from mindspore.context import ParallelMode
from tests.dataset_mock import MindData


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
    auto_parallel_context().set_enable_all_reduce_fusion(enable_all_reduce_fusion=True)
    context.set_auto_parallel_context(device_num=device_num, parameter_broadcast=False)
    context.set_context(mode=context.GRAPH_MODE)

    predict = Tensor(np.ones([batch_size, 128]), dtype=ms.float32)
    label = Tensor(np.ones([batch_size]), dtype=ms.int32)
    dataset = Dataset(predict, label, 2)

    loss = SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
    opt = Momentum(net.trainable_params(), learning_rate, momentum)
    model = Model(net, loss, opt)

    model.train(epoch_size, dataset, dataset_sink_mode=False)
    allreduce_fusion_dict = _executor._get_allreduce_fusion(model._train_network)

    print(allreduce_fusion_dict)
    return allreduce_fusion_dict


@pytest.mark.skip(reason="depreciated feature")
def test_allreduce_fusion_parameters():
    cost_model_context.reset_cost_model_context()
    cost_model_context.set_cost_model_context(costmodel_allreduce_fusion_algorithm=2)
    algorithm = cost_model_context.get_cost_model_context('costmodel_allreduce_fusion_algorithm')
    assert algorithm == 2
    cost_model_context.set_cost_model_context(costmodel_allreduce_fusion_algorithm=1)
    algorithm = cost_model_context.get_cost_model_context('costmodel_allreduce_fusion_algorithm')
    assert algorithm == 1
    cost_model_context.reset_cost_model_context()
    algorithm = cost_model_context.get_cost_model_context('costmodel_allreduce_fusion_algorithm')
    assert algorithm == 0

    cost_model_context.set_cost_model_context(costmodel_allreduce_fusion_times=2)
    fusion_times = cost_model_context.get_cost_model_context('costmodel_allreduce_fusion_times')
    assert fusion_times == 2

    cost_model_context.set_cost_model_context(costmodel_allreduce_fusion_tail_percent=0.2)
    tail_percent = cost_model_context.get_cost_model_context('costmodel_allreduce_fusion_tail_percent')
    assert tail_percent == 0.2
    cost_model_context.reset_cost_model_context()
    tail_percent = cost_model_context.get_cost_model_context('costmodel_allreduce_fusion_tail_percent')
    assert tail_percent == 0.1

    cost_model_context.set_cost_model_context(costmodel_allreduce_fusion_tail_time=0.2)
    tail_time = cost_model_context.get_cost_model_context('costmodel_allreduce_fusion_tail_time')
    assert tail_time == 0.2
    cost_model_context.reset_cost_model_context()
    tail_time = cost_model_context.get_cost_model_context('costmodel_allreduce_fusion_tail_time')
    assert tail_time == 0.1

    cost_model_context.set_cost_model_context(costmodel_allreduce_fusion_allreduce_inherent_time=0.2)
    allreduce_inherent_time = cost_model_context.get_cost_model_context(
        'costmodel_allreduce_fusion_allreduce_inherent_time')
    assert allreduce_inherent_time == 0.2
    cost_model_context.reset_cost_model_context()
    allreduce_inherent_time = cost_model_context.get_cost_model_context(
        'costmodel_allreduce_fusion_allreduce_inherent_time')
    assert allreduce_inherent_time == 0.1

    cost_model_context.set_cost_model_context(costmodel_allreduce_fusion_allreduce_bandwidth=0.2)
    allreduce_bandwidth = cost_model_context.get_cost_model_context('costmodel_allreduce_fusion_allreduce_bandwidth')
    assert allreduce_bandwidth == 0.2
    cost_model_context.reset_cost_model_context()
    allreduce_bandwidth = cost_model_context.get_cost_model_context('costmodel_allreduce_fusion_allreduce_bandwidth')
    assert allreduce_bandwidth == 0.1

    cost_model_context.set_cost_model_context(costmodel_allreduce_fusion_computation_time_parameter=0.2)
    computation_time_parameter = cost_model_context.get_cost_model_context(
        'costmodel_allreduce_fusion_computation_time_parameter')
    assert computation_time_parameter == 0.2
    cost_model_context.reset_cost_model_context()
    computation_time_parameter = cost_model_context.get_cost_model_context(
        'costmodel_allreduce_fusion_computation_time_parameter')
    assert computation_time_parameter == 0.1


@pytest.mark.skip(reason="depreciated feature")
def test_allreduce_fusion1():
    cost_model_context.set_cost_model_context(costmodel_allreduce_fusion_algorithm=1)
    cost_model_context.set_cost_model_context(costmodel_allreduce_fusion_times=2)
    cost_model_context.set_cost_model_context(costmodel_allreduce_fusion_tail_percent=0.5)
    context.reset_auto_parallel_context()
    context.set_auto_parallel_context(parallel_mode=ParallelMode.SEMI_AUTO_PARALLEL)
    net = SimpleDMLNet(DenseNet1(has_bias=False, activation=None), DenseNet2(has_bias=False, activation=None))
    allreduce_fusion_dict = train_common(net)
    expect_dict = {'backbone2.fc8.weight': 2,
                   'backbone2.fc7.weight': 2,
                   'backbone2.fc6.weight': 2,
                   'backbone1.fc4.weight': 2,
                   'backbone1.fc3.weight': 2,
                   'backbone1.fc2.weight': 2,
                   'backbone2.fc5.weight': 1,
                   'backbone2.fc4.weight': 1,
                   'backbone2.fc3.weight': 1,
                   'backbone2.fc2.weight': 1,
                   'backbone2.fc1.weight': 1,
                   'backbone1.fc1.weight': 1}
    assert allreduce_fusion_dict == expect_dict
    cost_model_context.reset_cost_model_context()


@pytest.mark.skip(reason="depreciated feature")
# reset_cost_model_context is called, the default value of costmodel_allreduce_fusion_times is 0, step_allreduce_fusion
# is bypassed.
def test_allreduce_fusion2():
    cost_model_context.set_cost_model_context(costmodel_allreduce_fusion_times=2)
    cost_model_context.set_cost_model_context(costmodel_allreduce_fusion_tail_percent=0.5)
    cost_model_context.reset_cost_model_context()
    context.reset_auto_parallel_context()
    context.set_auto_parallel_context(parallel_mode=ParallelMode.SEMI_AUTO_PARALLEL)
    net = SimpleDMLNet(DenseNet1(has_bias=False, activation=None), DenseNet2(has_bias=False, activation=None))
    allreduce_fusion_dict = train_common(net)
    expect_dict = {}
    assert allreduce_fusion_dict == expect_dict
    cost_model_context.reset_cost_model_context()


@pytest.mark.skip(reason="depreciated feature")
def test_allreduce_fusion3():
    cost_model_context.set_cost_model_context(costmodel_allreduce_fusion_algorithm=1)
    cost_model_context.set_cost_model_context(costmodel_allreduce_fusion_times=3)
    cost_model_context.set_cost_model_context(costmodel_allreduce_fusion_tail_percent=0.3333333)
    context.reset_auto_parallel_context()
    context.set_auto_parallel_context(parallel_mode=ParallelMode.SEMI_AUTO_PARALLEL)
    net = SimpleDMLNet(DenseNet1(has_bias=True, activation='relu'), DenseNet2(has_bias=False, activation='relu'))
    allreduce_fusion_dict = train_common(net)
    expect_dict = {'backbone2.fc8.weight': 3,
                   'backbone2.fc7.weight': 3,
                   'backbone2.fc6.weight': 2,
                   'backbone2.fc5.weight': 2,
                   'backbone2.fc4.weight': 2,
                   'backbone2.fc3.weight': 1,
                   'backbone2.fc2.weight': 1,
                   'backbone2.fc1.weight': 1,
                   'backbone1.fc4.bias': 3,
                   'backbone1.fc4.weight': 3,
                   'backbone1.fc3.bias': 3,
                   'backbone1.fc3.weight': 2,
                   'backbone1.fc2.bias': 2,
                   'backbone1.fc2.weight': 2,
                   'backbone1.fc1.bias': 2,
                   'backbone1.fc1.weight': 2}
    assert allreduce_fusion_dict == expect_dict
    cost_model_context.reset_cost_model_context()


@pytest.mark.skip(reason="depreciated feature")
def test_allreduce_fusion4():
    cost_model_context.set_cost_model_context(costmodel_allreduce_fusion_algorithm=1)
    cost_model_context.set_cost_model_context(costmodel_allreduce_fusion_times=2)
    cost_model_context.set_cost_model_context(costmodel_allreduce_fusion_tail_percent=0.5)
    context.reset_auto_parallel_context()
    context.set_auto_parallel_context(parallel_mode=ParallelMode.SEMI_AUTO_PARALLEL)
    net = SimpleDMLNet(DenseNet2(has_bias=False, activation=None), DenseNet2(has_bias=False, activation=None))
    allreduce_fusion_dict = train_common(net)
    expect_dict = {'backbone2.fc8.weight': 2,
                   'backbone2.fc7.weight': 2,
                   'backbone2.fc6.weight': 2,
                   'backbone1.fc8.weight': 2,
                   'backbone1.fc7.weight': 2,
                   'backbone1.fc6.weight': 2,
                   'backbone2.fc5.weight': 1,
                   'backbone2.fc4.weight': 1,
                   'backbone2.fc3.weight': 1,
                   'backbone2.fc2.weight': 1,
                   'backbone2.fc1.weight': 1,
                   'backbone1.fc5.weight': 1,
                   'backbone1.fc4.weight': 1,
                   'backbone1.fc3.weight': 1,
                   'backbone1.fc2.weight': 1,
                   'backbone1.fc1.weight': 1}

    assert allreduce_fusion_dict == expect_dict
    cost_model_context.reset_cost_model_context()


@pytest.mark.skip(reason="depreciated feature")
def test_allreduce_fusion5():
    cost_model_context.set_cost_model_context(costmodel_allreduce_fusion_algorithm=2)
    cost_model_context.set_cost_model_context(costmodel_allreduce_fusion_tail_time=0.1)
    cost_model_context.set_cost_model_context(costmodel_allreduce_fusion_allreduce_inherent_time=0.05)
    cost_model_context.set_cost_model_context(costmodel_allreduce_fusion_allreduce_bandwidth=0.000001)
    cost_model_context.set_cost_model_context(costmodel_allreduce_fusion_computation_time_parameter=0.0000015)
    context.reset_auto_parallel_context()
    context.set_auto_parallel_context(parallel_mode=ParallelMode.SEMI_AUTO_PARALLEL)
    net = SimpleDMLNet(DenseNet2(has_bias=False, activation=None), DenseNet2(has_bias=False, activation=None))
    allreduce_fusion_dict = train_common(net)

    expect_dict = {'backbone2.fc8.weight': 3,
                   'backbone2.fc7.weight': 3,
                   'backbone2.fc6.weight': 3,
                   'backbone2.fc5.weight': 3,
                   'backbone2.fc4.weight': 2,
                   'backbone2.fc3.weight': 2,
                   'backbone2.fc2.weight': 1,
                   'backbone2.fc1.weight': 1,
                   'backbone1.fc8.weight': 3,
                   'backbone1.fc7.weight': 3,
                   'backbone1.fc6.weight': 3,
                   'backbone1.fc5.weight': 3,
                   'backbone1.fc4.weight': 2,
                   'backbone1.fc3.weight': 2,
                   'backbone1.fc2.weight': 1,
                   'backbone1.fc1.weight': 1,}

    assert allreduce_fusion_dict == expect_dict
    cost_model_context.reset_cost_model_context()
