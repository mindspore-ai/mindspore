# Copyright 2024 Huawei Technologies Co., Ltd
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

import mindspore as ms
import mindspore.nn as nn
from mindspore import Tensor
from mindspore import context
from mindspore.common.api import _cell_graph_executor
from mindspore.common.parameter import Parameter
from mindspore.ops import composite as C
from mindspore.ops import operations as P
from tests.dataset_mock import MindData
from tests.ut.python.ops.test_math_ops import VirtualLoss


def setup_function():
    context.set_auto_parallel_context(dataset_strategy="full_batch")


context.set_context(mode=context.GRAPH_MODE)

grad_all = C.GradOperation(get_all=True)


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


class Net(nn.Cell):
    def __init__(self):
        super().__init__()
        self.grid_sampler2d = P.GridSampler2D(interpolation_mode='bilinear', padding_mode='zeros', align_corners=True)
        self.parameter = Parameter(Tensor(np.arange(8).reshape((1, 2, 2, 2)).astype(np.float32)),
                                   name="assignadd_weight")
        self.grid = Tensor(np.arange(-9, 9, 0.5).reshape((2, 3, 3, 2)).astype(np.float32))
        self.add = P.Add()
        self.tile = P.Tile()


    def construct(self, x, y):
        x = self.tile(self.parameter, (2, 1, 1, 1))
        output = self.grid_sampler2d(x, self.grid)
        output2 = self.add(output, y)
        return output2

    def shard(self, strategy):
        self.grid_sampler2d.shard(strategy)

class NetWithLoss(nn.Cell):
    def __init__(self, network):
        super(NetWithLoss, self).__init__()
        self.loss = VirtualLoss()
        self.network = network

    def construct(self, x, y):
        predict = self.network(x, y)
        return self.loss(predict)


class GradWrap(nn.Cell):
    def __init__(self, network):
        super(GradWrap, self).__init__()
        self.network = network

    def construct(self, x, y):
        return grad_all(self.network)(x, y)


def test_grid_sampler2d_semi_auto_parallel():
    """
    Feature: test gridsampler2d
    Description: test data parallel in semi auto parallel
    Expectation: compile success
    """
    context.set_auto_parallel_context(device_num=8, global_rank=0)
    net = GradWrap(NetWithLoss(Net()))
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")

    x = Tensor(np.ones([2, 2, 3, 3]), dtype=ms.float32)
    y = Tensor(np.ones([2, 2, 3, 3]), dtype=ms.float32)
    net.set_train()
    _cell_graph_executor.compile(net, x, y)


def test_grid_sampler2d_semi_auto_parallel_with_dp_shard():
    """
    Feature: test gridsampler2d with shard
    Description: test data parallel in semi auto parallel
    Expectation: compile success
    """
    context.set_auto_parallel_context(device_num=8, global_rank=0)
    grid_net = Net()
    grid_net.shard(((2, 1, 1, 1), (2, 1, 1, 1)))
    net = GradWrap(NetWithLoss(grid_net))
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")

    x = Tensor(np.ones([2, 2, 3, 3]), dtype=ms.float32)
    y = Tensor(np.ones([2, 2, 3, 3]), dtype=ms.float32)
    net.set_train()
    _cell_graph_executor.compile(net, x, y)

def test_grid_sampler2d_auto_parallel():
    """
    Feature: test gridsampler2d
    Description: test data parallel in auto parallel
    Expectation: compile success
    """
    context.set_auto_parallel_context(device_num=8, global_rank=0)
    grid_net = Net()
    grid_net.shard(((2, 1, 1, 1), (2, 1, 1, 1)))
    net = GradWrap(NetWithLoss(grid_net))
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")

    x = Tensor(np.ones([2, 2, 3, 3]), dtype=ms.float32)
    y = Tensor(np.ones([2, 2, 3, 3]), dtype=ms.float32)
    net.set_train()
    _cell_graph_executor.compile(net, x, y)
