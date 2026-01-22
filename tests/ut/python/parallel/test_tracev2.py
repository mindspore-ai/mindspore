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


import pytest
import numpy as np

from mindspore import Tensor, context
from mindspore.nn import Cell
from mindspore.ops import composite as C
from mindspore.ops import auto_generate as P
from mindspore.graph.api import _cell_graph_executor
from tests.ut.python.ops.test_math_ops import VirtualLoss


def setup_function():
    context.set_auto_parallel_context(dataset_strategy="full_batch")


a = Tensor(np.array([[[2, 3, 4], [3, 4, 5], [4, 5, 6]],
                     [[2, 3, 4], [3, 4, 5], [4, 5, 6]]]))

grad_all = C.GradOperation(get_all=True)


class NetWithLoss(Cell):
    def __init__(self, network):
        super(NetWithLoss, self).__init__()
        self.loss = VirtualLoss()
        self.network = network

    def construct(self, x):
        predict = self.network(x)
        return self.loss(predict)


class GradWrap(Cell):
    def __init__(self, network):
        super(GradWrap, self).__init__()
        self.network = network

    def construct(self, x):
        return grad_all(self.network)(x)


class Net(Cell):
    def __init__(self, offset, axis1, axis2, strategy=None):
        super(Net, self).__init__()
        self.tracev2 = P.TraceV2().shard(strategy)
        self.offset = offset
        self.axis1 = axis1
        self.axis2 = axis2

    def construct(self, x):
        return self.tracev2(x, self.offset, self.axis1, self.axis2)

def test_tracev2_auto_parallel():
    """
    Feature: test tracev2 auto parallel
    Description: auto parallel
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="auto_parallel", search_mode="sharding_propagation", device_num=2,
                                      global_rank=0)
    net = GradWrap(NetWithLoss(Net(offset=0, axis1=-2, axis2=-1)))
    net.set_train()
    _cell_graph_executor.compile(net, a)


def test_tracev2_model_parallel():
    """
    Feature: test tracev2 model parallel
    Description: model parallel
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=2, global_rank=0)
    net = GradWrap(NetWithLoss(Net(offset=0, axis1=-2, axis2=-1, strategy=((2, 1, 1),))))
    net.set_train()
    _cell_graph_executor.compile(net, a)


def test_tracev2_strategy_error():
    """
    Feature: test invalid strategy for tracev2
    Description: illegal strategy
    Expectation: raise RuntimeError
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=2, global_rank=0)
    net = GradWrap(NetWithLoss(Net(offset=0, axis1=-2, axis2=-1, strategy=((2, 2, 1),))))
    net.set_train()
    with pytest.raises(RuntimeError):
        _cell_graph_executor.compile(net, a)
