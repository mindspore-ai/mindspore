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
from mindspore.common.api import _cell_graph_executor
from tests.ut.python.ops.test_math_ops import VirtualLoss


def setup_function():
    context.set_auto_parallel_context(dataset_strategy="full_batch")


a = Tensor(np.array([[[3, 0, 0, 0], [2, 1, 0, 0], [1, 0, 1, 0], [1, 1, 1, 1]],
                     [[3, 0, 0, 0], [2, 1, 0, 0], [1, 0, 1, 0], [1, 1, 1, 1]]], np.float32))
b = Tensor(np.array([[3, 1, 3, 4], [3, 1, 3, 4]], np.float32))

grad_all = C.GradOperation(get_all=True)


class NetWithLoss(Cell):
    def __init__(self, network):
        super(NetWithLoss, self).__init__()
        self.loss = VirtualLoss()
        self.network = network

    def construct(self, x, y):
        predict = self.network(x, y)
        return self.loss(predict)


class GradWrap(Cell):
    def __init__(self, network):
        super(GradWrap, self).__init__()
        self.network = network

    def construct(self, x, y):
        return grad_all(self.network)(x, y)


class Net(Cell):
    def __init__(self, trans, lower, unit_diagonal, strategy=None):
        super(Net, self).__init__()
        self.solve_triangular = P.SolveTriangular().shard(strategy)
        self.trans = trans
        self.lower = lower
        self.unit_diagonal = unit_diagonal

    def construct(self, x, y):
        return self.solve_triangular(x, y, self.trans, self.lower, self.unit_diagonal)


def test_solvetriangular_auto_parallel():
    """
    Feature: test solvetriangular auto parallel
    Description: auto parallel
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="auto_parallel", search_mode="dynamic_programming", device_num=2,
                                      global_rank=0)
    net = GradWrap(NetWithLoss(Net(trans=0, lower=True, unit_diagonal=False)))
    net.set_train()
    _cell_graph_executor.compile(net, a, b)


def test_solvetriangular_model_parallel():
    """
    Feature: test solvetriangular model parallel
    Description: model parallel
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=2, global_rank=0)
    net = GradWrap(NetWithLoss(Net(trans=0, lower=True, unit_diagonal=False, strategy=((2, 1, 1), (2, 1)))))
    net.set_train()
    _cell_graph_executor.compile(net, a, b)


def test_solvetriangular_strategy_error():
    """
    Feature: test invalid strategy for solvetriangular
    Description: illegal strategy
    Expectation: raise RuntimeError
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=2, global_rank=0)
    net = GradWrap(NetWithLoss(Net(trans=0, lower=True, unit_diagonal=False, strategy=((2, 2, 1), (2, 1)))))
    net.set_train()
    with pytest.raises(RuntimeError):
        _cell_graph_executor.compile(net, a, b)
