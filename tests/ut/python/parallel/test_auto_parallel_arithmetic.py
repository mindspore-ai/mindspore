# Copyright 2019 Huawei Technologies Co., Ltd
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

import re
import numpy as np

import mindspore as ms
import mindspore.nn as nn
from mindspore import Tensor
from mindspore import context
from mindspore.common.api import _cell_graph_executor
from mindspore.ops import composite as C
from mindspore.ops import operations as P
from mindspore.parallel._utils import _reset_op_id as reset_op_id
from tests.ut.python.ops.test_math_ops import VirtualLoss


def setup_function():
    context.set_auto_parallel_context(dataset_strategy="full_batch")

context.set_context(mode=context.GRAPH_MODE)


grad_all = C.GradOperation(get_all=True)


class NetWithLoss(nn.Cell):
    def __init__(self, network):
        super(NetWithLoss, self).__init__()
        self.loss = VirtualLoss()
        self.network = network

    def construct(self, x, y, b):
        predict = self.network(x, y, b)
        return self.loss(predict)


class GradWrap(nn.Cell):
    def __init__(self, network):
        super(GradWrap, self).__init__()
        self.network = network

    def construct(self, x, y, b):
        return grad_all(self.network)(x, y, b)


def compile_net(net, x, y, b, phase):
    net.set_train()
    _cell_graph_executor.compile(net, x, y, b, phase=phase)


def test_auto_parallel_arithmetic():
    """
    Features: test auto parallel
    Description: search strategies
    Expectation: Generated strategies matching expectations
    """
    class Net(nn.Cell):
        def __init__(self):
            super().__init__()
            self.matmul = P.MatMul()
            self.floordiv = P.FloorDiv()

        def construct(self, x, y, b):
            out = self.matmul(x, y)
            out = self.floordiv(out, b)
            return out

    context.set_auto_parallel_context(device_num=8, global_rank=0)
    net = NetWithLoss(Net())
    context.set_auto_parallel_context(parallel_mode="auto_parallel")
    reset_op_id()

    x = Tensor(np.ones([64, 32]), dtype=ms.float32)
    y = Tensor(np.ones([32, 128]), dtype=ms.float32)
    b = Tensor(np.ones([64, 128]), dtype=ms.float32)
    compile_net(net, x, y, b, phase='train')
    strategies = _cell_graph_executor._get_shard_strategy(net)
    for (k, v) in strategies.items():
        if re.search('FloorDiv-op', k) is not None:
            assert v == [[8, 1], [8, 1]]
        elif re.search('MatMul-op', k) is not None:
            assert v == [[8, 1], [1, 1]]
        elif re.search('_VirtualDataset-op', k) is not None:
            assert v == [[1, 1], [1, 1], [1, 1]]


def test_auto_parallel_arithmetic_broadcast_both():
    """
    Features: test auto parallel
    Description: search strategies for broadcast
    Expectation: Generated strategies matching expectations
    """
    class Net(nn.Cell):
        def __init__(self):
            super().__init__()
            self.matmul = P.MatMul()
            self.floordiv = P.FloorDiv()

        def construct(self, x, y, b):
            out = self.matmul(x, y)
            out = self.floordiv(out, b)
            return out

    context.set_auto_parallel_context(dataset_strategy="full_batch")
    context.set_auto_parallel_context(device_num=8, global_rank=0)
    net = NetWithLoss(Net())
    context.set_auto_parallel_context(parallel_mode="auto_parallel")
    reset_op_id()

    x = Tensor(np.ones([64, 32]), dtype=ms.float32)
    y = Tensor(np.ones([32, 1]), dtype=ms.float32)
    b = Tensor(np.ones([1, 64]), dtype=ms.float32)
    compile_net(net, x, y, b, phase='train')
    strategies = _cell_graph_executor._get_shard_strategy(net)
    for (k, v) in strategies.items():
        if re.search('FloorDiv-op', k) is not None:
            assert v == [[1, 1], [1, 1]]
        elif re.search('MatMul-op', k) is not None:
            assert v == [[1, 1], [1, 1]]


def test_auto_parallel_arithmetic_broadcast_right():
    """
    Features: test auto parallel
    Description: search strategies for right broadcast
    Expectation: Generated strategies matching expectations
    """
    class Net(nn.Cell):
        def __init__(self):
            super().__init__()
            self.matmul = P.MatMul()
            self.floordiv = P.FloorDiv()

        def construct(self, x, y, b):
            out = self.matmul(x, y)
            out = self.floordiv(out, b)
            return out

    context.set_auto_parallel_context(device_num=8, global_rank=0)
    net = NetWithLoss(Net())
    context.set_auto_parallel_context(parallel_mode="auto_parallel")
    reset_op_id()

    x = Tensor(np.ones([64, 32]), dtype=ms.float32)
    y = Tensor(np.ones([32, 32]), dtype=ms.float32)
    b = Tensor(np.ones([32]), dtype=ms.float32)
    compile_net(net, x, y, b, phase='train')
    strategies = _cell_graph_executor._get_shard_strategy(net)
    for (k, v) in strategies.items():
        if re.search('FloorDiv-op', k) is not None:
            assert v == [[1, 8], [8]]
        elif re.search('MatMul-op', k) is not None:
            assert v == [[1, 1], [1, 8]]
        elif re.search('_VirtualDataset-op', k) is not None:
            assert v == [[1, 1], [1, 1], [1]]


def test_auto_parallel_arithmetic_broadcast_left():
    """
    Features: test auto parallel
    Description: search strategies for left broadcast
    Expectation: Generated strategies matching expectations
    """
    class Net(nn.Cell):
        def __init__(self):
            super().__init__()
            self.matmul = P.MatMul()
            self.floordiv = P.FloorDiv()

        def construct(self, x, y, b):
            out = self.matmul(x, y)
            out = self.floordiv(out, b)
            return out

    context.set_auto_parallel_context(device_num=8, global_rank=0)
    net = NetWithLoss(Net())
    context.set_auto_parallel_context(parallel_mode="auto_parallel")
    reset_op_id()

    x = Tensor(np.ones([64, 32]), dtype=ms.float32)
    y = Tensor(np.ones([32, 32]), dtype=ms.float32)
    b = Tensor(np.ones([128, 64, 32]), dtype=ms.float32)
    compile_net(net, x, y, b, phase="train")
    strategies = _cell_graph_executor._get_shard_strategy(net)
    for (k, v) in strategies.items():
        if re.search('FloorDiv-op', k) is not None:
            assert v == [[1, 1], [1, 1, 1]]
        elif re.search('MatMul-op', k) is not None:
            assert v == [[1, 1], [1, 1]]
        elif re.search('_VirtualDataset-op', k) is not None:
            assert v == [[1, 1], [1, 1], [1, 1, 1]]
