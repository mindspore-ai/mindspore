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
from mindspore import ops
from mindspore.ops import composite as C
from mindspore.ops import operations as P

from tests.ut.python.ops.test_math_ops import VirtualLoss

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


def compile_net(net, x, y, b):
    net.set_train()
    _cell_graph_executor.compile(net, x, y, b)


def test_rotary_mul_semi_auto():
    """
    Feature: op rotary_mul support distribution
    Description: test rotary_mul op with input and strategy under semi-auto parallel
    Expectation: compile success
    """

    class Net(nn.Cell):
        def __init__(self, strategy1, strategy2):
            super().__init__()
            self.rotary_mul = ops.auto_generate.RotaryMul().shard(strategy1)
            self.relu = P.ReLU().shard(strategy2)

        def construct(self, x, r1, r2):
            out = self.rotary_mul(x, r1, r2)
            out = self.relu(out)
            return out

    strategy1 = ((4, 2, 1, 1), (1, 1, 1, 1), (1, 1, 1, 1))
    strategy2 = ((8, 1, 1, 1),)
    net = GradWrap(NetWithLoss(Net(strategy1, strategy2)))
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", full_batch=True)
    context.set_auto_parallel_context(device_num=8, global_rank=0)

    x = Tensor(np.random.rand(64, 8, 128, 128), dtype=ms.float32)
    r1 = Tensor(np.random.rand(1, 1, 128, 128), dtype=ms.float32)
    r2 = Tensor(np.random.rand(1, 1, 128, 128), dtype=ms.float32)
    compile_net(net, x, r1, r2)


def test_rotary_mul_auto():
    """
    Feature: op rotary_mul support distribution
    Description: test rotary_mul op with input and strategy under auto parallel model
    Expectation: compile success
    """

    class Net(nn.Cell):
        def __init__(self, strategy1):
            super().__init__()
            self.rotary_mul = ops.auto_generate.RotaryMul()
            self.relu = P.ReLU().shard(strategy1)

        def construct(self, x, r1, r2):
            out = self.rotary_mul(x, r1, r2)
            out = self.relu(out)
            return out

    strategy1 = ((8, 1, 1, 1),)
    net = GradWrap(NetWithLoss(Net(strategy1)))
    context.set_auto_parallel_context(parallel_mode="auto_parallel", search_mode="sharding_propagation",
                                      full_batch=True)
    context.set_auto_parallel_context(device_num=8, global_rank=0)

    x = Tensor(np.random.rand(64, 8, 128, 128), dtype=ms.float32)
    r1 = Tensor(np.random.rand(1, 1, 128, 128), dtype=ms.float32)
    r2 = Tensor(np.random.rand(1, 1, 128, 128), dtype=ms.float32)
    compile_net(net, x, r1, r2)
