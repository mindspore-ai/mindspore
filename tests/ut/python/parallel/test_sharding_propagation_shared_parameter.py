# Copyright 2025 Huawei Technologies Co., Ltd
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
from mindspore import Tensor, context, mint
from mindspore.common import Parameter
from mindspore.graph.api import _cell_graph_executor
from mindspore.ops import composite as C
from mindspore.ops import operations as P
import re


def setup_function():
    context.set_auto_parallel_context(dataset_strategy="full_batch")


grad_all = C.GradOperation(get_all=True)


class GradWrapTwoInput(nn.Cell):
    def __init__(self, network):
        super(GradWrapTwoInput, self).__init__()
        self.network = network

    def construct(self, x, y):
        return grad_all(self.network)(x, y)


class NetWithLossTwoInput(nn.Cell):
    def __init__(self, network):
        super(NetWithLossTwoInput, self).__init__()
        self.mean = P.ReduceMean(keep_dims=False)
        self.network = network

    def construct(self, x, y):
        predict = self.network(x, y)
        return self.mean(predict, ())


def compile_graph(net, device_num, x):
    context.set_auto_parallel_context(device_num=device_num, global_rank=0, parallel_mode="auto_parallel",
                                      search_mode="sharding_propagation")
    net.set_train()
    _cell_graph_executor.compile(net, x)


def compile_graph_two_input(net, device_num, x, y):
    context.set_auto_parallel_context(device_num=device_num, global_rank=0, parallel_mode="auto_parallel",
                                      search_mode="sharding_propagation")
    net.set_train()
    _cell_graph_executor.compile(net, x, y)


def test_two_matmul_shared_parameter():
    """
    Feature: Operator construction for auto parallel.
    Description: set different strategy for the first input of matmul1 and matmul2
    Expectation: Two matmul get the same strategy on the shared parameter.
    """

    class Net(nn.Cell):
        def __init__(self,):
            super().__init__()
            ly1 = ms.Layout((4, 2), ("axis0", "axis1"))
            ly2 = ms.Layout((2, 4), ("axis0", "axis1"))
            self.rma_net1 = ms.shard(RMANet(), in_strategy=(ly1("axis0", "axis1"), ly2("axis0", "axis1")))

        def construct(self, x, y):
            out1 = self.rma_net1(x, y)
            return out1


    class RMANet(nn.Cell):
        def __init__(self,):
            super().__init__()
            self.gamma = Parameter(Tensor(np.ones([96, 96]), dtype=ms.float32), name="gamma")
            self.matmul1 = mint.matmul
            self.matmul2 = mint.matmul
            self.add = mint.add

        def construct(self, x, y):
            out1 = self.matmul1(x, self.gamma)
            out2 = self.matmul2(y, self.gamma)
            return self.add(out1, out2)

    device_num = 8
    context.set_auto_parallel_context(device_num=device_num, global_rank=0, parallel_mode="auto_parallel",
                                      search_mode="sharding_propagation")

    net = GradWrapTwoInput(NetWithLossTwoInput(Net()))
    net.set_train()

    x = Tensor(np.ones([128, 96]), dtype=ms.float32)
    y = Tensor(np.ones([128, 96]), dtype=ms.float32)


    _cell_graph_executor.compile(net, x, y, phase='train')
    strategies = _cell_graph_executor._get_shard_strategy(net)
    context._reset_auto_parallel_context()
    matmul1_strategy = [[1], [1]]
    matmul2_strategy = [[2], [2]]
    for (k, v) in strategies.items():
        if re.search('MatMulExt-op0', k) is not None:
            matmul1_strategy = v
        elif re.search('MatMulExt-op1', k) is not None:
            matmul2_strategy = v
    print("matmul1 strategy: ", matmul1_strategy)
    print("matmul2 strategy: ", matmul2_strategy)
    assert matmul1_strategy[1] == matmul2_strategy[1]
