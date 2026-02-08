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


def test_construct_two_matmul_info():
    """
    Feature: Operator construction for auto parallel.
    Description: set strategy for matmul1 and gelu
    Expectation: relu gets right strategy (8, 1).
    """

    class Net(nn.Cell):
        def __init__(self,):
            super().__init__()
            ly = ms.Layout((4, 2), ("axis0", "axis1"))
            in_matmul1_strategy = (ly("axis0", "axis1"), ly("axis1", "None"))
            in_gelu_strategy = (ly(("axis0", "axis1"), "None"),)
            self.rma_net1 = RMANet(matmul1_strategy=in_matmul1_strategy, gelu_strategy=in_gelu_strategy)

        def construct(self, x, y):
            out1 = self.rma_net1(x, y)
            return out1


    class RMANet(nn.Cell):
        def __init__(self, matmul1_strategy=None, gelu_strategy=None):
            super().__init__()
            self.gamma = Parameter(Tensor(np.ones([96, 96]), dtype=ms.float32), name="gamma")
            self.beta = Parameter(Tensor(np.ones([96, 96]), dtype=ms.float32), name="beta")
            self.add_weight = Parameter(Tensor(np.ones([128, 96]), dtype=ms.float32), name="add_weight")
            self.add1 = mint.add
            self.add2 = mint.add
            if matmul1_strategy:
                self.matmul1 = ms.shard(mint.matmul, in_strategy=matmul1_strategy)
            else:
                self.matmul1 = mint.matmul

            self.matmul2 = mint.matmul

            self.relu = mint.nn.ReLU()

            if gelu_strategy:
                self.gelu = ms.shard(mint.nn.GELU(), in_strategy=gelu_strategy)
            else:
                self.gelu = mint.nn.GELU()


        def construct(self, x, y):
            out1 = self.add1(x, y)
            out2 = self.relu(self.gamma)
            out3 = self.matmul1(out1, out2)
            out4 = self.add2(out3, self.add_weight)
            out5 = self.gelu(self.beta)
            out6 = self.matmul2(out4, out5)
            return out6

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
    for (k, v) in strategies.items():
        if re.search('GeluExt-op0', k) is not None:
            print("check GeluExt-op0")
            assert v == [[8, 1]]
