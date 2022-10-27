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
# ============================================================================
import numpy as np

import mindspore as ms
from mindspore import nn, context, Tensor
from mindspore.common.api import _cell_graph_executor
from mindspore.ops import operations as P


def setup_function():
    context.set_auto_parallel_context(dataset_strategy="full_batch")


class NetMul(nn.Cell):
    def __init__(self, strategy=None):
        super().__init__()
        self.mul = P.Mul().shard(strategy)

    def construct(self, x, y):
        return self.mul(x, y)


class NetMatMul(nn.Cell):
    def __init__(self, strategy=None):
        super().__init__()
        self.matmul = P.MatMul().shard(strategy)

    def construct(self, x, y):
        return self.matmul(x, y)

class NetRecursive(nn.Cell):
    def __init__(self):
        super().__init__()
        self.mul_net = NetMul()
        self.matmul_net = NetMatMul()

    def construct(self, x, y):
        out1 = self.matmul_net(x, y)
        out2 = self.matmul_net(x, y)
        return self.mul_net(out1, out2)

def compile_net(net, x, y):
    net.set_train()
    _cell_graph_executor.compile(net, x, y)


def test_batch_parallel_matmul():
    """
    Feature: shard at cell level
    Description: test batch matmul
    Expectation: using batch parallel mode to generate unspecified strategies in primitive ops
    """
    context.set_auto_parallel_context(device_num=8, global_rank=0)
    context.set_auto_parallel_context(parallel_mode="auto_parallel")
    net = NetMatMul()
    net.set_data_parallel()

    x = Tensor(np.ones([128, 32]), dtype=ms.float32)
    y = Tensor(np.ones([32, 128]), dtype=ms.float32)

    compile_net(net, x, y)


def test_batch_parallel_mul():
    """
    Feature: shard at cell level
    Description: test mul
    Expectation: using batch parallel mode to generate unspecified strategies in primitive ops
    """
    context.set_auto_parallel_context(device_num=8, global_rank=0)
    context.set_auto_parallel_context(parallel_mode="auto_parallel")
    net = NetMatMul()
    net.set_data_parallel()

    x = Tensor(np.ones([128, 128]), dtype=ms.float32)
    y = Tensor(np.ones([128, 128]), dtype=ms.float32)

    compile_net(net, x, y)


def test_batch_parallel_recursive():
    """
    Feature: shard at cell level
    Description: test primitive ops in cells wrapped by other cells
    Expectation: using batch parallel mode to generate unspecified strategies in primitive ops
    """
    context.set_auto_parallel_context(device_num=8, global_rank=0)
    context.set_auto_parallel_context(parallel_mode="auto_parallel")
    net = NetRecursive()
    net.set_data_parallel()

    x = Tensor(np.ones([128, 32]), dtype=ms.float32)
    y = Tensor(np.ones([32, 128]), dtype=ms.float32)

    compile_net(net, x, y)


def test_batch_parallel_with_user_strategy():
    """
    Feature: shard at cell level
    Description: test strategy gen mode while users have specified strategies
    Expectation: for those primitive ops who have users specified strategies, using those strategies;
                 for those who do not, using batch parallel mode to generate strategies
    """
    context.set_auto_parallel_context(device_num=8, global_rank=0)
    context.set_auto_parallel_context(parallel_mode="auto_parallel")
    net = NetMatMul(strategy=((1, 8), (8, 1)))
    net.set_data_parallel()

    x = Tensor(np.ones([128, 32]), dtype=ms.float32)
    y = Tensor(np.ones([32, 128]), dtype=ms.float32)

    compile_net(net, x, y)
