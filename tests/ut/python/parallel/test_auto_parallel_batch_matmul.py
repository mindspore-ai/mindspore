# Copyright 2023 Huawei Technologies Co., Ltd
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
import re
import mindspore as ms
import mindspore.nn as nn
from mindspore.ops import operations as P
from mindspore import Tensor
from mindspore import context
from mindspore.common.api import _cell_graph_executor
from parallel.utils.utils import compile_net


class BatchMatmulNet(nn.Cell):
    def __init__(self):
        super().__init__()
        self.bmm = P.BatchMatMul()

    def construct(self, x, y):
        output = self.bmm(x, y)
        return output


class MatmulBatchMatmulNet(nn.Cell):
    def __init__(self):
        super().__init__()
        self.matmul = P.MatMul()
        self.bmm = P.BatchMatMul()
        self.reshape = P.Reshape()

    def construct(self, x, y, z, shape):
        output = self.matmul(x, y)
        output = self.reshape(output, shape)
        output = self.bmm(output, z)
        return output


def test_batch_matmul():
    """
    Feature: batch matmul op
    Description:
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="auto_parallel", search_mode="recursive_programming",
                                      device_num=4, global_rank=0)
    input_x = Tensor(np.ones([32, 32, 32, 16]), dtype=ms.float32)
    input_y = Tensor(np.ones([32, 32, 16, 16]), dtype=ms.float32)
    net = BatchMatmulNet()

    compile_net(net, input_x, input_y)
    strategies = _cell_graph_executor._get_shard_strategy(net)
    for (k, v) in strategies.items():
        if re.search('BatchMatMul-op', k) is not None:
            assert v == [4, 1, 1, 1]


def test_batch_matmul_propagate():
    """
    Feature: batch matmul op stra is propagated
    Description:
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="auto_parallel", search_mode="recursive_programming",
                                      device_num=8, global_rank=0)
    input_x = Tensor(np.ones([256, 1024]), dtype=ms.float32)
    input_y = Tensor(np.ones([128, 128]), dtype=ms.float32)
    input_z = Tensor(np.ones([8, 32, 8, 32]), dtype=ms.float32)
    shape = (64, 32, 16, 8)
    net = MatmulBatchMatmulNet()

    compile_net(net, input_x, input_y, input_z, shape)
    strategies = _cell_graph_executor._get_shard_strategy(net)
    for (k, v) in strategies.items():
        if re.search('BatchMatMul-op', k) is not None:
            assert v == [4, 2, 1, 1]
