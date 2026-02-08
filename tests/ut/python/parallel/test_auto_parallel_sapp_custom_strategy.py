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

import re
import numpy as np
import os

import mindspore as ms
import mindspore.nn as nn
from mindspore.graph.api import _cell_graph_executor
from mindspore.ops import operations as P
from mindspore import context, Tensor

class MatMulNet(nn.Cell):
    def __init__(self):
        super().__init__()
        self.matmul1 = P.MatMul().shard(((1, 1), (1, 8)))
        self.matmul2 = P.MatMul()

    def construct(self, x, y, z):
        out = self.matmul1(x, y)
        out = self.matmul2(out, z)
        return out

def test_auto_parallel_sapp_custom_strategy1():
    """
    Feature: test Custom Strategy feature in SAPP
    Description: auto parallel
    Expectation: compile success and and strategy correct
    """
    context.set_auto_parallel_context(dataset_strategy="full_batch")
    context.set_auto_parallel_context(device_num=8, global_rank=0)
    context.set_auto_parallel_context(parallel_mode="auto_parallel", search_mode="recursive_programming")
    ms.set_algo_parameters(fully_use_devices=True)
    os.environ['MS_INTERFERED_SAPP'] = '1'

    x = Tensor(np.ones([32768, 1024]), dtype=ms.float32)
    y = Tensor(np.ones([1024, 1024]), dtype=ms.float32)
    z = Tensor(np.ones([1024, 1024]), dtype=ms.float32)

    net = MatMulNet()
    net.set_train()
    _cell_graph_executor.compile(net, x, y, z, phase='train')

    strategies = _cell_graph_executor._get_shard_strategy(net)
    for (k, v) in strategies.items():
        if re.search('Default/MatMul-op0', k) is not None:
            assert v == [[1, 8], [8, 1]]

def test_auto_parallel_sapp_custom_strategy2():
    """
    Feature: test Custom Strategy feature in SAPP
    Description: auto parallel
    Expectation: compile success and and strategy correct
    """
    context.set_auto_parallel_context(dataset_strategy="full_batch")
    context.set_auto_parallel_context(device_num=8, global_rank=0)
    context.set_auto_parallel_context(parallel_mode="auto_parallel", search_mode="recursive_programming")
    os.environ.pop('MS_INTERFERED_SAPP', None)

    x = Tensor(np.ones([32768, 1024]), dtype=ms.float32)
    y = Tensor(np.ones([1024, 1024]), dtype=ms.float32)
    z = Tensor(np.ones([1024, 1024]), dtype=ms.float32)

    net = MatMulNet()
    net.set_train()
    _cell_graph_executor.compile(net, x, y, z, phase='train')

    strategies = _cell_graph_executor._get_shard_strategy(net)
    for (k, v) in strategies.items():
        if re.search('Default/MatMul-op0', k) is not None:
            assert v == [[8, 1], [1, 1]]
