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

import mindspore as ms
import mindspore.nn as nn
from mindspore.common.api import _cell_graph_executor
from mindspore.ops import operations as P
from mindspore import context, Tensor

class GatherNet(nn.Cell):
    def __init__(self, axis=0, shape=None, target=""):
        super().__init__()
        self.matmul = P.MatMul()
        self.reshape = P.Reshape()
        if shape is None:
            shape = [64, 64]
        self.gatherv2 = P.Gather().set_device(target)
        self.index = Tensor(np.ones(shape), dtype=ms.int32)
        self.axis = axis

    def construct(self, x, y):
        out = self.gatherv2(x, self.index, self.axis)
        out = self.reshape(out, (128, 2048))
        out = self.matmul(out, y)
        return out

class GatherNet1D(nn.Cell):
    def __init__(self, axis=0, shape=None, target=""):
        super().__init__()
        self.gatherv2 = P.Gather().set_device(target)
        self.index = Tensor(np.ones(shape), dtype=ms.int32)
        self.axis = axis

    def construct(self, x):
        out = self.gatherv2(x, self.index, self.axis)
        return out

def test_auto_parallel_sapp_gather_1():
    """
    Feature: test Gather in SAPP
    Description: auto parallel
    Expectation: compile success and and strategy correct
    """
    context.set_auto_parallel_context(dataset_strategy="full_batch")
    context.set_auto_parallel_context(device_num=8, global_rank=0)
    context.set_auto_parallel_context(parallel_mode="auto_parallel", search_mode="recursive_programming")

    x = Tensor(np.ones([64, 64]), dtype=ms.float32)
    y = Tensor(np.ones([2048, 64]), dtype=ms.float32)

    net = GatherNet()
    net.set_train()
    _cell_graph_executor.compile(net, x, y, phase='train')

    strategies = _cell_graph_executor._get_shard_strategy(net)
    for (k, v) in strategies.items():
        if re.search('Default/Gather-op0', k) is not None:
            assert v == [[1, 1], [8, 1]]

def test_auto_parallel_sapp_gather_2():
    """
    Feature: test Gather in SAPP
    Description: auto parallel
    Expectation: compile success and and strategy correct
    """
    context.set_auto_parallel_context(dataset_strategy="full_batch")
    context.set_auto_parallel_context(device_num=8, global_rank=0)
    context.set_auto_parallel_context(parallel_mode="auto_parallel", search_mode="recursive_programming")

    x = Tensor(np.ones([1024, 128]), dtype=ms.float32)
    y = Tensor(np.ones([2048, 64]), dtype=ms.float32)

    net = GatherNet(shape=[2, 1024])
    net.set_train()
    _cell_graph_executor.compile(net, x, y, phase='train')

    strategies = _cell_graph_executor._get_shard_strategy(net)
    for (k, v) in strategies.items():
        if re.search('Default/Gather-op0', k) is not None:
            assert v == [[4, 1], [2, 1]]

def test_auto_parallel_sapp_gather_3():
    """
    Feature: test Gather in SAPP
    Description: auto parallel
    Expectation: compile success and and strategy correct
    """
    context.set_auto_parallel_context(dataset_strategy="full_batch")
    context.set_auto_parallel_context(device_num=8, global_rank=0)
    context.set_auto_parallel_context(parallel_mode="auto_parallel", search_mode="recursive_programming")

    x = Tensor(np.ones([128]), dtype=ms.float32)

    net = GatherNet1D(shape=[4])
    net.set_train()
    _cell_graph_executor.compile(net, x, phase='train')

    strategies = _cell_graph_executor._get_shard_strategy(net)
    for (k, v) in strategies.items():
        if re.search('Default/Gather-op0', k) is not None:
            assert v == [[1], [4]]
