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

import re
import numpy as np

import mindspore as ms
import mindspore.nn as nn
from mindspore import Tensor
from mindspore import context
from mindspore.common.api import _cell_graph_executor
from mindspore.ops import operations as P
from mindspore.parallel._cost_model_context import _set_rp_matmul_mem_coef


class TwoMatMulNet(nn.Cell):
    def __init__(self):
        super().__init__()
        self.matmul1 = P.MatMul()
        self.matmul2 = P.MatMul()
        self.sub = P.Sub()
        self.add2 = P.Add()

    def construct(self, x, y, b, p, q):
        out = self.matmul1(x, y)
        out = self.sub(out, b)
        out = self.matmul2(out, p)
        out = self.add2(out, q)
        return out


class TwoMatMulReshapeNet(nn.Cell):
    def __init__(self):
        super().__init__()
        self.matmul1 = P.MatMul()
        self.matmul2 = P.MatMul()
        self.sub = P.Sub()
        self.add2 = P.Add()
        self.reshape = P.Reshape()

    def construct(self, x, y, b, p, r, q):
        out = self.matmul1(x, y)
        out = self.sub(out, b)
        out = self.matmul2(out, p)
        out = self.reshape(out, r)
        out = self.add2(out, q)
        return out


def test_auto_parallel_two_matmul_seq_parallel():
    """
    Feature: test auto parallel sequence parallel
    Description: auto parallel
    Expectation: compile success
    """

    context.set_auto_parallel_context(dataset_strategy="full_batch")
    context.set_auto_parallel_context(device_num=64, global_rank=0)
    x = Tensor(np.ones([640, 1280]), dtype=ms.float32)
    y = Tensor(np.ones([1280, 1280]), dtype=ms.float32)
    b = Tensor(np.ones([640, 1280]), dtype=ms.float32)
    p = Tensor(np.ones([1280, 1280]), dtype=ms.float32)
    q = Tensor(np.ones([640, 1280]), dtype=ms.float32)

    net = TwoMatMulNet()
    context.set_auto_parallel_context(parallel_mode="auto_parallel", search_mode="recursive_programming")
    _set_rp_matmul_mem_coef(1024)

    net.set_train()
    _cell_graph_executor.compile(net, x, y, b, p, q, phase='train')
    strategies = _cell_graph_executor._get_shard_strategy(net)
    for (k, v) in strategies.items():
        if re.search('Add-op0', k) is not None:
            assert v == [[64, 1], [64, 1]]


def test_auto_parallel_two_matmul_reshape1_seq_parallel():
    """
    Feature: test auto parallel sequence parallel
    Description: auto parallel
    Expectation: compile success
    """

    context.set_auto_parallel_context(dataset_strategy="full_batch")
    context.set_auto_parallel_context(device_num=64, global_rank=0)
    x = Tensor(np.ones([640, 1280]), dtype=ms.float32)
    y = Tensor(np.ones([1280, 1280]), dtype=ms.float32)
    b = Tensor(np.ones([640, 1280]), dtype=ms.float32)
    p = Tensor(np.ones([1280, 1280]), dtype=ms.float32)
    r = (64, 10, 1280)
    q = Tensor(np.ones([64, 10, 1280]), dtype=ms.float32)

    net = TwoMatMulReshapeNet()
    context.set_auto_parallel_context(parallel_mode="auto_parallel", search_mode="recursive_programming")
    _set_rp_matmul_mem_coef(1024)

    net.set_train()
    _cell_graph_executor.compile(net, x, y, b, p, r, q, phase='train')
    strategies = _cell_graph_executor._get_shard_strategy(net)
    for (k, v) in strategies.items():
        if re.search('Add-op0', k) is not None:
            assert v == [[64, 1, 1], [64, 1, 1]]


def test_auto_parallel_two_matmul_reshape2_seq_parallel():
    """
    Feature: test auto parallel sequence parallel
    Description: auto parallel
    Expectation: compile success
    """

    context.set_auto_parallel_context(dataset_strategy="full_batch")
    context.set_auto_parallel_context(device_num=64, global_rank=0)
    x = Tensor(np.ones([640, 1280]), dtype=ms.float32)
    y = Tensor(np.ones([1280, 1280]), dtype=ms.float32)
    b = Tensor(np.ones([640, 1280]), dtype=ms.float32)
    p = Tensor(np.ones([1280, 1280]), dtype=ms.float32)
    r = (10, 64, 1280)
    q = Tensor(np.ones([10, 64, 1280]), dtype=ms.float32)

    net = TwoMatMulReshapeNet()
    context.set_auto_parallel_context(parallel_mode="auto_parallel", search_mode="recursive_programming")
    _set_rp_matmul_mem_coef(1024)

    net.set_train()
    _cell_graph_executor.compile(net, x, y, b, p, r, q, phase='train')
    strategies = _cell_graph_executor._get_shard_strategy(net)
    for (k, v) in strategies.items():
        if re.search('Add-op0', k) is not None:
            assert v == [[1, 64, 1], [1, 64, 1]]


def test_auto_parallel_two_matmul_no_seq_parallel():
    """
    Feature: test auto parallel sequence parallel
    Description: auto parallel
    Expectation: compile success
    """

    context.set_auto_parallel_context(dataset_strategy="full_batch")
    context.set_auto_parallel_context(device_num=8, global_rank=0)
    x = Tensor(np.ones([160, 320]), dtype=ms.float32)
    y = Tensor(np.ones([320, 320]), dtype=ms.float32)
    b = Tensor(np.ones([160, 320]), dtype=ms.float32)
    p = Tensor(np.ones([320, 320]), dtype=ms.float32)
    q = Tensor(np.ones([160, 320]), dtype=ms.float32)

    net = TwoMatMulNet()
    context.set_auto_parallel_context(parallel_mode="auto_parallel", search_mode="recursive_programming")
    _set_rp_matmul_mem_coef(1024)

    net.set_train()
    _cell_graph_executor.compile(net, x, y, b, p, q, phase='train')
    strategies = _cell_graph_executor._get_shard_strategy(net)
    for (k, v) in strategies.items():
        if re.search('Add-op0', k) is not None:
            assert v == [[1, 1], [1, 1]]
