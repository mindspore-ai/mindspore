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

import mindspore as ms
import mindspore.nn as nn
from mindspore import Tensor
from mindspore import context
from mindspore.common.api import _cell_graph_executor
from mindspore.ops import operations as P


class MatMulNet(nn.Cell):
    def __init__(self):
        super().__init__()
        self.matmul1 = P.MatMul()
        self.add1 = P.Add()
        self.reshape1 = P.Reshape()

    def construct(self, x, y, b):
        out = self.matmul1(x, y)
        out = self.reshape1(out, (2, 2048, 4096))
        out = self.add1(out, b)
        return out


def test_auto_parallel_sapp_broadcast():
    """
    Feature: test strategy propagation with broadcast in SAPP
    Description: auto parallel
    Expectation: compile success
    """

    context.set_auto_parallel_context(dataset_strategy="full_batch")
    context.set_auto_parallel_context(device_num=8, global_rank=0)
    context.set_auto_parallel_context(parallel_mode="auto_parallel", search_mode="recursive_programming")

    x = Tensor(np.ones([4096, 320]), dtype=ms.float32)
    y = Tensor(np.ones([320, 4096]), dtype=ms.float32)
    b = Tensor(np.ones([2048, 4096]), dtype=ms.float32)
    net = MatMulNet()
    net.set_train()
    _cell_graph_executor.compile(net, x, y, b, phase='train')
