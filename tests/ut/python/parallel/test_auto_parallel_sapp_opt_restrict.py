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


class MatMulNet(nn.Cell):
    def __init__(self):
        super().__init__()
        self.matmul1 = P.MatMul()

    def construct(self, x, y):
        out = self.matmul1(x, y)
        return out


def test_auto_parallel_sapp_optimizer_parallel_restrict():
    """
    Feature: test optimizer parallel restrict in SAPP
    Description: auto parallel
    Expectation: compile success and and strategy correct
    """

    context.set_auto_parallel_context(dataset_strategy="full_batch")
    context.set_auto_parallel_context(device_num=8, global_rank=0)
    context.set_auto_parallel_context(parallel_mode="auto_parallel", search_mode="recursive_programming")
    context.set_auto_parallel_context(enable_parallel_optimizer=True)

    x = Tensor(np.ones([160, 320]), dtype=ms.float32)
    y = Tensor(np.ones([320, 320000]), dtype=ms.float32)
    net = MatMulNet()
    net.set_train()
    _cell_graph_executor.compile(net, x, y, phase='train')

    strategies = _cell_graph_executor._get_shard_strategy(net)
    for (k, v) in strategies.items():
        if re.search('MatMul-op0', k) is not None:
            assert v == [[1, 1], [1, 8]]
