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

import mindspore.nn as nn
from mindspore import Tensor
from mindspore import context
from mindspore.common.api import _cell_graph_executor
from mindspore.ops import operations as P


class LpNormNet(nn.Cell):
    def __init__(self):
        super().__init__()
        self.LpNorm = P.LpNorm(axis=[0, 1], p=2, keep_dims=False)

    def construct(self, input_x):
        out = self.LpNorm(input_x)
        return out



def test_auto_parallel_sapp_batch_parallel_operator():
    """
    Feature: test batch parallel operator's strategy in SAPP
    Description: auto parallel
    Expectation: compile success and strategy correct
    """

    context.set_auto_parallel_context(dataset_strategy="full_batch")
    context.set_auto_parallel_context(device_num=2, global_rank=0)
    context.set_auto_parallel_context(parallel_mode="auto_parallel", search_mode="recursive_programming")

    input_x = Tensor(np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]).astype(np.float32))
    net = LpNormNet()
    net.set_train()
    _cell_graph_executor.compile(net, input_x, phase='train')
    strategies = _cell_graph_executor._get_shard_strategy(net)
    for (k, v) in strategies.items():
        if re.search('LpNorm-op0', k) is not None:
            assert v == [[2, 1, 1]]
