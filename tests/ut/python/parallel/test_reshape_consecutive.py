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
from mindspore import context, Tensor
from mindspore import nn
from mindspore.ops import operations as P
from mindspore.common.api import _cell_graph_executor
from parallel.utils.utils import compile_net


class ReshapeNetFirst(nn.Cell):
    def __init__(self, shape):
        super().__init__()
        self.relu = P.ReLU()
        self.reshape = P.Reshape()
        self.shape = shape

    def construct(self, x):
        output = self.relu(x)
        output = self.reshape(x, self.shape)
        return output


class ReshapeNetSecond(nn.Cell):
    def __init__(self, shape):
        super().__init__()
        self.relu = P.ReLU()
        self.reshape = P.Reshape()
        self.shape = shape

    def construct(self, x):
        output = self.reshape(x, self.shape)
        output = self.relu(x)
        return output


class ConsecutiveReshapeNet(nn.Cell):
    def __init__(self, shape1, shape2):
        super().__init__()
        self.net1 = ReshapeNetFirst(shape1)
        self.net2 = ReshapeNetSecond(shape2)

    def construct(self, x):
        output = self.net1(x)
        output = self.net2(output)
        return output


def test_consecutive_reshape():
    """
    Feature: consecutive reshape op
    Description:
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="auto_parallel", search_mode="recursive_programming",
                                      device_num=8, global_rank=0)
    input_x = Tensor(np.ones([32, 64]), dtype=ms.float32)
    shape1 = (256, 8, 8)
    shape2 = (256, 64)
    net = ConsecutiveReshapeNet(shape1, shape2)

    compile_net(net, input_x)
    strategies = _cell_graph_executor._get_shard_strategy(net)
    for (k, v) in strategies.items():
        if re.search('Reshape-op', k) is not None:
            assert v == []
