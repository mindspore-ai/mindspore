# Copyright 2022 Huawei Technologies Co., Ltd
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
from mindspore import context, Tensor, Parameter
from mindspore.common.api import _cell_graph_executor
from mindspore.nn import Cell
from mindspore.ops import operations as P


def setup_function():
    context.set_auto_parallel_context(dataset_strategy="full_batch")


class Net(Cell):
    def __init__(self, w, strategy):
        super().__init__()
        self.add = P.Add()
        self.weight = Parameter(w, "w1")
        self.softmax = P.Softmax(axis=(-1, -2)).shard(strategy)
        self.cholesky = P.Cholesky()

    def construct(self, x, y):
        out = self.add(x, self.weight)
        out = self.softmax(out)
        out = self.cholesky(out)
        return out


_x = Tensor(np.ones([4, 8, 8, 8]), dtype=ms.float32)
_w1 = Tensor(np.ones([4, 8, 8, 8]), dtype=ms.float32)
_b = Tensor(np.ones([4, 8, 8, 8]), dtype=ms.float32)


def test_element_wise_ops_with_axis():
    """
    Features: test sharding propagation for cholesky
    Description: the last two dimension of cholesky can not be split
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="auto_parallel", device_num=8, global_rank=0,
                                      search_mode="sharding_propagation")
    strategy = ((2, 4, 1, 1),)
    net = Net(_w1, strategy=strategy)
    _cell_graph_executor.compile(net, _x, _b, phase='train')
    strategies = _cell_graph_executor._get_shard_strategy(net)
    for (k, v) in strategies.items():
        if re.search("Cholesky", k) is not None:
            assert v == [[2, 4, 1, 1],]
