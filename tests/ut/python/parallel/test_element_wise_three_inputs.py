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
    def __init__(self, strategy):
        super().__init__()
        w = Tensor(np.ones([4, 8, 4, 8]), dtype=ms.float32)
        self.w1 = Parameter(w, "w1")
        self.w2 = Parameter(w, "w2")
        self.w3 = Parameter(w, "w3")
        self.w4 = Parameter(w, "w4")
        self.select = P.Select().shard(strategy)
        self.betainc = P.Betainc()

    def construct(self, x):
        out = self.select(x, self.w1, self.w2)
        out = self.betainc(out, self.w3, self.w4)
        return out


_x = Tensor(np.ones([4, 8, 4, 8]), dtype=ms.bool_)


def test_element_wise_three_inputs_ops():
    """
    Features: test sharding propagation for element wise ops with three inputs
    Description:
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="auto_parallel", device_num=64, global_rank=0,
                                      search_mode="sharding_propagation")
    strategy = ((1, 2, 4, 8), (1, 2, 4, 8), (1, 2, 4, 8))
    net = Net(strategy=strategy)
    _cell_graph_executor.compile(net, _x, phase='train')
    strategies = _cell_graph_executor._get_shard_strategy(net)
    for (k, v) in strategies.items():
        if re.search("Betainc", k) is not None:
            assert v == [[1, 2, 4, 8], [1, 2, 4, 8], [1, 2, 4, 8]]
