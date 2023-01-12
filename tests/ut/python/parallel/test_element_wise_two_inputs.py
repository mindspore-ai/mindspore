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
import mindspore.ops as ops
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
        w1 = Tensor(np.ones([4, 8, 1, 8]), dtype=ms.int32)
        self.hypot_w = Parameter(w, "w1")
        self.igamma_w = Parameter(w, "w2")
        self.igammac_w = Parameter(w, "w3")
        self.next_after_w = Parameter(w, "w4")
        self.zeta_w = Parameter(w, "w5")
        self.left_shift_w = Parameter(w1, "w6")
        self.right_shift_w = Parameter(w1, "w7")
        self.hypot = P.Hypot().shard(strategy)
        self.left_shift = P.LeftShift()
        self.right_shift = P.RightShift()
        self.next_after = P.NextAfter()
        self.zeta = P.Zeta()
        self.cast = P.Cast()
        self.gcd = P.Gcd()
        self.gcd_weight = Parameter(w1, "w8")

    def construct(self, x):
        out = self.hypot(x, self.hypot_w)
        out = ops.igamma(out, self.igamma_w)
        out = ops.igammac(out, self.igammac_w)
        out = self.next_after(out, self.next_after_w)
        out = self.zeta(out, self.zeta_w)
        out = self.cast(out, ms.int32)
        out = self.left_shift(out, self.left_shift_w)
        out = self.right_shift(out, self.right_shift_w)
        out = self.gcd(out, self.gcd_weight)
        return out


_x = Tensor(np.ones([4, 8, 4, 8]), dtype=ms.float32)


def test_element_wise_two_inputs_ops():
    """
    Features: test sharding propagation for element wise ops with two inputs
    Description:
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="auto_parallel", device_num=64, global_rank=0,
                                      search_mode="sharding_propagation")
    strategy = ((1, 2, 4, 8), (1, 2, 4, 8))
    net = Net(strategy=strategy)
    _cell_graph_executor.compile(net, _x, phase='train')
    strategies = _cell_graph_executor._get_shard_strategy(net)
    for (k, v) in strategies.items():
        if re.search("Igamma", k) is not None:
            assert v == [[1, 2, 4, 8], [1, 2, 4, 8]]
        elif re.search("Igammac", k) is not None:
            assert v == [[1, 2, 4, 8], [1, 2, 4, 8]]
        elif re.search("NextAfter", k) is not None:
            assert v == [[1, 2, 4, 8], [1, 2, 4, 8]]
        elif re.search("Zeta", k) is not None:
            assert v == [[1, 2, 4, 8], [1, 2, 4, 8]]
        elif re.search("LeftShift", k) is not None:
            assert v == [[1, 2, 4, 8], [1, 2, 1, 8]]
        elif re.search("RightShift", k) is not None:
            assert v == [[1, 2, 4, 8], [1, 2, 1, 8]]
        elif re.search("Gcd", k) is not None:
            assert v == [[1, 2, 4, 8], [1, 2, 1, 8]]
