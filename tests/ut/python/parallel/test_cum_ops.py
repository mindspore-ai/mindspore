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
        self.cum_sum = P.CumSum().shard(strategy)
        self.cum_prod = P.CumProd()
        self.cum_max = P.Cummax(axis=-2)
        self.cum_min = P.Cummin(axis=-2)
        self.reversev2 = P.ReverseV2(axis=[-2])
        self.log_softmax = P.LogSoftmax(axis=-2)
        self.softmax = P.Softmax(axis=-2)
        self.lgamma = P.Lgamma()
        self.trunc = P.Trunc()
        self.elu_act = P.Elu(alpha=1.0)

    def construct(self, x, y):
        out = self.add(x, self.weight)
        out = self.cum_sum(out, -2)
        out = self.cum_prod(out, -2)
        out, _ = self.cum_max(out)
        out, _ = self.cum_min(out)
        out = self.reversev2(out)
        out = self.lgamma(out)
        out = self.trunc(out)
        out = self.log_softmax(out)
        out = self.softmax(out)
        out = self.elu_act(out)
        return out


_x = Tensor(np.ones([4, 8, 4, 8]), dtype=ms.float32)
_w1 = Tensor(np.ones([4, 8, 4, 8]), dtype=ms.float32)
_b = Tensor(np.ones([4, 8, 4, 8]), dtype=ms.float32)


def test_cum_ops():
    """
    Features: test sharding propagation for cum ops
    Description:
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="auto_parallel", device_num=8, global_rank=0,
                                      search_mode="sharding_propagation")
    strategy = ((2, 2, 1, 2),)
    net = Net(_w1, strategy=strategy)
    _cell_graph_executor.compile(net, _x, _b, phase='train')
    strategies = _cell_graph_executor._get_shard_strategy(net)
    for (k, v) in strategies.items():
        if re.search("CumProd", k) is not None:
            assert v == [[2, 2, 1, 2],]
        elif re.search("Cummax", k) is not None:
            assert v == [[2, 2, 1, 2],]
        elif re.search("Cummin", k) is not None:
            assert v == [[2, 2, 1, 2],]
        elif re.search("ReverseV2", k) is not None:
            assert v == [[2, 2, 1, 2],]
        elif re.search("Lgamma", k) is not None:
            assert v == [[2, 2, 1, 2],]
        elif re.search("Trunc", k) is not None:
            assert v == [[2, 2, 1, 2],]
        elif re.search("LogSoftmax", k) is not None:
            assert v == [[2, 2, 1, 2],]
        elif re.search("Softmax", k) is not None:
            assert v == [[2, 2, 1, 2],]
        elif re.search("Elu", k) is not None:
            assert v == [[2, 2, 1, 2],]
