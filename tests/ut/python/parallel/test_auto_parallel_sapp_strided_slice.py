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
from mindspore import context, Tensor, Parameter

class StridedSliceNet(nn.Cell):
    def __init__(self, weight, w2, begin, end, strides, is_parameter=True,
                 begin_mask=0, end_mask=0, ellipsis_mask=0, new_axis_mask=0, shrink_axis_mask=0):
        super().__init__()
        self.matmul = P.MatMul()
        self.reshape = P.Reshape()
        self.strided_slice = P.StridedSlice(begin_mask=begin_mask,
                                            end_mask=end_mask,
                                            ellipsis_mask=ellipsis_mask, new_axis_mask=new_axis_mask,
                                            shrink_axis_mask=shrink_axis_mask)
        if is_parameter:
            self.weight = Parameter(weight, "w1")
        else:
            self.weight = weight
        self.mul2 = P.Mul()
        self.weight2 = Parameter(w2, "w2")
        self.begin = begin
        self.end = end
        self.strides = strides

    def construct(self, x):
        out = self.strided_slice(self.weight, self.begin, self.end, self.strides)
        out = self.reshape(out, (128, 2048))
        out = self.matmul(x, out)
        return out

def test_auto_parallel_sapp_strided_slice():
    """
    Feature: test Strided Slice in SAPP
    Description: auto parallel
    Expectation: compile success and and strategy correct
    """
    context.set_auto_parallel_context(dataset_strategy="full_batch")
    context.set_auto_parallel_context(device_num=8, global_rank=0)
    context.set_auto_parallel_context(parallel_mode="auto_parallel", search_mode="recursive_programming")

    x = Tensor(np.ones([128, 128]), dtype=ms.float32)
    _w1 = Tensor(np.ones([256, 64, 32]), dtype=ms.float32)
    _w2 = Tensor(np.ones([128, 64, 1]), dtype=ms.float32)

    net = StridedSliceNet(_w1, _w2, (0, 0, 0), (128, 64, 32), (1, 1, 1), is_parameter=True)
    net.set_train()
    _cell_graph_executor.compile(net, x, phase='train')

    strategies = _cell_graph_executor._get_shard_strategy(net)
    for (k, v) in strategies.items():
        if re.search('Default/StridedSlice-op0', k) is not None:
            assert v == [[1, 1, 1]]
