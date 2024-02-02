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
# ============================================================================
import numpy as np
import os

import mindspore as ms
import mindspore.nn as nn
from mindspore import Tensor
from mindspore import context
from mindspore.common.api import _cell_graph_executor
from mindspore.ops import operations as P


def setup_function():
    context.set_context(mode=context.GRAPH_MODE)
    os.putenv("PIPELINE_SLICE_SKIP_REDISTRIBUTION", "1")


def compile_net(net, inputs):
    net.set_train()
    phase, _ = _cell_graph_executor.compile(net, *inputs)
    context.reset_auto_parallel_context()
    return phase


class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.shape = P.Shape()
        self.micro_size = 2
        self.strided_slice = P.StridedSlice()
        self.strided_slice.add_prim_attr("strided_slice_flag", True)
        self.strided_slice.add_prim_attr("interleave_num", 2)

    def construct(self, input_x):
        i = 0
        input_shape = self.shape(input_x)
        micro_batch_begin = i * input_shape[0] // self.micro_size
        micro_batch_end = (i + 1) * input_shape[0] // self.micro_size
        strided_slice_begin = (micro_batch_begin,)
        strided_slice_strides = (1,)
        for _ in range(len(input_shape) - 1):
            strided_slice_begin += (0,)
            strided_slice_strides += (1,)
        strided_slice_end = (micro_batch_end,)
        strided_slice_end += input_shape[1:]
        micro_input = self.strided_slice(input_x, strided_slice_begin, strided_slice_end, strided_slice_strides)
        return micro_input


def test_set_stridedslice_strategy_semi_auto_parallel():
    """
    Feature: test PIPELINE_SLICE_SKIP_REDISTRIBUTION set_stridedslice_strategy
    Description: semi_auto_parallel
    Expectation: compile success
    """
    context.set_auto_parallel_context(device_num=8, global_rank=0)
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", full_batch=False)
    input_x = Tensor(np.ones((3, 3, 3)), ms.float32)
    inputs = [input_x]
    net = Net()
    compile_net(net, inputs)
