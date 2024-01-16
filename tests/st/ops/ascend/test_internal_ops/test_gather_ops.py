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
import pytest
import mindspore
import mindspore.context as context
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor
from mindspore.ops import operations as P
from bfloat16 import bfloat16

class GatherNet(nn.Cell):
    def __init__(self):
        super(GatherNet, self).__init__()
        self.gather = P.Gather()

    def construct(self, input_x, indices, axis):
        return ops.cast(self.gather(input_x, indices, axis), mindspore.float32)


def gather_net(input_params_shape, input_indices_shape, dtype):
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")

    net = GatherNet()

    axis = 0
    input_params_np = np.random.randn(*input_params_shape).astype(np.float16)
    input_indices_np = np.random.uniform(0, input_params_shape[axis], input_indices_shape).astype(np.int32)

    if dtype == bfloat16:
        output = net(Tensor(input_params_np, dtype=mindspore.bfloat16), Tensor(input_indices_np), axis)
    else:
        output = net(Tensor(input_params_np.astype(dtype)), Tensor(input_indices_np), axis)

    expected = np.take(input_params_np.astype(dtype), input_indices_np, axis).astype(np.float32)
    np.testing.assert_array_almost_equal(output.asnumpy(), expected)


def test_gather_fp16():
    """
    Feature: test add operator in graph mode
    Description: test add.
    Expectation: the result is correct
    """
    input_params_shape = (4, 2, 4096, 128)
    input_indices_shape = (1)
    dtype=np.float16
    gather_net(input_params_shape, input_indices_shape, dtype)


def test_gather_bf16():
    """
    Feature: test add operator in graph mode
    Description: test add.
    Expectation: the result is correct
    """
    input_params_shape = (50257, 1024)
    input_indices_shape = (4, 1)
    dtype=bfloat16
    gather_net(input_params_shape, input_indices_shape, dtype)


def test_gather_fp32():
    """
    Feature: test add operator in graph mode
    Description: test add.
    Expectation: the result is correct
    """
    input_params_shape = (50257, 1024)
    input_indices_shape = (1, 512)
    dtype=np.float32
    gather_net(input_params_shape, input_indices_shape, dtype)
