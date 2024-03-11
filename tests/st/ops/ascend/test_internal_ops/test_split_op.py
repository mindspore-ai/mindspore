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
import mindspore
import mindspore.context as context
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor
from mindspore.ops import operations as P
from bfloat16 import bfloat16

class SplitNet(nn.Cell):
    def __init__(self, axis=0, output_num=2):
        super(SplitNet, self).__init__()
        self.split = P.Split(axis, output_num)

    def construct(self, input_x):
        return self.split(input_x)

def split_net(input_params_shape, dtype, axis=0, output_num=2):
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")

    net = SplitNet(axis, output_num)

    input_np = np.random.randn(*input_params_shape).astype(np.float16)

    if dtype == bfloat16:
        output = net(Tensor(input_np, dtype=mindspore.bfloat16))
    else:
        output = net(Tensor(input_np.astype(dtype)))

    expected = np.split(input_np.astype(dtype), output_num, axis)

    for res, exp in zip(output, expected):
        np.testing.assert_array_almost_equal(ops.cast(res, mindspore.float16).asnumpy(), exp.astype(np.float16))


def test_split_fp16():
    """
    Feature: test add operator in graph mode
    Description: test add.
    Expectation: the result is correct
    """
    input_params_shape = (4096, 128)
    dtype = np.float16
    split_net(input_params_shape, dtype)


def test_split_bf16():
    """
    Feature: test add operator in graph mode
    Description: test add.
    Expectation: the result is correct
    """
    input_params_shape = (4, 1024)
    dtype = bfloat16
    split_net(input_params_shape, dtype, axis=1, output_num=2)
    input_params_shape = (4, 128, 1024*3)
    split_net(input_params_shape, dtype, axis=-1, output_num=3)
