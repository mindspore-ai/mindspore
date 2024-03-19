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
import scipy
import mindspore
import mindspore.context as context
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor
from mindspore.ops import operations as P

class SoftmaxNet(nn.Cell):
    def __init__(self, axis):
        super(SoftmaxNet, self).__init__()
        self.softmax = P.Softmax()

    def construct(self, input_x):
        return ops.cast(self.softmax(input_x), mindspore.float32)


def softmax_net(input_params_shape, dtype):
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")

    axis = -1
    net = SoftmaxNet(axis)

    input_np = np.random.randn(*input_params_shape).astype(dtype)

    output = net(Tensor(input_np))
    expected = scipy.special.softmax(input_np, axis).astype(np.float32)

    decimal = 6
    if dtype == np.float16:
        decimal = 3
    np.testing.assert_array_almost_equal(output.asnumpy(), expected, decimal=decimal)


def test_softmax_fp16():
    """
    Feature: test add operator in graph mode
    Description: test add.
    Expectation: the result is correct
    """
    input_params_shape = (4, 2, 4096, 128)
    dtype = np.float16
    softmax_net(input_params_shape, dtype)

def test_softmax_fp32():
    """
    Feature: test add operator in graph mode
    Description: test add.
    Expectation: the result is correct
    """
    input_params_shape = (50257, 1024)
    dtype = np.float32
    softmax_net(input_params_shape, dtype)
