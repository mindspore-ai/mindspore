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

import os
import numpy as np
import pytest
import mindspore
import mindspore.context as context
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor


class TransposeNet(nn.Cell):
    def __init__(self):
        super().__init__()
        self.transpose = ops.Transpose()

    def construct(self, x, perm):
        return self.transpose(x, perm)


def transpose_net(x_shape, perm, dtype, is_dyn=False):
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")

    x = np.random.randn(*x_shape).astype(dtype)

    net = TransposeNet()

    if is_dyn:
        # test dynamic shape
        x_dyn = Tensor(shape=[None] * x.ndim, dtype=mindspore.float16)
        print("python set_inputs start")
        net.set_inputs(x_dyn, perm)
        print("python set_inputs end")

    output = net(Tensor(x), perm)
    expected = x.transpose(perm)

    np.testing.assert_allclose(output.asnumpy(), expected, 0, 0)


def test_transpose(is_dyn=False):
    """
    Feature: test transpose operator in graph mode
    Description: test transpose.
    Expectation: the result is correct
    """
    transpose_net((4, 2, 4096, 128), (0, 3, 2, 1), np.float16, is_dyn)
    transpose_net((111, 222), (1, 0), np.float16, is_dyn)
    transpose_net((111, 222), (1, 0), np.int64, is_dyn)
    transpose_net((3, 4, 2), (0, 2, 1), np.float16, is_dyn)
    transpose_net((2, 2, 2, 32, 2), (0, 1, 2, 4, 3), np.float16, is_dyn)
    transpose_net((2, 2, 2, 2, 32, 2), (0, 1, 2, 3, 5, 4), np.float16, is_dyn)
    print("run transpose success")


def test_transpose_dyn():
    test_transpose(True)
