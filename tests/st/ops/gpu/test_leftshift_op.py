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
# ============================================================================
from tests.mark_utils import arg_mark

import numpy as np
import pytest
import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.ops.operations.array_ops import LeftShift


class NetLeftShift(nn.Cell):
    def __init__(self):
        super(NetLeftShift, self).__init__()
        self.leftshift = LeftShift()

    def construct(self, x, y):
        return self.leftshift(x, y)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_leftshift_1d():
    """
    Feature: LeftShift gpu TEST.
    Description: 1d test case for LeftShift
    Expectation: the result match to numpy
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    x_np = (np.array([-1, -5, -3, -14, 64])).astype(np.int8)
    y_np = (np.array([5, 0, 7, 11, 1])).astype(np.int8)
    z_np = np.left_shift(x_np, y_np)
    print(z_np)

    x_ms = Tensor(x_np)
    y_ms = Tensor(y_np)
    net = NetLeftShift()
    z_ms = net(x_ms, y_ms)
    print(z_ms.asnumpy())

    assert np.allclose(z_np, z_ms.asnumpy())


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_leftshift_2d():
    """
    Feature: LeftShift gpu TEST.
    Description: 2d test case for LeftShift
    Expectation: the result match to numpy
    """
    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
    x_np = (np.array([[-1, -5, -3], [-14, 64, 0]])).astype(np.int8)
    y_np = (np.array([[5, 0, 7], [11, 1, 0]])).astype(np.int8)
    z_np = np.left_shift(x_np, y_np)
    print(z_np)

    x_ms = Tensor(x_np)
    y_ms = Tensor(y_np)
    net = NetLeftShift()
    z_ms = net(x_ms, y_ms)
    print(z_ms.asnumpy())

    assert np.allclose(z_np, z_ms.asnumpy())
