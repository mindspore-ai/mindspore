# Copyright 2023 Huawei Technologies Co., Ltd
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

import mindspore as ms
import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.ops.operations.array_ops import RightShift, LeftShift


class NetRightShift(nn.Cell):

    def __init__(self):
        super(NetRightShift, self).__init__()
        self.rightshift = RightShift()

    def construct(self, x, y):
        return self.rightshift(x, y)


class NetLeftShift(nn.Cell):
    def __init__(self):
        super(NetLeftShift, self).__init__()
        self.leftshift = LeftShift()

    def construct(self, x, y):
        return self.leftshift(x, y)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_rightshift_2d(mode):
    """
    Feature: RightShift Ascend TEST.
    Description: 2d input test case for RightShift
    Expectation: the result match to numpy
    """
    context.set_context(mode=mode, device_target="Ascend")
    x_np = (np.array([[-1, -5, -3], [-14, 64, 0]])).astype(np.int8)
    y_np = (np.array([[5, 0, 7], [11, 1, 0]])).astype(np.int8)
    z_np = np.right_shift(x_np, y_np)

    x_ms = Tensor(x_np)
    y_ms = Tensor(y_np)
    net = NetRightShift()
    z_ms = net(x_ms, y_ms)

    assert np.allclose(z_np, z_ms.asnumpy())


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_rightshift_dyn(mode):
    """
    Feature: test RightShift ops in Ascend.
    Description: test the ops in dynamic shape.
    Expectation: expect correct shape result.
    """
    context.set_context(mode=mode, device_target="Ascend")

    net = NetRightShift()
    x_dyn = Tensor(shape=[None, None], dtype=ms.int8)
    y_dyn = Tensor(shape=[None], dtype=ms.int8)
    net.set_inputs(x_dyn, y_dyn)

    x = Tensor([[1, 2, 3], [1, 2, 3]], dtype=ms.int8)
    y = Tensor([1, 1, 1], dtype=ms.int8)
    out = net(x, y)

    expect_shape = (2, 3)
    assert out.asnumpy().shape == expect_shape


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_leftshift_2d(mode):
    """
    Feature: LeftShift Ascend TEST.
    Description: 2d input test case for LeftShift
    Expectation: the result match to numpy
    """
    context.set_context(mode=mode, device_target="Ascend")
    x_np = (np.array([[-1, -5, -3], [-14, 64, 0]])).astype(np.int8)
    y_np = (np.array([[5, 0, 7], [11, 1, 0]])).astype(np.int8)
    z_np = np.left_shift(x_np, y_np)

    x_ms = Tensor(x_np)
    y_ms = Tensor(y_np)
    net = NetLeftShift()
    z_ms = net(x_ms, y_ms)

    assert np.allclose(z_np, z_ms.asnumpy())


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_leftshift_dyn(mode):
    """
    Feature: test LeftShift ops in Ascend.
    Description: test the ops in dynamic shape.
    Expectation: expect correct shape result.
    """
    context.set_context(mode=mode, device_target="Ascend")

    net = NetLeftShift()
    x_dyn = Tensor(shape=[None, None], dtype=ms.int8)
    y_dyn = Tensor(shape=[None], dtype=ms.int8)
    net.set_inputs(x_dyn, y_dyn)

    x = Tensor([[1, 2, 3], [1, 2, 3]], dtype=ms.int8)
    y = Tensor([1, 1, 1], dtype=ms.int8)
    out = net(x, y)

    expect_shape = (2, 3)
    assert out.asnumpy().shape == expect_shape
