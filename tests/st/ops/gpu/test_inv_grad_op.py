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
from mindspore.ops.operations import _grad_ops as G
from mindspore.ops import functional as F
from mindspore.ops.operations import _inner_ops as inner


class NetInvGrad(nn.Cell):
    def __init__(self):
        super(NetInvGrad, self).__init__()
        self.grad = G.InvGrad()

    def construct(self, y, dy):
        return self.grad(y, dy)


class InvGradDynamicShapeNet(nn.Cell):
    def __init__(self):
        super(InvGradDynamicShapeNet, self).__init__()
        self.grad = G.InvGrad()
        self.test_dynamic = inner.GpuConvertToDynamicShape()

    def construct(self, y, dy):
        y = self.test_dynamic(y)
        dy = self.test_dynamic(dy)
        return self.grad(y, dy)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
@pytest.mark.parametrize('dtype', [np.float32, np.float64, np.complex64, np.complex128])
def test_inv_grad_float_and_complex(mode, dtype):
    """
    Feature: ALL To ALL
    Description: test cases for InvGrad for float32 and complex
    Expectation: the result match to numpy
    """
    context.set_context(mode=mode, device_target="GPU")
    y = Tensor(np.array([[[[-1, 1, 12],
                           [5, 34, 6],
                           [10, 2, -1]]]]).astype(dtype))
    dy = Tensor(np.array([[[[29, 1, 55],
                            [2.2, 63, 2],
                            [3, 3, 12]]]]).astype(dtype))
    expect = np.array([[[[-29, -1, -7920],
                         [-55, -72828, -72],
                         [-300, -12, -12]]]]).astype(dtype)
    net = NetInvGrad()
    output = net(y, dy)
    np.testing.assert_array_almost_equal(output.asnumpy(), expect)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_inv_grad_float16(mode):
    """
    Feature: ALL To ALL
    Description: test cases for InvGrad for float16
    Expectation: the result match to numpy
    """
    context.set_context(mode=mode, device_target="GPU")
    y = Tensor(np.array([[0.01, 0.2, 0.22],
                         [10.002, 2, -1]]).astype(np.float16))
    dy = Tensor(np.array([[34, 1, 55],
                          [3, 3, 63]]).astype(np.float16))
    expect = np.array([[-0.0034, -0.03998, -2.662],
                       [-300, -12, -63]]).astype(np.float16)
    net = NetInvGrad()
    output = net(y, dy)
    np.testing.assert_array_almost_equal(output.asnumpy(), expect)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_inv_grad_vmap(mode):
    """
    Feature: test inv_grad vmap feature.
    Description: test inv_grad vmap feature.
    Expectation: Success.
    """
    context.set_context(mode=mode, device_target="GPU")
    y = Tensor(np.array([[-1, 1, 12],
                         [5, 34, 6],
                         [10, 2, -1]]).astype(np.float32))
    dout = Tensor(np.array([[29, 1, 55],
                            [2.2, 63, 2],
                            [3, 3, 12]]).astype(np.float32))
    # Case 1
    output = F.vmap(NetInvGrad(), (0, 0), 0)(y, dout)
    expect_output = np.array([[-29, -1, -7920],
                              [-55, -72828, -72],
                              [-300, -12, -12]]).astype(np.float32)
    np.testing.assert_almost_equal(output.asnumpy(), expect_output)

    # Case 2
    output = F.vmap(NetInvGrad(), (0, 1), 0)(y, dout)
    expect_output = np.array([[-29, -2.2, -432],
                              [-25, -72828, -108],
                              [-5500, -8, -12]]).astype(np.float32)
    np.testing.assert_almost_equal(output.asnumpy(), expect_output)

    # Case 3
    output = F.vmap(NetInvGrad(), (0, 0), 1)(y, dout)
    expect_output = np.array([[-29, -55, -300],
                              [-1, -72828, -12],
                              [-7920, -72, -12]]).astype(np.float32)
    np.testing.assert_almost_equal(output.asnumpy(), expect_output)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_inv_grad_dynamic_shape(mode):
    """
    Feature: test inv_grad dynamic_shape feature.
    Description: test inv_grad dynamic_shape feature.
    Expectation: Success.
    """
    context.set_context(mode=mode, device_target="GPU")
    y = Tensor(np.array([[-1, 1, 12],
                         [5, 34, 6],
                         [10, 2, -1]]).astype(np.float32))
    dout = Tensor(np.array([[29, 1, 55],
                            [2.2, 63, 2],
                            [3, 3, 12]]).astype(np.float32))
    output = InvGradDynamicShapeNet()(y, dout)
    expect_output = np.array([[-29, -1, -7920],
                              [-55, -72828, -72],
                              [-300, -12, -12]]).astype(np.float32)
    np.testing.assert_almost_equal(output.asnumpy(), expect_output)
