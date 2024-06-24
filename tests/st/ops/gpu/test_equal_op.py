# Copyright 2020-2022 Huawei Technologies Co., Ltd
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
from mindspore.common.tensor import Tensor
from mindspore.nn import Cell
from mindspore.ops import operations as P
from mindspore.ops.operations import _inner_ops as inner
from mindspore.common import dtype as mstype


class NetEqual(Cell):
    def __init__(self):
        super(NetEqual, self).__init__()
        self.equal = P.Equal()

    def construct(self, x, y):
        return self.equal(x, y)


class NetEqualDynamic(Cell):
    def __init__(self):
        super(NetEqualDynamic, self).__init__()
        self.conv = inner.GpuConvertToDynamicShape()
        self.equal = P.Equal()

    def construct(self, x, y):
        x_conv = self.conv(x)
        y_conv = self.conv(y)
        return self.equal(x_conv, y_conv)


class NetNotEqual(Cell):
    def __init__(self):
        super(NetNotEqual, self).__init__()
        self.not_equal = P.NotEqual()

    def construct(self, x, y):
        return self.not_equal(x, y)


class NetGreaterEqual(Cell):
    def __init__(self):
        super(NetGreaterEqual, self).__init__()
        self.greater_equal = P.GreaterEqual()

    def construct(self, x, y):
        return self.greater_equal(x, y)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_equal():
    x0_np = np.arange(24).reshape((4, 3, 2)).astype(np.float32)
    x0 = Tensor(x0_np)
    y0_np = np.arange(24).reshape((4, 3, 2)).astype(np.float32)
    y0 = Tensor(y0_np)
    expect0 = np.equal(x0_np, y0_np)
    x1_np = np.array([0, 1, 3]).astype(np.float32)
    x1 = Tensor(x1_np)
    y1_np = np.array([0]).astype(np.float32)
    y1 = Tensor(y1_np)
    expect1 = np.equal(x1_np, y1_np)
    x2_np = np.array([0, 1, 3]).astype(np.int32)
    x2 = Tensor(x2_np)
    y2_np = np.array([0]).astype(np.int32)
    y2 = Tensor(y2_np)
    expect2 = np.equal(x2_np, y2_np)
    x3_np = np.array([0, 1, 3]).astype(np.int16)
    x3 = Tensor(x3_np)
    y3_np = np.array([0, 1, -3]).astype(np.int16)
    y3 = Tensor(y3_np)
    expect3 = np.equal(x3_np, y3_np)
    x4_np = np.array([0, 1, 4]).astype(np.uint8)
    x4 = Tensor(x4_np)
    y4_np = np.array([0, 1, 3]).astype(np.uint8)
    y4 = Tensor(y4_np)
    expect4 = np.equal(x4_np, y4_np)
    x5_np = np.array([True, False, True]).astype(bool)
    x5 = Tensor(x5_np)
    y5_np = np.array([True, False, False]).astype(bool)
    y5 = Tensor(y5_np)
    expect5 = np.equal(x5_np, y5_np)
    x6_np = np.array([0, 1, 4]).astype(np.int8)
    x6 = Tensor(x6_np)
    y6_np = np.array([0, 1, 3]).astype(np.int8)
    y6 = Tensor(y6_np)
    expect6 = np.equal(x6_np, y6_np)
    x7_np = np.array([0, 1, 4]).astype(np.int64)
    x7 = Tensor(x7_np)
    y7_np = np.array([0, 1, 3]).astype(np.int64)
    y7 = Tensor(y7_np)
    expect7 = np.equal(x7_np, y7_np)
    x8_np = np.array([0, 1, 4]).astype(np.float16)
    x8 = Tensor(x8_np)
    y8_np = np.array([0, 1, 3]).astype(np.float16)
    y8 = Tensor(y8_np)
    expect8 = np.equal(x8_np, y8_np)
    x9_np = np.array([0, 1, 4]).astype(np.float64)
    x9 = Tensor(x9_np)
    y9_np = np.array([0, 1, 3]).astype(np.float64)
    y9 = Tensor(y9_np)
    expect9 = np.equal(x9_np, y9_np)

    x = [x0, x1, x2, x3, x4, x5, x6, x7, x8, x9]
    y = [y0, y1, y2, y3, y4, y5, y6, y7, y8, y9]
    expect = [expect0, expect1, expect2, expect3, expect4, expect5, expect6, expect7, expect8, expect9]

    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
    equal = NetEqual()
    for i, xi in enumerate(x):
        output = equal(xi, y[i])
        assert np.all(output.asnumpy() == expect[i])
        assert output.shape == expect[i].shape
        print('test [%d/%d] passed!' % (i, len(x)))

    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    equal = NetEqual()
    for i, xi in enumerate(x):
        output = equal(xi, y[i])
        assert np.all(output.asnumpy() == expect[i])
        assert output.shape == expect[i].shape
        print('test [%d/%d] passed!' % (i, len(x)))


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_notequal():
    x0 = Tensor(np.array([[1.2, 1], [1, 0]]).astype(np.float32))
    y0 = Tensor(np.array([[1, 2]]).astype(np.float32))
    expect0 = np.array([[True, True], [False, True]])
    x1 = Tensor(np.array([[2, 1], [1, 1]]).astype(np.int16))
    y1 = Tensor(np.array([[1, 2]]).astype(np.int16))
    expect1 = np.array([[True, True], [False, True]])
    x2 = Tensor(np.array([[2, 1], [1, 2]]).astype(np.uint8))
    y2 = Tensor(np.array([[1, 2]]).astype(np.uint8))
    expect2 = np.array([[True, True], [False, False]])
    x3 = Tensor(np.array([[False, True], [True, False]]).astype(bool))
    y3 = Tensor(np.array([[True, False]]).astype(bool))
    expect3 = np.array([[True, True], [False, False]])
    x4 = Tensor(np.array([[1.2, 1], [1, 0]]).astype(np.float16))
    y4 = Tensor(np.array([[1, 2]]).astype(np.float16))
    expect4 = np.array([[True, True], [False, True]])
    x5 = Tensor(np.array([[2, 1], [1, 0]]).astype(np.int64))
    y5 = Tensor(np.array([[1, 2]]).astype(np.int64))
    expect5 = np.array([[True, True], [False, True]])
    x6 = Tensor(np.array([[2, 1], [1, 0]]).astype(np.int32))
    y6 = Tensor(np.array([[1, 2], [1, 2]]).astype(np.int32))
    expect6 = np.array([[True, True], [False, True]])

    x = [x0, x1, x2, x3, x4, x5, x6]
    y = [y0, y1, y2, y3, y4, y5, y6]
    expect = [expect0, expect1, expect2, expect3, expect4, expect5, expect6]

    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
    notequal = NetNotEqual()
    for i, xi in enumerate(x):
        output = notequal(xi, y[i])
        assert np.all(output.asnumpy() == expect[i])
        assert output.shape == expect[i].shape
        print('test [%d/%d] passed!' % (i, len(x)))

    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    notequal = NetNotEqual()
    for i, xi in enumerate(x):
        output = notequal(xi, y[i])
        assert np.all(output.asnumpy() == expect[i])
        assert output.shape == expect[i].shape
        print('test [%d/%d] passed!' % (i, len(x)))


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_greaterqual():
    x0 = Tensor(np.array([[1.2, 1], [1, 0]]).astype(np.float32))
    y0 = Tensor(np.array([[1, 2], [1, 2]]).astype(np.float32))
    expect0 = np.array([[True, False], [True, False]])
    x1 = Tensor(np.array([[2, 1], [1, 1]]).astype(np.int16))
    y1 = Tensor(np.array([[1, 2]]).astype(np.int16))
    expect1 = np.array([[True, False], [True, False]])
    x2 = Tensor(np.array([[2, 1], [1, 2]]).astype(np.uint8))
    y2 = Tensor(np.array([[1, 2]]).astype(np.uint8))
    expect2 = np.array([[True, False], [True, True]])

    x3 = Tensor(np.array([[2, 1], [1, 2]]).astype(np.float64))
    y3 = Tensor(np.array([[1, 2]]).astype(np.float64))
    expect3 = np.array([[True, False], [True, True]])
    x4 = Tensor(np.array([[2, 1], [1, 2]]).astype(np.float16))
    y4 = Tensor(np.array([[1, 2]]).astype(np.float16))
    expect4 = np.array([[True, False], [True, True]])
    x5 = Tensor(np.array([[2, 1], [1, 1]]).astype(np.int64))
    y5 = Tensor(np.array([[1, 2]]).astype(np.int64))
    expect5 = np.array([[True, False], [True, False]])
    x6 = Tensor(np.array([[2, 1], [1, 1]]).astype(np.int32))
    y6 = Tensor(np.array([[1, 2]]).astype(np.int32))
    expect6 = np.array([[True, False], [True, False]])
    x7 = Tensor(np.array([[2, 1], [1, 1]]).astype(np.int8))
    y7 = Tensor(np.array([[1, 2]]).astype(np.int8))
    expect7 = np.array([[True, False], [True, False]])

    x = [x0, x1, x2, x3, x4, x5, x6, x7]
    y = [y0, y1, y2, y3, y4, y5, y6, y7]
    expect = [expect0, expect1, expect2, expect3, expect4, expect5, expect6, expect7]

    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
    gequal = NetGreaterEqual()
    for i, xi in enumerate(x):
        output = gequal(xi, y[i])
        assert np.all(output.asnumpy() == expect[i])
        assert output.shape == expect[i].shape
        print('test [%d/%d] passed!' % (i, len(x)))

    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    gequal = NetGreaterEqual()
    for i, xi in enumerate(x):
        output = gequal(xi, y[i])
        assert np.all(output.asnumpy() == expect[i])
        assert output.shape == expect[i].shape
        print('test [%d/%d] passed!' % (i, len(x)))


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_equal_dynamic_shape():
    x0_np = np.arange(24).reshape((4, 3, 2)).astype(np.float32)
    x0 = Tensor(x0_np)
    y0_np = np.arange(24).reshape((4, 3, 2)).astype(np.float32)
    y0 = Tensor(y0_np)
    expect0 = np.equal(x0_np, y0_np)

    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    equal = NetEqualDynamic()
    output0 = equal(x0, y0)
    assert np.all(output0.asnumpy() == expect0)
    assert output0.shape == expect0.shape


def test_equal_tensor_api():
    """
    Feature: test equal tensor API.
    Description: testcase for equal tensor API.
    Expectation: the result match with expected result.
    """
    x = Tensor(np.array([1, 2, 3]), mstype.int32)
    y = Tensor(np.array([1, 2, 4]), mstype.int32)
    output = x.equal(y)
    expected = np.array([True, True, False])
    np.testing.assert_array_equal(output.asnumpy(), expected)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_equal_tensor_modes():
    """
    Feature: test equal tensor API in PyNative and Graph modes.
    Description: test case for equal tensor API.
    Expectation: the result match with expected result.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    test_equal_tensor_api()
    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
    test_equal_tensor_api()
