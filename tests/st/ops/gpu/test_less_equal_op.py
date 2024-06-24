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
from mindspore.ops import functional as F
from mindspore.common import dtype as mstype


class Net(Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.lessequal = P.LessEqual()

    def construct(self, x, y):
        return self.lessequal(x, y)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_lessequal():
    x = Tensor(np.array([[1, 2, 3]]).astype(np.float32))
    y = Tensor(np.array([[2, 2, 2]]).astype(np.float32))
    expect = np.array([[True, True, False]])
    x1 = Tensor(np.array([[1, 2, 3]]).astype(np.int16))
    y1 = Tensor(np.array([[2]]).astype(np.int16))
    expect1 = np.array([[True, True, False]])
    x2 = Tensor(np.array([[1, 2, 3]]).astype(np.uint8))
    y2 = Tensor(np.array([[2]]).astype(np.uint8))
    expect2 = np.array([[True, True, False]])
    x3 = Tensor(np.array([[1, 2, 3]]).astype(np.float64))
    y3 = Tensor(np.array([[2]]).astype(np.float64))
    expect3 = np.array([[True, True, False]])
    x4 = Tensor(np.array([[1, 2, 3]]).astype(np.float16))
    y4 = Tensor(np.array([[2]]).astype(np.float16))
    expect4 = np.array([[True, True, False]])
    x5 = Tensor(np.array([[1, 2, 3]]).astype(np.int64))
    y5 = Tensor(np.array([[2]]).astype(np.int64))
    expect5 = np.array([[True, True, False]])
    x6 = Tensor(np.array([[1, 2, 3]]).astype(np.int32))
    y6 = Tensor(np.array([[2, 2, 2]]).astype(np.int32))
    expect6 = np.array([[True, True, False]])
    x7 = Tensor(np.array([[1, 2, 3]]).astype(np.int8))
    y7 = Tensor(np.array([[2]]).astype(np.int8))
    expect7 = np.array([[True, True, False]])
    x8 = Tensor(np.array([[6.67054e-10, 6.67054e-10]]).astype(np.float32))
    y8 = Tensor(np.array([[0, 6.67054e-10]]).astype(np.float32))
    expect8 = np.array([[False, True]])

    x = [x, x1, x2, x3, x4, x5, x6, x7, x8]
    y = [y, y1, y2, y3, y4, y5, y6, y7, y8]
    expect = [expect, expect1, expect2, expect3, expect4, expect5, expect6, expect7, expect8]

    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
    lessequal = Net()
    for i, xi in enumerate(x):
        output = lessequal(xi, y[i])
        assert np.all(output.asnumpy() == expect[i])
        assert output.shape == expect[i].shape
        print('test [%d/%d] passed!' % (i, len(x)))

    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    lessequal = Net()
    for i, xi in enumerate(x):
        output = lessequal(xi, y[i])
        assert np.all(output.asnumpy() == expect[i])
        assert output.shape == expect[i].shape
        print('test [%d/%d] passed!' % (i, len(x)))


def test_less_equal_functional_api():
    """
    Feature: test less_equal functional API.
    Description: test less_equal functional API.
    Expectation: the result match with expected result.
    """
    x = Tensor(np.array([1, 2, 3]), mstype.int32)
    other = Tensor(np.array([1, 1, 4]), mstype.int32)
    output = F.less_equal(x, other)
    expected = np.array([True, False, True])
    np.testing.assert_array_equal(output.asnumpy(), expected)


def test_less_equal_tensor_api():
    """
    Feature: test less_equal tensor API.
    Description: test case for less_equal tensor API.
    Expectation: the result match with expected result.
    """
    x = Tensor(np.array([1, 2, 3]), mstype.int32)
    other = Tensor(np.array([1, 1, 4]), mstype.int32)
    output = x.less_equal(other)
    expected = np.array([True, False, True])
    np.testing.assert_array_equal(output.asnumpy(), expected)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_less_equal_functional_tensor_modes():
    """
    Feature: test less_equal functional and tensor APIs in PyNative and Graph modes.
    Description: test case for erfiless_equalnv functional API.
    Expectation: the result match with expected result.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    test_less_equal_functional_api()
    test_less_equal_tensor_api()
    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
    test_less_equal_functional_api()
    test_less_equal_tensor_api()
