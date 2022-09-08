# Copyright 2020 Huawei Technologies Co., Ltd
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

import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.common import dtype as mstype
from mindspore.ops import operations as P
from mindspore.ops import functional as F
from mindspore.ops.functional import vmap

context.set_context(mode=context.GRAPH_MODE, device_target='CPU')


class Slice(nn.Cell):
    def __init__(self):
        super(Slice, self).__init__()
        self.slice = P.Slice()

    def construct(self, x):
        return self.slice(x, (0, 1, 0), (2, 1, 3))


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_slice():
    x = Tensor(
        np.array([[[1, -1, 1], [2, -2, 2]], [[3, -3, 3], [4, -4, 4]], [[5, -5, 5], [6, -6, 6]]]), mstype.float32)
    expect = [[[2., -2., 2.]],
              [[4., -4., 4.]]]

    slice_op = Slice()
    output = slice_op(x)
    assert (output.asnumpy() == expect).all()


class Slice2(nn.Cell):
    def __init__(self):
        super(Slice2, self).__init__()
        self.slice = P.Slice()

    def construct(self, x):
        return self.slice(x, (1, 0, 0), (1, 2, 3))


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_slice2():
    x = Tensor(np.arange(3 * 2 * 3).reshape(3, 2, 3), mstype.float32)
    expect = [[[6., 7., 8.],
               [9., 10., 11.]]]

    slice_op = Slice2()
    output = slice_op(x)
    assert (output.asnumpy() == expect).all()


def test_slice_float64():
    data = Tensor(np.array([[[1, 1, 1], [2, 2, 2]],
                            [[3, 3, 3], [4, 4, 4]],
                            [[5, 5, 5], [6, 6, 6]]]).astype(np.float64))
    slice_op = P.Slice()
    output = slice_op(data, (1, 0, 0), (1, 1, 3))
    expect = [[[3.0, 3.0, 3.0]]]
    assert (output.asnumpy() == expect).all()


class Slice3(nn.Cell):
    def __init__(self):
        super(Slice3, self).__init__()
        self.relu = nn.ReLU()

    def construct(self, x):
        return (x[..., -1], x[..., 2:1:-1], x[1:3:1, 0, ...], x[-1, 0, ...])


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_slice3():
    inputx = np.random.rand(4, 4, 4, 4).astype(np.float32)
    x = Tensor(inputx)
    slice_op = Slice3()
    output = slice_op(x)
    assert (output[0].asnumpy() == inputx[..., -1]).all()
    assert (output[1].asnumpy() == inputx[..., 2:1:-1]).all()
    assert (output[2].asnumpy() == inputx[1:3:1, 0, ...]).all()
    assert (output[3].asnumpy() == inputx[-1, 0, ...]).all()


class Slice4(nn.Cell):
    def __init__(self):
        super(Slice4, self).__init__()
        self.relu = nn.ReLU()

    def construct(self, x):
        return x[:10:1, :, 2:3:1]


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_slice4():
    inputx = np.random.rand(4, 4, 4).astype(np.float32)
    x = Tensor(inputx)
    slice_op = Slice4()
    output = slice_op(x)
    assert (output.asnumpy() == inputx[:10:1, :, 2:3:1]).all()


class Slice5(nn.Cell):
    def __init__(self, begin, size):
        super(Slice5, self).__init__()
        self.relu = nn.ReLU()
        self.slice = P.Slice()
        self.begin = begin
        self.size = size

    def construct(self, x):
        return self.slice(x, self.begin, self.size)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_slice5():
    inputx = np.arange(3 * 5 * 4).reshape(3, 5, 4).astype(np.float32)
    x = Tensor(inputx)
    begin = (0, 1, 0)
    size = (3, 4, 4)
    slice_op = Slice5(begin, size)
    output = slice_op(x)
    assert (output.asnumpy() == inputx[0:3:1, 1:5:1, 0:4:1]).all()


class Slice6(nn.Cell):
    def __init__(self):
        super(Slice6, self).__init__()
        self.relu = nn.ReLU()

    def construct(self, x):
        return (x[-10:], x[-5:10:2, :, :], x[-10:10:1, :, -10:10:1])


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_slice6():
    inputx = np.random.rand(4, 4, 4).astype(np.float32)
    x = Tensor(inputx)
    slice_op = Slice6()
    output = slice_op(x)
    assert (output[0].asnumpy() == inputx[-10:]).all()
    assert (output[1].asnumpy() == inputx[-5:10:2, :, :]).all()
    assert (output[2].asnumpy() == inputx[-10:10:1, :, -10:10:1]).all()


class StridedSlice(nn.Cell):
    def __init__(self, begin, end, stride):
        super(StridedSlice, self).__init__()
        self.begin = begin
        self.end = end
        self.stride = stride
        self.stride_slice = P.StridedSlice()

    def construct(self, x):
        return self.stride_slice(x, self.begin, self.end, self.stride)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_strided_slice_bool_type():
    input_x = Tensor([[[False, False, True], [False, True, False]], [[False, True, False], [True, False, False]],
                      [[False, True, True], [True, False, True]]], mstype.bool_)
    begin = (1, 0, 0)
    end = (2, 1, 3)
    stride = (1, 1, 1)
    slice_op = StridedSlice(begin, end, stride)
    output = slice_op(input_x)
    expected_output = np.array([False, True, False])
    assert (output.asnumpy() == expected_output).all()


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_slice_functional():
    """
    Feature: test_slice_functional
    Description: test slice functional API
    Expectation: the output is as expected
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    x = Tensor(
        np.array([[[1, -1, 1], [2, -2, 2]], [[3, -3, 3], [4, -4, 4]], [[5, -5, 5], [6, -6, 6]]]), mstype.float32)
    expect = [[[2., -2., 2.]],
              [[4., -4., 4.]]]
    output = F.slice(x, begin=(0, 1, 0), size=(2, 1, 3))
    assert (output.asnumpy() == expect).all()


def vmap_1_batch():
    def calc(x):
        return Slice()(x)

    def vmap_calc(x):
        return vmap(calc, in_axes=0)(x)

    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")

    x1 = np.array([[[1, 1, 1], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]], [[5, 5, 5], [6, 6, 6]]]).astype(np.int32)
    expect1 = [[[2., 2., 2.]],
               [[4., 4., 4.]]]
    x2 = np.array([[[1, -1, 1], [2, -2, 2]], [[3, -3, 3], [4, -4, 4]], [[5, -5, 5], [6, -6, 6]]]).astype(np.int32)
    expect2 = [[[2., -2., 2.]],
               [[4., -4., 4.]]]
    xs = np.stack([x1, x2])
    expect = np.stack([expect1, expect2])
    output = vmap_calc(Tensor(xs)).asnumpy()

    error = np.ones(shape=output.shape) * 1.0e-6
    diff = output - expect
    assert np.all(diff < error)
    assert np.all(-diff < error)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_slice_vmap():
    """
    Feature: Slice
    Description: slice vmap with 1 batch dim
    Expectation: success
    """
    vmap_1_batch()
