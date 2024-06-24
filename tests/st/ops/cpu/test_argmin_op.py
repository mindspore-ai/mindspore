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

import random
from functools import reduce
import numpy as np
import pytest

import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor, ops
from mindspore.common import dtype as mstype

context.set_context(mode=context.GRAPH_MODE, device_target="CPU")


class NetArgmin(nn.Cell):
    def __init__(self, axis=0):
        super(NetArgmin, self).__init__()
        self.argmin = ops.Argmin(axis=axis, output_type=mstype.int32)

    def construct(self, x):
        return self.argmin(x)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_argmin_1d():
    """
    Features: The ops Argmin on CPU.
    Description: Test Argmin with 1d-input.
    Expectation: No exception.
    """
    x = Tensor(np.array([1., 20., 5.]).astype(np.float32))
    output = NetArgmin(axis=0)(x)
    expect = np.array([0]).astype(np.float32)
    assert (output.asnumpy() == expect).all()


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_argmin_2d():
    """
    Features: The ops Argmin on CPU.
    Description: Test Argmin with 2d-input.
    Expectation: No exception.
    """
    x = Tensor(np.array([[1., 20., 5.],
                         [67., 8., 9.],
                         [130., 24., 15.]]).astype(np.float32))
    output = NetArgmin(axis=0)(x)
    expect = np.array([0, 1, 0]).astype(np.float32)
    assert (output.asnumpy() == expect).all()
    output = NetArgmin(axis=1)(x)
    expect = np.array([0, 1, 2]).astype(np.float32)
    assert (output.asnumpy() == expect).all()


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_argmin_high_dims():
    """
    Features: The ops Argmin on CPU.
    Description: Test Argmin with random input.
    Expectation: No exception.
    """
    for dim in range(3, 10):
        shape = np.random.randint(1, 10, size=dim)
        x = np.random.randn(reduce(lambda x, y: x * y, shape)).astype(np.float32)
        x = x.reshape(shape)

        rnd_axis = random.randint(-dim + 1, dim - 1)
        ms_output = NetArgmin(axis=rnd_axis)(Tensor(x))
        np_output = np.argmin(x, axis=rnd_axis)
        assert (ms_output.asnumpy() == np_output).all()


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_function_argmin():
    """
    Features: The function argmin on CPU.
    Description: Test function argmin with random input.
    Expectation: No exception.
    """
    for dim in range(2, 5):
        shape = np.random.randint(1, 10, size=dim)
        x = np.random.randn(reduce(lambda x, y: x * y, shape)).astype(np.float32)
        x = x.reshape(shape)

        rnd_axis = random.randint(-dim + 1, dim - 1)
        ms_output = ops.argmin(Tensor(x), axis=rnd_axis)
        np_output = np.argmin(x, axis=rnd_axis)
        assert (ms_output.asnumpy() == np_output).all()


def cal_argmin_axis_zero(x):
    return ops.Argmin(axis=0)(x)


def cal_argmin_axis_negative(x):
    return ops.Argmin(axis=-1)(x)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_argmin_vmap_axis_zero():
    """
    Features: The argmin vmap on CPU.
    Description: Test basic vmap of argmin op.
    Expectation: No exception.
    """
    x = Tensor([[5., 3., 4.], [2., 4., 3.], [3., 1., 4.]], dtype=mstype.float32)
    outputs = ops.vmap(cal_argmin_axis_zero, in_axes=0, out_axes=0)(x)
    expect = np.array([1, 0, 1]).astype(np.int32)
    assert np.allclose(outputs.asnumpy(), expect)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_argmin_vmap_basic_axis_negative():
    """
    Features: The argmin vmap on CPU.
    Description: Test basic vmap of argmin op.
    Expectation: No exception.
    """
    x = Tensor([[[5., 3., 4.], [2., 4., 3.], [3., 1., 4.]],
                [[4., 2., 1.], [3., 4., 5.], [1., 2., 3.]]], dtype=mstype.float32)
    outputs = ops.vmap(cal_argmin_axis_negative, in_axes=0, out_axes=0)(x)
    expect = np.array([[1, 0, 1], [2, 0, 0]]).astype(np.int32)
    assert np.allclose(outputs.asnumpy(), expect)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_argmin_functional():
    """
    Feature: test ops.argmin.
    Description: test ops.argmin functional api.
    Expectation: the result match with expected result.
    """
    x = Tensor([[5., 3., 4.], [2., 4., 3.], [3., 1., 4.]], mstype.int32)
    out_dim_none = ops.argmin(x, axis=None, keepdims=False)
    out_dim_0 = ops.argmin(x, axis=0, keepdims=False)
    out_dim_1 = ops.argmin(x, axis=1, keepdims=False)
    out_dim_none_keepdim = ops.argmin(x, axis=None, keepdims=True)
    out_dim_0_keepdim = ops.argmin(x, axis=0, keepdims=True)
    out_dim_1_keepdim = ops.argmin(x, axis=1, keepdims=True)

    assert out_dim_none.asnumpy() == 7
    assert np.all(out_dim_0.asnumpy() == np.array([1, 2, 1]))
    assert np.all(out_dim_1.asnumpy() == np.array([1, 0, 1]))
    assert out_dim_none_keepdim.asnumpy() == 7
    assert np.all(out_dim_0_keepdim.asnumpy() == np.array([[1, 2, 1]]))
    assert np.all(out_dim_1_keepdim.asnumpy() == np.array([[1], [0], [1]]))


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_argmin_tensor():
    """
    Feature: test tensor.argmin.
    Description: test argmin tensor api.
    Expectation: the result match with expected result.
    """
    x = Tensor([[5., 3., 4.], [2., 4., 3.], [3., 1., 4.]], mstype.int32)
    out_dim_none = x.argmin(axis=None, keepdims=False)
    out_dim_0 = x.argmin(axis=0, keepdims=False)
    out_dim_1 = x.argmin(axis=1, keepdims=False)
    out_dim_none_keepdim = x.argmin(axis=None, keepdims=True)
    out_dim_0_keepdim = x.argmin(axis=0, keepdims=True)
    out_dim_1_keepdim = x.argmin(axis=1, keepdims=True)

    assert out_dim_none.asnumpy() == 7
    assert np.all(out_dim_0.asnumpy() == np.array([1, 2, 1]))
    assert np.all(out_dim_1.asnumpy() == np.array([1, 0, 1]))
    assert out_dim_none_keepdim.asnumpy() == 7
    assert np.all(out_dim_0_keepdim.asnumpy() == np.array([[1, 2, 1]]))
    assert np.all(out_dim_1_keepdim.asnumpy() == np.array([[1], [0], [1]]))
