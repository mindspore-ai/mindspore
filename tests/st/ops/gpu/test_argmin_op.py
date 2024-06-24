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
from tests.mark_utils import arg_mark

import random
from functools import reduce
import numpy as np
import pytest

import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.common import dtype as mstype
import mindspore.ops as ops


class NetArgmin(nn.Cell):
    def __init__(self, axis=0):
        super(NetArgmin, self).__init__()
        self.argmin = ops.Argmin(axis, output_type=mstype.int32)

    def construct(self, x):
        return self.argmin(x)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_argmin_1d():
    """
    Feature: None
    Description: test argmin 1d
    Expectation: just test
    """

    for mode in [context.PYNATIVE_MODE, context.GRAPH_MODE]:
        context.set_context(mode=mode, device_target="GPU")

        x = Tensor(np.array([1., 20., 5.]).astype(np.float32))
        argmin = NetArgmin(axis=0)
        output = argmin(x)
        expect = np.array([0]).astype(np.float32)
        assert (output.asnumpy() == expect).all()


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_argmin_2d():
    """
    Feature: None
    Description: test argmin 2d
    Expectation: just test
    """

    for mode in [context.PYNATIVE_MODE, context.GRAPH_MODE]:
        context.set_context(mode=mode, device_target="GPU")

    x = Tensor(np.array([[1., 20., 5.],
                         [67., 8., 9.],
                         [130., 24., 15.],
                         [0.3, -0.4, -15.]]).astype(np.float32))
    argmin_axis_0 = NetArgmin(axis=0)
    output = argmin_axis_0(x)
    expect = np.array([3, 3, 3]).astype(np.int32)
    assert (output.asnumpy() == expect).all()

    argmin_axis_1 = NetArgmin(axis=1)
    output = argmin_axis_1(x)
    expect = np.array([0, 1, 2, 2]).astype(np.int32)
    assert (output.asnumpy() == expect).all()


@arg_mark(plat_marks=['platform_gpu'], level_mark='level2', card_mark='onecard', essential_mark='unessential')
def test_argmin_high_dims():
    """
    Feature: None
    Description: test argmin high dim
    Expectation: just test
    """

    for mode in [context.PYNATIVE_MODE, context.GRAPH_MODE]:
        context.set_context(mode=mode, device_target="GPU")
        for dim in range(3, 10):
            shape = np.random.randint(1, 10, size=dim)
            x = np.random.randn(reduce(lambda x, y: x * y, shape)).astype(np.float32)
            x = x.reshape(shape)

            rnd_axis = random.randint(-dim + 1, dim - 1)
            argmin = NetArgmin(axis=rnd_axis)
            ms_output = argmin(Tensor(x))
            np_output = np.argmin(x, axis=rnd_axis)
            assert (ms_output.asnumpy() == np_output).all()


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_argmin_functional():
    """
    Feature: test ops.argmin.
    Description: test ops.argmin functional api.
    Expectation: the result match with expected result.
    """
    x = Tensor([[5., 3., 4.], [2., 4., 3.], [3., 1., 4.]], mstype.float32)
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


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_argmin_tensor():
    """
    Feature: test tensor.argmin.
    Description: test argmin tensor api.
    Expectation: the result match with expected result.
    """
    x = Tensor([[5., 3., 4.], [2., 4., 3.], [3., 1., 4.]], mstype.float32)
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
