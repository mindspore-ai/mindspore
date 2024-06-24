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

import numpy as np
import pytest

import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.ops import operations as P
from mindspore.ops.operations import _inner_ops as inner


class ReduceMin(nn.Cell):
    def __init__(self, keep_dims):
        super(ReduceMin, self).__init__()
        self.reduce_min = P.ReduceMin(keep_dims=keep_dims)

    def construct(self, x, axis):
        return self.reduce_min(x, axis)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('dtype', [np.float16, np.float32, np.float64])
@pytest.mark.parametrize('shape, axis, keep_dims',
                         [((2, 3, 4, 4), 3, True), ((2, 3, 4, 4), 3, False), ((2, 3, 1, 4), 2, True),
                          ((2, 3, 1, 4), 2, False), ((2, 3, 4, 4), None, True), ((2, 3, 4, 4), None, False),
                          ((2, 3, 4, 4), -2, False), ((2, 3, 4, 4), (-2, -1), False), ((1, 1, 1, 1), None, True)])
def test_reduce_min(dtype, shape, axis, keep_dims):
    """
    Feature: ALL To ALL
    Description: test cases for ReduceMin
    Expectation: the result match to numpy
    """
    context.set_context(mode=context.PYNATIVE_MODE, device_target='GPU')
    x = np.random.rand(*shape).astype(dtype)
    tensor_x = Tensor(x)

    reduce_min = ReduceMin(keep_dims)
    ms_axis = axis if axis is not None else ()
    output = reduce_min(tensor_x, ms_axis)

    expect = np.min(x, axis=axis, keepdims=keep_dims)
    diff = abs(output.asnumpy() - expect)
    error = np.ones(shape=expect.shape) * 1.0e-5
    assert np.all(diff < error)
    assert output.shape == expect.shape


class ReduceMinDynamic(nn.Cell):
    def __init__(self, x, axis):
        super(ReduceMinDynamic, self).__init__()
        self.reduce_min = P.ReduceMin(False)
        self.test_dynamic = inner.GpuConvertToDynamicShape()
        self.x = x
        self.axis = axis

    def construct(self):
        dynamic_x = self.test_dynamic(self.x)
        return self.reduce_min(dynamic_x, self.axis)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('dtype', [np.float32])
@pytest.mark.parametrize('shape, axis, keep_dims',
                         [((1, 1, 1, 1), 0, False), ((2, 3, 4, 4), 0, False)])
def test_reduce_min_dynamic(dtype, shape, axis, keep_dims):
    """
    Feature: ALL To ALL
    Description: test cases for ReduceMin with dynamic shape
    Expectation: the result match to numpy
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    x = np.random.rand(*shape).astype(dtype)
    ms_axis = axis if axis is not None else ()
    net = ReduceMinDynamic(Tensor(x), ms_axis)

    expect = np.min(x, axis=axis, keepdims=keep_dims)
    output = net()

    np.testing.assert_almost_equal(output.asnumpy(), expect)
