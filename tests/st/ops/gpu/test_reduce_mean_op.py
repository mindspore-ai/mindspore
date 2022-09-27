# Copyright 2019 Huawei Technologies Co., Ltd
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
from mindspore import dtype as mstype
from mindspore.ops import operations as P
from mindspore.ops.operations import _inner_ops as inner


class ReduceMean(nn.Cell):
    def __init__(self, keep_dims):
        super(ReduceMean, self).__init__()
        self.reduce_mean = P.ReduceMean(keep_dims=keep_dims)

    def construct(self, x, axis):
        return self.reduce_mean(x, axis)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('dtype', [np.float16, np.float32, np.float64])
@pytest.mark.parametrize('shape, axis, keep_dims',
                         [((2, 3, 4, 4), 3, True), ((2, 3, 4, 4), 3, False), ((2, 3, 1, 4), 2, True),
                          ((2, 3, 1, 4), 2, False), ((2, 3, 4, 1), 3, True), ((2, 3, 4, 1), 3, False),
                          ((2, 3, 4, 4), (1, 2), False), ((2, 3, 4, 4), (1, 2), True), ((2, 1, 1, 4), (1, 2), True),
                          ((2, 1, 1, 4), (1, 2), False), ((2, 3, 4, 4), (0, 1, 2, 3), False),
                          ((1, 1, 1, 1), (0, 1, 2, 3), False), ((2, 3, 4, 4, 5, 6), -2, False),
                          ((2, 3, 4, 4), (-2, -1), True), ((1, 1, 1, 1), (), True)])
def test_reduce_mean(dtype, shape, axis, keep_dims):
    """
    Feature: ALL To ALL
    Description: test cases for ReduceMean
    Expectation: the result match to numpy
    """
    context.set_context(mode=context.PYNATIVE_MODE, device_target='GPU')
    x = np.random.rand(*shape).astype(dtype)
    tensor_x = Tensor(x)

    reduce_mean = ReduceMean(keep_dims)
    output = reduce_mean(tensor_x, axis)

    expect = np.mean(x, axis=axis, keepdims=keep_dims)
    diff = abs(output.asnumpy() - expect)
    error = np.ones(shape=expect.shape) * 1.0e-5
    assert np.all(diff < error)
    assert output.shape == expect.shape


class ReduceMeanDynamic(nn.Cell):
    def __init__(self, x, axis, keepdims=False):
        super(ReduceMeanDynamic, self).__init__()
        self.test_dynamic = inner.GpuConvertToDynamicShape()
        self.reduce_mean = P.ReduceMean(keep_dims=keepdims)
        self.x = x
        self.axis = axis

    def construct(self):
        dynamic_x = self.test_dynamic(self.x)
        output = self.reduce_mean(dynamic_x, self.axis)
        return output


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('dtype', [np.float32])
@pytest.mark.parametrize('shape, axis, keep_dims',
                         [((2, 3, 4, 4), 3, True), ((1, 1, 1, 1), (), True), ((2, 3, 4, 4, 5, 6), -2, False)])
def test_dynamic_reduce_mean(dtype, shape, axis, keep_dims):
    """
    Feature: ALL To ALL
    Description: test cases for ReduceMean with dynamic shape
    Expectation: the result match to numpy
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    x = np.random.rand(*shape).astype(dtype)
    tensor_x = Tensor(x)
    net = ReduceMeanDynamic(tensor_x, axis, keepdims=keep_dims)
    output = net()

    expect = np.mean(x, axis=axis, keepdims=keep_dims)
    diff = abs(output.asnumpy() - expect)
    error = np.ones(shape=expect.shape) * 1.0e-5
    assert np.all(diff < error)
    assert output.shape == expect.shape


class ReduceMeanNegativeNet(nn.Cell):
    def __init__(self):
        super().__init__()
        self.mean0 = P.ReduceMean(True)
        self.mean1 = P.ReduceMean(False)

    def construct(self, x):
        t = self.mean0(x, ())
        return self.mean1(t, (-1,))


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_reduce_mean_negative():
    """
    Feature: ALL To ALL
    Description: test cases for ReduceMean with negative axis.
    Expectation: the result match expectation
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    x = Tensor([[[1, 2, 3,], [3, 2, 1]]], mstype.float32)
    net = ReduceMeanNegativeNet()
    out = net(x)
    assert out.shape == (1, 1)
