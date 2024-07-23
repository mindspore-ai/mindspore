# Copyright 2019-2022 Huawei Technologies Co., Ltd
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

from functools import reduce
import numpy as np
import pytest

from mindspore import Tensor
import mindspore.context as context
import mindspore.nn as nn
import mindspore.ops.operations as P
from mindspore.ops import composite as C
from mindspore.ops.functional import vmap
from mindspore.ops import functional as F

context.set_context(mode=context.GRAPH_MODE, device_target="GPU")


class AvgPool(nn.Cell):
    def __init__(self, dim, kernel_size, strides, pad_mode, count_include_pad=False):
        super(AvgPool, self).__init__()
        if dim == 2:
            self.avgpool = P.AvgPool(kernel_size=kernel_size, strides=strides, pad_mode=pad_mode)
        else:
            self.avgpool = P.AvgPool3D(kernel_size=kernel_size, strides=strides, pad_mode=pad_mode,
                                       count_include_pad=count_include_pad)

    def construct(self, x):
        return self.avgpool(x)


class AvgPoolGrad(nn.Cell):
    def __init__(self, forward):
        super(AvgPoolGrad, self).__init__()
        self.forward = forward
        self.grad = C.GradOperation(get_all=True)

    def construct(self, x):
        return self.grad(self.forward)(x)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_avgpool2d_valid():
    """
    Feature: test avgpool2d op.
    Description: including forward and backward.
    Expectation: expect correct forward and backward result.
    """
    x = Tensor(np.array([[[[10, 1, 2, 3, -4, -5],
                           [6, 7, 8, 9, -10, -11],
                           [12, 13, 24, -15, -16, -17],
                           [18, 19, 20, 21, 22, 23],
                           [32, 25, 26, 27, 28, 40],
                           [30, 31, 35, 33, 34, 35]]]]).astype(np.float32))
    avgpool = AvgPool(dim=2, kernel_size=2, strides=2, pad_mode="VALID")
    actual_output = avgpool(x)
    expect_output = np.array([[[[6, 5.5, -7.5],
                                [15.5, 12.5, 3],
                                [29.5, 30.25, 34.25]]]]).astype(np.float32)
    assert np.allclose(actual_output.asnumpy(), expect_output)

    avgpool_grad = AvgPoolGrad(avgpool)
    actual_grad = avgpool_grad(x)
    expect_grad = np.array([[[[0.25, 0.25, 0.25, 0.25, 0.25, 0.25],
                              [0.25, 0.25, 0.25, 0.25, 0.25, 0.25],
                              [0.25, 0.25, 0.25, 0.25, 0.25, 0.25],
                              [0.25, 0.25, 0.25, 0.25, 0.25, 0.25],
                              [0.25, 0.25, 0.25, 0.25, 0.25, 0.25],
                              [0.25, 0.25, 0.25, 0.25, 0.25, 0.25]]]]).astype(np.float32)
    assert np.allclose(actual_grad[0].asnumpy(), expect_grad)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_avgpool2d_same():
    """
    Feature: test avgpool2d op.
    Description: including forward and backward.
    Expectation: expect correct forward and backward result.
    """
    x = Tensor(np.array([[[[0, 1, 2, 3, -4, -5],
                           [6, 7, 8, 9, -10, -11],
                           [12, 13, 14, -15, -16, -17],
                           [18, 19, 20, 21, 22, 23],
                           [24, 25, 26, 27, 28, 29],
                           [30, 31, 32, 33, 34, 35]]]]).astype(np.float32))
    avgpool = AvgPool(dim=2, kernel_size=3, strides=2, pad_mode="SAME")
    actual_output = avgpool(x)
    expect_output = np.array([[[[7., -1., -10.5],
                                [19., 14.111111, 11.5],
                                [28., 30., 31.5]]]]).astype(np.float32)
    assert np.allclose(actual_output.asnumpy(), expect_output)

    avgpool_grad = AvgPoolGrad(avgpool)
    actual_grad = avgpool_grad(x)
    expect_grad = np.array([[[[0.11111111, 0.11111111, 0.22222222, 0.11111111, 0.27777779, 0.16666667],
                              [0.11111111, 0.11111111, 0.22222222, 0.11111111, 0.27777779, 0.16666667],
                              [0.22222222, 0.22222222, 0.44444448, 0.22222222, 0.55555558, 0.33333334],
                              [0.11111111, 0.11111111, 0.22222222, 0.11111111, 0.27777779, 0.16666667],
                              [0.27777779, 0.27777779, 0.55555558, 0.27777779, 0.69444447, 0.41666668],
                              [0.16666667, 0.16666667, 0.33333334, 0.16666667, 0.41666668, 0.25]]]]).astype(np.float32)
    assert np.allclose(actual_grad[0].asnumpy(), expect_grad)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_avgpool3d_1():
    """
    Feature: test avgpool3d op.
    Description: including forward and backward.
    Expectation: expect correct forward and backward result.
    """
    x_shape = (1, 3, 2, 3, 4)
    x = Tensor(np.arange(reduce(lambda x, y: x * y, x_shape))).reshape(x_shape).astype(np.float32)
    avgpool = AvgPool(dim=3, kernel_size=(2, 2, 3), strides=1, pad_mode='VALID')
    actual_output = avgpool(x)
    expect_output = np.array([[[[[9., 10.],
                                 [13., 14.]]],
                               [[[33., 34.],
                                 [37., 38.]]],
                               [[[57., 58.],
                                 [61., 62.]]]]])
    assert np.allclose(actual_output.asnumpy(), expect_output)

    avgpool_grad = AvgPoolGrad(avgpool)
    actual_grad = avgpool_grad(x)
    expect_grad = np.array([[[[[0.08333333, 0.16666667, 0.16666667, 0.08333333],
                               [0.16666667, 0.33333334, 0.33333334, 0.16666667],
                               [0.08333333, 0.16666667, 0.16666667, 0.08333333]],
                              [[0.08333333, 0.16666667, 0.16666667, 0.08333333],
                               [0.16666667, 0.33333334, 0.33333334, 0.16666667],
                               [0.08333333, 0.16666667, 0.16666667, 0.08333333]]],
                             [[[0.08333333, 0.16666667, 0.16666667, 0.08333333],
                               [0.16666667, 0.33333334, 0.33333334, 0.16666667],
                               [0.08333333, 0.16666667, 0.16666667, 0.08333333]],
                              [[0.08333333, 0.16666667, 0.16666667, 0.08333333],
                               [0.16666667, 0.33333334, 0.33333334, 0.16666667],
                               [0.08333333, 0.16666667, 0.16666667, 0.08333333]]],
                             [[[0.08333333, 0.16666667, 0.16666667, 0.08333333],
                               [0.16666667, 0.33333334, 0.33333334, 0.16666667],
                               [0.08333333, 0.16666667, 0.16666667, 0.08333333]],
                              [[0.08333333, 0.16666667, 0.16666667, 0.08333333],
                               [0.16666667, 0.33333334, 0.33333334, 0.16666667],
                               [0.08333333, 0.16666667, 0.16666667, 0.08333333]]]]]).astype(np.float32)


    assert np.allclose(actual_grad[0].asnumpy(), expect_grad)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_avgpool3d_2():
    """
    Feature: test avgpool3d op.
    Description: including forward and backward.
    Expectation: expect correct forward and backward result.
    """
    x_shape = (1, 3, 2, 3, 4)
    x = Tensor(np.arange(reduce(lambda x, y: x * y, x_shape))).reshape(x_shape).astype(np.float32)
    avgpool = AvgPool(dim=3, kernel_size=2, strides=1, pad_mode='VALID')
    actual_output = avgpool(x)
    expect_output = np.array([[[[[8.5, 9.5, 10.5],
                                 [12.5, 13.5, 14.5]]],
                               [[[32.5, 33.5, 34.5],
                                 [36.5, 37.5, 38.5]]],
                               [[[56.5, 57.5, 58.5],
                                 [60.5, 61.5, 62.5]]]]])
    assert np.allclose(actual_output.asnumpy(), expect_output)

    avgpool_grad = AvgPoolGrad(avgpool)
    actual_grad = avgpool_grad(x)
    expect_grad = np.array([[[[[0.125, 0.25, 0.25, 0.125],
                               [0.25, 0.5, 0.5, 0.25],
                               [0.125, 0.25, 0.25, 0.125]],
                              [[0.125, 0.25, 0.25, 0.125],
                               [0.25, 0.5, 0.5, 0.25],
                               [0.125, 0.25, 0.25, 0.125]]],
                             [[[0.125, 0.25, 0.25, 0.125],
                               [0.25, 0.5, 0.5, 0.25],
                               [0.125, 0.25, 0.25, 0.125]],
                              [[0.125, 0.25, 0.25, 0.125],
                               [0.25, 0.5, 0.5, 0.25],
                               [0.125, 0.25, 0.25, 0.125]]],
                             [[[0.125, 0.25, 0.25, 0.125],
                               [0.25, 0.5, 0.5, 0.25],
                               [0.125, 0.25, 0.25, 0.125]],
                              [[0.125, 0.25, 0.25, 0.125],
                               [0.25, 0.5, 0.5, 0.25],
                               [0.125, 0.25, 0.25, 0.125]]]]]).astype(np.float32)
    assert np.allclose(actual_grad[0].asnumpy(), expect_grad)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_avgpool3d_3():
    """
    Feature: test avgpool3d op.
    Description: including forward and backward.
    Expectation: expect correct forward and backward result.
    """
    x_shape = (1, 3, 2, 3, 4)
    x = Tensor(np.arange(reduce(lambda x, y: x * y, x_shape))).reshape(x_shape).astype(np.float32)
    avgpool = AvgPool(dim=3, kernel_size=2, strides=3, pad_mode='VALID')
    actual_output = avgpool(x)
    expect_output = np.array([[[[[8.5]]],
                               [[[32.5]]],
                               [[[56.5]]]]])
    assert np.allclose(actual_output.asnumpy(), expect_output)

    avgpool_grad = AvgPoolGrad(avgpool)
    actual_grad = avgpool_grad(x)
    expect_grad = np.array([[[[[0.125, 0.125, 0., 0.],
                               [0.125, 0.125, 0., 0.],
                               [0., 0., 0., 0.]],
                              [[0.125, 0.125, 0., 0.],
                               [0.125, 0.125, 0., 0.],
                               [0., 0., 0., 0.]]],
                             [[[0.125, 0.125, 0., 0.],
                               [0.125, 0.125, 0., 0.],
                               [0., 0., 0., 0.]],
                              [[0.125, 0.125, 0., 0.],
                               [0.125, 0.125, 0., 0.],
                               [0., 0., 0., 0.]]],
                             [[[0.125, 0.125, 0., 0.],
                               [0.125, 0.125, 0., 0.],
                               [0., 0., 0., 0.]],
                              [[0.125, 0.125, 0., 0.],
                               [0.125, 0.125, 0., 0.],
                               [0., 0., 0., 0.]]]]]).astype(np.float32)
    assert np.allclose(actual_grad[0].asnumpy(), expect_grad)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_avgpool3d_4():
    """
    Feature: test avgpool3d op.
    Description: including forward and backward.
    Expectation: expect correct forward and backward result.
    """
    x_shape = (1, 3, 2, 3, 4)
    x = Tensor(np.arange(reduce(lambda x, y: x * y, x_shape))).reshape(x_shape).astype(np.float32)
    avgpool = AvgPool(dim=3, kernel_size=(2, 2, 3), strides=1, pad_mode='SAME', count_include_pad=False)
    actual_output = avgpool(x)
    expect_output = np.array([[[[[8.5, 9., 10., 10.5],
                                 [12.5, 13., 14., 14.5],
                                 [14.5, 15., 16., 16.5]],
                                [[14.5, 15., 16., 16.5],
                                 [18.5, 19., 20., 20.5],
                                 [20.5, 21., 22., 22.5]]],
                               [[[32.5, 33., 34., 34.5],
                                 [36.5, 37., 38., 38.5],
                                 [38.5, 39., 40., 40.5]],
                                [[38.5, 39., 40., 40.5],
                                 [42.5, 43., 44., 44.5],
                                 [44.5, 45., 46., 46.5]]],
                               [[[56.5, 57., 58., 58.5],
                                 [60.5, 61., 62., 62.5],
                                 [62.5, 63., 64., 64.5]],
                                [[62.5, 63., 64., 64.5],
                                 [66.5, 67., 68., 68.5],
                                 [68.5, 69., 70., 70.5]]]]])
    assert np.allclose(actual_output.asnumpy(), expect_output)

    avgpool_grad = AvgPoolGrad(avgpool)
    actual_grad = avgpool_grad(x)
    expect_grad = np.array([[[[[0.20833334, 0.29166668, 0.29166668, 0.20833334],
                               [0.41666668, 0.58333337, 0.58333337, 0.41666668],
                               [0.625, 0.875, 0.875, 0.625]],
                              [[0.625, 0.875, 0.875, 0.625],
                               [1.25, 1.75, 1.75, 1.25],
                               [1.875, 2.625, 2.625, 1.875]]],
                             [[[0.20833334, 0.29166668, 0.29166668, 0.20833334],
                               [0.41666668, 0.58333337, 0.58333337, 0.41666668],
                               [0.625, 0.875, 0.875, 0.625]],
                              [[0.625, 0.875, 0.875, 0.625],
                               [1.25, 1.75, 1.75, 1.25],
                               [1.875, 2.625, 2.625, 1.875]]],
                             [[[0.20833334, 0.29166668, 0.29166668, 0.20833334],
                               [0.41666668, 0.58333337, 0.58333337, 0.41666668],
                               [0.625, 0.875, 0.875, 0.625]],
                              [[0.625, 0.875, 0.875, 0.625],
                               [1.25, 1.75, 1.75, 1.25],
                               [1.875, 2.625, 2.625, 1.875]]]]])
    assert np.allclose(actual_grad[0].asnumpy(), expect_grad)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_avgpool_vmap():
    """
    Feature: test vmap function.
    Description: test avgpool op vmap.
    Expectation: expect correct result.
    """
    in_axes = -1
    x = Tensor(np.random.randn(1, 1, 6, 6, 3, 6).astype(np.float32))
    net = AvgPool(dim=2, kernel_size=2, strides=2, pad_mode="VALID")
    nest_vmap = vmap(vmap(net, in_axes=in_axes, out_axes=0), in_axes=in_axes, out_axes=0)
    out = nest_vmap(x)
    assert out.shape == (6, 3, 1, 1, 3, 3)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_avgpool_grad_vmap():
    """
    Feature: test vmap function.
    Description: test avgpoolgrad op vmap.
    Expectation: expect correct result.
    """
    in_axes = -1
    x = Tensor(np.random.randn(1, 1, 6, 6, 3, 6).astype(np.float32))
    avgpool = AvgPool(dim=2, kernel_size=2, strides=2, pad_mode="VALID")
    net = AvgPoolGrad(avgpool)
    nest_vmap = vmap(vmap(net, in_axes=in_axes, out_axes=0), in_axes=in_axes, out_axes=0)
    out = nest_vmap(x)
    assert out[0].shape == (6, 3, 1, 1, 6, 6)


class DynamicShapeAvgPool3DGrad(nn.Cell):
    def __init__(self, net, axis=0):
        super(DynamicShapeAvgPool3DGrad, self).__init__()
        self.net = net
        self.unique = P.Unique()
        self.gather = P.Gather()
        self.axis = axis

    def construct(self, x_shape):
        return self.net(x_shape)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_avgpool3d_grad_dynamic_shape():
    """
    Feature: AvgPool3dGrad dynamic test.
    Description: Run unique and gather ops before AvgPool3dGrad.
    Expectation: success.
    """
    x_shape = (1, 3, 2, 3, 4)
    x = Tensor(np.arange(reduce(lambda x, y: x * y, x_shape))).reshape(x_shape).astype(np.float32)
    avgpool = AvgPool(dim=3, kernel_size=2, strides=1, pad_mode='VALID')
    avgpool_grad = AvgPoolGrad(avgpool)
    net = DynamicShapeAvgPool3DGrad(avgpool_grad)
    actual_grad = net(x)
    expect_grad = np.array([[[[[0.125, 0.25, 0.25, 0.125],
                               [0.25, 0.5, 0.5, 0.25],
                               [0.125, 0.25, 0.25, 0.125]],
                              [[0.125, 0.25, 0.25, 0.125],
                               [0.25, 0.5, 0.5, 0.25],
                               [0.125, 0.25, 0.25, 0.125]]],
                             [[[0.125, 0.25, 0.25, 0.125],
                               [0.25, 0.5, 0.5, 0.25],
                               [0.125, 0.25, 0.25, 0.125]],
                              [[0.125, 0.25, 0.25, 0.125],
                               [0.25, 0.5, 0.5, 0.25],
                               [0.125, 0.25, 0.25, 0.125]]],
                             [[[0.125, 0.25, 0.25, 0.125],
                               [0.25, 0.5, 0.5, 0.25],
                               [0.125, 0.25, 0.25, 0.125]],
                              [[0.125, 0.25, 0.25, 0.125],
                               [0.25, 0.5, 0.5, 0.25],
                               [0.125, 0.25, 0.25, 0.125]]]]]).astype(np.float32)
    assert np.allclose(actual_grad[0].asnumpy(), expect_grad)


def avg_pool1d_forward_functional(nptype):
    """
    Feature: test avg_pool1d forward for given input dtype.
    Description: test inputs for given input dtype.
    Expectation: the result match with expected result.
    """
    input_x = Tensor(np.ones((2, 3, 6)).astype(nptype))
    output = F.avg_pool1d(input_x, kernel_size=6, stride=1)
    expected = np.ones((2, 3, 1)).astype(nptype)
    np.testing.assert_array_almost_equal(output.asnumpy(), expected)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_avg_pool1d_forward_float32_functional():
    """
    Feature: test avg_pool1d forward.
    Description: test float32 inputs.
    Expectation: the result match with expected result.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    avg_pool1d_forward_functional(np.float32)
    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
    avg_pool1d_forward_functional(np.float32)


def avg_pool2d_forward_functional(nptype):
    """
    Feature: test avg_pool2d forward for given input dtype.
    Description: test inputs for given input dtype.
    Expectation: the result match with expected result.
    """
    input_x = Tensor(np.ones((2, 3, 4, 6)).astype(nptype))
    output = F.avg_pool2d(input_x, kernel_size=3, stride=1)
    expected = np.ones((2, 3, 2, 4)).astype(nptype)
    np.testing.assert_array_almost_equal(output.asnumpy(), expected)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_avg_pool2d_forward_float32_functional():
    """
    Feature: test avg_pool2d forward.
    Description: test float32 inputs.
    Expectation: the result match with expected result.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    avg_pool2d_forward_functional(np.float32)
    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
    avg_pool2d_forward_functional(np.float32)


def avg_pool3d_forward_functional(nptype):
    """
    Feature: test avg_pool3d forward for given input dtype.
    Description: test inputs for given input dtype.
    Expectation: the result match with expected result.
    """
    input_x = Tensor(np.ones((2, 3, 6, 3, 6)).astype(nptype))
    output = F.avg_pool3d(input_x, kernel_size=3, stride=1)
    expected = np.ones((2, 3, 4, 1, 4)).astype(nptype)
    np.testing.assert_array_almost_equal(output.asnumpy(), expected)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_avg_pool3d_forward_float32_functional():
    """
    Feature: test avg_pool3d forward.
    Description: test float32 inputs.
    Expectation: the result match with expected result.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    avg_pool3d_forward_functional(np.float32)
    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
    avg_pool3d_forward_functional(np.float32)
