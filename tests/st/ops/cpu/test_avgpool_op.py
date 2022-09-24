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

from functools import reduce
import numpy as np
import pytest

from mindspore import Tensor
from mindspore import dtype as msdtype
import mindspore.context as context
import mindspore.nn as nn
import mindspore.ops.operations as P
from mindspore.ops import composite as C
from mindspore.ops import functional as F

context.set_context(mode=context.GRAPH_MODE, device_target="CPU")


class AvgPool(nn.Cell):
    def __init__(self, dim, kernel_size, strides, pad_mode, count_include_pad=False):
        super(AvgPool, self).__init__()
        if dim == 2:
            self.avgpool = P.AvgPool(
                kernel_size=kernel_size, strides=strides, pad_mode=pad_mode)
        else:
            self.avgpool = P.AvgPool3D(kernel_size=kernel_size, strides=strides, pad_mode=pad_mode,
                                       count_include_pad=count_include_pad)

    def construct(self, x):
        return self.avgpool(x)


class AvgPoolGrad(nn.Cell):
    def __init__(self, forward):
        super(AvgPoolGrad, self).__init__()
        self.forward = forward
        self.grad = C.GradOperation(get_all=True, sens_param=True)

    def construct(self, x, sens):
        return self.grad(self.forward)(x, sens)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
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
    assert (actual_output.asnumpy() == expect_output).all()

    avgpool_grad = AvgPoolGrad(avgpool)
    sens = Tensor(np.arange(1, 10).reshape(
        actual_output.shape).astype(np.float32))
    actual_grad = avgpool_grad(x, sens)
    expect_grad = np.array([[[[0.25, 0.25, 0.5, 0.5, 0.75, 0.75],
                              [0.25, 0.25, 0.5, 0.5, 0.75, 0.75],
                              [1., 1., 1.25, 1.25, 1.5, 1.5],
                              [1., 1., 1.25, 1.25, 1.5, 1.5],
                              [1.75, 1.75, 2., 2., 2.25, 2.25],
                              [1.75, 1.75, 2., 2., 2.25, 2.25]]]]).astype(np.float32)
    assert (actual_grad[0].asnumpy() == expect_grad).all()


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
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
    assert (actual_output.asnumpy() == expect_output).all()

    avgpool_grad = AvgPoolGrad(avgpool)
    sens = Tensor(np.arange(1, 10).reshape(
        actual_output.shape).astype(np.float32))
    actual_grad = avgpool_grad(x, sens)
    expect_grad = np.array([[[[0.11111111, 0.11111111, 0.33333334, 0.22222222, 0.7222222, 0.5],
                              [0.11111111, 0.11111111, 0.33333334,
                               0.22222222, 0.7222222, 0.5],
                              [0.5555556, 0.5555556, 1.3333334,
                               0.7777778, 2.2777777, 1.5],
                              [0.44444445, 0.44444445, 1.,
                               0.5555556, 1.5555556, 1.],
                              [1.611111, 1.611111, 3.5, 1.888889, 5.138889, 3.25],
                              [1.1666666, 1.1666666, 2.5, 1.3333334, 3.5833335, 2.25]]]]).astype(np.float32)
    assert (actual_grad[0].asnumpy() == expect_grad).all()


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_avgpool3d_1():
    """
    Feature: test avgpool3d op.
    Description: including forward and backward.
    Expectation: expect correct forward and backward result.
    """
    x_shape = (1, 3, 2, 3, 4)
    x = Tensor(np.arange(reduce(lambda x, y: x * y, x_shape))
               ).reshape(x_shape).astype(np.float32)
    avgpool = AvgPool(dim=3, kernel_size=(2, 2, 3),
                      strides=1, pad_mode='VALID')
    actual_output = avgpool(x)
    expect_output = np.array([[[[[9., 10.],
                                 [13., 14.]]],
                               [[[33., 34.],
                                 [37., 38.]]],
                               [[[57., 58.],
                                 [61., 62.]]]]])
    assert (actual_output.asnumpy() == expect_output).all()

    avgpool_grad = AvgPoolGrad(avgpool)
    sens = actual_output + 1
    actual_grad = avgpool_grad(x, sens)
    expect_grad = np.array([[[[[0.8333333, 1.75, 1.75, 0.9166667],
                               [2., 4.1666665, 4.1666665, 2.1666667],
                               [1.1666666, 2.4166665, 2.4166665, 1.25]],
                              [[0.8333333, 1.75, 1.75, 0.9166667],
                               [2., 4.1666665, 4.1666665, 2.1666667],
                               [1.1666666, 2.4166665, 2.4166665, 1.25]]],
                             [[[2.8333333, 5.75, 5.75, 2.9166667],
                               [6., 12.166667, 12.166667, 6.166667],
                               [3.1666667, 6.416667, 6.416667, 3.25]],
                              [[2.8333333, 5.75, 5.75, 2.9166667],
                               [6., 12.166667, 12.166667, 6.166667],
                               [3.1666667, 6.416667, 6.416667, 3.25]]],
                             [[[4.8333335, 9.75, 9.75, 4.9166665],
                               [10., 20.166666, 20.166666, 10.166666],
                               [5.1666665, 10.416666, 10.416666, 5.25]],
                              [[4.8333335, 9.75, 9.75, 4.9166665],
                               [10., 20.166666, 20.166666, 10.166666],
                               [5.1666665, 10.416666, 10.416666, 5.25]]]]]).astype(np.float32)
    assert (actual_grad[0].asnumpy() == expect_grad).all()


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_avgpool3d_2():
    """
    Feature: test avgpool3d op.
    Description: including forward and backward.
    Expectation: expect correct forward and backward result.
    """
    x_shape = (1, 3, 2, 3, 4)
    x = Tensor(np.arange(reduce(lambda x, y: x * y, x_shape))
               ).reshape(x_shape).astype(np.float32)
    avgpool = AvgPool(dim=3, kernel_size=2, strides=1, pad_mode='VALID')
    actual_output = avgpool(x)
    expect_output = np.array([[[[[8.5, 9.5, 10.5],
                                 [12.5, 13.5, 14.5]]],
                               [[[32.5, 33.5, 34.5],
                                 [36.5, 37.5, 38.5]]],
                               [[[56.5, 57.5, 58.5],
                                 [60.5, 61.5, 62.5]]]]])
    assert (actual_output.asnumpy() == expect_output).all()

    avgpool_grad = AvgPoolGrad(avgpool)
    sens = actual_output + 1
    actual_grad = avgpool_grad(x, sens)
    expect_grad = np.array([[[[[1.1875, 2.5, 2.75, 1.4375],
                               [2.875, 6., 6.5, 3.375],
                               [1.6875, 3.5, 3.75, 1.9375]],
                              [[1.1875, 2.5, 2.75, 1.4375],
                               [2.875, 6., 6.5, 3.375],
                               [1.6875, 3.5, 3.75, 1.9375]]],
                             [[[4.1875, 8.5, 8.75, 4.4375],
                               [8.875, 18., 18.5, 9.375],
                               [4.6875, 9.5, 9.75, 4.9375]],
                              [[4.1875, 8.5, 8.75, 4.4375],
                               [8.875, 18., 18.5, 9.375],
                               [4.6875, 9.5, 9.75, 4.9375]]],
                             [[[7.1875, 14.5, 14.75, 7.4375],
                               [14.875, 30., 30.5, 15.375],
                               [7.6875, 15.5, 15.75, 7.9375]],
                              [[7.1875, 14.5, 14.75, 7.4375],
                               [14.875, 30., 30.5, 15.375],
                               [7.6875, 15.5, 15.75, 7.9375]]]]]).astype(np.float32)
    assert (actual_grad[0].asnumpy() == expect_grad).all()


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_avgpool3d_3():
    """
    Feature: test avgpool3d op.
    Description: including forward and backward.
    Expectation: expect correct forward and backward result.
    """
    x_shape = (1, 3, 2, 3, 4)
    x = Tensor(np.arange(reduce(lambda x, y: x * y, x_shape))
               ).reshape(x_shape).astype(np.float32)
    avgpool = AvgPool(dim=3, kernel_size=2, strides=3, pad_mode='VALID')
    actual_output = avgpool(x)
    expect_output = np.array([[[[[8.5]]],
                               [[[32.5]]],
                               [[[56.5]]]]])
    assert (actual_output.asnumpy() == expect_output).all()

    avgpool_grad = AvgPoolGrad(avgpool)
    sens = actual_output + 1
    actual_grad = avgpool_grad(x, sens)
    expect_grad = np.array([[[[[1.1875, 1.1875, 0., 0.],
                               [1.1875, 1.1875, 0., 0.],
                               [0., 0., 0., 0.]],
                              [[1.1875, 1.1875, 0., 0.],
                               [1.1875, 1.1875, 0., 0.],
                               [0., 0., 0., 0.]]],
                             [[[4.1875, 4.1875, 0., 0.],
                               [4.1875, 4.1875, 0., 0.],
                               [0., 0., 0., 0.]],
                              [[4.1875, 4.1875, 0., 0.],
                               [4.1875, 4.1875, 0., 0.],
                               [0., 0., 0., 0.]]],
                             [[[7.1875, 7.1875, 0., 0.],
                               [7.1875, 7.1875, 0., 0.],
                               [0., 0., 0., 0.]],
                              [[7.1875, 7.1875, 0., 0.],
                               [7.1875, 7.1875, 0., 0.],
                               [0., 0., 0., 0.]]]]]).astype(np.float32)
    assert (actual_grad[0].asnumpy() == expect_grad).all()


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_avgpool3d_4():
    """
    Feature: test avgpool3d op.
    Description: including forward and backward.
    Expectation: expect correct forward and backward result.
    """
    x_shape = (1, 3, 2, 3, 4)
    x = Tensor(np.arange(reduce(lambda x, y: x * y, x_shape))
               ).reshape(x_shape).astype(np.float32)
    avgpool = AvgPool(dim=3, kernel_size=(2, 2, 3), strides=1,
                      pad_mode='SAME', count_include_pad=False)
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
    assert (actual_output.asnumpy() == expect_output).all()

    avgpool_grad = AvgPoolGrad(avgpool)
    sens = actual_output + 1
    actual_grad = avgpool_grad(x, sens)
    expect_grad = np.array([[[[[2.0208333, 2.9375, 3.1875, 2.3541667],
                               [4.875, 7.0416665, 7.5416665, 5.541667],
                               [9.395833, 13.479166, 14.229166, 10.395833]],
                              [[8.5625, 12.3125, 13.0625, 9.5625],
                               [19.625002, 28.125, 29.625, 21.625],
                               [35.6875, 50.9375, 53.1875, 38.6875]]],
                             [[[7.020833, 9.9375, 10.1875, 7.354167],
                               [14.875, 21.041666, 21.541668, 15.541667],
                               [24.395834, 34.479168, 35.229168, 25.395834]],
                              [[23.562498, 33.3125, 34.0625, 24.5625],
                               [49.624996, 70.125, 71.625, 51.625],
                               [80.68751, 113.9375, 116.1875, 83.6875]]],
                             [[[12.020834, 16.9375, 17.1875, 12.354166],
                               [24.875, 35.041664, 35.541664, 25.541666],
                               [39.395832, 55.479164, 56.229164, 40.395832]],
                              [[38.5625, 54.3125, 55.0625, 39.5625],
                               [79.62501, 112.125, 113.625, 81.625],
                               [125.6875, 176.9375, 179.1875, 128.6875]]]]]).astype(np.float32)
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


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_avg_pool1d_forward_float32_functional():
    """
    Feature: test avg_pool1d forward.
    Description: test float32 inputs.
    Expectation: the result match with expected result.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    avg_pool1d_forward_functional(np.float32)
    context.set_context(mode=context.PYNATIVE_MODE, device_target="CPU")
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


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_avg_pool2d_forward_float32_functional():
    """
    Feature: test avg_pool2d forward.
    Description: test float32 inputs.
    Expectation: the result match with expected result.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    avg_pool2d_forward_functional(np.float32)
    context.set_context(mode=context.PYNATIVE_MODE, device_target="CPU")
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


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_avg_pool3d_forward_float32_functional():
    """
    Feature: test avg_pool3d forward.
    Description: test float32 inputs.
    Expectation: the result match with expected result.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    avg_pool3d_forward_functional(np.float32)
    context.set_context(mode=context.PYNATIVE_MODE, device_target="CPU")
    avg_pool3d_forward_functional(np.float32)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_avgpool_cpu_dynamic_shape():
    """
    Feature: test dynamic shape of avgpool.
    Description: test the dynamic shape output of avgpool.
    Expectation: correct output shape.
    """

    x_dyn = Tensor(shape=[None, 32, None, None], dtype=msdtype.float32)
    net = AvgPool(dim=2, kernel_size=2, strides=2, pad_mode="VALID")
    net.set_inputs(x_dyn)
    x = np.random.randn(2, 32, 9, 9)
    output = net(Tensor(x, msdtype.float32))
    expect_out_shape = (2, 32, 4, 4)
    assert output.asnumpy().shape == expect_out_shape
