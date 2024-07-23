# Copyright 2019-2023 Huawei Technologies Co., Ltd
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

import mindspore
import mindspore.context as context
import mindspore.nn as nn
import mindspore.ops.operations as P
from mindspore import Tensor
from mindspore.ops import composite as C
from mindspore.ops.functional import vmap

context.set_context(mode=context.GRAPH_MODE, device_target="CPU")


class MaxPool(nn.Cell):
    def __init__(self, dim, kernel_size, strides, pad_mode):
        super(MaxPool, self).__init__()
        if dim == 2:
            self.maxpool = P.MaxPool(kernel_size=kernel_size, strides=strides, pad_mode=pad_mode)
        else:
            self.maxpool = P.MaxPool3D(kernel_size=kernel_size, strides=strides, pad_mode=pad_mode)

    def construct(self, x):
        return self.maxpool(x)


class MaxPoolGrad(nn.Cell):
    def __init__(self, forward):
        super(MaxPoolGrad, self).__init__()
        self.forward = forward
        self.grad = C.GradOperation(get_all=True, sens_param=True)

    def construct(self, x, sens):
        return self.grad(self.forward)(x, sens)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_maxpool2d_valid():
    """
    Feature: test maxpool2d op.
    Description: including forward and backward.
    Expectation: expect correct forward and backward result.
    """
    x = Tensor(np.array([[[[10, 1, 2, 3, -4, -5],
                           [6, 7, 8, 9, -10, -11],
                           [12, 13, 24, -15, -16, -17],
                           [18, 19, 20, 21, 22, 23],
                           [32, 25, 26, 27, 28, 40],
                           [30, 31, 35, 33, 34, 35]]]]).astype(np.float32))
    maxpool = MaxPool(dim=2, kernel_size=2, strides=2, pad_mode="VALID")
    actual_output = maxpool(x)
    expect_output = np.array([[[[10, 9, -4],
                                [19, 24, 23],
                                [32, 35, 40]]]]).astype(np.float32)
    assert (actual_output.asnumpy() == expect_output).all()

    maxpool_grad = MaxPoolGrad(maxpool)
    sens = Tensor(np.arange(1, 10).reshape(actual_output.shape).astype(np.float32))
    actual_grad = maxpool_grad(x, sens)
    expect_dx = np.array([[[[1, 0, 0, 0, 3, 0],
                            [0, 0, 0, 2, 0, 0],
                            [0, 0, 5, 0, 0, 0],
                            [0, 4, 0, 0, 0, 6],
                            [7, 0, 0, 0, 0, 9],
                            [0, 0, 8, 0, 0, 0]]]]).astype(np.float32)
    assert (actual_grad[0].asnumpy() == expect_dx).all()


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_maxpool2d_same():
    """
    Feature: test maxpool2d op.
    Description: including forward and backward.
    Expectation: expect correct forward and backward result.
    """
    x = Tensor(np.array([[[[0, 1, 2, 3, -4, -5],
                           [6, 7, 8, 9, -10, -11],
                           [12, 13, 14, -15, -16, -17],
                           [18, 19, 20, 21, 22, 23],
                           [24, 25, 26, 27, 28, 29],
                           [30, 31, 32, 33, 34, 35]]]]).astype(np.float32))
    maxpool = MaxPool(dim=2, kernel_size=3, strides=2, pad_mode="SAME")
    actual_output = maxpool(x)
    expect_output = np.array([[[[14, 14, -4],
                                [26, 28, 29],
                                [32, 34, 35]]]])
    assert (actual_output.asnumpy() == expect_output).all()

    maxpool_grad = MaxPoolGrad(maxpool)
    sens = Tensor(np.arange(1, 10).reshape(actual_output.shape).astype(np.float32))
    actual_grad = maxpool_grad(x, sens)
    expect_dx = np.array([[[[0, 0, 0, 0, 3, 0],
                            [0, 0, 0, 0, 0, 0],
                            [0, 0, 3, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0],
                            [0, 0, 4, 0, 5, 6],
                            [0, 0, 7, 0, 8, 9]]]]).astype(np.float32)
    assert (actual_grad[0].asnumpy() == expect_dx).all()


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize("dtype", [np.float32, np.float16, np.float64])
def test_maxpool3d_1(dtype):
    """
    Feature: test maxpool3d op.
    Description: including forward and backward.
    Expectation: expect correct forward and backward result.
    """
    x_shape = (1, 3, 2, 3, 4)
    x = Tensor(np.arange(reduce(lambda x, y: x * y, x_shape))).reshape(x_shape).astype(dtype)
    maxpool = MaxPool(dim=3, kernel_size=(2, 2, 3), strides=1, pad_mode='VALID')
    actual_output = maxpool(x)
    expect_output = np.array([[[[[18, 19],
                                 [22, 23]]],
                               [[[42, 43],
                                 [46, 47]]],
                               [[[66, 67],
                                 [70, 71]]]]])
    assert (actual_output.asnumpy() == expect_output).all()

    maxpool_grad = MaxPoolGrad(maxpool)
    sens = actual_output + 1
    actual_grad = maxpool_grad(x, sens)
    expect_dx = np.array([[[[[0, 0, 0, 0],
                             [0, 0, 0, 0],
                             [0, 0, 0, 0]],
                            [[0, 0, 0, 0],
                             [0, 0, 19, 20],
                             [0, 0, 23, 24]]],
                           [[[0, 0, 0, 0],
                             [0, 0, 0, 0],
                             [0, 0, 0, 0]],
                            [[0, 0, 0, 0],
                             [0, 0, 43, 44],
                             [0, 0, 47, 48]]],
                           [[[0, 0, 0, 0],
                             [0, 0, 0, 0],
                             [0, 0, 0, 0]],
                            [[0, 0, 0, 0],
                             [0, 0, 67, 68],
                             [0, 0, 71, 72]]]]]).astype(np.float32)
    assert (actual_grad[0].asnumpy() == expect_dx).all()


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize("dtype", [np.float32, np.float16, np.float64])
def test_maxpool3d_2(dtype):
    """
    Feature: test maxpool3d op.
    Description: including forward and backward.
    Expectation: expect correct forward and backward result.
    """
    x_shape = (1, 3, 2, 3, 4)
    x = Tensor(np.arange(reduce(lambda x, y: x * y, x_shape))).reshape(x_shape).astype(dtype)
    maxpool = MaxPool(dim=3, kernel_size=2, strides=1, pad_mode='VALID')
    actual_output = maxpool(x)
    expect_output = np.array([[[[[17, 18, 19],
                                 [21, 22, 23]]],
                               [[[41, 42, 43],
                                 [45, 46, 47]]],
                               [[[65, 66, 67],
                                 [69, 70, 71]]]]])
    assert (actual_output.asnumpy() == expect_output).all()

    maxpool_grad = MaxPoolGrad(maxpool)
    sens = actual_output + 1
    actual_grad = maxpool_grad(x, sens)
    expect_dx = np.array([[[[[0, 0, 0, 0],
                             [0, 0, 0, 0],
                             [0, 0, 0, 0]],
                            [[0, 0, 0, 0],
                             [0, 18, 19, 20],
                             [0, 22, 23, 24]]],
                           [[[0, 0, 0, 0],
                             [0, 0, 0, 0],
                             [0, 0, 0, 0]],
                            [[0, 0, 0, 0],
                             [0, 42, 43, 44],
                             [0, 46, 47, 48]]],
                           [[[0, 0, 0, 0],
                             [0, 0, 0, 0],
                             [0, 0, 0, 0]],
                            [[0, 0, 0, 0],
                             [0, 66, 67, 68],
                             [0, 70, 71, 72]]]]]).astype(np.float32)
    assert (actual_grad[0].asnumpy() == expect_dx).all()


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize("dtype", [np.float32, np.float16, np.float64])
def test_maxpool3d_3(dtype):
    """
    Feature: test maxpool3d op.
    Description: including forward and backward.
    Expectation: expect correct forward and backward result.
    """
    x_shape = (1, 3, 2, 3, 4)
    x = Tensor(np.arange(reduce(lambda x, y: x * y, x_shape))).reshape(x_shape).astype(dtype)
    maxpool = MaxPool(dim=3, kernel_size=2, strides=3, pad_mode='VALID')
    actual_output = maxpool(x)
    expect_output = np.array([[[[[17]]],
                               [[[41]]],
                               [[[65]]]]])
    assert (actual_output.asnumpy() == expect_output).all()

    maxpool_grad = MaxPoolGrad(maxpool)
    sens = actual_output + 1
    actual_grad = maxpool_grad(x, sens)
    expect_dx = np.array([[[[[0, 0, 0, 0],
                             [0, 0, 0, 0],
                             [0, 0, 0, 0]],
                            [[0, 0, 0, 0],
                             [0, 18, 0, 0],
                             [0, 0, 0, 0]]],
                           [[[0, 0, 0, 0],
                             [0, 0, 0, 0],
                             [0, 0, 0, 0]],
                            [[0, 0, 0, 0],
                             [0, 42, 0, 0],
                             [0, 0, 0, 0]]],
                           [[[0, 0, 0, 0],
                             [0, 0, 0, 0],
                             [0, 0, 0, 0]],
                            [[0, 0, 0, 0],
                             [0, 66, 0, 0],
                             [0, 0, 0, 0]]]]]).astype(np.float32)
    assert (actual_grad[0].asnumpy() == expect_dx).all()


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize("dtype", [np.float32, np.float16, np.float64])
def test_maxpool3d_4(dtype):
    """
    Feature: test maxpool3d op.
    Description: including forward and backward.
    Expectation: expect correct forward and backward result.
    """
    x_shape = (1, 3, 2, 3, 4)
    x = Tensor(np.arange(reduce(lambda x, y: x * y, x_shape))).reshape(x_shape).astype(dtype)
    maxpool = MaxPool(dim=3, kernel_size=(2, 2, 3), strides=1, pad_mode='SAME')
    actual_output = maxpool(x)
    expect_output = np.array([[[[[17, 18, 19, 19],
                                 [21, 22, 23, 23],
                                 [21, 22, 23, 23]],
                                [[17, 18, 19, 19],
                                 [21, 22, 23, 23],
                                 [21, 22, 23, 23]]],
                               [[[41, 42, 43, 43],
                                 [45, 46, 47, 47],
                                 [45, 46, 47, 47]],
                                [[41, 42, 43, 43],
                                 [45, 46, 47, 47],
                                 [45, 46, 47, 47]]],
                               [[[65, 66, 67, 67],
                                 [69, 70, 71, 71],
                                 [69, 70, 71, 71]],
                                [[65, 66, 67, 67],
                                 [69, 70, 71, 71],
                                 [69, 70, 71, 71]]]]])
    assert (actual_output.asnumpy() == expect_output).all()

    maxpool_grad = MaxPoolGrad(maxpool)
    sens = actual_output + 1
    actual_grad = maxpool_grad(x, sens)
    expect_dx = np.array([[[[[0, 0, 0, 0],
                             [0, 0, 0, 0],
                             [0, 0, 0, 0]],
                            [[0, 0, 0, 0],
                             [0, 36, 38, 80],
                             [0, 88, 92, 192]]],
                           [[[0, 0, 0, 0],
                             [0, 0, 0, 0],
                             [0, 0, 0, 0]],
                            [[0, 0, 0, 0],
                             [0, 84, 86, 176],
                             [0, 184, 188, 384]]],
                           [[[0, 0, 0, 0],
                             [0, 0, 0, 0],
                             [0, 0, 0, 0]],
                            [[0, 0, 0, 0],
                             [0, 132, 134, 272],
                             [0, 280, 284, 576]]]]]).astype(np.float32)
    assert (actual_grad[0].asnumpy() == expect_dx).all()


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_max_pool2d_vmap():
    """
    Feature: Test maxpool op.
    Description: Vmap test--P.MaxPool.
    Expectation: Consistent with the assertion.
    """
    context.set_context(device_target="CPU")
    def max_pool(x):
        return P.MaxPool(kernel_size=2, strides=1, pad_mode="valid", data_format="NCHW")(x)

    # once vmap
    x1 = Tensor(np.arange(1 * 2 * 3 * 4).reshape(1, 1, 2, 3, 4), mindspore.float32)
    vmap_max_pool = vmap(max_pool, in_axes=-1)
    outputs = vmap_max_pool(x1)
    assert outputs.asnumpy().shape == (4, 1, 1, 1, 2)

    # twice vmap
    x2 = Tensor(np.arange(1 * 2 * 3 * 4).reshape(1, 1, 1, 2, 3, 4), mindspore.float32)
    vmap_max_pool = vmap(vmap(max_pool, in_axes=0), in_axes=0)
    outputs = vmap_max_pool(x2)
    assert outputs.asnumpy().shape == (1, 1, 1, 2, 2, 3)
