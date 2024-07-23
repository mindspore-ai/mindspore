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
from tests.mark_utils import arg_mark

from functools import reduce
import numpy as np
import pytest

import mindspore
import mindspore.context as context
import mindspore.nn as nn
import mindspore.ops.operations as P
from mindspore import Tensor
from mindspore.ops.functional import vmap


class Net_Pool(nn.Cell):
    def __init__(self):
        super(Net_Pool, self).__init__()
        self.maxpool_fun = nn.MaxPool2d(kernel_size=2, stride=2, pad_mode="VALID")

    def construct(self, x):
        return self.maxpool_fun(x)


class Net_Pool2(nn.Cell):
    def __init__(self):
        super(Net_Pool2, self).__init__()
        self.maxpool_fun = nn.MaxPool2d(kernel_size=3, stride=2, pad_mode="SAME")

    def construct(self, x):
        return self.maxpool_fun(x)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_maxpool2d():
    x = Tensor(np.array([[[
        [0, 1, 2, 3, -4, -5],
        [6, 7, 8, 9, -10, -11],
        [12, 13, 14, -15, -16, -17],
        [18, 19, 20, 21, 22, 23],
        [24, 25, 26, 27, 28, 29],
        [30, 31, 32, 33, 34, 35]
    ]]]).astype(np.float32))
    expect_result = (np.array([[[
        [7, 9, -4],
        [19, 21, 23],
        [31, 33, 35]
    ]]]))
    expect_result2 = (np.array([[[
        [14, 14, -4],
        [26, 28, 29],
        [32, 34, 35]
    ]]]))

    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
    maxpool2d = Net_Pool()
    maxpool2d2 = Net_Pool2()
    output2 = maxpool2d2(x)
    output = maxpool2d(x)
    assert (output.asnumpy() == expect_result).all()
    assert (output2.asnumpy() == expect_result2).all()

    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    maxpool2d = Net_Pool()
    maxpool2d2 = Net_Pool2()
    output2 = maxpool2d2(x)
    output = maxpool2d(x)
    assert (output.asnumpy() == expect_result).all()
    assert (output2.asnumpy() == expect_result2).all()


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_max_pool3d_1():
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    x_shape = (2, 3, 2, 3, 4)
    kernel_size = (2, 2, 3)
    strides = 1
    pad_mode = 'VALID'
    x_val = np.arange(reduce(lambda x, y: x * y, x_shape))
    x_ms = Tensor(x_val).reshape(x_shape).astype(np.float32)
    output_ms = P.MaxPool3D(kernel_size=kernel_size, strides=strides, pad_mode=pad_mode)(x_ms)
    expert_result = (np.array([[[[[18, 19],
                                  [22, 23]]],
                                [[[42, 43],
                                  [46, 47]]],
                                [[[66, 67],
                                  [70, 71]]]],
                               [[[[90, 91],
                                  [94, 95]]],
                                [[[114, 115],
                                  [118, 119]]],
                                [[[138, 139],
                                  [142, 143]]]]]))
    assert (output_ms.asnumpy() == expert_result).all()


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_max_pool3d_2():
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    x_shape = (2, 3, 2, 3, 4)
    kernel_size = 2
    strides = 1
    pad_mode = 'VALID'
    x_val = np.arange(reduce(lambda x, y: x * y, x_shape))
    x_ms = Tensor(x_val).reshape(x_shape).astype(np.float32)
    output_ms = P.MaxPool3D(kernel_size=kernel_size, strides=strides, pad_mode=pad_mode)(x_ms)
    expert_result = (np.array([[[[[17, 18, 19],
                                  [21, 22, 23]]],
                                [[[41, 42, 43],
                                  [45, 46, 47]]],
                                [[[65, 66, 67],
                                  [69, 70, 71]]]],
                               [[[[89, 90, 91],
                                  [93, 94, 95]]],
                                [[[113, 114, 115],
                                  [117, 118, 119]]],
                                [[[137, 138, 139],
                                  [141, 142, 143]]]]]))
    assert (output_ms.asnumpy() == expert_result).all()


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_max_pool3d_3():
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    x_shape = (2, 3, 2, 3, 4)
    kernel_size = 2
    strides = 3
    pad_mode = 'VALID'
    x_val = np.arange(reduce(lambda x, y: x * y, x_shape))
    x_ms = Tensor(x_val).reshape(x_shape).astype(np.float32)
    output_ms = P.MaxPool3D(kernel_size=kernel_size, strides=strides, pad_mode=pad_mode)(x_ms)
    expert_result = (np.array([[[[[17]]],
                                [[[41]]],
                                [[[65]]]],
                               [[[[89]]],
                                [[[113]]],
                                [[[137]]]]]))
    assert (output_ms.asnumpy() == expert_result).all()


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_max_pool3d_4():
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    x_shape = (2, 3, 2, 3, 4)
    kernel_size = (2, 2, 3)
    strides = 1
    pad_mode = 'SAME'
    x_val = np.arange(reduce(lambda x, y: x * y, x_shape))
    x_ms = Tensor(x_val).reshape(x_shape).astype(np.float32)
    output_ms = P.MaxPool3D(kernel_size=kernel_size, strides=strides, pad_mode=pad_mode)(x_ms)
    expert_result = (np.array([[[[[17, 18, 19, 19],
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
                                  [69, 70, 71, 71]]]],
                               [[[[89, 90, 91, 91],
                                  [93, 94, 95, 95],
                                  [93, 94, 95, 95]],
                                 [[89, 90, 91, 91],
                                  [93, 94, 95, 95],
                                  [93, 94, 95, 95]]],
                                [[[113, 114, 115, 115],
                                  [117, 118, 119, 119],
                                  [117, 118, 119, 119]],
                                 [[113, 114, 115, 115],
                                  [117, 118, 119, 119],
                                  [117, 118, 119, 119]]],
                                [[[137, 138, 139, 139],
                                  [141, 142, 143, 143],
                                  [141, 142, 143, 143]],
                                 [[137, 138, 139, 139],
                                  [141, 142, 143, 143],
                                  [141, 142, 143, 143]]]]]))
    assert (output_ms.asnumpy() == expert_result).all()


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_max_pool3d_5():
    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
    x_shape = (2, 3, 2, 3, 4)
    kernel_size = (2, 2, 3)
    strides = 1
    pad_mode = 'SAME'
    x_val = np.arange(reduce(lambda x, y: x * y, x_shape))
    x_ms = Tensor(x_val).reshape(x_shape).astype(np.float32)
    output_ms = P.MaxPool3D(kernel_size=kernel_size, strides=strides, pad_mode=pad_mode)(x_ms)
    expert_result = (np.array([[[[[17, 18, 19, 19],
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
                                  [69, 70, 71, 71]]]],
                               [[[[89, 90, 91, 91],
                                  [93, 94, 95, 95],
                                  [93, 94, 95, 95]],
                                 [[89, 90, 91, 91],
                                  [93, 94, 95, 95],
                                  [93, 94, 95, 95]]],
                                [[[113, 114, 115, 115],
                                  [117, 118, 119, 119],
                                  [117, 118, 119, 119]],
                                 [[113, 114, 115, 115],
                                  [117, 118, 119, 119],
                                  [117, 118, 119, 119]]],
                                [[[137, 138, 139, 139],
                                  [141, 142, 143, 143],
                                  [141, 142, 143, 143]],
                                 [[137, 138, 139, 139],
                                  [141, 142, 143, 143],
                                  [141, 142, 143, 143]]]]]))
    assert (output_ms.asnumpy() == expert_result).all()


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_max_pool2d_vmap():
    """
    Feature: Test MaxPool op.
    Description: Vmap test--P.MaxPoolWithArgmax.
    Expectation: Consistent with the assertion.
    """
    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
    def max_pool(x):
        return P.MaxPoolWithArgmax(kernel_size=2, strides=1, pad_mode="valid", data_format="NCHW")(x)

    # once vmap
    x1 = Tensor(np.arange(1 * 2 * 3 * 4).reshape(1, 1, 2, 3, 4), mindspore.float32)
    vmap_max_pool = vmap(max_pool, in_axes=-1)
    outputs, indices1 = vmap_max_pool(x1)
    assert outputs.asnumpy().shape == (4, 1, 1, 1, 2)
    assert indices1.asnumpy().shape == (4, 1, 1, 1, 2)

    # twice vmap
    x2 = Tensor(np.arange(1 * 2 * 3 * 4).reshape(1, 1, 1, 2, 3, 4), mindspore.float32)
    vmap_max_pool = vmap(vmap(max_pool, in_axes=0), in_axes=0)
    outputs, indices2 = vmap_max_pool(x2)
    assert outputs.asnumpy().shape == (1, 1, 1, 2, 2, 3)
    assert indices2.asnumpy().shape == (1, 1, 1, 2, 2, 3)
