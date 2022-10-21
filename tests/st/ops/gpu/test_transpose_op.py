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
from mindspore.common.api import jit
from mindspore.common.initializer import initializer
from mindspore.common.parameter import Parameter
from mindspore.ops import operations as P
from mindspore.ops.operations import _inner_ops as inner

context.set_context(device_target='GPU')

class Transpose(nn.Cell):
    def __init__(self, nptype):
        super(Transpose, self).__init__()
        self.transpose = P.Transpose()
        self.x_2D = Parameter(initializer(Tensor(np.arange(5 * 6).reshape(5, 6).astype(nptype)), [5, 6]),
                              name='x_2D')
        self.perm_2D = (1, 0)
        self.x_3D = Parameter(initializer(Tensor(np.arange(2 * 2 * 4).reshape(2, 2, 4).astype(nptype)), [2, 2, 4]),
                              name='x_3D')
        self.perm_3D = (1, 0, 2)
        self.x_4D = Parameter(
            initializer(Tensor(np.arange(2 * 3 * 4 * 5).reshape(2, 3, 4, 5).astype(nptype)), [2, 3, 4, 5]),
            name='x_4D')
        self.perm_4D = (0, 1, 2, 3)
        self.x_5D = Parameter(
            initializer(Tensor(np.arange(1 * 2 * 3 * 4 * 5).reshape(1, 2, 3, 4, 5).astype(nptype)),
                        [1, 2, 3, 4, 5]), name='x_5D')
        self.perm_5D = (1, 0, 3, 4, 2)

    @jit
    def construct(self):
        return (self.transpose(self.x_2D, self.perm_2D), self.transpose(self.x_3D, self.perm_3D),
                self.transpose(self.x_4D, self.perm_4D), self.transpose(self.x_5D, self.perm_5D))

class Transpose_dynamic(nn.Cell):
    def __init__(self, nptype):
        super(Transpose_dynamic, self).__init__()
        self.transpose = P.Transpose()
        self.test_dynamic = inner.GpuConvertToDynamicShape()
        self.x = Parameter(
            initializer(Tensor(np.arange(1 * 2 * 3 * 4 * 5).reshape(1, 2, 3, 4, 5).astype(nptype)),
                        [1, 2, 3, 4, 5]), name='5D')
        self.perm = (1, 0, 3, 4, 2)

    @jit
    def construct(self):
        out = self.test_dynamic(self.x)
        return self.transpose(out, self.perm)

class Transpose_dynamic2(nn.Cell):
    def __init__(self, input_1, input_2, perm_1, perm_2):
        super(Transpose_dynamic2, self).__init__()
        self.transpose = P.Transpose()
        self.test_dynamic = inner.GpuConvertToDynamicShape()
        self.x_1 = input_1
        self.x_2 = input_2
        self.perm_1 = perm_1
        self.perm_2 = perm_2

    @jit
    def construct(self):
        out_1 = self.test_dynamic(self.x_1)
        out_1 = self.transpose(out_1, self.perm_1)
        out_2 = self.test_dynamic(self.x_2)
        out_2 = self.transpose(out_2, self.perm_2)
        return (out_1, out_2)

def transpose1(nptype):
    context.set_context(mode=context.GRAPH_MODE, device_target='GPU')
    transpose = Transpose(nptype)
    output = transpose()
    expect0 = np.array([[[0, 6, 12, 18, 24],
                         [1, 7, 13, 19, 25],
                         [2, 8, 14, 20, 26],
                         [3, 9, 15, 21, 27],
                         [4, 10, 16, 22, 28],
                         [5, 11, 17, 23, 29]]]).astype(nptype)
    expect1 = np.array([[[[0, 1, 2, 3],
                          [8, 9, 10, 11]],
                         [[4, 5, 6, 7],
                          [12, 13, 14, 15]]]]).astype(nptype)
    expect2 = np.array([[[[[0, 1, 2, 3, 4],
                           [5, 6, 7, 8, 9],
                           [10, 11, 12, 13, 14],
                           [15, 16, 17, 18, 19]],
                          [[20, 21, 22, 23, 24],
                           [25, 26, 27, 28, 29],
                           [30, 31, 32, 33, 34],
                           [35, 36, 37, 38, 39]],
                          [[40, 41, 42, 43, 44],
                           [45, 46, 47, 48, 49],
                           [50, 51, 52, 53, 54],
                           [55, 56, 57, 58, 59]]],
                         [[[60, 61, 62, 63, 64],
                           [65, 66, 67, 68, 69],
                           [70, 71, 72, 73, 74],
                           [75, 76, 77, 78, 79]],
                          [[80, 81, 82, 83, 84],
                           [85, 86, 87, 88, 89],
                           [90, 91, 92, 93, 94],
                           [95, 96, 97, 98, 99]],
                          [[100, 101, 102, 103, 104],
                           [105, 106, 107, 108, 109],
                           [110, 111, 112, 113, 114],
                           [115, 116, 117, 118, 119]]]]]).astype(nptype)
    expect3 = np.array([[[[[[0, 20, 40],
                            [1, 21, 41],
                            [2, 22, 42],
                            [3, 23, 43],
                            [4, 24, 44]],
                           [[5, 25, 45],
                            [6, 26, 46],
                            [7, 27, 47],
                            [8, 28, 48],
                            [9, 29, 49]],
                           [[10, 30, 50],
                            [11, 31, 51],
                            [12, 32, 52],
                            [13, 33, 53],
                            [14, 34, 54]],
                           [[15, 35, 55],
                            [16, 36, 56],
                            [17, 37, 57],
                            [18, 38, 58],
                            [19, 39, 59]]]],
                         [[[[60, 80, 100],
                            [61, 81, 101],
                            [62, 82, 102],
                            [63, 83, 103],
                            [64, 84, 104]],
                           [[65, 85, 105],
                            [66, 86, 106],
                            [67, 87, 107],
                            [68, 88, 108],
                            [69, 89, 109]],
                           [[70, 90, 110],
                            [71, 91, 111],
                            [72, 92, 112],
                            [73, 93, 113],
                            [74, 94, 114]],
                           [[75, 95, 115],
                            [76, 96, 116],
                            [77, 97, 117],
                            [78, 98, 118],
                            [79, 99, 119]]]]]]).astype(nptype)
    assert (output[0].asnumpy() == expect0).all()
    assert (output[1].asnumpy() == expect1).all()
    assert (output[2].asnumpy() == expect2).all()
    assert (output[3].asnumpy() == expect3).all()

def transpose_d(nptype):
    context.set_context(mode=context.GRAPH_MODE, device_target='GPU')
    transpose = Transpose_dynamic(nptype)
    output = transpose()
    expect = np.array([[[[[[0, 20, 40],
                           [1, 21, 41],
                           [2, 22, 42],
                           [3, 23, 43],
                           [4, 24, 44]],
                          [[5, 25, 45],
                           [6, 26, 46],
                           [7, 27, 47],
                           [8, 28, 48],
                           [9, 29, 49]],
                          [[10, 30, 50],
                           [11, 31, 51],
                           [12, 32, 52],
                           [13, 33, 53],
                           [14, 34, 54]],
                          [[15, 35, 55],
                           [16, 36, 56],
                           [17, 37, 57],
                           [18, 38, 58],
                           [19, 39, 59]]]],
                        [[[[60, 80, 100],
                           [61, 81, 101],
                           [62, 82, 102],
                           [63, 83, 103],
                           [64, 84, 104]],
                          [[65, 85, 105],
                           [66, 86, 106],
                           [67, 87, 107],
                           [68, 88, 108],
                           [69, 89, 109]],
                          [[70, 90, 110],
                           [71, 91, 111],
                           [72, 92, 112],
                           [73, 93, 113],
                           [74, 94, 114]],
                          [[75, 95, 115],
                           [76, 96, 116],
                           [77, 97, 117],
                           [78, 98, 118],
                           [79, 99, 119]]]]]]).astype(nptype)
    assert (output.asnumpy() == expect).all()

def transpose_d2(nptype):
    context.set_context(mode=context.GRAPH_MODE, device_target='GPU')
    input_1 = Parameter(Tensor(np.arange(5 * 6).reshape(5, 6).astype(nptype)),
                        name="input_1")
    input_2 = Parameter(Tensor(np.arange(2 * 2 * 4).reshape(2, 2, 4).astype(nptype)),
                        name="input_2")
    perm_1 = (1, 0)
    perm_2 = (1, 0, 2)
    expect_1 = np.array([[[0, 6, 12, 18, 24],
                          [1, 7, 13, 19, 25],
                          [2, 8, 14, 20, 26],
                          [3, 9, 15, 21, 27],
                          [4, 10, 16, 22, 28],
                          [5, 11, 17, 23, 29]]]).astype(nptype)
    expect_2 = np.array([[[[0, 1, 2, 3],
                           [8, 9, 10, 11]],
                          [[4, 5, 6, 7],
                           [12, 13, 14, 15]]]]).astype(nptype)
    net = Transpose_dynamic2(input_1, input_2, perm_1, perm_2)
    output_1, output_2 = net()
    assert (output_1.asnumpy() == expect_1).all()
    assert (output_2.asnumpy() == expect_2).all()

@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_transpose_float32():
    transpose1(np.float32)

@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_transpose_float16():
    transpose1(np.float16)

@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_transpose_int32():
    transpose1(np.int32)

@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_transpose_int64():
    transpose1(np.int64)

@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_transpose_dynamic_int64():
    transpose_d(np.int64)

@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_transpose_dynamic_two_inputs_int64():
    transpose_d2(np.int64)

@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_transpose_dynamic_float32():
    transpose_d(np.float32)

@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_transpose_dynamic_float16():
    transpose_d(np.float16)

@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_transpose_dynamic_int32():
    transpose_d(np.int32)

@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_transpose_dynamic_two_inputs_float32():
    transpose_d2(np.float32)

@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_transpose_dynamic_two_inputs_float16():
    transpose_d2(np.float16)

@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_transpose_dynamic_two_inputs_int32():
    transpose_d2(np.int32)
