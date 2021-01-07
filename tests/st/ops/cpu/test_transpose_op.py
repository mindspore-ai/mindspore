# Copyright 2021 Huawei Technologies Co., Ltd
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

import pytest
import numpy as np
from mindspore import Tensor
from mindspore.ops import operations as P
from mindspore.common.api import ms_function
from mindspore.common.initializer import initializer
from mindspore.common.parameter import Parameter
import mindspore.nn as nn
import mindspore.context as context

context.set_context(device_target='CPU')


class Transpose(nn.Cell):
    def __init__(self):
        super(Transpose, self).__init__()
        self.transpose = P.Transpose()

        self.x_2D = Parameter(initializer(Tensor(np.arange(5 * 6).reshape(5, 6).astype(np.float32)), [5, 6]),
                              name='x_2D')
        self.perm_2D = (1, 0)

        self.x_3D = Parameter(initializer(Tensor(np.arange(2 * 2 * 4).reshape(2, 2, 4).astype(np.float32)), [2, 2, 4]),
                              name='x_3D')
        self.perm_3D = (1, 0, 2)

        self.x_4D = Parameter(
            initializer(Tensor(np.arange(2 * 3 * 4 * 5).reshape(2,
                                                                3, 4, 5).astype(np.float32)), [2, 3, 4, 5]),
            name='x_4D')
        self.perm_4D = (0, 1, 2, 3)

        self.x_5D = Parameter(
            initializer(Tensor(np.arange(1 * 2 * 3 * 4 * 5).reshape(1, 2, 3, 4, 5).astype(np.float32)),
                        [1, 2, 3, 4, 5]), name='x_5D')
        self.perm_5D = (1, 0, 3, 4, 2)

    @ms_function
    def construct(self):
        return (self.transpose(self.x_2D, self.perm_2D), self.transpose(self.x_3D, self.perm_3D),
                self.transpose(self.x_4D, self.perm_4D), self.transpose(self.x_5D, self.perm_5D))


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_transpose():
    transpose = Transpose()
    output = transpose()

    expect0 = np.array([[[0, 6, 12, 18, 24],
                         [1, 7, 13, 19, 25],
                         [2, 8, 14, 20, 26],
                         [3, 9, 15, 21, 27],
                         [4, 10, 16, 22, 28],
                         [5, 11, 17, 23, 29]]]).astype(np.float32)
    expect1 = np.array([[[[0, 1, 2, 3],
                          [8, 9, 10, 11]],
                         [[4, 5, 6, 7],
                          [12, 13, 14, 15]]]]).astype(np.float32)
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
                           [115, 116, 117, 118, 119]]]]]).astype(np.float32)
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
                            [79, 99, 119]]]]]]).astype(np.float32)
    assert (output[0].asnumpy() == expect0).all()
    assert (output[1].asnumpy() == expect1).all()
    assert (output[2].asnumpy() == expect2).all()
    assert (output[3].asnumpy() == expect3).all()


test_transpose()


class Transpose_int64(nn.Cell):
    def __init__(self):
        super(Transpose_int64, self).__init__()
        self.transpose = P.Transpose()

        self.x_2D = Parameter(initializer(Tensor(np.arange(5 * 6).reshape(5, 6).astype(np.int64)), [5, 6]),
                              name='x_2D')
        self.perm_2D = (1, 0)

        self.x_3D = Parameter(initializer(Tensor(np.arange(2 * 2 * 4).reshape(2, 2, 4).astype(np.int64)), [2, 2, 4]),
                              name='x_3D')
        self.perm_3D = (1, 0, 2)

        self.x_4D = Parameter(
            initializer(Tensor(np.arange(2 * 3 * 4 * 5).reshape(2,
                                                                3, 4, 5).astype(np.int64)), [2, 3, 4, 5]),
            name='x_4D')
        self.perm_4D = (0, 1, 2, 3)

        self.x_5D = Parameter(
            initializer(Tensor(np.arange(1 * 2 * 3 * 4 * 5).reshape(1, 2, 3, 4, 5).astype(np.int64)),
                        [1, 2, 3, 4, 5]), name='x_5D')
        self.perm_5D = (1, 0, 3, 4, 2)

    @ms_function
    def construct(self):
        return (self.transpose(self.x_2D, self.perm_2D), self.transpose(self.x_3D, self.perm_3D),
                self.transpose(self.x_4D, self.perm_4D), self.transpose(self.x_5D, self.perm_5D))


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_transpose_int64():
    transpose = Transpose_int64()
    output = transpose()

    expect0 = np.array([[[0, 6, 12, 18, 24],
                         [1, 7, 13, 19, 25],
                         [2, 8, 14, 20, 26],
                         [3, 9, 15, 21, 27],
                         [4, 10, 16, 22, 28],
                         [5, 11, 17, 23, 29]]]).astype(np.int64)
    expect1 = np.array([[[[0, 1, 2, 3],
                          [8, 9, 10, 11]],
                         [[4, 5, 6, 7],
                          [12, 13, 14, 15]]]]).astype(np.int64)
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
                           [115, 116, 117, 118, 119]]]]]).astype(np.int64)
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
                            [79, 99, 119]]]]]]).astype(np.int64)
    assert (output[0].asnumpy() == expect0).all()
    assert (output[1].asnumpy() == expect1).all()
    assert (output[2].asnumpy() == expect2).all()
    assert (output[3].asnumpy() == expect3).all()


test_transpose_int64()


class Transpose_uint8(nn.Cell):
    def __init__(self):
        super(Transpose_uint8, self).__init__()
        self.transpose = P.Transpose()

        self.x_2D = Parameter(initializer(Tensor(np.arange(5 * 6).reshape(5, 6).astype(np.uint8)), [5, 6]),
                              name='x_2D')
        self.perm_2D = (1, 0)

        self.x_3D = Parameter(initializer(Tensor(np.arange(2 * 2 * 4).reshape(2, 2, 4).astype(np.uint8)), [2, 2, 4]),
                              name='x_3D')
        self.perm_3D = (1, 0, 2)

        self.x_4D = Parameter(
            initializer(Tensor(np.arange(2 * 3 * 4 * 5).reshape(2,
                                                                3, 4, 5).astype(np.uint8)), [2, 3, 4, 5]),
            name='x_4D')
        self.perm_4D = (0, 1, 2, 3)

        self.x_5D = Parameter(
            initializer(Tensor(np.arange(1 * 2 * 3 * 4 * 5).reshape(1, 2, 3, 4, 5).astype(np.uint8)),
                        [1, 2, 3, 4, 5]), name='x_5D')
        self.perm_5D = (1, 0, 3, 4, 2)

    @ms_function
    def construct(self):
        return (self.transpose(self.x_2D, self.perm_2D), self.transpose(self.x_3D, self.perm_3D),
                self.transpose(self.x_4D, self.perm_4D), self.transpose(self.x_5D, self.perm_5D))


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_transpose_uint8():
    transpose = Transpose_uint8()
    output = transpose()

    expect0 = np.array([[[0, 6, 12, 18, 24],
                         [1, 7, 13, 19, 25],
                         [2, 8, 14, 20, 26],
                         [3, 9, 15, 21, 27],
                         [4, 10, 16, 22, 28],
                         [5, 11, 17, 23, 29]]]).astype(np.uint8)
    expect1 = np.array([[[[0, 1, 2, 3],
                          [8, 9, 10, 11]],
                         [[4, 5, 6, 7],
                          [12, 13, 14, 15]]]]).astype(np.uint8)
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
                           [115, 116, 117, 118, 119]]]]]).astype(np.uint8)
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
                            [79, 99, 119]]]]]]).astype(np.uint8)
    assert (output[0].asnumpy() == expect0).all()
    assert (output[1].asnumpy() == expect1).all()
    assert (output[2].asnumpy() == expect2).all()
    assert (output[3].asnumpy() == expect3).all()


test_transpose_uint8()
