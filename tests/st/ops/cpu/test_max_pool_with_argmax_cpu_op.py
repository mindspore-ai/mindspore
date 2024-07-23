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


import numpy as np
import pytest
import mindspore as ms
from mindspore import Tensor
import mindspore.context as context
import mindspore.nn as nn
import mindspore.ops.operations as P


context.set_context(mode=context.GRAPH_MODE, device_target="CPU")


class MaxPoolWithArgmaxOp(nn.Cell):
    def __init__(self, kernel_size=1, strides=1, pad_mode="valid", data_format="NCHW"):
        super(MaxPoolWithArgmaxOp, self).__init__()
        self.max_pool_op = P.MaxPoolWithArgmax(
            kernel_size=kernel_size, strides=strides, pad_mode=pad_mode, data_format=data_format)

    def construct(self, x):
        return self.max_pool_op(x)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_maxpool_with_argmax_valid():
    """
    Feature: test maxPoolWithArgmax cpu op.
    Description: document test with 'VALID' padding
    Expectation: expect correct result.
    """
    x = Tensor(np.array([[[[10, 1, 2, 3, -4, -5],
                           [6, 7, 8, 9, -10, -11],
                           [12, 13, 24, -15, -16, -17],
                           [18, 19, 20, 21, 22, 23],
                           [32, 25, 26, 27, 28, 40],
                           [30, 31, 35, 33, 34, 35]]]]).astype(np.float32))
    maxpool_with_argmax = MaxPoolWithArgmaxOp(kernel_size=2, strides=2, pad_mode="valid", data_format="NCHW")
    actual_output, argmax = maxpool_with_argmax(x)
    expect_output = np.array([[[[10, 9, -4],
                                [19, 24, 23],
                                [32, 35, 40]]]]).astype(np.float32)
    expect_argmax = np.array([[[[0, 9, 4],
                                [19, 14, 23],
                                [24, 32, 29]]]]).astype(np.int32)
    assert (actual_output.asnumpy() == expect_output).all()
    assert (argmax.asnumpy() == expect_argmax).all()


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_maxpool_with_argmax_same():
    """
    Feature: test maxPoolWithArgmax cpu.
    Description: document test with 'SAME' padding
    Expectation: expect correct result.
    """
    x = Tensor(np.array([[[[0, 1, 2, 3, -4, -5],
                           [6, 7, 8, 9, -10, -11],
                           [12, 13, 14, -15, -16, -17],
                           [18, 19, 20, 21, 22, 23],
                           [24, 25, 26, 27, 28, 29],
                           [30, 31, 32, 33, 34, 35]]]]).astype(np.float32))
    maxpool_with_argmax = MaxPoolWithArgmaxOp(kernel_size=3, strides=2, pad_mode="same", data_format="NCHW")
    actual_output, argmax = maxpool_with_argmax(x)
    expect_output = np.array([[[[14, 14, -4],
                                [26, 28, 29],
                                [32, 34, 35]]]])
    expect_argmax = np.array([[[[14, 14, 4],
                                [26, 28, 29],
                                [32, 34, 35]]]])
    assert (actual_output.asnumpy() == expect_output).all()
    assert (argmax.asnumpy() == expect_argmax).all()


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_maxpool_with_argmax_tensorflow():
    """
    Feature: test maxPoolWithArgmax op.
    Description: comparing to tensorflow's MaxPoolWithArgmax with 'VALID' padding
    Expectation: expect correct result.
    """
    x = Tensor(np.array([[[[80., 70., -19.],
                           [14., 46., -67.],
                           [15., -41., -80.]],
                          [[-80., 62., 85.],
                           [-3., -68., 35.],
                           [84., 54., 32.]]],
                         [[[-65., 57., 10.],
                           [-1., 38., -43.],
                           [36., -64., 5.]],
                          [[32., 8., 70.],
                           [-20., -92., -31.],
                           [-73., -27., -87.]]]]).astype(np.float32))
    maxpool_with_argmax = MaxPoolWithArgmaxOp(kernel_size=2, strides=1, pad_mode="valid", data_format="NCHW")
    actual_output, argmax = maxpool_with_argmax(x)
    expect_output = np.array([[[[80., 70.],
                                [46., 46.]],
                               [[62., 85.],
                                [84., 54.]]],
                              [[[57., 57.],
                                [38., 38.]],
                               [[32., 70.],
                                [-20., -27.]]]])
    expect_argmax = np.array([[[[0, 1],
                                [4, 4]],
                               [[10, 11],
                                [15, 16]]],
                              [[[1, 1],
                                [4, 4]],
                               [[9, 11],
                                [12, 16]]]])
    assert (actual_output.asnumpy() == expect_output).all()
    assert (argmax.asnumpy() == expect_argmax).all()
    assert np.allclose(actual_output.asnumpy(), expect_output)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_maxpool_with_argmax_dynamic_shape():
    """
    Feature: test maxPoolWithArgmax op with dynamic shapes.
    Description: input x has the dynamic shapes
    Expectation: expect correct result.
    """
    x = Tensor(np.array([[[[80., 70., -19.],
                           [14., 46., -67.],
                           [15., -41., -80.]],
                          [[-80., 62., 85.],
                           [-3., -68., 35.],
                           [84., 54., 32.]]],
                         [[[-65., 57., 10.],
                           [-1., 38., -43.],
                           [36., -64., 5.]],
                          [[32., 8., 70.],
                           [-20., -92., -31.],
                           [-73., -27., -87.]]]]), dtype=ms.float32)
    x_dyn = Tensor(shape=[2, None, 3, 3], dtype=ms.float32)
    maxpool_with_argmax = MaxPoolWithArgmaxOp(kernel_size=2, strides=1, pad_mode="valid", data_format="NCHW")
    maxpool_with_argmax.set_inputs(x_dyn)
    actual_output, argmax = maxpool_with_argmax(x)
    expect_output = np.array([[[[80., 70.],
                                [46., 46.]],
                               [[62., 85.],
                                [84., 54.]]],
                              [[[57., 57.],
                                [38., 38.]],
                               [[32., 70.],
                                [-20., -27.]]]])
    expect_argmax = np.array([[[[0, 1],
                                [4, 4]],
                               [[10, 11],
                                [15, 16]]],
                              [[[1, 1],
                                [4, 4]],
                               [[9, 11],
                                [12, 16]]]])
    assert (actual_output.asnumpy() == expect_output).all()
    assert (argmax.asnumpy() == expect_argmax).all()
