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
"""GPU Col2Im primitive coverage with NumPy numerical references."""
from tests.mark_utils import arg_mark

import numpy as np
import pytest

import mindspore.context as context
import mindspore.nn as nn
from mindspore import dtype as mstype
from mindspore import Tensor
from mindspore.ops.operations.array_ops import Col2Im

np.random.seed(1)


class Col2ImTest(nn.Cell):

    def __init__(self, kernel_size, dilation, padding, stride):
        super().__init__()
        self.c2i = Col2Im(kernel_size, dilation, padding, stride)

    def construct(self, x, output_size):
        return self.c2i(x, output_size)


def col2im_numpy(data, output_size, kernel_size, dilation, padding, stride):
    """Accumulate each block element at its dilated, unpadded output position."""
    height, width = output_size
    kernel_h, kernel_w = kernel_size
    blocks_h = (height + 2 * padding[0] - dilation[0] * (kernel_h - 1) - 1) // stride[0] + 1
    blocks_w = (width + 2 * padding[1] - dilation[1] * (kernel_w - 1) - 1) // stride[1] + 1
    accumulator_dtype = np.complex128 if np.iscomplexobj(data) else np.float64
    expected = np.zeros((data.shape[0], data.shape[1], height, width), dtype=accumulator_dtype)
    for block_h in range(blocks_h):
        for block_w in range(blocks_w):
            for kernel_y in range(kernel_h):
                out_y = block_h * stride[0] - padding[0] + kernel_y * dilation[0]
                for kernel_x in range(kernel_w):
                    out_x = block_w * stride[1] - padding[1] + kernel_x * dilation[1]
                    if 0 <= out_y < height and 0 <= out_x < width:
                        expected[:, :, out_y, out_x] += data[:, :, kernel_y * kernel_w + kernel_x,
                                                            block_h * blocks_w + block_w]
    return expected


def dyn_case():
    kernel_size = [2, 2]
    dilation = [2, 2]
    padding = [2, 2]
    stride = [2, 2]
    col2im = Col2ImTest(kernel_size=kernel_size,
                        dilation=dilation,
                        padding=padding,
                        stride=stride)

    x_dyn = Tensor(shape=[None, None, None, None], dtype=mstype.float32)
    output_size_dyn = Tensor(shape=[None], dtype=mstype.int32)
    col2im.set_inputs(x_dyn, output_size_dyn)

    data = np.random.rand(16, 16, 4, 25).astype(np.float32)
    x = Tensor(data)
    output_size = Tensor([8, 8], dtype=mstype.int32)
    output = col2im(x, output_size)

    expect_shape = (16, 16, 8, 8)
    assert output.shape == expect_shape
    expected = col2im_numpy(data, (8, 8), kernel_size, dilation, padding, stride)
    np.testing.assert_allclose(output.asnumpy(), expected, rtol=1e-5, atol=1e-6)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard')
def test_col2im_dyn():
    """
    Feature: Col2Im function.
    Description:  test the ops in dynamic shape.
    Expectation: shape and values match the NumPy reference in both modes.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    dyn_case()
    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
    dyn_case()


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard')
@pytest.mark.parametrize("mode, input_type",
                         [(context.GRAPH_MODE, np.float32),
                          (context.PYNATIVE_MODE, np.float32),
                          (context.GRAPH_MODE, np.float16),
                          (context.PYNATIVE_MODE, np.float16),
                          (context.GRAPH_MODE, np.float64),
                          (context.PYNATIVE_MODE, np.float64),
                          (context.GRAPH_MODE, np.complex64),
                          (context.PYNATIVE_MODE, np.complex64),
                          (context.GRAPH_MODE, np.complex128),
                          (context.PYNATIVE_MODE, np.complex128)])
def test_col2im_op(mode, input_type):
    """
    Feature: Col2Im GPU kernel
    Description: test the retained Col2Im primitive for different dtypes and modes.
    Expectation: match to np benchmark.
    """
    context.set_context(mode=mode, device_target='GPU')
    data = np.random.rand(16, 16, 4, 25).astype(input_type)
    if np.issubdtype(input_type, np.complexfloating):
        data += 1j * np.random.rand(*data.shape).astype(input_type)
    x = Tensor(data)
    output_size = Tensor([8, 8], dtype=mstype.int32)
    kernel_size = [2, 2]
    dilation = [2, 2]
    padding = [2, 2]
    stride = [2, 2]
    expect_shape = (16, 16, 8, 8)
    col2im = Col2ImTest(kernel_size=kernel_size,
                        dilation=dilation,
                        padding=padding,
                        stride=stride)
    output = col2im(x, output_size)
    assert output.shape == expect_shape
    expected = col2im_numpy(data, (8, 8), kernel_size, dilation, padding, stride)
    if input_type == np.float16:
        tolerance = 1e-3
    elif input_type in (np.float32, np.complex64):
        tolerance = 1e-5
    else:
        tolerance = 1e-12
    np.testing.assert_allclose(output.asnumpy(), expected, rtol=tolerance, atol=tolerance)
