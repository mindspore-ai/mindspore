# Copyright 2024 Huawei Technologies Co., Ltd
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
import mindspore as ms
from mindspore import context, Tensor
from bfloat16 import bfloat16

ms.set_context(mode=context.GRAPH_MODE, device_target="Ascend")


class StridedSlice(ms.nn.Cell):
    def __init__(self, begin_mask=0, end_mask=0, ellipsis_mask=0, new_axis_mask=0, shrink_axis_mask=0):
        super().__init__()
        self.net = ms.ops.StridedSlice(begin_mask, end_mask, ellipsis_mask, new_axis_mask, shrink_axis_mask)

    def construct(self, input_x, begin, end, strides):
        output = self.net(input_x, begin, end, strides)
        return output



def stride_slice(npDtype=np.float16, msDtype=ms.float16):
    input_x = np.random.random([3, 4, 5])
    sslice = StridedSlice()
    output = sslice(Tensor(input_x, msDtype), (1, 0, 1), (2, 3, 4), (1, 1, 1))
    expected = input_x[1:2, 0:3, 1:4].astype(npDtype)
    np.testing.assert_array_almost_equal(output.asnumpy(), expected)

def stride_slice_bf16():
    input_x = np.random.random([3, 4, 5])
    sslice = StridedSlice()
    output = sslice(Tensor(input_x, ms.bfloat16), (1, 0, 1), (2, 3, 4), (1, 1, 1))
    output = ms.ops.cast(output, ms.float32)
    expected = input_x[1:2, 0:3, 1:4].astype(bfloat16)
    expected = expected.astype(np.float32)
    np.testing.assert_array_almost_equal(output.asnumpy(), expected)

def test_strided_slice_bf16():
    """
    Feature: test strided_slice operator in graph mode
    Description: test strided_slice datatype: bf16
    Expectation: the result is correct
    """
    stride_slice_bf16()

def test_strided_slice_float16():
    """
    Feature: test strided_slice operator in graph mode
    Description: test strided_slice datatype: float16
    Expectation: the result is correct
    """
    stride_slice(np.float16, ms.float16)

def test_strided_slice_int8():
    """
    Feature: test strided_slice operator in graph mode
    Description: test strided_slice datatype: int8
    Expectation: the result is correct
    """
    stride_slice(np.int8, ms.int8)

def test_strided_slice_uint8():
    """
    Feature: test strided_slice operator in graph mode
    Description: test strided_slice datatype: uint8
    Expectation: the result is correct
    """
    stride_slice(np.uint8, ms.uint8)

def test_strided_slice_int32():
    """
    Feature: test strided_slice operator in graph mode
    Description: test strided_slice datatype: int32
    Expectation: the result is correct
    """
    stride_slice(np.int32, ms.int32)

def test_strided_slice_uint32():
    """
    Feature: test strided_slice operator in graph mode
    Description: test strided_slice datatype: uint32
    Expectation: the result is correct
    """
    stride_slice(np.uint32, ms.uint32)
