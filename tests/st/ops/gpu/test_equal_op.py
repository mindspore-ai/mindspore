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
from mindspore.common.tensor import Tensor
from mindspore.nn import Cell
from mindspore.ops import operations as P
from mindspore.ops.operations import _inner_ops as inner


class NetEqual(Cell):
    def __init__(self):
        super(NetEqual, self).__init__()
        self.Equal = P.Equal()

    def construct(self, x, y):
        return self.Equal(x, y)

class NetEqualDynamic(Cell):
    def __init__(self):
        super(NetEqualDynamic, self).__init__()
        self.conv = inner.GpuConvertToDynamicShape()
        self.Equal = P.Equal()

    def construct(self, x, y):
        x_conv = self.conv(x)
        y_conv = self.conv(y)
        return self.Equal(x_conv, y_conv)

class NetNotEqual(Cell):
    def __init__(self):
        super(NetNotEqual, self).__init__()
        self.NotEqual = P.NotEqual()

    def construct(self, x, y):
        return self.NotEqual(x, y)

class NetGreaterEqual(Cell):
    def __init__(self):
        super(NetGreaterEqual, self).__init__()
        self.GreaterEqual = P.GreaterEqual()

    def construct(self, x, y):
        return self.GreaterEqual(x, y)

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_equal():
    x0_np = np.arange(24).reshape((4, 3, 2)).astype(np.float32)
    x0 = Tensor(x0_np)
    y0_np = np.arange(24).reshape((4, 3, 2)).astype(np.float32)
    y0 = Tensor(y0_np)
    expect0 = np.equal(x0_np, y0_np)
    x1_np = np.array([0, 1, 3]).astype(np.float32)
    x1 = Tensor(x1_np)
    y1_np = np.array([0, 1, -3]).astype(np.float32)
    y1 = Tensor(y1_np)
    expect1 = np.equal(x1_np, y1_np)
    x2_np = np.array([0, 1, 3]).astype(np.int32)
    x2 = Tensor(x2_np)
    y2_np = np.array([0, 1, -3]).astype(np.int32)
    y2 = Tensor(y2_np)
    expect2 = np.equal(x2_np, y2_np)
    x3_np = np.array([0, 1, 3]).astype(np.int16)
    x3 = Tensor(x3_np)
    y3_np = np.array([0, 1, -3]).astype(np.int16)
    y3 = Tensor(y3_np)
    expect3 = np.equal(x3_np, y3_np)
    x4_np = np.array([0, 1, 4]).astype(np.uint8)
    x4 = Tensor(x4_np)
    y4_np = np.array([0, 1, 3]).astype(np.uint8)
    y4 = Tensor(y4_np)
    expect4 = np.equal(x4_np, y4_np)
    x5_np = np.array([True, False, True]).astype(bool)
    x5 = Tensor(x5_np)
    y5_np = np.array([True, False, False]).astype(bool)
    y5 = Tensor(y5_np)
    expect5 = np.equal(x5_np, y5_np)
    x6_np = np.array([0, 1, 4]).astype(np.int8)
    x6 = Tensor(x4_np)
    y6_np = np.array([0, 1, 3]).astype(np.int8)
    y6 = Tensor(y4_np)
    expect6 = np.equal(x6_np, y6_np)
    x7_np = np.array([0, 1, 4]).astype(np.int64)
    x7 = Tensor(x4_np)
    y7_np = np.array([0, 1, 3]).astype(np.int64)
    y7 = Tensor(y4_np)
    expect7 = np.equal(x7_np, y7_np)
    x8_np = np.array([0, 1, 4]).astype(np.float16)
    x8 = Tensor(x4_np)
    y8_np = np.array([0, 1, 3]).astype(np.float16)
    y8 = Tensor(y4_np)
    expect8 = np.equal(x8_np, y8_np)

    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
    equal = NetEqual()
    output0 = equal(x0, y0)
    assert np.all(output0.asnumpy() == expect0)
    assert output0.shape == expect0.shape
    output1 = equal(x1, y1)
    assert np.all(output1.asnumpy() == expect1)
    assert output1.shape == expect1.shape
    output2 = equal(x2, y2)
    assert np.all(output2.asnumpy() == expect2)
    assert output2.shape == expect2.shape
    output3 = equal(x3, y3)
    assert np.all(output3.asnumpy() == expect3)
    assert output3.shape == expect3.shape
    output4 = equal(x4, y4)
    assert np.all(output4.asnumpy() == expect4)
    assert output4.shape == expect4.shape
    output5 = equal(x5, y5)
    assert np.all(output5.asnumpy() == expect5)
    assert output5.shape == expect5.shape



    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    equal = NetEqual()
    output0 = equal(x0, y0)
    assert np.all(output0.asnumpy() == expect0)
    assert output0.shape == expect0.shape
    output1 = equal(x1, y1)
    assert np.all(output1.asnumpy() == expect1)
    assert output1.shape == expect1.shape
    output2 = equal(x2, y2)
    assert np.all(output2.asnumpy() == expect2)
    assert output2.shape == expect2.shape
    output3 = equal(x3, y3)
    assert np.all(output3.asnumpy() == expect3)
    assert output3.shape == expect3.shape
    output4 = equal(x4, y4)
    assert np.all(output4.asnumpy() == expect4)
    assert output4.shape == expect4.shape
    output5 = equal(x5, y5)
    assert np.all(output5.asnumpy() == expect5)
    assert output5.shape == expect5.shape
    output6 = equal(x6, y6)
    assert np.all(output6.asnumpy() == expect6)
    assert output6.shape == expect6.shape
    output7 = equal(x7, y7)
    assert np.all(output7.asnumpy() == expect7)
    assert output7.shape == expect7.shape
    output8 = equal(x8, y8)
    assert np.all(output8.asnumpy() == expect8)
    assert output8.shape == expect8.shape


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_notequal():
    x0 = Tensor(np.array([[1.2, 1], [1, 0]]).astype(np.float32))
    y0 = Tensor(np.array([[1, 2]]).astype(np.float32))
    expect0 = np.array([[True, True], [False, True]])
    x1 = Tensor(np.array([[2, 1], [1, 1]]).astype(np.int16))
    y1 = Tensor(np.array([[1, 2]]).astype(np.int16))
    expect1 = np.array([[True, True], [False, True]])
    x2 = Tensor(np.array([[2, 1], [1, 2]]).astype(np.uint8))
    y2 = Tensor(np.array([[1, 2]]).astype(np.uint8))
    expect2 = np.array([[True, True], [False, False]])
    x3 = Tensor(np.array([[False, True], [True, False]]).astype(bool))
    y3 = Tensor(np.array([[True, False]]).astype(bool))
    expect3 = np.array([[True, True], [False, False]])

    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
    notequal = NetNotEqual()
    output0 = notequal(x0, y0)
    assert np.all(output0.asnumpy() == expect0)
    assert output0.shape == expect0.shape
    output1 = notequal(x1, y1)
    assert np.all(output1.asnumpy() == expect1)
    assert output1.shape == expect1.shape
    output2 = notequal(x2, y2)
    assert np.all(output2.asnumpy() == expect2)
    assert output2.shape == expect2.shape
    output3 = notequal(x3, y3)
    assert np.all(output3.asnumpy() == expect3)
    assert output3.shape == expect3.shape

    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    notequal = NetNotEqual()
    output0 = notequal(x0, y0)
    assert np.all(output0.asnumpy() == expect0)
    assert output0.shape == expect0.shape
    output1 = notequal(x1, y1)
    assert np.all(output1.asnumpy() == expect1)
    assert output1.shape == expect1.shape
    output2 = notequal(x2, y2)
    assert np.all(output2.asnumpy() == expect2)
    assert output2.shape == expect2.shape
    output3 = notequal(x3, y3)
    assert np.all(output3.asnumpy() == expect3)
    assert output3.shape == expect3.shape


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_greaterqual():
    x0 = Tensor(np.array([[1.2, 1], [1, 0]]).astype(np.float32))
    y0 = Tensor(np.array([[1, 2]]).astype(np.float32))
    expect0 = np.array([[True, False], [True, False]])
    x1 = Tensor(np.array([[2, 1], [1, 1]]).astype(np.int16))
    y1 = Tensor(np.array([[1, 2]]).astype(np.int16))
    expect1 = np.array([[True, False], [True, False]])
    x2 = Tensor(np.array([[2, 1], [1, 2]]).astype(np.uint8))
    y2 = Tensor(np.array([[1, 2]]).astype(np.uint8))
    expect2 = np.array([[True, False], [True, True]])

    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
    gequal = NetGreaterEqual()
    output0 = gequal(x0, y0)
    assert np.all(output0.asnumpy() == expect0)
    assert output0.shape == expect0.shape
    output1 = gequal(x1, y1)
    assert np.all(output1.asnumpy() == expect1)
    assert output1.shape == expect1.shape
    output2 = gequal(x2, y2)
    assert np.all(output2.asnumpy() == expect2)
    assert output2.shape == expect2.shape

    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    gequal = NetGreaterEqual()
    output0 = gequal(x0, y0)
    assert np.all(output0.asnumpy() == expect0)
    assert output0.shape == expect0.shape
    output1 = gequal(x1, y1)
    assert np.all(output1.asnumpy() == expect1)
    assert output1.shape == expect1.shape
    output2 = gequal(x2, y2)
    assert np.all(output2.asnumpy() == expect2)
    assert output2.shape == expect2.shape


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_equal_dynamic_shape():
    x0_np = np.arange(24).reshape((4, 3, 2)).astype(np.float32)
    x0 = Tensor(x0_np)
    y0_np = np.arange(24).reshape((4, 3, 2)).astype(np.float32)
    y0 = Tensor(y0_np)
    expect0 = np.equal(x0_np, y0_np)

    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    equal = NetEqualDynamic()
    output0 = equal(x0, y0)
    assert np.all(output0.asnumpy() == expect0)
    assert output0.shape == expect0.shape
