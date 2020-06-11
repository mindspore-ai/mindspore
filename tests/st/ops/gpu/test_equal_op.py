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


class NetEqual(Cell):
    def __init__(self):
        super(NetEqual, self).__init__()
        self.Equal = P.Equal()

    def construct(self, x, y):
        return self.Equal(x, y)

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

    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
    equal = NetEqual()
    output0 = equal(x0, y0)
    assert np.all(output0.asnumpy() == expect0)
    assert output0.shape() == expect0.shape
    output1 = equal(x1, y1)
    assert np.all(output1.asnumpy() == expect1)
    assert output1.shape() == expect1.shape

    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    equal = NetEqual()
    output0 = equal(x0, y0)
    assert np.all(output0.asnumpy() == expect0)
    assert output0.shape() == expect0.shape
    output1 = equal(x1, y1)
    assert np.all(output1.asnumpy() == expect1)
    assert output1.shape() == expect1.shape


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_notequal():
    x0 = Tensor(np.array([[1.2, 1], [1, 0]]).astype(np.float32))
    y0 = Tensor(np.array([[1, 2]]).astype(np.float32))
    expect0 = np.array([[True, True], [False, True]])

    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
    notequal = NetNotEqual()
    output0 = notequal(x0, y0)
    assert np.all(output0.asnumpy() == expect0)
    assert output0.shape() == expect0.shape

    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    notequal = NetNotEqual()
    output0 = notequal(x0, y0)
    assert np.all(output0.asnumpy() == expect0)
    assert output0.shape() == expect0.shape


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_greaterqual():
    x0 = Tensor(np.array([[1.2, 1], [1, 0]]).astype(np.float32))
    y0 = Tensor(np.array([[1, 2]]).astype(np.float32))
    expect0 = np.array([[True, False], [True, False]])

    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
    gequal = NetGreaterEqual()
    output0 = gequal(x0, y0)
    assert np.all(output0.asnumpy() == expect0)
    assert output0.shape() == expect0.shape

    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    gequal = NetGreaterEqual()
    output0 = gequal(x0, y0)
    assert np.all(output0.asnumpy() == expect0)
    assert output0.shape() == expect0.shape
