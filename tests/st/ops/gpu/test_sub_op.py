# Copyright 2019-2021 Huawei Technologies Co., Ltd
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
from mindspore.ops import operations as P
from mindspore.ops.operations import _inner_ops as inner

class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.sub = P.Sub()

    def construct(self, x, y):
        return self.sub(x, y)

class NetDynamic(nn.Cell):
    def __init__(self):
        super(NetDynamic, self).__init__()
        self.d = inner.GpuConvertToDynamicShape()
        self.sub = P.Sub()

    def construct(self, x, y):
        x = self.d(x)
        y = self.d(y)
        out = self.sub(x, y)
        return out


def sub(nptype):
    np_x0 = np.random.uniform(-2, 2, (2, 3, 4, 4)).astype(nptype)
    np_y0 = np.random.uniform(-2, 2, (2, 3, 4, 4)).astype(nptype)
    np_x1 = np.random.uniform(-2, 2, (2, 3, 4, 4)).astype(nptype)
    np_y1 = np.random.uniform(-2, 2, (2, 1, 4, 4)).astype(nptype)
    np_x2 = np.random.uniform(-2, 2, (2, 1, 1, 4)).astype(nptype)
    np_y2 = np.random.uniform(-2, 2, (2, 3, 4, 4)).astype(nptype)
    np_x3 = np.random.uniform(-2, 2, 1).astype(nptype)
    np_y3 = np.random.uniform(-2, 2, 1).astype(nptype)
    np_x4 = np.array(768).astype(nptype)
    np_y4 = np.array(3072.5).astype(nptype)
    x0 = Tensor(np_x0)
    y0 = Tensor(np_y0)
    x1 = Tensor(np_x1)
    y1 = Tensor(np_y1)
    x2 = Tensor(np_x2)
    y2 = Tensor(np_y2)
    x3 = Tensor(np_x3)
    y3 = Tensor(np_y3)
    x4 = Tensor(np_x4)
    y4 = Tensor(np_y4)

    expect0 = np.subtract(np_x0, np_y0)
    error0 = np.ones(shape=expect0.shape) * 1.0e-5
    expect1 = np.subtract(np_x1, np_y1)
    error1 = np.ones(shape=expect1.shape) * 1.0e-5
    expect2 = np.subtract(np_x2, np_y2)
    error2 = np.ones(shape=expect2.shape) * 1.0e-5
    expect3 = np.subtract(np_x3, np_y3)
    error3 = np.ones(shape=expect3.shape) * 1.0e-5
    expect4 = np.subtract(np_x4, np_y4)
    error4 = np.ones(shape=expect4.shape) * 1.0e-5

    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
    sub_net = Net()
    output0 = sub_net(x0, y0)
    output1 = sub_net(x1, y1)
    output2 = sub_net(x2, y2)
    output3 = sub_net(x3, y3)
    output4 = sub_net(x4, y4)
    diff0 = output0.asnumpy() - expect0
    assert np.all(diff0 < error0)
    assert output0.shape == expect0.shape
    diff1 = output1.asnumpy() - expect1
    assert np.all(diff1 < error1)
    assert output1.shape == expect1.shape
    diff2 = output2.asnumpy() - expect2
    assert np.all(diff2 < error2)
    assert output2.shape == expect2.shape
    diff3 = output3.asnumpy() - expect3
    assert np.all(diff3 < error3)
    assert output3.shape == expect3.shape
    diff4 = output4.asnumpy() - expect4
    assert np.all(diff4 < error4)
    assert output4.shape == expect4.shape

    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    sub_net = Net()
    output0 = sub_net(x0, y0)
    output1 = sub_net(x1, y1)
    output2 = sub_net(x2, y2)
    output3 = sub_net(x3, y3)
    output4 = sub_net(x4, y4)
    diff0 = output0.asnumpy() - expect0
    assert np.all(diff0 < error0)
    assert output0.shape == expect0.shape
    diff1 = output1.asnumpy() - expect1
    assert np.all(diff1 < error1)
    assert output1.shape == expect1.shape
    diff2 = output2.asnumpy() - expect2
    assert np.all(diff2 < error2)
    assert output2.shape == expect2.shape
    diff3 = output3.asnumpy() - expect3
    assert np.all(diff3 < error3)
    assert output3.shape == expect3.shape
    diff4 = output4.asnumpy() - expect4
    assert np.all(diff4 < error4)
    assert output4.shape == expect4.shape

    #dynamic shape
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    d_sub_net = NetDynamic()
    output3 = d_sub_net(x3, y3)
    output0 = d_sub_net(x0, y0)
    diff3 = output3.asnumpy() - expect3
    assert np.all(diff3 < error3)
    assert output3.shape == expect3.shape
    diff0 = output0.asnumpy() - expect0
    assert np.all(diff0 < error0)
    assert output0.shape == expect0.shape


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_sub_float64():
    sub(np.float64)

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_sub_float32():
    sub(np.float32)

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_sub_float16():
    sub(np.float16)

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_sub_int64():
    sub(np.int64)

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_sub_int32():
    sub(np.int32)
