# Copyright 2020-2021 Huawei Technologies Co., Ltd
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

class NetDiv(nn.Cell):
    def __init__(self):
        super(NetDiv, self).__init__()
        self.div = P.Div()

    def construct(self, x, y):
        return self.div(x, y)

def div(nptype):
    x0_np = np.random.randint(1, 5, (2, 3, 4, 4)).astype(nptype)
    y0_np = np.random.randint(1, 5, (2, 3, 4, 4)).astype(nptype)
    x1_np = np.random.randint(1, 5, (2, 3, 4, 4)).astype(nptype)
    y1_np = np.random.randint(1, 5, (2, 1, 4, 4)).astype(nptype)
    x2_np = np.random.randint(1, 5, (2, 1, 1, 4)).astype(nptype)
    y2_np = np.random.randint(1, 5, (2, 3, 4, 4)).astype(nptype)
    x3_np = np.random.randint(1, 5, 1).astype(nptype)
    y3_np = np.random.randint(1, 5, 1).astype(nptype)
    x4_np = np.array(78).astype(nptype)
    y4_np = np.array(37.5).astype(nptype)

    x0 = Tensor(x0_np)
    y0 = Tensor(y0_np)
    x1 = Tensor(x1_np)
    y1 = Tensor(y1_np)
    x2 = Tensor(x2_np)
    y2 = Tensor(y2_np)
    x3 = Tensor(x3_np)
    y3 = Tensor(y3_np)
    x4 = Tensor(x4_np)
    y4 = Tensor(y4_np)

    context.set_context(mode=context.GRAPH_MODE, device_target='GPU')
    div_net = NetDiv()
    output0 = div_net(x0, y0)
    expect0 = np.divide(x0_np, y0_np)
    diff0 = output0.asnumpy() - expect0
    error0 = np.ones(shape=expect0.shape) * 1.0e-5
    assert np.all(diff0 < error0)
    assert output0.shape == expect0.shape

    output1 = div_net(x1, y1)
    expect1 = np.divide(x1_np, y1_np)
    diff1 = output1.asnumpy() - expect1
    error1 = np.ones(shape=expect1.shape) * 1.0e-5
    assert np.all(diff1 < error1)
    assert output1.shape == expect1.shape

    output2 = div_net(x2, y2)
    expect2 = np.divide(x2_np, y2_np)
    diff2 = output2.asnumpy() - expect2
    error2 = np.ones(shape=expect2.shape) * 1.0e-5
    assert np.all(diff2 < error2)
    assert output2.shape == expect2.shape

    context.set_context(mode=context.PYNATIVE_MODE, device_target='GPU')
    output3 = div_net(x3, y3)
    expect3 = np.divide(x3_np, y3_np)
    diff3 = output3.asnumpy() - expect3
    error3 = np.ones(shape=expect3.shape) * 1.0e-5
    assert np.all(diff3 < error3)
    assert output3.shape == expect3.shape

    output4 = div_net(x4, y4)
    expect4 = np.divide(x4_np, y4_np)
    diff4 = output4.asnumpy() - expect4
    error4 = np.ones(shape=expect4.shape) * 1.0e-5
    assert np.all(diff4 < error4)
    assert output4.shape == expect4.shape

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_div_float64():
    div(np.float64)

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_div_float32():
    div(np.float32)

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_div_float16():
    div(np.float16)

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_div_int64():
    div(np.int64)

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_div_int32():
    div(np.int32)
