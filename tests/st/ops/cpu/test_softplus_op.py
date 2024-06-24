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
from tests.mark_utils import arg_mark

import numpy as np
import pytest

import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.ops import operations as P

context.set_context(mode=context.GRAPH_MODE, device_target="CPU")


class SoftplusNet(nn.Cell):
    def __init__(self):
        super(SoftplusNet, self).__init__()
        self.softplus = P.Softplus()

    def construct(self, x):
        return self.softplus(x)


def SoftplusCompute(x):
    return np.log(1 + np.exp(x))


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_softplus_0d_fp32():
    """
    Feature: test ops.Softplus
    Description: test the Softplus with scalar input
    Expectation: success
    """
    x_np = np.array(1.2, np.float32)
    y_np = SoftplusCompute(x_np)

    x_ms = Tensor(x_np)
    net = SoftplusNet()
    y_ms = net(x_ms)

    assert np.allclose(y_np, y_ms.asnumpy())


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_softplus_1d_fp32():
    """
    Feature: test ops.Softplus
    Description: test the Softplus with 1d fp32 inputs
    Expectation: success
    """
    x_np = np.random.random((50,)).astype(np.float32)
    y_np = SoftplusCompute(x_np)

    x_ms = Tensor(x_np)
    net = SoftplusNet()
    y_ms = net(x_ms)

    assert np.allclose(y_np, y_ms.asnumpy())


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_softplus_2d_fp32():
    """
    Feature: test ops.Softplus
    Description: test the Softplus with 2d fp32 inputs
    Expectation: success
    """
    x_np = np.random.random((50, 40)).astype(np.float32)
    y_np = SoftplusCompute(x_np)

    x_ms = Tensor(x_np)
    net = SoftplusNet()
    y_ms = net(x_ms)

    assert np.allclose(y_np, y_ms.asnumpy())


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_softplus_4d_fp32():
    """
    Feature: test ops.Softplus
    Description: test the Softplus with 4d fp32 inputs
    Expectation: success
    """
    x_np = np.random.random((32, 3, 224, 224)).astype(np.float32)
    y_np = SoftplusCompute(x_np)

    x_ms = Tensor(x_np)
    net = SoftplusNet()
    y_ms = net(x_ms)

    assert np.allclose(y_np, y_ms.asnumpy())


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_softplus_neg():
    """
    Feature: test ops.Softplus
    Description: test the Softplus with neg inputs
    Expectation: success
    """
    x_np = np.random.random((32, 3, 224, 224)).astype(np.float32) * -1
    y_np = SoftplusCompute(x_np)

    x_ms = Tensor(x_np)
    net = SoftplusNet()
    y_ms = net(x_ms)

    assert np.allclose(y_np, y_ms.asnumpy())


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_softplus_4d_fp16():
    """
    Feature: test ops.Softplus
    Description: test the Softplus with 4d fp16 inputs
    Expectation: success
    """
    x_np = np.random.random((32, 3, 224, 224)).astype(np.float16)
    y_np = SoftplusCompute(x_np)

    x_ms = Tensor(x_np)
    net = SoftplusNet()
    y_ms = net(x_ms)

    assert np.allclose(y_np, y_ms.asnumpy(), rtol=5e-3)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_softplus_7d_fp32():
    """
    Feature: test ops.Softplus
    Description: test the Softplus with 7d fp32 inputs
    Expectation: success
    """
    x_np = np.random.random((32, 3, 20, 20, 20, 10, 10)).astype(np.float32)
    y_np = SoftplusCompute(x_np)

    x_ms = Tensor(x_np)
    net = SoftplusNet()
    y_ms = net(x_ms)

    assert np.allclose(y_np, y_ms.asnumpy(), rtol=5e-3)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_softplus_with_big_input():
    """
    Feature: test ops.Softplus
    Description: test the Softplus with input value greater than log(FLT_MAX)
    Expectation: success
    """
    softplus = SoftplusNet()
    x = Tensor(np.array([0.1, 42.3, 88, 109.2]).astype(np.float32))

    output = softplus(x)
    expect_output = np.array([0.7443967, 42.3, 88, 109.2]).astype(np.float32)
    assert np.allclose(output.asnumpy(), expect_output)
