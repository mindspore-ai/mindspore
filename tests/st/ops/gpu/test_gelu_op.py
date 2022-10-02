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
from mindspore.ops import operations as P
from mindspore.ops import functional as F
from mindspore.common import dtype as mstype

context.set_context(mode=context.GRAPH_MODE, device_target="GPU")


class GeluNet(nn.Cell):
    def __init__(self):
        super(GeluNet, self).__init__()
        self.gelu = P.GeLU()

    def construct(self, x):
        return self.gelu(x)


def gelu_compute(x):
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x * x * x)))


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_gelu_1d():
    x_np = np.random.random((50,)).astype(np.float32)
    y_np = gelu_compute(x_np)

    x_ms = Tensor(x_np)
    net = GeluNet()
    y_ms = net(x_ms)

    assert np.allclose(y_np, y_ms.asnumpy())


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_gelu_2d():
    x_np = np.random.random((50, 40)).astype(np.float32)
    y_np = gelu_compute(x_np)

    x_ms = Tensor(x_np)
    net = GeluNet()
    y_ms = net(x_ms)

    assert np.allclose(y_np, y_ms.asnumpy())


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_gelu_4d():
    x_np = np.random.random((32, 3, 224, 224)).astype(np.float32)
    y_np = gelu_compute(x_np)

    x_ms = Tensor(x_np)
    net = GeluNet()
    y_ms = net(x_ms)

    assert np.allclose(y_np, y_ms.asnumpy())


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_gelu_neg():
    x_np = np.random.random((32, 3, 224, 224)).astype(np.float32) * -1
    y_np = gelu_compute(x_np)

    x_ms = Tensor(x_np)
    net = GeluNet()
    y_ms = net(x_ms)

    assert np.allclose(y_np, y_ms.asnumpy())


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_gelu_4d_fp16():
    x_np = np.random.random((32, 3, 224, 224)).astype(np.float16)
    y_np = gelu_compute(x_np)

    x_ms = Tensor(x_np)
    net = GeluNet()
    y_ms = net(x_ms)

    assert np.allclose(y_np, y_ms.asnumpy(), rtol=1e-3)


def test_gelu_functional_api():
    """
    Feature: test gelu functional API.
    Description: test gelu functional API and compare with expected output.
    Expectation: output should be equal to expected value.
    """
    input_x = Tensor([1.0, 2.0, 3.0], mstype.float32)
    output = F.gelu(input_x)
    expected = np.array([0.841192, 1.9545976, 2.9963627], np.float32)
    np.testing.assert_array_almost_equal(output.asnumpy(), expected, decimal=4)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_gelu_functional_api_modes():
    """
    Feature: test gelu functional API for different modes.
    Description: test gelu functional API and compare with expected output.
    Expectation: output should be equal to expected value.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    test_gelu_functional_api()
    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
    test_gelu_functional_api()
