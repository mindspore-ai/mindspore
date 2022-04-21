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

import numpy as np
import pytest

import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.ops import operations as P

context.set_context(mode=context.GRAPH_MODE, device_target="CPU")


class FastGeluNet(nn.Cell):
    """FastGeluNet"""

    def __init__(self):
        """Init."""
        super(FastGeluNet, self).__init__()
        self.fast_gelu = P.FastGeLU()

    def construct(self, x):
        """Construct."""
        return self.fast_gelu(x)


def fast_gelu_compute(x):
    """FastGeluCompute."""
    return x * np.exp(0.851 * (x - np.abs(x))) / (1 + np.exp(-1.702 * np.abs(x)))


def np_all_close_with_loss(out, expect):
    """np_all_close_with_loss"""
    return np.allclose(out, expect, 0.005, 0.005, equal_nan=True)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_fast_gelu_1d_float32():
    """
    Feature: FastGeLU cpu kernel
    Description: test the rightness of FastGeLU cpu kernel, type of input data is 1d float32.
    Expectation: Success.
    """
    x_np = np.random.random((50,)).astype(np.float32)
    y_np = fast_gelu_compute(x_np)

    x_ms = Tensor(x_np)
    net = FastGeluNet()
    y_ms = net(x_ms)

    assert np_all_close_with_loss(y_np, y_ms.asnumpy())


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_fast_gelu_2d_float32():
    """
    Feature: FastGeLU cpu kernel
    Description: test the rightness of FastGeLU cpu kernel, type of input data is 2d float32.
    Expectation: Success.
    """
    x_np = np.random.random((50, 40)).astype(np.float32)
    y_np = fast_gelu_compute(x_np)

    x_ms = Tensor(x_np)
    net = FastGeluNet()
    y_ms = net(x_ms)

    assert np_all_close_with_loss(y_np, y_ms.asnumpy())


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_fast_gelu_4d_float32():
    """
    Feature: FastGeLU cpu kernel
    Description: test the rightness of FastGeLU cpu kernel, type of input data is 4d float32.
    Expectation: Success.
    """
    x_np = np.random.random((32, 3, 224, 224)).astype(np.float32)
    y_np = fast_gelu_compute(x_np)

    x_ms = Tensor(x_np)
    net = FastGeluNet()
    y_ms = net(x_ms)

    assert np_all_close_with_loss(y_np, y_ms.asnumpy())


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_fast_gelu_neg_float32():
    """
    Feature: FastGeLU cpu kernel
    Description: test the rightness of FastGeLU cpu kernel, type of input data is neg float32.
    Expectation: Success.
    """
    x_np = np.random.random((32, 3, 224, 224)).astype(np.float32) * -1
    y_np = fast_gelu_compute(x_np)

    x_ms = Tensor(x_np)
    net = FastGeluNet()
    y_ms = net(x_ms)

    assert np_all_close_with_loss(y_np, y_ms.asnumpy())


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_fast_gelu_1d_float16():
    """
    Feature: FastGeLU cpu kernel
    Description: test the rightness of FastGeLU cpu kernel, type of input data is 1d float16.
    Expectation: Success.
    """
    x_np = np.random.random((50,)).astype(np.float16)
    y_np = fast_gelu_compute(x_np)

    x_ms = Tensor(x_np)
    net = FastGeluNet()
    y_ms = net(x_ms)

    assert np_all_close_with_loss(y_np, y_ms.asnumpy())


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_fast_gelu_2d_float16():
    """
    Feature: FastGeLU cpu kernel
    Description: test the rightness of FastGeLU cpu kernel, type of input data is 2d float16.
    Expectation: Success.
    """
    x_np = np.random.random((50, 40)).astype(np.float16)
    y_np = fast_gelu_compute(x_np)

    x_ms = Tensor(x_np)
    net = FastGeluNet()
    y_ms = net(x_ms)

    assert np_all_close_with_loss(y_np, y_ms.asnumpy())


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_fast_gelu_4d_float16():
    """
    Feature: FastGeLU cpu kernel
    Description: test the rightness of FastGeLU cpu kernel, type of input data is 4d float16.
    Expectation: Success.
    """
    x_np = np.random.random((32, 3, 224, 224)).astype(np.float16)
    y_np = fast_gelu_compute(x_np)

    x_ms = Tensor(x_np)
    net = FastGeluNet()
    y_ms = net(x_ms)

    assert np_all_close_with_loss(y_np, y_ms.asnumpy())


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_fast_gelu_neg_float16():
    """
    Feature: FastGeLU cpu kernel
    Description: test the rightness of FastGeLU cpu kernel, type of input data is neg float16.
    Expectation: Success.
    """
    x_np = np.random.random((32, 3, 224, 224)).astype(np.float16) * -1
    y_np = fast_gelu_compute(x_np)

    x_ms = Tensor(x_np)
    net = FastGeluNet()
    y_ms = net(x_ms)

    assert np_all_close_with_loss(y_np, y_ms.asnumpy())
