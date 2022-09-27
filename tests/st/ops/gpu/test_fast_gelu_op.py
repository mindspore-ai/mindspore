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
from mindspore.ops import composite as C
from mindspore.ops import operations as P
from mindspore.ops import functional as F
from mindspore.ops.functional import vmap
from mindspore.common.api import ms_function

context.set_context(mode=context.GRAPH_MODE, device_target="GPU")


def fast_gelu_grad_compute(x, dy):
    """FastGeluGradCompute."""
    div_up = np.exp(-1.702 * x) + 1.702 * x * np.exp(-1.702 * x) + 1
    div_down = (np.exp(-1.702 * x) + 1) ** 2
    return dy * div_up / div_down


def fast_gelu_compute(x):
    """FastGeluCompute."""
    return x * np.exp(0.851 * (x - np.abs(x))) / (1 + np.exp(-1.702 * np.abs(x)))


class FastGeluNet(nn.Cell):
    """FastGeluNet."""

    def __init__(self):
        """Init."""
        super(FastGeluNet, self).__init__()
        self.fast_gelu = P.FastGeLU()

    def construct(self, x):
        """Construct."""
        return self.fast_gelu(x)


class FastGeLUGrad(nn.Cell):
    """FastGeLUGrad."""

    def __init__(self, network):
        """Init."""
        super(FastGeLUGrad, self).__init__()
        self.fast_gelu_grad = C.GradOperation(get_all=True, sens_param=True)
        self.network = network

    def construct(self, input_data, sens):
        """Construct."""
        gout = self.fast_gelu_grad(self.network)(input_data, sens)
        return gout


def np_all_close_with_loss(out, expect):
    """np_all_close_with_loss"""
    return np.allclose(out, expect, 0.005, 0.005, equal_nan=True)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('shape', [(2,), (4, 5), (3, 4, 5, 6)])
@pytest.mark.parametrize('dtype', [np.float32, np.float16])
def test_fast_gelu_grad(shape, dtype):
    """
    Feature: FastGeLUGrad gpu kernel
    Description: test the rightness of FastGeLUGrad gpu kernel.
    Expectation: Success.
    """
    prop = 1 if np.random.random() > 0.5 else -1
    dy_np = (np.random.randn(*shape) * prop).astype(dtype)
    x_np = (np.random.randn(*shape) * prop).astype(dtype)
    expect = fast_gelu_grad_compute(dy_np, x_np)

    dy_ms = Tensor(dy_np)
    x_ms = Tensor(x_np)
    net = FastGeluNet()
    grad = FastGeLUGrad(net)
    output = grad(dy_ms, x_ms)
    assert np_all_close_with_loss(output[0].asnumpy(), expect)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('shape', [(2,), (4, 5), (3, 4, 5, 6)])
@pytest.mark.parametrize('dtype', [np.float32, np.float16])
def test_fast_gelu(shape, dtype):
    """
    Feature: FastGeLU gpu kernel
    Description: test the rightness of FastGeLU gpu kernel.
    Expectation: Success.
    """
    prop = 100 if np.random.random() > 0.5 else -100
    x_np = (np.random.randn(*shape) * prop).astype(dtype)
    y_np = fast_gelu_compute(x_np)

    x_ms = Tensor(x_np)
    net = FastGeluNet()
    y_ms = net(x_ms)
    assert np_all_close_with_loss(y_np, y_ms.asnumpy())

    x_ms = Tensor(x_np)
    y_fun = F.fast_gelu(x_ms)
    assert np_all_close_with_loss(y_np, y_fun.asnumpy())

    x_ms = Tensor(x_np)
    fast_gelu_nn = nn.FastGelu()
    y_nn = fast_gelu_nn(x_ms)
    assert np_all_close_with_loss(y_np, y_nn.asnumpy())


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('dtype', [np.float32, np.float16])
def test_fast_gelu_grad_vmap(dtype, shape=(100, 2)):
    """
    Feature: FastGeLUGrad gpu kernel
    Description: test the rightness of FastGeLUGrad gpu kernel vmap feature.
    Expectation: Success.
    """
    net = FastGeluNet()
    grad = FastGeLUGrad(net)

    def fast_gelu_grad_func(dy, x):
        """fast_gelu_grad_func"""
        output = grad(dy, x)
        return output[0]

    prop = 1 if np.random.random() > 0.5 else -1
    dy_np = (np.random.randn(*shape) * prop).astype(dtype)
    x_np = (np.random.randn(*shape) * prop).astype(dtype)
    dy = Tensor(dy_np)
    x = Tensor(x_np)
    dy = F.sub(dy, 0)
    x = F.sub(x, 0)

    output_vmap = vmap(fast_gelu_grad_func, in_axes=(0, 0))(dy, x)

    @ms_function
    def manually_batched(dys, xs):
        """manually_batched"""
        output = []
        for i in range(dys.shape[0]):
            output.append(fast_gelu_grad_func(dys[i], xs[i]))
        return F.stack(output)

    output_manually = manually_batched(dy, x)
    assert np_all_close_with_loss(output_vmap.asnumpy(), output_manually.asnumpy())


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('dtype', [np.float32, np.float16])
def test_fast_gelu_vmap(dtype, shape=(100, 2)):
    """
    Feature: FastGeLU gpu kernel
    Description: test the rightness of FastGeLU gpu kernel vmap feature.
    Expectation: Success.
    """

    def fast_gelu_func(x):
        """fast_gelu_func"""
        return P.FastGeLU()(x)

    prop = 100 if np.random.random() > 0.5 else -100
    x_np = (np.random.randn(*shape) * prop).astype(dtype)
    x = Tensor(x_np)
    x = F.sub(x, 0)

    output_vmap = vmap(fast_gelu_func, in_axes=(0,))(x)

    @ms_function
    def manually_batched(xs):
        """manually_batched"""
        output = []
        for i in range(xs.shape[0]):
            output.append(fast_gelu_func(xs[i]))
        return F.stack(output)

    output_manually = manually_batched(x)

    assert np_all_close_with_loss(output_vmap.asnumpy(), output_manually.asnumpy())
