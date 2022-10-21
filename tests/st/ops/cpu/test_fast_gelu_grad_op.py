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
from mindspore.ops.functional import vmap
from mindspore.common.api import jit
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.ops import operations as P
from mindspore.ops import functional as F
from mindspore.ops import composite as C

context.set_context(mode=context.GRAPH_MODE, device_target="CPU")


def fast_gelu_grad_compute(x, dy):
    """FastGeluGradCompute."""
    div_up = np.exp(-1.702 * x) + 1.702 * x * np.exp(-1.702 * x) + 1
    div_down = (np.exp(-1.702 * x) + 1) ** 2
    return dy * div_up / div_down


class FastGeluNet(nn.Cell):
    """FastGeluNet."""

    def __init__(self):
        """Init."""
        super(FastGeluNet, self).__init__()
        self.fast_gelu = P.FastGeLU()

    def construct(self, x):
        """Construct."""
        return self.fast_gelu(x)


class FastGeLUGradCPU(nn.Cell):
    """FastGeLUGrad."""

    def __init__(self, network_cpu):
        """Init."""
        super(FastGeLUGradCPU, self).__init__()
        self.fast_gelu_grad = C.GradOperation(get_all=True, sens_param=True)
        self.network_cpu = network_cpu

    def construct(self, input_data, sens):
        """Construct."""
        gout = self.fast_gelu_grad(self.network_cpu)(input_data, sens)
        return gout


def np_all_close_with_loss(out_cpu, expect_cpu):
    """np_all_close_with_loss"""
    return np.allclose(out_cpu, expect_cpu, 0.005, 0.005, equal_nan=True)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize('shape', [(2,), (4, 5), (3, 4, 5, 6)])
@pytest.mark.parametrize('dtype', [np.float32, np.float16])
def test_fast_gelu_grad_float32(shape, dtype):
    """
    Feature: FastGeLUGrad cpu kernel
    Description: test the rightness of FastGeLUGrad cpu kernel.
    Expectation: Success.
    """
    np.random.seed(1)
    prop_cpu = 1 if np.random.random() > 0.5 else -1
    dy_np = (np.random.randn(*shape) * prop_cpu).astype(dtype)
    x_np = (np.random.randn(*shape) * prop_cpu).astype(dtype)
    expect_cpu = fast_gelu_grad_compute(dy_np, x_np)

    dy_ms = Tensor(dy_np)
    x_ms = Tensor(x_np)
    net_cpu = FastGeluNet()
    grad = FastGeLUGradCPU(net_cpu)
    output = grad(dy_ms, x_ms)
    assert np_all_close_with_loss(output[0].asnumpy(), expect_cpu)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize('dtype', [np.float32, np.float16])
def test_fast_gelu_grad_vmap_cpu(dtype, shape=(100, 2)):
    """
    Feature: FastGeLUGrad cpu kernel
    Description: test the rightness of FastGeLUGrad cpu kernel vmap feature.
    Expectation: Success.
    """
    np.random.seed(1)

    net_cpu = FastGeluNet()
    grad = FastGeLUGradCPU(net_cpu)

    def fast_gelu_grad_func(dy, x):
        """fast_gelu_grad_func"""
        output_cpu = grad(dy, x)
        return output_cpu[0]

    prop_cpu = 1 if np.random.random() > 0.5 else -1
    dy_np = (np.random.randn(*shape) * prop_cpu).astype(dtype)
    x_np = (np.random.randn(*shape) * prop_cpu).astype(dtype)
    dy_cpu = Tensor(dy_np)
    x_cpu = Tensor(x_np)
    dy_cpu = F.sub(dy_cpu, 0)
    x_cpu = F.sub(x_cpu, 0)

    output_vmap_cpu = vmap(fast_gelu_grad_func, in_axes=(0, 0))(dy_cpu, x_cpu)

    @jit
    def manually_batched_cpu(dys_cpu, xs_cpu):
        """manually_batched_cpu"""
        output = []
        for i in range(dys_cpu.shape[0]):
            output.append(fast_gelu_grad_func(dys_cpu[i], xs_cpu[i]))
        return F.stack(output)

    output_manually = manually_batched_cpu(dy_cpu, x_cpu)

    assert np_all_close_with_loss(output_vmap_cpu.asnumpy(), output_manually.asnumpy())
