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
from mindspore.ops.functional import vmap
from mindspore.common.api import jit
from mindspore.ops import operations as P
from mindspore.ops import functional as F

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
@pytest.mark.parametrize('shape', [(2,), (4, 5), (3, 4, 5, 6)])
@pytest.mark.parametrize('dtype', [np.float32, np.float16])
def test_fast_gelu_cpu(shape, dtype):
    """
    Feature: FastGeLU cpu kernel
    Description: test the rightness of FastGeLU cpu kernel.
    Expectation: Success.
    """
    prop_cpu = 100 if np.random.random() > 0.5 else -100
    x_np = (np.random.randn(*shape) * prop_cpu).astype(dtype)
    y_np_cpu = fast_gelu_compute(x_np)

    x_ms_cpu = Tensor(x_np)
    net = FastGeluNet()
    y_ms = net(x_ms_cpu)
    assert np_all_close_with_loss(y_np_cpu, y_ms.asnumpy())

    x_ms_cpu = Tensor(x_np)
    y_fun = F.fast_gelu(x_ms_cpu)
    assert np_all_close_with_loss(y_np_cpu, y_fun.asnumpy())

    x_ms_cpu = Tensor(x_np)
    fast_gelu_nn = nn.FastGelu()
    y_nn = fast_gelu_nn(x_ms_cpu)
    assert np_all_close_with_loss(y_np_cpu, y_nn.asnumpy())


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize('dtype', [np.float32, np.float16])
def test_fast_gelu_vmap(dtype, shape=(100, 2)):
    """
    Feature: FastGeLU cpu kernel
    Description: test the rightness of FastGeLU cpu kernel vmap feature.
    Expectation: Success.
    """

    def fast_gelu_fun(x):
        """fast_gelu_fun"""
        return P.FastGeLU()(x)

    prop = 100 if np.random.random() > 0.5 else -100
    x_np = (np.random.randn(*shape) * prop).astype(dtype)
    x = Tensor(x_np)
    x = F.sub(x, 0)

    output_vmap = vmap(fast_gelu_fun, in_axes=(0,))(x)

    @jit
    def manually_batched(xs):
        """manually_batched"""
        output = []
        for i in range(xs.shape[0]):
            output.append(fast_gelu_fun(xs[i]))
        return F.stack(output)

    output_manually = manually_batched(x)

    assert np_all_close_with_loss(output_vmap.asnumpy(), output_manually.asnumpy())
