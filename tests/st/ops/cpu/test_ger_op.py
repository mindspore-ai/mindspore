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

import mindspore.nn as nn
from mindspore import Tensor
from mindspore import context
from mindspore.ops import operations as P
from mindspore.ops import functional as F
from mindspore.ops.functional import vmap
from mindspore.common.api import jit


class NetGer(nn.Cell):
    """Net of ger."""

    def __init__(self):
        """Init."""
        super(NetGer, self).__init__()
        self.ger = P.Ger()

    def construct(self, x, y):
        """Construct."""
        return self.ger(x, y)


def np_all_close_with_loss(out, expect):
    """np_all_close_with_loss"""
    return np.allclose(out, expect, 0.0005, 0.0005, equal_nan=True)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
@pytest.mark.parametrize('dtype', [np.float16, np.float32, np.float64])
@pytest.mark.parametrize('xshape', [(2,), (3,), (4,)])
@pytest.mark.parametrize('yshape', [(2,), (3,), (4,)])
def test_ger_float16(dtype, mode, xshape, yshape):
    """
    Feature: Ger cpu kernel
    Description: test the rightness of Ger cpu kernel.
    Expectation: Success.
    """
    context.set_context(mode=mode, device_target="CPU")
    prop = 100 if np.random.random() > 0.5 else -100
    x_array = (np.random.randn(*xshape) * prop).astype(dtype)
    y_array = (np.random.randn(*yshape) * prop).astype(dtype)
    input_x = Tensor(x_array)
    input_y = Tensor(y_array)
    expect = x_array.reshape(xshape[0], 1) * y_array.reshape(1, yshape[0])

    net = NetGer()
    output = net(input_x, input_y)
    assert np.allclose(output.asnumpy(), expect)

    output_functional = F.ger(input_x, input_y)
    assert np.allclose(output_functional.asnumpy(), expect)

    output_tensor = input_x.ger(input_y)
    assert np.allclose(output_tensor.asnumpy(), expect)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize('dtype', [np.float32, np.float16, np.float64])
def test_ger_vmap(dtype):
    """
    Feature: Ger cpu kernel
    Description: test the rightness of Ger cpu kernel vmap feature.
    Expectation: Success.
    """

    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")

    def ger_func(x1, x2):
        """ger_func"""
        return P.Ger()(x1, x2)

    @jit
    def manually_batched(x1s, x2s):
        """manually_batched"""
        output = []
        for i in range(x1s.shape[0]):
            output.append(ger_func(x1s[i], x2s[i]))
        return F.stack(output)

    x1shape = (100, 3)
    x2shape = (100, 4)
    prop = 100 if np.random.random() > 0.5 else -100
    x1_np = (np.random.randn(*x1shape) * prop).astype(dtype)
    x2_np = (np.random.randn(*x2shape) * prop).astype(dtype)
    x1 = Tensor(x1_np)
    x2 = Tensor(x2_np)
    x1 = F.sub(x1, 0)
    x2 = F.sub(x2, 0)

    output_vmap = vmap(ger_func, in_axes=(0, 0))(x1, x2)
    output_manually = manually_batched(x1, x2)

    assert np_all_close_with_loss(output_vmap.asnumpy(), output_manually.asnumpy())


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize('dtype', [np.float32, np.float16, np.float64])
def test_ger_vmap_two(dtype):
    """
    Feature: Ger cpu kernel
    Description: test the rightness of Ger cpu kernel vmap feature.
    Expectation: Success.
    """

    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")

    def ger_func_two(x1, x2):
        """ger_func_two"""
        return P.Ger()(x1, x2)

    @jit
    def manually_batched_two(x1s, x2s):
        """manually_batched_two"""
        output = []
        for i in range(x1s.shape[0]):
            output.append(ger_func_two(x1s[i], x2s[:, i]))
        return F.stack(output)

    x1shape_2 = (100, 3)
    x2shape_2 = (4, 100)
    prop_2 = 100 if np.random.random() > 0.5 else -100
    x1_np_2 = (np.random.randn(*x1shape_2) * prop_2).astype(dtype)
    x2_np_2 = (np.random.randn(*x2shape_2) * prop_2).astype(dtype)
    x1_2 = Tensor(x1_np_2)
    x2_2 = Tensor(x2_np_2)
    x1_2 = F.sub(x1_2, 0)
    x2_2 = F.sub(x2_2, 0)

    output_vmap_2 = vmap(ger_func_two, in_axes=(0, 1))(x1_2, x2_2)
    output_manually_2 = manually_batched_two(x1_2, x2_2)

    assert np_all_close_with_loss(output_vmap_2.asnumpy(), output_manually_2.asnumpy())
