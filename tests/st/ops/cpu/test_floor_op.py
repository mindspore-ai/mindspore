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
from mindspore import Tensor, ops


class FloorNet(nn.Cell):
    def __init__(self):
        super().__init__()
        self.floor = ops.Floor()

    def construct(self, x):
        out = self.floor(x)
        return out


class FloorFunc(nn.Cell):
    def construct(self, x):
        out = ops.floor(x)
        return out


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
@pytest.mark.parametrize('dtype', [np.float16, np.float32, np.float64])
def test_floor_op_dtype(mode, dtype):
    """
    Feature: CPU floor op.
    Description: test floor with the different types.
    Expectation: success.
    """
    context.set_context(mode=mode, device_target="CPU")

    net = FloorNet()
    x_np = np.array([1.1, 2.5, -1.5]).astype(dtype)
    out_ms = net(Tensor(x_np))

    out_np = np.floor(x_np)
    assert np.allclose(out_np, out_ms.asnumpy())


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_floor_op_functional(mode):
    """
    Feature: CPU floor op.
    Description: test floor with the functional interface.
    Expectation: success.
    """
    context.set_context(mode=mode, device_target="CPU")

    net = FloorFunc()
    x_np = np.array([1.1, 2.5, -1.5]).astype(np.float32)
    out_ms = net(Tensor(x_np))

    out_np = np.floor(x_np)
    assert np.allclose(out_np, out_ms.asnumpy())


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_floor_op_tensor(mode):
    """
    Feature: CPU floor op.
    Description: test floor with the Tensor interface.
    Expectation: success.
    """
    context.set_context(mode=mode, device_target="CPU")

    x_np = np.array([1.1, 2.5, -1.5]).astype(np.float32)
    out_ms = Tensor(x_np).floor()

    out_np = np.floor(x_np)
    assert np.allclose(out_np, out_ms.asnumpy())
