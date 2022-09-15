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


class FlattenNet(nn.Cell):
    def __init__(self):
        super().__init__()
        self.flatten = ops.Flatten()

    def construct(self, x):
        out = self.flatten(x)
        return out


class FlattenFunc(nn.Cell):
    def construct(self, x):
        out = ops.flatten(x)
        return out


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
@pytest.mark.parametrize("dtype", [np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16,
                                   np.uint32, np.uint64, np.float16, np.float32, np.float64,
                                   np.bool, np.complex64, np.complex128])
def test_flatten_op_dtype(mode, dtype):
    """
    Feature: cpu Flatten op.
    Description: test flatten with the different types.
    Expectation: success.
    """
    context.set_context(mode=mode, device_target="CPU")

    x = Tensor(np.array([[-0.1, 0.3, 3.6], [0.4, 0.5, -3.2]]).astype(dtype))
    expect = np.array([[-0.1, 0.3, 3.6], [0.4, 0.5, -3.2]]).astype(dtype)

    net = FlattenNet()
    out = net(x)

    assert np.allclose(expect, out.asnumpy())


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_flatten_op_functional(mode):
    """
    Feature: cpu Flatten op.
    Description: test flatten with functional interface.
    Expectation: success.
    """
    context.set_context(mode=mode, device_target="CPU")
    x = Tensor(np.array([[-0.1, 0.3, 3.6], [0.4, 0.5, -3.2]]).astype(np.float32))
    expect = np.array([[-0.1, 0.3, 3.6], [0.4, 0.5, -3.2]]).astype(np.float32)

    net = FlattenFunc()
    out = net(x)

    assert np.allclose(expect, out.asnumpy())


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_flatten_op_nn(mode):
    """
    Feature: cpu Flatten ops.
    Description: test flatten with nn interface.
    Expectation: success.
    """
    context.set_context(mode=mode, device_target="CPU")
    x = Tensor(np.array([[-0.1, 0.3, 3.6], [0.4, 0.5, -3.2]]).astype(np.float32))
    expect = np.array([[-0.1, 0.3, 3.6], [0.4, 0.5, -3.2]]).astype(np.float32)

    net = nn.Flatten()
    out = net(x)

    assert np.allclose(expect, out.asnumpy())
