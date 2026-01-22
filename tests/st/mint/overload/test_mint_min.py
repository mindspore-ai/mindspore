# Copyright 2024 Huawei Technologies Co., Ltd
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
from tests.mark_utils import arg_mark

import mindspore as ms
import mindspore.nn as nn
from mindspore import Tensor, mint
from mindspore.graph.api import _pynative_executor

class MinNet(nn.Cell):
    def construct(self, x):
        return mint.min(x)


class MinDimNet(nn.Cell):
    def construct(self, x, dim, keepdim=False):
        return mint.min(x, dim, keepdim)


class MinimumNet(nn.Cell):
    def construct(self, x, other):
        return mint.min(x, other)


@arg_mark(plat_marks=['platform_ascend'],
          level_mark='level1',
          card_mark='onecard',
          essential_mark='essential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_min(mode):
    """
    Feature: test mint.min
    Description: Verify the result of mint.min(input)
    Expectation: expect correct forward result
    """
    ms.set_context(mode=mode, jit_config={"jit_level": "O0"})
    net = MinNet()

    x = Tensor(np.array([[1., 25., 5., 7.], [4., 11., 6., 21.]]), ms.float32)
    output = net(x)
    expect_output = 1.0
    assert np.allclose(output.asnumpy(), expect_output)

    x_np = np.array([[1., 25., 5., 7.], [4., 11., 6., 21.]]).astype(np.float32)
    with pytest.raises(TypeError):
        net(x_np)
        _pynative_executor.sync()


@arg_mark(plat_marks=['platform_ascend'],
          level_mark='level1',
          card_mark='onecard',
          essential_mark='essential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_min_dim(mode):
    """
    Feature: test mint.min
    Description: Verify the result of mint.min(input, dim, keepdim=False)
    Expectation: expect correct forward result
    """
    ms.set_context(mode=mode, jit_config={"jit_level": "O0"})
    net = MinDimNet()

    x_np = np.array([[1., 25., 5., 7.], [7., 11., 6., 21.]]).astype(np.float32)
    x = Tensor(x_np, ms.float32)
    output = net(x, dim=-1, keepdim=True)
    expect_output0 = np.array([[1.], [6.]], dtype=np.float32)
    expect_output1 = np.array([[0], [2]], dtype=np.float32)
    assert np.allclose(output[0].asnumpy(), expect_output0)
    assert np.allclose(output[1].asnumpy(), expect_output1)

    with pytest.raises(TypeError):
        net(x_np, dim=-1, keepdim=True)
        _pynative_executor.sync()

    with pytest.raises(TypeError):
        net(x, dim=None, keepdim=True)
        _pynative_executor.sync()

    with pytest.raises(TypeError):
        net(x_np, dim=None, keepdim=False)
        _pynative_executor.sync()

    with pytest.raises(TypeError):
        net(x_np, dim=-1, keepdim=-1)
        _pynative_executor.sync()


@arg_mark(plat_marks=['platform_ascend'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_minimum(mode):
    """
    Feature: test mint.min
    Description: Verify the result of mint.min(input, other)
    Expectation: expect correct forward result
    """
    ms.set_context(mode=mode, jit_config={"jit_level": "O0"})
    net = MinimumNet()

    x = Tensor(np.array([[1., 25., 5., 7.], [4., 11., 6., 21.]]), ms.float32)
    other = Tensor(np.array([[2., 26., 4., 1.], [3., 41., 16., 1.]]), ms.float32)
    output = net(x, other)
    expect_output = np.array([[1., 25., 4., 1.], [3., 11., 6., 1.]], dtype=np.float32)
    assert np.allclose(output.asnumpy(), expect_output)
