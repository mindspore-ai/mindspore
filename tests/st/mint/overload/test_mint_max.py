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


class MaxNet(nn.Cell):
    def construct(self, x):
        return mint.max(x)


class MaxDimNet(nn.Cell):
    def construct(self, x, dim, keepdim=False):
        return mint.max(x, dim, keepdim)


class MaximumNet(nn.Cell):
    def construct(self, x, other):
        return mint.max(x, other)


@arg_mark(plat_marks=['platform_ascend'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_max(mode):
    """
    Feature: test mint.max
    Description: Verify the result of mint.max(input)
    Expectation: expect correct forward result
    """
    ms.set_context(mode=mode, jit_config={"jit_level": "O0"})
    net = MaxNet()

    x = Tensor(np.array([[1., 25., 5., 7.], [4., 11., 6., 21.]]), ms.float32)
    output = net(x)
    expect_output = 25.0
    assert np.allclose(output.asnumpy(), expect_output)

    x_np = np.array([[1., 25., 5., 7.], [4., 11., 6., 21.]]).astype(np.float32)
    with pytest.raises(TypeError):
        net(x_np)
        _pynative_executor.sync()


@arg_mark(plat_marks=['platform_ascend'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_max_dim(mode):
    """
    Feature: test mint.max
    Description: Verify the result of mint.max(input, dim, keepdim=False)
    Expectation: expect correct forward result
    """
    ms.set_context(mode=mode, jit_config={"jit_level": "O0"})
    net = MaxDimNet()

    x_np = np.array([[1., 25., 5., 7.], [7., 11., 6., 21.]]).astype(np.float32)
    x = Tensor(x_np, ms.float32)
    output = net(x, dim=-1, keepdim=True)
    expect_output0 = np.array([[25.], [21.]], dtype=np.float32)
    expect_output1 = np.array([[1], [3]], dtype=np.float32)
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
    Feature: test mint.max
    Description: Verify the result of mint.max(input, other)
    Expectation: expect correct forward result
    """
    ms.set_context(mode=mode, jit_config={"jit_level": "O0"})
    net = MaximumNet()

    x = Tensor(np.array([[1., 25., 5., 7.], [4., 11., 6., 21.]]), ms.float32)
    other = Tensor(
        np.array([[2., 26., 4., 1.], [3., 41., 16., 1.]]), ms.float32)
    output = net(x, other)
    expect_output = np.array(
        [[2., 26., 5., 7.], [4., 41., 16., 21.]], dtype=np.float32)
    assert np.allclose(output.asnumpy(), expect_output)


@arg_mark(plat_marks=['platform_ascend'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_max_empty_input():
    """
    Feature: test mint.max
    Description: mint.max getting empty input
    Expectation: expect raise value error
    """
    x = Tensor([], ms.float32)
    with pytest.raises(ValueError):
        ms.mint.max(x)
        _pynative_executor.sync()
