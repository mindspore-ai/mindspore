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
from mindspore import Tensor

class LogicalAndNet(nn.Cell):
    def construct(self, x, y):
        return x.logical_and(y)


class LogicalNotNet(nn.Cell):
    def construct(self, x):
        return x.logical_not()


class LogicalOrNet(nn.Cell):
    def construct(self, x, y):
        return x.logical_or(y)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos', 'platform_gpu', 'platform_ascend'],
          level_mark='level1',
          card_mark='onecard',
          essential_mark='unessential', mem_peak=0.01)
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_tensor_logical_and(mode):
    """
    Feature: test Tensor.logical_and.
    Description: Verify the result of Tensor.logical_and.
    Expectation: expect correct forward result.
    """
    ms.set_context(mode=mode, jit_config={"jit_level": "O0"})
    x_np = np.array([True, False, True]).astype(np.bool_)
    y_np = np.array([True, True, False]).astype(np.bool_)
    x = Tensor(x_np, dtype=ms.bool_)
    y = Tensor(y_np, dtype=ms.bool_)
    expect_output = np.logical_and(x_np, y_np)
    net = LogicalAndNet()
    output = net(x, y)
    assert np.allclose(output.asnumpy(), expect_output)

@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos', 'platform_gpu', 'platform_ascend'],
          level_mark='level1',
          card_mark='onecard',
          essential_mark='unessential', mem_peak=2048.00)
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_tensor_logical_not(mode):
    """
    Feature: test Tensor.logical_not.
    Description: Verify the result of Tensor.logical_not.
    Expectation: expect correct forward result.
    """
    ms.set_context(mode=mode, jit_config={"jit_level": "O0"})
    x_np = np.array([True, False, True]).astype(np.bool_)
    x = Tensor(x_np, dtype=ms.bool_)
    expect_output = np.logical_not(x_np)
    net = LogicalNotNet()
    output = net(x)
    output = x.logical_not()
    assert np.allclose(output.asnumpy(), expect_output)

@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos', 'platform_gpu', 'platform_ascend'],
          level_mark='level1',
          card_mark='onecard',
          essential_mark='unessential', mem_peak=2048.01)
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_tensor_logical_or(mode):
    """
    Feature: test Tensor.logical_or.
    Description: Verify the result of Tensor.logical_or.
    Expectation: expect correct forward result.
    """
    ms.set_context(mode=mode, jit_config={"jit_level": "O0"})
    x_np = np.array([True, False, True]).astype(np.bool_)
    y_np = np.array([True, True, False]).astype(np.bool_)
    x = Tensor(x_np, dtype=ms.bool_)
    y = Tensor(y_np, dtype=ms.bool_)
    expect_output = np.logical_or(x_np, y_np)
    net = LogicalOrNet()
    output = net(x, y)
    assert np.allclose(output.asnumpy(), expect_output)
