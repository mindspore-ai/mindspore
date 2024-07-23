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
from tests.mark_utils import arg_mark

import mindspore as ms
import mindspore.nn as nn
from mindspore import Tensor


class Deg2radNet(nn.Cell):
    def construct(self, x):
        return x.deg2rad()


class Rad2degNet(nn.Cell):
    def construct(self, x):
        return x.rad2deg()


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos', 'platform_gpu', 'platform_ascend'],
          level_mark='level2',
          card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_deg2rad(mode):
    """
    Feature: test Tensor.deg2rad
    Description: Verify the result of Tensor.deg2rad
    Expectation: expect correct forward result
    """
    ms.set_context(mode=mode)
    x_np = np.array([[90.0, -90.0], [180.0, -180.0], [270.0, -270.0]]).astype(np.float32)
    x = Tensor(x_np, ms.float32)
    net = Deg2radNet()
    output_ms = net(x)
    expect_output = np.deg2rad(x_np)
    assert np.allclose(output_ms.asnumpy(), expect_output)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos', 'platform_gpu', 'platform_ascend'],
          level_mark='level2',
          card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_rad2deg(mode):
    """
    Feature: test Tensor.rad2deg
    Description: Verify the result of Tensor.rad2deg
    Expectation: expect correct forward result
    """
    ms.set_context(mode=mode)
    x_np = np.array([[6.283, -3.142], [1.570, -6.283], [3.142, -1.570]]).astype(np.float32)
    x = Tensor(x_np, ms.float32)
    net = Rad2degNet()
    output_ms = net(x)
    expect_output = np.rad2deg(x_np)
    assert np.allclose(output_ms.asnumpy(), expect_output)
