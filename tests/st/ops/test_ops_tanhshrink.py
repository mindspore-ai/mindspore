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
from tests.mark_utils import arg_mark
import numpy as np
import pytest

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor


class Net(nn.Cell):
    def construct(self, x):
        return ops.tanhshrink(x)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level2', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_tanhshrink_normal(mode):
    """
    Feature: Tanhshrink
    Description: Verify the result of Tanhshrink with normal input
    Expectation: success
    """
    ms.set_context(mode=mode)
    net = Net()
    a = Tensor(np.array([1, 2, 3, 2, 1]).astype(np.float16))
    output = net(a).asnumpy()
    expected_output = np.array([0.2383, 1.036, 2.004, 1.036, 0.2383]).astype(np.float16)
    assert np.allclose(output, expected_output, 1e-3, 1e-3)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level2', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_tanhshrink_negative(mode):
    """
    Feature: Tanhshrink
    Description: Verify the result of Tanhshrink with negative input
    Expectation: success
    """
    ms.set_context(mode=mode)
    net = Net()
    a = Tensor(np.array([-1, -2, -3, -2, -1]).astype(np.float16))
    output = net(a).asnumpy()
    expected_output = np.array([-0.2383, -1.036, -2.004, -1.036, -0.2383]).astype(np.float16)
    assert np.allclose(output, expected_output, 1e-3, 1e-3)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level2', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_tanhshrink_zeros(mode):
    """
    Feature: Tanhshrink
    Description: Verify the result of Tanhshrink with zeros
    Expectation: success
    """
    ms.set_context(mode=mode)
    net = Net()
    a = Tensor(np.array([0, 0, 0, 0, 0]).astype(np.float16))
    output = net(a).asnumpy()
    expected_output = np.array([0, 0, 0, 0, 0]).astype(np.float16)
    assert np.allclose(output, expected_output, 1e-3, 1e-3)
