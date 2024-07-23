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
import mindspore.nn as nn
from mindspore import Tensor
from mindspore import context


class Net(nn.Cell):
    def construct(self, x, k):
        return x.triu(diagonal=k)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos', 'platform_gpu'],
          level_mark='level2',
          card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_triu_0(mode):
    """
    Feature: test_triu_0
    Description: Verify the result of test_triu_0
    Expectation: success
    """
    context.set_context(mode=mode)
    net = Net()
    x = Tensor(np.array([[1, 2, 3, 4],
                         [5, 6, 7, 8],
                         [10, 11, 12, 13],
                         [14, 15, 16, 17]]))
    output = net(x, 1)
    expected = np.array([[0, 2, 3, 4],
                         [0, 0, 7, 8],
                         [0, 0, 0, 13],
                         [0, 0, 0, 0]])
    assert np.array_equal(output.asnumpy(), expected)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos', 'platform_gpu'],
          level_mark='level2',
          card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_triu_1(mode):
    """
    Feature: test_triu_1
    Description: test_triu_1
    Expectation: success
    """
    context.set_context(mode=mode)
    net = Net()
    x = Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    output = net(x, 0)
    assert np.sum(output.asnumpy()) == 26


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos', 'platform_gpu'],
          level_mark='level2',
          card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_triu_2(mode):
    """
    Feature: test_triu_2
    Description: test_triu_2
    Expectation: success
    """
    context.set_context(mode=mode)
    net = Net()
    x = Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    output = net(x, 1)
    assert np.sum(output.asnumpy()) == 11


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos', 'platform_gpu'],
          level_mark='level2',
          card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_triu_3(mode):
    """
    Feature: test_triu_3
    Description: test_triu_3
    Expectation: success
    """
    context.set_context(mode=mode)
    net = Net()
    x = Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    output = net(x, -1)
    assert np.sum(output.asnumpy()) == 38
