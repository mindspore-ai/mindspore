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
from mindspore import Tensor, context


class Net(nn.Cell):
    def construct(self, x, diagonal=0):
        return x.tril(diagonal)


class TrilNet(nn.Cell):
    def __init__(self):
        super(TrilNet, self).__init__()
        self.tril = nn.Tril()

    def construct(self, value, k):

        return self.tril(value, k)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos', 'platform_gpu', 'platform_ascend'],
          level_mark='level2',
          card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_tril(mode):
    """
    Feature: tril
    Description: Verify the result of tril
    Expectation: success
    """
    ms.set_context(mode=mode)
    x = Tensor([[-1.8297, -0.8474, 1.0292], [-1.2167, 0.5574, -0.6753], [-0.6702, 0.2276, 1.2421]])
    net = Net()
    output = net(x)
    expect_output = np.array([[-1.8297, 0., 0.], [-1.2167, 0.5574, 0.], [-0.6702, 0.2276, 1.2421]], dtype=np.float32)
    assert np.allclose(output.asnumpy(), expect_output)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos', 'platform_gpu'],
          level_mark='level2',
          card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [context.GRAPH_MODE,])
def test_tril_0(mode):
    """
    Feature: test_tril
    Description: Verify the result of test_tril
    Expectation: success
    """
    value = Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    net = TrilNet()
    out = net(value, 0)
    assert np.sum(out.asnumpy()) == 34


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos', 'platform_gpu'],
          level_mark='level2',
          card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [context.GRAPH_MODE,])
def test_tril_1(mode):
    """
    Feature: test_tril_1
    Description: Verify the result of test_tril_1
    Expectation: success
    """
    value = Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    net = TrilNet()
    out = net(value, 1)
    assert np.sum(out.asnumpy()) == 42


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos', 'platform_gpu'],
          level_mark='level2',
          card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [context.GRAPH_MODE,])
def test_tril_2(mode):
    """
    Feature: test_tril_2
    Description: Verify the result of test_tril_2
    Expectation: success
    """
    value = Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    net = TrilNet()
    out = net(value, -1)
    assert np.sum(out.asnumpy()) == 19


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos', 'platform_gpu'],
          level_mark='level2',
          card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [context.GRAPH_MODE,])
def test_tril_parameter(mode):
    """
    Feature: test_tril_parameter
    Description: Verify the result of test_tril_parameter
    Expectation: success
    """
    net = TrilNet()
    net(Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), 0)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos', 'platform_gpu'],
          level_mark='level2',
          card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [context.GRAPH_MODE,])
def test_tril_parameter_1(mode):
    """
    Feature: test_tril_parameter_1
    Description: Verify the result of test_tril_parameter_1
    Expectation: success
    """
    net = TrilNet()
    net(Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), 0)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos', 'platform_gpu'],
          level_mark='level2',
          card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [context.GRAPH_MODE,])
def test_tril_parameter_2(mode):
    """
    Feature: test_tril_parameter_2
    Description: Verify the result of test_tril_parameter_2
    Expectation: success
    """
    net = TrilNet()
    net(Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), 0)
