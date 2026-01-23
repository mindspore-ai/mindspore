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
from mindspore.graph.api import _pynative_executor


class Net(nn.Cell):
    def construct(self, x, y):
        output = x.expand_as(other=y)
        return output


class NetPy(nn.Cell):
    def construct(self, x, y):
        output = x.expand_as(x=y)
        return output


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos', 'platform_gpu', 'platform_ascend',
                      'platform_ascend910b'],
          level_mark='level1',
          card_mark='onecard',
          essential_mark='unessential', mem_peak=4.0)
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_expand_as_pyboost(mode):
    """
    Feature: tensor.expand_as
    Description: Verify the result of expand_as in pyboost
    Expectation: success
    """
    ms.set_context(mode=mode, jit_config={"jit_level": "O0"})
    net = Net()
    x = Tensor([1, 2, 3], dtype=ms.float32)
    y = Tensor(np.ones((2, 3)), dtype=ms.float32)
    # For the time being, cpu or gpu does not work in graph mode.
    if ms.get_context('device_target') != 'Ascend' and ms.get_context('mode') == ms.GRAPH_MODE:
        with pytest.raises(RuntimeError):
            net(x, y)
            _pynative_executor.sync()
        return
    expect_output = np.array([[1., 2., 3.], [1., 2., 3.]])
    assert np.allclose(net(x, y).asnumpy(), expect_output)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos', 'platform_gpu', 'platform_ascend',
                      'platform_ascend910b'],
          level_mark='level1',
          card_mark='onecard',
          essential_mark='unessential', mem_peak=2048.00)
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_expand_as_python(mode):
    """
    Feature: tensor.expand_as
    Description: Verify the result of expand_as in python
    Expectation: success
    """
    ms.set_context(mode=mode, jit_config={"jit_level": "O0"})
    netpy = NetPy()
    x = Tensor([1, 2, 3], dtype=ms.float32)
    y = Tensor(np.ones((2, 3)), dtype=ms.float32)
    expect_output = np.array([[1., 2., 3.], [1., 2., 3.]])
    assert np.allclose(netpy(x, y).asnumpy(), expect_output)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos', 'platform_gpu', 'platform_ascend',
                      'platform_ascend910b'],
          level_mark='level1',
          card_mark='onecard',
          essential_mark='unessential', mem_peak=4.0)
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_expand_as_python_ge(mode):
    """
    Feature: tensor.expand_as
    Description: Verify the result of expand_as in python in GE mode
    Expectation: success
    """
    ms.set_context(mode=mode, jit_config={"jit_level": "O2"})
    netpy = NetPy()
    x = Tensor([1, 2, 3], dtype=ms.float32)
    y = Tensor(np.ones((2, 3)), dtype=ms.float32)
    expect_output = np.array([[1., 2., 3.], [1., 2., 3.]])
    assert np.allclose(netpy(x, y).asnumpy(), expect_output)
