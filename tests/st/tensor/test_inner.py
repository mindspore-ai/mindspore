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


class Net(nn.Cell):
    def construct(self, x, y):
        output = x.inner(y)
        return output


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos', 'platform_gpu', 'platform_ascend'],
          level_mark='level2',
          card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_inner_normal(mode):
    """
    Feature: tensor.inner
    Description: Verify the result of tensor.inner
    Expectation: success
    """
    ms.set_context(mode=mode)
    net = Net()
    x = ms.Tensor([[[1, 2, 3], [3, 2, 1]], [[4, 5, 6], [4, 5, 6]]], ms.float32)
    y = ms.Tensor([[2, 3, 4], [4, 3, 2]], ms.float32)
    out = net(x, y)
    expect_out = np.array([[[20, 16], [16, 20]], [[47, 43], [47, 43]]], dtype=np.float32)
    assert np.allclose(out.asnumpy(), expect_out)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos', 'platform_gpu', 'platform_ascend'],
          level_mark='level2',
          card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_inner_with_scalar(mode):
    """
    Feature: tensor.inner
    Description: Verify the result of tensor.inner
    Expectation: success
    """
    ms.set_context(mode=mode)
    net = Net()
    x = ms.Tensor([[[1, 2, 3], [3, 2, 1]], [[4, 5, 6], [4, 5, 6]]], ms.float32)
    y = ms.Tensor(2, ms.float32)
    out = net(x, y)
    expect_out = np.array([[[2, 4, 6], [6, 4, 2]], [[8, 10, 12], [8, 10, 12]]], dtype=np.float32)
    assert np.allclose(out.asnumpy(), expect_out)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos', 'platform_gpu', 'platform_ascend'],
          level_mark='level1',
          card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_inner_1d(mode):
    """
    Feature: tensor.inner
    Description: Verify the result of tensor.inner
    Expectation: success
    """
    ms.set_context(mode=mode)
    net = Net()
    x = ms.Tensor([1, 2, 3], ms.float32)
    y = ms.Tensor([4, 5, 6], ms.float32)
    out = net(x, y)
    expect_out = np.array(32, dtype=np.float32)
    assert np.allclose(out.asnumpy(), expect_out)
