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


class Log10Net(nn.Cell):
    def construct(self, x):
        return x.log10()


class Log2Net(nn.Cell):
    def construct(self, x):
        return x.log2()


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos', 'platform_ascend'],
          level_mark='level2',
          card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_log10(mode):
    """
    Feature: test Tensor.log10.
    Description: Verify the result of Tensor.log10..
    Expectation: expect correct forward result.
    """
    ms.set_context(mode=mode)
    x = Tensor([10, 100, 1000], dtype=ms.float32)
    log10 = Log10Net()
    output = log10(x)
    expect_output = np.array([1, 2, 3], dtype=np.float32)

    assert np.allclose(output.asnumpy(), expect_output)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos', 'platform_ascend'],
          level_mark='level2',
          card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_log2(mode):
    """
    Feature: test Tensor.log2.
    Description: Verify the result of Tensor.log2..
    Expectation: expect correct forward result.
    """
    ms.set_context(mode=mode)
    x = Tensor([2, 4, 8], dtype=ms.float32)
    log2 = Log2Net()
    output = log2(x)
    expect_output = np.array([1, 2, 3], dtype=np.float32)

    assert np.allclose(output.asnumpy(), expect_output)
