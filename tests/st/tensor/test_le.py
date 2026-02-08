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

class LeNet(nn.Cell):
    def construct(self, x, y):
        return x.le(y)


class LessEqualNet(nn.Cell):
    def construct(self, x, y):
        return x.less_equal(y)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos', 'platform_gpu', 'platform_ascend'],
          level_mark='level1',
          card_mark='onecard',
          essential_mark='essential', mem_peak=2048.0)
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_tensor_le(mode):
    """
    Feature: test Tensor.le.
    Description: Verify the result of Tensor.le..
    Expectation: expect correct forward result.
    """
    ms.set_context(mode=mode, jit_config={"jit_level": "O0"})
    x_np = np.array([1, 2, 3]).astype(np.int32)
    y_np = np.array([1, 1, 4]).astype(np.int32)
    x = Tensor(x_np, dtype=ms.int32)
    y = Tensor(y_np, dtype=ms.int32)
    expect_output = x_np <= y_np
    net = LeNet()
    output = net(x, y)
    assert np.allclose(output.asnumpy(), expect_output)

@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos', 'platform_gpu', 'platform_ascend'],
          level_mark='level1',
          card_mark='onecard',
          essential_mark='essential', mem_peak=2048.0)
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_tensor_less_equal(mode):
    """
    Feature: test Tensor.less_equal.
    Description: Verify the result of Tensor.less_equal.
    Expectation: expect correct forward result.
    """
    ms.set_context(mode=mode, jit_config={"jit_level": "O0"})
    x_np = np.array([1, 2, 3]).astype(np.int32)
    y_np = np.array([1, 1, 4]).astype(np.int32)
    x = Tensor(x_np, dtype=ms.int32)
    y = Tensor(y_np, dtype=ms.int32)
    expect_output = x_np <= y_np
    net = LessEqualNet()
    output = net(x, y)
    assert np.allclose(output.asnumpy(), expect_output)
