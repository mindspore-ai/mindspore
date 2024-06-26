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


class Net(nn.Cell):
    def construct(self, x, y):
        return x.lcm(y)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos', 'platform_gpu'],
          level_mark='level2',
          card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_lcm(mode):
    """
    Feature: test Tensor.lcm
    Description: Verify the result of Tensor.lcm
    Expectation: expect correct forward result
    """
    ms.set_context(mode=mode)
    x1_np = np.array([10, 15, 20]).astype(np.int32)
    x2_np = np.array([5]).astype(np.int64)
    input_x1 = Tensor(x1_np)
    input_x2 = Tensor(x2_np)
    net = Net()
    output_ms = net(input_x1, input_x2)
    expect_output = np.lcm(x1_np, x2_np)
    assert np.allclose(output_ms.asnumpy(), expect_output)
