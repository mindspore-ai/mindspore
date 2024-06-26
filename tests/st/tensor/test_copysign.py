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


class CopysginNet(nn.Cell):
    def construct(self, x, other):
        return x.copysign(other)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos', 'platform_gpu', 'platform_ascend'],
          level_mark='level2',
          card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_copysign(mode):
    """
    Feature: test Tensor.copysign
    Description: Verify the result of Tensor.copysign
    Expectation: expect correct forward result
    """
    ms.set_context(mode=mode)
    x_np = np.array([[0.3, -0.7], [0.5, 0.5]]).astype(np.float32)
    other_np = np.array([[-0.4, 0.6], [0.4, -0.6]]).astype(np.float32)
    x = Tensor(x_np, ms.float32)
    other = Tensor(other_np, ms.float32)
    net = CopysginNet()
    output_ms = net(x, other)
    expect_output = np.copysign(x_np, other_np)
    assert np.allclose(output_ms.asnumpy(), expect_output)
