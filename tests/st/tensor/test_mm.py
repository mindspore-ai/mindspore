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
    def construct(self, x1, x2):
        output = x1.mm(x2)
        return output


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos', 'platform_gpu', 'platform_ascend'],
          level_mark='level2',
          card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_mm_normal(mode):
    """
    Feature: mm
    Description: Verify the result of mm
    Expectation: success
    """
    ms.set_context(mode=mode)
    net = Net()
    x1 = ms.Tensor(np.arange(6).reshape((2, 3)), dtype=ms.float32)
    x2 = ms.Tensor(np.arange(12).reshape((3, 4)), dtype=ms.float32)
    out = net(x1, x2)
    expect_out = np.array([[20, 23, 26, 29],
                           [56, 68, 80, 92]])
    assert np.allclose(out.asnumpy(), expect_out)
