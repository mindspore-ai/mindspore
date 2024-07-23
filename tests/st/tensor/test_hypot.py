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

import pytest
from tests.mark_utils import arg_mark
import numpy as np

import mindspore as ms
import mindspore.nn as nn


class Net(nn.Cell):
    def construct(self, x, other):
        output = x.hypot(other)
        return output


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos', 'platform_gpu', 'platform_ascend'],
          level_mark='level2',
          card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_hypot_normal(mode):
    """
    Feature: tensor.hypot
    Description: Verify the result of hypot
    Expectation: success
    """
    ms.set_context(mode=mode)
    net = Net()
    x = ms.Tensor([4.], ms.float32)
    other = ms.Tensor([3., 4., 5.], ms.float64)
    out = net(x, other)
    expect_out = np.array([5.0000, 5.6569, 6.4031], dtype=np.float64)
    assert np.allclose(out.asnumpy(), expect_out)
