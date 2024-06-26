# Copyright 2023 Huawei Technologies Co., Ltd
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
from mindspore import Tensor, nn


class Net(nn.Cell):
    def construct(self, x, y):
        return x.floor_divide(y)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos', 'platform_gpu', 'platform_ascend'],
          level_mark='level1',
          card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_net(mode):
    """
    Feature: test floor_divide op
    Description: verify the result of floor_divide
    Expectation: assertion success
    """
    ms.set_context(mode=mode)
    x1 = Tensor(np.array([2, 4, -1]), ms.int32)
    x2 = Tensor(np.array([3, 3, 3]), ms.int32)
    floor_divide = Net()
    output = floor_divide(x1, x2)
    np_out = np.array([0, 1, -1])
    assert np.allclose(output.asnumpy(), np_out)
