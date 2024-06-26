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
    def construct(self, x, divisor):
        return x.remainder(divisor)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos', 'platform_gpu', 'platform_ascend'],
          level_mark='level2',
          card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_tensor_remainder(mode):
    """
    Feature: tensor.remainder
    Description: Verify the result of remainder
    Expectation: success
    """
    ms.set_context(mode=mode)
    x = Tensor(np.array([-3, -2, -1, 1, 2, 3]), ms.float32)
    net = Net()
    output = net(x, -1.5)
    expect_output1 = [0, -0.5, -1, -0.5, -1, 0]
    assert np.allclose(output.asnumpy(), expect_output1)
    x = Tensor(np.array([-30, -17, -3, 61, 17, 30]), ms.float32)
    y = Tensor(np.array([-1.5, -2, -3.5, 1.5, 2, 3.5]), ms.float32)
    output = net(x, y)
    expect_output2 = [0, -1, -3, 1, 1, 2]
    assert np.allclose(output.asnumpy(), expect_output2)
