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
    def construct(self, x, other):
        return x.arctan2(other)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos', 'platform_gpu', 'platform_ascend'],
          level_mark='level2',
          card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_tensor_arctan2(mode):
    """
    Feature: tensor.arctan2
    Description: Verify the result of arctan2
    Expectation: success
    """
    ms.set_context(mode=mode)
    x = Tensor(np.array([0.9041, 0.0196, -0.3108, -2.4423]), ms.float32)
    y = Tensor(np.array([0.5, 0.5, 0.5, 0.5]), ms.float32)
    net = Net()
    output = net(x, y)
    expect_output = [1.06562507e+000, 3.91799398e-002, -5.56150615e-001, -1.36886156e+000]
    assert np.allclose(output.asnumpy(), expect_output)
