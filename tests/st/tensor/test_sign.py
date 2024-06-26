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
    def construct(self, x):
        return x.sign()


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos', 'platform_gpu'],
          level_mark='level2',
          card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_sign_normal(mode):
    """
    Feature: sign
    Description: Verify the result of sign
    Expectation: success
    """
    ms.set_context(mode=mode)
    net = Net()
    x1 = ms.Tensor([[-1, 0, 2, 4, 6], [2, 3, 5, -6, 0]])
    output1 = net(x1)
    expect_output1 = np.array([[-1, 0, 1, 1, 1],
                               [1, 1, 1, - 1, 0]])

    x2 = ms.Tensor([[-1, 0, float('inf'), 4, float('nan')], [2, 3, float('-inf'), -6, 0]])
    output2 = net(x2)
    expect_output2 = np.array([[-1., 0., 1., 1., 0.],
                               [1., 1., -1., -1., 0.]])
    assert np.allclose(output1.asnumpy(), expect_output1)
    assert np.allclose(output2.asnumpy(), expect_output2)
