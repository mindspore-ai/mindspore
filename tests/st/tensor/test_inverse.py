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
        return x.inverse()


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos', 'platform_gpu'],
          level_mark='level2',
          card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_tensor_inverse(mode):
    """
    Feature: tensor.inverse
    Description: Verify the result of inverse
    Expectation: success
    """
    ms.set_context(mode=mode)
    x = ms.Tensor([[1., 2, 3],
                   [4, 5., 6],
                   [8, 8, 9]], ms.float32)
    net = Net()
    output = net(x)
    expect_output = [[1.0000008, -2.000001, 1.0000005],
                     [-4.0000014, 5.000002, -2.000001],
                     [2.6666675, -2.6666675, 1.0000002]]
    assert np.allclose(output.asnumpy(), expect_output)
