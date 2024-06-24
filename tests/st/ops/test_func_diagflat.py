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
from tests.mark_utils import arg_mark
import numpy as np
import pytest

import mindspore as ms
import mindspore.nn as nn
from mindspore import ops


class Net(nn.Cell):
    def construct(self, x):
        out1 = ops.diagflat(x, -1)
        out2 = ops.diagflat(x, 1)
        return out1, out2


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level2', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ops_diagflat(mode):
    """
    Feature: ops.diagflat
    Description: Verify the result of diagflat
    Expectation: success
    """
    ms.set_context(mode=mode)
    x = ms.Tensor([-0.5, 0.5, 3], ms.float32)
    net = Net()
    output1, output2 = net(x)
    expect_output1 = [[0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
                      [-5.00000000e-01, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
                      [0.00000000e+00, 5.00000000e-01, 0.00000000e+00, 0.00000000e+00],
                      [0.00000000e+00, 0.00000000e+00, 3.00000000e+00, 0.00000000e+00]]
    expect_output2 = [[0.00000000e+00, -5.00000000e-01, 0.00000000e+00, 0.00000000e+00],
                      [0.00000000e+00, 0.00000000e+00, 5.00000000e-01, 0.00000000e+00],
                      [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 3.00000000e+00],
                      [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00]]
    assert np.allclose(output1.asnumpy(), expect_output1)
    assert np.allclose(output2.asnumpy(), expect_output2)
