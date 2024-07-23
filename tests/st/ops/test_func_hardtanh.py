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
import mindspore.ops as ops


class Net(nn.Cell):
    def construct(self, x):
        return ops.hardtanh(x, min_val=-50, max_val=50)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level2', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_hardtanh(mode):
    """
    Feature: Test hardtanh
    Description: Test the functionality of hardtanh
    Expectation: Success
    """
    ms.set_context(mode=mode)
    net = Net()
    x = ms.Tensor([[[71, -25, -50, -47],
                    [-94, -41, -23, 49],
                    [65, 31, 2, 38]],
                   [[-75, 54, -63, 62],
                    [-70, -29, -80, -67],
                    [-19, -98, -79, -84]]], ms.int32)
    out = net(x)
    expect_out = [[[50, -25, -50, -47],
                   [-50, -41, -23, 49],
                   [50, 31, 2, 38]],
                  [[-50, 50, -50, 50],
                   [-50, -29, -50, -50],
                   [-19, -50, -50, -50]]]
    assert np.allclose(out.asnumpy(), expect_out)
