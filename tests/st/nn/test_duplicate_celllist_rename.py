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

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from tests.mark_utils import arg_mark


class Net(nn.Cell):
    def __init__(self):
        super().__init__()
        self.cell_list1 = nn.CellList()
        self.cell_list2 = nn.CellList()

        m1 = nn.Dense(2, 2, weight_init='ones', bias_init='zeros')
        m2 = nn.Dense(2, 2, weight_init='ones', bias_init='zeros')
        self.cell_list1.append(m1)
        self.cell_list2.append(m2)

    def construct(self, x1, x2):
        a = self.cell_list1[0](x1)
        b = self.cell_list2[0](x2)
        return a + b


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos', 'platform_gpu', 'platform_ascend'],
          level_mark='level2',
          card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_celllist_rename_normal(mode):
    """
    Feature: duplicate celllist rename
    Description: Verify the result of duplicate celllist rename
    Expectation: success
    """
    ms.set_context(mode=mode)
    x1 = ops.ones((2, 2), ms.float32)
    x2 = ops.ones((2, 2), ms.float32)
    net = Net()
    out = net(x1, x2)
    expect_out = np.array([[4., 4.],
                           [4., 4.]])
    assert np.allclose(out.asnumpy(), expect_out)
