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

import pytest
import mindspore as ms
import mindspore.nn as nn
import numpy as np


class TestCellListInsertNet(nn.Cell):
    def __init__(self):
        super(TestCellListInsertNet, self).__init__()
        self.cell_list = nn.CellList()
        self.cell_list.insert(0, nn.Cell())

    def construct(self):
        return len(self.cell_list)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_arm_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_celllist_insert_method_boundary_cond(mode):
    """
    Feature: CellList.insert()
    Description: Verify the result of CellDict.insert(index, cell) in boundary conditions.
    Expectation: success
    """
    ms.set_context(mode=mode)
    net = TestCellListInsertNet()
    expect_output = 1
    output = net()
    assert np.allclose(output, expect_output)
