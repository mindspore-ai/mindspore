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
import numpy as np
import mindspore as ms
import mindspore.nn as nn
from mindspore import context, Tensor
from mindspore.common import dtype as mstype
from tests.mark_utils import arg_mark

context.set_context(mode=context.GRAPH_MODE)


class TestCellListInsertNet(nn.Cell):
    def __init__(self):
        super(TestCellListInsertNet, self).__init__()
        self.cell_list = nn.CellList()
        self.cell_list.insert(0, nn.Cell())
        self.cell_list.insert(1, nn.Dense(1, 2))

    def construct(self):
        return len(self.cell_list)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos', 'platform_gpu', 'platform_ascend'],
          level_mark='level1',
          card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_celllist_insert_method_boundary_cond(mode):
    """
    Feature: CellList.insert()
    Description: Verify the result of CellDict.insert(index, cell) in boundary conditions.
    Expectation: success
    """
    ms.set_context(mode=mode)
    net = TestCellListInsertNet()
    expect_output = 2
    output = net()
    assert np.allclose(output, expect_output)
    x = nn.Dense(1, 2)
    assert type(x) is type(net.cell_list[1])


class EmbeddedCellDictNet(nn.Cell):
    def __init__(self):
        super(EmbeddedCellDictNet, self).__init__()
        self.cell_dict = nn.CellDict({'conv': nn.Conv2d(3, 2, 2), "relu": nn.ReLU()})
        self.cell_list = nn.CellList([self.cell_dict])

    def construct(self, x):
        for cell_dict in self.cell_list:
            for net in cell_dict.values():
                x = net(x)
        return x


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos', 'platform_gpu', 'platform_ascend'],
          level_mark='level1',
          card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_celllist_embed_celldict_case(mode):
    """
    Feature: CellList.extend()
    Description: Verify the result of initializing CellList by CellDict
    Expectation: success
    """
    with pytest.raises(TypeError):
        EmbeddedCellDictNet()


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_parse_cell_list():
    """
    Feature: Parse CellList.
    Description: Parse CellList.
    Expectation: success
    """
    class NormLayerBlock(nn.CellList):
        def __init__(self, in_channels, out_channels):
            norm_layer = nn.BatchNorm2d
            _layers = [norm_layer(out_channels)] if out_channels == in_channels else []
            super().__init__(_layers)

        def construct(self):
            return len(self)

    net = NormLayerBlock(in_channels=4, out_channels=4)
    out = net()
    assert out == 1


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_iter_for_in_cell_list():
    """
    Feature: Parse CellList.
    Description: Parse CellList.
    Expectation: success
    """
    class IterNet(nn.CellList):
        def __init__(self):
            _layers = [nn.Conv2d(1, 1, 1)]
            super().__init__(_layers)

        def construct(self, x):
            return sum([mod(x) for mod in self])

    net = IterNet()
    x = Tensor(np.ones([1, 1, 2, 2]), mstype.float32)
    context.set_context(mode=context.PYNATIVE_MODE)
    pynative_out = net(x)
    context.set_context(mode=context.GRAPH_MODE)
    graph_out = net(x)
    assert (pynative_out == graph_out).all()
