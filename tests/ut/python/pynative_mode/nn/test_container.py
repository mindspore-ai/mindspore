# Copyright 2020 Huawei Technologies Co., Ltd
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
""" test_container """
from collections import OrderedDict
import numpy as np
import pytest

import mindspore.nn as nn
from mindspore import context, Tensor

context.set_context(mode=context.PYNATIVE_MODE)


weight = Tensor(np.ones([2, 2]))
conv2 = nn.Conv2d(3, 64, (3, 3), stride=2, padding=0)

kernel_size = 3
stride = 2
padding = 1
avg_pool = nn.AvgPool2d(kernel_size, stride)


class TestSequentialCell():
    """ TestSequentialCell definition """

    def test_SequentialCell_init(self):
        m = nn.SequentialCell()
        assert not m

    def test_SequentialCell_init2(self):
        m = nn.SequentialCell([conv2])
        assert len(m) == 1

    def test_SequentialCell_init3(self):
        m = nn.SequentialCell([conv2, avg_pool])
        assert len(m) == 2

    def test_SequentialCell_init4(self):
        m = nn.SequentialCell(OrderedDict(
            [('cov2d', conv2), ('avg_pool', avg_pool)]))
        assert len(m) == 2

    def test_getitem1(self):
        m = nn.SequentialCell(OrderedDict(
            [('cov2d', conv2), ('avg_pool', avg_pool)]))
        assert m[0] == conv2

    def test_getitem2(self):
        m = nn.SequentialCell(OrderedDict(
            [('cov2d', conv2), ('avg_pool', avg_pool)]))
        assert len(m[0:2]) == 2
        assert m[:2][1] == avg_pool

    def test_setitem1(self):
        m = nn.SequentialCell(OrderedDict(
            [('cov2d', conv2), ('avg_pool', avg_pool)]))
        m[1] = conv2
        assert m[1] == m[0]

    def test_setitem2(self):
        m = nn.SequentialCell(OrderedDict(
            [('cov2d', conv2), ('avg_pool', avg_pool)]))
        with pytest.raises(TypeError):
            m[1.0] = conv2

    def test_delitem1(self):
        m = nn.SequentialCell(OrderedDict(
            [('cov2d', conv2), ('avg_pool', avg_pool)]))
        del m[0]
        assert len(m) == 1

    def test_delitem2(self):
        m = nn.SequentialCell(OrderedDict(
            [('cov2d', conv2), ('avg_pool', avg_pool)]))
        del m[:]
        assert not m

    def test_construct(self):
        m = nn.SequentialCell(OrderedDict(
            [('cov2d', conv2), ('avg_pool', avg_pool)]))
        m.construct(Tensor(np.ones([1, 3, 16, 50], dtype=np.float32)))


class TestCellList():
    """ TestCellList definition """

    def test_init1(self):
        cell_list = nn.CellList([conv2, avg_pool])
        assert len(cell_list) == 2

    def test_init2(self):
        with pytest.raises(TypeError):
            nn.CellList(["test"])

    def test_getitem(self):
        cell_list = nn.CellList([conv2, avg_pool])
        assert cell_list[0] == conv2
        temp_cells = cell_list[:]
        assert temp_cells[1] == avg_pool

    def test_setitem(self):
        cell_list = nn.CellList([conv2, avg_pool])
        cell_list[0] = avg_pool
        assert cell_list[0] == cell_list[1]

    def test_delitem(self):
        cell_list = nn.CellList([conv2, avg_pool])
        del cell_list[0]
        assert len(cell_list) == 1
        del cell_list[:]
        assert not cell_list

    def test_iter(self):
        cell_list = nn.CellList([conv2, avg_pool])
        for _ in cell_list:
            break

    def test_add(self):
        cell_list = nn.CellList([conv2, avg_pool])
        cell_list += [conv2]
        assert len(cell_list) == 3
        assert cell_list[0] == cell_list[2]

    def test_insert(self):
        cell_list = nn.CellList([conv2, avg_pool])
        cell_list.insert(0, avg_pool)
        assert len(cell_list) == 3
        assert cell_list[0] == cell_list[2]

    def test_append(self):
        cell_list = nn.CellList([conv2, avg_pool])
        cell_list.append(conv2)
        assert len(cell_list) == 3
        assert cell_list[0] == cell_list[2]

    def test_extend(self):
        cell_list = nn.CellList()
        cell_list.extend([conv2, avg_pool])
        assert len(cell_list) == 2
        assert cell_list[0] == conv2
