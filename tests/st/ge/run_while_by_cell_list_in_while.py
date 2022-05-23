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
import mindspore.context as context
from mindspore import Tensor, nn
from mindspore.common import dtype as mstype

context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")


class OneCell(nn.Cell):
    def __init__(self, i):
        super().__init__()
        self.i = i

    def construct(self, x):
        return self.i * x


class WhileByCellListInWhile(nn.Cell):
    def __init__(self):
        super().__init__()
        self.cell_list = nn.CellList()
        self.cell_list.append(OneCell(4))
        self.cell_list.append(OneCell(5))
        self.cell_list.append(OneCell(6))

    def construct(self, n, x):
        out = n
        while x < 3:
            out += 4
            x += 1
        x = 0
        while x < 3:
            add = self.cell_list[x](n)
            out = out + add
            x += 1
        return out


def while_by_cell_list_in_while():
    net = WhileByCellListInWhile()
    n = Tensor(10, mstype.int32)
    x = Tensor(0, mstype.int32)
    out = net(n, x)
    return out


def test_while_by_cell_list_in_while_ge():
    """
    Feature: Control flow(while and case) implement in ge
    Description: run the whole graph sink in ascend in ge backend
    Expectation: success
    """
    out = while_by_cell_list_in_while()
    assert out == Tensor(172, mstype.int32)

if __name__ == "__main__":
    test_while_by_cell_list_in_while_ge()
