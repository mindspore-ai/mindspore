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
"""test for CellList."""

import mindspore.nn as nn
from mindspore.rewrite import SymbolTree


class MyNet(nn.Cell):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(16, 16, 3)
        self.blocks = nn.CellList()
        for _ in range(3):
            block = nn.ReLU()
            self.blocks.append(block)

    def construct(self, x):
        x = self.conv(x)
        for block in self.blocks:
            x = block(x)
        return x


def test_for_celllist():
    """
    Feature: for parser for nn.CellList.
    Description: parse for statement when the iterator is nn.CellList.
    Expectation: Success.
    """
    net = MyNet()
    stree = SymbolTree.create(net)
    assert len(stree.get_handler().nodes()) == 6
    assert stree.get_handler().get_node("blocks0")
    assert stree.get_handler().get_node("blocks1")
    assert stree.get_handler().get_node("blocks2")

    codes = stree.get_code()
    assert codes.count("for block in self.blocks:") == 1
    assert codes.count("pass") == 1
    assert codes.count("x = self.blocks0(x)") == 1
    assert codes.count("x = self.blocks1(x)") == 1
    assert codes.count("x = self.blocks2(x)") == 1
