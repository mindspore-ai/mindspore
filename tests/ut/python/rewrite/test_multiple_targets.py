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
from mindspore.nn import Cell, Conv2d
from mindspore.rewrite import SymbolTree
from mindspore.ops import operations as P
from .utils import get_node_by_index


class SubNet(Cell):
    """Sample cell which returns multiple features."""
    def __init__(self):
        """Init."""
        super().__init__()
        self.conv = Conv2d(1, 10, 3)

    def construct(self, x):
        """Construct."""
        c1 = self.conv(x)
        c2 = self.conv(c1)
        c3 = self.conv(c2)
        return c1, c2, c3


class NetMultiTargets(Cell):
    """Test cls for multiple targets."""
    def __init__(self):
        """Init."""
        super(NetMultiTargets, self).__init__()
        self.conv1 = SubNet()
        self.add = P.Add()

    def construct(self, x):
        """Construct."""
        c1, c2, c3 = self.conv1(x)
        x = self.add(c1, c2)
        x = self.add(x, c3)
        return x


def test_multi_targets():
    """
    Feature: Test multi-targets.
    Description: Test multi-targets.
    Expectation: Success.
    """
    test_cls = NetMultiTargets()
    stree = SymbolTree.create(test_cls)
    node = get_node_by_index(stree, 2)
    assert node is not None
    targets = node.get_targets()
    assert targets[0].value == 'c1'
    assert targets[1].value == 'c2'
    assert targets[2].value == 'c3'
