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
"""Test mindspore.rewrite"""
import sys

from mindspore.nn import Cell, Conv2d
from mindspore.rewrite import SymbolTree, ScopedValue
from mindspore.ops import operations as P


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
        super().__init__()
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
    node = stree.get_node('conv1')
    assert node is not None
    targets = node.get_targets()
    assert targets[0].value == 'c1'
    assert targets[1].value == 'c2'
    assert targets[2].value == 'c3'

class NetMultiTargetsWithAttribute(Cell):
    """Test cls for multiple targets."""
    def __init__(self):
        """Init."""
        super().__init__()
        self.conv1 = SubNet()
        self.add = P.Add()
        self.c1 = None
        self.c2 = None
        self.c3 = None

    def construct(self, x):
        """Construct."""
        self.c1, self.c2, self.c3 = self.conv1(x)
        x = self.add(self.c1, self.c2)
        x = self.add(x, self.c3)
        return x

def test_multi_targets_with_attribute():
    """
    Feature: Test multi-targets.
    Description: Test multi-targets with attribute.
    Expectation: Success.
    """
    net = NetMultiTargetsWithAttribute()
    stree = SymbolTree.create(net)
    conv1_node = stree.get_node("conv1")
    add = stree.get_node("add") # code: x = self.add(self.c1, self.c2)
    add_1 = stree.get_node("add_1") # x = self.add(x, self.c3)
    assert conv1_node.get_handler().get_target_users(0)[0] == (add.get_handler(), 0)
    assert conv1_node.get_handler().get_target_users(1)[0] == (add.get_handler(), 1)
    assert conv1_node.get_handler().get_target_users(2)[0] == (add_1.get_handler(), 1)
    # modify order of targets
    stree.get_handler().set_node_target(conv1_node.get_handler(), 0, ScopedValue.create_naming_value("c2", "self"))
    stree.get_handler().set_node_target(conv1_node.get_handler(), 1, ScopedValue.create_naming_value("c3", "self"))
    stree.get_handler().set_node_target(conv1_node.get_handler(), 2, ScopedValue.create_naming_value("c1", "self"))
    assert conv1_node.get_handler().get_target_users(0)[0] == (add.get_handler(), 1)
    assert conv1_node.get_handler().get_target_users(1)[0] == (add_1.get_handler(), 1)
    assert conv1_node.get_handler().get_target_users(2)[0] == (add.get_handler(), 0)
    codes = stree.get_code()
    if sys.version_info >= (3, 12):
        assert codes.count("self.c2, self.c3, self.c1 = self.conv1(x)")
    else:
        assert codes.count("(self.c2, self.c3, self.c1) = self.conv1(x)")

class NetMultiTargetsWithContinuousAssign(Cell):
    """Test cls for multiple targets."""
    def __init__(self):
        """Init."""
        super().__init__()
        self.conv1 = SubNet()
        self.add = P.Add()
        self.c1 = None
        self.c2 = None
        self.c3 = None

    def construct(self, x):
        """Construct."""
        c1, c2, c3 = self.c1, self.c2, self.c3 = self.conv1(x)
        x = self.add(c1, c2)
        x = self.add(x, c3)
        return x

def test_multi_targets_with_continuous_assign():
    """
    Feature: Test multi-targets.
    Description: Test multi-targets with continuous assign.
    Expectation: Success.
    """
    net = NetMultiTargetsWithContinuousAssign()
    stree = SymbolTree.create(net)
    conv1_node = stree.get_node("conv1")
    tuple_node = stree.get_node("tuple") # code: (c1, c2, c3) = (self.c1,self.c2, self.c3)
    add_node = stree.get_node("add") # code: x = self.add(c1, c2)
    add_1_node = stree.get_node("add_1") # code: x = self.add(c1, c2)
    assert conv1_node.get_handler().get_target_users(0)[0] == (tuple_node.get_handler(), 0)
    assert conv1_node.get_handler().get_target_users(1)[0] == (tuple_node.get_handler(), 1)
    assert conv1_node.get_handler().get_target_users(2)[0] == (tuple_node.get_handler(), 2)
    assert tuple_node.get_handler().get_target_users(0)[0] == (add_node.get_handler(), 0)
    assert tuple_node.get_handler().get_target_users(1)[0] == (add_node.get_handler(), 1)
    assert tuple_node.get_handler().get_target_users(2)[0] == (add_1_node.get_handler(), 1)
    codes = stree.get_code()
    if sys.version_info >= (3, 12):
        assert codes.count("self.c1, self.c2, self.c3 = self.conv1(x)")
        assert codes.count("c1, c2, c3 = self.c1, self.c2, self.c3")
    else:
        assert codes.count("(self.c1, self.c2, self.c3) = self.conv1(x)")
        assert codes.count("(c1, c2, c3) = (self.c1, self.c2, self.c3)")
