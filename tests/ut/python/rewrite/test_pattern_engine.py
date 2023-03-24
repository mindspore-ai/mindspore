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
import ast
from collections import OrderedDict

from mindspore.nn import Cell, Conv2d, BatchNorm2d, ReLU
from mindspore.ops import Add, AddN
from mindspore.rewrite import ScopedValue, Node, SymbolTree
from mindspore.rewrite import PatternEngine, PatternNode, Replacement, VarNode
from .utils import get_symbol_tree_nodes_count


def test_tree_pattern_match():
    """
    Feature: Python api PatternEngine.
    Description: Construct a tree PatternEngine and apply it on a SymbolTree, check SymbolTree after PatternEngine
                 applied.
    Expectation: Success.
    """
    assert True


def test_leak_pattern_match():
    """
    Feature: Python api PatternEngine.
    Description: Construct a leaked tree PatternEngine and apply it on a SymbolTree, check SymbolTree after
                 PatternEngine applied.
    Expectation: Failure.
    """
    assert True


class ChainNetwork(Cell):
    def __init__(self):
        super().__init__()
        self.conv = Conv2d(16, 16, 3)
        self.bn = BatchNorm2d(16)
        self.relu1 = ReLU()
        self.relu2 = ReLU()
        self.relu3 = ReLU()

    def construct(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu1(x)
        x = self.relu2(x)
        x = self.relu3(x)
        return x


def test_one_to_one_pattern():
    """
    Feature: Python api PatternEngine.
    Description: Construct a one-to-one PatternEngine and apply it on a SymbolTree, check SymbolTree after PatternEngine
                 applied.
    Expectation: Success.
    """

    class BnReplacement(Replacement):
        def build(self, pattern: PatternNode, is_chain_pattern: bool, matched: OrderedDict) -> [Node]:
            bn_node: Node = matched.get(pattern.name())

            conv = Conv2d(16, 16, 3)
            conv_node = Node.create_call_cell(conv, ['x1'], bn_node.get_args(), bn_node.get_kwargs())
            return [conv_node]

    class BnReplace(PatternEngine):
        def __init__(self):
            super().__init__([BatchNorm2d], BnReplacement())

    net = ChainNetwork()
    stree = SymbolTree.create(net)
    conv = stree.get_node("conv")
    bn = stree.get_node("bn")
    relu1 = stree.get_node("relu1")
    construct_ast: ast.FunctionDef = getattr(stree.get_handler(), "_root_ast")
    assert conv is not None
    assert bn is not None
    assert relu1 is not None
    assert len(construct_ast.body) == 6
    assert get_symbol_tree_nodes_count(stree) == 7

    bn_replace = BnReplace()
    bn_replace.apply(stree)

    assert len(construct_ast.body) == 6
    assert get_symbol_tree_nodes_count(stree) == 7
    conv = stree.get_node("conv")
    bn = stree.get_node("bn")
    relu1 = stree.get_node("relu1")
    new_conv = stree.get_node("x1")
    assert conv is not None
    assert bn is None
    assert relu1 is not None
    assert new_conv is not None

    # check conv topological order
    assert len(conv.get_users()) == 1
    assert conv.get_users()[0] == new_conv
    # check new_conv topological order
    assert len(new_conv.get_inputs()) == 1
    assert new_conv.get_inputs()[0] == conv
    assert len(new_conv.get_users()) == 1
    assert new_conv.get_users()[0] == relu1
    # check source code order
    assert getattr(conv.get_handler(), "_next") == new_conv.get_handler()
    assert getattr(new_conv.get_handler(), "_next") == relu1.get_handler()
    assert getattr(relu1.get_handler(), "_prev") == new_conv.get_handler()
    assert getattr(new_conv.get_handler(), "_prev") == conv.get_handler()
    # # check arg edge
    assert len(conv.get_targets()) == 1
    assert len(new_conv.get_args()) == 1
    assert conv.get_targets()[0] == new_conv.get_args()[0]
    assert len(new_conv.get_targets()) == 1
    assert len(relu1.get_args()) == 1
    assert new_conv.get_targets()[0] == relu1.get_args()[0]


def test_one_to_multi_chain_pattern():
    """
    Feature: Python api PatternEngine.
    Description: Construct a one-to-multi PatternEngine and apply it on a SymbolTree, check SymbolTree after
                 PatternEngine applied.
    Expectation: Success.
    """

    class BnReplacement(Replacement):
        def build(self, pattern: PatternNode, is_chain_pattern: bool, matched: OrderedDict) -> [Node]:
            assert is_chain_pattern
            assert pattern.type() == BatchNorm2d
            bn_node: Node = matched.get(pattern.name())
            assert bn_node is not None

            # Replacement should ensure target is unique in result
            # Replacement should ensure args and kwargs are well set by topological relation
            conv1 = Conv2d(16, 16, 3)
            conv_node1 = Node.create_call_cell(conv1, ['x1'], bn_node.get_args(), bn_node.get_kwargs())
            conv2 = Conv2d(16, 16, 5)
            conv_node2 = Node.create_call_cell(conv2, ['x2'], [ScopedValue.create_naming_value('x1')])
            return [conv_node1, conv_node2]

    class BnReplace(PatternEngine):
        def __init__(self):
            super().__init__([BatchNorm2d], BnReplacement())

    net = ChainNetwork()
    stree = SymbolTree.create(net)
    conv = stree.get_node("conv")
    bn = stree.get_node("bn")
    relu1 = stree.get_node("relu1")
    construct_ast: ast.FunctionDef = getattr(stree.get_handler(), "_root_ast")
    assert conv is not None
    assert bn is not None
    assert relu1 is not None
    assert len(construct_ast.body) == 6
    assert get_symbol_tree_nodes_count(stree) == 7

    bn_replace = BnReplace()
    bn_replace.apply(stree)

    assert len(construct_ast.body) == 7
    assert get_symbol_tree_nodes_count(stree) == 8
    conv = stree.get_node("conv")
    bn = stree.get_node("bn")
    relu1 = stree.get_node("relu1")
    new_conv1 = stree.get_node("x1")
    new_conv2 = stree.get_node("x2")
    assert conv is not None
    assert bn is None
    assert relu1 is not None
    assert new_conv1 is not None
    assert new_conv2 is not None

    # check conv topological order
    assert len(conv.get_users()) == 1
    assert conv.get_users()[0] == new_conv1
    # check new_conv1 topological order
    assert len(new_conv1.get_inputs()) == 1
    assert new_conv1.get_inputs()[0] == conv
    assert len(new_conv1.get_users()) == 1
    assert new_conv1.get_users()[0] == new_conv2
    # check new_conv2 topological order
    assert len(new_conv2.get_inputs()) == 1
    assert new_conv2.get_inputs()[0] == new_conv1
    assert len(new_conv2.get_users()) == 1
    assert new_conv2.get_users()[0] == relu1
    # check source code order
    assert getattr(conv.get_handler(), "_next") == new_conv1.get_handler()
    assert getattr(new_conv1.get_handler(), "_next") == new_conv2.get_handler()
    assert getattr(new_conv2.get_handler(), "_next") == relu1.get_handler()
    assert getattr(relu1.get_handler(), "_prev") == new_conv2.get_handler()
    assert getattr(new_conv2.get_handler(), "_prev") == new_conv1.get_handler()
    assert getattr(new_conv1.get_handler(), "_prev") == conv.get_handler()
    # check arg edge
    assert len(conv.get_targets()) == 1
    assert len(new_conv1.get_args()) == 1
    assert conv.get_targets()[0] == new_conv1.get_args()[0]

    assert len(new_conv1.get_targets()) == 1
    assert len(new_conv2.get_args()) == 1
    assert new_conv1.get_targets()[0] == new_conv2.get_args()[0]

    assert len(new_conv2.get_targets()) == 1
    assert len(relu1.get_args()) == 1
    assert new_conv2.get_targets()[0] == relu1.get_args()[0]


class TreeNetwork(Cell):
    def __init__(self):
        super().__init__()
        self.conv1 = Conv2d(16, 16, 3)
        self.conv2 = Conv2d(16, 16, 5)
        self.add = Add()
        self.relu = ReLU()
        self.relu1 = ReLU()
        self.relu2 = ReLU()

    def construct(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x = self.add(x1, x2)
        x = self.relu(x)
        x1 = self.relu1(x)
        x2 = self.relu2(x)
        x = self.add(x1, x2)
        return x


def test_tree_pattern():
    """
    Feature: Python api PatternEngine.
    Description: Construct a multi-to-multi PatternEngine and apply it on a SymbolTree, check SymbolTree after
                 PatternEngine applied.
    Expectation: Success.
    """

    class AddReluReplacement(Replacement):
        def build(self, pattern: PatternNode, is_chain_pattern: bool, matched: OrderedDict) -> [Node]:
            assert is_chain_pattern
            assert pattern.type() == ReLU
            relu_node: Node = matched.get(pattern.name())
            assert relu_node is not None
            assert len(pattern.get_inputs()) == 1
            add_pattern = pattern.get_inputs()[0]
            assert add_pattern.type() == Add
            add_node: Node = matched.get(add_pattern.name())
            assert add_node is not None
            assert not add_pattern.get_inputs()

            # can not use add_node here
            new_add1 = Add()
            new_add1_node = Node.create_call_cell(new_add1, ['new_add_1'], add_node.get_args(), add_node.get_kwargs())
            new_relu1 = ReLU()
            new_relu1_node = Node.create_call_cell(new_relu1, ['new_relu_1'],
                                                   [ScopedValue.create_naming_value('new_add_1')])
            new_relu2 = ReLU()
            new_relu2_node = Node.create_call_cell(new_relu2, ['new_relu_2'],
                                                   [ScopedValue.create_naming_value('new_add_1')])
            new_add2 = Add()
            new_add2_node = Node.create_call_cell(new_add2, ['new_add_2'],
                                                  [ScopedValue.create_naming_value('new_relu_1'),
                                                   ScopedValue.create_naming_value('new_relu_2')])
            return [new_add1_node, new_relu1_node, new_relu2_node, new_add2_node]

    class AddReluPattern(PatternEngine):
        def __init__(self):
            super().__init__([Add, ReLU], AddReluReplacement())

    net = TreeNetwork()
    stree = SymbolTree.create(net)
    conv1 = stree.get_node("conv1")
    conv2 = stree.get_node("conv2")
    add = stree.get_node("add")
    relu = stree.get_node("relu")
    relu1 = stree.get_node("relu1")
    relu2 = stree.get_node("relu2")
    assert conv1 is not None
    assert conv2 is not None
    assert add is not None
    assert relu is not None
    assert relu1 is not None
    assert relu2 is not None
    construct_ast: ast.FunctionDef = getattr(stree.get_handler(), "_root_ast")
    assert len(construct_ast.body) == 8
    assert get_symbol_tree_nodes_count(stree) == 9

    add_relu_pattern = AddReluPattern()
    add_relu_pattern.apply(stree)

    assert len(construct_ast.body) == 10
    assert get_symbol_tree_nodes_count(stree) == 11
    conv1 = stree.get_node("conv1")
    conv2 = stree.get_node("conv2")
    add = stree.get_node("add")
    relu = stree.get_node("relu")
    relu1 = stree.get_node("relu1")
    relu2 = stree.get_node("relu2")
    new_add = stree.get_node("new_add")
    new_relu = stree.get_node("new_relu")
    new_relu_1 = stree.get_node("new_relu_1")
    new_add_1 = stree.get_node("new_add_1")

    assert conv1 is not None
    assert conv2 is not None
    assert add is None
    assert relu is None
    assert relu1 is not None
    assert relu2 is not None
    assert new_add is not None
    assert new_relu is not None
    assert new_relu_1 is not None
    assert new_add_1 is not None

    # check conv1 topological order
    assert len(conv1.get_users()) == 1
    assert conv1.get_users()[0] == new_add
    # check conv2 topological order
    assert len(conv2.get_users()) == 1
    assert conv2.get_users()[0] == new_add
    # check new_add topological order
    assert len(new_add.get_inputs()) == 2
    assert new_add.get_inputs()[0] == conv1
    assert new_add.get_inputs()[1] == conv2
    assert len(new_add.get_users()) == 2
    assert new_add.get_users()[0] == new_relu
    assert new_add.get_users()[1] == new_relu_1
    # check new_relu topological order
    assert len(new_relu.get_inputs()) == 1
    assert new_relu.get_inputs()[0] == new_add
    assert len(new_relu.get_users()) == 1
    assert new_relu.get_users()[0] == new_add_1
    # check new_relu_1 topological order
    assert len(new_relu_1.get_inputs()) == 1
    assert new_relu_1.get_inputs()[0] == new_add
    assert len(new_relu_1.get_users()) == 1
    assert new_relu_1.get_users()[0] == new_add_1
    # check new_add_1 topological order
    assert len(new_add_1.get_inputs()) == 2
    assert new_add_1.get_inputs()[0] == new_relu_1
    assert new_add_1.get_inputs()[1] == new_relu
    assert len(new_add_1.get_users()) == 2
    assert new_add_1.get_users()[0] == relu1
    assert new_add_1.get_users()[1] == relu2
    # check source code order
    assert getattr(conv1.get_handler(), "_next") == conv2.get_handler()
    assert getattr(conv2.get_handler(), "_next") == new_add.get_handler()
    assert getattr(new_add.get_handler(), "_next") == new_relu.get_handler()
    assert getattr(new_relu.get_handler(), "_next") == new_relu_1.get_handler()
    assert getattr(new_relu_1.get_handler(), "_next") == new_add_1.get_handler()
    assert getattr(new_add_1.get_handler(), "_next") == relu1.get_handler()
    assert getattr(relu1.get_handler(), "_prev") == new_add_1.get_handler()
    assert getattr(new_add_1.get_handler(), "_prev") == new_relu_1.get_handler()
    assert getattr(new_relu_1.get_handler(), "_prev") == new_relu.get_handler()
    assert getattr(new_relu.get_handler(), "_prev") == new_add.get_handler()
    assert getattr(new_add.get_handler(), "_prev") == conv2.get_handler()
    assert getattr(conv2.get_handler(), "_prev") == conv1.get_handler()
    # check arg edge
    assert len(conv1.get_targets()) == 1
    assert len(conv2.get_targets()) == 1
    assert len(new_add.get_args()) == 2
    assert conv1.get_targets()[0] == new_add.get_args()[0]
    assert conv2.get_targets()[0] == new_add.get_args()[1]

    assert len(new_add.get_targets()) == 1
    assert len(new_relu.get_args()) == 1
    assert len(new_relu_1.get_args()) == 1
    assert new_add.get_targets()[0] == new_relu.get_args()[0]
    assert new_add.get_targets()[0] == new_relu_1.get_args()[0]

    assert len(new_relu.get_targets()) == 1
    assert len(new_relu_1.get_targets()) == 1
    assert len(new_add_1.get_args()) == 2
    assert new_relu.get_targets()[0] == new_add_1.get_args()[1]
    assert new_relu_1.get_targets()[0] == new_add_1.get_args()[0]

    assert len(new_add_1.get_targets()) == 1
    assert len(relu1.get_args()) == 1
    assert len(relu2.get_args()) == 1
    assert new_add_1.get_targets()[0] == relu1.get_args()[0]
    assert new_add_1.get_targets()[0] == relu2.get_args()[0]


class TreeNetwork2(Cell):
    def __init__(self):
        super().__init__()
        self.conv1 = Conv2d(16, 16, 1)
        self.conv2 = Conv2d(16, 16, 3)
        self.add1 = AddN()
        self.add2 = AddN()
        self.relu = ReLU()

    def construct(self, x, y, z):
        x = self.conv1(x)
        y = self.conv2(y)
        z = self.add1(x, y, z)
        z = self.add2(x, y, z)
        z = self.relu(z)
        return z


class MultiInputPattern(PatternEngine):
    class MultiInputReplacement(Replacement):
        def build(self, pattern: PatternNode, is_chain_pattern: bool, matched: OrderedDict) -> [Node]:
            assert not is_chain_pattern
            assert pattern.type() == AddN
            addn2_node: Node = matched.get(pattern.name())
            assert addn2_node is not None
            assert len(pattern.get_inputs()) == 3
            conv1_pn = pattern.get_inputs()[0]
            conv2_pn = pattern.get_inputs()[1]
            addn1_pn = pattern.get_inputs()[2]
            assert conv1_pn.type() == Conv2d
            assert conv2_pn.type() == Conv2d
            assert addn1_pn.type() == AddN
            conv1_node: Node = matched.get(conv1_pn.name())
            conv2_node: Node = matched.get(conv2_pn.name())
            addn1_node: Node = matched.get(addn1_pn.name())
            assert conv1_node is not None
            assert conv2_node is not None
            assert addn1_node is not None
            assert len(conv1_node.get_inputs()) == 1
            assert len(conv2_node.get_inputs()) == 1
            assert len(addn1_node.get_inputs()) == 3
            arg1 = conv1_node.get_args()[0]
            arg2 = conv2_node.get_args()[0]
            arg3 = addn1_node.get_args()[2]

            # can not use add_node here
            new_add1 = Add()
            new_add1_node = Node.create_call_cell(new_add1, ['new_add1'], [arg1, arg2])
            new_add2 = Add()
            new_add2_node = Node.create_call_cell(new_add2, ['new_add2'], [ScopedValue.create_naming_value('new_add1'),
                                                                           arg3])
            return [new_add1_node, new_add2_node]

    def __init__(self):
        conv1_pn = PatternNode("conv1", Conv2d)
        conv2_pn = PatternNode("conv2", Conv2d)
        addn1_pn = PatternNode("addn1", AddN)
        addn2_pn = PatternNode("addn2", AddN)
        conv1_pn.set_inputs([VarNode()])
        conv2_pn.set_inputs([VarNode()])
        addn1_pn.set_inputs([conv1_pn, conv2_pn, VarNode()])
        addn2_pn.set_inputs([conv1_pn, conv2_pn, addn1_pn])
        super().__init__(addn2_pn, MultiInputPattern.MultiInputReplacement())


def test_multi_input_to_multi_pattern_tree_pattern():
    """
    Feature: Python api PatternEngine.
    Description: Construct a multi-to-multi PatternEngine and apply it on a SymbolTree, check SymbolTree after
                 PatternEngine applied.
    Expectation: Success.
    """

    net = TreeNetwork2()
    stree = SymbolTree.create(net)
    conv1 = stree.get_node("conv1")
    conv2 = stree.get_node("conv2")
    add1 = stree.get_node("add1")
    add2 = stree.get_node("add2")
    relu = stree.get_node("relu")
    assert conv1 is not None
    assert conv2 is not None
    assert add1 is not None
    assert add2 is not None
    assert relu is not None
    construct_ast: ast.FunctionDef = getattr(stree.get_handler(), "_root_ast")
    assert len(construct_ast.body) == 6
    assert get_symbol_tree_nodes_count(stree) == 9

    multi_input_pattern = MultiInputPattern()
    multi_input_pattern.apply(stree)

    assert len(construct_ast.body) == 4
    assert get_symbol_tree_nodes_count(stree) == 7
    conv1 = stree.get_node("conv1")
    conv2 = stree.get_node("conv2")
    add1 = stree.get_node("add1")
    add2 = stree.get_node("add2")
    relu = stree.get_node("relu")
    new_add1 = stree.get_node("new_add1")
    new_add2 = stree.get_node("new_add2")
    inputx = stree.get_node("input_x")
    inputy = stree.get_node("input_y")
    inputz = stree.get_node("input_z")
    assert conv1 is None
    assert conv2 is None
    assert add1 is None
    assert add2 is None
    assert relu is not None
    assert new_add1 is not None
    assert new_add2 is not None
    assert inputx is not None
    assert inputy is not None
    assert inputz is not None

    # check inputx topological order
    assert len(inputx.get_users()) == 1
    assert inputx.get_users()[0] == new_add1
    # check inputy topological order
    assert len(inputy.get_users()) == 1
    assert inputy.get_users()[0] == new_add1
    # check inputz topological order
    assert len(inputz.get_users()) == 1
    assert inputz.get_users()[0] == new_add2
    # check new_add1 topological order
    assert len(new_add1.get_inputs()) == 2
    assert new_add1.get_inputs()[0] == inputx
    assert new_add1.get_inputs()[1] == inputy
    assert len(new_add1.get_users()) == 1
    assert new_add1.get_users()[0] == new_add2
    # check new_add2 topological order
    assert len(new_add2.get_inputs()) == 2
    assert new_add2.get_inputs()[0] == new_add1
    assert new_add2.get_inputs()[1] == inputz
    assert len(new_add2.get_users()) == 1
    assert new_add2.get_users()[0] == relu
    # check relu topological order
    assert len(relu.get_inputs()) == 1
    assert relu.get_inputs()[0] == new_add2
    # check source code order
    assert getattr(inputz.get_handler(), "_next") == new_add1.get_handler()
    assert getattr(new_add1.get_handler(), "_next") == new_add2.get_handler()
    assert getattr(new_add2.get_handler(), "_next") == relu.get_handler()
    assert getattr(relu.get_handler(), "_prev") == new_add2.get_handler()
    assert getattr(new_add2.get_handler(), "_prev") == new_add1.get_handler()
    assert getattr(new_add1.get_handler(), "_prev") == inputz.get_handler()
    # check arg edge
    assert len(inputx.get_targets()) == 1
    assert len(inputy.get_targets()) == 1
    assert len(new_add1.get_args()) == 2
    assert inputx.get_targets()[0] == new_add1.get_args()[0]
    assert inputy.get_targets()[0] == new_add1.get_args()[1]

    assert len(inputz.get_targets()) == 1
    assert len(new_add1.get_targets()) == 1
    assert len(new_add2.get_args()) == 2
    assert new_add1.get_targets()[0] == new_add2.get_args()[0]
    assert inputz.get_targets()[0] == new_add2.get_args()[1]

    assert len(new_add2.get_targets()) == 1
    assert len(relu.get_args()) == 1
    assert new_add2.get_targets()[0] == relu.get_args()[0]


class TreeNetwork3(Cell):
    def __init__(self):
        super().__init__()
        self.conv1 = Conv2d(16, 16, 1)
        self.conv2 = Conv2d(16, 16, 3)
        self.add1 = AddN()
        self.add2 = AddN()
        self.relu = ReLU()

    def construct(self, x):
        y = self.conv1(x)
        z = self.conv2(x)
        x = self.add1(y, z, x)
        x = self.add2(y, z, x)
        x = self.relu(x)
        return x


def test_one_input_to_multi_pattern_tree_pattern():
    """
    Feature: Python api PatternEngine.
    Description: Construct a multi-to-multi PatternEngine and apply it on a SymbolTree, check SymbolTree after
                 PatternEngine applied.
    Expectation: Success.
    """

    net = TreeNetwork3()
    stree = SymbolTree.create(net)
    conv1 = stree.get_node("conv1")
    conv2 = stree.get_node("conv2")
    add1 = stree.get_node("add1")
    add2 = stree.get_node("add2")
    relu = stree.get_node("relu")
    assert conv1 is not None
    assert conv2 is not None
    assert add1 is not None
    assert add2 is not None
    assert relu is not None
    construct_ast: ast.FunctionDef = getattr(stree.get_handler(), "_root_ast")
    assert len(construct_ast.body) == 6
    assert get_symbol_tree_nodes_count(stree) == 7

    multi_input_pattern = MultiInputPattern()
    multi_input_pattern.apply(stree)

    assert len(construct_ast.body) == 4
    assert get_symbol_tree_nodes_count(stree) == 5
    conv1 = stree.get_node("conv1")
    conv2 = stree.get_node("conv2")
    add1 = stree.get_node("add1")
    add2 = stree.get_node("add2")
    relu = stree.get_node("relu")
    new_add1 = stree.get_node("new_add1")
    new_add2 = stree.get_node("new_add2")
    inputx = stree.get_node("input_x")
    assert conv1 is None
    assert conv2 is None
    assert add1 is None
    assert add2 is None
    assert relu is not None
    assert new_add1 is not None
    assert new_add2 is not None
    assert inputx is not None

    # check inputx topological order
    assert len(inputx.get_users()) == 2
    assert inputx.get_users()[0] == new_add1
    assert inputx.get_users()[1] == new_add2
    # check new_add1 topological order
    assert len(new_add1.get_inputs()) == 2
    assert new_add1.get_inputs()[0] == inputx
    assert new_add1.get_inputs()[1] == inputx
    assert len(new_add1.get_users()) == 1
    assert new_add1.get_users()[0] == new_add2
    # check new_add2 topological order
    assert len(new_add2.get_inputs()) == 2
    assert new_add2.get_inputs()[0] == new_add1
    assert new_add2.get_inputs()[1] == inputx
    assert len(new_add2.get_users()) == 1
    assert new_add2.get_users()[0] == relu
    # check relu topological order
    assert len(relu.get_inputs()) == 1
    assert relu.get_inputs()[0] == new_add2
    # check source code order
    assert getattr(inputx.get_handler(), "_next") == new_add1.get_handler()
    assert getattr(new_add1.get_handler(), "_next") == new_add2.get_handler()
    assert getattr(new_add2.get_handler(), "_next") == relu.get_handler()
    assert getattr(relu.get_handler(), "_prev") == new_add2.get_handler()
    assert getattr(new_add2.get_handler(), "_prev") == new_add1.get_handler()
    assert getattr(new_add1.get_handler(), "_prev") == inputx.get_handler()
    # check arg edge
    assert len(inputx.get_targets()) == 1
    assert len(new_add1.get_args()) == 2
    assert inputx.get_targets()[0] == new_add1.get_args()[0]
    assert inputx.get_targets()[0] == new_add1.get_args()[1]

    assert len(inputx.get_targets()) == 1
    assert len(new_add1.get_targets()) == 1
    assert len(new_add2.get_args()) == 2
    assert new_add1.get_targets()[0] == new_add2.get_args()[0]
    assert inputx.get_targets()[0] == new_add2.get_args()[1]

    assert len(new_add2.get_targets()) == 1
    assert len(relu.get_args()) == 1
    assert new_add2.get_targets()[0] == relu.get_args()[0]
