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

from mindspore.nn import Cell
from mindspore import nn, ops
from mindspore.rewrite import ScopedValue
from mindspore.rewrite import SymbolTree as SymbolTreeApi
from mindspore.rewrite import Node as NodeApi
from mindspore.rewrite.node import Node


class FakeCell(Cell):
    def construct(self, input1, input2, cool_boy=None):
        return input1 + input2 + cool_boy


class FakeCell2(Cell):
    def construct(self, a, b, d, e, *args, f=6, **kwargs):
        return a + b + d + e + sum(args) + f + sum(kwargs.values())


class FakeCell3(Cell):
    def construct(self, a, b, *args, f=6, h=7, **kwargs):
        return a + b + f + h + sum(args) + sum(kwargs.values())


def test_create_by_cell():
    """
    Feature: Python api create_call_buildin_op of Node of Rewrite.
    Description: Call create_call_buildin_op to create a CallCell node.
    Expectation: Success.
    """
    node = Node.create_call_buildin_op(FakeCell(), None, [ScopedValue.create_naming_value('x')],
                                       ScopedValue.create_naming_value('new_conv'),
                                       [ScopedValue.create_naming_value('x'), ScopedValue.create_variable_value(1)],
                                       {"cool_boy": ScopedValue.create_naming_value('Naroto')}, 'new_conv')
    assert node._args_num == 2
    assert node._kwargs_num == 1
    assert node._normalized_args_keys == ["input1", "input2", "cool_boy"]

    assert node._normalized_args == {
        "input1": ScopedValue.create_naming_value('x'),
        "input2": ScopedValue.create_variable_value(1),
        "cool_boy": ScopedValue.create_naming_value('Naroto')
    }

    node.update_ast_node()
    ast_node: ast.Assign = node.get_ast()
    assign_value: ast.Call = ast_node.value
    args_ast = assign_value.args
    keywords_ast = assign_value.keywords
    assert len(args_ast) == 2
    assert len(keywords_ast) == 1
    assert keywords_ast[0].arg == "cool_boy"
    assert isinstance(args_ast[0], ast.Name)
    assert args_ast[0].id == "x"
    assert isinstance(args_ast[1], ast.Constant)
    assert args_ast[1].value == 1
    keyword_value_3 = keywords_ast[0].value
    assert isinstance(keyword_value_3, ast.Name)
    assert keyword_value_3.id == "Naroto"

    node.set_arg(ScopedValue.create_variable_value(2), 1)
    assert isinstance(node.get_normalized_args().get("input2"), ScopedValue)
    assert node.get_normalized_args().get("input2").value == 2
    ast_node: ast.Assign = node.get_ast()
    assign_value: ast.Call = ast_node.value
    args_ast = assign_value.args
    assert args_ast[1].value == 2

    args = node.get_args()
    assert args == [ScopedValue.create_naming_value('x'), ScopedValue.create_variable_value(2)]
    kwargs = node.get_kwargs()
    assert kwargs == {"cool_boy": ScopedValue.create_naming_value('Naroto')}


def test_create_by_cell2():
    """
    Feature: Python api create_call_buildin_op of Node of Rewrite.
    Description: Call create_call_buildin_op to create a CallCell node.
    Expectation: Success.
    """
    node = Node.create_call_buildin_op(FakeCell2(), None, [ScopedValue.create_naming_value('x')],
                                       ScopedValue.create_naming_value('new_conv'),
                                       [ScopedValue.create_naming_value('x'), ScopedValue.create_naming_value("x"),
                                        ScopedValue.create_naming_value('x'), ScopedValue.create_naming_value("x"),
                                        ScopedValue.create_naming_value('x'), ScopedValue.create_naming_value("x")],
                                       {"cool_boy": ScopedValue.create_naming_value('Naroto')}, 'new_conv')
    assert node.get_normalized_args() == {
        "a": ScopedValue.create_naming_value('x'),
        "b": ScopedValue.create_naming_value('x'),
        "d": ScopedValue.create_naming_value('x'),
        "e": ScopedValue.create_naming_value('x'),
        "args_4": ScopedValue.create_naming_value('x'),
        "args_5": ScopedValue.create_naming_value('x'),
        "cool_boy": ScopedValue.create_naming_value('Naroto'),
    }


def test_create_by_cell3():
    """
    Feature: Python api create_call_buildin_op of Node of Rewrite.
    Description: Call create_call_buildin_op to create a CallCell node.
    Expectation: Success.
    """
    node = Node.create_call_buildin_op(FakeCell3(), None, [ScopedValue.create_naming_value('x')],
                                       ScopedValue.create_naming_value('new_conv'),
                                       [ScopedValue.create_naming_value('x'), ScopedValue.create_naming_value("x"),
                                        ScopedValue.create_naming_value('x'), ScopedValue.create_naming_value("x")],
                                       {"h": ScopedValue.create_variable_value(1),
                                        "f": ScopedValue.create_variable_value(2),
                                        "cool_boy": ScopedValue.create_naming_value('Naroto')}, 'new_conv')
    assert node.get_normalized_args() == {
        "a": ScopedValue.create_naming_value('x'),
        "b": ScopedValue.create_naming_value('x'),
        "args_2": ScopedValue.create_naming_value('x'),
        "args_3": ScopedValue.create_naming_value('x'),
        "f": ScopedValue.create_variable_value(2),
        "h": ScopedValue.create_variable_value(1),
        "cool_boy": ScopedValue.create_naming_value('Naroto'),
    }


def test_create_by_cell4():
    """
    Feature: Python api create_call_buildin_op of Node of Rewrite.
    Description: Call create_call_buildin_op to create a CallCell node.
    Expectation: Success.
    """
    node = Node.create_call_buildin_op(FakeCell3(), None, [ScopedValue.create_naming_value('x')],
                                       ScopedValue.create_naming_value('new_conv'),
                                       [ScopedValue.create_naming_value('x'), ScopedValue.create_naming_value("x"),
                                        ScopedValue.create_naming_value('x'), ScopedValue.create_naming_value("x")],
                                       {"h": ScopedValue.create_variable_value([1]),
                                        "f": ScopedValue.create_variable_value((2,)),
                                        "cool_boy": ScopedValue.create_naming_value('Naroto')}, 'new_conv')
    assert node.get_normalized_args() == {
        "a": ScopedValue.create_naming_value('x'),
        "b": ScopedValue.create_naming_value('x'),
        "args_2": ScopedValue.create_naming_value('x'),
        "args_3": ScopedValue.create_naming_value('x'),
        "f": ScopedValue.create_variable_value((2,)),
        "h": ScopedValue.create_variable_value([1]),
        "cool_boy": ScopedValue.create_naming_value('Naroto'),
    }


def test_create_by_cell5():
    """
    Feature: Python api create_call_buildin_op of Node of Rewrite.
    Description: Call create_call_buildin_op to create a CallCell node.
    Expectation: Success.
    """
    node = Node.create_call_buildin_op(FakeCell3(), None, [ScopedValue.create_naming_value('x')],
                                       ScopedValue.create_naming_value('new_conv'),
                                       [ScopedValue.create_variable_value((4,)), ScopedValue.create_variable_value(5),
                                        ScopedValue.create_variable_value([5]), ScopedValue.create_naming_value("x")],
                                       {"h": ScopedValue.create_variable_value(1),
                                        "f": ScopedValue.create_variable_value(2),
                                        "cool_boy": ScopedValue.create_naming_value('Naroto')}, 'new_conv')
    assert node.get_normalized_args() == {
        "a": ScopedValue.create_variable_value((4,)),
        "b": ScopedValue.create_variable_value(5),
        "args_2": ScopedValue.create_variable_value([5]),
        "args_3": ScopedValue.create_naming_value('x'),
        "f": ScopedValue.create_variable_value(2),
        "h": ScopedValue.create_variable_value(1),
        "cool_boy": ScopedValue.create_naming_value('Naroto'),
    }


def external_func(x):
    x = ops.abs(x)
    return x


class SubNet(nn.Cell):
    def __init__(self):
        super().__init__()
        self.abs = ops.Abs()
        self.relu = nn.ReLU()
        self.container = nn.SequentialCell(nn.ReLU())

    def construct(self, x):
        x = self.relu(x)
        x = external_func(x)
        x = self.subnet_internal_func(x)
        return x

    def subnet_internal_func(self, x):
        x = self.abs(x)
        x = self.subsubnet(x)
        x = self.container(x)
        return x


class MyNet(nn.Cell):
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU()
        self.sub_net = SubNet()
        self.abs = ops.Abs()

    def construct(self, x):
        x = self.relu(x)
        x = external_func(x)
        x = self.internal_func(x)
        return x

    def internal_func(self, x):
        x = self.sub_net(self.abs(x))
        return x


def test_get_symbol_tree():
    """
    Feature: Python api get_symbol_tree of Node of Rewrite.
    Description: Call get_symbol_tree to get symbol tree of node.
    Expectation: Success.
    """
    net = MyNet()
    stree = SymbolTreeApi.create(net)
    for node in stree.nodes():
        assert node.get_symbol_tree().get_handler() == stree.get_handler()
    internal_func_node = stree.get_node('internal_func')
    for node in internal_func_node.get_handler().nodes():
        assert NodeApi(node).get_symbol_tree().get_handler() == stree.get_handler()
    subtree = SymbolTreeApi(stree.get_node("sub_net").get_handler().symbol_tree)
    for node in subtree.nodes():
        assert node.get_symbol_tree().get_handler() == subtree.get_handler()
    subnet_internal_func_node = subtree.get_node('subnet_internal_func')
    for node in subnet_internal_func_node.get_handler().nodes():
        assert NodeApi(node).get_symbol_tree().get_handler() == subtree.get_handler()
    subnet_cell_container = subtree.get_node('container')
    for node in subnet_cell_container.get_handler().nodes():
        assert NodeApi(node).get_symbol_tree().get_handler() == subtree.get_handler()
