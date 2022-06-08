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
import os
import ast
import inspect

from mindspore.nn import Cell, Conv2d, BatchNorm2d, ReLU
from mindspore.ops import Add
from mindspore.rewrite import ScopedValue, ValueType, NodeType
from mindspore.rewrite import Node as NodeApi
from mindspore.rewrite.symbol_tree import SymbolTree
from mindspore.rewrite.node import Node
from .utils import get_symbol_tree_nodes_count


class Network(Cell):
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


def create_symbol_tree():
    net = Network()
    source = inspect.getsource(type(net))
    ast_root = ast.parse(source)
    ast_module = ast_root
    assert isinstance(ast_root, ast.Module)
    ast_class = ast_module.body[0]
    assert isinstance(ast_class, ast.ClassDef)
    ast_init_func = ast_class.body[0]
    assert isinstance(ast_init_func, ast.FunctionDef)
    ast_construct_func = ast_class.body[1]
    assert isinstance(ast_construct_func, ast.FunctionDef)
    ast_conv = ast_construct_func.body[0]
    ast_bn = ast_construct_func.body[1]
    ast_relu1 = ast_construct_func.body[2]
    ast_relu2 = ast_construct_func.body[3]
    ast_relu3 = ast_construct_func.body[4]
    ast_return = ast_construct_func.body[5]
    stree = SymbolTree(net, ast_module)
    stree.set_class_ast(ast_class)
    stree.set_init_func_ast(ast_init_func)
    stree.set_ast_root(ast_construct_func)
    stree.append_input_node(ast.arg(arg="x", annotation=None, type_comment=None), "x")
    conv_node = Node.create_call_buildin_op(net.conv, ast_conv, [ScopedValue.create_naming_value("x")],
                                            ScopedValue.create_naming_value("conv", "self"),
                                            [ScopedValue.create_naming_value("x")],
                                            {},
                                            "conv")
    stree.append_origin_field(conv_node)
    bn_node = Node.create_call_buildin_op(net.bn, ast_bn, [ScopedValue.create_naming_value("x")],
                                          ScopedValue.create_naming_value("bn", "self"),
                                          [ScopedValue.create_naming_value("x")], {},
                                          "bn")
    bn_node = stree.append_origin_field(bn_node)
    relu1_node = Node.create_call_buildin_op(net.relu1, ast_relu1, [ScopedValue.create_naming_value("x")],
                                             ScopedValue.create_naming_value("relu1", "self"),
                                             [ScopedValue.create_naming_value("x")],
                                             {}, "relu1")
    relu1_node = stree.append_origin_field(relu1_node)
    relu2_node = Node.create_call_buildin_op(net.relu2, ast_relu2, [ScopedValue.create_naming_value("x")],
                                             ScopedValue.create_naming_value("relu2", "self"),
                                             [ScopedValue.create_naming_value("x")],
                                             {}, "relu2")
    relu2_node = stree.append_origin_field(relu2_node)
    relu3_node = Node.create_call_buildin_op(net.relu3, ast_relu3, [ScopedValue.create_naming_value("x")],
                                             ScopedValue.create_naming_value("relu3", "self"),
                                             [ScopedValue.create_naming_value("x")],
                                             {}, "relu3")
    stree.append_origin_field(relu3_node)
    node_return = Node.create_output_node(ast_return, ["x"])
    stree.append_origin_field(node_return)
    return stree, bn_node, relu1_node, relu2_node


def test_insert_node():
    """
    Feature: Python api insert_node of SymbolTree of Rewrite.
    Description: Call insert_node to insert a node into SymbolTree.
    Expectation: Success.
    """
    stree, _, relu1, relu2 = create_symbol_tree()
    construct_ast: ast.FunctionDef = getattr(stree, "_root_ast")
    providers = getattr(getattr(stree, "_topo_mgr"), "_target_provider")
    consumers = getattr(getattr(stree, "_topo_mgr"), "_target_consumer")
    providers_len = len(providers)
    consumers_len = len(consumers)
    assert get_symbol_tree_nodes_count(stree) == 7
    assert len(construct_ast.body) == 6
    assert len(relu1.get_targets()) == 1
    assert len(relu2.get_normalized_args().values()) == 1
    assert relu1.get_targets()[0] == list(relu2.get_normalized_args().values())[0]
    input1 = 1
    node = Node.create_call_buildin_op(Add(), None, [ScopedValue.create_naming_value('x')],
                                       ScopedValue.create_naming_value('new_conv'),
                                       [ScopedValue.create_naming_value('x'),
                                        ScopedValue.create_variable_value(input1)], {},
                                       'new_conv')
    position = stree.before(relu2)
    node = stree.insert_node(position, node)
    # check nodes size
    assert get_symbol_tree_nodes_count(stree) == 8
    # check args
    assert len(relu2.get_normalized_args().values()) == 1
    assert relu1.get_targets()[0] == list(relu2.get_normalized_args().values())[0]
    assert len(node.get_normalized_args().values()) == 2
    assert list(node.get_normalized_args().values())[0] == ScopedValue.create_naming_value('x')
    assert list(node.get_normalized_args().values())[1].type == ValueType.IntValue
    # check provider
    assert len(providers) == providers_len + 1
    assert len(node.get_targets()) == 1
    assert providers.get(node.get_targets()[0])[0] == node
    assert providers.get(node.get_targets()[0])[1] == 0
    # check consumer
    assert len(consumers) == consumers_len + 1
    assert consumers.get(list(node.get_normalized_args().values())[1]) is not None
    # check inputs
    assert len(relu2.get_inputs()) == 1
    assert relu2.get_inputs()[0] == relu1
    assert len(node.get_inputs()) == 1
    assert node.get_inputs()[0].get_node_type() == NodeType.Input
    # check ast
    node_ast = node.get_ast()
    assert isinstance(node_ast, ast.Assign)
    args = node_ast.value.args
    assert isinstance(args, list)
    assert len(args) == 2
    assert isinstance(args[0], ast.Name)
    assert isinstance(args[1], ast.Constant)
    assert len(construct_ast.body) == 7


def test_insert_node_before_input():
    """
    Feature: Python api insert_node of SymbolTree of Rewrite.
    Description: Call insert_node to insert a node before an input node.
    Expectation: Failure.
    """
    stree, _, _, _ = create_symbol_tree()
    input1 = 1
    node = Node.create_call_buildin_op(Add(), None, [ScopedValue.create_naming_value('x')],
                                       ScopedValue.create_naming_value('new_conv'),
                                       [ScopedValue.create_naming_value('x'),
                                        ScopedValue.create_variable_value(input1)], {},
                                       'new_conv')
    input_node = stree.get_inputs()[0]
    position = stree.before(input_node)
    failed = False
    try:
        stree.insert_node(position, node)
    except RuntimeError:
        failed = True
    assert failed


def test_set_node_arg():
    """
    Feature: Python api set_node_arg of SymbolTree of Rewrite.
    Description: Call set_node_arg to change topological-order of a node.
    Expectation: Success.
    """
    stree, bn, relu1, relu2 = create_symbol_tree()
    assert get_symbol_tree_nodes_count(stree) == 7
    assert len(bn.get_targets()) == 1
    bn_output = bn.get_targets()[0]
    # check bn topological order
    assert len(stree.get_node_users(bn)) == 1
    assert stree.get_node_users(bn)[0][0] == relu1
    # check relu1 topological order
    assert len(stree.get_node_inputs(relu1)) == 1
    assert stree.get_node_inputs(relu1)[0] == bn
    assert len(stree.get_node_users(relu1)) == 1
    assert stree.get_node_users(relu1)[0][0] == relu2
    # check relu2 topological order
    assert len(stree.get_node_inputs(relu2)) == 1
    assert stree.get_node_inputs(relu2)[0] == relu1
    # check relu1 and relu2 edge
    assert len(relu1.get_targets()) == 1
    assert len(relu2.get_normalized_args().values()) == 1
    assert relu1.get_targets()[0] == list(relu2.get_normalized_args().values())[0]

    stree.set_node_arg(relu2, 0, bn_output)
    # check bn topological order
    assert len(stree.get_node_users(bn)) == 2
    assert stree.get_node_users(bn)[0][0] == relu1
    assert stree.get_node_users(bn)[1][0] == relu2
    # check relu1 topological order
    assert len(stree.get_node_inputs(relu1)) == 1
    assert stree.get_node_inputs(relu1)[0] == bn
    assert len(stree.get_node_users(relu1)) == 0
    # check relu2 topological order
    assert len(stree.get_node_inputs(relu2)) == 1
    assert stree.get_node_inputs(relu2)[0] == bn
    # check bn and relu2 edge
    assert len(relu1.get_targets()) == 1
    assert len(relu2.get_normalized_args().values()) == 1
    assert bn_output == list(relu2.get_normalized_args().values())[0]
    # check ast
    node_ast = relu2.get_ast()
    assert isinstance(node_ast, ast.Assign)
    args = node_ast.value.args
    assert isinstance(args, list)
    assert len(args) == 1
    assert isinstance(args[0], ast.Name)
    assert args[0].id == bn_output.value


def test_set_node_arg_by_node():
    """
    Feature: Python api set_node_arg_by_node of SymbolTree of Rewrite.
    Description: Call set_node_arg_by_node to change topological-order of a node.
    Expectation: Success.
    """
    stree, bn, relu1, relu2 = create_symbol_tree()
    assert get_symbol_tree_nodes_count(stree) == 7
    assert len(bn.get_targets()) == 1
    bn_output = bn.get_targets()[0]
    # check bn topological order
    assert len(stree.get_node_users(bn)) == 1
    assert stree.get_node_users(bn)[0][0] == relu1
    # check relu1 topological order
    assert len(stree.get_node_inputs(relu1)) == 1
    assert stree.get_node_inputs(relu1)[0] == bn
    assert len(stree.get_node_users(relu1)) == 1
    assert stree.get_node_users(relu1)[0][0] == relu2
    # check relu2 topological order
    assert len(stree.get_node_inputs(relu2)) == 1
    assert stree.get_node_inputs(relu2)[0] == relu1
    # check relu1 and relu2 edge
    assert len(relu1.get_targets()) == 1
    assert len(relu2.get_normalized_args().values()) == 1
    assert relu1.get_targets()[0] == list(relu2.get_normalized_args().values())[0]

    stree.set_node_arg_by_node(relu2, 0, bn)
    # check bn topological order
    assert len(stree.get_node_users(bn)) == 2
    assert stree.get_node_users(bn)[0][0] == relu1
    assert stree.get_node_users(bn)[1][0] == relu2
    # check relu1 topological order
    assert len(stree.get_node_inputs(relu1)) == 1
    assert stree.get_node_inputs(relu1)[0] == bn
    assert len(stree.get_node_users(relu1)) == 0
    # check relu2 topological order
    assert len(stree.get_node_inputs(relu2)) == 1
    assert stree.get_node_inputs(relu2)[0] == bn
    # check bn and relu2 edge
    assert len(relu1.get_targets()) == 1
    assert len(relu2.get_normalized_args().values()) == 1
    assert bn_output == list(relu2.get_normalized_args().values())[0]
    # check ast
    node_ast = relu2.get_ast()
    assert isinstance(node_ast, ast.Assign)
    args = node_ast.value.args
    assert isinstance(args, list)
    assert len(args) == 1
    assert isinstance(args[0], ast.Name)
    assert args[0].id == bn_output.value


def test_erase_succeed():
    """
    Feature: Python api erase_node of SymbolTree of Rewrite.
    Description: Call erase_node to erase a node from SymbolTree.
    Expectation: Success.
    """
    stree, bn, relu1, relu2 = create_symbol_tree()
    construct_ast: ast.FunctionDef = getattr(stree, "_root_ast")
    providers = getattr(getattr(stree, "_topo_mgr"), "_target_provider")
    providers_len = len(providers)
    assert get_symbol_tree_nodes_count(stree) == 7
    assert len(construct_ast.body) == 6

    stree.set_node_arg_by_node(relu2, 0, bn)
    stree.erase_node(relu1)

    assert get_symbol_tree_nodes_count(stree) == 6
    assert len(providers) == providers_len - 1
    assert len(construct_ast.body) == 5


def test_erase_failed():
    """
    Feature: Python api erase_node of SymbolTree of Rewrite.
    Description: Call erase_node to erase a node from SymbolTree which is not isolated.
    Expectation: Failure.
    """
    stree, _, relu1, _ = create_symbol_tree()
    catched_error = False
    try:
        stree.erase_node(relu1)
    except RuntimeError:
        catched_error = True
    assert catched_error


def test_replace_one_to_one():
    """
    Feature: Python api replace of SymbolTree of Rewrite.
    Description: Call replace to replace an origin node to a new node.
    Expectation: Success.
    """
    stree, bn, relu1, relu2 = create_symbol_tree()
    construct_ast: ast.FunctionDef = getattr(stree, "_root_ast")
    assert len(construct_ast.body) == 6
    assert get_symbol_tree_nodes_count(stree) == 7

    new_conv = Conv2d(16, 16, 5)
    new_conv_node = NodeApi.create_call_cell(new_conv, [ScopedValue.create_naming_value("new_conv")],
                                             bn.get_targets()).get_handler()
    new_conv_node = stree.replace(relu1, [new_conv_node])
    assert get_symbol_tree_nodes_count(stree) == 7
    # check ast
    assert len(construct_ast.body) == 6
    node_ast: ast.Assign = construct_ast.body[2]
    func_ast: ast.Attribute = node_ast.value.func
    assert func_ast.attr == new_conv_node.get_name()
    # check bn topological order
    assert len(stree.get_node_users(bn)) == 1
    assert stree.get_node_users(bn)[0][0] == new_conv_node
    # check new_conv_node topological order
    assert len(stree.get_node_inputs(new_conv_node)) == 1
    assert stree.get_node_inputs(new_conv_node)[0] == bn
    assert len(stree.get_node_users(new_conv_node)) == 1
    assert stree.get_node_users(new_conv_node)[0][0] == relu2
    # check relu2 topological order
    assert len(stree.get_node_inputs(relu2)) == 1
    assert stree.get_node_inputs(relu2)[0] == new_conv_node
    # check arg edge
    assert len(bn.get_targets()) == 1
    assert len(new_conv_node.get_normalized_args().values()) == 1
    assert bn.get_targets()[0] == list(new_conv_node.get_normalized_args().values())[0]
    assert len(new_conv_node.get_targets()) == 1
    assert len(relu2.get_normalized_args().values()) == 1
    assert new_conv_node.get_targets()[0] == list(relu2.get_normalized_args().values())[0]


def test_replace_one_to_one_with_same_arg_and_target():
    """
    Feature: Python api replace of SymbolTree of Rewrite.
    Description: Call replace to replace an origin node to a new node whose arg and target are same.
    Expectation: Success.
    """
    stree, _, relu1, _ = create_symbol_tree()
    construct_ast: ast.FunctionDef = getattr(stree, "_root_ast")
    assert len(construct_ast.body) == 6
    assert get_symbol_tree_nodes_count(stree) == 7

    new_conv = Conv2d(16, 16, 5)
    new_conv_node = NodeApi.create_call_cell(new_conv, [ScopedValue.create_naming_value("x_2")],
                                             [ScopedValue.create_naming_value("x_2")], name="new_conv").get_handler()
    stree.replace(relu1, [new_conv_node])
    assert new_conv_node.get_args()[0].value == "x_2"
    assert get_symbol_tree_nodes_count(stree) == 7
    assert stree.get_node("new_conv")


def test_replace_one_to_multi():
    """
    Feature: Python api replace of SymbolTree of Rewrite.
    Description: Call replace to replace an origin node to a new node-tree.
    Expectation: Success.
    """
    stree, bn, relu1, relu2 = create_symbol_tree()
    construct_ast: ast.FunctionDef = getattr(stree, "_root_ast")
    assert len(construct_ast.body) == 6
    assert get_symbol_tree_nodes_count(stree) == 7

    new_conv_node = NodeApi.create_call_cell(Conv2d(16, 16, 5), [ScopedValue.create_naming_value("new_conv")],
                                             bn.get_targets()).get_handler()
    new_relu_node = NodeApi.create_call_cell(ReLU(), [ScopedValue.create_naming_value("new_relu")],
                                             new_conv_node.get_targets()).get_handler()
    new_relu_node = stree.replace(relu1, [new_relu_node, new_conv_node])
    new_conv_node = new_relu_node.get_inputs()[0]

    assert get_symbol_tree_nodes_count(stree) == 8
    # check ast
    assert len(construct_ast.body) == 7
    new_conv_ast: ast.Assign = construct_ast.body[2]
    new_conv_func_ast: ast.Attribute = new_conv_ast.value.func
    assert new_conv_func_ast.attr == new_conv_node.get_name()
    new_relu_ast: ast.Assign = construct_ast.body[3]
    new_relu_func_ast: ast.Attribute = new_relu_ast.value.func
    assert new_relu_func_ast.attr == new_relu_node.get_name()
    # check bn topological order
    assert len(stree.get_node_users(bn)) == 1
    assert stree.get_node_users(bn)[0][0] == new_conv_node
    # check new_conv_node topological order
    assert len(stree.get_node_inputs(new_conv_node)) == 1
    assert stree.get_node_inputs(new_conv_node)[0] == bn
    assert len(stree.get_node_users(new_conv_node)) == 1
    assert stree.get_node_users(new_conv_node)[0][0] == new_relu_node
    # check new_relu_node topological order
    assert len(stree.get_node_inputs(new_relu_node)) == 1
    assert stree.get_node_inputs(new_relu_node)[0] == new_conv_node
    assert len(stree.get_node_users(new_relu_node)) == 1
    assert stree.get_node_users(new_relu_node)[0][0] == relu2
    # check relu2 topological order
    assert len(stree.get_node_inputs(relu2)) == 1
    assert stree.get_node_inputs(relu2)[0] == new_relu_node
    # check arg edge
    assert len(bn.get_targets()) == 1
    assert len(new_conv_node.get_normalized_args().values()) == 1
    assert bn.get_targets()[0] == list(new_conv_node.get_normalized_args().values())[0]

    assert len(new_conv_node.get_targets()) == 1
    assert len(new_relu_node.get_normalized_args().values()) == 1
    assert new_conv_node.get_targets()[0] == list(new_relu_node.get_normalized_args().values())[0]

    assert len(new_relu_node.get_targets()) == 1
    assert len(relu2.get_normalized_args().values()) == 1
    assert new_relu_node.get_targets()[0] == list(relu2.get_normalized_args().values())[0]


def test_set_saved_file_name():
    """
    Feature: Python api set_saved_file_name and get_saved_file_name of SymbolTree of Rewrite.
    Description: Call set_saved_file_name to set the filename used to save the network.
                 Call get_saved_file_name to get the filename used to save the network.
    Expectation: Success.
    """
    stree, _, _, _ = create_symbol_tree()

    stree.set_saved_file_name("new_network.py")
    new_file_name = stree.get_saved_file_name()
    assert new_file_name == "new_network.py"

    stree.set_saved_file_name("new_network_01")
    new_file_name = stree.get_saved_file_name()
    assert new_file_name == "new_network_01.py"


def test_save_network_to_file():
    """
    Feature: Python api save_network_to_file of SymbolTree of Rewrite.
    Description: Call save_network_to_file to save the network to a file.
    Expectation: Success.
    """
    stree, bn, relu1, relu2 = create_symbol_tree()
    stree.set_node_arg_by_node(relu2, 0, bn)
    stree.erase_node(relu1)

    stree.set_saved_file_name("new_network.py")
    stree.save_network_to_file()
    assert os.path.exists("./new_network.py")

    os.system("rm -f new_network.py")
