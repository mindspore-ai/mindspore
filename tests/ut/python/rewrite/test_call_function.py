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

import mindspore.nn as nn
import mindspore.ops as ops
from mindspore.rewrite import SymbolTree, NodeType, Node, ScopedValue
from mindspore import Tensor
import mindspore.common.dtype as mstype # pylint:disable=unused-import
from mindspore.rewrite.node import LocalPrim


def external_func(x):
    x = ops.abs(x)
    return x


class SubSubNet(nn.Cell):
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU()
        self.abs = ops.Abs()

    def construct(self, x):
        x = self.relu(x)
        x = external_func(x)
        x = self.subsubnet_internal_func(x)
        return x

    def subsubnet_internal_func(self, x):
        x = self.abs(x)
        return x


class SubNet(nn.Cell):
    def __init__(self):
        super().__init__()
        self.subsubnet = SubSubNet()
        self.abs = ops.Abs()
        self.relu = nn.ReLU()

    def construct(self, x):
        x = self.relu(x)
        x = external_func(x)
        x = self.subnet_internal_func(x)
        return x

    def subnet_internal_func(self, x):
        x = self.abs(x)
        x = self.subsubnet(x)
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


def test_call_function():
    """
    Feature: Python api get_code of Node of Rewrite.
    Description: Test rewrite CallFunction nodes.
    Expectation: Success.
    """
    net = MyNet()
    stree = SymbolTree.create(net)

    # check CallFunction node of internal function
    internal_func_node = stree.get_node('internal_func')
    assert internal_func_node
    assert internal_func_node.get_node_type() == NodeType.CallFunction
    internal_func_subnodes = internal_func_node.get_handler().nodes()
    assert internal_func_subnodes
    assert len(internal_func_subnodes) == 4
    # check CallFunction node of external function
    external_func_node = stree.get_node('external_func')
    assert external_func_node
    assert external_func_node.get_node_type() == NodeType.CallFunction
    external_func_subnodes = external_func_node.get_handler().nodes()
    assert external_func_subnodes
    assert len(external_func_subnodes) == 3
    # check CallFunction node of subtree's internal function
    subtree = internal_func_node.get_handler().get_node("sub_net").symbol_tree
    subnet_internal_func_node = subtree.get_node('subnet_internal_func')
    assert subnet_internal_func_node
    assert subnet_internal_func_node.get_node_type() == NodeType.CallFunction
    subnet_internal_func_subnodes = subnet_internal_func_node.nodes()
    assert subnet_internal_func_subnodes
    assert len(subnet_internal_func_subnodes) == 4
    # check CallFunction node of subtree's external function
    subnet_external_func_node = subtree.get_node('external_func_1')
    assert subnet_external_func_node
    assert subnet_external_func_node.get_node_type() == NodeType.CallFunction
    subnet_external_func_subnodes = subnet_external_func_node.nodes()
    assert subnet_external_func_subnodes
    assert len(subnet_external_func_subnodes) == 3
    # check CallFunction node of subsubtree's internal function
    subsubtree = subnet_internal_func_node.get_node("subsubnet").symbol_tree
    subsubnet_internal_func_node = subsubtree.get_node('subsubnet_internal_func')
    assert subsubnet_internal_func_node
    assert subsubnet_internal_func_node.get_node_type() == NodeType.CallFunction
    subsubnet_internal_func_subnodes = subsubnet_internal_func_node.nodes()
    assert subsubnet_internal_func_subnodes
    assert len(subsubnet_internal_func_subnodes) == 3
    # check CallFunction node of subsubtree's external function
    subsubtree = subnet_internal_func_node.get_node("subsubnet").symbol_tree
    subsubnet_external_func_node = subsubtree.get_node('external_func_2')
    assert subsubnet_external_func_node
    assert subsubnet_external_func_node.get_node_type() == NodeType.CallFunction
    subsubnet_external_func_subnodes = subsubnet_external_func_node.nodes()
    assert subsubnet_external_func_subnodes
    assert len(subsubnet_external_func_subnodes) == 3

    # insert cell node to CallFunction node of internal function
    new_node = Node.create_call_cell(ops.Abs(), targets=['x'],
                                     args=[ScopedValue.create_naming_value("x")],
                                     name="new_abs")
    internal_func_abs = internal_func_node.get_handler().get_node('abs')
    stree.insert(stree.after(Node(internal_func_abs)), new_node)

    internal_func_subnodes = internal_func_node.get_handler().nodes()
    assert len(internal_func_subnodes) == 5
    assert internal_func_node.get_handler().get_node('new_abs')

    codes = stree.get_code()
    assert codes.count("self.new_abs = obj.new_abs") == 1
    assert codes.count("x = self.new_abs(x)") == 1

    # erase node in CallFunction nodes of internal function
    assert codes.count("abs_var = self.abs(x)") == 1

    internal_func_node.get_handler().get_node("sub_net").set_arg('x', 0)
    stree.erase(Node(internal_func_abs))

    internal_func_subnodes = internal_func_node.get_handler().nodes()
    assert len(internal_func_subnodes) == 4
    assert not internal_func_node.get_handler().get_node('abs')

    codes = stree.get_code()
    assert codes.count("abs_var = self.abs(x)") == 0

    # erase node in CallFunction nodes of external function
    assert codes.count("x = abs_opt(x)") == 1

    external_func_abs = external_func_node.get_handler().get_node("abs_opt")
    stree.erase(Node(external_func_abs))

    external_func_subnodes = external_func_node.get_handler().nodes()
    assert len(external_func_subnodes) == 2
    assert not external_func_node.get_handler().get_node('abs_opt')

    codes = stree.get_code()
    assert codes.count("x = abs_opt(x)") == 0

    # insert cell node to CallFunction node of external function, a TypeError exception is expected.
    new_node = Node.create_call_cell(ops.Abs(), targets=['x'],
                                     args=[ScopedValue.create_naming_value("x")],
                                     name="new_abs")
    external_func_input = external_func_node.get_handler().get_node('input_x_1')
    with pytest.raises(TypeError, match="Cannot insert NodeType.CallPrimitive node '.*' "
                                        "into no-method function '.*'."):
        stree.insert(stree.after(Node(external_func_input)), new_node)


def external_func1(x):
    x = ops.abs(x)
    return x


class SubSubNet1(nn.Cell):
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU()
        self.abs = ops.Abs()

    def construct(self, x):
        x = self.relu(x)
        x = external_func1(x)
        x = self.subsubnet_internal_func(x)
        return x

    def subsubnet_internal_func(self, x):
        x = self.abs(x)
        return x


class SubNet1(nn.Cell):
    def __init__(self):
        super().__init__()
        self.subsubnet = SubSubNet1()
        self.abs = ops.Abs()
        self.relu = nn.ReLU()

    def construct(self, x):
        x = self.relu(x)
        x = external_func1(x)
        x = self.subnet_internal_func(x)
        return x

    def subnet_internal_func(self, x):
        x = self.abs(x)
        x = self.subsubnet(x)
        return x


class MyNet1(nn.Cell):
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU()
        self.sub_net = SubNet1()
        self.abs = ops.Abs()

    def construct(self, x):
        x = self.relu(x)
        x = external_func1(x)
        x = self.internal_func(x)
        return x

    def internal_func(self, x):
        x = self.sub_net(self.abs(x))
        return x

def test_create_call_function():
    """
    Feature: Python Rewrite api.
    Description: Test rewrite create CallFunction nodes.
    Expectation: Success.
    """
    net = MyNet1()
    stree = SymbolTree.create(net)
    # insert to construct function
    new_node = Node.create_call_function(function=ops.abs, targets=['x'],
                                         args=[ScopedValue.create_naming_value('x')])
    stree.insert(stree.after(stree.get_node('relu')), new_node)
    assert len(list(stree.nodes())) == 6
    codes = stree.get_code()
    assert codes.count("import abs\n") == 1, codes # from math_func/gen_ops_def
    assert codes.count("x = abs(x)") == 1
    # insert to class internal function
    internal_func_node = stree.get_node('internal_func')
    internal_abs_node = Node(internal_func_node.get_handler().get_node('abs'))
    new_node = Node.create_call_function(function=ops.abs, targets=['x'],
                                         args=[ScopedValue.create_naming_value('x')])
    stree.insert(stree.after(internal_abs_node), new_node)
    assert len(list(internal_func_node.get_handler().nodes())) == 5
    codes = stree.get_code()
    assert codes.count("import abs\n") == 1 # from math_func/gen_ops_def
    assert codes.count("x = abs(x)") == 2
    # insert to external function
    external_func_node = stree.get_node('external_func1')
    external_abs_node = Node(external_func_node.get_handler().get_node('abs_opt_3'))
    external_abs_nodes = [Node(n) for n in external_func_node.get_handler().nodes() if n.get_instance_type() == ops.abs]
    assert len(external_abs_nodes) == 1, [node.get_name() for node in external_func_node.get_handler().nodes()]
    external_abs_node = external_abs_nodes[0]
    new_node = Node.create_call_function(function=ops.abs, targets=['x'],
                                         args=[ScopedValue.create_naming_value('x')])
    stree.insert(stree.after(external_abs_node), new_node)
    assert len(list(external_func_node.get_handler().nodes())) == 4
    codes = stree.get_code()
    assert codes.count("import abs\n") == 1 # from math_func/gen_ops_def
    assert codes.count("x = abs(x)") == 3
    # insert to sub symbol tree in internal function
    subtree = SymbolTree(internal_func_node.get_handler().get_node("sub_net").symbol_tree)
    new_node = Node.create_call_function(function=ops.abs, targets=['x'],
                                         args=[ScopedValue.create_naming_value('x')])
    subtree.insert(subtree.after(subtree.get_node('relu')), new_node)
    assert len(list(subtree.nodes())) == 6
    codes = stree.get_code()
    assert codes.count("import abs\n") == 1 # from math_func/gen_ops_def
    assert codes.count("x = abs(x)") == 4
    # insert to internal function of sub symbol tree
    subnet_internal_func_node = subtree.get_node('subnet_internal_func')
    subnet_internal_abs_node = Node(subnet_internal_func_node.get_handler().get_node('abs'))
    new_node = Node.create_call_function(function=ops.abs, targets=['x'],
                                         args=[ScopedValue.create_naming_value('x')])
    subtree.insert(subtree.after(subnet_internal_abs_node), new_node)
    assert len(list(subnet_internal_func_node.get_handler().nodes())) == 5
    codes = stree.get_code()
    assert codes.count("import abs\n") == 1 # from math_func/gen_ops_def
    assert codes.count("x = abs(x)") == 5
    # insert to external function of sub sub symbol tree
    subnet_external_func_node = subtree.get_node('external_func1_1')
    subnet_external_abs_nodes = [Node(n) for n in subnet_external_func_node.get_handler().nodes() \
                                 if n.get_instance_type() == ops.abs]
    assert len(subnet_external_abs_nodes) == 1, [node.get_name() for node in \
                                                 subnet_external_func_node.get_handler().nodes()]
    subnet_external_abs_node = subnet_external_abs_nodes[0]
    new_node = Node.create_call_function(function=ops.abs, targets=['x'],
                                         args=[ScopedValue.create_naming_value('x')])
    subtree.insert(stree.after(subnet_external_abs_node), new_node)
    assert len(list(subnet_external_func_node.get_handler().nodes())) == 4
    codes = stree.get_code()
    assert codes.count("import abs\n") == 1 # from math_func/gen_ops_def
    assert codes.count("x = abs(x)") == 6


class TensorAddNet(nn.Cell):
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU()

    def construct(self, x):
        x = self.relu(x)
        return x

def test_create_call_function_tensor():
    """
    Feature: Python Rewrite api.
    Description: Test rewrite create tensor CallFunction nodes.
    Expectation: Success.
    """
    net = TensorAddNet()
    stree = SymbolTree.create(net)
    # insert tensor and add nodes to construct function
    relu_node = stree.get_node("relu")
    tensor_node = Node.create_call_function(function=Tensor, targets=['a'],
                                            args=[ScopedValue.create_variable_value(1.0),
                                                  ScopedValue.create_naming_value("float32", "mstype")])
    add_node = Node.create_call_function(function=ops.add, targets=['x'], args=[relu_node.get_targets()[0],
                                                                                tensor_node.get_targets()[0]])
    stree.insert(stree.after(relu_node), tensor_node)
    stree.insert(stree.after(tensor_node), add_node)
    # code check
    codes = stree.get_code()
    assert codes.count('from mindspore.common.tensor import Tensor') == 1, codes
    assert codes.count('import add\n') == 1, codes # from math_func/gen_ops_def
    assert codes.count('a = Tensor(1.0, mstype.float32)') == 1, codes
    assert codes.count('x = add(x, a)') == 1, codes


# loop function
def loop_func2(x):
    if x.shape:
        return x
    x = loop_func(x)
    return x

def loop_func(x):
    if x.shape:
        return x
    x = loop_func2(x)
    return x


class LoopFuncNet(nn.Cell):
    def construct(self, x):
        x = loop_func(x)
        return x

def test_loop_function():
    """
    Feature: Python Rewrite api.
    Description: Test rewrite parse function with loop.
    Expectation: Success.
    """
    net = LoopFuncNet()
    stree = SymbolTree.create(net)
    # code check
    codes = stree.get_code()
    assert codes.count('def loop_func(x):') == 1
    assert codes.count('def loop_func2(x):') == 1


class AbsNet(nn.Cell):
    def construct(self, x):
        _absolute = ops.Abs()
        x = _absolute(x)
        return x

def test_call_ops_local():
    """
    Feature: Python Rewrite api.
    Description: Test rewrite get node type of local variable.
    Expectation: Success.
    """
    net = AbsNet()
    stree = SymbolTree.create(net)
    node = stree.get_node("_absolute")
    assert node is not None
    assert isinstance(node.get_instance(), ops.Primitive)
    assert isinstance(node.get_instance(), LocalPrim)
    assert issubclass(node.get_instance_type(), ops.Abs)
    assert node.get_node_type() == NodeType.CallPrimitive


def closure_func(x):
    def inner_func(x):
        return x
    x = inner_func(x)
    return x


class ClosureNet(nn.Cell):
    def construct(self, x):
        x = closure_func(x)
        return x

def test_closure_func():
    """
    Feature: Python Rewrite api.
    Description: Test rewrite process closure function.
    Expectation: Success.
    """
    net = ClosureNet()
    stree = SymbolTree.create(net)
    node = stree.get_node("closure_func")
    assert node is not None
    assert node.get_node_type() == NodeType.CallFunction
    assert not node.get_handler().nodes()
    codes = stree.get_code()
    assert codes.count("x = closure_func(x)") == 1, codes
    assert codes.count("import closure_func") == 1, codes
    assert codes.count("def closure_func(x):") == 0, codes
    assert codes.count("def inner_func(x):") == 0, codes
