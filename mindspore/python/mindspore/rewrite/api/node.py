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
"""Rewrite module api: Node."""

from typing import Union, Optional, List, Dict
from types import FunctionType

from mindspore.nn import Cell
from mindspore.ops.primitive import Primitive
from mindspore import _checkparam as Validator
from ..node.node import Node as NodeImpl
from ..symbol_tree import SymbolTree as SymbolTreeImpl
from .node_type import NodeType
from .scoped_value import ScopedValue


class Node:
    """
    A node is a data structure that expresses source code statements in a network.

    Each node usually corresponds to a statement in expanded forward evaluation process.

    Nodes can express a ``Cell`` call statement, a ``Primitive`` call statement, an arithmetic operation statement, a
    return statements, etc. of the forward calculation process.

    Args:
        node (NodeImpl): A handler of `NodeImpl`. It is recommended to call the specific methods in Node to create
            a Node, such as 'create_call_cell', rather than calling the Node's constructor directly.
            Don't care what `NodeImpl` is, just treat it as a handle.
    """

    def __init__(self, node: NodeImpl):
        self._node = node


    def __eq__(self, other: 'Node'):
        if not isinstance(other, Node):
            return False
        return self._node == other._node

    @staticmethod
    def create_call_cell(cell: Cell, targets: List[Union[ScopedValue, str]], args: List[ScopedValue] = None,
                         kwargs: Dict[str, ScopedValue] = None, name: str = "", is_sub_net: bool = False) -> 'Node':
        """
        Create a node. Only support create from a `Cell` now.

        A node is corresponding to source code like:

        .. code-block::

            `targets` = self.`name`(*`args`, **`kwargs`)

        Args:
            cell (Cell): Cell-operator of this forward-layer.
            targets (List[Union[ScopedValue, str]]): Indicate output names. Used as targets of an assign statement in
                source code.
            args (List[ScopedValue]): Indicate input names. Used as args of a call expression of an assign statement in
                source code. Default: ``None`` , which indicates the `cell` has no args inputs.
            kwargs (Dict[str, ScopedValue]): Type of key must be `str` and type of value must be `ScopedValue`.
                Indicate keyword input names. Used as kwargs of a call expression of an assign statement in source
                code. Default: ``None`` , which indicates the `cell` has no kwargs inputs.
            name (str): Indicate the name of node. Used as field name in source code. Default is None. Rewrite will
                generate name from `cell` when name is None. Rewrite will check and ensure the uniqueness of `name`
                while node being inserted. Default: ``""`` .
            is_sub_net (bool): Indicate that is `cell` a network. If `is_sub_net` is true, Rewrite will try to parse
                the `cell` to a TreeNode, otherwise the `cell` is parsed to a CallCell node. Default: ``False`` .

        Returns:
            An instance of `Node`.

        Raises:
            TypeError: If `cell` is not a `Cell`.
            TypeError: If `targets` is not `list`.
            TypeError: If the type of `targets` is not in `[ScopedValue, str]`.
            TypeError: If arg in `args` is not a `ScopedValue`.
            TypeError: If key of `kwarg` is not a str or value of kwarg in `kwargs` is not a `ScopedValue`.

        Examples:
            >>> from mindspore.rewrite import SymbolTree, ScopedValue
            >>> import mindspore.nn as nn
            >>> # Define the network structure of LeNet5. Refer to
            >>> # https://gitee.com/mindspore/docs/blob/master/docs/mindspore/code/lenet.py
            >>> net = LeNet5()
            >>> stree = SymbolTree.create(net)
            >>> node = stree.get_node("conv1")
            >>> position = stree.after(node)
            >>> new_node = node.create_call_cell(cell=nn.ReLU(), targets=['x'],
            ...                                  args=[ScopedValue.create_naming_value('x')], name='new_relu')
            >>> stree.insert(position, new_node)
            >>> print(type(new_node))
            <class 'mindspore.rewrite.api.node.Node'>
        """
        Validator.check_value_type("cell", cell, [Cell, Primitive], "Node")
        Validator.check_element_type_of_iterable("targets", targets, [ScopedValue, str], "Node")
        Validator.check_value_type("name", name, [str], "Node")
        Validator.check_value_type("is_sub_net", is_sub_net, [bool], "Node")
        if args is not None:
            Validator.check_element_type_of_iterable("args", args, [ScopedValue], "Node")
        if kwargs is not None:
            Validator.check_element_type_of_dict("kwargs", kwargs, [str], [ScopedValue], "Node")
        return Node(NodeImpl.create_call_op(cell, None, targets, args, kwargs, name, is_sub_net))

    @staticmethod
    def create_call_function(function: FunctionType, targets: List[Union[ScopedValue, str]],
                             args: List[ScopedValue] = None, kwargs: Dict[str, ScopedValue] = None) -> 'Node':
        """
        Create a node that corresponds to a function call. The `function` object is saved into network, and used via
        getting object from `self.` .

        Args:
            function (FunctionType): The function to be called.
            targets (List[Union[ScopedValue, str]]): indicates output names. Used as targets of an assign statement in
                source code.
            args (List[ScopedValue]): Indicate input names. Used as args of a call expression of an assign statement in
                source code. Default: ``None`` , which indicates the `function` has no args inputs.
            kwargs (Dict[str, ScopedValue]): Type of key must be `str` and type of value must be `ScopedValue`.
                Indicate keyword input names. Used as kwargs of a call expression of an assign statement in source
                code. Default: ``None`` , which indicates the `function` has no kwargs inputs.

        Returns:
            An instance of `Node`.

        Raises:
            TypeError: If `function` is not a `FunctionType`.
            TypeError: If `targets` is not `list`.
            TypeError: If the type of `targets` is not in `[ScopedValue, str]`.
            TypeError: If arg in `args` is not a `ScopedValue`.
            TypeError: If key of `kwarg` is not a str or value of kwarg in `kwargs` is not a `ScopedValue`.

        Examples:
            >>> from mindspore.rewrite import SymbolTree, ScopedValue
            >>> import mindspore.nn as nn
            >>> import mindspore.ops as ops
            >>> # Define the network structure of LeNet5. Refer to
            >>> # https://gitee.com/mindspore/docs/blob/master/docs/mindspore/code/lenet.py
            >>> net = LeNet5()
            >>> stree = SymbolTree.create(net)
            >>> node = stree.get_node("conv1")
            >>> position = stree.after(node)
            >>> new_node = node.create_call_function(function=ops.abs, targets=['x'],
            ...                                      args=[ScopedValue.create_naming_value('x')])
            >>> stree.insert(position, new_node)
            >>> print(new_node.get_node_type())
            NodeType.CallFunction
        """
        Validator.check_value_type("function", function, [FunctionType, type], "create_call_function")
        Validator.check_element_type_of_iterable("targets", targets, [ScopedValue, str], "create_call_function")
        if args is not None:
            Validator.check_element_type_of_iterable("args", args, [ScopedValue], "create_call_function")
        if kwargs is not None:
            Validator.check_element_type_of_dict("kwargs", kwargs, [str], [ScopedValue], "create_call_function")
        return Node(NodeImpl._create_call_function(function, targets, args, kwargs))

    @staticmethod
    def create_input(param_name: str, default: Optional[ScopedValue] = None) -> 'Node':
    # pylint: disable=missing-function-docstring
        Validator.check_value_type("param_name", param_name, [str], "Node")
        if default is not None:
            Validator.check_value_type("default", default, [ScopedValue], "Node")
        return Node(NodeImpl.create_input_node(None, param_name, default, name=f"input_{param_name}"))

    def get_handler(self) -> NodeImpl:
        return self._node

    def get_inputs(self) -> ['Node']:
        """
        Gets a list of nodes whose output values are used as input values for the current node.

        Returns:
            A list of nodes.

        Examples:
            >>> from mindspore.rewrite import SymbolTree
            >>> # Define the network structure of LeNet5. Refer to
            >>> # https://gitee.com/mindspore/docs/blob/master/docs/mindspore/code/lenet.py
            >>> net = LeNet5()
            >>> stree = SymbolTree.create(net)
            >>> node = stree.get_node("conv2")
            >>> inputs = node.get_inputs()
            >>> print([input.get_name() for input in inputs])
            ['max_pool2d']
        """
        return [Node(node_impl) for node_impl in self._node.get_inputs()]

    def get_users(self) -> ['Node']:
        """
        Get a list of nodes that use the output of the current node as input.

        Returns:
            A list of nodes.

        Examples:
            >>> from mindspore.rewrite import SymbolTree
            >>> # Define the network structure of LeNet5. Refer to
            >>> # https://gitee.com/mindspore/docs/blob/master/docs/mindspore/code/lenet.py
            >>> net = LeNet5()
            >>> stree = SymbolTree.create(net)
            >>> node = stree.get_node("conv1")
            >>> users = node.get_users()
            >>> print([user.get_name() for user in users])
            ['relu']
        """
        belong_symbol_tree: SymbolTreeImpl = self._node.get_belong_symbol_tree()
        if belong_symbol_tree is None:
            return []
        unique_results = []
        for node_user in belong_symbol_tree.get_node_users(self._node):
            node = node_user[0]
            if node not in unique_results:
                unique_results.append(node)
        return [Node(node_impl) for node_impl in unique_results]

    def set_arg(self, index: int, arg: Union[ScopedValue, str]):
        """
        Set argument of current node.

        Args:
            index (int): Indicate which input being modified.
            arg (Union[ScopedValue, str]): New argument to been set.

        Raises:
            TypeError: If `index` is not a `int` number.
            TypeError: If the type of `arg` is not in [`ScopedValue`, `str`].

        Examples:
            >>> from mindspore.rewrite import SymbolTree
            >>> # Define the network structure of LeNet5. Refer to
            >>> # https://gitee.com/mindspore/docs/blob/master/docs/mindspore/code/lenet.py
            >>> net = LeNet5()
            >>> stree = SymbolTree.create(net)
            >>> node = stree.get_node("relu_3")
            >>> node.set_arg(0, "fc1")
            >>> print(node.get_args())
            [fc1]
        """
        Validator.check_value_type("index", index, [int], "Node")
        Validator.check_value_type("arg", arg, [ScopedValue, str], "Node")
        belong_symbol_tree: SymbolTreeImpl = self._node.get_belong_symbol_tree()
        if belong_symbol_tree is None:
            self._node.set_arg(arg, index)
        else:
            belong_symbol_tree.set_node_arg(self._node, index, arg)

    def set_arg_by_node(self, arg_idx: int, src_node: 'Node', out_idx: Optional[int] = None):
        """
        Set argument of current node by another Node.

        Args:
            arg_idx (int): Indicate which input being modified.
            src_node (Node): A `Node` as new input. Can be a node or name of node.
            out_idx (int, optional): Indicate which output of `src_node` as new input of current node.
                Default: ``None`` ,
                which means use first output of `src_node` as new input.

        Raises:
            RuntimeError: If `src_node` is not belong to current `SymbolTree`.
            TypeError: If `arg_idx` is not a `int` number.
            ValueError: If `arg_idx` is out of range.
            TypeError: If `src_node` is not a `Node` instance.
            TypeError: If `out_idx` is not a `int` number.
            ValueError: If `out_idx` is out of range.
            ValueError: If `src_node` has multi-outputs while `out_idx` is None or `out_idx` is not offered.

        Examples:
            >>> from mindspore.rewrite import SymbolTree
            >>> # Define the network structure of LeNet5. Refer to
            >>> # https://gitee.com/mindspore/docs/blob/master/docs/mindspore/code/lenet.py
            >>> net = LeNet5()
            >>> stree = SymbolTree.create(net)
            >>> src_node = stree.get_node("fc1")
            >>> dst_node = stree.get_node("relu_3")
            >>> dst_node.set_arg_by_node(0, src_node, 0)
            >>> print(dst_node.get_args())
            [fc1]
        """
        Validator.check_value_type("arg_idx", arg_idx, [int], "Node")
        Validator.check_value_type("src_node", src_node, [Node], "Node")
        if out_idx is not None:
            Validator.check_value_type("out_idx", out_idx, [int], "Node")
        belong_symbol_tree: SymbolTreeImpl = self._node.get_belong_symbol_tree()
        if belong_symbol_tree is None:
            self._node.set_arg_by_node(arg_idx, src_node._node, out_idx)
        else:
            belong_symbol_tree.set_node_arg_by_node(self._node, arg_idx, src_node.get_handler(), out_idx)

    def get_targets(self) -> [ScopedValue]:
        """
        Gets a list of output values for the current node.

        Returns:
            A list of outputs of type ``ScopedValue`` .
        """
        return self._node.get_targets()

    def get_name(self) -> str:
        """
        Get the name of current node.

        When node has been inserted into `SymbolTree`, the name of node should be unique in `SymbolTree`.

        Returns:
            A string as name of node.

        Examples:
            >>> from mindspore.rewrite import SymbolTree
            >>> # Define the network structure of LeNet5. Refer to
            >>> # https://gitee.com/mindspore/docs/blob/master/docs/mindspore/code/lenet.py
            >>> net = LeNet5()
            >>> stree = SymbolTree.create(net)
            >>> node = stree.get_node("conv1")
            >>> name = node.get_name()
            >>> print(name)
            conv1
        """
        return self._node.get_name()

    def get_node_type(self) -> NodeType:
        """
        Get the node_type of current node. See :class:`mindspore.rewrite.NodeType` for details on node types.

        Returns:
            A NodeType as node_type of node.

        Examples:
            >>> from mindspore.rewrite import SymbolTree
            >>> # Define the network structure of LeNet5. Refer to
            >>> # https://gitee.com/mindspore/docs/blob/master/docs/mindspore/code/lenet.py
            >>> net = LeNet5()
            >>> stree = SymbolTree.create(net)
            >>> node = stree.get_node("conv1")
            >>> node_type = node.get_node_type()
            >>> print(node_type)
            NodeType.CallCell
        """
        return self._node.get_node_type()

    def get_instance_type(self) -> type:
        """
        Gets the instance type called in the code corresponding to the current node.

        - When `node_type` of current node is `CallCell`, the code for that node calls an instance of type ``Cell`` .
        - When `node_type` of current node is `CallPrimitive`, the code for that node calls an instance of
          type ``Primitive`` .
        - When `node_type` of current node is `Tree`, the code for that node calls an instance of network type.
        - When `node_type` of current node is `Python`, `Input`, `Output` or `CallMethod`, the instance type
          is ``NoneType`` .

        Returns:
            The type of instance called in the statement corresponding to the current node.

        Examples:
            >>> from mindspore.rewrite import SymbolTree
            >>> # Define the network structure of LeNet5. Refer to
            >>> # https://gitee.com/mindspore/docs/blob/master/docs/mindspore/code/lenet.py
            >>> net = LeNet5()
            >>> stree = SymbolTree.create(net)
            >>> node = stree.get_node("conv1")
            >>> instance_type = node.get_instance_type()
            >>> print(instance_type)
            <class 'mindspore.nn.layer.conv.Conv2d'>
        """
        return self._node.get_instance_type()

    def get_instance(self):
        return self._node.get_instance()

    def get_args(self) -> [ScopedValue]:
        """
        Get arguments of current node.

        Returns:
            A list of arguments of type ``ScopedValue`` .

        Examples:
            >>> from mindspore.rewrite import SymbolTree
            >>> # Define the network structure of LeNet5. Refer to
            >>> # https://gitee.com/mindspore/docs/blob/master/docs/mindspore/code/lenet.py
            >>> net = LeNet5()
            >>> stree = SymbolTree.create(net)
            >>> node = stree.get_node("conv1")
            >>> print(node.get_args())
            [x]
        """
        return self._node.get_args()

    def get_symbol_tree(self) -> 'SymbolTree':
        """
        Get the symbol tree which current node belongs to.

        Returns:
            SymbolTree, None if current node does not belong to any SymbolTree.

        Examples:
        >>> from mindspore.rewrite import SymbolTree
        >>> # Define the network structure of LeNet5. Refer to
        >>> # https://gitee.com/mindspore/docs/blob/master/docs/mindspore/code/lenet.py
        >>> net = LeNet5()
        >>> stree = SymbolTree.create(net)
        >>> node = stree.get_node("conv1")
        >>> print(type(node.get_symbol_tree()))
        <class 'mindspore.rewrite.api.symbol_tree.SymbolTree'>
        """
        from .symbol_tree import SymbolTree
        stree_impl = self._node.get_belong_symbol_tree()
        if not stree_impl:
            return None
        return SymbolTree(stree_impl)

    def get_kwargs(self) -> {str: ScopedValue}:
        return self._node.get_kwargs()

    def set_attribute(self, key: str, value):
        Validator.check_value_type("key", key, [str], "Node attribute")
        self._node.set_attribute(key, value)

    def get_attributes(self) -> {str: object}:
        return self._node.get_attributes()

    def get_attribute(self, key: str):
        Validator.check_value_type("key", key, [str], "Node attribute")
        return self._node.get_attribute(key)
