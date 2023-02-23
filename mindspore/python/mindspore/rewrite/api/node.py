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

from typing import Union, Optional

from mindspore.nn import Cell
from mindspore.ops.primitive import Primitive
from ..._checkparam import Validator
from ..node import Node as NodeImpl
from ..symbol_tree import SymbolTree as SymbolTreeImpl
from .node_type import NodeType
from .scoped_value import ScopedValue


class Node:
    """
    Node is a data structure represents a source code line in network.

    For the most part, Node represents an operator invoking in forward which could be an instance of `Cell`, an instance
    of `Primitive` or a callable method.

    Args:
        node (NodeImpl): A handler of `NodeImpl`. `NodeImpl` mentioned below is implementation of `Node` which is not
            an interface of Rewrite. Rewrite recommend invoking specific create method of `Node`
            to instantiate an instance of Node such as `create_call_cell` rather than invoking constructor of `Node`
            directly, so don't care about what is `NodeImpl` and use its instance just as a handler.
    """

    def __init__(self, node: NodeImpl):
        self._node = node


    def __eq__(self, other: 'Node'):
        if not isinstance(other, Node):
            return False
        return self._node == other._node

    @staticmethod
    def create_call_cell(cell: Cell, targets: [Union[ScopedValue, str]], args: [ScopedValue] = None,
                         kwargs: {str: ScopedValue}=None, name: str = "", is_sub_net: bool = False) -> 'Node':
        """
        Create a node. Only support create from a `Cell` now.

        A node is corresponding to source code like:

        .. code-block::

            `targets` = self.`name`(*`args`, **`kwargs`)

        Args:
            cell (Cell): Cell-operator of this forward-layer.
            targets (list[ScopedValue]): Indicate output names. Used as targets of an assign statement in source code.
                Rewrite will check and ensure the uniqueness of each target while node being inserted.
            args (list[ScopedValue]): Indicate input names. Used as args of a call expression of an assign statement in
                source code. Default is None indicate the `cell` has no args inputs. Rewrite will check and ensure the
                uniqueness of each arg while node being inserted.
            kwargs (dict): Type of key must be `str` and type of value must be `ScopedValue`.
                Indicate keyword input names. Used as kwargs of a call expression of an assign statement in source code.
                Default is None indicate the `cell` has no kwargs inputs. Rewrite will check and ensure the uniqueness
                of each kwarg while node being inserted.
            name (str): Indicate the name of node. Used as field name in source code. Default is None. Rewrite will
                generate name from `targets` when name is None. Rewrite will check and ensure the uniqueness of `name`
                while node being inserted.
            is_sub_net (bool): Indicate that is `cell` a network. If `is_sub_net` is true, Rewrite will try to parse the
                `cell` to a TreeNode, else a CallCell Node. Default is a False.

        Returns:
            An instance of `Node`.

        Raises:
            TypeError: If `cell` is not a `Cell`.
            TypeError: If `targets` is not `list`.
            TypeError: If the type of `targets` is not in `[ScopedValue, str]`.
            TypeError: If arg in `args` is not a `ScopedValue`.
            TypeError: If key of `kwarg` is not a str or value of kwarg in `kwargs` is not a `ScopedValue`.
        """
        Validator.check_value_type("cell", cell, [Cell, Primitive], "Node")
        Validator.check_element_type_of_iterable("targets", targets, [ScopedValue, str], "Node")
        Validator.check_value_type("name", name, [str], "Node")
        Validator.check_value_type("is_sub_net", is_sub_net, [bool], "Node")
        if args is not None:
            Validator.check_element_type_of_iterable("args", args, [ScopedValue], "Node")
        if kwargs is not None:
            Validator.check_element_type_of_dict("kwargs", kwargs, [str], [ScopedValue], "Node")
        return Node(NodeImpl.create_call_op(cell, None, targets, ScopedValue.create_naming_value(name, "self"),
                                            args, kwargs, name, is_sub_net))

    def get_handler(self) -> NodeImpl:
        return self._node

    def get_inputs(self) -> ['Node']:
        """
        Get input nodes of current node in topological order.

        Returns:
            A list of instances of `Node` as input nodes.
        """
        return [Node(node_impl) for node_impl in self._node.get_inputs()]

    def get_users(self) -> ['Node']:
        """
        Get output nodes of current node in topological order.

        Returns:
            A list of nodes represents users.

        Examples:
            >>> from mindspore.rewrite import SymbolTree
            >>> from lenet import Lenet
            >>> net = Lenet()
            >>> stree = SymbolTree.create(net)
            >>> node = stree.get_node("conv1")
            >>> users = node.get_users()
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
            >>> from lenet import Lenet
            >>> net = Lenet()
            >>> stree = SymbolTree.create(net)
            >>> node = stree.get_node("conv1")
            >>> node.set_arg(0, "x")
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
            out_idx (int, optional): Indicate which output of `src_node` as new input of current node. Default is None
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
            >>> from lenet import Lenet
            >>> net = Lenet()
            >>> stree = SymbolTree.create(net)
            >>> src_node = stree.get_node("conv1")
            >>> dst_node = stree.get_node("conv2")
            >>> dst_node.set_arg_by_node(0, src_node)
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
        return self._node.get_targets()

    def get_name(self) -> str:
        """
        Get the name of current node.

        When node has been inserted into `SymbolTree`, the name of node should be unique in `SymbolTree`.

        Returns:
            A string as name of node.

        Examples:
            >>> from mindspore.rewrite import SymbolTree
            >>> from lenet import Lenet
            >>> net = Lenet()
            >>> stree = SymbolTree.create(net)
            >>> node = stree.get_node("conv1")
            >>> name = node.get_name()
        """
        return self._node.get_name()

    def get_node_type(self) -> NodeType:
        """
        Get the node_type of current node.

        Returns:
            A NodeType as node_type of node.

        Examples:
            >>> from mindspore.rewrite import SymbolTree
            >>> from lenet import Lenet
            >>> net = Lenet()
            >>> stree = SymbolTree.create(net)
            >>> node = stree.get_node("conv1")
            >>> node_type = node.get_node_type()
        """
        return self._node.get_node_type()

    def get_instance_type(self) -> type:
        """
        Get the instance_type of current node.

        - When node_type of current node is `CallCell`, instance_type is type of cell-op.
        - When node_type of current node is `CallPrimitive`, instance_type is type of primitive-op.
        - When node_type of current node is `Tree`, instance_type is type of network-cell.
        - When node_type of current node is `Python`, `Input`, `Output` or `CallMethod`, instance_type should be
          NoneType.

        Returns:
            A type object represents corresponding instance type of current node.
        """
        return self._node.get_instance_type()

    def get_instance(self):
        return self._node.get_instance()

    def get_args(self) -> [ScopedValue]:
        return self._node.get_args()

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
