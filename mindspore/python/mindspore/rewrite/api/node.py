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
from ..node import Node as NodeImpl
from ..symbol_tree import SymbolTree as SymbolTreeImpl
from .node_type import NodeType
from .scoped_value import ScopedValue


class Node:
    """
    Node is a data structure represents a source code line in network.

    For the most part, Node represents an operator invoking in forward which could be an instance of `Cell`, an instance
    of `Primitive` or a callable method.

    `NodeImpl` mentioned below is implementation of `Node` which is not an interface of Rewrite. Rewrite recommend
    invoking specific create method of `Node` to instantiate an instance of Node such as `create_call_cell` rather than
    invoking constructor of `Node` directly, so don't care about what is `NodeImpl` and use its instance just as a
    handler.

    Args:
        node (NodeImpl): A handler of `NodeImpl`.
    """

    def __init__(self, node: NodeImpl):
        self._node = node

    def get_handler(self) -> NodeImpl:
        """
        Get handler of node implementation.

        Returns:
            An instance of `NodeImpl`.
        """
        return self._node

    @staticmethod
    def create_call_cell(cell: Cell, targets: [Union[ScopedValue, str]], args: [ScopedValue] = None,
                         kwargs: {str: ScopedValue}=None, name: str = "") -> 'Node':
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
            kwargs (dict{str: ScopedValue}): Indicate keyword input names. Used as kwargs of a call expression of an
                assign statement in source code. Default is None indicate the `cell` has no kwargs inputs. Rewrite will
                check and ensure the uniqueness of each kwarg while node being inserted.
            name (str): Indicate the name of node. Used as field name in source code. Default is None. Rewrite will
                generate name from `targets` when name is None. Rewrite will check and ensure the uniqueness of `name`
                while node being inserted.

        Returns:
            An instance of `Node`.

        Raises:
            RuntimeError: If `cell` is not a `Cell`.
            RuntimeError: If `targets` is None.
            RuntimeError: If target in `targets` is not a `NamingValue`-`ScopedValue`.
            RuntimeError: If arg in `args` is not a `NamingValue`-`ScopedValue` or a `CustomObjValue`-`ScopedValue`.
            RuntimeError: If value of kwarg in `kwargs` is not a `NamingValue`-`ScopedValue` or a
                `CustomObjValue`-`ScopedValue`.
        """
        return Node(NodeImpl.create_call_buildin_op(cell, None, targets, ScopedValue.create_naming_value(name, "self"),
                                                    args, kwargs, name))

    def get_prev(self) -> 'Node':
        """
        Get previous node of current node in source code order.

        Returns:
            An instance of `Node` as previous node.
        """
        return Node(self._node.get_prev())

    def get_next(self) -> 'Node':
        """
        Get next node of current node in source code order.

        Returns:
            An instance of `Node` as next node.
        """
        return Node(self._node.get_next())

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
            RuntimeError: If `index` is out of range.
            RuntimeError: If `arg` a `NamingValue`-`ScopedValue` or a `CustomObjValue`-`ScopedValue` when `arg` is an
                `ScopedValue`.
        """
        belong_symbol_tree: SymbolTreeImpl = self._node.get_belong_symbol_tree()
        if belong_symbol_tree is None:
            self._node.set_arg(arg, index)
        else:
            belong_symbol_tree.set_node_arg(self._node, index, arg)

    def set_arg_by_node(self, arg_idx: int, src_node: 'Node', out_idx: Optional[int] = None):
        """
        Set argument of current node by another `Node`.

        Args:
            arg_idx (int): Indicate which input being modified.
            src_node (Node): A `Node` as new input. Can be a node or name of node.
            out_idx (int, optional): Indicate which output of `src_node` as new input of current node. Default is None
                which means use first output of `src_node` as new input.

        Raises:
            RuntimeError: If `src_node` is not belong to current `SymbolTree`.
            RuntimeError: If current node and `src_node` is not belong to same `SymbolTree`.
            RuntimeError: If `arg_idx` is out of range.
            RuntimeError: If `out_idx` is out of range.
            RuntimeError: If `src_node` has multi-outputs while `out_idx` is None or `out_idx` is not offered.
        """
        belong_symbol_tree: SymbolTreeImpl = self._node.get_belong_symbol_tree()
        if belong_symbol_tree is None:
            self._node.set_arg_by_node(arg_idx, src_node._node, out_idx)
        else:
            belong_symbol_tree.set_node_arg_by_node(self._node, arg_idx, src_node._node, out_idx)

    def get_targets(self) -> [ScopedValue]:
        """
        Get targets of current node.

        - When node_type of current node is `CallCell`, `CallPrimitive`, `CallMethod` or `Tree`, `targets` are strings
          represents invoke result of the cell-op or primitive-op or function-call which are corresponding to targets of
          ast.Assign.
        - When node_type of current node is Input, `targets` should have only one element which is a string represents
          parameter of function.
        - When node_type of current node is `Python` or `Output`, `targets` are don't-care.

        Returns:
            A list of instances of ScopedValue as targets of node.
        """
        return self._node.get_targets()

    def get_name(self) -> str:
        """
        Get the name of current node.

        When node has been inserted into `SymbolTree`, the name of node should be unique in `SymbolTree`.

        Returns:
            A string as name of node.
        """
        return self._node.get_name()

    def get_node_type(self) -> NodeType:
        """
        Get the node_type of current node.

        Returns:
            A NodeType as node_type of node.
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
        """
        Get the instance of current node.

        - When node_type of current node is `CallCell`, instance is an instance of Cell.
        - When node_type of current node is `CallPrimitive`, instance is an instance of primitive.
        - When node_type of current node is `Tree`, instance is an instance of network-cell.
        - When node_type of current node is `Python`, `Input`, `Output` or `CallMethod`, instance should be None.

        Returns:
            A object represents corresponding instance of current node.
        """
        return self._node.get_instance()

    def get_args(self) -> [ScopedValue]:
        """
        Get the arguments of current node.

        - When node_type of current node is `CallCell`, `CallPrimitive` or `Tree`, arguments are corresponding to args
          of ast.Call which represents arguments to invoke forward method of cell-op or primitive-op.
        - When node_type of current node is `Input`, arguments represents default-value of argument of function.
        - When node_type of current node is `Output`, arguments represents return values.
        - When node_type of current node is `Python`, arguments are don't-care.

        Returns:
            A list of instances of `ScopedValue`.
        """
        return self._node.get_args()

    def get_kwargs(self) -> {str: ScopedValue}:
        """
        Get the keyword arguments of current node.

        - When node_type of current node is `CallCell`, `CallPrimitive` or `Tree`, keyword arguments are corresponding
          to kwargs of ast.Call which represents arguments to invoke forward method of cell-op or primitive-op.
        - When node_type of current node is `Python`, `Input` or `Output`, keyword arguments are don't-care.

        Returns:
            A dict of str to instance of `ScopedValue`.
        """
        return self._node.get_kwargs()

    def set_attribute(self, key: str, value):
        """
        Set attribute of current node.

        Args:
            key (str): Key of attribute.
            value (object): Value of attribute.
        """
        self._node.set_attribute(key, value)

    def get_attributes(self) -> {str: object}:
        """
        Get all attributes of current node.

        Returns:
            A dict of str to instance of object as attributes.
        """
        return self._node.get_attributes()

    def get_attribute(self, key: str):
        """
        Get attribute of current node by key.

        Returns:
            A object as attribute.
        """
        return self._node.get_attribute(key)

    def __eq__(self, other: 'Node'):
        return self._node == other._node
