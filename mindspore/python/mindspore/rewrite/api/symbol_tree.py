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
"""Rewrite module api: SymbolTree."""
from typing import Optional
from types import FunctionType
import mindspore as ms

from mindspore.nn import Cell
from ..._checkparam import Validator
from .node import Node
from ..symbol_tree_builder import SymbolTreeBuilder
from ..symbol_tree import Position, SymbolTree as SymbolTreeImpl

ParamTypes = (int, str, float, bool, Node)
MsDtypes = (ms.float16, ms.float32, ms.float64)


class SymbolTree:
    """
    A `SymbolTree` usually corresponding to forward method of a network.

    Args:
        network (Cell): Network to be rewritten. Only support `Cell`-type-network now.

    Raises:
        RuntimeError: If `network` is not a Cell.
        RuntimeError: If there is any unsupported ast node type while parsing or optimizing.
    """

    def __init__(self, handler: SymbolTreeImpl):
        Validator.check_value_type("handler", handler, [SymbolTreeImpl], "SymbolTree")
        self._symbol_tree: SymbolTreeImpl = handler

    @classmethod
    def create(cls, network):
        """
        Create a new `SymbolTree` of the input `network`.

        Args:
            network (Cell): `network` used to create `SymbolTree`.

        Returns:
            Symboltree, a `Symboltree` created based on `network`.

        Raises:
            TypeError: If `network` is not a `Cell` instance.
        """
        Validator.check_value_type("network", network, [Cell], "SymbolTree")
        return cls(SymbolTreeBuilder(network).build())

    @staticmethod
    def _check_args_type(args):
        for arg in args:
            if arg not in MsDtypes and not isinstance(arg, ParamTypes):
                raise TypeError(f"For call-function Node, got unsupported arg: {arg}, type: {type(arg)}")

    @staticmethod
    def _check_kwargs_type(kwargs):
        for k, v in kwargs.items():
            if not isinstance(k, str):
                raise TypeError(f"For call-function Node, key in kwarg must be a str, but got: {type(v)}",)
            if v not in MsDtypes and not isinstance(v, ParamTypes):
                raise TypeError(f"For call-function Node, got unsupported kwarg value: {v}, type: {type(v)}")

    def create_call_function(self, func, targets, *args, **kwargs):
        """
        Create a Node object and generate the execution code to insert into the source code.
        The source code calls the 'func' function with 'args' and' kwargs' as parameters.

        Args:
            func (FunctionType) - The function to be called.
            targets (list [str]) - indicates the output name. As the output of the node in the source code.
            args (Union[MsDtypes, ParamTypes]) - parameter name of the node. Used as a parameter to a code statement in
                source code. The default value is None, which means there is no parameter input in the cell.
            kwargs ({str: Union[MsDtypes, ParamTypes]}) - The key type must be str, and the value must be value or type
                must be ParamTypes. The input parameter name used to describe the formal parameter with a keyword.
                Enter the name in the source code as the 'kwargs' in the statement expression.The default value is
                None, which means there is no 'kwargs' input.

        Returns:
            An instance of `Node`.

        Raises:
            TypeError: If `func` is not FuntionType.
            TypeError: If `targets` is not `list`.
            TypeError: If the type of `targets` is not str.
            TypeError: If arg in `args` is not ParamType.
            TypeError: If key of `kwarg` is not a str or value of kwarg in `kwargs` is not ParamType.
        """
        Validator.check_value_type("func", func, [FunctionType], "SymbolTree node")
        Validator.check_element_type_of_iterable("targets", targets, [str], "SymbolTree node")
        args_ = list(args)
        SymbolTree._check_args_type(args_)
        for i, arg in enumerate(args_):
            if isinstance(arg, Node):
                args_[i] = arg.get_handler()
        SymbolTree._check_kwargs_type(kwargs)
        for key, value in kwargs.items():
            if isinstance(value, Node):
                kwargs[key] = value.get_handler()
        return Node(self._symbol_tree.create_call_function(func, targets, args_, kwargs))

    def get_handler(self) -> SymbolTreeImpl:
        """
        Get handler of `SymbolTree` implementation.

        Returns:
            An instance of `SymbolTree`.
        """
        return self._symbol_tree

    def nodes(self):
        """
        Get a generator for node of corresponding network.

        Returns:
            A generator for node of current `SymbolTree`.
        """
        for node in self._symbol_tree.nodes():
            yield Node(node)

    def get_node(self, node_name: str) -> Optional[Node]:
        """
        Get node by `node_name`.

        Args:
            node_name (str): A string represents name of node.

        Returns:
            An instance of node if find else None.

        Raises:
            TypeError: If `node_name` is not `str`.
        """
        Validator.check_value_type("node_name", node_name, [str], "SymbolTree")
        node_impl = self._symbol_tree.get_node(node_name)
        if node_impl is None:
            return None
        return Node(node_impl)

    def get_inputs(self) -> [Node]:
        """
        Get input nodes of current `SymbolTree`.

        Returns:
            [Node], the node list of the current `Symboltree`.
        """
        return [Node(node_impl) for node_impl in self._symbol_tree.get_inputs()]

    def before(self, node: Node):
        """
        Get insert position before input `node`.

        `Position` is used to indicate where to insert node, it indicates position in source code rather than position
        in topological order. We don't need to care about what `Position` is, just treat it as a handler and use it as
        an arguments of `insert` api of `SymbolTree`.

        Args:
            node (Node): Indicate the position before which node. Can be a node or name of node.

        Returns:
            A `Position` to indicate where to insert node.

        Raises:
            TypeError: if `node` is not a `Node`.
        """
        Validator.check_value_type("node", node, [Node], "SymbolTree")
        return self._symbol_tree.before(node.get_handler())

    def after(self, node: Node):
        """
        Get insert position after input `node`.

        `Position` is used to indicate where to insert node, it indicates position in source code rather than position
        in topological order. We don't need to care about what `Position` is, just treat it as a handler and use it as
        an arguments of `insert` api of `SymbolTree`.

        Args:
            node (Node): Indicate the position after which node. Can be a node or name of node.

        Returns:
            A `Position` to indicate where to insert node.

        Raises:
            TypeError: If `node` is not a `Node`.
        """
        Validator.check_value_type("node", node, [Node], "SymbolTree")
        return self._symbol_tree.after(node.get_handler())

    def insert(self, position, node: Node) -> Node:
        """
        Insert a `node` into `SymbolTree` at `position`.

        `position` is obtained from `before` api or `after` api of `SymbolTree`.

        Args:
            position (Position): Indicate where to insert `node`.
            node (Node): An instance of Node to be inserted.

        Returns:
            An instance of Node being inserted. `node` could be changed while calling this method for uniqueness and
            custom-object in args or kwargs.

        Raises:
            RuntimeError: If `position` is not belong to current `SymbolTree`.
            TypeError: If `position` is not a `Position`.
            TypeError: If `node` is not a `Node`.
        """
        Validator.check_value_type("position", position, [Position], "SymbolTree")
        Validator.check_value_type("node", node, [Node], "SymbolTree")
        return Node(self._symbol_tree.insert_node(position, node.get_handler()))

    def erase_node(self, node: Node) -> Optional[Node]:
        """
        Erase a `node` from rewrite. Can only erase a node not being depended on.

        Args:
            node (Node): A `Node` to be erased. Can be a node or name of node.

        Returns:
            An instance of `Node` being erased if node is in `SymbolTree` else None.

        Raises:
            TypeError: If `node` is not a `Node`.
        """
        Validator.check_value_type("node", node, [Node], "SymbolTree")
        return Node(self._symbol_tree.erase_node(node.get_handler()))

    def replace(self, old_node: Node, new_nodes: [Node]) -> Node:
        """
        Replace `old_node` with a node_tree.

        Note:

            1. Replace support one-to-one replacement or one-to-multi replacement. If you need multi-to-multi
               replacement, please refer to `PatternEngine`.
            2. When applying one-to-multi replacement, Rewrite will insert all `new_nodes` into symbol_tree.
            3. Caller should maintain arguments and targets of nodes intra sub-tree for specifying topological relation
               intra sub-tree.
            4. Caller should maintain arguments of input nodes of sub-tree and for specifying topological relation of
               inputs of sub-tree.
            5. Rewrite will maintain arguments of prepend node of sub-tree for specifying topological relation of
               outputs of sub-tree.
            6. Rewrite will maintain all inputs of nodes after replace `new_nodes` into `SymbolTree`.

        Args:
            old_node (Node): Node to be replaced.
            new_nodes (list[Node]): Nodes of the node_tree to replace in.

        Returns:
            An instance of Node represents root of node_tree been replaced in.

        Raises:
            RuntimeError: Old node is isolated.
            TypeError: If `old_node` is not a `Node`.
            TypeError: If `new_nodes` is not a `list` or node in `new_nodes` is not a `Node`.
        """
        Validator.check_value_type("old_node", old_node, [Node], "SymbolTree")
        Validator.check_element_type_of_iterable("new_nodes", new_nodes, [Node], "SymbolTree")
        nodes_impl = [node.get_handler() for node in new_nodes]
        return Node(self._symbol_tree.replace(old_node.get_handler(), nodes_impl))

    def set_output(self, index: int, return_value: str) -> Node:
        """
        Set return value of network.

        Args:
            index (int): Indicate which output being modified.
            return_value (str): New return value to been set.

        Returns:
            Return node of current rewrite.

        Raises:
            RuntimeError: If `index` is out of range.
            TypeError: If `index` is not a `int` number.
            TypeError: If `return_value` is not a `str`.
        """
        Validator.check_value_type("index", index, [int], "SymbolTree")
        Validator.check_value_type("return_value", return_value, [str], "SymbolTree")
        return Node(self._symbol_tree.set_output(return_value, index))

    def dump(self):
        """
        Dump graph to console.
        """
        self._symbol_tree.dump()

    def print_node_tabulate(self):
        """
        Print node information of graph.
        """
        self._symbol_tree.print_node_tabulate()

    def get_code(self) -> str:
        """
        Get source code of modified network.

        Returns:
            A str represents source code of modified network.
        """
        return self._symbol_tree.get_code()

    def get_network(self) -> Cell:
        """
        Get modified network.
        The source code of network is saved to a file, the default file name is `network_define.py`.

        Returns:
            A network object.
        """
        return self._symbol_tree.get_network()

    def set_saved_file_name(self, file_name: str):
        """
        Set the name of the file used to save the network.

        Args:
            file_name (str): filename to be set.
        """
        Validator.check_value_type("file_name", file_name, [str], "Saving network")
        self._symbol_tree.set_saved_file_name(file_name)

    def get_saved_file_name(self):
        """Gets the filename used to save the network."""
        return self._symbol_tree.get_saved_file_name()

    def save_network_to_file(self):
        """
        Save the modified network to a file. Default file name is `network_define.py`.
        """
        self._symbol_tree.save_network_to_file()
