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
from typing import Optional, Union
from types import FunctionType
import mindspore as ms

from mindspore.nn import Cell
from mindspore import _checkparam as Validator
from .node import Node
from ..symbol_tree_builder import SymbolTreeBuilder
from ..symbol_tree import Position, SymbolTree as SymbolTreeImpl

ParamTypes = (int, str, float, bool, Node)
MsDtypes = (ms.float16, ms.float32, ms.float64)


class SymbolTree:
    """
    SymbolTree stores information about a network, including statements of the network's forward
    computation process and the topological relationship between statement input and output.

    The statements in the network are saved in the SymbolTree in the form of nodes, and by processing
    the nodes in the SymbolTree, you can delete the network code, insert and replace it, and get the
    modified network code and network instances.

    Args:
        handler (SymbolTreeImpl): SymbolTree internal implementation instance. It is recommended to call the `create`
            method in SymbolTree to create a SymbolTree, rather than calling SymbolTree's constructor directly.
            Don't care what `SymbolTreeImpl` is, just treat it as a handle.
    """

    def __init__(self, handler: SymbolTreeImpl):
        Validator.check_value_type("handler", handler, [SymbolTreeImpl], "SymbolTree")
        self._symbol_tree: SymbolTreeImpl = handler

    @classmethod
    def create(cls, network):
        """
        Create a SymbolTree object by passing in the network instance `network`.

        This interface parses the `network` instance, expands each source
        code statement of the forward computation process, and parses it into nodes,
        which is stored in the SymbolTree.

        Args:
            network (Cell): `network` used to create SymbolTree.

        Returns:
            Symboltree, a SymbolTree created based on `network`.

        Raises:
            TypeError: If `network` is not a `Cell` instance.

        Examples:
            >>> from mindspore.rewrite import SymbolTree
            >>> # Define the network structure of LeNet5. Refer to
            >>> # https://gitee.com/mindspore/docs/blob/master/docs/mindspore/code/lenet.py
            >>> net = LeNet5()
            >>> stree = SymbolTree.create(net)
            >>> print(type(stree))
            <class 'mindspore.rewrite.api.symbol_tree.SymbolTree'>
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

    def create_call_function(self, func, targets, *args, **kwargs): # pylint: disable=C0111
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
        return Node(self._symbol_tree._create_call_function(func, targets, args_, kwargs)) # pylint: disable=W0212

    def get_handler(self) -> SymbolTreeImpl:
        return self._symbol_tree

    def nodes(self):
        """
        Get the generator of the node in the current SymbolTree, which is used to iterate
        through the nodes in SymbolTree.

        Returns:
            A generator for node of current SymbolTree.

        Examples:
            >>> from mindspore.rewrite import SymbolTree
            >>> # Define the network structure of LeNet5. Refer to
            >>> # https://gitee.com/mindspore/docs/blob/master/docs/mindspore/code/lenet.py
            >>> net = LeNet5()
            >>> stree = SymbolTree.create(net)
            >>> print([node.get_name() for node in stree.nodes()])
            ['input_x', 'conv1', 'relu', 'max_pool2d', 'conv2', 'relu_1', 'max_pool2d_1',
             'flatten', 'fc1', 'relu_2', 'fc2', 'relu_3', 'fc3', 'return']
        """
        for node in self._symbol_tree.nodes():
            yield Node(node)

    def get_node(self, node_name: str) -> Optional[Node]:
        """
        Get the node with the name `node_name` in the SymbolTree.

        Args:
            node_name (str): The name of node.

        Returns:
            Node with name of `node_name` . Return ``None`` if there is no node named `node_name` in SymbolTree.

        Examples:
            >>> from mindspore.rewrite import SymbolTree
            >>> # Define the network structure of LeNet5. Refer to
            >>> # https://gitee.com/mindspore/docs/blob/master/docs/mindspore/code/lenet.py
            >>> net = LeNet5()
            >>> stree = SymbolTree.create(net)
            >>> node = stree.get_node('conv1')
            >>> print(node.get_name())
            conv1
        """
        Validator.check_value_type("node_name", node_name, [str], "SymbolTree")
        node_impl = self._symbol_tree.get_node(node_name)
        if node_impl is None:
            return None
        return Node(node_impl)

    def get_inputs(self) -> [Node]:
        return [Node(node_impl) for node_impl in self._symbol_tree.get_inputs()]

    def before(self, node: Union[Node, str]):
        """
        Returns a location information before `node`. The return value of this interface is
        used as a parameter for the insert operation.

        Args:
            node (Union[Node, str]): Indicate the position before which node. Can be a node or name of node.

        Returns:
            A `Position` to indicate where to insert node.

        Raises:
            TypeError: if `node` is not a `Node`.

        Examples:
            >>> from mindspore.rewrite import SymbolTree
            >>> # Define the network structure of LeNet5. Refer to
            >>> # https://gitee.com/mindspore/docs/blob/master/docs/mindspore/code/lenet.py
            >>> net = LeNet5()
            >>> stree = SymbolTree.create(net)
            >>> for node in stree.nodes():
            ...     if node.get_name() == "conv1":
            ...         position = stree.before(node)
        """
        Validator.check_value_type("node", node, [Node], "SymbolTree")
        return self._symbol_tree.before(node.get_handler())

    def after(self, node: Union[Node, str]):
        """
        Returns a location information after `node`. The return value of this interface is
        used as a parameter for the insert operation.

        Args:
            node (Union[Node, str]): Indicate the position after which node. Can be a node or name of node.

        Returns:
            A `Position` to indicate where to insert node.

        Raises:
            TypeError: If `node` is not a `Node`.

        Examples:
            >>> from mindspore.rewrite import SymbolTree
            >>> # Define the network structure of LeNet5. Refer to
            >>> # https://gitee.com/mindspore/docs/blob/master/docs/mindspore/code/lenet.py
            >>> net = LeNet5()
            >>> stree = SymbolTree.create(net)
            >>> for node in stree.nodes():
            ...     if node.get_name() == "conv1":
            ...         position = stree.after(node)
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
            An instance of Node being inserted.

        Raises:
            RuntimeError: If `position` is not belong to current `SymbolTree`.
            TypeError: If `position` is not a `Position`.
            TypeError: If `node` is not a `Node`.

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
        """
        Validator.check_value_type("position", position, [Position], "SymbolTree")
        Validator.check_value_type("node", node, [Node], "SymbolTree")
        return Node(self._symbol_tree.insert_node(position, node.get_handler()))

    def erase(self, node: Union[Node, str]) -> Optional[Node]:
        """
        Erase a `node` from rewrite.

        Args:
            node (Union[Node, str]): A `Node` to be erased. Can be a node or name of node.

        Returns:
            An instance of `Node` being erased if node is in `SymbolTree` else None.

        Raises:
            TypeError: The type of `node` is not Node.

        Examples:
            >>> from mindspore.rewrite import SymbolTree
            >>> # Define the network structure of LeNet5. Refer to
            >>> # https://gitee.com/mindspore/docs/blob/master/docs/mindspore/code/lenet.py
            >>> net = LeNet5()
            >>> stree = SymbolTree.create(net)
            >>> node = stree.get_node("conv1")
            >>> stree.erase(node)
        """
        Validator.check_value_type("node", node, [Node], "SymbolTree")
        return Node(self._symbol_tree.erase_node(node.get_handler()))

    def replace(self, old_node: Node, new_nodes: [Node]) -> Node:
        """
        Replace the `old_node` with nodes in the `new_nodes` list.

        Nodes in `new_nodes` will be inserted into SymbolTree sequentially, and then `old_node` will be deleted.

        Note:
            - Replace support one-to-one replacement or one-to-multi replacement. If you need multi-to-multi
              replacement, please refer to `PatternEngine`.
            - Caller should maintain the topological relationship between each node in the `new_nodes` , as well as
              the topological relationship between nodes in the `new_nodes` and nodes in the original tree.

        Args:
            old_node (Node): Node to be replaced.
            new_nodes (list[Node]): Nodes of the node_tree to replace in.

        Returns:
            An instance of Node represents root of node_tree been replaced in.

        Raises:
            RuntimeError: Old node is not isolated.
            TypeError: If `old_node` is not a `Node`.
            TypeError: If `new_nodes` is not a `list` or node in `new_nodes` is not a `Node`.

        Examples:
            >>> from mindspore.rewrite import SymbolTree, ScopedValue
            >>> import mindspore.nn as nn
            >>> # Define the network structure of LeNet5. Refer to
            >>> # https://gitee.com/mindspore/docs/blob/master/docs/mindspore/code/lenet.py
            >>> net = LeNet5()
            >>> stree = SymbolTree.create(net)
            >>> node = stree.get_node("conv1")
            >>> new_node = node.create_call_cell(cell=nn.ReLU(), targets=['x'],
            ...                                  args=[ScopedValue.create_naming_value('x')], name='new_relu')
            >>> stree.replace(node, [new_node])
        """
        Validator.check_value_type("old_node", old_node, [Node], "SymbolTree")
        Validator.check_element_type_of_iterable("new_nodes", new_nodes, [Node], "SymbolTree")
        nodes_impl = [node.get_handler() for node in new_nodes]
        return Node(self._symbol_tree.replace(old_node.get_handler(), nodes_impl))

    def set_output(self, index: int, return_value: str) -> Node:
        Validator.check_value_type("index", index, [int], "SymbolTree")
        Validator.check_value_type("return_value", return_value, [str], "SymbolTree")
        return Node(self._symbol_tree.set_output(return_value, index))

    def dump(self):
        self._symbol_tree.dump()

    def print_node_tabulate(self):
        """
        Print the topology information of nodes in SymbolTree, including node type, node name, node code,
        and node input-output relationship.
        The information is output to the screen using the print interface.

        .. warning::
            This is an experimental API that is subject to change or deletion.
        """
        self._symbol_tree.print_node_tabulate()

    def get_code(self) -> str:
        """
        Get source code corresponding to the network information in SymbolTree.
        If the network has already been modified, the source code of modified network is returned.

        Returns:
            A str represents source code of modified network.

        Examples:
            >>> from mindspore.rewrite import SymbolTree
            >>> # Define the network structure of LeNet5. Refer to
            >>> # https://gitee.com/mindspore/docs/blob/master/docs/mindspore/code/lenet.py
            >>> net = LeNet5()
            >>> stree = SymbolTree.create(net)
            >>> codes = stree.get_code()
            >>> print(codes)
        """
        return self._symbol_tree.get_code()

    def get_network(self) -> Cell:
        """
        Get the network object generated based on SymbolTree.
        The source code is saved to a file in the 'rewritten_network' folder of the current directory.

        Returns:
            A network object generated from SymbolTree.

        Examples:
            >>> from mindspore.rewrite import SymbolTree
            >>> # Define the network structure of LeNet5. Refer to
            >>> # https://gitee.com/mindspore/docs/blob/master/docs/mindspore/code/lenet.py
            >>> net = LeNet5()
            >>> stree = SymbolTree.create(net)
            >>> new_net = stree.get_network()
        """
        return self._symbol_tree.get_network()

    def set_saved_file_name(self, file_name: str):
        Validator.check_value_type("file_name", file_name, [str], "Saving network")
        self._symbol_tree.set_saved_file_name(file_name)

    def get_saved_file_name(self):
        return self._symbol_tree.get_saved_file_name()

    def save_network_to_file(self):
        self._symbol_tree.save_network_to_file()

    def unique_name(self, name: str = "output"):
        """
        Based on the given `name` , returns a new name that is unique within the symbol tree.
        This interface can be used when a variable name that does not conflict is required.

        Args:
            name (str, optional): The prefix of the name. Defaults to ``"output"`` .

        Returns:
            str, A new, unique name within a symbol tree in the format `name_n`, where `n` is a numeric subscript.
            If there is no name conflict when entered `name`, there is no numeric subscript.
        """
        return self._symbol_tree.unique_name(name)
