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
"""SymbolTree nodes manager."""
from typing import Optional, Union
import ast
from .node import Node
from .node_topological_manager import TopoManager
from ..api.node_type import NodeType
from ..api.scoped_value import ScopedValue


class NodeManager:
    """
    NodeManager saves nodes and manager nodes' topological relationship.
    """
    def __init__(self):
        """Initializer of NodeManager"""
        self._topo_mgr = TopoManager()
        self._nodes: {str, Node} = {}
        self._manager_node_namer = None
        # record all tree nodes, which is used when generating codes
        self._tree_nodes: [Node] = []
        # head node is always point to the first node of nodes
        self._head = None
        # tail node is always point to the last node of nodes
        self._tail = None
        # nodes of Input type
        self._inputs: [Node] = []
        # nodes of Output type
        self._returns: [Node] = []
        # ast of node manager
        # SymbolTree    -> ast.FunctionDef
        # CallFunction  -> ast.FunctionDef
        # ControlFlow   -> list
        # CellContainer -> ast.Assign
        self._node_manager_ast: Union[ast.AST, list] = None
        # name of manager
        self._manager_name = "OriginNodeManager"

    @property
    def node_list(self):
        """ Get node list. """
        nodes = []
        node = self._head
        while node is not None:
            nodes.append(node)
            node = node.get_next()
        return nodes

    @property
    def node_count(self):
        """Number of nodes."""
        node_num = 0
        node = self._head
        while node is not None:
            node_num = node_num + 1
            node = node.get_next()
        return node_num

    def insert_node(self, new_node: Node, base_node: Node, before_node: bool):
        """
        Insert a node before or after base_node.

        Args:
            new_node (Node): Node to be inserted.
            base_node (Node): New node will be inserted before or after base_node.
            before_node (bool): Indicate whether new node is inserted before base_node.
        """
        # update node name
        new_node_name = self._manager_node_namer.get_name(new_node)
        new_node.set_name(new_node_name)
        if isinstance(new_node, NodeManager):
            new_node.set_manager_name(new_node_name)
        # insert node to list table
        if base_node is None:
            if self._nodes:
                raise ValueError("base_node cannot be None when node inserted is not the first node.")
            self._head = new_node
            self._tail = new_node
        elif before_node:
            base_node.insert_before(new_node)
            if self._head == base_node:
                self._head = new_node
        else:
            base_node.insert_after(new_node)
            if self._tail == base_node:
                self._tail = new_node
        self._add_node_to_nodes(new_node)
        self._topo_mgr.on_insert_node(new_node)
        new_node.set_node_manager(self)
        # record Input nodes, Output nodes and tree nodes
        if new_node.get_node_type() == NodeType.Output:
            self._returns.append(new_node)
        elif new_node.get_node_type() == NodeType.Input:
            self._inputs.append(new_node)
        elif new_node.get_node_type() == NodeType.Tree:
            self._tree_nodes.append(new_node)

    def erase_node(self, node: Node):
        """
        Erase a node from nodes.

        Args:
            node (Node): Node to be erased.
        """
        self._topo_mgr.on_erase_node(node)
        for key, value in self._nodes.items():
            if id(value) == id(node):
                # update self._head and self._tail
                if self._head == node:
                    self._head = node.get_next()
                if self._tail == node:
                    self._tail = node.get_prev()
                # erase node
                self._nodes.pop(key)
                value.isolate()
                break

    def nodes(self):
        """
        Get nodes.

        Returns:
            A list of nodes.
        """
        # If iterating nodes directly without new list, iteration may stuck caused
        # by node topology being modified during iteration.
        nodes = []
        node = self._head
        while node is not None:
            nodes.append(node)
            node = node.get_next()
        return nodes

    def get_node(self, node_name: str) -> Optional[Node]:
        """
        Get node of current NodeManager by `node_name`.

        Args:
            node_name (str): A str represents name of node as key of query.

        Returns:
            An instance of Node if found else None.
        """
        return self._nodes.get(node_name)

    def append_python_node(self, new_node: Node):
        """Append python node"""
        NodeManager.insert_node(self, new_node, self._tail, False)

    def get_head(self):
        """Get head node of nodes"""
        return self._head

    def get_tail(self):
        """Get tail node of nodes"""
        return self._tail

    def reg_observer(self, observer):
        """Register observer to monitor code changes."""
        self._topo_mgr.reg_observer(observer)
        for node in self.nodes():
            if isinstance(node, NodeManager):
                node.reg_observer(observer)
            if node.get_node_type() == NodeType.Tree:
                node.symbol_tree.reg_observer(observer)

    def get_tree_nodes(self):
        """Get tree nodes inserted into symbol tree, include nodes later erased by user."""
        tree_nodes = []
        tree_nodes.extend(self._tree_nodes)
        for node in self.nodes():
            if isinstance(node, NodeManager):
                tree_nodes.extend(node.get_tree_nodes())
        return tree_nodes

    def set_manager_ast(self, node_manager_ast: Union[ast.AST, list]):
        """Set _node_manager_ast."""
        self._node_manager_ast = node_manager_ast

    def get_manager_ast(self):
        """Get _node_manager_ast."""
        return self._node_manager_ast

    def get_input_nodes(self):
        """Get _inputs"""
        return self._inputs

    def get_returns(self):
        """Get _returns"""
        return self._returns

    def set_manager_name(self, name: str):
        """Set _manager_name"""
        self._manager_name = name

    def get_manager_name(self):
        """Get _manager_name"""
        return self._manager_name

    def on_update_arg(self, node: Node, arg_idx: int, old_arg: ScopedValue, new_arg: ScopedValue):
        """
        Update node topological when node arg is modified.
        """
        self._topo_mgr.on_update_arg(node, arg_idx, old_arg, new_arg)

    def on_update_arg_by_node(self, dst_node: Node, arg_idx: int, src_node: Node, out_idx: int):
        """
        Update node topological when node arg is modified by another node.
        """
        self._topo_mgr.on_update_arg_by_node(dst_node, arg_idx, src_node, out_idx)

    def dump(self, title="") -> str:
        """
        Dump topological relation.

        title (str): A string as a title will be printed before dumping topological relation.
        """
        try:
            from tabulate import tabulate # pylint: disable=unused-import,reportMissingModuleSource
        except ImportError:
            return ""
        dump_str = f"\n[{title}]\n"
        node_specs = [[
            n.get_node_type(),
            n.get_name(),
            n.get_source_code(),
            [[key, ((value[0].get_name(), value[1]) if value else ())]
             for key, value in n.get_arg_providers().items()],
            [[
                key,
                [(val[0].get_name(), val[1]) if val else ()
                 for val in value] if value else []
            ] for key, value in n.get_target_users().items()]
        ] for n in NodeManager.nodes(self)]
        dump_str += tabulate(node_specs, headers=['node type', 'name', 'codes', 'arg providers', 'target users'])
        dump_str += '\n'
        return dump_str

    def get_top_manager(self) -> 'NodeManager':
        """
        Get the top node_manager with type of no-method CallFunction or SymbolTree this
        node_manager belongs to.
        """
        from .call_function import CallFunction
        from ..symbol_tree import SymbolTree
        if isinstance(self, SymbolTree):
            return self
        if isinstance(self, CallFunction) and not self.is_method():
            return self
        return self.get_node_manager().get_top_manager()

    def set_manager_node_namer(self, node_namer):
        """Set manager node namer"""
        self._manager_node_namer = node_namer

    def _add_node_to_nodes(self, node: Node):
        """
        Add `node` to `_nodes` dict.

        Args:
            node (Node): A Node to be added into `_nodes`.

        Raises:
            ValueError: If name of the node is duplicated.
        """
        node_name = node.get_name()
        if self._nodes.get(node_name) is not None:
            raise ValueError(f"Duplicated node name: {node_name} in"
                             f"{self.get_name() if isinstance(self, Node) else 'construct'}")
        self._nodes[node_name] = node
