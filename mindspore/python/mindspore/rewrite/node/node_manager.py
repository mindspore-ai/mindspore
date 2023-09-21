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
import sys
from typing import Optional
import ast
from .node import Node
from .node_topological_manager import TopoManager
from ..api.node_type import NodeType

if sys.version_info >= (3, 9):
    import ast as astunparse # pylint: disable=reimported, ungrouped-imports
else:
    import astunparse

class NodeManager:
    """
    NodeManager saves nodes and manager nodes' topological relationship.
    """
    def __init__(self, node_namer):
        """Initializer of NodeManager"""
        self._topo_mgr = TopoManager()
        self._nodes: {str, Node} = {}
        self._manager_node_namer = node_namer
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
        # ast of ast.FunctionDef
        self._ast_functiondef = None
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
            node (Node): _description_
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

    def set_ast_functiondef(self, ast_functiondef: ast.FunctionDef):
        """Set _ast_functiondef."""
        self._ast_functiondef = ast_functiondef

    def get_ast_functiondef(self):
        """Get _ast_functiondef."""
        return self._ast_functiondef

    def get_inputs(self):
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

    def dump(self, title="") -> str:
        """
        Dump topological relation.

        title (str): A string as a title will be printed before dumping topological relation.
        """
        try:
            from tabulate import tabulate # pylint: disable=unused-import,reportMissingModuleSource
        except ImportError:
            return ""
        dump_str = "=" * 40 + title + "=" * 40 + '\n'
        node_specs = [[
            n.get_node_type(),
            n.get_name(),
            astunparse.unparse(n.get_ast()).strip(),
            [[key, ((value[0].get_name(), value[1]) if value else ())]
             for key, value in n.get_arg_providers().items()],
            [[
                key,
                [(val[0].get_name(), val[1]) if val else ()
                 for val in value] if value else []
            ] for key, value in n.get_target_users().items()]
        ] for n in NodeManager.nodes(self)]
        dump_str += tabulate(node_specs, headers=['node type', 'name', 'codes', 'arg providers', 'target users'])
        dump_str += '\n' + "=" * (82 + len(title)) + '\n'
        return dump_str

    def _add_node_to_nodes(self, node: Node):
        """
        Add `node` to `_nodes` dict.

        Args:
            node (Node): A Node to be added into `_nodes`.

        Raises:
            RuntimeError: If name of the node is duplicated.
        """
        node_name = node.get_name()
        if self._nodes.get(node_name) is not None:
            raise ValueError(f"Duplicated node name: {node_name} in"
                             f"{self.get_name() if isinstance(self, Node) else 'construct'}")
        self._nodes[node_name] = node
