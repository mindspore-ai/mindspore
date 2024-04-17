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
"""ControlFlow Node."""
from typing import Union, List
import ast
from mindspore import log as logger
from .node import Node
from .node_manager import NodeManager
from ..api.node_type import NodeType
from ..ast_helpers import AstModifier


class ControlFlow(Node, NodeManager):
    """ControlFlow node is used for statements like loops and `if` ."""
    def __init__(self, node_name: str, ast_node: Union[ast.If, ast.IfExp, ast.For, ast.While], is_orelse: bool,
                 args: None, stree):
        """
        Constructor of ControlFlow.

        Args:
            node_name (str): A string represents name of node. Name of node will be unique when inserted into
                SymbolTree. Name of node also used as field name in network class.
            ast_node (ast.AST): An instance of ast.AST represents control flow statements, can be one of ast.If,
                ast.IfExp, ast.For, ast.While.
            is_orelse (bool): Whether current node presents the else branch of node.
            args (list[ScopedValue]): A list of instance of ScopedValue.
            stree (SymbolTree): Symbol tree used to get node_namer.
        """
        Node.__init__(self, NodeType.ControlFlow, ast_node, None, node_name, args, {}, node_name, None)
        NodeManager.__init__(self)
        NodeManager.set_manager_node_namer(self, stree.get_node_namer())
        NodeManager.set_manager_name(self, node_name)
        self.is_orelse = is_orelse
        self.body_node = None
        self.orelse_node = None
        # record node of another branch
        if is_orelse:
            NodeManager.set_manager_ast(self, ast_node.orelse)
            self.orelse_node = self
        else:
            NodeManager.set_manager_ast(self, ast_node.body)
            self.body_node = self
        # record eval result of test code, used for ast.If
        self.test_result = None
        # record loop variables of control flow, e.g. 'item' of 'for item in self.cell_list:'
        self.loop_vars: List[str] = []

    def erase_node(self, node):
        """Erase node from container."""
        NodeManager.erase_node(self, node)
        # erase node's ast
        if isinstance(node, ControlFlow):
            ret = AstModifier.earse_ast_of_control_flow(self.get_manager_ast(), node.get_ast(), node.is_orelse)
        else:
            ret = AstModifier.erase_ast_from_bodies(self.get_manager_ast(), node.get_ast())
        if not ret:
            raise ValueError(f"Erase node failed, node {node.get_name()} is not in ControlFlow ast tree.")

    def insert_node(self, new_node: Node, base_node: Node, before_node: bool, insert_to_ast: bool = True):
        """
        Insert a node before or after base_node.

        Args:
            new_node (Node): Node to be inserted.
            base_node (Node): New node will be inserted before or after base_node.
            before_node (bool): Indicate whether new node is inserted before base_node.
            insert_to_ast (bool): Indicate whether ast nodes need to be updated.
        """
        NodeManager.insert_node(self, new_node, base_node, before_node)
        if insert_to_ast:
            stree = self.get_belong_symbol_tree()
            stree.insert_to_ast_while_insert_node(new_node, base_node, before_node)

    def set_belong_symbol_tree(self, symbol_tree):
        """Set the symbol tree to which node belongs."""
        self._belong_tree = symbol_tree
        for node in self.nodes():
            node.set_belong_symbol_tree(symbol_tree)

    def set_body_node(self, body_node):
        """Set body_node of control flow"""
        self.body_node = body_node

    def set_orelse_node(self, orelse_node):
        """Set orelse_node of control flow"""
        self.orelse_node = orelse_node

    def get_source_code(self) -> str:
        """Print source code of control flow, overwriting the implementation in Node."""
        source_code = Node.get_source_code(self)
        if self.orelse_node:
            else_pos = source_code.find("else:")
            if else_pos == -1:
                logger.warning(f"Failed to find code 'else:' in control flow node {self.get_name()}, "
                               f"return all codes.")
                return source_code
            if self.is_orelse:
                source_code = source_code[else_pos:].strip()
            else:
                source_code = source_code[:else_pos].strip()
        return source_code
