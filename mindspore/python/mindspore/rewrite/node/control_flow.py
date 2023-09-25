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
from typing import List
import ast
from .node import Node, TreeNode
from .node_manager import NodeManager
from ..api.scoped_value import ScopedValue
from ..api.node_type import NodeType
from ..ast_helpers import AstModifier


class ControlFlow(Node, NodeManager):
    """ControlFlow node is used for statements like loops and `if` ."""
    def __init__(self, node_name: str, ast_body: List[ast.AST], stree):
        """
        Constructor of ControlFlow.

        Args:
            node_name (str): A string represents name of node. Name of node will be unique when inserted into
                SymbolTree. Name of node also used as field name in network class.
            ast_node (ast.AST): An instance of ast.AST represents control flow statements, can be one of ast.If,
                ast.Ifexp, ast.For, ast.While.
            is_orelse (bool): Whether process else branch of node.
            stree (SymbolTree): Symbol tree used to get node_namer.
        """
        Node.__init__(self, NodeType.ControlFlow, ast_body, None, node_name, [], [], node_name, None)
        NodeManager.__init__(self, stree.get_node_namer())
        NodeManager.set_manager_name(self, node_name)
        self.ast_body = ast_body

    def erase_node(self, node):
        """Erase node from container."""
        NodeManager.erase_node(self, node)
        # erase node's ast
        ret = AstModifier.erase_ast_from_bodies(self.ast_body, node.get_ast())
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
            ast_assign = new_node.get_ast()
            if ast_assign is None:
                func_name = new_node.get_belong_symbol_tree().unique_func_name(new_node.get_name())
                new_node.set_func_name(ScopedValue.create_naming_value(func_name, "self"))
                ast_assign = new_node.update_ast_node()
            # Save instance into _origin_network.
            stree = self.get_belong_symbol_tree()
            setattr(stree.get_origin_network(), new_node.get_name(), new_node.get_instance())
            # Insert ast_assign to __init__ function
            if isinstance(new_node, TreeNode):
                init_code = f"self.{new_node.get_name()} = " \
                            f"{new_node.symbol_tree.get_opt_cls_name()}(obj.{new_node.get_name()})"
            else:
                init_code = f"self.{new_node.get_name()} = obj.{new_node.get_name()}"
            init_ast = ast.parse(init_code).body[0]
            AstModifier.insert_assign_ast_to_function(stree.get_init_func_ast(), init_ast)
            # Insert ast_assign to bodies
            ast_base_node = base_node.get_ast() if base_node else None
            AstModifier.insert_assign_ast_to_bodies(self.ast_body, ast_assign, ast_base_node, before_node)

    def set_belong_symbol_tree(self, symbol_tree):
        """Set the symbol tree to which node belongs."""
        self._belong_tree = symbol_tree
        for node in self.nodes():
            node.set_belong_symbol_tree(symbol_tree)
