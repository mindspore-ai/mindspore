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
"""CallFunction Node."""
import ast
from .node import Node
from .node_manager import NodeManager
from ..api.scoped_value import ScopedValue
from ..api.node_type import NodeType
from ..ast_helpers import AstModifier


class CallFunction(Node, NodeManager):
    """CallFunction is used for class internal function."""
    def __init__(self, targets: [ScopedValue], func_name: ScopedValue, args: [ScopedValue],
                 kwargs: {str: ScopedValue}, node_name: str, ast_node: ast.AST, ast_functiondef: ast.FunctionDef,
                 stree, instance):
        """
        Constructor of CallFunction.

        Args:
            targets (list[ScopedValue]): A list of instance of ScopedValue. See detail in docstring of Node class.
            args (list[ScopedValue]): A list of instance of ScopedValue. See detail in docstring of Node class.
            kwargs (dict{str: ScopedValue}): A list of instance of ScopedValue. See detail in docstring of Node class.
            func_name ([ScopedValue, optional]): An instance of ScopedValue. See detail in docstring of Node class.
            node_name (str): A string represents name of node. Name of node will be unique when inserted into
                SymbolTree. Name of node also used as field name in network class.
            ast_node (ast.AST): An instance of ast.AST represents corresponding node in ast.
            ast_functiondef (ast.FunctionDef): An instance of ast.FunctionDef represents corresponding function
                definition in ast.
            stree (SymbolTree): Symbol tree used to get node_namer.
            instance: Object in network corresponding to this node.
        """
        if isinstance(func_name, str):
            func_name = ScopedValue.create_naming_value(func_name)
        Node.__init__(self, NodeType.CallFunction, ast_node, targets, func_name, args, kwargs, node_name, instance)
        NodeManager.__init__(self, stree.get_node_namer())
        NodeManager.set_ast_functiondef(self, ast_functiondef)
        NodeManager.set_manager_name(self, func_name.value)

    def erase_node(self, node):
        """Erase node from CallFunction."""
        NodeManager.erase_node(self, node)
        # erase asts
        ret = AstModifier.erase_ast_from_function(self.get_ast_functiondef(), node.get_ast())
        if not ret:
            raise ValueError(f"erase node failed, node {node.get_name()} not in function ast tree.")

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
            stree.insert_to_ast_while_insert_node(new_node, base_node, before_node, self)

    def set_belong_symbol_tree(self, symbol_tree):
        """Set the symbol tree to which node belongs."""
        self._belong_tree = symbol_tree
        for node in self.nodes():
            node.set_belong_symbol_tree(symbol_tree)
