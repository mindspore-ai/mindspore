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
"""Parse ast.arguments to input-node of SymbolTree."""
import ast

from .parser import Parser
from .parser_register import reg_parser
from ..symbol_tree import SymbolTree
from ..common import error_str
from ..node.node_manager import NodeManager


class ArgumentsParser(Parser):
    """Parse ast.arguments to input-node of SymbolTree."""

    def target(self):
        """Parse target type"""
        return ast.arguments

    def process(self, stree: SymbolTree, ast_node: ast.arguments, node_manager: NodeManager):
        """
        Parse ast.arguments and create input-node to stree.

        Args:
            stree (SymbolTree): symbol tree under parsing.
            ast_node (ast.arguments): ast argument node in construct.
            node_manager (NodeManager): NodeManager those asts belong to.

        Raises:
            RuntimeError: Types of ast_node.args elements are not ast.arg.
        """
        if hasattr(ast_node, "posonlyargs"):
            stree.try_append_python_node(ast_node, ast_node.posonlyargs, node_manager)

        for arg in ast_node.args:
            if not isinstance(arg, ast.arg):
                raise RuntimeError(error_str(f"only support ast.arg in arguments arg, but got '{type(arg).__name__}'",
                                             arg, ast_node))
            stree.append_input_node(arg, arg.arg, node_manager=node_manager)
        if hasattr(ast_node, "vararg"):
            stree.try_append_python_node(ast_node, ast_node.vararg, node_manager)
        if hasattr(ast_node, "kwonlyargs"):
            stree.try_append_python_node(ast_node, ast_node.kwonlyargs, node_manager)
        if hasattr(ast_node, "kw_defaults"):
            stree.try_append_python_node(ast_node, ast_node.kw_defaults, node_manager)
        if hasattr(ast_node, "kwarg"):
            stree.try_append_python_node(ast_node, ast_node.kwarg, node_manager)
        if hasattr(ast_node, "defaults"):
            stree.try_append_python_node(ast_node, ast_node.defaults, node_manager)


g_arguments_parser = reg_parser(ArgumentsParser())
