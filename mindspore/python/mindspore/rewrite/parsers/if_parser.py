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
"""Parse ast.If in construct function to node of SymbolTree."""

import ast
import astunparse

from ..symbol_tree import SymbolTree
from .parser import Parser
from .parser_register import reg_parser
from ..node.node_manager import NodeManager


class IfParser(Parser):
    """Parse ast.If in construct function to node of SymbolTree."""

    def target(self):
        """Parse target type"""
        return ast.If

    def process(self, stree: SymbolTree, node: ast.If, node_manager: NodeManager):
        """
        Parse ast.If and create a node in symbol tree.

        Args:
            stree ([SymbolTree]): Symbol Tree under parsing.
            node ([ast.If]): An ast.If node.
            node_manager (NodeManager): NodeManager those asts belong to.

        Raises:
            NotImplementedError: If test of ast.If can not be eval.
        """

        test_code = astunparse.unparse(node.test)
        test_code = test_code.replace("self", "stree.get_origin_network()")
        bodies = None
        try:
            test_value = eval(test_code)
        except (NameError, TypeError):
            stree.try_append_python_node(node, node, node_manager)
            return

        bodies = node.body if test_value else node.orelse
        ast_functiondef = node_manager.get_ast_functiondef()
        if not ast_functiondef:
            raise RuntimeError(f"ast_functiondef is None in node_manager {node_manager.get_manager_name()} "
                               "when parsing 'If' statement.")
        index = ast_functiondef.body.index(node) + 1
        info_node = ast.Name(id=f"# If node has been replaced by {bool(test_value)} branch.",
                             lineno=0, col_offset=0, ctx=ast.Load)
        exp_node = ast.Expr(value=info_node, lineno=0, col_offset=0, ctx=ast.Load)
        ast_functiondef.body.insert(index-1, exp_node)
        for body in bodies:
            ast_functiondef.body.insert(index, body)
            index += 1
        ast_functiondef.body.remove(node)
g_if_parser = reg_parser(IfParser())
