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

from ..symbol_tree import SymbolTree
from .parser import Parser
from .parser_register import ParserRegister, reg_parser
from ..node import NodeManager, ControlFlow
from ..ast_transformers.flatten_recursive_stmt import FlattenRecursiveStmt


class IfParser(Parser):
    """Parse ast.If in construct function to node of SymbolTree."""

    def target(self):
        """Parse target type"""
        return ast.If

    def process(self, stree: SymbolTree, node: ast.If, node_manager: NodeManager):
        """
        Parse ast.If and create nodes into symbol tree.

        Args:
            stree ([SymbolTree]): Symbol Tree under parsing.
            node ([ast.If]): An ast.If node.
            node_manager (NodeManager): NodeManager those asts belong to.

        Raises:
            NotImplementedError: If test of ast.If can not be eval.
        """
        # expand codes in ast.if
        ast_if = FlattenRecursiveStmt().transform_if(node, stree)
        # parse ast codes of if branch into ControlFlow Node
        if_node = ControlFlow("if_node", ast_if.body, stree)
        for body in ast_if.body:
            parser: Parser = ParserRegister.instance().get_parser(type(body))
            if parser is None:
                stree.append_python_node(ast_if, body, node_manager=if_node)
            else:
                parser.process(stree, body, node_manager=if_node)
        stree.append_origin_field(if_node, node_manager)
        # parse ast codes of else branch into ControlFlow Node
        if ast_if.orelse:
            else_node = ControlFlow("else_node", ast_if.orelse, stree)
            for body in ast_if.orelse:
                parser: Parser = ParserRegister.instance().get_parser(type(body))
                if parser is None:
                    stree.append_python_node(ast_if, body, node_manager=else_node)
                else:
                    parser.process(stree, body, node_manager=else_node)
            stree.append_origin_field(else_node, node_manager)

g_if_parser = reg_parser(IfParser())
