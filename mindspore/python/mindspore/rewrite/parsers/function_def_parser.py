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
"""Parse bodies of ast.FunctionDef which is construct function to nodes of SymbolTree."""
import ast
from mindspore import log as logger
from .parser_register import ParserRegister, reg_parser
from .parser import Parser
from ..symbol_tree import SymbolTree
from ..api.node_type import NodeType
from ..node.node_manager import NodeManager


class FunctionDefParser(Parser):
    """Parse bodies of ast.FunctionDef in SymbolTree."""

    def target(self):
        """Parse target type"""
        return ast.FunctionDef

    def remove_dead_code(self, node_manager: NodeManager):
        """Remove dead codes"""
        # Find out return node position
        return_idx = -1
        for idx, node in enumerate(node_manager.nodes()):
            if node.get_node_type() == NodeType.Output:
                return_idx = idx
                break
        if return_idx == -1:
            return
        # Remove nodes after return node.
        # Reverse traversal to ensure that nodes are orphaned and can be deleted.
        for idx, node in reversed(list(enumerate(node_manager.nodes()))):
            if idx <= return_idx:
                break
            logger.info(f"Remove dead code node:{node.get_name()}")
            node_manager.erase_node(node)

    def process(self, stree: SymbolTree, ast_node: ast.FunctionDef, node_manager: NodeManager):
        """
        Parse bodies of ast.FunctionDef in SymbolTree.

        Args:
            stree (SymbolTree): symbol tree under parsing.
            ast_node (ast.FunctionDef): Ast FunctionDef node in construct.
            node_manager (NodeManager): NodeManager those asts belong to.
        """
        # parse args as inputs of stree
        arguments: ast.arguments = ast_node.args
        parser: Parser = ParserRegister.instance().get_parser(ast.arguments)
        parser.process(stree, arguments, node_manager)

        # parse body as node of stree
        for body in ast_node.body:
            # avoid add dead code, so we need to break if return is added.
            parser: Parser = ParserRegister.instance().get_parser(type(body))
            if parser is None:
                stree.append_python_node(ast_node, body, node_manager)
            else:
                parser.process(stree, body, node_manager)

        self.remove_dead_code(node_manager)


g_functiondef_parser = reg_parser(FunctionDefParser())
