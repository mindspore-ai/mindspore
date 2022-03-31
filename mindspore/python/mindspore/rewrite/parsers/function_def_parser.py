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

from ..parser_register import ParserRegister, reg_parser
from ..parser import Parser
from ..symbol_tree import SymbolTree


class FunctionDefParser(Parser):
    """Parse bodies of ast.FunctionDef which is construct function to nodes of SymbolTree."""

    def target(self):
        """Parse target type"""
        return ast.FunctionDef

    def process(self, stree: SymbolTree, node: ast.FunctionDef):
        """Parse bodies of ast.FunctionDef which is construct function to nodes of SymbolTree."""
        stree.set_ast_root(node)
        # parse args as inputs of stree
        arguments: ast.arguments = node.args
        parser: Parser = ParserRegister.instance().get_parser(ast.arguments)
        parser.process(stree, arguments)

        # parse body as node of stree
        for body in node.body:
            # avoid add dead code, so we need to break if return is added.
            parser: Parser = ParserRegister.instance().get_parser(type(body))
            if parser is None:
                stree.append_python_node(node, body)
            else:
                parser.process(stree, body)

        if hasattr(node, "decorator_list"):
            stree.try_append_python_node(node, node.decorator_list)
        if hasattr(node, "returns"):
            stree.try_append_python_node(node, node.returns)


g_functiondef_parser = reg_parser(FunctionDefParser())
