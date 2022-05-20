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
from ..parser import Parser
from ..parser_register import ParserRegister, reg_parser


class IfParser(Parser):
    """Parse ast.If in construct function to node of SymbolTree."""

    def target(self):
        """Parse target type"""
        return ast.If

    def process(self, stree: SymbolTree, node: ast.If):
        """
        Parse ast.If and create a node in symbol tree.

        Args:
            stree ([SymbolTree]): Symbol Tree under parsing.
            node ([ast.If]): An ast.If node.

        Raises:
            NotImplementedError: If test of ast.If can not be eval.
        """

        test_code = astunparse.unparse(node.test)
        test_code = test_code.replace("self", "stree.get_origin_network()")
        bodies = None
        try:
            test_value = eval(test_code)
            if test_value:
                bodies = node.body
            else:
                bodies = node.orelse
        except Exception:
            raise NotImplementedError("Only support ast.If whose test can be eval, got:", test_code)

        for body in bodies:
            parser: Parser = ParserRegister.instance().get_parser(type(body))
            if parser is None:
                stree.append_python_node(node, body)
            else:
                parser.process(stree, body)


g_if_parser = reg_parser(IfParser())
