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
        src_bodies = None
        dst_bodies = None
        test_value = True
        try:
            test_value = eval(test_code)
        except Exception:
            raise NotImplementedError("Only support ast.If whose test can be eval, got:", test_code)

        bodies = node.body if test_value else node.orelse
        for body in bodies:
            parser: Parser = ParserRegister.instance().get_parser(type(body))
            if parser is None:
                stree.append_python_node(node, body)
            else:
                parser.process(stree, body)

        # hardcode for if, ME need both branch of ast.If has same output
        src_bodies = node.body if test_value else node.orelse
        dst_bodies = node.orelse if test_value else node.body
        dst_bodies.clear()
        if src_bodies:
            for ast_node in src_bodies:
                if not isinstance(ast_node, ast.Assign):
                    continue
                targets = ast_node.targets
                for target in targets:
                    dst_bodies.append(ast.Assign(targets=[target], value=ast.Constant(value=0, kind=None,
                                                                                      ctx=ast.Load())))
        else:
            dst_bodies.append(ast.Pass())


g_if_parser = reg_parser(IfParser())
