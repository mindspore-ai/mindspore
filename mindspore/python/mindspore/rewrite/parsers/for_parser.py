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
""" Parse ast.For node """
import ast

from . import Parser, ParserRegister, reg_parser
from ..node import NodeManager, ControlFlow
from ..symbol_tree import SymbolTree
from ..ast_helpers import AstConverter, AstFlattener


class ForParser(Parser):
    """ Class that implements parsing ast.For nodes """

    def target(self):
        """Parse target type"""
        return ast.For

    def process(self, stree: SymbolTree, node: ast.For, node_manager: NodeManager):
        """ Process ast.For node """
        # expand codes in ast.for
        ast_for = AstFlattener().transform_control_flow(node, stree)
        # parse ast codes of for branch into ControlFlow Node
        args = [AstConverter.create_scopedvalue(node.iter)]
        for_node = ControlFlow("for_node", ast_for, False, args, stree)
        for_node.loop_vars = AstConverter.get_ast_target_elems(node.target, True)
        stree.append_origin_field(for_node, node_manager)
        for_node.set_node_manager(node_manager)
        for body in ast_for.body:
            parser: Parser = ParserRegister.instance().get_parser(type(body))
            if parser is None:
                stree.append_python_node(ast_for, body, node_manager=for_node)
            else:
                parser.process(stree, body, node_manager=for_node)
        # parse ast codes of else branch into ControlFlow Node
        if ast_for.orelse:
            for_else_node = ControlFlow("for_else_node", ast_for, True, args, stree)
            for_else_node.loop_vars = AstConverter.get_ast_target_elems(node.target, True)
            stree.append_origin_field(for_else_node, node_manager)
            for body in ast_for.orelse:
                parser: Parser = ParserRegister.instance().get_parser(type(body))
                if parser is None:
                    stree.append_python_node(ast_for, body, node_manager=for_else_node)
                else:
                    parser.process(stree, body, node_manager=for_else_node)
            for_else_node.set_body_node(for_node)
            for_node.set_orelse_node(for_else_node)

g_for_parser = reg_parser(ForParser())
