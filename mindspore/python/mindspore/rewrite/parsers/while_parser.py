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
""" Parse ast.While node """
import ast

from . import Parser, ParserRegister, reg_parser
from ..node import NodeManager, ControlFlow
from ..symbol_tree import SymbolTree
from ..ast_helpers import AstConverter, AstFlattener


class WhileParser(Parser):
    """ Class that implements parsing ast.While nodes """

    def target(self):
        """Parse target type"""
        return ast.While

    def process(self, stree: SymbolTree, node: ast.While, node_manager: NodeManager):
        """ Process ast.While node """
        # expand codes in ast.While
        ast_while = AstFlattener().transform_control_flow(node, stree)
        # parse ast codes of for branch into ControlFlow Node
        args = [AstConverter.create_scopedvalue(node.test)]
        while_node = ControlFlow("while_node", ast_while, False, args, stree)
        stree.append_origin_field(while_node, node_manager)
        while_node.set_node_manager(node_manager)
        for body in ast_while.body:
            parser: Parser = ParserRegister.instance().get_parser(type(body))
            if parser is None:
                stree.append_python_node(ast_while, body, node_manager=while_node)
            else:
                parser.process(stree, body, node_manager=while_node)
        # parse ast codes of else branch into ControlFlow Node
        if ast_while.orelse:
            while_else_node = ControlFlow("while_else_node", ast_while, True, args, stree)
            stree.append_origin_field(while_else_node, node_manager)
            for body in ast_while.orelse:
                parser: Parser = ParserRegister.instance().get_parser(type(body))
                if parser is None:
                    stree.append_python_node(ast_while, body, node_manager=while_else_node)
                else:
                    parser.process(stree, body, node_manager=while_else_node)
            while_else_node.set_body_node(while_node)
            while_node.set_orelse_node(while_else_node)

g_while_parser = reg_parser(WhileParser())
