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
from . import Parser, ParserRegister, reg_parser
from ..node import NodeManager, ControlFlow
from ..symbol_tree import SymbolTree
from ..ast_helpers import AstFinder, AstConverter, AstFlattener


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
        ast_if = AstFlattener().transform_control_flow(node, stree)
        # parse ast codes of if branch into ControlFlow Node
        args = [AstConverter.create_scopedvalue(node.test)]
        if_node = ControlFlow("if_node", ast_if, False, args, stree)
        stree.append_origin_field(if_node, node_manager)
        for body in ast_if.body:
            parser: Parser = ParserRegister.instance().get_parser(type(body))
            if parser is None:
                stree.append_python_node(ast_if, body, node_manager=if_node)
            else:
                parser.process(stree, body, node_manager=if_node)
        # parse ast codes of else branch into ControlFlow Node
        else_node = None
        if ast_if.orelse:
            else_node = ControlFlow("else_node", ast_if, True, args, stree)
            stree.append_origin_field(else_node, node_manager)
            for body in ast_if.orelse:
                parser: Parser = ParserRegister.instance().get_parser(type(body))
                if parser is None:
                    stree.append_python_node(ast_if, body, node_manager=else_node)
                else:
                    parser.process(stree, body, node_manager=else_node)
            else_node.set_body_node(if_node)
            if_node.set_orelse_node(else_node)
        # record eval result of ast.If's test
        if ast_if in AstFlattener.ast_if_test_cache:
            origin_test_ast = AstFlattener.ast_if_test_cache[ast_if]
            # replace self.xxx to self._origin_network.xxx
            ast_attributes = AstFinder(origin_test_ast).find_all(ast.Attribute)
            for ast_attr in ast_attributes:
                if isinstance(ast_attr.value, ast.Name) and ast_attr.value.id == 'self':
                    new_ast_attr = ast.Attribute(value=ast.Name(id='self', ctx=ast.Load()),
                                                 attr='_origin_network', ctx=ast.Load())
                    ast_attr.value = new_ast_attr
            # get result of ast.If's test
            eval_success, eval_result = stree.eval_ast_result(origin_test_ast)
            if eval_success:
                if_node.test_result = eval_result
                if else_node:
                    else_node.test_result = eval_result

g_if_parser = reg_parser(IfParser())
