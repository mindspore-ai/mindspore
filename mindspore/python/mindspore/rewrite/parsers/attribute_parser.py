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
"""Parse ast.Assign in construct function to node of SymbolTree."""
import ast

from mindspore.rewrite.parsers.parser import Parser
from mindspore.rewrite.symbol_tree import SymbolTree
from mindspore.rewrite.parsers.parser_register import ParserRegister, reg_parser
from ..common import error_str
from ..node.node_manager import NodeManager

class AttributeParser(Parser):
    """Parse ast.Attribute in construct function to node of SymbolTree."""

    def target(self):
        """Parse target type."""
        return ast.Attribute

    def process(self, stree: SymbolTree, node: ast.Attribute, node_manager: NodeManager):
        """
        Parse ast.Attribute node.

        Args:
            stree ([SymbolTree]): Symbol Tree under parsing.
            node ([ast.Attribute]): An ast.Attribute node.
            node_manager (NodeManager): NodeManager those asts belong to.

        Returns:
            The value of node.

        Raises:
            TypeError: Attribute parser only supports parsing ast.Attribute type nodes.
        """
        if not isinstance(node, ast.Attribute):
            raise TypeError(error_str(f"Attribute parser only supports parsing ast.Attribute type nodes, but got "
                                      f"'{type(node).__name__}'", father_node=node))
        if not isinstance(node.value, (ast.Name, ast.Attribute)):
            raise RuntimeError(error_str(f"Attribute parser only supports (ast.Attribute, ast.Name) as value of "
                                         f"ast.Attribute, but got '{type(node).__name__}'", father_node=node))
        parser = ParserRegister.instance().get_parser(type(node.value))
        value = parser.process(stree, node.value, node_manager)

        return ".".join([value, node.attr])


g_attribute_parser = reg_parser(AttributeParser())
