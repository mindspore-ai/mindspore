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
"""Parse ast.Return output-node of SymbolTree."""
from __future__ import absolute_import
import ast

from ..symbol_tree import SymbolTree
from ..node import Node
from ..parser import Parser
from ..parser_register import reg_parser


class ReturnParser(Parser):
    """Parse ast.Return output-node of SymbolTree."""

    def target(self):
        """Parse target type"""
        return ast.Return

    def process(self, stree: SymbolTree, node: ast.Return):
        """Parse ast.Return to output-node of SymbolTree."""
        return_value = node.value
        if not isinstance(return_value, ast.Name):
            raise RuntimeError("Only ast.Name as return value")
        node_return = Node.create_output_node(node, [return_value.id])
        stree.append_origin_field(node_return)


g_return_parser = reg_parser(ReturnParser())
