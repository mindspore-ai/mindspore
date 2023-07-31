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
import ast

from ..symbol_tree import SymbolTree
from ..node.node import Node
from ..node.node_manager import NodeManager
from .parser import Parser
from .parser_register import reg_parser
from ..common import error_str


class ReturnParser(Parser):
    """Parse ast.Return output-node of SymbolTree."""

    def target(self):
        """Parse target type"""
        return ast.Return

    def process(self, stree: SymbolTree, node: ast.Return, node_manager: NodeManager):
        """Parse ast.Return to output-node of SymbolTree."""
        return_value = node.value
        if not isinstance(return_value, ast.Name):
            raise RuntimeError(error_str(f"only support ast.Name as return value, but got ast type "
                                         f"'{type(return_value).__name__}'", father_node=node, child_node=return_value))
        node_return = Node.create_output_node(node, [return_value.id])
        stree.append_origin_field(node_return, node_manager)

g_return_parser = reg_parser(ReturnParser())
