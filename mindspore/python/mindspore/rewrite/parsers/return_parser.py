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

from . import Parser, reg_parser
from ..symbol_tree import SymbolTree
from ..node import Node, NodeManager
from ..ast_helpers import AstConverter


class ReturnParser(Parser):
    """Parse ast.Return output-node of SymbolTree."""

    def target(self):
        """Parse target type"""
        return ast.Return

    def process(self, stree: SymbolTree, node: ast.Return, node_manager: NodeManager):
        """Parse ast.Return to output-node of SymbolTree."""
        return_scoped_value = AstConverter.create_scopedvalue(node.value)
        node_return = Node.create_output_node(node, [return_scoped_value])
        stree.append_origin_field(node_return, node_manager)

g_return_parser = reg_parser(ReturnParser())
