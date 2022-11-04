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
"""Parse ast.arguments to input-node of SymbolTree."""
import ast

from ..parser import Parser
from ..parser_register import reg_parser
from ..symbol_tree import SymbolTree
from ..common import error_str


class ArgumentsParser(Parser):
    """Parse ast.arguments to input-node of SymbolTree."""

    def target(self):
        """Parse target type"""
        return ast.arguments

    def process(self, stree: SymbolTree, node: ast.arguments):
        """
        Parse ast.arguments and create input-node to stree.

        Args:
            stree (SymbolTree): symbol tree under parsing.
            node (ast.arguments): argument node in construct.

        Raises:
            RuntimeError: Types of node.args elements are not ast.arg.
        """
        if hasattr(node, "posonlyargs"):
            stree.try_append_python_node(node, node.posonlyargs)

        for arg in node.args:
            if not isinstance(arg, ast.arg):
                raise RuntimeError(error_str(f"only support ast.arg in arguments arg, but got '{type(arg).__name__}'",
                                             child_node=arg, father_node=node))
            stree.append_input_node(arg, arg.arg)

        if hasattr(node, "vararg"):
            stree.try_append_python_node(node, node.vararg)
        if hasattr(node, "kwonlyargs"):
            stree.try_append_python_node(node, node.kwonlyargs)
        if hasattr(node, "kw_defaults"):
            stree.try_append_python_node(node, node.kw_defaults)
        if hasattr(node, "kwarg"):
            stree.try_append_python_node(node, node.kwarg)
        if hasattr(node, "defaults"):
            stree.try_append_python_node(node, node.defaults)


g_arguments_parser = reg_parser(ArgumentsParser())
