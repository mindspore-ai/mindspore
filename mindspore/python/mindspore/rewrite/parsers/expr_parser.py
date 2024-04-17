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
""" Parse ast.Expr node """
import ast
import sys

from mindspore import log as logger
from . import Parser, reg_parser, AssignParser
from ..node import NodeManager
from ..symbol_tree import SymbolTree
if sys.version_info >= (3, 9):
    import ast as astunparse # pylint: disable=reimported, ungrouped-imports
else:
    import astunparse


class ExprParser(Parser):
    """ Class that implements parsing ast.Expr nodes """

    def target(self):
        """Parse target type"""
        return ast.Expr

    def process(self, stree: SymbolTree, node: ast.Expr, node_manager: NodeManager):
        """ Process ast.Expr node """
        # ast.Expr is not supported to be parsed.
        stree.append_python_node(node, node, node_manager)
        # when value of ast.Expr is ast.Call, add imports to make sure that the
        # function object can be accessed.
        if isinstance(node.value, ast.Call):
            ast_call = node.value
            func_full_name = astunparse.unparse(ast_call.func).strip()
            func_name = func_full_name.split('.')[0]
            module, _ = AssignParser._get_module_of_node_manager(node_manager) # pylint: disable=protected-access
            if module is None:
                logger.warning(f"Cannot get module where '{func_full_name}' is located.")
                return
            top_node_manager = node_manager.get_top_manager()
            belonging_ast = None if isinstance(top_node_manager, SymbolTree) else top_node_manager.get_manager_ast()
            stree.add_import(module, func_name, belonging_ast)


g_expr_parser = reg_parser(ExprParser())
