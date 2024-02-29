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
"""Parse ast.Module to SymbolTrees."""
import sys
from typing import Any
import os
import ast
import inspect

from mindspore import log as logger
from . import Parser, ParserRegister, reg_parser
from ..node import NodeManager
from ..symbol_tree import SymbolTree
from ..ast_helpers import AstFinder
from ..common.error_log import error_str

if sys.version_info >= (3, 9):
    import ast as astunparse # pylint: disable=reimported, ungrouped-imports
else:
    import astunparse

class ModuleParser(Parser):
    """Parse ast.Module to SymbolTrees."""

    # a denied_class_decorator_list represents the decorators should be banned, which is registered by user
    denied_class_decorator_list = []

    @staticmethod
    def _find_class(ast_node: ast.Module) -> ast.ClassDef:
        """Find all ast.ClassDef in ast.Module, only support one ast.ClassDef in ast.Module now."""
        visitor = AstFinder(ast_node)
        classes = visitor.find_all(ast.ClassDef)
        if not classes:
            raise RuntimeError(error_str("no class in module.", father_node=ast_node))
        if len(classes) > 1:
            raise RuntimeError(error_str("multi-class in module is not supported now", father_node=ast_node))
        return classes[0]

    @staticmethod
    def _add_decorator_to_class(class_ast: ast.ClassDef, origin_net):
        """Add decorators to class"""
        origin_net_source_code_file = inspect.getfile(type(origin_net))
        if not os.path.exists(origin_net_source_code_file):
            raise FileNotFoundError(f"For MindSpore Rewrite, in module parser, File {origin_net_source_code_file} "
                                    f"not exist")
        with open(origin_net_source_code_file, "r", encoding="utf-8") as f:
            source_code = f.read()
            decorators = ModuleParser._get_decorator(ast.parse(source_code), origin_net)
        if decorators:
            for decorator_index, decorator_node in enumerate(decorators):
                class_ast.decorator_list.insert(decorator_index, decorator_node)
        ast.fix_missing_locations(class_ast)

    @staticmethod
    def _get_decorator(ast_root, origin_net):
        """Get the decorators of function"""
        net_name = type(origin_net).__name__
        decorators = []

        class GetClassNode(ast.NodeVisitor):
            """Find the class node from input ast node."""
            def visit_ClassDef(self, node: ast.ClassDef) -> Any:
                """Visit the class node and add the decorators to class node"""
                if node.name == net_name:
                    for decorator in node.decorator_list[:]:
                        decorator_name = ""
                        if isinstance(decorator, ast.Call):
                            func = decorator.func
                            if isinstance(func, ast.Name):
                                decorator_name = func.id
                        elif isinstance(decorator, ast.Name):
                            decorator_name = decorator.id
                        # User should set the denied class_decorator,
                        # because the symbol_tree cant pass the correct parameters to decorators but the instance "obj".
                        if decorator_name not in ModuleParser.denied_class_decorator_list:
                            decorators.append(decorator)
                return node

            def get_node(self, input_ast):
                """Interface of GetClassNode."""
                self.generic_visit(input_ast)
                return True

        get_node_handler = GetClassNode()
        get_node_handler.get_node(ast_root)
        return decorators

    def target(self):
        """Parse target type"""
        return ast.Module

    def process(self, stree: SymbolTree, node: ast.Module, node_manager: NodeManager):
        """Process ast.ClassDef nodes in ast.Module."""
        class_ast = ModuleParser._find_class(node)
        ModuleParser._add_decorator_to_class(class_ast, stree.get_origin_network())
        stree.set_class_ast(class_ast)
        for body in node.body:
            if isinstance(body, ast.ClassDef):
                parser: Parser = ParserRegister.instance().get_parser(ast.ClassDef)
                parser.process(stree, body, stree)
            else:
                logger.info(f"For MindSpore Rewrite, in module parser, Ignoring unsupported "
                            f"node({astunparse.unparse(body)}) in ast.Module.")

g_module_parser = reg_parser(ModuleParser())
