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
from typing import Any
import os
import ast
import copy
import inspect
import astunparse

from mindspore import log as logger
from ..symbol_tree import SymbolTree
from ..parser import Parser
from ..parser_register import ParserRegister, reg_parser
from ..ast_helpers import AstFinder
from ..common import error_str


class ModuleParser(Parser):
    """Parse ast.Module to SymbolTrees."""
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
    def _get_import_node(ast_root):
        """Iterate over ast_root and return all ast.Import nodes or ast.ImportFrom nodes in ast_root."""
        import_nodes = []

        class GetImportNode(ast.NodeVisitor):
            """Find all import nodes from input ast node."""

            def visit_Import(self, node: ast.Import) -> Any:
                """Iterate over all nodes and save ast.Import nodes."""
                import_nodes.append(copy.deepcopy(node))
                return node

            def visit_ImportFrom(self, node: ast.ImportFrom) -> Any:
                """Iterate over all nodes and save ast.ImportFrom nodes."""
                import_nodes.append(copy.deepcopy(node))
                return node

            def get_node(self, input_ast):
                """Interface of GetImportNode."""
                self.generic_visit(input_ast)
                return True

        get_node_handler = GetImportNode()
        get_node_handler.get_node(ast_root)
        return import_nodes

    @staticmethod
    def _save_net_file_path(stree, level_num):
        """
        Save net file path into stree._import_asts. `level_num` is used when level exist in ast.ImportFrom.

        When level_num = 0(e.g. from xxx import yyy), current path will be saved.
        When level_num = 1(e.g. from .xxx import yyy), current path will be saved.
        When level_num = 2(e.g. from ..xxx import yyy), the path one level above the current path will be saved.
        """
        origin_net_file = inspect.getfile(type(stree.get_origin_network()))
        file_full_path = os.path.abspath(origin_net_file)
        file_path = os.path.dirname(file_full_path)
        if level_num > 1:
            for _ in range(level_num - 1):
                file_path = os.path.dirname(file_path)
        sys_path_append_ast = ast.parse(f"sys.path.append(r'{file_path}')").body[0]
        stree.get_import_asts().append(ast.Import([ast.alias(name='sys', asname=None)]))
        stree.get_import_asts().append(sys_path_append_ast)

    @staticmethod
    def _save_imports(stree):
        """Insert two groups of import nodes to ast.Module, common ones and those from class definition file."""
        stree.get_import_asts().append(ast.Import([ast.alias(name='mindspore', asname=None)]))
        stree.get_import_asts().append(ast.ImportFrom(module='mindspore', names=[ast.alias(name='nn', asname=None)],
                                                      level=0))
        stree.get_import_asts().append(ast.ImportFrom(module='mindspore.nn',
                                                      names=[ast.alias(name='Cell', asname=None)], level=0))
        stree.get_import_asts().append(ast.ImportFrom(module='mindspore.ops',
                                                      names=[ast.alias(name='functional', asname='F')], level=0))
        ModuleParser._save_net_file_path(stree, 0)
        origin_net = stree.get_origin_network()
        origin_net_source_code_file = inspect.getfile(type(origin_net))
        if not os.path.exists(origin_net_source_code_file):
            raise RuntimeError("For MindSpore Rewrite, in module parser, File ", origin_net_source_code_file,
                               " not exist")
        try:
            with open(origin_net_source_code_file, "r", encoding="utf-8") as f:
                source_code = f.read()
                import_nodes = ModuleParser._get_import_node(ast.parse(source_code))
        except RuntimeError as err:
            raise RuntimeError(f"For MindSpore Rewrite, in module parser, get import nodes error: {err}")
        if import_nodes:
            for import_node in import_nodes:
                if isinstance(import_node, ast.ImportFrom) and import_node.module:
                    if import_node.level > 1:
                        # For ImportFrom with dots(e.g. from ..file import abc), dot will be removed.
                        # The corresponding path will be saved into sys.path according to `import_node.level`.
                        ModuleParser._save_net_file_path(stree, import_node.level)
                    import_node.level = 0
                stree.get_import_asts().append(import_node)

    def target(self):
        """Parse target type"""
        return ast.Module

    def process(self, stree: SymbolTree, node: ast.Module):
        """Process ast.ClassDef nodes in ast.Module."""
        ModuleParser._save_imports(stree)
        class_ast = ModuleParser._find_class(node)
        stree.set_class_ast(class_ast)
        for body in node.body:
            if isinstance(body, ast.ClassDef):
                parser: Parser = ParserRegister.instance().get_parser(ast.ClassDef)
                parser.process(stree, body)
            else:
                logger.info(f"For MindSpore Rewrite, in module parser, Ignoring unsupported "
                            f"node({astunparse.unparse(body)}) in ast.Module.")

g_module_parser = reg_parser(ModuleParser())
