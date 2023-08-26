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
from .parser import Parser
from .parser_register import ParserRegister, reg_parser
from ..ast_helpers import AstFinder
from ..common import error_str
from ..node.node_manager import NodeManager

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
    def save_file_path_to_sys(stree, level_num, file_path):
        """
        Save file path into stree._import_asts. `level_num` is used when level exist in ast.ImportFrom.

        When level_num = 0(e.g. from xxx import yyy), current path will be saved.
        When level_num = 1(e.g. from .xxx import yyy), current path will be saved.
        When level_num = 2(e.g. from ..xxx import yyy), the path one level above the current path will be saved.
        """
        file_path = os.path.dirname(os.path.abspath(file_path))
        if level_num > 1:
            for _ in range(level_num - 1):
                file_path = os.path.dirname(file_path)
        sys_path_append_ast = ast.parse(f"sys.path.insert(0, r'{file_path}')").body[0]
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
        origin_net = stree.get_origin_network()
        net_path = inspect.getfile(type(origin_net))
        ModuleParser.save_file_path_to_sys(stree, 0, net_path)
        ModuleParser.save_imports_from_file(stree, net_path)

    @staticmethod
    def get_valid_import_info(import_node, file_path):
        """Get valid import info while import_node.module is None"""
        # copy to a new node to avoid origin import_node being modified.
        import_node_test = copy.deepcopy(import_node)
        file_path = os.path.dirname(os.path.abspath(file_path))
        # get real path from import_node.level
        # from . import xxx: current path
        # from .. import xxx: last level path
        if import_node.level > 1:
            for _ in range(import_node.level - 1):
                file_path = os.path.dirname(file_path)
        file_path_tmp = file_path[:]
        max_level_count = file_path.count('/') + file_path.count('\\') - 1
        level_count = 0
        while level_count < max_level_count:
            file_path_tmp = os.path.dirname(file_path_tmp)
            import_node_test.module = file_path[len(file_path_tmp) + 1:].replace('/', '.')
            import_node_test.level = 0
            import_code = astunparse.unparse(import_node_test).strip()
            test_code = f"import sys\nsys.path.insert(0, r'{file_path_tmp}')\n{import_code}"
            try:
                exec(test_code) # pylint: disable=W0122
            except ValueError as e:
                # try upper level to avoid ValueError: attempted relative import beyond top-level package
                logger.warning(f"For MindSpore Rewrite, in module parser, test import code: "
                               f"{import_code} failed: {e}. Try upper level.")
                level_count += 1
                continue
            except Exception as e: # pylint: disable=W0703
                logger.error(f"For MindSpore Rewrite, in module parser, process import code: "
                             f"{import_code} failed: {e}. Ignore this import code.")
                return None, None
            else:
                # try test code success
                return import_node_test.module, file_path_tmp
        # try codes with all level failed
        logger.error(f"For MindSpore Rewrite, in module parser, test import code: "
                     f"{astunparse.unparse(import_node).strip()} failed. Ignore this import code.")
        return None, None

    @staticmethod
    def save_imports_from_file(stree, file_path):
        """Save imports from file"""
        if not os.path.exists(file_path):
            raise RuntimeError(f"For MindSpore Rewrite, in module parser, file {file_path} not exist.")
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                source_code = f.read()
                import_nodes = ModuleParser._get_import_node(ast.parse(source_code))
        except RuntimeError as err:
            raise RuntimeError(f"For MindSpore Rewrite, in module parser, get import nodes error: {err}")
        if not import_nodes:
            return
        for import_node in import_nodes:
            if isinstance(import_node, ast.ImportFrom):
                if import_node.module is None:
                    import_module, import_path = ModuleParser.get_valid_import_info(import_node, file_path)
                    if not import_module:
                        continue
                    ModuleParser.save_file_path_to_sys(stree, 0, import_path)
                    import_node.module = import_module
                elif import_node.level > 1:
                    # For ImportFrom with dots(e.g. from ..file import abc), dot will be removed.
                    # The corresponding path will be saved into sys.path according to `import_node.level`.
                    ModuleParser.save_file_path_to_sys(stree, import_node.level, file_path)
                import_node.level = 0
            stree.get_import_asts().append(import_node)

    def target(self):
        """Parse target type"""
        return ast.Module

    def process(self, stree: SymbolTree, node: ast.Module, node_manager: NodeManager):
        """Process ast.ClassDef nodes in ast.Module."""
        ModuleParser._save_imports(stree)
        class_ast = ModuleParser._find_class(node)
        stree.set_class_ast(class_ast)
        for body in node.body:
            if isinstance(body, ast.ClassDef):
                parser: Parser = ParserRegister.instance().get_parser(ast.ClassDef)
                parser.process(stree, body, stree)
            else:
                logger.info(f"For MindSpore Rewrite, in module parser, Ignoring unsupported "
                            f"node({astunparse.unparse(body)}) in ast.Module.")

g_module_parser = reg_parser(ModuleParser())
