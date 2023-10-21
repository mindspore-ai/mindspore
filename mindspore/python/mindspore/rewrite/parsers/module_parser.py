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
import copy
import inspect

from mindspore import log as logger
from ..symbol_tree import SymbolTree
from .parser import Parser
from .parser_register import ParserRegister, reg_parser
from ..ast_helpers import AstFinder
from ..common import error_str
from ..node.node_manager import NodeManager

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
    def _get_import_node(ast_root):
        """Iterate over ast_root and return all ast.Import nodes or ast.ImportFrom nodes in ast_root."""
        import_nodes = []
        try_nodes = []
        imports_str = []

        class GetImportNode(ast.NodeVisitor):
            """Find all import nodes from input ast node."""

            def visit_Try(self, node: ast.Try) -> Any:
                if isinstance(node.body[0], (ast.Import, ast.ImportFrom)):
                    try_nodes.append(copy.deepcopy(node))
                return node

            def visit_Import(self, node: ast.Import) -> Any:
                """Iterate over all nodes and save ast.Import nodes."""
                import_nodes.append(copy.deepcopy(node))
                imports_str.append(astunparse.unparse(node))
                return node

            def visit_ImportFrom(self, node: ast.ImportFrom) -> Any:
                """Iterate over all nodes and save ast.ImportFrom nodes."""
                import_nodes.append(copy.deepcopy(node))
                imports_str.append(astunparse.unparse(node))
                return node

            def get_node(self, input_ast):
                """Interface of GetImportNode."""
                self.generic_visit(input_ast)
                return True

        def _remove_duplicated_import_in_try(node: [ast.Import, ast.ImportFrom]):
            import_str = astunparse.unparse(node)
            if import_str in imports_str:
                import_nodes.remove(import_nodes[imports_str.index(import_str)])

        get_node_handler = GetImportNode()
        get_node_handler.get_node(ast_root)
        for Try in try_nodes:
            for body in Try.body:
                _remove_duplicated_import_in_try(body)
            for handler in Try.handlers:
                for body in handler.body:
                    _remove_duplicated_import_in_try(body)
        import_nodes.extend(try_nodes)
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
        """Get valid import info while import_node.module is at form of relative path"""
        # copy to a new node to avoid origin import_node being modified.
        import_node_test = copy.deepcopy(import_node)
        file_path = os.path.dirname(os.path.abspath(file_path))
        # get real path from import_node.level
        # from .(A) import xxx: current path
        # from ..(A) import xxx: last level path
        import_node_module_name = import_node.module
        level = import_node.level
        # from A import xxx: it does not need to pad, directly return the module name
        if level == 0:
            return import_node_module_name, None
        if level > 1:
            for _ in range(level - 1):
                file_path = os.path.dirname(file_path)
        file_path_tmp = file_path[:]
        max_level_count = file_path.count('/') + file_path.count('\\') - 1
        level_count = 0
        # suffix is the module_name, e.g. 'A' in 'from ..(A) import xxx'
        suffix = ''
        if import_node_module_name:
            suffix = '.' + import_node_module_name
        while level_count < max_level_count:
            file_path_tmp = os.path.dirname(file_path_tmp)
            import_node_test.module = file_path[len(file_path_tmp) + 1:].replace('/', '.') + suffix
            import_node_test.level = 0
            import_code = astunparse.unparse(import_node_test).strip()
            test_code = f"import sys\nsys.path.insert(0, r'{file_path_tmp}')\n{import_code}"
            try:
                exec(test_code) # pylint: disable=W0122
            except (ValueError, ImportError) as e:
                # try upper level to avoid ValueError: attempted relative import beyond top-level package
                # this exception is changed to ImportError after python3.9
                logger.info(f"For MindSpore Rewrite, in module parser, test import code: "
                            f"{import_code} failed: {e}. Try upper level.")
                level_count += 1
                continue
            except Exception as e: # pylint: disable=W0703
                logger.warning(f"For MindSpore Rewrite, in module parser, process import code: "
                               f"{import_code} failed: {e}. Ignore this import code.")
                return None, None
            else:
                # try test code success
                return import_node_test.module, file_path_tmp
        # try codes with all level failed
        logger.warning(f"For MindSpore Rewrite, in module parser, test import code: "
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
            import_node = ModuleParser._process_relative_import(stree, import_node, file_path)
            if import_node:
                stree.get_import_asts().append(import_node)

    @staticmethod
    def _process_relative_import(stree, import_node, file_path):
        """Process relative imports"""
        if isinstance(import_node, ast.ImportFrom):
            # pad the ImportFrom with parent path
            # e.g. from ..C import xxx -> from A.B.C import xxx
            import_module, import_path = ModuleParser.get_valid_import_info(import_node, file_path)
            if import_path:
                ModuleParser.save_file_path_to_sys(stree, 0, import_path)
            module_name_list = [alias.name.strip() for alias in import_node.names]
            # add the module into _imported_modules to direct the class
            stree.save_imported_modules(file_path, import_module, module_name_list)
            import_node = ast.ImportFrom(module=import_module, names=import_node.names, level=0)
        elif isinstance(import_node, ast.Import):
            for alias in import_node.names:
                name = alias.name
                stree.save_imported_modules(file_path, name.strip(), [])
        return import_node

    @staticmethod
    def _add_decorator_to_class(class_ast: ast.ClassDef, origin_net):
        """Add decorators to class"""
        origin_net_source_code_file = inspect.getfile(type(origin_net))
        if not os.path.exists(origin_net_source_code_file):
            raise RuntimeError("For MindSpore Rewrite, in module parser, File ", origin_net_source_code_file,
                               " not exist")
        try:
            with open(origin_net_source_code_file, "r", encoding="utf-8") as f:
                source_code = f.read()
                decorators = ModuleParser._get_decorator(ast.parse(source_code), origin_net)
        except RuntimeError:
            raise RuntimeError("For MindSpore Rewrite, in module parser, get decorators error")
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
        ModuleParser._save_imports(stree)
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
