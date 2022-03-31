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
"""SymbolTree builder."""
from copy import copy
from typing import Optional
import ast
import inspect

from mindspore.nn import Cell
from .symbol_tree import SymbolTree
from .node import TreeNode
from .parser_register import ParserRegister
from .parser import Parser
from .ast_transformers import FlattenRecursiveStmt
from .ast_helpers import AstModifier
from .ast_helpers import AstFinder


class SymbolTreeBuilder:
    """
    `SymbolTreeBuilder` for building a SymbolTree from network.

    Args:
         network (Cell): An instance of Cell represents a network from which SymbolTree will be built.
    """

    def __init__(self, network: Cell):
        if not isinstance(network, Cell):
            raise RuntimeError("Only support network with Cell type now")
        self._origin_net = network
        network_str = inspect.getsource(type(network))
        self._ast_root: ast.Module = ast.parse(network_str)
        self._root_tree: Optional[SymbolTree] = None

    @staticmethod
    def _ast_transform(ast_root: ast.AST) -> ast.AST:
        """
        Optimize ast before parse.

        Args:
             ast_root (ast.AST): An instance of ast to be optimized.

        Returns:
             An instance of ast been optimized.
        """
        transform_list = [FlattenRecursiveStmt()]
        for transformer in transform_list:
            ast_root = transformer.transform(ast_root)
        return ast_root

    @staticmethod
    def _merge_import_of_module(main_mod: ast.Module, sub_mod: ast.Module):
        """
        Merge imports of ast.Module of sub-network to ast.Module of main-network.

        Note:
            Imports of sub_module would be added ahead of imports in main_module.

            Error will occur if import name is over-load because of alise.

        Args:
             main_mod (ast.Module): An ast.Module corresponding to main-network.
             sub_mod (ast.Module): An ast.Module corresponding to sub-network.

        """

        sub_mod_finder = AstFinder(sub_mod)
        main_mod_finder = AstFinder(main_mod)
        imports_in_sub = copy(sub_mod_finder.find_all((ast.Import, ast.ImportFrom)))
        imports_in_main = copy(main_mod_finder.find_all((ast.Import, ast.ImportFrom)))
        assert imports_in_main
        first_import = imports_in_main[0]
        for clazz in imports_in_sub:
            AstModifier.insert_sub_ast(main_mod, clazz, first_import, True)

    @staticmethod
    def _merge_class_of_module(main_mod: ast.Module, sub_mod: ast.Module):
        """
        Merge classes of ast.Module of sub-network to ast.Module of main-network.

        Note:
            Classes of sub_module would be added ahead of classes in main_module.

        Args:
             main_mod (ast.Module): An ast.Module corresponding to main-network.
             sub_mod (ast.Module): An ast.Module corresponding to sub-network.

        """

        sub_mod_finder = AstFinder(sub_mod)
        main_mod_finder = AstFinder(main_mod)
        classes_in_sub = copy(sub_mod_finder.find_all(ast.ClassDef))
        classes_in_main = copy(main_mod_finder.find_all(ast.ClassDef))
        assert classes_in_main
        first_class = classes_in_main[0]
        for clazz in classes_in_sub:
            AstModifier.insert_class_into_module(main_mod, clazz, first_class, True)

    def _merge_module_of_subtree(self):
        """
        Merge ast.Module of all sub-networks into ast.Module of main-network.

        1. Merge imports of ast.Module.
        2. Merge classes of ast.Module.
        3. Use merged ast.Module as module of main-network and sub-network.
        """

        father_mod = self._root_tree.get_module_ast()
        for node in self._root_tree.nodes():
            if isinstance(node, TreeNode):
                sub_stree: SymbolTree = node.symbol_tree
                SymbolTreeBuilder._merge_import_of_module(father_mod, sub_stree.get_module_ast())
                SymbolTreeBuilder._merge_class_of_module(father_mod, sub_stree.get_module_ast())
                sub_stree.set_module_ast(father_mod)

    def _reduce_redundant_import(self):
        """
        Reduce redundant imports of ast.Module.

        Redundant imports may be introduced into ast.Module while merging ast.Module of sub-network to main-network.
        """

        module: ast.Module = self._root_tree.get_module_ast()
        import_list = []
        exist_import = []
        exist_import_from = []
        for body in module.body:
            if isinstance(body, ast.Import):
                names = body.names
                for name in names:
                    assert isinstance(name, ast.alias)
                    import_hash = hash((name.name, name.asname))
                    if import_hash in exist_import:
                        continue
                    exist_import.append(import_hash)
                    import_list.append(ast.Import(names=[ast.alias(name=name.name, asname=name.asname)]))
            if isinstance(body, ast.ImportFrom):
                import_module = body.module
                names = body.names
                for name in names:
                    assert isinstance(name, ast.alias)
                    import_hash = hash((import_module, name.name, name.asname))
                    if import_hash in exist_import_from:
                        continue
                    exist_import_from.append(import_hash)
                    import_list.append(ast.ImportFrom(module=import_module,
                                                      names=[ast.alias(name=name.name, asname=name.asname)],
                                                      level=0))
        insert_pos = None
        for i in range(len(module.body) - 1, -1, -1):
            body = module.body[i]
            if not isinstance(body, (ast.Import, ast.ImportFrom)):
                insert_pos = body
                continue
            module.body.pop(i)
        for import_ast in import_list:
            AstModifier.insert_sub_ast(module, import_ast, insert_pos, True)

    def build(self) -> SymbolTree:
        """
        Build SymbolTree.

        Returns:
             An instance of SymbolTree.
        """

        self._ast_root = SymbolTreeBuilder._ast_transform(self._ast_root)
        if not isinstance(self._ast_root, ast.Module):
            raise RuntimeError("ast_root should be a ast.Module")
        self._root_tree: SymbolTree = SymbolTree(self._origin_net, self._ast_root)
        parser: Parser = ParserRegister.instance().get_parser(ast.Module)
        parser.process(self._root_tree, self._ast_root)
        self._merge_module_of_subtree()
        self._reduce_redundant_import()
        ast.fix_missing_locations(self._root_tree.get_module_ast())
        self._root_tree.start_observe()
        return self._root_tree
