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
from typing import Optional
import ast
import inspect

from mindspore.nn import Cell
from .symbol_tree import SymbolTree
from .parsers.parser_register import ParserRegister
from .parsers.parser import Parser
from .ast_transformers import FlattenRecursiveStmt


class SymbolTreeBuilder:
    """
    `SymbolTreeBuilder` for building a SymbolTree from network.

    Args:
         network (Cell): An instance of Cell represents a network from which SymbolTree will be built.
    """

    # Entry function of the forward computation process
    entry_function = "construct"

    def __init__(self, network: Cell):
        if not isinstance(network, Cell):
            raise RuntimeError("Only support network with Cell type now, ", network)
        self._origin_net = network
        network_str = inspect.getsource(type(network))
        self._ast_root: ast.Module = ast.parse(network_str)
        self._root_tree: Optional[SymbolTree] = None

    @staticmethod
    def ast_transform(ast_root: ast.AST) -> ast.AST:
        """
        Optimize ast before parse.

        Args:
             ast_root (ast.AST): An instance of ast to be optimized.

        Returns:
             An instance of ast been optimized.
        """
        ast_root = FlattenRecursiveStmt().transform(ast_root, [SymbolTreeBuilder.entry_function])
        return ast_root

    def build(self) -> SymbolTree:
        """
        Build SymbolTree.

        Returns:
             An instance of SymbolTree.
        """

        self._ast_root = SymbolTreeBuilder.ast_transform(self._ast_root)
        if not isinstance(self._ast_root, ast.Module):
            raise RuntimeError("ast_root should be a ast.Module")
        self._root_tree: SymbolTree = SymbolTree(self._origin_net, self._ast_root)
        parser: Parser = ParserRegister.instance().get_parser(ast.Module)
        parser.process(self._root_tree, self._ast_root, None)
        ast.fix_missing_locations(self._root_tree.get_module_ast())
        self._root_tree.finish_build()
        return self._root_tree
