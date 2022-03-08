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
from .parser_register import ParserRegister
from .parser import Parser
from .ast_transformers import FlattenRecursiveStmt


class SymbolTreeBuilder:
    """SymbolTree builder."""

    def __init__(self, network: Cell):
        """
        Constructor of SymbolTreeBuilder.

        Args:
             network (Cell): An instance of Cell represents a network from which SymbolTree will be built.
        """
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
        return self._root_tree
