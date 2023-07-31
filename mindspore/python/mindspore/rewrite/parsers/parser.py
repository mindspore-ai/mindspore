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
"""Base class of parser."""
import abc
import ast

from ..symbol_tree import SymbolTree
from ..node.node_manager import NodeManager


class Parser(abc.ABC):
    """
    DFS into a ast_node until add node into SymbolTree
    """

    def target(self) -> type:
        """
        Get type of ast which could be accepted by current parser.

        Returns:
            A type of ast.
        """
        return type(None)

    @abc.abstractmethod
    def process(self, stree: SymbolTree, node: ast.AST, node_manager: NodeManager):
        """
        Parse input ast node and add parse result into SymbolTree.

        Args:
             stree (SymbolTree): current symbol_tree
             node (ast.AST): node who is tried to be parsed
             node_manager (NodeManager): NodeManager those asts belong to.
        """
        raise NotImplementedError
