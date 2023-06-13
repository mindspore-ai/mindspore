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
"""Rewrite module api: TreeNodeHelper."""
from typing import Optional

from mindspore import log as logger
from mindspore import _checkparam as Validator
from .symbol_tree import SymbolTree
from .node import Node
from .node_type import NodeType
from ..symbol_tree import SymbolTree as SymbolTreeImpl


class TreeNodeHelper:
    """
    `TreeNodeHelper` is used to break circle reference while getting symbol_tree from a `Tree` type `Node`.

    `TreeNodeHelper` provides a staticmethod `get_sub_tree` for getting symbol_tree from a `Tree` type `Node`.

    .. warning::
        This is a set of experimental APIs that is subject to change or deletion.
    """

    @staticmethod
    def get_sub_tree(node: Node) -> Optional[SymbolTree]:
        """
        Getting symbol_tree from a `Tree` type `Node`.

        Args:
            node (Node): A `Node` which may hold a sub-symbol_tree.

        Returns:
            An instance of SymbolTree represents sub-symbol_tree. Note that `node`'s symbol_tree maybe None, in this
            case, method will return None.

        Raises:
            RuntimeError: If `node`'s type is not `NodeType.Tree`.
            TypeError: If `node` is not a `Node` instance.
        """
        Validator.check_value_type("node", node, [Node], "TreeNodeHelper")
        if node.get_node_type() == NodeType.Tree:
            node_impl = node.get_handler()
            subtree: SymbolTreeImpl = node_impl.symbol_tree
            if subtree is None:
                return None
            return SymbolTree(subtree)
        logger.info(f"Current node is not a Tree node, current node type: {type(node)}")
        return None
