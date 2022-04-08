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
"""Find specific type ast node in specific scope."""

from typing import Type
import ast


class AstFinder(ast.NodeVisitor):
    """
    Find all specific type ast node in specific scope.

    Args:
        node (ast.AST): An instance of ast node as search scope.
    """

    def __init__(self, node: ast.AST):
        self._scope: ast.AST = node
        self._targets: tuple = ()
        self._results: [ast.AST] = []

    def generic_visit(self, node):
        """
        An override method, iterating over all nodes and save target ast nodes.

        Args:
            node (ast.AST): An instance of ast node which is visited currently.
        """

        if isinstance(node, self._targets):
            self._results.append(node)
        super(AstFinder, self).generic_visit(node)

    def find_all(self, ast_types) -> [ast.AST]:
        """
        Find all matched ast node.

        Args:
            ast_types (Union[tuple(Type), Type]): A tuple of Type or a Type indicates target ast node type.

        Returns:
            A list of instance of ast.AST as matched result.

        Raises:
            ValueError: If input `ast_types` is not a type nor a tuple.
        """

        if isinstance(ast_types, Type):
            self._targets: tuple = (ast_types,)
        else:
            if not isinstance(ast_types, tuple):
                raise ValueError("Input ast_types should be a tuple or a type")
            self._targets: tuple = ast_types

        self._results.clear()
        self.visit(self._scope)
        return self._results
