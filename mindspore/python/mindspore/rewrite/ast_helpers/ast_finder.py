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

from typing import Type, Any
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


class StrChecker(ast.NodeVisitor):
    """
    Check if specific String exists in specific scope.

    Args:
        node (ast.AST): An instance of ast node as check scope.
    """

    def __init__(self, node: ast.AST):
        self._context = node
        self._pattern = ""
        self._hit = False

    def visit_Attribute(self, node: ast.Attribute) -> Any:
        """Visit a node of type ast.Attribute."""
        if isinstance(node.value, ast.Name) and node.value.id == self._pattern:
            self._hit = True
        return super(StrChecker, self).generic_visit(node)

    def visit_Name(self, node: ast.Name) -> Any:
        """Visit a node of type ast.Name."""
        if node.id == self._pattern:
            self._hit = True
        return super(StrChecker, self).generic_visit(node)

    def generic_visit(self, node: ast.AST) -> Any:
        for _, value in ast.iter_fields(node):
            if self._hit:
                break
            if isinstance(value, (list, tuple)):
                for item in value:
                    if isinstance(item, ast.AST):
                        self.visit(item)
                    if self._hit:
                        break
            elif isinstance(value, dict):
                for item in value.values():
                    if isinstance(item, ast.AST):
                        self.visit(item)
                    if self._hit:
                        break
            elif isinstance(value, ast.AST):
                self.visit(value)

    def check(self, pattern: str) -> bool:
        """
        Check if `pattern` exists in `_context`.

        Args:
            pattern (str): A string indicates target pattern.

        Returns:
            A bool indicates if `pattern` exists in `_context`.
        """
        self._pattern = pattern
        self._hit = False
        self.generic_visit(self._context)
        return self._hit


class FindConstValueInInit(ast.NodeVisitor):
    """
    Check if specific String exists in specific scope.

    Args:
        node (ast.AST): An instance of ast node as check scope.
    """
    def __init__(self, node: ast.AST):
        self._context = node
        self._pattern = ""
        self._hit = False

    def visit_Constant(self, node: ast.Constant):
        if node.value == self._pattern:
            self._hit = True
        return node

    def check(self, pattern: str) -> bool:
        """
        Check if `pattern` exists in `_context`.

        Args:
            pattern (str): A string indicates target pattern.

        Returns:
            A bool indicates if `pattern` exists in `_context`.
        """
        self._pattern = pattern
        self._hit = False
        self.generic_visit(self._context)
        return self._hit


class CheckPropertyIsUsed(ast.NodeVisitor):
    """
    Check whether a property is used.

    Args:
        node (ast.AST): An instance of ast node.
    """
    def __init__(self, node: ast.AST):
        self._context = node
        self._value = ""
        self._attr = ""
        self._hit = False

    def visit_Attribute(self, node: ast.Attribute) -> Any:  # pylint: disable=invalid-name
        """Visit a node of type ast.Attribute."""
        if isinstance(node.value, ast.Name) and node.value.id == self._value and node.attr == self._attr:
            self._hit = True
        return super(CheckPropertyIsUsed, self).generic_visit(node)

    def generic_visit(self, node: ast.AST) -> Any:
        """
        An override method, iterating over all nodes and save target ast nodes.
        """
        if self._hit:
            return
        super(CheckPropertyIsUsed, self).generic_visit(node)

    def check(self, value, attr) -> bool:
        """
        Check whether `value` and `attr` exists.
        """
        self._value = value
        self._attr = attr
        self._hit = False
        self.generic_visit(self._context)
        return self._hit
