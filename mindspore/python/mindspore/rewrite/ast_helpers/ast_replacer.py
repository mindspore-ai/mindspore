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
"""Replacing specific symbol name with another symbol name in specific scope."""

from typing import Any
import ast


class AstReplacer(ast.NodeTransformer):
    """
    Replace all specific symbol name in specific scope with another symbol name.

    Args:
        node (ast.AST): An instance of ast node as replace scope.
    """

    def __init__(self, node: ast.AST):
        self._scope = node
        self._src = ""
        self._dst = ""
        self._trace = []

    def visit_ClassDef(self, node: ast.ClassDef) -> Any:
        """
        An override method, call back when visiting an ast.ClassDef node.

        Args:
            node (ast.ClassDef): An instance of ast.ClassDef which is visited currently.
        """

        if node.name == self._src:
            node.name = self._dst
            self._trace.append((node, "name", self._src, self._dst))
        return self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> Any:
        """
        An override method, call back when visiting an ast.Name node.

        Args:
            node (ast.Name): An instance of ast.Name which is visited currently.
        """

        if node.id == self._src:
            node.id = self._dst
            self._trace.append((node, "id", self._src, self._dst))
        return self.generic_visit(node)

    def replace_all(self, src: str, dst: str):
        """
        Replace all matched symbol to new symbol name.

        Args:
            src (str): Target symbol name to be replaced out.
            dst (str): New symbol name to be replaced in.
        """

        self._src = src
        self._dst = dst
        self.visit(self._scope)

    def undo_all(self):
        """Undo all replace-actions applied on current scope."""

        for trace in self._trace:
            setattr(trace[0], trace[1], trace[2])
        self._trace.clear()
