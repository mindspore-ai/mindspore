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
"""Ast optimizer for flatten recursive call."""

import sys
from typing import Any, Tuple, List
import keyword
import ast

from mindspore import log as logger
from ..common import error_str

if sys.version_info >= (3, 9):
    import ast as astunparse # pylint: disable=reimported, ungrouped-imports
else:
    import astunparse

FLATTEN_BLACK_LIST = ["set_vertex_attr",]

class FlattenRecursiveStmt(ast.NodeTransformer):
    """Ast optimizer for flatten recursive call."""

    def __init__(self):
        """
        Constructor of FlattenRecursiveStmt.

        Returns:
            An instance of ast optimizer for flatten recursive call.
        """
        self._flatten_table: dict = {
            ast.Return: ["value"],
            ast.Call: ["args"],
            ast.BinOp: ["left", "right"],
            ast.BoolOp: ["values"],
            ast.UnaryOp: ["operand"],
            ast.Compare: ["left", "comparators"],
            ast.If: ["test"]
        }
        self._transform_functions = []
        self._transform_if = False
        self._symbol_tree = None # Used to get unique name

    @staticmethod
    def _check_flatten_black_list(node: ast.AST):
        """Check whether node in flatten black list"""
        func_name = ""
        # Get func name of node
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                func_name = node.func.id
            elif isinstance(node.func, ast.Attribute):
                func_name = node.func.attr
        # Check func name of node
        if func_name and func_name in FLATTEN_BLACK_LIST:
            return True
        return False

    def _generate_target_name(self, node: ast.AST, target_names):
        """Generate unique target name."""
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name):
                target_name = func.id + "_var"
            elif isinstance(func, ast.Attribute):
                target_name = func.attr + "_var"
            else:
                logger.info("unhandled type of func of ast.Call while generating new target name: %s ", type(func))
                target_name = "function"
        elif isinstance(node, ast.Return):
            target_name = "return_value"
        elif isinstance(node, (ast.BinOp, ast.BoolOp, ast.UnaryOp)):
            target_name = type(node.op).__name__.lower() + "_var"
        elif isinstance(node, ast.Tuple):
            target_name = type(node).__name__.lower() + "_var"
        elif isinstance(node, ast.Name):
            target_name = node.id
        elif isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            target_name = f"{node.value.id}_{node.attr}"
        else:
            logger.info("unhandled type of node while generating new target name: %s ", type(node))
            target_name = type(node).__name__.lower() + "_var"
        # avoid python keyword
        if keyword.iskeyword(target_name):
            target_name = target_name + "_var"
        suffix = 0
        result = target_name
        while result in target_names:
            suffix += 1
            result = f"{target_name}_{suffix}"
        if self._symbol_tree:
            result = self._symbol_tree.unique_name(result)
        target_names.append(result)
        return result

    def _create_new_assign_node(self, node: ast.AST, target_names) -> Tuple[str, ast.AST]:
        """Create new assign node to be inserted into ast.FunctionDef."""
        if isinstance(node, (ast.Name, ast.Constant, ast.Num, ast.Str, ast.NameConstant, ast.Bytes, ast.Ellipsis)):
            return "", node
        new_target_name = self._generate_target_name(node, target_names)
        return new_target_name, ast.Assign(targets=[ast.Name(id=new_target_name, ctx=ast.Store())], value=node)

    def _flatten_list(self, node_list, target_names):
        """Flatten a list of node."""
        results = list()
        new_list = list()
        for node in node_list:
            if isinstance(node, ast.Call):
                new_target, new_node = self._create_new_assign_node(node, target_names)
                results.append(new_node)
                new_list.append(ast.Name(id=new_target, ctx=ast.Load()))
            else:
                new_list.append(node)
        return results, new_list

    def _flatten_statement(self, node: ast.AST, target_names) -> [ast.AST]:
        """Flatten recursive statement according to different node type."""
        if FlattenRecursiveStmt._check_flatten_black_list(node):
            return []
        flatten_config = self._flatten_table.get(type(node))
        if flatten_config is None:
            return []
        results = []
        for todo_name in flatten_config:
            todos = getattr(node, todo_name)
            if isinstance(todos, list):
                new_list = []
                for todo in todos:
                    # Starred expression(e.g. *args) cannot be flatten.
                    if isinstance(todo, ast.Starred):
                        new_list.append(todo)
                        continue
                    new_target_name, new_node = self._create_new_assign_node(todo, target_names)
                    if id(new_node) == id(todo):
                        new_list.append(todo)
                    else:
                        new_list.append(ast.Name(id=new_target_name, ctx=ast.Load()))
                        results.append(new_node)
                    if isinstance(todo, (ast.Tuple, tuple)):
                        _res, _new_list = self._flatten_list(new_node.value.elts, [new_target_name])
                        new_node.value.elts = _new_list
                        results.extend(_res)
                setattr(node, todo_name, new_list)
            elif isinstance(todos, dict):
                new_dict = []
                for key, value in todos:
                    new_target_name, new_node = self._create_new_assign_node(value, target_names)
                    if id(new_node) == id(value):
                        new_dict[key] = value
                    else:
                        new_dict[key] = ast.Name(id=new_target_name, ctx=ast.Load())
                        results.append(new_node)
                setattr(node, todo_name, new_dict)
            else:
                new_target_name, new_node = self._create_new_assign_node(todos, target_names)
                if id(new_node) != id(todos):
                    setattr(node, todo_name, ast.Name(id=new_target_name, ctx=ast.Load()))
                    results.append(new_node)
        return results

    def _save_target_names(self, target_names, ast_body: List[ast.AST]):
        """Saving target names in ast_body before getting unique names."""
        for child in ast_body:
            if isinstance(child, (ast.Assign, ast.Expr)):
                child_value = child.value
            else:
                child_value = child
            if not self._flatten_table.get(type(child_value)):
                continue

            if not isinstance(child, ast.Assign):
                continue
            targets = child.targets
            for target in targets:
                if not isinstance(target, (ast.Name, ast.Tuple, ast.List)):
                    raise RuntimeError(
                        error_str(f"currently only support ast.Name targets, but got ast type "
                                  f"'{type(target).__name__}'", child_node=target, father_node=child))
                if isinstance(target, ast.Name):
                    target_name = target.id
                    if target_name not in target_names:
                        target_names.append(target_name)
                elif isinstance(target, (ast.Tuple, ast.List)):
                    for elt in target.elts:
                        if not isinstance(elt, ast.Name):
                            raise RuntimeError(
                                error_str(f"currently only support ast.Name in ast.Tuple, "
                                          f"but got ast type '{type(elt).__name__}'", child_node=elt,
                                          father_node=child))
                        target_name = elt.id
                        if target_name not in target_names:
                            target_names.append(target_name)

    def _visit_ast_bodies(self, ast_body: List[ast.AST]):
        """Traverse nodes in ast_body and flatten nodes recursive."""
        target_names = []
        self._save_target_names(target_names, ast_body)
        index = len(ast_body) - 1
        while index >= 0:
            child = ast_body[index]
            if isinstance(child, ast.Assign):
                stmt = child.value
            elif isinstance(child, ast.If):
                if isinstance(child.body[0], ast.Return) and not isinstance(child.test, ast.UnaryOp):
                    if not isinstance(child.body[0].value, (ast.Name, ast.Constant)):
                        return_val_ast = child.body[0].value
                        return_name = self._generate_target_name(return_val_ast, target_names)
                        new_assign_code = f"{return_name} = {astunparse.unparse(return_val_ast)}"
                        new_assign_ast = ast.parse(new_assign_code).body[0]
                        new_return_ast = ast.parse(f"return {return_name}").body[0]
                        child.body = [new_assign_ast, new_return_ast]
                stmt = child
            elif isinstance(child, ast.Expr):
                stmt = child.value
            else:
                stmt = child
            results = self._flatten_statement(stmt, target_names)
            if results:
                for result in reversed(results):
                    ast_body.insert(index, result)
                    index += 1
            index -= 1

    def visit_FunctionDef(self, node: ast.FunctionDef) -> Any: # pylint: disable=invalid-name
        """Traverse nodes in _transform_functions and flatten recursive nodes."""
        if node.name not in self._transform_functions:
            return node
        self._visit_ast_bodies(node.body)
        return node

    def visit_If(self, node: ast.If) -> Any: # pylint: disable=invalid-name
        """Traverse nodes in if node and flatten recursive nodes."""
        if not self._transform_if:
            return node
        self._visit_ast_bodies(node.body)
        if node.orelse:
            self._visit_ast_bodies(node.orelse)
        return node

    def transform(self, ast_root, transform_functions=None, stree=None):
        """Interface of FlattenRecursiveStmt."""
        self._transform_functions = transform_functions if transform_functions else ["construct"]
        self._transform_if = False
        self._symbol_tree = stree
        ast_root = self.visit(ast_root)
        ast_root = ast.fix_missing_locations(ast_root)
        return ast_root

    def transform_if(self, ast_if, stree=None):
        """Interface of FlattenRecursiveStmt."""
        self._transform_functions = []
        self._transform_if = True
        self._symbol_tree = stree
        ast_if = self.visit(ast_if)
        ast_if = ast.fix_missing_locations(ast_if)
        return ast_if
