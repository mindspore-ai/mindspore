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

from typing import Any, Tuple
import ast
from ast import FunctionDef

from mindspore import log as logger
import astunparse
from ..common import error_str


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
        }

    @staticmethod
    def _generate_target_name(node: ast.AST, target_names):
        """Generate unique target name."""
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name):
                target_name = func.id
            elif isinstance(func, ast.Attribute):
                target_name = func.attr
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
        suffix = 0
        result = target_name
        while result in target_names:
            suffix += 1
            result = f"{target_name}_{suffix}"
        target_names.append(result)
        return result

    @staticmethod
    def _create_new_assign_node(node: ast.AST, target_names) -> Tuple[str, ast.AST]:
        """Create new assign node to be inserted into ast.FunctionDef."""
        if isinstance(node, (ast.Name, ast.Constant, ast.Num, ast.Str, ast.NameConstant, ast.Bytes, ast.Ellipsis)):
            return "", node
        new_target_name = FlattenRecursiveStmt._generate_target_name(node, target_names)
        return new_target_name, ast.Assign(targets=[ast.Name(id=new_target_name, ctx=ast.Store())], value=node)

    @staticmethod
    def _flatten_list(node_list, target_names):
        """Flatten a list of node."""
        results = list()
        new_list = list()
        for node in node_list:
            if isinstance(node, ast.Call):
                new_target, new_node = FlattenRecursiveStmt._create_new_assign_node(node, target_names)
                results.append(new_node)
                new_list.append(ast.Name(id=new_target, ctx=ast.Load()))
            else:
                new_list.append(node)
        return results, new_list

    def _flatten_statement(self, node: ast.AST, target_names) -> [ast.AST]:
        """Flatten recursive statement according to different node type."""
        flatten_config = self._flatten_table.get(type(node))
        if flatten_config is None:
            return []
        results = []
        for todo_name in flatten_config:
            todos = getattr(node, todo_name)
            if isinstance(todos, list):
                new_list = []
                for todo in todos:
                    new_target_name, new_node = FlattenRecursiveStmt._create_new_assign_node(todo, target_names)
                    if id(new_node) == id(todo):
                        new_list.append(todo)
                    else:
                        new_list.append(ast.Name(id=new_target_name, ctx=ast.Load()))
                        results.append(new_node)
                    if isinstance(todo, (ast.Tuple, tuple)):
                        _res, _new_list = FlattenRecursiveStmt._flatten_list(new_node.value.elts, [new_target_name])
                        new_node.value.elts = _new_list
                        results.extend(_res)
                setattr(node, todo_name, new_list)
            elif isinstance(todos, dict):
                new_dict = []
                for key, value in todos:
                    new_target_name, new_node = FlattenRecursiveStmt._create_new_assign_node(value, target_names)
                    if id(new_node) == id(value):
                        new_dict[key] = value
                    else:
                        new_dict[key] = ast.Name(id=new_target_name, ctx=ast.Load())
                        results.append(new_node)
                setattr(node, todo_name, new_dict)
            else:
                new_target_name, new_node = FlattenRecursiveStmt._create_new_assign_node(todos, target_names)
                if id(new_node) != id(todos):
                    setattr(node, todo_name, ast.Name(id=new_target_name, ctx=ast.Load()))
                    results.append(new_node)
        return results

    def _fill_in_original_target_names(self, target_names, node):
        """Fill in original target names before getting unique names."""
        for function_index in range(len(node.body)):
            child = node.body[function_index]
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
                if not isinstance(target, (ast.Name, ast.Tuple)):
                    raise RuntimeError(
                        error_str(f"currently only support ast.Name targets, but got ast type "
                                  f"'{type(target).__name__}'", child_node=target, father_node=child))
                if isinstance(target, ast.Name):
                    target_name = target.id
                    if target_name not in target_names:
                        target_names.append(target_name)
                elif isinstance(target, ast.Tuple):
                    for elt in target.elts:
                        if not isinstance(elt, ast.Name):
                            raise RuntimeError(
                                error_str(f"currently only support ast.Name in ast.Tuple, "
                                          f"but got ast type '{type(elt).__name__}'", child_node=elt,
                                          father_node=child))
                        target_name = elt.id
                        if target_name not in target_names:
                            target_names.append(target_name)

    def visit_FunctionDef(self, node: FunctionDef) -> Any:
        """Traverse construct node and flatten recursive nodes."""
        if node.name != "construct":
            return node

        target_names = []
        self._fill_in_original_target_names(target_names, node)
        index = len(node.body) - 1
        while index >= 0:
            child = node.body[index]
            if isinstance(child, ast.Assign):
                stmt = child.value
            elif isinstance(child, ast.If):
                if isinstance(child.body[0], ast.Return) and not isinstance(child.test, ast.UnaryOp):
                    if isinstance(child.body[0].value, ast.Call):
                        if_body = child.body
                        if_func = if_body[0].value
                        expr = "x = " + astunparse.unparse(if_func)
                        if_body = ast.parse(expr)
                        if_body = if_body.body+ast.parse("return x").body
                        child.body = if_body
                        stmt = child
                    else:
                        stmt = child
                else:
                    stmt = child
            elif isinstance(child, ast.Expr):
                stmt = child.value
            else:
                stmt = child
            results = self._flatten_statement(stmt, target_names)
            if results:
                results.reverse()
                for result in results:
                    node.body.insert(index, result)
                    index += 1
            index -= 1
        return node

    def transform(self, ast_root):
        """Interface of FlattenRecursiveStmt."""
        ast_root = self.visit(ast_root)
        ast_root = ast.fix_missing_locations(ast_root)
        return ast_root
