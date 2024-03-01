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

from typing import Any, Tuple, List, Dict, Union
import keyword
import ast
import copy

from mindspore import log as logger

FLATTEN_BLACK_LIST = ["set_vertex_attr"]


class AstFlattener(ast.NodeTransformer):
    """Ast optimizer for flatten recursive call."""

    # Record origin test ast, used to judge direction of static if control flow.
    ast_if_test_cache: Dict[ast.If, ast.AST] = {}

    def __init__(self):
        """
        Constructor of AstFlattener.

        Returns:
            An instance of ast optimizer for flatten recursive call.
        """
        self._flatten_table: dict = {
            ast.Return: ["value"],
            ast.Call: ["func", "args", "keywords"],
            ast.BinOp: ["left", "right"],
            ast.BoolOp: ["values"],
            ast.UnaryOp: ["operand"],
            ast.Compare: ["left", "comparators"],
            ast.If: ["test"],
            ast.For: ["iter"],
            ast.Tuple: ["elts"],
            ast.List: ["elts"],
        }
        self._transform_functions = []
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

    @staticmethod
    def _flatten_continuous_assign(ast_body: List[ast.AST]):
        """
        Flatten ast.Assign with continuous targets.
        """
        for pos, ast_node in enumerate(ast_body):
            if not isinstance(ast_node, ast.Assign):
                continue
            if not len(ast_node.targets) > 1:
                continue
            for idx, ast_target in enumerate(ast_node.targets[:-1]):
                new_assign = ast.Assign(targets=[ast_target], value=ast_node.targets[idx + 1])
                ast_body.insert(pos + idx + 1, new_assign)
            ast_node.targets = [ast_node.targets[-1]]

    @staticmethod
    def _save_target_names(ast_body: List[ast.AST]):
        """Saving target names in ast_body before getting unique names."""
        target_names = []
        for child in ast_body:
            if not isinstance(child, ast.Assign):
                continue
            targets = child.targets
            for target in targets:
                if isinstance(target, ast.Name) and target.id not in target_names:
                    target_names.append(target.id)
                elif isinstance(target, (ast.Tuple, ast.List)):
                    # get target names from list recursively
                    ast_queue = [target.elts]
                    while ast_queue:
                        elt = ast_queue.pop()
                        if isinstance(elt, ast.Name) and elt.id not in target_names:
                            target_names.append(elt.id)
                        elif isinstance(elt, (ast.Tuple, ast.List)):
                            ast_queue.extend(elt.elts)
                        elif isinstance(elt, (list, tuple)):
                            ast_queue.extend(elt)
        return target_names

    def _generate_target_name(self, node: ast.AST, target_names):
        """Generate unique target name."""
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name):
                target_name = func.id + "_var"
            elif isinstance(func, ast.Attribute):
                target_name = func.attr + "_var"
            else:
                logger.debug("unhandled type of func of ast.Call while generating new target name: %s ", type(func))
                target_name = "function_var"
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
            logger.debug("unhandled type of node while generating new target name: %s ", type(node))
            target_name = type(node).__name__.lower() + "_var"
        # avoid python built-in keyword
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

    def _create_new_assign_node(self, node: ast.AST, target_names, father_node: ast.AST) \
            -> Tuple[Union[ast.Name, ast.Attribute], ast.AST]:
        """Create new assign node to be inserted into ast.FunctionDef."""
        ast_unflattens = (ast.Name, ast.NameConstant, ast.Constant, ast.Num, ast.Str, ast.Bytes, ast.Ellipsis)
        if isinstance(node, ast_unflattens):
            return node, None
        # ast.Attribute in ast.For will be force flatten
        # when ast.Attribute is not in ast.For, it's value which is not type of ast.Name will be flatten
        if isinstance(node, ast.Attribute) and not isinstance(father_node, ast.For):
            iter_node = node
            while isinstance(iter_node.value, ast.Attribute):
                iter_node = iter_node.value
            if isinstance(iter_node.value, ast.Name):
                return node, None
            new_target_name = self._generate_target_name(iter_node.value, target_names)
            new_node = ast.Attribute(value=ast.Name(id=new_target_name, ctx=ast.Load()),
                                     attr=iter_node.attr, ctx=iter_node.ctx)
            return new_node, ast.Assign(targets=[ast.Name(id=new_target_name, ctx=ast.Store())], value=iter_node.value)
        # flatten nodes
        new_target_name = self._generate_target_name(node, target_names)
        new_node = ast.Name(id=new_target_name, ctx=ast.Load())
        return new_node, ast.Assign(targets=[ast.Name(id=new_target_name, ctx=ast.Store())], value=node)

    def _flatten_statement(self, node: ast.AST, target_names) -> [ast.AST]:
        """Flatten recursive statement according to different node type."""
        if AstFlattener._check_flatten_black_list(node):
            return []
        flatten_config = self._flatten_table.get(type(node))
        if flatten_config is None:
            return []
        results = []
        for todo_name in flatten_config:
            todos = getattr(node, todo_name)
            if isinstance(todos, list):
                new_list = []
                for idx, todo in enumerate(todos):
                    # Starred expression(e.g. *args) cannot be flatten.
                    if isinstance(todo, ast.Starred):
                        new_list.append(todo)
                        continue
                    # For codes like 'xxx and yyy and zzz', only 'xxx' can be flatten and parsed,
                    # otherwise executing 'yyy' may raise an exception when 'xxx' is False
                    if isinstance(node, ast.BoolOp) and isinstance(node.op, ast.And) and idx > 0:
                        new_list.append(todo)
                        continue
                    # ast.keywords are processed individually:
                    # y = func(key=value) => new_target_name = value & y = func(key=new_target_name)
                    if isinstance(todo, ast.keyword):
                        new_node, new_assign = self._create_new_assign_node(todo.value, target_names, node)
                        if id(new_node) != id(todo.value):
                            todo.value = new_node
                            results.append(new_assign)
                        new_list.append(todo)
                        continue
                    new_node, new_assign = self._create_new_assign_node(todo, target_names, node)
                    if id(new_node) != id(todo):
                        new_list.append(new_node)
                        results.append(new_assign)
                    else:
                        new_list.append(todo)
                setattr(node, todo_name, new_list)
            elif isinstance(todos, dict):
                new_dict = []
                for key, value in todos:
                    new_node, new_assign = self._create_new_assign_node(value, target_names, node)
                    if id(new_node) != id(value):
                        new_dict[key] = new_node
                        results.append(new_assign)
                    else:
                        new_dict[key] = value
                setattr(node, todo_name, new_dict)
            else:
                new_node, new_assign = self._create_new_assign_node(todos, target_names, node)
                if id(new_node) != id(todos):
                    setattr(node, todo_name, new_node)
                    results.append(new_assign)
        return results

    def _visit_ast_bodies(self, ast_body: List[ast.AST]):
        """Traverse nodes in ast_body and flatten nodes recursive."""
        # Flatten continuous assign statements in ast_body
        AstFlattener._flatten_continuous_assign(ast_body)
        # save target names, used when create new assign ast node
        target_names = AstFlattener._save_target_names(ast_body)
        index = len(ast_body) - 1
        while index >= 0:
            child = ast_body[index]
            # Record origin test ast, used to judge direction of static if control flow.
            if isinstance(child, ast.If) and child not in AstFlattener.ast_if_test_cache:
                AstFlattener.ast_if_test_cache[child] = copy.deepcopy(child.test)

            stmt = child.value if isinstance(child, (ast.Assign, ast.Expr)) else child
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

    def transform(self, ast_root, transform_functions=None, stree=None):
        """Interface of AstFlattener."""
        self._transform_functions = transform_functions if transform_functions else ["construct"]
        self._symbol_tree = stree
        ast_root = self.visit(ast_root)
        ast_root = ast.fix_missing_locations(ast_root)
        return ast_root

    def transform_control_flow(self, ast_control_flow: Union[ast.If, ast.For, ast.While], stree=None):
        """Interface of AstFlattener."""
        self._transform_functions = []
        self._symbol_tree = stree
        self._visit_ast_bodies(ast_control_flow.body)
        if ast_control_flow.orelse:
            self._visit_ast_bodies(ast_control_flow.orelse)
        ast_control_flow = ast.fix_missing_locations(ast_control_flow)
        return ast_control_flow
