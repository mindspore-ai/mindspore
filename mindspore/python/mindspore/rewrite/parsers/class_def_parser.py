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
"""Parse ast.ClassDef which is subclass of Cell to SymbolTree."""
import sys
import ast
import inspect
from mindspore import log as logger
from mindspore.nn import Cell
from mindspore._extends.parse.namespace import CellNamespace
from ..symbol_tree import SymbolTree
from ..parser import Parser
from ..parser_register import ParserRegister, reg_parser
from ..ast_helpers import AstReplacer
from ..common import error_str


class AstScopeChecker:
    """
    Check scope of ast node meets the constraints recursively.

    Args:
        scope_constraints (str): A string represents constraints of scope.
    """

    def __init__(self, scope_constraints: str):
        self._scope = scope_constraints

    def check(self, node: ast.AST) -> bool:
        """
        Check scope of `node` meets the constraints recursively.

        Args:
            node (ast.AST): A ast.AST node to be checked.

        Returns:
            A bool represents if input `node` meets constraints.
        """
        if isinstance(node, ast.Compare):
            return self._check_compare(node)
        if isinstance(node, ast.Attribute):
            return self._check_attribute(node)
        if isinstance(node, ast.Name):
            return False
        if isinstance(node, ast.BoolOp):
            return self._check_bool(node)
        if isinstance(node, ast.UnaryOp):
            return self._check_unary(node)
        if isinstance(node, ast.Call):
            return self._check_call(node)
        if isinstance(node, (ast.Constant, ast.NameConstant, ast.Bytes, ast.Str, ast.Num)):
            return True
        raise RuntimeError(error_str(f"only support (ast.Compare, ast.Attribute, ast.Name, ast.BoolOp, ast.UnaryOp"
                                     f"ast.Call, ast.Constant, ast.NameConstant, ast.Bytes, ast.Str, ast.Num"
                                     f") as test check, but got ast type '{type(node).__name__}'", father_node=node))

    def _check_attribute(self, node: ast.Attribute):
        """Check an ast.Attribute meets the constraints recursively."""
        if not isinstance(node.value, ast.Name):
            return False
        if node.value.id != self._scope:
            return False
        return True

    def _check_compare(self, node: ast.Compare):
        """Check an ast.Compare meets the constraints recursively."""
        left = node.left
        for comparator in node.comparators:
            if not self.check(comparator):
                return False
        return self.check(left)

    def _check_bool(self, node: ast.BoolOp):
        """Check an ast.BoolOp meets the constraints recursively."""
        for value in node.values:
            if not self.check(value):
                return False
        return True

    def _check_call(self, node: ast.Call):
        """Check an ast.Call meets the constraints recursively."""
        for arg in node.args:
            if not self.check(arg):
                return False
        for kwarg in node.keywords:
            if not self.check(kwarg):
                return False
        return self.check(node.func)

    def _check_unary(self, node: ast.UnaryOp):
        """Check an ast.UnaryOp meets the constraints recursively."""
        return self.check(node.operand)


class ClassDefParser(Parser):
    """Parse ast.ClassDef which is subclass of Cell to SymbolTree."""

    def __init__(self):
        """Constructor"""
        super(ClassDefParser, self).__init__()
        self._cell_namespace = CellNamespace('mindspore.nn')

    @staticmethod
    def _is_super_expr(expr: ast.AST) -> bool:
        """Check whether ast node is super().__init__()"""
        if not isinstance(expr, ast.Expr):
            return False
        if not isinstance(expr.value, ast.Call):
            return False
        if not isinstance(expr.value.func, ast.Attribute):
            return False
        if expr.value.func.attr != "__init__" or not isinstance(expr.value.func.value, ast.Call):
            return False
        if not isinstance(expr.value.func.value.func, ast.Name) or expr.value.func.value.func.id != "super":
            return False
        return True

    @staticmethod
    def _modify_arguments_of_init_func(ast_init_fn: ast.FunctionDef):
        """Replace init function input parameters to self and global_vars."""
        arg_self = ast.arg(arg="self", annotation="")
        arg_global_vars = ast.arg(arg="obj", annotation="")
        ast_init_fn.args = ast.arguments(args=[arg_self, arg_global_vars], posonlyargs=[], kwonlyargs=[],
                                         kw_defaults=[], defaults=[], vararg=None, kwarg=None)
        ast.fix_missing_locations(ast_init_fn)

    def target(self):
        """Parse target type"""
        return ast.ClassDef

    def process(self, stree: SymbolTree, node: ast.ClassDef):
        """
        Parse init and construct in ast.ClassDef.

        Args:
            stree ([SymbolTree]): Symbol Tree under parsing.
            node ([ast.ClassDef]): An ast.ClassDef node.
        """
        replacer = AstReplacer(node)
        replacer.replace_all(stree.get_ori_cls_name(), stree.get_opt_cls_name())

        stree.set_class_ast(node)
        has_father_class = self._handle_father_class(stree, node)

        if self._need_add_init_func(stree, node):
            self._add_init_func(node)

        for body in node.body:
            if isinstance(body, ast.FunctionDef):
                if body.name == "__init__":
                    self._process_init_func_ast(body, has_father_class)
                    stree.set_init_func_ast(body)
                elif body.name == "construct":
                    parser: Parser = ParserRegister.instance().get_parser(ast.FunctionDef)
                    parser.process(stree, body)
                else:
                    logger.info(
                        "Ignoring ast.FunctionDef in ast.ClassDef except __init__ and construct function: %s",
                        body.name)
            else:
                logger.info("Ignoring unsupported node(%s) in ast.ClassDef.", type(body).__name__)

    def _is_subtree_field(self, ori_net, field) -> bool:
        op = getattr(ori_net, field)
        return not type(op).__name__ in self._cell_namespace

    def _process_init_func_ast(self, init_ast: ast.FunctionDef, has_father_class: bool):
        """Process init func"""
        ClassDefParser._modify_arguments_of_init_func(init_ast)
        new_bodies = self._replace_ori_field_of_init_func(init_ast.body, has_father_class)
        init_ast.body = new_bodies

    def _need_add_init_func(self, stree: SymbolTree, cls_ast: ast.ClassDef) -> bool:
        """If class is child class of nn.Cell but not have init func, then we need to add init func"""
        if not isinstance(stree.get_origin_network(), Cell):
            return False
        for body in cls_ast.body:
            if isinstance(body, ast.FunctionDef) and body.name == '__init__':
                return False
        return True

    def _add_init_func(self, cls_ast: ast.ClassDef):
        """Add init func with super().__init__()"""
        init_func_ast = ast.parse("def __init__(self): super().__init__()").body[0]
        cls_ast.body.insert(0, init_func_ast)
        ast.fix_missing_locations(cls_ast)

    def _replace_ori_field_of_init_func(self, bodies: [], has_father_class: bool):
        """
        Replace original field in init func to self.XX = getattr(self._handler, "XX").
        Only keep following two kinds of ast nodes in bodies right now:
            1. Ast.If and test is self.XX.
            2. Ast.Assign and target is self.XX.

        Args:
            bodies ([]): bodied of init ast.FunctionDef.
            has_father_class (bool): whether class has father class that is not nn.Cell

        Raises:
            RuntimeError: Not support multi-targets in assign.
            RuntimeError: Only support target.value in [ast.Name] in assign node.
        """
        new_bodies = []
        for body in bodies:
            if self._is_super_expr(body):
                if has_father_class:
                    body.value.args = [ast.Name(id='obj', ctx=ast.Load())]
                new_bodies.append(body)
                continue
        ast_copy_attr = ast.parse(
            "for key, value in obj.__dict__.items(): setattr(self, key, value)").body[0]
        new_bodies.append(ast_copy_attr)
        return new_bodies

    def _handle_father_class(self, stree, node: ast.ClassDef) -> bool:
        """Handle father class."""
        has_father_class = False
        for base in node.bases:
            parser: Parser = ParserRegister.instance().get_parser(type(base))
            father_class = parser.process(stree, base)
            if "Cell" not in father_class:
                for k, m in sys.modules.items():
                    if k in ("_ast", "ast"):
                        continue
                    if hasattr(m, father_class):
                        cls = getattr(m, father_class)
                        if not inspect.isclass(cls):
                            continue
                        source_code = inspect.getsource(cls)
                        father_class_ast: ast.Module = ast.parse(source_code)
                        self._father_class_process_init_func_ast(stree, father_class_ast)
                        stree._father_class_ast.append(father_class_ast) # pylint: disable=protected-access
                        has_father_class = True
                        break
        return has_father_class

    def _father_class_process_init_func_ast(self, stree: SymbolTree, father_class_ast: ast.Module):
        father_class_stree: SymbolTree = SymbolTree(stree.get_origin_network(), father_class_ast)
        for ast_body in father_class_ast.body:
            if isinstance(ast_body, ast.ClassDef):
                has_father_class = self._handle_father_class(stree, ast_body)
                if self._need_add_init_func(father_class_stree, ast_body):
                    self._add_init_func(ast_body)
                for body in ast_body.body:
                    if isinstance(body, ast.FunctionDef) and body.name == "__init__":
                        self._process_init_func_ast(body, has_father_class)


g_classdef_parser = reg_parser(ClassDefParser())
