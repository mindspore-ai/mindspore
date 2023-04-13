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
import ast
import astunparse
from mindspore import log as logger
from mindspore._extends.parse.namespace import CellNamespace
from mindspore.rewrite.ast_creator_register import ast_creator_registry
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

        if self._need_add_init_func(node):
            self._add_init_func(node)

        for body in node.body:
            if isinstance(body, ast.FunctionDef):
                if body.name == "__init__":
                    self._process_init_func_ast(stree, node, body)
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

    def _process_init_func_ast(self, stree: SymbolTree, cls_ast: ast.ClassDef, init_ast: ast.FunctionDef):
        """Process init func"""
        ClassDefParser._modify_arguments_of_init_func(init_ast)
        new_bodies = self._replace_ori_field_of_init_func(stree, cls_ast, init_ast.body)
        init_ast.body = new_bodies

    @staticmethod
    def _is_super_expr(expr: ast.AST) -> bool:
        """Check whether ast node is super().__init__()"""
        if not isinstance(expr, ast.Expr):
            return False
        expr_value = expr.value
        if not isinstance(expr_value, ast.Call):
            return False
        expr_value_func = expr_value.func
        if not isinstance(expr_value_func, ast.Attribute):
            return False
        expr_value_func_value = expr_value_func.value
        if expr_value_func.attr != "__init__" or not isinstance(expr_value_func_value, ast.Call):
            return False
        expr_value_func_value_func = expr_value_func_value.func
        if not isinstance(expr_value_func_value_func, ast.Name) or expr_value_func_value_func.id != "super":
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

    @staticmethod
    def _remove_empty_ast_in_init_func(bodies: []):
        """Remove ast.If, ast.For or other ast node with body when their body is empty recursively."""
        body_index_to_be_deleted = []
        for body_index, body in enumerate(bodies):
            if isinstance(body, ast.If):
                ClassDefParser._remove_empty_ast_in_init_func(body.body)
                ClassDefParser._remove_empty_ast_in_init_func(body.orelse)
                if not body.body and not body.orelse:
                    body_index_to_be_deleted.append(body_index)
                if not body.body and body.orelse:
                    body.body.append(ast.Pass())
                continue
            if isinstance(body, ast.For):
                ClassDefParser._remove_empty_ast_in_init_func(body.body)
                ClassDefParser._remove_empty_ast_in_init_func(body.orelse)
                if not body.body or not body.orelse:
                    body_index_to_be_deleted.append(body_index)
                continue
            if hasattr(body, "body"):
                ClassDefParser._remove_empty_ast_in_init_func(body.body)
                if not body.body:
                    body_index_to_be_deleted.append(body_index)
        for counter, index in enumerate(body_index_to_be_deleted):
            bodies.pop(index - counter)

    def _handle_tuple_for_replace_ori_field(self, ast_tuple: ast.Tuple, new_bodies):
        """ Handle ast.Assign node with target of ast.Tuple in init func to new ast nodes. """
        for e in ast_tuple.elts:
            if isinstance(e, ast.Attribute):
                field_name = e.attr
                value = ast.Call(ast.Name('getattr', ast.Load()),
                                 [ast.Name('obj', ast.Load()),
                                  ast.Constant(value=field_name, kind=None)], [])
                new_assign = ast_creator_registry.get("Assign")(targets=[e], value=value)
                new_bodies.append(new_assign)

    def _handle_express_for_replace_ori_field(self, cls_ast, ast_expr: ast.Expr, stree, new_bodies):
        """ Handle ast.Expr node in init func to new ast nodes. """
        ast_call = ast_expr.value
        if not isinstance(ast_call.func, ast.Attribute) or not isinstance(ast_call.func.value, ast.Name)\
            or ast_call.func.value.id != 'self':
            return
        for func_def in cls_ast.body:
            if isinstance(func_def, ast.FunctionDef) and func_def.name == ast_call.func.attr:
                for func_def_body in func_def.body:
                    self._handle_bodies_for_replace_ori_field(cls_ast, func_def_body, stree, new_bodies)
                return

    def _handle_assign_for_replace_ori_field(self, ast_assign: ast.Assign, stree, new_bodies):
        """ Handle ast.Assign node in init func to new ast nodes. """
        if len(ast_assign.targets) != 1:
            raise RuntimeError("not support multi-targets in assign now!", father_node=ast_assign)
        target = ast_assign.targets[0]
        if isinstance(target, ast.Tuple):
            self._handle_tuple_for_replace_ori_field(target, new_bodies)
            return
        if not isinstance(target, ast.Attribute) or not isinstance(target.value, ast.Name)\
            or target.value.id != 'self':
            logger.info(f"Ignoring {astunparse.unparse(target)} in __init__ function.")
            return
        field_name = target.attr
        # Ensure that the instance has corresponding attribute
        if not hasattr(stree.get_origin_network(), field_name):
            return
        # Check to avoid repeat code
        for new_ast in new_bodies:
            if isinstance(new_ast, ast.Assign) and isinstance(new_ast.targets[0], ast.Attribute)\
                and new_ast.targets[0].attr == field_name:
                return
        value = ast.Call(ast.Name('getattr', ast.Load()),
                         [ast.Name('obj', ast.Load()),
                          ast.Constant(value=field_name, kind=None)], [])
        new_assign = ast_creator_registry.get("Assign")(targets=[target], value=value)
        new_bodies.append(new_assign)

    def _handle_bodies_for_replace_ori_field(self, cls_ast, body, stree, new_bodies):
        """ handle_bodies_for_replace_ori_field. """
        if self._is_super_expr(body):
            new_bodies.append(body)
            return
        if isinstance(body, ast.If):
            for if_body in body.body + body.orelse:
                self._handle_bodies_for_replace_ori_field(cls_ast, if_body, stree, new_bodies)
            return
        if isinstance(body, ast.Expr) and isinstance(body.value, ast.Call):
            self._handle_express_for_replace_ori_field(cls_ast, body, stree, new_bodies)
            return
        if isinstance(body, ast.Assign):  # if not assign node, delete
            self._handle_assign_for_replace_ori_field(body, stree, new_bodies)
            return

    def _need_add_init_func(self, cls_ast: ast.ClassDef) -> bool:
        """If class has base nn.Cell but not have init func, then we need to add init func"""
        base_nn_cell = False
        for base in cls_ast.bases:
            if isinstance(base, ast.Name) and base.id == 'Cell'\
                or isinstance(base, ast.Attribute) and isinstance(base.value, ast.Name)\
                and base.value.id == "nn" and base.attr == 'Cell':
                base_nn_cell = True
                break
        if not base_nn_cell:
            return False
        for body in cls_ast.body:
            if isinstance(body, ast.FunctionDef) and body.name == '__init__':
                return False
        return True

    def _add_init_func(self, cls_ast: ast.ClassDef):
        """Add init func with super().__init__()"""
        init_func_ast = ast.FunctionDef(
            name='__init__',
            args=ast.arguments(
                posonlyargs=[],
                args=[
                    ast.arg(arg='self')],
                kwonlyargs=[],
                kw_defaults=[],
                defaults=[]),
            body=[
                ast.Expr(
                    value=ast.Call(
                        func=ast.Attribute(
                            value=ast.Call(
                                func=ast.Name(id='super', ctx=ast.Load()),
                                args=[],
                                keywords=[]),
                            attr='__init__',
                            ctx=ast.Load()),
                        args=[],
                        keywords=[]))],
            decorator_list=[])
        cls_ast.body.insert(0, init_func_ast)
        ast.fix_missing_locations(cls_ast)

    def _replace_ori_field_of_init_func(self, stree: SymbolTree, cls_ast: ast.ClassDef, bodies: []):
        """
        Replace original field in init func to self.XX = getattr(self._handler, "XX").
        Only keep following two kinds of ast nodes in bodies right now:
            1. Ast.If and test is self.XX.
            2. Ast.Assign and target is self.XX.

        Args:
            bodies ([]): bodied of init ast.FunctionDef.

        Raises:
            RuntimeError: Not support multi-targets in assign.
            RuntimeError: Only support target.value in [ast.Name] in assign node.
        """

        new_bodies = []
        for body in bodies:
            self._handle_bodies_for_replace_ori_field(cls_ast, body, stree, new_bodies)
        return new_bodies


g_classdef_parser = reg_parser(ClassDefParser())
