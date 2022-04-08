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

from mindspore import log as logger
from mindspore._extends.parse.namespace import CellNamespace
from ..symbol_tree import SymbolTree
from ..parser import Parser
from ..parser_register import ParserRegister, reg_parser
from ..api.scoped_value import ScopedValue
from ..ast_helpers import AstReplacer, AstModifier


class ClassDefParser(Parser):
    """Parse ast.ClassDef which is subclass of Cell to SymbolTree."""

    def __init__(self):
        """Constructor"""
        super(ClassDefParser, self).__init__()
        self._cell_namespace = CellNamespace('mindspore.nn')

    def target(self):
        """Parse target type"""
        return ast.ClassDef

    def _is_subtree_field(self, ori_net, field) -> bool:
        op = getattr(ori_net, field)
        assert op is not None
        return not type(op).__name__ in self._cell_namespace

    def _process_init_func_ast(self, stree: SymbolTree, init_ast: ast.FunctionDef):
        """Process init func"""
        super_index = ClassDefParser._find_super_expr_of_init_func(init_ast)
        ClassDefParser._modify_arguments_of_init_func(init_ast)
        self._replace_ori_field_of_init_func(stree, init_ast.body, super_index)
        ClassDefParser._insert_handler_to_init_func(init_ast, super_index)

    @staticmethod
    def _find_super_expr_of_init_func(ast_init_fn: ast.FunctionDef) -> int:
        """Find index of super(XXnet).__init__() in body of init ast.FunctionDef"""
        if not ast_init_fn.body:
            return -1
        super_index = -1
        while True:
            super_index += 1
            expr = ast_init_fn.body[super_index]
            if not isinstance(expr, ast.Expr):
                continue
            expr_value = expr.value
            if not isinstance(expr_value, ast.Call):
                continue
            expr_value_func = expr_value.func
            if not isinstance(expr_value_func, ast.Attribute):
                continue
            expr_value_func_value = expr_value_func.value
            if expr_value_func.attr != "__init__" or not isinstance(expr_value_func_value, ast.Call):
                continue
            expr_value_func_value_func = expr_value_func_value.func
            if not isinstance(expr_value_func_value_func, ast.Name) or expr_value_func_value_func.id != "super":
                continue
            break
        return super_index

    @staticmethod
    def _modify_arguments_of_init_func(ast_init_fn: ast.FunctionDef):
        """Replace init function input parameters to self and global_vars."""
        arg_self = ast.arg(arg="self", annotation="")
        arg_global_vars = ast.arg(arg="global_vars", annotation="")
        ast_init_fn.args = ast.arguments(args=[arg_self, arg_global_vars], posonlyargs=[], kwonlyargs=[],
                                         kw_defaults=[], defaults=[], vararg=None, kwarg=None)
        ast.fix_missing_locations(ast_init_fn)

    def _replace_ori_field_of_init_func(self, stree: SymbolTree, bodies: [], super_index: int):
        """
        Replace original field in init func to self.XX = getattr(self._handler, "XX").
        Only keep following two kinds of ast nodes in bodies right now:
            1. Ast.If and test is self.XX.
            2. Ast.Assign and target is self.XX.

        Args:
            bodies ([]): bodied of init ast.FunctionDef.
            super_index (int): index of super().__init__() in bodies.

        Raises:
            RuntimeError: Not support multi-targets in assign.
            RuntimeError: Only support target.value in [ast.Name] in assign node.
        """
        body_index_to_be_deleted = []
        for body_index, body in enumerate(bodies):
            if body_index == super_index:
                continue  # ignoring super.__init__()
            if isinstance(body, ast.If) and isinstance(body.test, ast.Attribute) \
                    and isinstance(body.test.value, ast.Name) and body.test.value.id == 'self':
                self._replace_ori_field_of_init_func(stree, body.body, -1)
                self._replace_ori_field_of_init_func(stree, body.orelse, -1)
                continue
            if not isinstance(body, ast.Assign):  # if not assign node, delete
                body_index_to_be_deleted.append(body_index)
                continue
            if len(body.targets) != 1:
                raise RuntimeError("Not support multi-targets in assign now!")
            target = body.targets[0]
            if not isinstance(target, ast.Attribute):  # only keep class member
                body_index_to_be_deleted.append(body_index)
                continue
            if not isinstance(target.value, ast.Name):
                raise RuntimeError("Only support target.value in ast.Name now!")
            target_value: ast.Name = target.value
            if target_value.id != "self":
                body_index_to_be_deleted.append(body_index)
                continue
            field_name = target.attr
            body.value = ast.Call(ast.Name('getattr', ast.Load()),
                                  [ast.Attribute(ast.Name('self', ast.Load()), '_handler', ast.Load()),
                                   ast.Constant(value=field_name, kind=None)], [])
        for counter, index in enumerate(body_index_to_be_deleted):
            bodies.pop(index - counter)

    @staticmethod
    def _insert_handler_to_init_func(ast_init_fn: ast.FunctionDef, super_index):
        """Insert 'self._handler = global_vars.get('handler')' to init ast.FunctionDef.body"""
        if super_index == -1:
            super_index = 0
        AstModifier.insert_assign_to_function(ast_init_fn, [ScopedValue.create_naming_value("_handler", "self")],
                                              ScopedValue.create_naming_value("get", "global_vars"),
                                              [ScopedValue.create_variable_value("handler")], None,
                                              ast_init_fn.body[super_index], False)

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

        for body in node.body:
            if isinstance(body, ast.FunctionDef):
                if body.name == "__init__":
                    self._process_init_func_ast(stree, body)
                    stree.set_init_func_ast(body)
                elif body.name == "construct":
                    parser: Parser = ParserRegister.instance().get_parser(ast.FunctionDef)
                    parser.process(stree, body)
                else:
                    logger.warning(
                        "Ignoring ast.FunctionDef in ast.ClassDef except __init__ and construct function: %s",
                        body.name)
            else:
                logger.warning("Ignoring unsupported node(%s) in ast.ClassDef.", type(body).__name__)


g_classdef_parser = reg_parser(ClassDefParser())
