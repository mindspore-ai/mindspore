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
"""Ast utils for create or update ast node."""
from typing import Optional, List
import ast

from ..api.scoped_value import ScopedValue, ValueType


class AstModifier(ast.NodeTransformer):
    """Ast utils for create or update ast node."""

    @staticmethod
    def erase_ast_from_function(ast_func: ast.FunctionDef, to_erase: ast.AST) -> bool:
        """
        Erase ast node from ast.FunctionDef.

        Args:
            ast_func (ast.FunctionDef): From which to search to_erase-node and erase.
            to_erase (ast.AST): Node to be erased.

        Returns:
            A bool if to_erase-node been found and been erased.
        """
        return AstModifier.erase_ast_from_bodies(ast_func.body, to_erase)

    @staticmethod
    def erase_ast_from_bodies(ast_bodies: List[ast.AST], to_erase: ast.AST) -> bool:
        """Erase ast node from ast bodies."""
        for body in ast_bodies:
            if id(body) == id(to_erase):
                ast_bodies.remove(body)
                return True
        return False

    @staticmethod
    def erase_func_from_class_by_name(ast_class: ast.ClassDef, func_name: str):
        """
        Erase ast FunctionDef from ast.ClassDef by name.

        Args:
            ast_class (ast.ClassDef): From which to search to_erase-node and erase.
            func_name (str): Function name to be erased.
        """
        for body in ast_class.body:
            if isinstance(body, ast.FunctionDef) and body.name == func_name:
                ast_class.body.remove(body)

    @staticmethod
    def insert_sub_ast(ast_father: ast.AST, ast_son: ast.AST, index_ast: Optional[ast.AST] = None,
                       insert_before=True) -> ast.AST:
        """
        Insert an ast node into another ast node's body.

        Args:
            ast_father (ast.AST): Where new ast node to be inserted into.
            ast_son (ast.AST): An ast node to be inserted in.
            index_ast ([ast.AST, optional]): An ast_node indicates a position in 'ast_father' where new ast node to be
                inserted into. Default is None which means append new ast node to body of 'ast_father'.
            insert_before (bool): A bool indicates at before or at after of 'index_ast' where new ast node to be
                inserted into. Only valid when 'index_ast' is not None. Default is True which means inserting new ast
                node before 'index_ast'.

        Returns:
            An instance of ast.AST which has been inserted into 'ast_father'.

        Raises:
            ValueError: If 'ast_father' has no attribute named 'body'.
            RuntimeError: If 'index_ast' is not contained in 'ast_father'.
        """
        if not hasattr(ast_father, "body"):
            raise ValueError("Input ast_father has no attribute body:", type(ast_father))
        if index_ast is None:
            ast_father.body.append(ast_son)
            ast.fix_missing_locations(ast_father)
            return ast_son
        for index in range(0, len(ast_father.body)):
            if id(ast_father.body[index]) == id(index_ast):
                if insert_before:
                    ast_father.body.insert(index, ast_son)
                else:
                    ast_father.body.insert(index + 1, ast_son)
                ast.fix_missing_locations(ast_father)
                return ast_son
        raise RuntimeError("index_ast is not contained in ast_father")

    @staticmethod
    def insert_class_into_module(ast_mod: ast.Module, ast_class: ast.ClassDef, index_ast: Optional[ast.AST] = None,
                                 insert_before=True) -> ast.ClassDef:
        """
        Insert an ast.ClassDef into an ast.Module.

        Args:
            ast_mod (ast.Module): Where new ast.ClassDef to be inserted into.
            ast_class (ast.ClassDef): ClassDef to be inserted.
            index_ast ([ast.AST, optional]): An ast_node indicates a position in 'ast_mod' where new ast.ClassDef node
                to be inserted into. Default is None which means append new ast.ClassDef into 'ast_mod'.
            insert_before (bool): A bool indicates at before or at after of 'index_ast' where new ast.ClassDef node to
                be inserted into. Only valid when 'index_ast' is not None. Default is True which means inserting new
                ast.ClassDef before 'index_ast'.

        Returns:
            An instance of ast.ClassDef which has been inserted into 'ast_mod'.
        """
        return AstModifier.insert_sub_ast(ast_mod, ast_class, index_ast, insert_before)

    @staticmethod
    def insert_assign_to_function(ast_func: ast.FunctionDef, targets: [ScopedValue], expr: ScopedValue,
                                  args: [ScopedValue] = None, kwargs: {str, ScopedValue}=None,
                                  index_ast: Optional[ast.AST] = None, insert_before=True) -> ast.AST:
        """
        Insert an ast.Assign into an ast.FunctionDef.

        Args:
            ast_func (ast.FunctionDef): Where new ast.Assign to be inserted into.
            targets ([ScopedValue]): Targets of ast.Assign.
            expr (ScopedValue): Func of ast.Call which is value of new ast.Assign.
            args ([ScopedValue]): Args of ast.Call which is value of new ast.Assign.
            kwargs ({str, ScopedValue}): Kwargs of ast.Call which is value of new ast.Assign.
            index_ast ([ast.AST, optional]): An ast_node indicates a position in 'ast_func' where new ast.Assign node to
                be inserted into. Default is None which means append new ast.Assign into 'ast_func'.
            insert_before (bool): A bool indicates at before or at after of 'index_ast' where new ast.Assign node to be
                inserted into. Only valid when 'index_ast' is not None. Default is True which means inserting new
                ast.Assign before 'index_ast'.

        Returns:
            An instance of ast.Assign which has been inserted into 'ast_func'.

        Raises:
            RuntimeError: If 'index_ast' is not contained in 'ast_func'.
        """
        assign = AstModifier.create_call_assign(targets, expr, args, kwargs)
        return AstModifier.insert_assign_ast_to_function(ast_func, assign, index_ast, insert_before)

    @staticmethod
    def insert_assign_ast_to_function(ast_func: ast.FunctionDef, ast_assign: ast.Assign,
                                      index_ast: Optional[ast.AST] = None, insert_before=True) -> ast.AST:
        """
        Insert an ast.Assign into an ast.FunctionDef.

        Args:
            ast_func (ast.FunctionDef): Where new ast.Assign to be inserted into.
            ast_assign (ast.Assign): An instance of ast.Assign to be inserted in.
            index_ast ([ast.AST, optional]): An ast_node indicates a position in 'ast_func' where new ast.Assign node to
                be inserted into. Default is None which means append new ast.Assign to 'ast_func'.
            insert_before (bool): A bool indicates at before or at after of 'index_ast' where new ast.Assign node to be
                inserted into. Only valid when 'index_ast' is not None. Default is True which means inserting new
                ast.Assign before 'index_ast'.

        Returns:
            An instance of ast.Assign which has been inserted into 'ast_func'.

        Raises:
            RuntimeError: If 'index_ast' is not contained in 'ast_func'.
        """
        # Insert ast at the frontmost position of function body when index_ast is an argument of function
        arguments: ast.arguments = ast_func.args
        if index_ast and arguments.args:
            for arg in arguments.args:
                if id(arg) == id(index_ast):
                    ast_func.body.insert(0, ast_assign)
                    ast.fix_missing_locations(ast_func)
                    return ast_assign
        # Insert ast at position specified by index_ast in function body
        ast_assign = AstModifier.insert_assign_ast_to_bodies(ast_func.body, ast_assign, index_ast, insert_before)
        ast.fix_missing_locations(ast_assign)
        return ast_assign

    @staticmethod
    def insert_assign_ast_to_bodies(ast_bodies: List[ast.AST], ast_assign: ast.Assign,
                                    index_ast: Optional[ast.AST] = None, insert_before=True) -> ast.AST:
        """Insert ast at position specified by index_ast of ast_bodies"""
        # Append ast_assign to ast_bodies when index_ast is None
        if index_ast is None:
            ast_bodies.append(ast_assign)
            return ast_assign
        # Append ast_assign to ast_bodies
        for index, body in enumerate(ast_bodies):
            if id(body) == id(index_ast):
                if not insert_before:
                    index += 1
                ast_bodies.insert(index, ast_assign)
                ast.fix_missing_locations(body)
                break
        else:
            raise ValueError("insert position is not contained in ast_bodies")
        return ast_assign

    @staticmethod
    def append_arg_to_function(ast_func: ast.FunctionDef, ast_arg: ast.arg) -> ast.AST:
        """
        Append an ast.arg to an ast.FunctionDef (e.g. self.construct).

        Args:
            ast_func (ast.FunctionDef): An instance of ast.FunctionDef which is "construct" function of network.
            ast_arg (ast.arg): An instance of ast.arg to be inserted in.

        Returns:
            An instance of ast.arg which has been appended to 'ast_func'.

        Raises:
            RuntimeError: If 'ast_arg' is not an instance of ast_arg.
        """
        if not isinstance(ast_arg, ast.arg):
            raise RuntimeError("ast_arg should be an instance of ast.arg.")
        arguments: ast.arguments = ast_func.args
        args: [ast.arg] = arguments.args
        args.append(ast_arg)
        defaults = arguments.defaults
        arg_default = ast.Constant(value=None, kind=None)
        defaults.append(arg_default)
        return ast_arg

    @staticmethod
    def append_global_vars_expr_to_init(init_func: ast.FunctionDef, targets: [ScopedValue],
                                        field: str) -> ast.AST:
        """
        Append an ast.Assign to an ast.FunctionDef which is function named "__init__" in network. Value of new
        ast.Assign is an ast.Call represents get an object from global_vars dict.

        While user inserting a custom op, the instance of new custom op is saved in a dict named global_vars. Rewrite
        need to get the custom op instance from global_vars in new "__init__" function of network:
        self.var1 = global_vars.get("var1")

        Args:
            init_func (ast.FunctionDef): An instance of ast.FunctionDef which is "__init__" function of network.
            targets ([ScopedValue]): Targets of ast.Assign.
            field (str): A string represents name of new custom op field.

        Returns:
            An instance of ast.Assign which has been appended to 'init_func'.
        """
        return AstModifier.insert_assign_to_function(init_func, targets=targets,
                                                     expr=ScopedValue(ValueType.NamingValue, "", "getattr"),
                                                     args=[ScopedValue(ValueType.NamingValue, "obj"),
                                                           ScopedValue.create_variable_value(field)])


    @staticmethod
    def create_call_assign(targets: [ScopedValue], expr: ScopedValue, args: [ScopedValue],
                           kwargs: {str, ScopedValue}) -> ast.Assign:
        """
        Create an instance of ast.Assign whose value must ba a ast.Call.

        Args:
            targets ([ScopedValue]): Targets of ast.Assign.
            expr (ScopedValue): Func of ast.Call which is value of new ast.Assign.
            args ([ScopedValue]): Args of ast.Call which is value of new ast.Assign.
            kwargs ({str, ScopedValue}): Kwargs of ast.Call which is value of new ast.Assign.

        Returns:
            An instance of ast.Assign.

        Raises:
            RuntimeError: If 'targets' is None.
            RuntimeError: If value_type of element of 'targets' is not ValueType.NamingValue.

        """
        if targets is None:
            raise RuntimeError("'Targets should not be None.")
        targets_list = []
        for target in targets:
            if target.type != ValueType.NamingValue:
                raise RuntimeError("Target must be a right-value, got: ", target)
            if target.scope:
                ast_target = ast.Attribute(ast.Name(target.scope, ast.Load()), target.value, ast.Store())
            else:
                ast_target = ast.Name(target.value, ast.Store())
            targets_list.append(ast_target)
        call = AstModifier.create_call(expr, args, kwargs)

        if len(targets) == 1:
            result = ast.Assign(targets=[targets_list[0]], value=call)
        elif len(targets) > 1:
            ast_targets = ast.Tuple(elts=targets_list, ctx=ast.Store())
            result = ast.Assign(targets=[ast_targets], value=call)
        ast.fix_missing_locations(result)
        return result

    @staticmethod
    def _create_arg_by_constant_value(value: ScopedValue):
        """
        Create an instance of ast.Constant.

        Args:
            value (ScopedValue): value used to create arg.

        Raises:
            RuntimeError: if scope of value is not empty.
            TypeError: type of arg is not ValueType.ConstantValue

        Returns:
            ast.Constant: An instance of ast.Constant
        """
        if value.type == ValueType.ConstantValue:
            if value.scope:
                raise RuntimeError("For arg the scope should be empty")
            return ast.Constant(value=value.value, kind=None)
        raise TypeError("Type of arg only support ValueType.ConstantValue, but got {type(value)}")

    @staticmethod
    def _create_list_or_tuple(value: ScopedValue):
        """
        Create an instance of ast.List or ast.Tuple.

        Args:
            value (ScopedValue): value used to create ast node.

        Returns:
            ast.List or ast.Tuple: An instance of ast.List or ast.Tuple.
        """
        elts = []
        for v in value.value:
            elts.append(AstModifier._create_arg_by_constant_value(v))
        if isinstance(value, list):
            return ast.List(elts=elts)
        return ast.Tuple(elts=elts)

    @staticmethod
    def _create_keyword(arg: str, value: ScopedValue):
        """
        Create an instance of ast.keyword.

        Args:
            arg (str): key of keyword.
            value (ScopedValue): value used to create ast.keywrod instance.

        Raises:
            RuntimeError:  if scope of value is not empty.
            TypeError: type of arg is not ValueType.ConstantValue

        Returns:
            ast.keyword: a instance of ast.keyword.
        """
        if value.scope:
            raise RuntimeError("value.scope should be empty")
        if value.type == ValueType.ConstantValue:
            v = ast.Constant(value=value.value, kind=None)
        elif value.type in (ValueType.ListValue, ValueType.TupleValue):
            v = AstModifier._create_list_or_tuple(value)
        else:
            raise TypeError("Type of keyword value only support [ValueType.ConstantValue, ValueType.ListValue, "
                            f"ValueType.TupleValue], but got {type(value)}")
        return ast.keyword(arg=arg, value=v)

    @staticmethod
    def _create_call_args(args: [ScopedValue]) -> [ast.AST]:
        """
        Create a list of ast.AST as args of ast.Call from a list of `ScopedValue`.

        Args:
            args (list[ScopedValue]): Args of ast.Call.

        Returns:
            A list of ast.AST as args of ast.Call.

        Raises:
            RuntimeError: If element of 'args' is not an instance of `ScopedValue`.
            RuntimeError: If value_type of element of 'args' is `ValueType.CustomObjValue`.
        """

        if args is None:
            return []
        results = []
        for arg in args:
            if not isinstance(arg, ScopedValue):
                raise TypeError("arg should be ScopedValue, got: ", type(arg))
            if arg.type == ValueType.ConstantValue:
                results.append(AstModifier._create_arg_by_constant_value(arg))
            elif arg.type == ValueType.NamingValue:
                if arg.scope:
                    results.append(ast.Attribute(ast.Name(arg.scope, ast.Load()), arg.value, ast.Store()))
                else:
                    results.append(ast.Name(arg.value, ast.Store()))
            elif arg.type in (ValueType.ListValue, ValueType.TupleValue):
                results.append(AstModifier._create_list_or_tuple(arg))
            else:
                raise RuntimeError("Please handle custom-object first")
        return results

    @staticmethod
    def _create_call_kwargs(kwargs: {str: ScopedValue}) -> [ast.keyword]:
        """
        Create a list of ast.keyword as kwargs of ast.Call from a dict of string to `ScopedValue`.

        Args:
            kwargs (dict{str: ScopedValue}): Kwargs of ast.Call.

        Returns:
            A list of ast.AST as args of ast.Call.

        Raises:
            RuntimeError: If element of 'args' is not an instance of `ScopedValue`.
            RuntimeError: If value_type of element of 'args' is `ValueType.CustomObjValue`.
        """

        if kwargs is None:
            return []
        results = []
        for arg, value in kwargs.items():
            if not isinstance(value, ScopedValue):
                raise TypeError("value should be ScopedValue, got: ", type(value))
            if value.type in (ValueType.ConstantValue, ValueType.ListValue, ValueType.TupleValue):
                results.append(AstModifier._create_keyword(arg, value))
            elif value.type == ValueType.NamingValue:
                if value.scope:
                    results.append(ast.keyword(arg=arg, value=ast.Attribute(ast.Name(value.scope, ast.Load()),
                                                                            value.value, ast.Store())))
                else:
                    results.append(ast.keyword(arg=arg, value=ast.Name(value.value, ast.Store())))
            else:
                raise RuntimeError("Please handle custom-object first")
        return results

    @staticmethod
    def create_call(expr: ScopedValue, args: [ScopedValue] = None, kwargs: {str: ScopedValue}=None) -> ast.Call:
        """
        Create an instance of ast.Call.

        Args:
            expr (ScopedValue): Func of ast.Call.
            args ([ScopedValue]): Args of ast.Call.
            kwargs ({str, ScopedValue}): Kwargs of ast.Call.

        Returns:
            An instance of ast.Call.

        Raises:
            RuntimeError: If value_type of 'expr' is ValueType.CustomObjValue.
            RuntimeError: If value_type of 'expr' is not ValueType.NamingValue.
            TypeError: If expr is not an instance of ScopedValue.
        """
        if not isinstance(expr, ScopedValue):
            raise TypeError("expr should be ScopedValue, got: ", type(expr))
        if expr.type == ValueType.CustomObjValue:
            raise RuntimeError("Please handle custom-object first")
        if expr.type != ValueType.NamingValue:
            raise RuntimeError("Expr must not be a constant, because constant can not been called: ", expr.type)
        if expr.scope:
            ast_func = ast.Attribute(ast.Name(expr.scope, ast.Load()), expr.value, ast.Store())
        else:
            ast_func = ast.Name(expr.value, ast.Store())

        ast_args = AstModifier._create_call_args(args)
        keywords = AstModifier._create_call_kwargs(kwargs)
        result = ast.Call(func=ast_func, args=ast_args, keywords=keywords)
        ast.fix_missing_locations(result)
        return result

    @staticmethod
    def update_arg_value(src_argument: ScopedValue, dst_ast: ast.AST):
        """
        Update 'arg_value' by 'input_argument'

        Args:
            src_argument (ScopedValue): An instance of ScopedValue represents new argument.
            dst_ast (ast.AST): Ast node to be updated by ScopedValue.

        Raises:
            TypeError: Input src_argument is not a ScopedValue
            RuntimeError: If 'dst_ast' is an instance of ast.Constant but type of 'src_argument' is not
                ValueType.ConstantValue.
            RuntimeError: If 'dst_ast' is an instance of ast.Name or ast.Attribute but type of 'src_argument' is not
                ValueType.NamingValue.
            RuntimeError: When 'dst_ast' is an instance of ast.Name, scope of 'src_argument' is not empty.
            RuntimeError: When 'dst_ast' is an instance of ast.Attribute, value of 'dst_ast' is not an instance of
                ast.Name.
            RuntimeError: If 'dst_ast' is an instance of ast.Tuple but type of 'src_argument' is not
                ValueType.TupleValue.
            RuntimeError: If 'dst_ast' is an instance of ast.Constant, ast.Name, ast.Attribute or ast.Tuple.
            RuntimeError: When 'dst_ast' is an instance of ast.Tuple, length of elts of 'dst_ast' is not equal to length
                of value of 'src_argument'.
        """
        if not isinstance(src_argument, ScopedValue):
            raise TypeError("src_argument should be ScopedValue, got: ", type(src_argument))
        if isinstance(dst_ast, (ast.Constant, ast.Num, ast.Str)):
            AstModifier.update_arg_value_constant(src_argument, dst_ast)
            return
        if isinstance(dst_ast, ast.Name):
            if src_argument.type not in [ValueType.NamingValue, ValueType.ConstantValue]\
                or not isinstance(src_argument.value, str):
                raise RuntimeError("src_argument.type should be ValueType.NamingValue or ValueType.ConstantValue, "
                                   "but got:", type(src_argument.value).__name__)
            if src_argument.scope:
                raise RuntimeError("src_argument.scope should be empty")
            dst_ast.id = src_argument.value
            return
        if isinstance(dst_ast, ast.Attribute):
            if src_argument.type != ValueType.NamingValue:
                raise RuntimeError("src_argument.type should equal to ValueType.NamingValue")
            attr_value = dst_ast.value
            if not isinstance(attr_value, ast.Name):
                raise RuntimeError("Only support ast.Name as value of attribute ", type(attr_value))
            attr_value.id = src_argument.scope
            dst_ast.attr = src_argument.value
            return
        if isinstance(dst_ast, ast.Tuple):
            if src_argument.type != ValueType.TupleValue:
                raise RuntimeError("src_argument.type should equal to ValueType.TupleValue")
            if len(src_argument.value) != len(dst_ast.elts):
                raise RuntimeError("src_argument.value and elts in ast should have same length")
            for elt_index, elt in enumerate(dst_ast.elts):
                AstModifier.update_arg_value(src_argument.value[elt_index], elt)
            return
        raise RuntimeError("keyword value type is not supported", type(dst_ast))

    @staticmethod
    def update_arg_value_constant(src_argument: ScopedValue, dst_ast: ast.AST):
        """Update 'arg_value' of type constant by 'input_argument'"""
        if src_argument.type != ValueType.ConstantValue:
            raise RuntimeError("src_argument should be a ConstantValue, got:", str(src_argument.type))
        if isinstance(dst_ast, ast.Constant):
            dst_ast.value = src_argument.value
            return
        if isinstance(dst_ast, ast.Num):
            if not isinstance(src_argument.value, (int, float)):
                raise RuntimeError("Type of src_argument should be int or float, but got:",
                                   type(src_argument.value).__name__)
            dst_ast.n = src_argument.value
            return
        if isinstance(dst_ast, ast.Str):
            if not isinstance(src_argument.value, str):
                raise RuntimeError("Type of src_argument should be str, but got:", type(src_argument.value).__name__)
            dst_ast.s = src_argument.value
            return
