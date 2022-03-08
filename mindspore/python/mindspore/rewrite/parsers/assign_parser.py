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
"""Parse ast.Assign in construct function to node of SymbolTree."""
import ast
import astunparse

from mindspore import log as logger
from ..symbol_tree import SymbolTree
from ..node import Node, TreeNode
from ..parser import Parser
from ..parser_register import reg_parser
from ..api.scoped_value import ScopedValue
from ..symbol_tree_builder import SymbolTreeBuilder


class AssignParser(Parser):
    """Parse ast.Assign in construct function to node of SymbolTree."""

    def target(self):
        """Parse target type."""
        return ast.Assign

    @staticmethod
    def _create_scopedvalue_from_tuple_ast(node: ast.Tuple) -> ScopedValue:
        """
        Create ScopedValue from a tuple ast node.

        Args:
            node (ast.Tuple): A tuple node.

        Returns:
            An instance of ScopedValue.

        Raises:
            RuntimeError: Only support ast.Constant as elts of ast.Tuple.
        """
        tuple_elts = node.elts
        tuple_values = []
        for tuple_elt in tuple_elts:
            if not isinstance(tuple_elt, ast.Constant):
                raise RuntimeError("Only support ast.Constant as elts of ast.Tuple.")
            tuple_values.append(tuple_elt.value)
        return ScopedValue.create_variable_value(tuple(tuple_values))

    @staticmethod
    def _create_scopedvalue(node: ast.expr) -> ScopedValue:
        """
        Create ScopedValue from an ast node.

        Args:
            node (ast.expr): An ast node.

        Returns:
            An instance of ScopedValue.

        Raises:
            RuntimeError: Value of target of ast.Assign should be an ast.Name when target is an ast.Attribute.
            RuntimeError: Type of input node is unsupported.
        """
        if isinstance(node, ast.Name):
            return ScopedValue.create_naming_value(node.id)
        if isinstance(node, ast.Attribute):
            scope = node.value
            if not isinstance(scope, ast.Name):
                raise RuntimeError("value of target of ast.Assign should be a ast.Name when target is a ast.Attribute.")
            return ScopedValue.create_naming_value(node.attr, scope.id)
        if isinstance(node, ast.Tuple):
            return AssignParser._create_scopedvalue_from_tuple_ast(node)
        if isinstance(node, ast.Constant):
            return ScopedValue.create_variable_value(node.value)
        raise RuntimeError("Unsupported ast type to argument:", node)

    @staticmethod
    def _get_func_name(ast_node: ast.Call) -> str:
        """
        Get the func name from ast.Call.

        Args:
            ast_node (ast.Call): Input ast.Call node.

        Returns:
            Func name.

        Raises:
            RuntimeError: Func of input ast node is not ast.Name or ast.Attribute.
        """
        func = ast_node.func
        if isinstance(func, ast.Name):
            return func.id
        if isinstance(func, ast.Attribute):
            return func.attr
        raise RuntimeError("FuncValue is should be Name or a Attribute:", astunparse.unparse(func))

    @staticmethod
    def _get_func_scope(ast_node: ast.Call) -> str:
        """
        Get the func scope from ast.Call.

        Args:
            ast_node (ast.Call): Input ast.Call node.

        Returns:
            Func scope.

        Raises:
            RuntimeError: FuncValue is not an ast.Name when func is an ast.Attribute.
            RuntimeError: Func of input ast node is not ast.Name or ast.Attribute.
        """
        func = ast_node.func
        if isinstance(func, ast.Name):
            return ""
        if isinstance(func, ast.Attribute):
            value = func.value
            if not isinstance(value, ast.Name):
                raise RuntimeError("FuncValue is should be Name:", ast.dump(func))
            return value.id
        raise RuntimeError("FuncValue is should be Name or a Attribute:", ast.dump(func))

    @staticmethod
    def _get_symbol_object(symbol_name, origin_net):
        """
        Get the func scope from ast.Call.

        Args:
            symbol_name (str): Func name.
            origin_net ([nn.Cell]): Network instance.

        Returns:
            Symbol Object.
        """
        var_dict = origin_net.__dict__
        for key, value in var_dict["_cells"].items():
            if key == symbol_name:
                return value

        for key, value in var_dict["_primitives"].items():
            if key == symbol_name:
                return value
        return None

    @staticmethod
    def _create_kwargs(keywords: [ast.keyword]) -> {str, ScopedValue}:
        """
        Transfer ast.Call keywords to a dict of ScopedValue when creating a symbol tree node.

        Args:
            keywords ([ast.keyword]): Keywords of ast.Call node.

        Returns:
            A dict of ScopedValue.
        """
        results = {}
        for keyword in keywords:
            results[keyword.arg] = AssignParser._create_scopedvalue(keyword.value)
        return results

    @staticmethod
    def _convert_ast_call_to_node(ast_node: ast.Call, father_ast_node: ast.Assign, stree: SymbolTree) -> Node:
        """
        Convert ast.Call to a symbol tree node.

        Args:
            ast_node ([ast.Call]): An ast.Call of assign node in construct.
            father_ast_node ([ast.Assign]): Assign node in construct.
            stree ([SymbolTree]): Symbol Tree under parsing.

        Returns:
            An instance of Node in Symbol Tree.

        Raises:
            RuntimeError: kwargs in construct function assign is unsupported.
        """
        target = AssignParser._create_scopedvalue(father_ast_node.targets[0])
        func_name = AssignParser._get_func_name(ast_node)
        if func_name is None or func_name == "":
            raise RuntimeError("function name not exist")
        func_scope = AssignParser._get_func_scope(ast_node)
        func = ScopedValue.create_naming_value(func_name, func_scope)
        call_args = [AssignParser._create_scopedvalue(arg) for arg in ast_node.args]
        call_kwargs = AssignParser._create_kwargs(ast_node.keywords)
        if ast_node.keywords:
            raise RuntimeError("kwargs in construct function assign is unsupported.")

        obj = AssignParser._get_symbol_object(func_name, stree.get_origin_network())
        # need check if node is a callmethod, like: x = len(x)
        # need check if node is a callprimitive, like: x = x * 5
        is_sub_tree = False
        if is_sub_tree:
            stb = SymbolTreeBuilder(obj)
            new_stree = stb.build()
            return TreeNode(new_stree, father_ast_node, [target], func, call_args, call_kwargs, func_name,
                            new_stree.get_origin_network())
        return Node.create_call_cell(obj, father_ast_node, [target], func, call_args, call_kwargs, func_name)

    def process(self, stree: SymbolTree, node: ast.Assign):
        """
        Parse ast.Assign and create a node in symbol tree.
        Will create node when value of ast.Assign is in [ast.Call, ast.Name, ast.Constant, ast.Attribute].
        Will create python node when value of ast.Assign is in
        [ast.BinOp, ast.BoolOp, ast.Subscript, ast.List, ast.Tuple, ast.Dict].
        Other value types are not supported.

        Args:
            stree ([SymbolTree]): Symbol Tree under parsing.
            node ([ast.Assign]): An ast.Assign node.

        Raises:
            RuntimeError: Only support one target in assign now.
            RuntimeError: Unsupported node type in construct function.
        """
        targets = node.targets
        if len(targets) != 1:
            raise RuntimeError("Only support one target in assign now")
        value = node.value
        if isinstance(value, ast.Call):
            node_ = AssignParser._convert_ast_call_to_node(value, node, stree)
            stree.append_origin_field(node_)
        elif isinstance(value, (ast.BinOp, ast.BoolOp, ast.Subscript)):
            logger.warning(f"ops-call({astunparse.unparse(node)}) in assign will be supported in near feature, "
                           f"ignored as a python node now")
            stree.try_append_python_node(node, node)
        elif isinstance(value, (ast.Name, ast.Constant, ast.Attribute, ast.Num, ast.NameConstant, ast.Bytes, ast.Str)):
            if isinstance(value, ast.Name):
                node_name = "name_assign"
            elif isinstance(value, ast.Constant):
                node_name = "constant_assign"
            else:
                node_name = "attribute_assign"
            target = AssignParser._create_scopedvalue(node.targets[0])
            call_args = [AssignParser._create_scopedvalue(value)]
            node_ = Node.create_call_pass_through_method(node, [target], call_args, {}, node_name)
            stree.append_origin_field(node_)
        elif isinstance(value, (ast.List, ast.Tuple, ast.Dict)):
            # add these as callmethod node if necessary
            stree.try_append_python_node(node, node)
        else:
            raise RuntimeError(f"Unsupported statement({astunparse.unparse(node)}) in construct function!")


g_assign_parser = reg_parser(AssignParser())
