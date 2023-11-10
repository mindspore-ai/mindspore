# Copyright 2023 Huawei Technologies Co., Ltd
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
"""Convert ast node to other type."""
from typing import Union, List
import ast
from mindspore.rewrite.api.scoped_value import ScopedValue
from mindspore.rewrite.common import error_str


class AstConverter():
    """
    Get information from ast node and convert to other type.
    """
    @staticmethod
    def create_scopedvalue(node: ast.AST) -> ScopedValue:
        """
        Create ScopedValue from an ast node.

        Args:
            node (ast.AST): An ast node.

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
                raise RuntimeError(error_str(f"value of target of ast.Assign should be a ast.Name when target is a "
                                             f"ast.Attribute, but got ast type '{type(scope).__name__}'",
                                             child_node=scope, father_node=node))
            return ScopedValue.create_naming_value(node.attr, scope.id)
        if isinstance(node, (ast.List, ast.Tuple)):
            return AstConverter.create_scopedvalue_from_list(node.elts)
        if isinstance(node, (ast.Constant, ast.NameConstant)):
            return ScopedValue.create_variable_value(node.value)
        if isinstance(node, ast.Num):
            return ScopedValue.create_variable_value(node.n)
        if isinstance(node, (ast.Str, ast.Bytes)):
            return ScopedValue.create_variable_value(node.s)
        raise RuntimeError(error_str(f"create_scopedvalue failed, only support (ast.Name, ast.Attribute, ast.Tuple, "
                                     f"ast.Constant, ast.Num, ast.Str, ast.Bytes) as argument, but got ast type "
                                     f"'{type(node).__name__}'", child_node=node))

    @staticmethod
    def create_scopedvalue_from_list(ast_list: List[ast.AST]) -> ScopedValue:
        """
        Create ScopedValue from a list of ast nodes.

        Args:
            ast_list (List[Union[ast.Constant, ast.Name, ast.Attribute]]): A list of ast nodes.

        Returns:
            An instance of ScopedValue.

        Raises:
            RuntimeError: Only support [ast.Constant, ast.Name, ast.Attribute] as elts of node_list.
        """
        tuple_values = []
        for tuple_elt in ast_list:
            if not isinstance(tuple_elt, (ast.Constant, ast.Name, ast.Attribute)):
                raise RuntimeError(error_str(f"Only support ast.Constant, ast.Name and ast.Attribute as elts of "
                                             f"ast.Tuple, but got ast type {type(tuple_elt).__name__}",
                                             child_node=tuple_elt))
            if isinstance(tuple_elt, ast.Constant):
                tuple_values.append(tuple_elt.value)
            elif isinstance(tuple_elt, ast.Name):
                tuple_values.append(tuple_elt.id)
            elif isinstance(tuple_elt, ast.Attribute):
                tuple_values.append("".join([tuple_elt.value.id, '.', tuple_elt.attr]))
        return ScopedValue.create_variable_value(tuple(tuple_values))

    @staticmethod
    def get_ast_name(ast_node: Union[ast.Name, ast.Attribute]) -> str:
        """Get name from ast.Name or ast.Attribute"""
        if isinstance(ast_node, ast.Name):
            return ast_node.id
        if isinstance(ast_node, ast.Attribute):
            return ast_node.attr
        return ""

    @staticmethod
    def get_ast_target_elems(ast_target: ast.AST):
        target_ast_elems = []
        if isinstance(ast_target, (ast.Tuple, ast.List)):
            for ast_elem in ast_target.elts:
                target_ast_elems.extend(AstConverter.get_ast_target_elems(ast_elem))
        else:
            target_ast_elems.append(ast_target)
        return target_ast_elems
