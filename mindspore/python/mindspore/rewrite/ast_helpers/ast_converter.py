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
import sys

from mindspore import log as logger
from ..api.scoped_value import ScopedValue, ValueType

if sys.version_info >= (3, 9):
    import ast as astunparse # pylint: disable=reimported, ungrouped-imports
else:
    import astunparse

AST_CONSTANTS = (ast.Constant, ast.Num, ast.Str, ast.NameConstant, ast.Bytes)


class AstConverter():
    """
    Get information from ast node and convert to other type.
    """

    @staticmethod
    def get_ast_constant_value(node: Union[ast.Constant, ast.NameConstant, ast.Num, ast.Str, ast.Bytes]):
        """Get value from ast constant"""
        if isinstance(node, (ast.Constant, ast.NameConstant)):
            return node.value
        if isinstance(node, ast.Num):
            return node.n
        if isinstance(node, (ast.Str, ast.Bytes)):
            return node.s
        raise ValueError(f"For get_ast_constant_value, node cannot be {type(node)}")

    @staticmethod
    def create_scopedvalue(node: ast.AST) -> ScopedValue:
        """
        Create ScopedValue from an ast node.

        Args:
            node (ast.AST): An ast node.

        Returns:
            An instance of ScopedValue.
        """
        if isinstance(node, ast.Name):
            return ScopedValue.create_naming_value(node.id)
        if isinstance(node, ast.Attribute):
            scope = node.value
            if not isinstance(scope, ast.Name):
                node_str = astunparse.unparse(node).strip()
                logger.info(f"When creating scopedvalue for '{node_str}', value of ast.Attribute should be ast.Name, "
                            f"but got ast type '{type(scope).__name__}'")
                return ScopedValue(ValueType.UnsupportedValue, "", node_str)
            return ScopedValue.create_naming_value(node.attr, scope.id)
        if isinstance(node, (ast.List, ast.Tuple)):
            return AstConverter.create_scopedvalue_from_list(node.elts)
        if isinstance(node, AST_CONSTANTS):
            value = AstConverter.get_ast_constant_value(node)
            return ScopedValue.create_variable_value(value)
        node_str = astunparse.unparse(node).strip()
        logger.info(f"For '{node_str}', type '{type(node).__name__}' is not supported for ScopedValue now.")
        return ScopedValue(ValueType.UnsupportedValue, "", node_str)

    @staticmethod
    def create_scopedvalue_from_list(ast_list: List[ast.AST]) -> ScopedValue:
        """
        Create ScopedValue from a list of ast nodes.

        Args:
            ast_list (List[Union[ast.Constant, ast.Name, ast.Attribute]]): A list of ast nodes.

        Returns:
            An instance of ScopedValue.
        """
        tuple_values = []
        for tuple_elt in ast_list:
            if not isinstance(tuple_elt, (ast.Constant, ast.Name, ast.Attribute)):
                node_str = astunparse.unparse(tuple_elt).strip()
                logger.info(f"When create scopedvalue for '{node_str}' only support (ast.Constant, ast.Name, "
                            f"ast.Attribute) as elts of ast.Tuple, but got ast type {type(tuple_elt).__name__}")
                return ScopedValue(ValueType.UnsupportedValue, "", node_str)
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
    def ast_tuple_elts_support_scopledvalue(value: ast.Tuple) -> bool:
        """ check whether each element's type in tuple is supported by scopled value. """
        for elt in value.elts:
            if not isinstance(elt, (ast.Name, ast.Attribute, ast.Tuple, ast.Constant, ast.Num, ast.Str, ast.Bytes)):
                return False
        return True

    @staticmethod
    def ast_dict_support_scopledvalue(ast_dict: ast.Dict) -> bool:
        """ check whether each element's type in dict is supported by scopled value. """
        for key in ast_dict.keys:
            if not (isinstance(key, ast.Constant) and isinstance(key.value, str)):
                return False
        for value in ast_dict.values:
            if not isinstance(value, (ast.Name, ast.Attribute, ast.Tuple, ast.Constant, ast.Num, ast.Str, ast.Bytes)):
                return False
        return True

    @staticmethod
    def get_ast_target_elems(ast_target: ast.AST, convert_to_str: bool = False):
        """Get elements in ast"""
        target_ast_elems = []
        if isinstance(ast_target, (ast.Tuple, ast.List)):
            for ast_elem in ast_target.elts:
                target_ast_elems.extend(AstConverter.get_ast_target_elems(ast_elem))
        else:
            if convert_to_str:
                target_ast_elems.append(astunparse.unparse(ast_target).strip())
            else:
                target_ast_elems.append(ast_target)
        return target_ast_elems
