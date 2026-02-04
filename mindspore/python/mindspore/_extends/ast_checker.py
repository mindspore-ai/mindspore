# Copyright 2025 Huawei Technologies Co., Ltd
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
"""AST type checker compatible with Python 3.7-3.12+."""

import ast
import sys
from typing import Dict, Tuple

_PY312_OR_LATER = sys.version_info >= (3, 12)

if _PY312_OR_LATER:
    _AST_CONSTANT_TYPES: Dict[str, Tuple[type, ...]] = {
        "ast.Str": (str,),
        "ast.Num": (int, float, complex),
        "ast.Bytes": (bytes,),
        "ast.NameConstant": (bool, type(None)),
        "ast.Ellipsis": (type(Ellipsis),),
    }
else:
    _AST_CONSTANT_TYPES: Dict[str, type] = {
        "ast.Str": ast.Str,
        "ast.Num": ast.Num,
        "ast.Bytes": ast.Bytes,
        "ast.NameConstant": ast.NameConstant,
        "ast.Ellipsis": ast.Ellipsis,
    }


class AstChecker:
    """AST type checker compatible with Python 3.7-3.12+."""

    @staticmethod
    def check_type(node: ast.AST, *types: str) -> bool:
        """
        Check if node matches any of the specified AST constant types.

        Compatible with Python 3.7-3.12+. In Python 3.12+, deprecated types
        (ast.Num, ast.Str, etc.) are unified as ast.Constant with different value types.

        Args:
            node: The AST node to check
            *types: Variable length argument of type names as strings,
                   e.g., "ast.Str", "ast.Num", "ast.Bytes", "ast.NameConstant", "ast.Ellipsis"

        Returns:
            True if node matches any of the specified types, False otherwise.

        Raises:
            ValueError: If argument type is not in ["ast.Str", "ast.Num", "ast.Bytes",
            "ast.NameConstant", "ast.Ellipsis"].

        Examples:
            >>> AstChecker.check_type(node, "ast.Str")
            >>> AstChecker.check_type(node, "ast.Num", "ast.Str")
            >>> AstChecker.check_type(node, "ast.Str", "ast.Num", "ast.Bytes")
        """
        if not types:
            return False

        # Python 3.12+ deprecated ast.Num, ast.Str, ast.Bytes, ast.NameConstant, ast.Ellipsis
        # They will be removed in Python 3.14
        # Refer to: https://docs.python.org/3/whatsnew/3.12.html#deprecated
        if _PY312_OR_LATER:
            # All deprecated types are unified as ast.Constant
            if not isinstance(node, ast.Constant):
                return False
            for type_name in types:
                if type_name not in _AST_CONSTANT_TYPES:
                    raise ValueError(
                        f"Unknown AST type: {type_name}. " f"Valid types: {list(_AST_CONSTANT_TYPES.keys())}"
                    )
                value_types = _AST_CONSTANT_TYPES[type_name]
                if isinstance(node.value, value_types):
                    return True
        else:
            for type_name in types:
                if type_name not in _AST_CONSTANT_TYPES:
                    raise ValueError(
                        f"Unknown AST type: {type_name}. " f"Valid types: {list(_AST_CONSTANT_TYPES.keys())}"
                    )
                ast_type = _AST_CONSTANT_TYPES[type_name]
                if isinstance(node, ast_type):
                    return True

        return False
