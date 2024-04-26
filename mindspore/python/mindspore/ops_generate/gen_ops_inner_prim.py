# Copyright 2023-2024 Huawei Technologies Co., Ltd
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
"""Primitive defined for arg handler."""

from mindspore.ops.primitive import Primitive, prim_attr_register, prim_arg_register
from mindspore._c_expression import typing
from mindspore._c_expression import op_enum


class DtypeToEnum(Primitive):
    r"""
    Convert mindspore dtype to enum.

    Inputs:
        - **op_name** (str) - The op name
        - **arg_name** (str) - The arg name
        - **dtype** (mindspore.dtype) - The data type.

    Outputs:
        An integer.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``
    """

    @prim_attr_register
    def __init__(self):
        """Initialize"""

    def __call__(self, op_name, arg_name, dtype):
        """Run in PyNative mode"""
        if not isinstance(dtype, typing.Type):
            raise TypeError(f"For '{op_name}', the input '{arg_name}' should be mindpsore dtype, but got {dtype}.")
        return typing.type_to_type_id(dtype)


class StringToEnum(Primitive):
    r"""
    Convert string to enum.

    Inputs:
        - **op_name** (str) - The op name
        - **arg_name** (str) - The arg name
        - **enum_str** (str) - The str data.

    Outputs:
        An integer.

    Supported Platforms:
        ``CPU``
    """

    @prim_attr_register
    def __init__(self):
        """Initialize"""

    def __call__(self, op_name, arg_name, enum_str):
        """Run in PyNative mode"""
        if enum_str is None:
            return None
        if not isinstance(enum_str, str):
            raise TypeError(f"For '{op_name}', the input '{arg_name}' should be a str, but got {type(enum_str)}.")
        return op_enum.str_to_enum(op_name, arg_name, enum_str)


class TupleToList(Primitive):
    r"""
    Convert tuple to list.

    Inputs:
        - **x** (tuple) - The input

    Outputs:
        List, has the same elements as the `input`.

    Supported Platforms:
        ``CPU``

    Examples:
        >>> from mindspore.ops_generate import TupleToList
        >>> x = (1, 2, 3)
        >>> result = TupleToList()(x)
        >>> print(result)
        [1, 2, 3]
    """
    @prim_arg_register
    def __init__(self):
        """Initialize TupleToList"""

    def __call__(self, input):
        return list(input)


class ListToTuple(Primitive):
    r"""
    Convert list to tuple.

    Inputs:
        - **x** (list) - The input

    Outputs:
        Tuple, has the same elements as the `input`.

    Supported Platforms:
        ``CPU``

    Examples:
        >>> from mindspore.ops_generate import ListToTuple
        >>> x = [1, 2, 3]
        >>> result = ListToTuple()(x)
        >>> print(result)
        (1, 2, 3)
    """
    @prim_arg_register
    def __init__(self):
        """Initialize TupleToList"""

    def __call__(self, input):
        return tuple(input)
