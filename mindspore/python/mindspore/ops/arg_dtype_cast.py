# This is the Python adaptation and derivative work of Myia (https://github.com/mila-iqia/myia/).
#
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
"""Operator argument data type cast function."""
from enum import Enum


class TypeCastKind(Enum):
    INT_TO_TUPLE = 1
    INT_OR_TUPLE_TO_LIST = 2
    SCALAR_TO_TENSOR = 3


def type_it(src_data, cast_type):
    """
    cast operator argument data type.
    """
    if cast_type == TypeCastKind.INT_TO_TUPLE:
        if isinstance(src_data, tuple):
            return src_data

        if isinstance(src_data, int):
            return (src_data,)

        raise TypeError(f'{src_data} is the wrong data type.')

    if cast_type == TypeCastKind.INT_OR_TUPLE_TO_LIST:
        if isinstance(src_data, list):
            return src_data

        if isinstance(src_data, int):
            return [
                src_data,
            ]

        if isinstance(src_data, tuple):
            dst_list = [item for item in src_data]
            return dst_list

        raise TypeError(f'{src_data} is the wrong data type.')

    if cast_type == TypeCastKind.SCALAR_TO_TENSOR:
        if isinstance(src_data, tensor):
            return src_data

        if isinstance(src_data, scalar):
            return Tensor(input_data=src_data, shape=None, dtype=src_data.dtype)

        raise TypeError(f'{src_data} is the wrong data type.')

    raise TypeError("Unsupported type cast")


class ArgHandleKind(Enum):
    ToKernelSize = 1
    ToStrides = 2
    ToDilations = 3
    ToPaddings = 4


def expand_int_tuple(arg_name, arg_value, dst_len):
    """
    Process an int number or a tuple with one, two or four ints.
    If arg is an int number or a tuple with one int number, s
    return (arg, arg) if dst_len==2 or (1, 1, arg, arg) if dst_len==4.
    If arg is a tuple with two int number, return (arg, arg) if dst_len==2.
    If arg is a tuple with four int number, return (arg[2], arg[3]) if dst_len==2 or arg if dst_len==4.
    """
    if isinstance(arg_value, int) or len(arg_value) == 1:
        if dst_len == 2:
            return (arg_value, arg_value)
        if dst_len == 4:
            return (arg_value, arg_value, arg_value, arg_value)
    if len(arg_value) == 2:
        if dst_len == 2:
            return arg_value
    if len(arg_value) == 4:
        if dst_len == 2:
            return (arg_value[2], arg_value[3])
        if dst_len == 4:
            return arg_value
    raise TypeError(f"For arg '{arg_name}', the value must be an int number or a tuple of two "
                    "or four int numbers.")


def arg_handle(arg_name, arg_value, handle_type):
    if handle_type == ArgHandleKind.ToKernelSize or handle_type == ArgHandleKind.ToStrides or \
            handle_type == ArgHandleKind.ToDilations:
        return expand_int_tuple(arg_name, arg_value, 2)
    if handle_type == ArgHandleKind.ToPaddings:
        return expand_int_tuple(arg_name, arg_value, 4)
    raise TypeError(f"Unsupported handle type for arg '{arg_name}'")
