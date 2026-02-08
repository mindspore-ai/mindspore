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

import mindspore as ms
from mindspore import ops
from mindspore.common.tensor import Tensor
from mindspore.common import dtype as mstype
from mindspore.ops.operations._sequence_ops import TensorToScalar, TensorToTuple
from mindspore._c_expression import OpDtype
from mindspore._c_expression import typing
from mindspore._c_expression import op_enum
from mindspore.ops.primitive import Primitive, prim_attr_register, prim_arg_register

tensor_to_tuple_ = TensorToTuple()
tensor_to_scalar_ = TensorToScalar()


class IsDimUnKnown(Primitive):
    @prim_attr_register
    def __init__(self):
        super().__init__("IsDimUnKnown")

    def __call__(self, data):
        return False

is_sequence_shape_unknown = IsDimUnKnown()


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
        >>> from mindspore.ops._utils.arg_dtype_cast import TupleToList
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
        >>> from mindspore.ops._utils.arg_dtype_cast import ListToTuple
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


tuple_to_list = TupleToList()
list_to_tuple = ListToTuple()


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
            raise TypeError(
                f"For '{op_name}', the input '{arg_name}' should be mindspore dtype, but got {dtype}.")
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
            raise TypeError(
                f"For '{op_name}', the input '{arg_name}' should be a str, but got {type(enum_str)}.")
        return op_enum.str_to_enum(op_name, arg_name, enum_str)


def int_to_float(data):
    return float(data)


def scalar_to_tuple(data):
    return (data,)


def tensor_to_tuple(data):
    # Since tuple is not supported for precision conversion during KernelSelect, the original int32 tensor input cases
    # would be failed. Thus, the tuple precision is raised from int32 to int64 at frontend. But sequence data type cast
    # must be adapted in future version.
    if data.dtype == ms.int32:
        data = ops.cast(data, ms.int64)
    return tensor_to_tuple_(data)


def scalar_to_tensor(data):
    if isinstance(data, bool):
        return ops.scalar_to_tensor(data, ms.bool_)
    if isinstance(data, int):
        return ops.scalar_to_tensor(data, ms.int32)
    if isinstance(data, float):
        return ops.scalar_to_tensor(data, ms.float32)
    return ops.scalar_to_tensor(data)


def tuple_to_tensor(data):
    return ops.tuple_to_array(data)


def list_to_tensor(data):
    return ops.tuple_to_array(list_to_tuple(data))


# There will be some problems in using OpDtype.xxx directly in GRAPH_MODE, so convert it to int.
# type
DT_TYPE_VAL = int(OpDtype.DT_TYPE)
# scalar
DT_INT_VAL = int(OpDtype.DT_INT)
DT_FLOAT_VAL = int(OpDtype.DT_FLOAT)
DT_BOOL_VAL = int(OpDtype.DT_BOOL)
DT_NUMBER_VAL = int(OpDtype.DT_NUMBER)
# tuple
DT_TUPLE_BOOL_VAL = int(OpDtype.DT_TUPLE_BOOL)
DT_TUPLE_INT_VAL = int(OpDtype.DT_TUPLE_INT)
DT_TUPLE_FLOAT_VAL = int(OpDtype.DT_TUPLE_FLOAT)
DT_TUPLE_NUMBER_VAL = int(OpDtype.DT_TUPLE_NUMBER)
DT_TUPLE_TENSOR_VAL = int(OpDtype.DT_TUPLE_TENSOR)
DT_TUPLE_STR_VAL = int(OpDtype.DT_TUPLE_STR)
DT_TUPLE_ANY_VAL = int(OpDtype.DT_TUPLE_ANY)
# list
DT_LIST_BOOL_VAL = int(OpDtype.DT_LIST_BOOL)
DT_LIST_INT_VAL = int(OpDtype.DT_LIST_INT)
DT_LIST_FLOAT_VAL = int(OpDtype.DT_LIST_FLOAT)
DT_LIST_NUMBER_VAL = int(OpDtype.DT_LIST_NUMBER)
DT_LIST_TENSOR_VAL = int(OpDtype.DT_LIST_TENSOR)
DT_LIST_STR_VAL = int(OpDtype.DT_LIST_STR)
DT_LIST_ANY_VAL = int(OpDtype.DT_LIST_ANY)
# tensor
DT_TENSOR_VAL = int(OpDtype.DT_TENSOR)

dtype_to_string = {
    DT_INT_VAL: "int",
    DT_FLOAT_VAL: "float",
    DT_BOOL_VAL: "bool",
    DT_NUMBER_VAL: "number",
    DT_TENSOR_VAL: "Tensor",
    DT_TUPLE_BOOL_VAL: "tuple of bool",
    DT_TUPLE_INT_VAL: "tuple of int",
    DT_TUPLE_FLOAT_VAL: "tuple of float",
    DT_TUPLE_NUMBER_VAL: "tuple of number",
    DT_TUPLE_TENSOR_VAL: "tuple of tensor",
    DT_TUPLE_STR_VAL: "tuple of string",
    DT_TUPLE_ANY_VAL: "tuple of Any",
    DT_LIST_BOOL_VAL: "list of bool",
    DT_LIST_INT_VAL: "list of int",
    DT_LIST_FLOAT_VAL: "list of float",
    DT_LIST_NUMBER_VAL: "list of number",
    DT_LIST_TENSOR_VAL: "list of Tensor",
    DT_LIST_STR_VAL: "list of string",
    DT_LIST_ANY_VAL: "list of Any"
}


def is_tuple(type_id):
    """
    Check type id is tuple.
    """
    return type_id in (DT_TUPLE_BOOL_VAL, DT_TUPLE_INT_VAL, DT_TUPLE_FLOAT_VAL, DT_TUPLE_NUMBER_VAL,
                       DT_TUPLE_TENSOR_VAL, DT_TUPLE_STR_VAL, DT_TUPLE_ANY_VAL)


def is_list(type_id):
    """
    Check type id is list.
    """
    return type_id in (DT_LIST_BOOL_VAL, DT_LIST_INT_VAL, DT_LIST_FLOAT_VAL, DT_LIST_NUMBER_VAL,
                       DT_LIST_TENSOR_VAL,
                       DT_LIST_STR_VAL, DT_LIST_ANY_VAL)


def is_integral_mstype(dtype):
    """
    Check dtype is integral.
    """
    integral_mstype = (mstype.int8, mstype.int16, mstype.int32, mstype.int64,
                       mstype.uint8, mstype.uint16, mstype.uint32, mstype.uint64)
    return dtype in integral_mstype


def is_floating_point_mstype(dtype):
    """
    Check dtype is floating point.
    """
    fp_mstype = (mstype.float16, mstype.float32, mstype.float64)
    return dtype in fp_mstype


def is_number(type_id):
    """
    Check type id is number.
    """
    return type_id in (DT_INT_VAL, DT_FLOAT_VAL, DT_BOOL_VAL, DT_NUMBER_VAL)


def is_instance_of(data, type_id):
    """
    Instead isinstance(obj, type).
    """
    if type_id == DT_INT_VAL:
        return isinstance(data, int)
    if type_id == DT_FLOAT_VAL:
        return isinstance(data, float)
    if type_id == DT_BOOL_VAL:
        return isinstance(data, bool)
    if is_number(type_id):
        return isinstance(data, (int, float, bool))
    if is_tuple(type_id):
        return isinstance(data, tuple)
    if is_list(type_id):
        return isinstance(data, list)
    if type_id == DT_TENSOR_VAL:
        return isinstance(data, Tensor)
    return False


def is_instance_in(data, type_id):
    """
    Instead isinstance(obj, tuple_types).
    """
    if not isinstance(type_id, tuple):
        return is_instance_of(data, type_id)
    for type_id_i in type_id:
        if is_instance_of(data, type_id_i):
            return True
    return False


def get_support_dtype_list(src_type, dst_type):
    """
    Get support dtype list.
    """
    support_list = ""
    if isinstance(src_type, tuple):
        for dtype in src_type:
            support_list += dtype_to_string.get(dtype) + ", "
    else:
        support_list += dtype_to_string.get(src_type) + ", "
    support_list += dtype_to_string.get(dst_type)
    return support_list


def tensor_to_number(data, dst_type, op_name):
    """Convert tensor to python number"""
    if dst_type == DT_INT_VAL:
        data = ops.cast(data, ms.int64)
    elif dst_type == DT_FLOAT_VAL:
        data = ops.cast(data, ms.float32)
    elif dst_type == DT_NUMBER_VAL:
        src_type = data.dtype
        if src_type in (ms.uint8, ms.uint16, ms.uint32, ms.uint64,
                        ms.int8, ms.int16, ms.int32, ms.int64):
            data = ops.cast(data, ms.int64)
        elif src_type in (ms.bfloat16, ms.float16, ms.float32, ms.float64):
            data = ops.cast(data, ms.float32)
    return tensor_to_scalar_(data)


_tensor_to_scalar = TensorToScalar()


def normalize_int_tensor(op_name, arg_name, data):
    """Normalize int tensor."""
    if not is_integral_mstype(data.dtype):
        raise ValueError(f"For '{op_name}', the input '{arg_name}' should be integral tensor, but got {data.dtype}.")
    data = ops.cast(data, ms.int64)
    return _tensor_to_scalar(data)


def normalize_int_sequence(op_name, arg_name, data, to_tuple=True):
    """Normalize mixed int sequence."""
    if is_sequence_shape_unknown(data):
        return data
    res = []
    for x in data:
        if isinstance(x, int):
            res.append(x)
        elif isinstance(x, Tensor):
            res.append(normalize_int_tensor(op_name, arg_name, x))
        else:
            return data # invalid input, pass it through
    return tuple(res) if to_tuple else res


def do_type_cast(data, dst_type, op_name):
    """Type conversion."""
    if is_instance_of(data, dst_type):
        return data
    if dst_type == DT_FLOAT_VAL:
        if isinstance(data, int):
            return int_to_float(data)
    elif is_tuple(dst_type):
        if isinstance(data, (int, float, bool)):
            return scalar_to_tuple(data)
        if isinstance(data, list):
            return list_to_tuple(data)
        if isinstance(data, Tensor):
            return tensor_to_tuple(data)
    elif is_list(dst_type):
        if isinstance(data, (int, float, bool)):
            return tuple_to_list(scalar_to_tuple(data))
        if isinstance(data, tuple):
            return tuple_to_list(data)
        if isinstance(data, Tensor):
            return tuple_to_list(tensor_to_tuple(data))
    elif dst_type == DT_TENSOR_VAL:
        if isinstance(data, (int, float, bool)):
            return scalar_to_tensor(data)
        if isinstance(data, tuple):
            return tuple_to_tensor(data)
        if isinstance(data, list):
            return list_to_tensor(data)
    elif is_number(dst_type):
        if isinstance(data, Tensor):
            return tensor_to_number(data, dst_type, op_name)
    raise TypeError("Type conversion failed: {}".format(op_name))


def type_it(op_name, arg_name, data, src_type, dst_type):
    """
    cast operator argument data type.
    """
    if isinstance(data, type(None)):
        return data
    if not isinstance(src_type, tuple):
        src_type = int(src_type)
    else:
        src_type = tuple((int(t) for t in src_type))
    dst_type = int(dst_type)
    if not is_instance_in(data, src_type) and not is_instance_of(data, dst_type):
        support_list = get_support_dtype_list(src_type, dst_type)
        raise TypeError(f"For '{op_name}', the type of '{arg_name}' should be one of '[{support_list}]', "
                        f"but got {type(data)}.")
    return do_type_cast(data, dst_type, op_name)
