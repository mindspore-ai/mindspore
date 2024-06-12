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
from mindspore.ops.operations._sequence_ops import TensorToScalar, TensorToTuple
from mindspore.ops_generate.gen_ops_inner_prim import TupleToList, ListToTuple
from mindspore._c_expression import OpDtype

tensor_to_tuple_ = TensorToTuple()
tuple_to_list = TupleToList()
list_to_tuple = ListToTuple()


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


def to_py_number(data, dst_type):
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
    return TensorToScalar()(data)


def do_type_cast(data, dst_type):
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
            return to_py_number(data, dst_type)
    raise TypeError("Type conversion failed.")


def type_it(op_name, arg_name, data, src_type, dst_type):
    """
    cast operator argument data type.
    """
    if not isinstance(src_type, tuple):
        src_type = int(src_type)
    else:
        src_type = tuple((int(t) for t in src_type))
    dst_type = int(dst_type)
    if not is_instance_in(data, src_type) and not is_instance_of(data, dst_type):
        support_list = get_support_dtype_list(src_type, dst_type)
        raise TypeError(f"For '{op_name}', the type of '{arg_name}' should be one of '[{support_list}]', "
                        f"but got {type(data)}.")
    return do_type_cast(data, dst_type)
