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

from mindspore import ops
from mindspore.common.tensor import Tensor
from mindspore.ops.operations._sequence_ops import TensorToScalar
from mindspore.ops.auto_generate.gen_enum_def import OpDtype


def int_to_float(data):
    return float(data)


def scalar_to_tuple(data):
    return (data,)


def list_to_tuple(data):
    return tuple(data)


def tensor_to_tuple(data):
    return ops.tuple_to_array(data)


def scalar_to_tensor(data):
    return ops.scalar_to_tensor(data)


def tuple_to_tensor(data):
    return ops.tuple_to_array(data)


def list_to_tensor(data):
    return ops.tuple_to_array(tuple(data))


# scalar
DT_INT = OpDtype.DT_INT.value
DT_FLOAT = OpDtype.DT_FLOAT.value
DT_BOOL = OpDtype.DT_BOOL.value
DT_NUMBER = OpDtype.DT_NUMBER.value
# tuple
DT_TUPLE_BOOL = OpDtype.DT_TUPLE_ANY.value
DT_TUPLE_INT = OpDtype.DT_TUPLE_INT.value
DT_TUPLE_FLOAT = OpDtype.DT_TUPLE_FLOAT.value
DT_TUPLE_NUMBER = OpDtype.DT_TUPLE_NUMBER.value
DT_TUPLE_TENSOR = OpDtype.DT_TUPLE_TENSOR.value
DT_TUPLE_STR = OpDtype.DT_TUPLE_STR.value
DT_TUPLE_ANY = OpDtype.DT_TUPLE_ANY.value
# list
DT_LIST_BOOL = OpDtype.DT_LIST_BOOL.value
DT_LIST_INT = OpDtype.DT_LIST_INT.value
DT_LIST_FLOAT = OpDtype.DT_LIST_FLOAT.value
DT_LIST_NUMBER = OpDtype.DT_LIST_NUMBER.value
DT_LIST_TENSOR = OpDtype.DT_LIST_TENSOR.value
DT_LIST_STR = OpDtype.DT_LIST_STR.value
DT_LIST_ANY = OpDtype.DT_LIST_ANY.value
# tensor
DT_TENSOR = OpDtype.DT_TENSOR.value


def is_tuple(type_id):
    """
    Check type id is tuple.
    """
    return type_id in (DT_TUPLE_BOOL, DT_TUPLE_INT, DT_TUPLE_FLOAT, type_id == DT_TUPLE_NUMBER, DT_TUPLE_TENSOR,
                       DT_TUPLE_STR, DT_TUPLE_ANY)


def is_list(type_id):
    """
    Check type id is list.
    """
    return type_id in (DT_LIST_BOOL, DT_LIST_INT, DT_LIST_FLOAT, DT_LIST_NUMBER, DT_LIST_TENSOR,
                       DT_LIST_STR, DT_LIST_ANY)


def is_numer(type_id):
    """
    Check type id is number.
    """
    return type_id in (DT_INT, DT_FLOAT, DT_BOOL, DT_NUMBER)


def is_instance_of(data, type_id):
    """
    Instead isinstance(obj, type).
    """
    if type_id == DT_INT:
        return isinstance(data, int)
    if type_id == DT_FLOAT:
        return isinstance(data, float)
    if type_id == DT_BOOL:
        return isinstance(data, bool)
    if is_numer(type_id):
        return isinstance(data, (int, float, bool))
    if is_tuple(type_id):
        return isinstance(data, tuple)
    if is_list(type_id):
        return isinstance(data, list)
    if type_id == DT_TENSOR:
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


def type_it(data, src_type, dst_type):
    """
    cast operator argument data type.
    """
    if is_instance_of(data, dst_type):
        return data
    if not is_instance_in(data, src_type):
        raise TypeError(f"For type_it, the {data} should be {src_type}, but got {type(data)}")
    if dst_type == DT_FLOAT:
        if isinstance(data, int):
            return int_to_float(data)
    elif is_tuple(dst_type):
        if isinstance(data, (int, float, bool)):
            return scalar_to_tuple(data)
        if isinstance(data, list):
            return list_to_tuple(data)
        if isinstance(data, Tensor):
            return tensor_to_tuple(data)
    elif dst_type == DT_TENSOR:
        if isinstance(data, (int, float, bool)):
            return scalar_to_tensor(data)
        if isinstance(data, tuple):
            return tuple_to_tensor(data)
        if isinstance(data, list):
            return list_to_tensor(data)
    # is dst_type is number:
    elif is_numer(dst_type):
        if isinstance(data, Tensor):
            ret = TensorToScalar()(data)
            return ret

    raise TypeError("Unsupported type cast.")
