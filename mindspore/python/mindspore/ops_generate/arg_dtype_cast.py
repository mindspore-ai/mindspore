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
from mindspore.ops.auto_generate.gen_enum_def import OpDtype

tensor_to_tuple_ = TensorToTuple()


def int_to_float(data):
    return float(data)


def scalar_to_tuple(data, dst_type):
    if dst_type == PY_DT_TUPLE_INT:
        return (int(data),)
    return (data,)


def list_to_tuple(data, dst_type):
    # tuple() currently does not support Any from JIT Fallback.
    res = ()
    for element in data:
        if dst_type == PY_DT_TUPLE_INT:
            res += (int(element),)
        else:
            res += (element,)
    return res


def tensor_to_tuple(data, dst_type):
    if dst_type == PY_DT_TUPLE_INT:
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


def list_to_tensor(data, dst_type):
    return ops.tuple_to_array(list_to_tuple(data, dst_type))


# scalar
PY_DT_INT = OpDtype.PY_DT_INT.value
PY_DT_FLOAT = OpDtype.PY_DT_FLOAT.value
PY_DT_BOOL = OpDtype.PY_DT_BOOL.value
PY_DT_NUMBER = OpDtype.PY_DT_NUMBER.value
# tuple
PY_DT_TUPLE_BOOL = OpDtype.PY_DT_TUPLE_BOOL.value
PY_DT_TUPLE_INT = OpDtype.PY_DT_TUPLE_INT.value
PY_DT_TUPLE_FLOAT = OpDtype.PY_DT_TUPLE_FLOAT.value
PY_DT_TUPLE_NUMBER = OpDtype.PY_DT_TUPLE_NUMBER.value
PY_DT_TUPLE_TENSOR = OpDtype.PY_DT_TUPLE_TENSOR.value
PY_DT_TUPLE_STR = OpDtype.PY_DT_TUPLE_STR.value
PY_DT_TUPLE_ANY = OpDtype.PY_DT_TUPLE_ANY.value
# list
PY_DT_LIST_BOOL = OpDtype.PY_DT_LIST_BOOL.value
PY_DT_LIST_INT = OpDtype.PY_DT_LIST_INT.value
PY_DT_LIST_FLOAT = OpDtype.PY_DT_LIST_FLOAT.value
PY_DT_LIST_NUMBER = OpDtype.PY_DT_LIST_NUMBER.value
PY_DT_LIST_TENSOR = OpDtype.PY_DT_LIST_TENSOR.value
PY_DT_LIST_STR = OpDtype.PY_DT_LIST_STR.value
PY_DT_LIST_ANY = OpDtype.PY_DT_LIST_ANY.value
# tensor
PY_DT_TENSOR = OpDtype.PY_DT_TENSOR.value


def is_tuple(type_id):
    """
    Check type id is tuple.
    """
    return type_id in (PY_DT_TUPLE_BOOL, PY_DT_TUPLE_INT, PY_DT_TUPLE_FLOAT, PY_DT_TUPLE_NUMBER,
                       PY_DT_TUPLE_TENSOR, PY_DT_TUPLE_STR, PY_DT_TUPLE_ANY)


def is_list(type_id):
    """
    Check type id is list.
    """
    return type_id in (PY_DT_LIST_BOOL, PY_DT_LIST_INT, PY_DT_LIST_FLOAT, PY_DT_LIST_NUMBER, PY_DT_LIST_TENSOR,
                       PY_DT_LIST_STR, PY_DT_LIST_ANY)


def is_number(type_id):
    """
    Check type id is number.
    """
    return type_id in (PY_DT_INT, PY_DT_FLOAT, PY_DT_BOOL, PY_DT_NUMBER)


def is_instance_of(data, type_id):
    """
    Instead isinstance(obj, type).
    """
    if type_id == PY_DT_INT:
        return isinstance(data, int)
    if type_id == PY_DT_FLOAT:
        return isinstance(data, float)
    if type_id == PY_DT_BOOL:
        return isinstance(data, bool)
    if is_number(type_id):
        return isinstance(data, (int, float, bool))
    if is_tuple(type_id):
        return isinstance(data, tuple)
    if is_list(type_id):
        return isinstance(data, list)
    if type_id == PY_DT_TENSOR:
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
        raise TypeError(f"For type_it, the {data} should be OpDtype::{src_type}, but got {type(data)}.")
    if dst_type == PY_DT_FLOAT:
        if isinstance(data, int):
            return int_to_float(data)
    elif is_tuple(dst_type):
        if isinstance(data, (int, float, bool)):
            return scalar_to_tuple(data, dst_type)
        if isinstance(data, list):
            return list_to_tuple(data, dst_type)
        if isinstance(data, Tensor):
            return tensor_to_tuple(data, dst_type)
    elif dst_type == PY_DT_TENSOR:
        if isinstance(data, (int, float, bool)):
            return scalar_to_tensor(data)
        if isinstance(data, tuple):
            return tuple_to_tensor(data)
        if isinstance(data, list):
            return list_to_tensor(data, dst_type)
    elif is_number(dst_type):
        if isinstance(data, Tensor):
            ret = TensorToScalar()(data)
            return ret

    raise TypeError(f"Failed to convert dtype of {data} from OpDtype::{src_type} to OpDtype::{dst_type}.")
