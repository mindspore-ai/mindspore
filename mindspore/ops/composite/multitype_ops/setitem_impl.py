# Copyright 2020 Huawei Technologies Co., Ltd
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

"""Implementation for setitem."""

from ...composite import base
from ....common import dtype as mstype
from ... import functional as F
from . import _multitype_ops_util as mult_util

setitem = base.MultitypeFuncGraph('setitem')

@setitem.register("List", "Number", "String")
def _list_setitem_with_string(data, number_index, value):
    """
    Assign value to list.

    Inputs:
        data (list): Data of type lis.
        number_index (Number): Index of data.
        value (String): Value given.

    Outputs:
        List, type is same as the element type of data.
    """
    return F.list_setitem(data, number_index, value)


@setitem.register("List", "Number", "Number")
def _list_setitem_with_number(data, number_index, value):
    """
    Assign value to list.

    Inputs:
        data (list): Data of type lis.
        number_index (Number): Index of data.
        value (Number): Value given.

    Outputs:
        List, type is same as the element type of data.
    """
    return F.list_setitem(data, number_index, value)


@setitem.register("List", "Number", "Tensor")
def _list_setitem_with_Tensor(data, number_index, value):
    """
    Assign value to list.

    Inputs:
        data (list): Data of type lis.
        number_index (Number): Index of data.
        value (Tensor): Value given.

    Outputs:
        List, type is same as the element type of data.
    """
    return F.list_setitem(data, number_index, value)


@setitem.register("List", "Number", "List")
def _list_setitem_with_List(data, number_index, value):
    """
    Assign value to list.

    Inputs:
        data (list): Data of type lis.
        number_index (Number): Index of data.
        value (List): Value given.

    Outputs:
        List, type is same as the element type of data.
    """
    return F.list_setitem(data, number_index, value)


@setitem.register("Dictionary", "String", "Tensor")
def _dict_setitem_with_tensor(data, key, value):
    """
    Assign value to dictionary.

    Inputs:
        data (Dictionary): Data of type dict.
        key (str): Key of the data.
        value (Tensor): Value given.

    Outputs:
        Dict, type is as same as the element type of data.
    """
    return F.dict_setitem(data, key, value)


@setitem.register("Dictionary", "String", "Number")
def _dict_setitem_with_number(data, key, value):
    """
    Assign value to dictionary.

    Inputs:
        data (Dictionary): Data of type dict.
        key (str): Key of the data.
        value (Number): Value given.

    Outputs:
        Dict, type is as same as the element type of data.
    """
    return F.dict_setitem(data, key, value)


@setitem.register("Tensor", "Tensor", "Tensor")
def _tensor_setitem_by_tensor_v1(data, index, value_tensor):
    """
    Tensor assignment.

    Note:
        Syntax support: A[B] = U and A[A>n] = U.
        Restraint condition: 1) A, U is a Tensor, and B is a bool Tensor.
                             2) A.shape == B.shape
                             3) U.size == 1
                             4) n is a number

    Inputs:
        data (Tensor): Assigned tensor.
        index (Tensor): Tensor of bool type.
        value_tensor (Tensor): Tensor with size 1.

    Outputs:
        Tensor, element type and shape is same as data.
    """
    index_dtype = F.dtype(index)
    index_shape = F.shape(index)
    is_bool = mult_util.is_same_type(index_dtype, mstype.bool_)
    if not is_bool:
        return mult_util.error_msg(
            "The tensor index should be a bool type tensor. {} type tensor is not supported yet.", (index_dtype,))
    data_shape = F.shape(data)
    if index_shape != data_shape:
        return mult_util.error_msg(
            "The tensor(shape={}) and tensor index(shape={}) should be the same shape.", (data_shape, index_shape))
    size = F.size(value_tensor)
    if size != 1:
        return mult_util.error_msg(
            "When assign value is a tensor, its size should be 1, but current size is {}.", (size,))
    dtype = F.dtype(data)
    u_cast = F.cast(value_tensor, dtype)
    one_data = F.ones_like(data)
    u = F.tensor_mul(one_data, u_cast)
    return F.select(index, u, data)


@setitem.register("Tensor", "Tensor", "Number")
def _tensor_setitem_by_tensor_v2(data, index, value):
    """
    Tensor assignment.

    Note:
        Syntax support: A[B] = u and A[A>n] = u.
        Restraint condition: 1) A is a Tensor, and B is a bool Tensor.
                             2) A.shape == B.shape
                             3) u is a scalar
                             4) n is a number

    Inputs:
        data (Tensor): Assigned tensor.
        index (Tensor): Tensor of bool type.
        value_tensor (Number): Assignment value.

    Outputs:
        Tensor, element type and shape is same as data.
    """
    index_dtype = F.dtype(index)
    index_shape = F.shape(index)
    is_bool = mult_util.is_same_type(index_dtype, mstype.bool_)
    if not is_bool:
        return mult_util.error_msg(
            "The tensor index should be a bool type tensor. {} type tensor is not supported yet.", (index_dtype,))
    shape = F.shape(data)
    if index_shape != shape:
        return mult_util.error_msg(
            "The tensor(shape={}) and tensor index(shape={}) should be the same shape.", (shape, index_shape))
    dtype = F.dtype(data)
    u = F.fill(dtype, shape, value)
    return F.select(index, u, data)
