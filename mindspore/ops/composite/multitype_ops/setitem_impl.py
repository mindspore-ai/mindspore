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
    result = None
    index_dtype = F.dtype(index)
    index_shape = F.shape(index)
    check_result = mult_util.check_tensor_setitem_index(mstype.tensor, index_dtype)
    if check_result:
        data_shape = F.shape(data)
        data_shape = mult_util.check_equal(data_shape, index_shape,
                                           "The tensor(shape={}) and tensor index(shape={}) should be the same shape.")
        size = F.size(value_tensor)
        size = mult_util.check_equal(1, size,
                                     "When assign value is a tensor, its size should be {}, but current size is {}.")
        dtype = F.dtype(data)
        u_cast = F.cast(value_tensor, dtype)
        one_data = F.ones_like(data)
        u = F.tensor_mul(one_data, u_cast)
        result = F.select(index, u, data)
    return result


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
    result = None
    index_dtype = F.dtype(index)
    index_shape = F.shape(index)
    check_result = mult_util.check_tensor_setitem_index(mstype.tensor, index_dtype)
    if check_result:
        shape = F.shape(data)
        shape = mult_util.check_equal(
            shape, index_shape, "The tensor(shape={}) and tensor index(shape={}) should be the same shape.")
        dtype = F.dtype(data)
        u = F.fill(dtype, shape, value)
        result = F.select(index, u, data)
    return result


@setitem.register("Tensor", "Slice", "Tensor")
def _tensor_setitem_with_slice_v3(data, input_slice, value):
    """
    Tensor assignment.

    Note:
        Syntax support: A[Slice] = U
        Restraint condition: A is a Tensor
                             Slice like "1:3"
                             U is a Tensor(size=1) or Tensor(size>1)

    Inputs:
        data (Tensor): Assigned tensor.
        input_slice (Slice): Slice expression.
        value (Number): Assignment value.

    Outputs:
        Tensor, element type and shape is same as data.
    """
    return _tensor_assgin_tensor(data, input_slice, value)


@setitem.register("Tensor", "Tuple", "Tensor")
def _tensor_setitem_with_slice_v4(data, input_slice, value):
    """
    Tensor assignment.

    Note:
        Syntax support: A[Slice] = U
        Restraint condition: A is a Tensor
                             Slice like "1:3, ::, :4:-1"
                             U is a Tensor(size=1) or Tensor(size>1)

    Inputs:
        data (Tensor): Assigned tensor.
        input_slice (Tuple(Slice)): Slice expression.
        value (Number): Assignment value.

    Outputs:
        Tensor, element type and shape is same as data.
    """
    return _tensor_assgin_tensor(data, input_slice, value)


def _tensor_assgin_tensor(data, input_slice, value):
    """Given a tensor value assign to tensor by slice"""
    # 1. condition
    result = None
    check_result = mult_util.check_tensor_setitem_index(input_slice)
    if check_result:
        data_shape = F.shape(data)
        data_size = F.size(data)
        data_dtype = F.dtype(data)
        indices = mult_util.slice2indices(input_slice, data_shape)
        indices_size = F.size(indices)
        indices_size = mult_util.check_indices(indices_size, input_slice)
        update = F.fill(data_dtype, (indices_size,), 1)
        condition_1d = F.scatter_nd(indices, update, (data_size,))
        condition_1d = F.cast(condition_1d, mstype.bool_)
        condition = F.reshape(condition_1d, data_shape)
        # 2. u
        value_fill = None
        value_size = F.size(value)

        value_size = mult_util.check_indices_value_size(indices_size, value_size)
        if value_size == 1:
            value_fill = F.fill(data_dtype, (indices_size,), 1)
            value = F.cast(value, data_dtype)
            value_fill = F.tensor_mul(value_fill, value)
        elif value_size > 1:
            value_fill = F.reshape(value, (indices_size,))
        value_1d = F.scatter_nd(indices, value_fill, (data_size,))
        u = F.reshape(value_1d, data_shape)
        # A[slice]= u -> A[B]=U -> select(B, U, A)
        result = F.select(condition, u, data)
    return result


@setitem.register("Tensor", "Slice", "Number")
def _tensor_setitem_with_slice_v1(data, input_slice, value):
    """
    Tensor assignment.

    Note:
        Syntax support: A[Slice] = u
        Restraint condition: A is a Tensor.
                             Slice like "1:3"
                             u is a scalar

    Inputs:
        data (Tensor): Assigned tensor.
        input_slice (Slice): slice expression.
        value (Number): Assignment value.

    Outputs:
        Tensor, element type and shape is same as data.
    """
    return _tensor_assgin_number(data, input_slice, value)


@setitem.register("Tensor", "Tuple", "Number")
def _tensor_setitem_with_slice_v2(data, input_slice, value):
    """
    Tensor assignment.

    Note:
        Syntax support: A[Slice] = u
        Restraint condition: A is a Tensor.
                             Slice like "1:3, ::, :4:-1"
                             u is a scalar

    Inputs:
        data (Tensor): Assigned tensor.
        input_slice (Tuple(Slice)): slice expression.
        value (Number): Assignment value.

    Outputs:
        Tensor, element type and shape is same as data.
    """
    return _tensor_assgin_number(data, input_slice, value)


def _tensor_assgin_number(data, input_slice, value):
    """Given a scalar assign to tensor by slice"""
    # 1. condition
    check_result = mult_util.check_tensor_setitem_index(input_slice)
    result = None
    if check_result:
        data_shape = F.shape(data)
        data_size = F.size(data)
        data_dtype = F.dtype(data)
        indices = mult_util.slice2indices(input_slice, data_shape)
        indices_size = F.size(indices)
        indices_size = mult_util.check_indices(indices_size, input_slice)
        update = F.fill(data_dtype, (indices_size,), 1)
        condition_1d = F.scatter_nd(indices, update, (data_size,))
        condition_1d = F.cast(condition_1d, mstype.bool_)
        condition = F.reshape(condition_1d, data_shape)
        # 2. u
        value_fill = F.fill(data_dtype, (indices_size,), value)
        value_1d = F.scatter_nd(indices, value_fill, (data_size,))
        u = F.reshape(value_1d, data_shape)
        # A[slice]= u -> A[B]=U -> select(B, U, A)
        result = F.select(condition, u, data)
    return result
