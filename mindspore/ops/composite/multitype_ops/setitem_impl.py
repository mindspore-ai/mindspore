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
from . import _utils as multi_utils

setitem = base.MultitypeFuncGraph('setitem')


@setitem.register("List", "Number", "String")
def _list_setitem_with_string(data, number_index, value):
    """
    Assigns value to list.

    Inputs:
        data (list): Data of type lis.
        number_index (Number): Index of data.

    Outputs:
        list, type is same as the element type of data.
    """
    return F.list_setitem(data, number_index, value)


@setitem.register("List", "Number", "Number")
def _list_setitem_with_number(data, number_index, value):
    """
    Assigns value to list.

    Inputs:
        data (list): Data of type lis.
        number_index (Number): Index of data.
        value (Number): Value given.

    Outputs:
        list, type is same as the element type of data.
    """
    return F.list_setitem(data, number_index, value)


@setitem.register("List", "Number", "Tensor")
def _list_setitem_with_Tensor(data, number_index, value):
    """
    Assigns value to list.

    Inputs:
        data (list): Data of type lis.
        number_index (Number): Index of data.
        value (Tensor): Value given.

    Outputs:
        list, type is same as the element type of data.
    """
    return F.list_setitem(data, number_index, value)


@setitem.register("List", "Number", "List")
def _list_setitem_with_List(data, number_index, value):
    """
    Assigns value to list.

    Inputs:
        data (list): Data of type lis.
        number_index (Number): Index of data.
        value (list): Value given.

    Outputs:
        list, type is same as the element type of data.
    """
    return F.list_setitem(data, number_index, value)


@setitem.register("Dictionary", "String", "Tensor")
def _dict_setitem_with_tensor(data, key, value):
    """
    Assigns value to dictionary.

    Inputs:
        data (dict): Data of type dict.
        key (str): Key of the data.
        value (Tensor): Value given.

    Outputs:
        dict, type is as same as the element type of data.
    """
    return F.dict_setitem(data, key, value)


@setitem.register("Dictionary", "String", "Number")
def _dict_setitem_with_number(data, key, value):
    """
    Assigns value to dictionary.

    Inputs:
        data (dict): Data of type dict.
        key (str): Key of the data.
        value (Number): Value given.

    Outputs:
        dict, type is as same as the element type of data.
    """
    return F.dict_setitem(data, key, value)


@setitem.register("Tensor", "Tensor", "Tensor")
def _tensor_setitem_by_tensor_with_tensor(data, index, value_tensor):
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
    tensor_dtype = multi_utils.get_index_tensor_dtype(index_dtype)
    if tensor_dtype == multi_utils.INT_:
        return _tensor_setitem_by_int_tensor_with_tensor(data, index, value_tensor)
    return _tensor_setitem_by_bool_tensor_with_tensor(data, index, value_tensor)


@setitem.register("Tensor", "Tensor", "Number")
def _tensor_setitem_by_tensor_with_number(data, index, value):
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
        value (Number): Assignment value.

    Outputs:
        Tensor, element type and shape is same as data.
    """
    index_dtype = F.dtype(index)
    tensor_dtype = multi_utils.get_index_tensor_dtype(index_dtype)
    if tensor_dtype == multi_utils.BOOL_:
        return _tensor_setitem_by_bool_tensor_with_scalar(data, index, value)
    return _tensor_setitem_by_int_tensor_with_scalar(data, index, value)


@setitem.register("Tensor", "Tuple", "Number")
def _tensor_setitem_by_tuple_with_number(data, tuple_index, value):
    """
    Tensor assignment.

    Note:
        Syntax support: A[B, C, D] = u.
        Restraint condition: 1) A is a Tensor, and B, C, D are index.
                             2) u is a scalar.

    Inputs:
        data (Tensor): Assigned tensor.
        index (Tuple): An index tuple.
        value (Number): Assignment value.

    Outputs:
        Tensor, element type and shape is same as data.
    """
    index_types = multi_utils.hyper_map(F.typeof, tuple_index)
    index_elements_type = multi_utils.tuple_index_elements_type(index_types, multi_utils.TENSOR_SETITEM)
    result = None
    if index_elements_type == multi_utils.NO_TENSOR:
        result = _tensor_assgin_number(data, tuple_index, value)
    if index_elements_type == multi_utils.ALL_TENSOR:
        indices = multi_utils.generate_indeices_from_tuple_of_tensor(data, tuple_index, multi_utils.TENSOR_SETITEM)
        updates = multi_utils.generate_updates_from_scalar(data, indices, value,
                                                           multi_utils.SET_ITEM_BY_TUPLE_OF_TENSOR)
        result = F.scatter_nd_update(data, indices, updates)
    return result


@setitem.register("Tensor", "Tuple", "Tensor")
def _tensor_setitem_by_tuple_with_tensor(data, tuple_index, value):
    """
    Tensor assignment.

    Note:
        Syntax support: A[B, C, D] = U.
        Restraint condition: 1) A is a Tensor, and B, C, D are index Tensors.
                             2) U is a Tensor.

    Inputs:
        data (Tensor): Assigned tensor.
        index (Tuple): An index tuple.
        value (Tensor): Assignment tensor, should has the same data type as 'data'.

    Outputs:
        Tensor, element type and shape is same as data.
    """
    index_types = multi_utils.hyper_map(F.typeof, tuple_index)
    index_elements_type = multi_utils.tuple_index_elements_type(index_types, multi_utils.TENSOR_SETITEM)
    result = None
    if index_elements_type == multi_utils.NO_TENSOR:
        result = _tensor_assgin_tensor(data, tuple_index, value)
    if index_elements_type == multi_utils.ALL_TENSOR:
        indices = multi_utils.generate_indeices_from_tuple_of_tensor(data, tuple_index, multi_utils.TENSOR_SETITEM)
        updates = multi_utils.generate_updates_from_tensor(data, indices, value,
                                                           multi_utils.SET_ITEM_BY_TUPLE_OF_TENSOR)
        result = F.scatter_nd_update(data, indices, updates)
    return result


@setitem.register("Tensor", "Tuple", "Tuple")
def _tensor_setitem_by_tuple_with_tuple(data, tuple_index, value):
    """
    Tensor assignment.

    Note:
        Syntax support: A[B, C, D] = U.
        Restraint condition: 1) A is a Tensor, and B, C, D are index Tensors.
                             2) A B and C could be broadcast.
                             3) U is a Tensor.

    Inputs:
        data (Tensor): Assigned tensor.
        index (Tuple): A tuple of tensor, these tensor could be broadcast.
        value (Tensor): Assignment tensor, should has the same data type as 'data'.

    Outputs:
        Tensor, element type and shape is same as data.
    """
    index_types = multi_utils.hyper_map(F.typeof, tuple_index)
    index_elements_type = multi_utils.tuple_index_elements_type(index_types, multi_utils.TENSOR_SETITEM)
    result = None
    if index_elements_type == multi_utils.ALL_TENSOR:
        indices = multi_utils.generate_indeices_from_tuple_of_tensor(data, tuple_index, multi_utils.TENSOR_SETITEM)
        updates = multi_utils.generate_updates_from_tuple(data, indices, value,
                                                          multi_utils.SET_ITEM_BY_TUPLE_OF_TENSOR)
        result = F.scatter_nd_update(data, indices, updates)
    return result


@setitem.register("Tensor", "Tensor", "Tuple")
def _tensor_setitem_by_tensor_v2(data, index, value):
    """
    Tensor assignment.

    Inputs:
        data (Tensor): Assigned tensor.
        index (Tensor): Tensor of bool type.
        value (Tuple): Assignment value.

    Outputs:
        Tensor, element type and shape is same as data.
    """
    index_dtype = F.dtype(index)
    check_dtype = multi_utils.check_tensor_dtype_valid(index_dtype, (mstype.int32, mstype.int64))
    result = None
    if check_dtype:
        result = _tensor_setitem_by_tensor_with_tuple(data, index, value)
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


def _tensor_assgin_number(data, input_slice, value):
    """Givens a scalar assign to tensor by slice"""
    check_result = multi_utils.check_tensor_setitem_index(input_slice)
    result = None
    if check_result:
        data_shape = F.shape(data)
        indices = multi_utils.slice2indices(input_slice, data_shape)
        is_tuple_int = multi_utils.tuple_element_is_int(input_slice)
        if is_tuple_int:
            indices = multi_utils.integer_to_indices(input_slice, data_shape)
        result = _tensor_indices_number(data, data_shape, input_slice, indices, value)
    return result


@setitem.register("Tensor", "Number", "Number")
def _tensor_setitem_with_int_v1(data, index, value):
    """Syntax: A[1] = 3"""
    data_shape = F.shape(data)
    indices = multi_utils.integer_to_indices(index, data_shape)
    return _tensor_indices_number(data, data_shape, index, indices, value)


@setitem.register("Tensor", "Number", "Tensor")
def _tensor_setitem_with_int_v2(data, index, value):
    """Syntax: A[1] = Tensor"""
    data_shape = F.shape(data)
    indices = multi_utils.integer_to_indices(index, data_shape)
    return _tensor_indices_tensor(data, data_shape, index, indices, value)


@setitem.register("Tensor", "Ellipsis", "Number")
def _tensor_setitem_with_ellipsis_v1(data, index, value):
    """Syntax: A[...] = number."""
    data_shape = F.shape(data)
    data_dtype = F.dtype(data)
    return F.fill(data_dtype, data_shape, value)


@setitem.register("Tensor", "Ellipsis", "Tensor")
def _tensor_setitem_with_ellipsis_v2(data, index, value):
    """Syntax: A[...] = Tensor."""
    result = None
    data_shape = F.shape(data)
    data_dtype = F.dtype(data)
    data_size = F.size(data)
    value_shape = F.shape(value)
    value_size = F.size(value)
    check_result = multi_utils.check_ellipsis_shape_size(data_shape, value_shape, data_size, value_size)
    if check_result:
        if data_size == value_size:
            result = F.reshape(value, data_shape)
            result = F.cast(result, data_dtype)
        elif value_size == 1:
            param1 = F.fill(data_dtype, data_shape, 1)
            param2 = F.cast(value, data_dtype)
            result = F.tensor_mul(param1, param2)
    return result


def _tensor_assgin_tensor(data, input_slice, value):
    """Assigns a tensor value to the tensor by slice."""
    result = None
    check_result = multi_utils.check_tensor_setitem_index(input_slice)
    if check_result:
        data_shape = F.shape(data)
        indices = multi_utils.slice2indices(input_slice, data_shape)
        is_tuple_int = multi_utils.tuple_element_is_int(input_slice)
        if is_tuple_int:
            indices = multi_utils.integer_to_indices(input_slice, data_shape)
        result = _tensor_indices_tensor(data, data_shape, input_slice, indices, value)
    return result


def _tensor_indices_tensor(data, data_shape, index, indices, value):
    """Assigns a tensor value to the tensor."""
    data_size = F.size(data)
    data_dtype = F.dtype(data)
    indices_size = F.size(indices)
    indices_size = multi_utils.check_indices(indices_size, index)
    update = F.fill(mstype.int32, (indices_size,), 1)
    condition_1d = F.scatter_nd(indices, update, (data_size,))
    condition = F.reshape(condition_1d, data_shape)
    condition = F.cast(condition, mstype.bool_)
    value_fill = None
    value_size = F.size(value)

    value_size = multi_utils.check_indices_value_size(indices_size, value_size)
    if value_size == 1:
        value_fill = F.fill(data_dtype, (indices_size,), 1)
        value = F.cast(value, data_dtype)
        value_fill = F.tensor_mul(value_fill, value)
    elif value_size > 1:
        value_fill = F.reshape(value, (indices_size,))
    value_1d = F.scatter_nd(indices, value_fill, (data_size,))
    u = F.reshape(value_1d, data_shape)
    return F.select(condition, u, data)


def _tensor_indices_number(data, data_shape, index, indices, value):
    """Assigns a scalar value to the tensor."""
    data_size = F.size(data)
    data_dtype = F.dtype(data)
    indices_size = F.size(indices)
    indices_size = multi_utils.check_indices(indices_size, index)
    update = F.fill(mstype.int32, (indices_size,), 1)
    condition_1d = F.scatter_nd(indices, update, (data_size,))
    condition = F.reshape(condition_1d, data_shape)
    condition = F.cast(condition, mstype.bool_)
    value_fill = F.fill(data_dtype, (indices_size,), value)
    value_1d = F.scatter_nd(indices, value_fill, (data_size,))
    u = F.reshape(value_1d, data_shape)
    return F.select(condition, u, data)


def _tensor_setitem_by_tensor_with_tuple(data, index, value):
    """Set a tensor item by a tensor with a tuple."""
    updates = multi_utils.generate_updates_from_tuple(data, index, value,
                                                      multi_utils.SET_ITEM_BY_ONE_TENSOR)
    result = F.scatter_update(data, index, updates)
    return result


def _tensor_setitem_by_int_tensor_with_scalar(data, index, value):
    """Set a tensor item by a int tensor with a scalar."""
    updates = multi_utils.generate_updates_from_scalar(data, index, value,
                                                       multi_utils.SET_ITEM_BY_ONE_TENSOR)
    return F.scatter_update(data, index, updates)


def _tensor_setitem_by_bool_tensor_with_scalar(data, index, value):
    """Set a tensor item by a bool tensor with a scalar."""
    index_shape = F.shape(index)
    shape = F.shape(data)
    shape = multi_utils.check_equal(
        shape, index_shape, "The tensor(shape={}) and tensor index(shape={}) should be the same shape.")
    dtype = F.dtype(data)
    u = F.fill(dtype, shape, value)
    return F.select(index, u, data)


def _tensor_setitem_by_int_tensor_with_tensor(data, index, value):
    """Set a tensor item by a int tensor with a tensor."""
    updates = multi_utils.generate_updates_from_tensor(data, index, value,
                                                       multi_utils.SET_ITEM_BY_ONE_TENSOR)
    return F.scatter_update(data, index, updates)


def _tensor_setitem_by_bool_tensor_with_tensor(data, index, value):
    """Set a tensor item by a bool tensor with a tensor."""
    index_shape = F.shape(index)
    data_shape = F.shape(data)
    data_shape = multi_utils.check_equal(data_shape, index_shape,
                                         "The tensor(shape={}) and tensor index(shape={}) should be the same shape.")
    size = F.size(value)
    size = multi_utils.check_equal(1, size,
                                   "When assign value is a tensor, its size should be {}, but current size is {}.")
    dtype = F.dtype(data)
    u_cast = F.cast(value, dtype)
    one_data = F.ones_like(data)
    u = F.tensor_mul(one_data, u_cast)
    result = F.select(index, u, data)
    return result
