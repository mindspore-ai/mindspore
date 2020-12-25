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

"""Implementation for internal polymorphism `setitem` operations."""

from . import _compile_utils as compile_utils
from ... import functional as F
from ...composite import base

setitem = base.MultitypeFuncGraph('setitem')


@setitem.register("List", "Number", "String")
def _list_setitem_with_string(data, number_index, value):
    """
    Assigns value to list.

    Inputs:
        data (list): Data of type list.
        number_index (Number): Index of data.

    Outputs:
        list, type is the same as the element type of data.
    """
    return F.list_setitem(data, number_index, value)


@setitem.register("List", "Number", "Number")
def _list_setitem_with_number(data, number_index, value):
    """
    Assigns value to list.

    Inputs:
        data (list): Data of type list.
        number_index (Number): Index of data.
        value (Number): Value given.

    Outputs:
        list, type is the same as the element type of data.
    """
    return F.list_setitem(data, number_index, value)


@setitem.register("List", "Number", "Tensor")
def _list_setitem_with_Tensor(data, number_index, value):
    """
    Assigns value to list.

    Inputs:
        data (list): Data of type list.
        number_index (Number): Index of data.
        value (Tensor): Value given.

    Outputs:
        list, type is the same as the element type of data.
    """
    return F.list_setitem(data, number_index, value)


@setitem.register("List", "Number", "List")
def _list_setitem_with_List(data, number_index, value):
    """
    Assigns value to list.

    Inputs:
        data (list): Data of type list.
        number_index (Number): Index of data.
        value (list): Value given.

    Outputs:
        list, type is the same as the element type of data.
    """
    return F.list_setitem(data, number_index, value)


@setitem.register("List", "Number", "Tuple")
def _list_setitem_with_Tuple(data, number_index, value):
    """
    Assigns value to list.

    Inputs:
        data (list): Data of type list.
        number_index (Number): Index of data.
        value (list): Value given.

    Outputs:
        list, type is the same as the element type of data.
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


@setitem.register("Dictionary", "String", "Tuple")
def _dict_setitem_with_tuple(data, key, value):
    """
    Assigns value to dictionary.

    Inputs:
        data (dict): Data of type dict.
        key (str): Key of the data.
        value (Tuple): Value given.

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
    return compile_utils.tensor_setitem_by_tensor_with_tensor(data, index, value_tensor)


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
    return compile_utils.tensor_setitem_by_tensor_with_number(data, index, value)


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
    return compile_utils.tensor_setitem_by_tuple_with_number(data, tuple_index, value)


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
    return compile_utils.tensor_setitem_by_tuple_with_tensor(data, tuple_index, value)


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
    return compile_utils.tensor_setitem_by_tuple_with_tuple(data, tuple_index, value)


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
    return compile_utils.tensor_setitem_by_tensor_with_tuple(data, index, value)


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
    return compile_utils.tensor_setitem_by_slice_with_tensor(data, input_slice, value)


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
    return compile_utils.tensor_setitem_by_slice_with_number(data, input_slice, value)


@setitem.register("Tensor", "Number", "Number")
def _tensor_setitem_with_int_v1(data, index, value):
    return compile_utils.tensor_setitem_by_number_with_number(data, index, value)


@setitem.register("Tensor", "Number", "Tensor")
def _tensor_setitem_with_int_v2(data, index, value):
    return compile_utils.tensor_setitem_by_number_with_tensor(data, index, value)


@setitem.register("Tensor", "Ellipsis", "Number")
def _tensor_setitem_with_ellipsis_v1(data, index, value):
    return compile_utils.tensor_setitem_by_ellipsis_with_number(data, index, value)


@setitem.register("Tensor", "Ellipsis", "Tensor")
def _tensor_setitem_with_ellipsis_v2(data, index, value):
    return compile_utils.tensor_setitem_by_ellipsis_with_tensor(data, index, value)
