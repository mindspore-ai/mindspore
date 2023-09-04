# Copyright 2020-2022 Huawei Technologies Co., Ltd
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

from __future__ import absolute_import
from mindspore.ops.composite.multitype_ops import _compile_utils as compile_utils
from mindspore.ops import functional as F
from mindspore.ops.operations._inner_ops import SliceGetItem
from mindspore.ops.operations import _map_tensor_ops
from mindspore.ops.composite import base
from mindspore.common import Tensor
from mindspore.ops.composite.base import _dict_setitem
from ...operations._sequence_ops import SequenceSliceSetItem

DOC_URL = "https://mindspore.cn/search/en?inputValue=Index%20value%20assignment"

setitem = base.MultitypeFuncGraph('setitem')
setitem.set_doc_url(DOC_URL)
slice_get_item = SliceGetItem()
sequence_slice_setitem = SequenceSliceSetItem()


class _ListSliceSetItem(base.ListSliceSetItem_):
    """
    List slice assign.

    Inputs:
        data (List): A List to be sliced.
        s (slice): The index to slice list data.
        value : The value to be assign

    Outputs:
        List, consists of some elements of data.
    """

    def __init__(self, name):
        """Initialize _TupleSlice."""
        base.ListSliceSetItem_.__init__(self, name)

    def __call__(self, *args):
        pass


_list_slice_set_item = _ListSliceSetItem('list_slice_set_item')
"""_list_slice_set_item is a MetaFuncGraph object which assign a list will slice."""


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
def _list_setitem_with_tensor(data, number_index, value):
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
def _list_setitem_with_list(data, number_index, value):
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
def _list_setitem_with_tuple(data, number_index, value):
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


@setitem.register("List", "Slice", "Tuple")
def _list_slice_setitem_with_tuple(data, slice_index, value):
    """
    Assigns value to list.

    Inputs:
        data (list): Data of type list.
        slice_index (slice): Index of data.
        value (tuple): Value given.

    Outputs:
        list, type is the same as the element type of data.
    """
    if F.is_sequence_shape_unknown(data) or F.is_sequence_shape_unknown(value) or not F.isconstant(slice_index):
        start = slice_get_item(slice_index, "start")
        stop = slice_get_item(slice_index, "stop")
        step = slice_get_item(slice_index, "step")
        return sequence_slice_setitem(data, value, start, stop, step)
    list_value = list(value)
    return _list_slice_set_item(data, slice_index, list_value)


@setitem.register("List", "Slice", "List")
def _list_slice_setitem_with_list(data, slice_index, value):
    """
    Assigns value to list.

    Inputs:
        data (list): Data of type list.
        slice_index (slice): Index of data.
        value (list): Value given.

    Outputs:
        list, type is the same as the element type of data.
    """
    if F.is_sequence_shape_unknown(data) or F.is_sequence_shape_unknown(value) or not F.isconstant(slice_index):
        start = slice_get_item(slice_index, "start")
        stop = slice_get_item(slice_index, "stop")
        step = slice_get_item(slice_index, "step")
        return sequence_slice_setitem(data, value, start, stop, step)
    return _list_slice_set_item(data, slice_index, value)


@setitem.register("List", "Slice", "Tensor")
def _list_slice_setitem_with_tensor(data, slice_index, value):
    """
    Assigns value to list.

    Inputs:
        data (list): Data of type list.
        slice_index (slice): Index of data.
        value (Tensor): Value given.

    Outputs:
        list, type is the same as the element type of data.
    """
    value_list = list(value)
    if F.is_sequence_shape_unknown(data) or F.is_sequence_shape_unknown(value_list) or not F.isconstant(slice_index):
        start = slice_get_item(slice_index, "start")
        stop = slice_get_item(slice_index, "stop")
        step = slice_get_item(slice_index, "step")
        return sequence_slice_setitem(data, value_list, start, stop, step)
    return _list_slice_set_item(data, slice_index, value_list)


@setitem.register("List", "Slice", "Number")
def _list_slice_setitem_with_number(data, slice_index, value):
    """
    Assigns value to list.

    Inputs:
        data (list): Data of type list.
        slice_index (slice): Index of data.
        value (number): Value given.

    Outputs:
        lis/t, type is the same as the element type of data.
    """
    step = slice_get_item(slice_index, "step")
    if step == 1 or step is None:
        raise TypeError("can only assign an iterable")
    raise TypeError("must assign iterable to extended slice")


@setitem.register("Dictionary", "Tensor", "Tuple")
@setitem.register("Dictionary", "Tensor", "Dictionary")
@setitem.register("Dictionary", "Tensor", "List")
@setitem.register("Dictionary", "Tensor", "Number")
@setitem.register("Dictionary", "Tensor", "Tensor")
@setitem.register("Dictionary", "Tuple", "Tuple")
@setitem.register("Dictionary", "Tuple", "Dictionary")
@setitem.register("Dictionary", "Tuple", "List")
@setitem.register("Dictionary", "Tuple", "Number")
@setitem.register("Dictionary", "Tuple", "Tensor")
@setitem.register("Dictionary", "Number", "Tuple")
@setitem.register("Dictionary", "Number", "Dictionary")
@setitem.register("Dictionary", "Number", "List")
@setitem.register("Dictionary", "Number", "Number")
@setitem.register("Dictionary", "Number", "Tensor")
@setitem.register("Dictionary", "String", "Tuple")
@setitem.register("Dictionary", "String", "Dictionary")
@setitem.register("Dictionary", "String", "List")
@setitem.register("Dictionary", "String", "Number")
@setitem.register("Dictionary", "String", "Tensor")
def _dict_setitem_with_tensor(data, key, value):
    """
    Assigns value to dictionary.

    Inputs:
        data: Data of type dict.
        key: Key of the data.
        value: Value given.

    Outputs:
        dict, type is as same as the element type of data.
    """
    return _dict_setitem(data, key, value)


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
                             3) U is a Tuple.

    Inputs:
        data (Tensor): Assigned tensor.
        index (Tuple): A tuple of tensor, these tensor could be broadcast.
        value (Tuple): Assignment tuple.

    Outputs:
        Tensor, element type and shape is same as data.
    """
    return compile_utils.tensor_setitem_by_tuple_with_sequence(data, tuple_index, value)


@setitem.register("Tensor", "Tuple", "List")
def _tensor_setitem_by_tuple_with_list(data, tuple_index, value):
    """
    Tensor assignment.

    Note:
        Syntax support: A[B, C, D] = U.
        Restraint condition: 1) A is a Tensor, and B, C, D are index Tensors.
                             2) A B and C could be broadcast.
                             3) U is a List.

    Inputs:
        data (Tensor): Assigned tensor.
        index (Tuple): A tuple of tensor, these tensor could be broadcast.
        value (List): Assignment tuple.

    Outputs:
        Tensor, element type and shape is same as data.
    """
    return compile_utils.tensor_setitem_by_tuple_with_sequence(data, tuple_index, value)


@setitem.register("Tensor", "Tensor", "Tuple")
def _tensor_setitem_by_tensor_with_tuple(data, index, value):
    """
    Tensor assignment.

    Inputs:
        data (Tensor): Assigned tensor.
        index (Tensor): Tensor of bool type.
        value (Tuple): Assignment value.

    Outputs:
        Tensor, element type and shape is same as data.
    """
    return compile_utils.tensor_setitem_by_tensor_with_sequence(data, index, value)


@setitem.register("Tensor", "Tensor", "List")
def _tensor_setitem_by_tensor_with_list(data, index, value):
    """
    Tensor assignment.

    Inputs:
        data (Tensor): Assigned tensor.
        index (Tensor): Tensor of bool type.
        value (List): Assignment value.

    Outputs:
        Tensor, element type and shape is same as data.
    """
    return compile_utils.tensor_setitem_by_tensor_with_sequence(data, index, value)


@setitem.register("Tensor", "Slice", "Tensor")
def _tensor_setitem_by_slice_with_tensor(data, input_slice, value):
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
    return compile_utils.tensor_setitem_by_slice(data, input_slice, value)


@setitem.register("Tensor", "Slice", "Number")
def _tensor_setitem_by_slice_with_number(data, input_slice, value):
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
    return compile_utils.tensor_setitem_by_slice(data, input_slice, value)


@setitem.register("Tensor", "Slice", "List")
def _tensor_setitem_by_slice_with_list(data, input_slice, value):
    """
    Tensor assignment.

    Note:
        Syntax support: A[Slice] = u
        Restraint condition: A is a Tensor.
                             Slice like "1:3"
                             u is a list

    Inputs:
        data (Tensor): Assigned tensor.
        input_slice (Slice): slice expression.
        value (List): Assignment value.

    Outputs:
        Tensor, element type and shape is same as data.
    """
    return compile_utils.tensor_setitem_by_slice(data, input_slice, value)


@setitem.register("Tensor", "Slice", "Tuple")
def _tensor_setitem_by_slice_with_tuple(data, input_slice, value):
    """
    Tensor assignment.

    Note:
        Syntax support: A[Slice] = u
        Restraint condition: A is a Tensor.
                             Slice like "1:3"
                             u is a tuple

    Inputs:
        data (Tensor): Assigned tensor.
        input_slice (Slice): slice expression.
        value (Tuple): Assignment value.

    Outputs:
        Tensor, element type and shape is same as data.
    """
    return compile_utils.tensor_setitem_by_slice(data, input_slice, value)


@setitem.register("Tensor", "Number", "Number")
def _tensor_setitem_by_number_with_number(data, index, value):
    """
    Tensor assignment.

    Note:
        Syntax support: A[Number] = u
        Restraint condition: A is a Tensor.
                             u is a Number.

    Inputs:
        data (Tensor): Assigned tensor.
        index (Number): An integer index.
        value (Tuple): Assignment value.

    Outputs:
        Tensor, element type and shape is same as data.
    """
    if isinstance(index, bool):
        return compile_utils.tensor_setitem_by_bool(data, index, value)
    return compile_utils.tensor_setitem_by_number_with_number(data, index, value)


@setitem.register("Tensor", "Number", "Tensor")
def _tensor_setitem_by_number_with_tensor(data, index, value):
    """
    Tensor assignment.

    Note:
        Syntax support: A[Number] = u
        Restraint condition: A is a Tensor.
                             u is a Tensor.

    Inputs:
        data (Tensor): Assigned tensor.
        index (Number): An integer index.
        value (Tensor): Assignment value.

    Outputs:
        Tensor, element type and shape is same as data.
    """
    if isinstance(index, bool):
        return compile_utils.tensor_setitem_by_bool(data, index, value)
    return compile_utils.tensor_setitem_by_number_with_tensor(data, index, value)


@setitem.register("Tensor", "Number", "Tuple")
def _tensor_setitem_by_number_with_tuple(data, index, value):
    """
    Tensor assignment.

    Note:
        Syntax support: A[Number] = u
        Restraint condition: A is a Tensor.
                             u is a Tuple, with all elements equal in length.

    Inputs:
        data (Tensor): Assigned tensor.
        index (Number): An integer index.
        value (Tuple): Assignment value.

    Outputs:
        Tensor, element type and shape is same as data.
    """
    if isinstance(index, bool):
        return compile_utils.tensor_setitem_by_bool(data, index, value)
    return compile_utils.tensor_setitem_by_number_with_sequence(data, index, value)


@setitem.register("Tensor", "Number", "List")
def _tensor_setitem_by_number_with_list(data, index, value):
    """
    Tensor assignment.

    Note:
        Syntax support: A[Number] = u
        Restraint condition: A is a Tensor.
                             u is a List, with all elements equal in length.

    Inputs:
        data (Tensor): Assigned tensor.
        index (Number): An integer index.
        value (List): Assignment value.

    Outputs:
        Tensor, element type and shape is same as data.
    """
    if isinstance(index, bool):
        return compile_utils.tensor_setitem_by_bool(data, index, value)
    return compile_utils.tensor_setitem_by_number_with_sequence(data, index, value)


@setitem.register("Tensor", "Ellipsis", "Number")
def _tensor_setitem_by_ellipsis_with_number(data, index, value):
    """
    Tensor assignment.

    Note:
        Syntax support: A[...] = u
        Restraint condition: A is a Tensor.
                             u is a Number.
    Inputs:
        data (Tensor): Assigned tensor.
        index (Ellipsis): Index is ``...``.
        value (Number): Assignment value.

    Outputs:
        Tensor, element type and shape is same as data.
    """
    return compile_utils.tensor_setitem_by_ellipsis_with_number(data, value)


@setitem.register("Tensor", "Ellipsis", "Tensor")
def _tensor_setitem_by_ellipsis_with_tensor(data, index, value):
    """
    Tensor assignment.

    Note:
        Syntax support: A[...] = u
        Restraint condition: A is a Tensor.
                             u is a Tensor.
    Inputs:
        data (Tensor): Assigned tensor.
        index (Ellipsis): Index is ``...``.
        value (Tensor): Assignment value.

    Outputs:
        Tensor, element type and shape is same as data.
    """
    return compile_utils.tensor_setitem_by_ellipsis_with_tensor(data, value)


@setitem.register("Tensor", "Ellipsis", "List")
def _tensor_setitem_by_ellipsis_with_list(data, index, value):
    """
    Tensor assignment.

    Note:
        Syntax support: A[...] = u
        Restraint condition: A is a Tensor.
                             u is a List, with all elements equal in length.
    Inputs:
        data (Tensor): Assigned tensor.
        index (Ellipsis): Index is ``...``.
        value (Number): Assignment value.

    Outputs:
        Tensor, element type and shape is same as data.
    """
    return compile_utils.tensor_setitem_by_ellipsis_with_sequence(data, value)


@setitem.register("Tensor", "Ellipsis", "Tuple")
def _tensor_setitem_by_ellipsis_with_tuple(data, index, value):
    """
    Tensor assignment.

    Note:
        Syntax support: A[...] = u
        Restraint condition: A is a Tensor.
                             u is a Tuple, with all elements equal in length.
    Inputs:
        data (Tensor): Assigned tensor.
        index (Ellipsis): Index is ``...``.
        value (Number): Assignment value.

    Outputs:
        Tensor, element type and shape is same as data.
    """
    return compile_utils.tensor_setitem_by_ellipsis_with_sequence(data, value)


@setitem.register("Tensor", "None", "Number")
def _tensor_setitem_by_none_with_number(data, index, value):
    """
    Tensor assignment.

    Note:
        Syntax support: A[...] = u
        Restraint condition: A is a Tensor.
                             u is a Number.
    Inputs:
        data (Tensor): Assigned tensor.
        index (None): Index is ``...``.
        value (Number): Assignment value.

    Outputs:
        Tensor, element type and shape is same as data.
    """
    return compile_utils.tensor_setitem_by_ellipsis_with_number(data, value)


@setitem.register("Tensor", "None", "Tensor")
def _tensor_setitem_by_none_with_tensor(data, index, value):
    """
    Tensor assignment.

    Note:
        Syntax support: A[...] = u
        Restraint condition: A is a Tensor.
                             u is a Tensor.
    Inputs:
        data (Tensor): Assigned tensor.
        index (None): Index is ``...``.
        value (Tensor): Assignment value.

    Outputs:
        Tensor, element type and shape is same as data.
    """
    return compile_utils.tensor_setitem_by_ellipsis_with_tensor(data, value)


@setitem.register("Tensor", "None", "List")
def _tensor_setitem_by_none_with_list(data, index, value):
    """
    Tensor assignment.

    Note:
        Syntax support: A[...] = u
        Restraint condition: A is a Tensor.
                             u is a List, with all elements equal in length.
    Inputs:
        data (Tensor): Assigned tensor.
        index (None): Index is ``...``.
        value (Number): Assignment value.

    Outputs:
        Tensor, element type and shape is same as data.
    """
    return compile_utils.tensor_setitem_by_ellipsis_with_sequence(data, value)


@setitem.register("Tensor", "None", "Tuple")
def _tensor_setitem_by_none_with_tuple(data, index, value):
    """
    Tensor assignment.

    Note:
        Syntax support: A[...] = u
        Restraint condition: A is a Tensor.
                             u is a Tuple, with all elements equal in length.
    Inputs:
        data (Tensor): Assigned tensor.
        index (None): Index is ``...``.
        value (Number): Assignment value.

    Outputs:
        Tensor, element type and shape is same as data.
    """
    return compile_utils.tensor_setitem_by_ellipsis_with_sequence(data, value)


@setitem.register("Tensor", "List", "Number")
def _tensor_setitem_by_list_with_number(data, index, value):
    """
    Tensor assignment.

    Note:
        Syntax support: A[List] = u
        Restraint condition: A is a Tensor.
                             u is a Number.
    Inputs:
        data (Tensor): Assigned tensor.
        index (List).
        value (Number): Assignment value.

    Outputs:
        Tensor, element type and shape is same as data.
    """
    # list indices will be converted to tuple or tensor based on its contents.
    index = compile_utils.format_list_indices(index, data.shape[0])
    if isinstance(index, Tensor):
        return compile_utils.tensor_setitem_by_tensor_with_number(data, index, value)
    return compile_utils.tensor_setitem_by_tuple_with_number(data, index, value)


@setitem.register("Tensor", "List", "Tensor")
def _tensor_setitem_by_list_with_tensor(data, index, value):
    """
    Tensor assignment.

    Note:
        Syntax support: A[List] = u
        Restraint condition: A is a Tensor.
                             u is a Tensor.
    Inputs:
        data (Tensor): Assigned tensor.
        index (List).
        value (Tensor): Assignment value.

    Outputs:
        Tensor, element type and shape is same as data.
    """
    # list indices will be converted to tuple or tensor based on its contents.
    index = compile_utils.format_list_indices(index, data.shape[0])
    if isinstance(index, Tensor):
        return compile_utils.tensor_setitem_by_tensor_with_tensor(data, index, value)
    return compile_utils.tensor_setitem_by_tuple_with_tensor(data, index, value)


@setitem.register("Tensor", "List", "Tuple")
def _tensor_setitem_by_list_with_tuple(data, index, value):
    """
    Tensor assignment.

    Note:
        Syntax support: A[List] = u
        Restraint condition: A is a Tensor.
                             u is a Tuple, with all elements equal in length.
    Inputs:
        data (Tensor): Assigned tensor.
        index (List).
        value (Tuple): Assignment value.

    Outputs:
        Tensor, element type and shape is same as data.
    """
    # list indices will be converted to tuple or tensor based on its contents.
    index = compile_utils.format_list_indices(index, data.shape[0])
    if isinstance(index, Tensor):
        return compile_utils.tensor_setitem_by_tensor_with_sequence(data, index, value)
    return compile_utils.tensor_setitem_by_tuple_with_sequence(data, index, value)


@setitem.register("Tensor", "List", "List")
def _tensor_setitem_by_list_with_list(data, index, value):
    """
    Tensor assignment.

    Note:
        Syntax support: A[List] = u
        Restraint condition: A is a Tensor.
                             u is a List, with all elements equal in length.
    Inputs:
        data (Tensor): Assigned tensor.
        index (List).
        value (List): Assignment value.

    Outputs:
        Tensor, element type and shape is same as data.
    """
    # list indices will be converted to tuple or tensor based on its contents.
    index = compile_utils.format_list_indices(index, data.shape[0])
    if isinstance(index, Tensor):
        return compile_utils.tensor_setitem_by_tensor_with_sequence(data, index, value)
    return compile_utils.tensor_setitem_by_tuple_with_sequence(data, index, value)


@setitem.register("MapTensor", "Tensor", "Tensor")
def _map_tensor_setitem(map_tensor, key_tensor, value_tensor):
    """
    Update or insert to map tensor by key tensor and value tensor.

    Inputs:
        map_tensor (MapTensor): A map tensor.
        key_tensor (Tensor): The key tensor.
        value_tensor (Tensor): The value tensor.

    Outputs:
        MapTensor, the map tensor be updated.
    """
    _map_tensor_ops.put(map_tensor, key_tensor, value_tensor)
    return map_tensor
