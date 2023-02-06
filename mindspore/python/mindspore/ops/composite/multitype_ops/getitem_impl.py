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

"""Implementation for internal polymorphism `getitem` operations."""
from __future__ import absolute_import
from mindspore.ops.operations import _inner_ops as inner
from mindspore.ops.operations import _map_tensor_ops
from mindspore.ops.composite.multitype_ops import _compile_utils as compile_utils
from mindspore.ops.composite import base
from mindspore.ops import functional as F
from mindspore.ops.operations._inner_ops import SliceGetItem
from ...operations._sequence_ops import SequenceSlice

DOC_URL = "https://mindspore.cn/docs/zh-CN/master/note/index_support.html"

getitem = base.MultitypeFuncGraph('getitem', doc_url=DOC_URL)
"""
getitem is a metafuncgraph object which will get item from an object according to input type
using ".register" decorator.
"""

slice_getitem = SliceGetItem()
sequence_slice = SequenceSlice()


class _TupleSlice(base.SequenceSliceGetItem_):
    """
    Slices a tuple.

    Inputs:
        data (tuple): A tuple to be sliced.
        s (slice): The index to slice tuple data.

    Outputs:
        Tuple, consists of some elements of data.
    """

    def __init__(self, name):
        """Initialize _TupleSlice."""
        base.SequenceSliceGetItem_.__init__(self, name, "MakeTuple", "TupleGetItem")

    def __call__(self, *args):
        pass


_tuple_slice = _TupleSlice('tuple_slice')
"""_tuple_slice is a metafuncgraph object which will slice a tuple."""


class _ListSlice(base.SequenceSliceGetItem_):
    """
    Slices a List.

    Inputs:
        data (List): A List to be sliced.
        s (slice): The index to slice list data.

    Outputs:
        List, consists of some elements of data.
    """

    def __init__(self, name):
        """Initialize _TupleSlice."""
        base.SequenceSliceGetItem_.__init__(self, name, "make_list", "list_getitem")

    def __call__(self, *args):
        pass


_list_slice = _ListSlice('list_slice')
"""_list_slice is a metafuncgraph object which will slice a list."""


class _TupleGetItemTensor(base.TupleGetItemTensor_):
    """
    Getting item of tuple by tensor index.

    Inputs:
        data (tuple): A tuple of items.
        index (Tensor): The index in tensor.
    Outputs:
        Type, is the same as the element type of data.
    """

    def __init__(self, name):
        """Initialize _TupleGetItemTensor."""
        base.TupleGetItemTensor_.__init__(self, name)

    def __call__(self, *args):
        pass


_tuple_get_item_tensor = _TupleGetItemTensor('tuple_get_item_tensor')
"""_tuple_get_item_tensor is an metafuncgraph object which will select indexed item."""


@getitem.register("Tuple", "Number")
def _tuple_getitem_by_number(data, number_index):
    """
    Getting item of tuple by number index.

    Inputs:
        data (tuple): A tuple to be sliced.
        number_index (Number): Index in scalar.

    Outputs:
        Type, is the same as the element type of data.
    """
    return F.tuple_getitem(data, number_index)


@getitem.register("Tuple", "Slice")
def _tuple_getitem_by_slice(data, slice_index):
    """
    Getting item of tuple by slice index.

    Inputs:
        data (tuple): data
        slice_index (Slice): Index in slice.

    Outputs:
        Tuple, element type is the same as the element type of data.
    """
    if F.is_sequence_shape_unknown(data) or not F.isconstant(slice_index):
        start = slice_getitem(slice_index, "start")
        stop = slice_getitem(slice_index, "stop")
        step = slice_getitem(slice_index, "step")
        return sequence_slice(data, start, stop, step)
    return _tuple_slice(data, slice_index)


@getitem.register("Tuple", "Tensor")
def _tuple_getitem_by_tensor(data, tensor_index):
    """
    Getting item out of tuple by tensor index.

    Inputs:
        data (tuple): A tuple of items to index.
        tensor_index (Tensor): Index to select item.

    Outputs:
        Type, is the same as the element type of data.
    """
    tensor_index = F.select(tensor_index >= 0, tensor_index, tensor_index + len(data))
    return _tuple_get_item_tensor(data, tensor_index)


@getitem.register("List", "Number")
def _list_getitem_by_number(data, number_index):
    """
    Getting item of list by number index.

    Inputs:
        data (list): data in list.
        number_index (Number): Index in scalar.

    Outputs:
        Type is the same as the element type of data.
    """
    return F.list_getitem(data, number_index)


@getitem.register("List", "Slice")
def _list_getitem_by_slice(data, slice_index):
    """
    Getting item of list by slice index.

    Inputs:
        data (list): data
        slice_index (Slice): Index in slice.

    Outputs:
        List, element type is the same as the element type of data.
    """
    if F.is_sequence_shape_unknown(data) or not F.isconstant(slice_index):
        start = slice_getitem(slice_index, "start")
        stop = slice_getitem(slice_index, "stop")
        step = slice_getitem(slice_index, "step")
        return sequence_slice(data, start, stop, step)
    return _list_slice(data, slice_index)


@getitem.register("Dictionary", "Tensor")
@getitem.register("Dictionary", "Tuple")
@getitem.register("Dictionary", "Number")
@getitem.register("Dictionary", "String")
def _dict_getitem_by_key(data, key):
    """
    Getting item of dictionary by key.

    Inputs:
        data (Dictionary): data
        key (Tensor, Tuple, Number, String): Key of the data.

    Outputs:
        Type, is as same as the element type of data.
    """
    return F.dict_getitem(data, key)


@getitem.register("String", "Number")
def _string_getitem_by_number(data, number_index):
    """
    Getting item of string by number index.

    Inputs:
        data (String): A string.
        number_index (Number): Index in scalar.

    Outputs:
        String.
    """
    return inner.string_getitem(data, number_index)


@getitem.register("Tensor", "Number")
def _tensor_getitem_by_number(data, number_index):
    """
    Getting item of tensor by number index.

    Inputs:
        data (Tensor): A tensor.
        number_index (Number): Index in scalar.

    Outputs:
        Tensor, element type is as same as the element type of data.
    """
    return compile_utils.tensor_index_by_number(data, number_index)


@getitem.register("Tensor", "None")
def _tensor_getitem_by_none(data, _):
    """
    For none indexing , expand data with one dim.

    Inputs:
        data (Tensor): A tensor.
        index (None): None.

    Outputs:
        Tensor, element type is as same as the element type of data.
    """
    return F.expand_dims(data, 0)


@getitem.register("Tensor", "Slice")
def _tensor_getitem_by_slice(data, slice_index):
    """
    Getting item of tensor by slice.

    Inputs:
        data (Tensor): A tensor.
        slice_index (Slice): Index in slice.

    Outputs:
        Tensor, element type is the same as the element type of data.
    """
    return compile_utils.tensor_index_by_slice(data, slice_index)


@getitem.register("Tensor", "Tensor")
def _tensor_getitem_by_tensor(data, tensor_index):
    """
    Getting item of tensor by tensor indices.

    Inputs:
        data (Tensor): A tensor.
        tensor_index (Tensor): An index expressed by tensor.

    Outputs:
        Tensor, element type is the same as the element type of data.
    """
    return compile_utils.tensor_index_by_tensor(data, tensor_index)


@getitem.register("Tensor", "Ellipsis")
def _tensor_getitem_by_ellipsis(data, _):
    """
    Getting item of tensor by Ellipsis.

    Inputs:
        data (Tensor): A tensor.
        ellipsis (Ellipsis): A Ellipsis object.

    Outputs:
        Tensor, same as data.
    """
    return data


@getitem.register("Tensor", "List")
def _tensor_getitem_by_list(data, list_index):
    """
    Getting item of tensor by list.

    Inputs:
        data (Tensor): A tensor
        list_index (List): A list object.

    Outputs:
        Tensor ,same as data.
    """
    return compile_utils.tensor_index_by_list(data, list_index)


@getitem.register("Tensor", "Tuple")
def _tensor_getitem_by_tuple(data, tuple_index):
    """
    Getting item of tensor by tuple.

    Inputs:
        data (Tensor): A tensor.
        tuple_index (tuple): Index in tuple which include ellipsis, slice, int, Tensor, None, list, tuple.

    Outputs:
        Tensor, element type is the same as the element type of data.
    """
    return compile_utils.tensor_index_by_tuple(data, tuple_index)


@getitem.register("MapTensor", "Tensor")
def _map_tensor_getitem(map_tensor, key_tensor):
    """
    Getting value tensor from map tensor by key tensor.

    Inputs:
        map_tensor (MapTensor): A map tensor.
        key_tensor (Tensor): The key tensor.

    Outputs:
        Tensor, value tensor according the key tensor.
    """
    return _map_tensor_ops.MapTensorGet(True)(map_tensor, key_tensor)
