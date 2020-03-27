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

"""Implementation for getitem."""

from ...composite import base
from ... import functional as F


getitem = base.MultitypeFuncGraph('getitem')
"""
getitem is a metafuncgraph object which will get item from an object according to input type
using ".register" decorator.
"""


class _TupleSlice(base.TupleSlice_):
    """
    Slices a tuple.

    Inputs:
        data (tuple): A tuple to be sliced.
        s (slice): The index to slice tuple data.

    Outputs:
        Tuple, consists of some elements of data.
    """

    def __init__(self, name):
        base.TupleSlice_.__init__(self, name)

    def __call__(self, *args):
        pass


_tuple_slice = _TupleSlice('tuple_slice')
"""_tuple_slice is an metafuncgraph object which will slice a tuple."""


class _TensorSlice(base.TensorSlice_):
    """
    Slices a tensor.

    Inputs:
        data (Tensor): A tensor to be sliced.
        s (slice): The index to slice tuple data.

    Outputs:
        Tensor, consists of some elements of data.
    """

    def __init__(self, name):
        base.TensorSlice_.__init__(self, name)

    def __call__(self, *args):
        pass


_tensor_slice = _TensorSlice('tensor_slice')
"""_tensor_slice is an metafuncgraph object which will slice a tensor."""


@getitem.register("Tuple", "Number")
def _tuple_getitem_by_number(data, number_index):
    """
    Getting item of tuple by number index.

    Inputs:
        data (tuple): A tuple to be sliced.
        number_index (Number): Index in scalar.

    Outputs:
        Type, is same as the element type of data.
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
        Tuple, element type is same as the element type of data.
    """
    return _tuple_slice(data, slice_index)


@getitem.register("List", "Number")
def _list_getitem_by_number(data, number_index):
    """
    Getting item of list by number index.

    Inputs:
        data (list): data in list.
        number_index (Number): Index in scalar.

    Outputs:
        Type is same as the element type of data.
    """
    return F.list_getitem(data, number_index)


@getitem.register("Dictionary", "String")
def _dict_getitem_by_key(data, key):
    """
    Getting item of dictionary by key which is a string.

    Inputs:
        data (Dictionary): data
        key (str): Key of the data.

    Outputs:
        Type, is as same as the element type of data.
    """
    return F.dict_getitem(data, key)


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
    return _tensor_slice(data, number_index)


@getitem.register("Tensor", "Slice")
def _tensor_getitem_by_slice(data, slice_index):
    """
    Getting item of tensor by slice index.

    Inputs:
        data (Tensor): A tensor.
        slice_index (Slice): Index in slice.

    Outputs:
        Tensor, element type is same as the element type of data.
    """
    return _tensor_slice(data, slice_index)


@getitem.register("Tensor", "Tuple")
def _tensor_getitem_by_slice_tuple(data, slice_tuple_index):
    """
    Getting item of tensor by slice tuple index.

    Inputs:
        data (Tensor): A tensor.
        slice_tuple_index (tuple): Index in tuple.

    Outputs:
        Tensor, element type is same as the element type of data.
    """
    return _tensor_slice(data, slice_tuple_index)
