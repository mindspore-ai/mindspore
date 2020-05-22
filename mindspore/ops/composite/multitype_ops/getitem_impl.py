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

from . import _utils as multi_utils
from ..import base
from ... import functional as F
from ....common import dtype as mstype

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


class _TupleGetItemTensor(base.TupleGetItemTensor_):
    """
    Getting item of tuple by tensor index.

    Inputs:
        data (tuple): A tuple of items.
        index (Tensor): The index in tensor.
    Outputs:
        Type, is same as the element type of data.
    """

    def __init__(self, name):
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


@getitem.register("Tuple", "Tensor")
def _tuple_getitem_by_tensor(data, tensor_index):
    """
    Getting item out of tuple by tensor index.

    Inputs:
        data (tuple): A tuple of items to index.
        tensor_index (Tensor): Index to select item.

    Outputs:
        Type, is same as the element type of data.
    """
    return _tuple_get_item_tensor(data, tensor_index)


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


@getitem.register("Tensor", "None")
def _tensor_getitem_by_none(data, index):
    """
    Getting item of tensor by None.

    Inputs:
        data (Tensor): A tensor.
        index (None): None.

    Outputs:
        Tensor, element type is as same as the element type of data.
    """
    return _tensor_slice(data, index)


@getitem.register("Tensor", "Slice")
def _tensor_getitem_by_slice(data, slice_index):
    """
    Getting item of tensor by slice.

    Inputs:
        data (Tensor): A tensor.
        slice_index (Slice): Index in slice.

    Outputs:
        Tensor, element type is same as the element type of data.
    """
    return _tensor_slice(data, slice_index)


@getitem.register("Tensor", "Tensor")
def _tensor_getitem_by_tensor(data, tensor_index):
    """
    Getting item of tensor by slice.

    Inputs:
        data (Tensor): A tensor.
        tensor_index (Tensor): An index expressed by tensor.

    Outputs:
        Tensor, element type is same as the element type of data.
    """
    check_dtypes = multi_utils.check_tensor_dtype_valid(F.dtype(tensor_index), (mstype.int32, mstype.int64))
    result = None
    if check_dtypes:
        result = F.gather(data, tensor_index, 0)
    return result


@getitem.register("Tensor", "Tuple")
def _tensor_getitem_by_tuple(data, tuple_index):
    """
    Getting item of tensor by slice tuple.

    Inputs:
        data (Tensor): A tensor.
        tuple_index (tuple): Index in tuple.

    Outputs:
        Tensor, element type is same as the element type of data.
    """
    index_types = multi_utils.hyper_map(F.typeof, tuple_index)
    index_elements_type = multi_utils.tuple_index_elements_type(index_types, multi_utils.TENSOR_GETITEM)
    result = None
    if index_elements_type == multi_utils.NO_TENSOR:
        result = _tensor_slice(data, tuple_index)
    if index_elements_type == multi_utils.ALL_TENSOR:
        result = _tensor_getitem_by_tuple_of_tensor(data, tuple_index)
    return result


@getitem.register("Tensor", "Ellipsis")
def _tensor_getitem_by_ellipsis(data, ellipsis_index):
    """
    Getting item of tensor by Ellipsis.

    Inputs:
        data (Tensor): A tensor.
        ellipsis (Ellipsis): A Ellipsis object.

    Outputs:
        Tensor, same as data.
    """
    return _tensor_slice(data, ellipsis_index)


def _tensor_getitem_by_tuple_of_tensor(data, tuple_index):
    """Tensor getitem by a tuple of tensor."""
    indices = multi_utils.generate_indeices_from_tuple_of_tensor(data, tuple_index, multi_utils.TENSOR_GETITEM)
    result = F.gather_nd(data, indices)
    return result
