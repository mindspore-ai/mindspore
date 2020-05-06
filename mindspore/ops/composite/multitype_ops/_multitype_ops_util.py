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

"""constexpr util"""

from functools import reduce
import numpy as np
from ...primitive import constexpr
from ....common.tensor import Tensor
from ....common import dtype as mstype
from ...._extends.utils import Slice, Ellipsis_

@constexpr
def check_equal(param1, param2, msg="{},{}"):
    """Checks whether the two parameters are equal or not."""
    if param1 != param2:
        raise ValueError(msg.format(param1, param2))
    return param1


@constexpr
def check_ellipsis_shape_size(data_shape, value_shape, data_size, value_size):
    """Checks the shape and size of the sensor and value."""
    if data_shape == value_shape or data_size == value_size or value_size == 1:
        return True
    raise ValueError("The value(shape={}), can not assign to tensor(shape={}).".format(value_shape, data_shape))


@constexpr
def check_tensor_setitem_index(index, element_type=None):
    """Checks tuple index type of tensor assignment."""
    if index is None:
        raise IndexError("Tensor's index cannot be None.")
    # eg. Tensor[Slice] = u
    if isinstance(index, Slice):
        return True
    # eg. Tensor[tuple] = u
    if isinstance(index, tuple):
        if not index:
            raise IndexError("Tensor's index cannot be empty.")
        # eg. Tensor[tuple(Slice...)] = u
        if isinstance(index[0], (Slice, Ellipsis_, int)):
            return True
        raise IndexError("Index of type '{}' is not supported yet.".format(type(index[0])))
    # eg. Tensor[Tensor[dtype=bool]] = u
    if index == mstype.tensor:
        if element_type is None or element_type != mstype.bool_:
            raise TypeError(
                "The index of tensor should be a bool type tensor. "
                "{} type is not supported yet.".format(element_type))
        return True

    raise IndexError("Index of type '{}' is not supported yet.".format(type(index)))


@constexpr
def is_same_type(inst, type_):
    """
    Checks whether an object is an instance of a target type.

    Inputs:
        inst (mindspore.dtype): Inspected type.
        type_ (mindspore.dtype): Target type.

    Outputs:
        bool, the check result.
    """
    return inst == type_


def slice_expand(input_slices, shape):
    """
    Converts slice to indices.

    Inputs:
        slices (Union[Slice, tuple[Slice]]): Slice tuple or slice.
        shape (tuple): The shape of a sensor is an integer element tuple.

    Outputs:
        tuple[list], This is expressed as (begins, ends, strides).
    """
    begin = []
    end = []
    strides = []
    index = 0
    slices = None
    # Slice or tuple(Slice...)
    if isinstance(input_slices, Slice):
        slices = (input_slices,)
    elif isinstance(input_slices, (tuple, list)) and input_slices and isinstance(input_slices[0], (Slice, Ellipsis_)):
        is_have_ellipsis = False
        for _, element in enumerate(input_slices):
            if isinstance(element, Ellipsis_):
                is_have_ellipsis = True
                break
        if is_have_ellipsis:
            slices = ellipsis2slice(input_slices, shape)
        else:
            slices = input_slices
    else:
        raise IndexError("Tensor's index type is not supported yet.")

    for s in slices:
        start = 0 if (s.start is None) else s.start
        stop = shape[index] if (s.end is None) else s.end
        step = 1 if (s.step is None) else s.step
        begin.append(start)
        end.append(stop)
        strides.append(step)
        index += 1
    while index < len(shape):
        begin.append(0)
        end.append(shape[index])
        strides.append(1)
        index += 1
    return begin, end, strides


def ellipsis2slice(input_, shape):
    """Converts ellipsis to slice."""
    input_slice = input_
    result = []
    if isinstance(input_, Ellipsis_):
        input_slice = (input_,)
    ell_count = 0
    for _, element in enumerate(input_slice):
        if not isinstance(element, Ellipsis_):
            result.append(element)
            continue
        ell_count += 1
        if ell_count > 1:
            raise IndexError("There cannot be more than one ellisis (...) in the index of the tensor, "
                             "but it is currently {}".format(input_slice))
        for _ in range(len(shape) - len(input_slice) + 1):
            result.append(Slice(None, None, None))
    return tuple(result)


@constexpr
def slice2indices(input_slices, shape):
    """
    Converts slice to indices.

    Inputs:
        slices (Union[Slice, tuple[Slice]]): Slice tuple or slice.
        shape (tuple): The shape of a tensor is an integer element tuple.

    Outputs:
        Tensor, the shape is (n, 1).
    """
    begin, end, strides = slice_expand(input_slices, shape)
    np_r = []
    for i, element in enumerate(shape):
        s = begin[i] if (begin[i] >= 0) else (element + begin[i])
        e = end[i] if (end[i] >= 0) else (element + end[i])
        np_r.append(np.r_[s:e:strides[i]])
    # Reference: np.ravel_multi_index((np.ix_(np.r_[1:3:1], np.r_[0:4:1], np.r_[4:0:-1])), a.shape)
    np_ix = np.ix_(*np_r)
    ravel = np.ravel_multi_index(np_ix, shape)
    ravel = Tensor(ravel.reshape(-1, 1), dtype=mstype.int32)
    return ravel

@constexpr
def check_indices(indices_size, index):
    """Checks indices whether is empty."""
    if indices_size < 1:
        raise IndexError("The tensor's index is unreasonable. index:{}".format(index))
    return indices_size


@constexpr
def check_indices_value_size(indices_size, value_size):
    """Checks if the sizes are already matched."""
    if value_size < 1:
        raise ValueError("The value assigned to tensor cannot be empty.")
    if value_size > 1:
        if value_size != indices_size:
            raise ValueError(
                "The value given to tensor does not match the index size,"
                " value size:{}, indics size:{}".format(value_size, indices_size))
    return value_size

@constexpr
def integer_to_indices(index, shape):
    """Converts int or tuple[int] to indices."""
    size = reduce(lambda x, y: x * y, shape)
    range_ = np.arange(size).reshape(shape)
    value = range_[index]
    value = value.reshape(-1, 1)
    return Tensor(value, dtype=mstype.int32)

@constexpr
def tuple_element_is_slice(indexs):
    """Judges tuple element type."""
    if not indexs:
        raise IndexError("Tensor's index cannot be empty.")
    if isinstance(indexs, tuple):
        for _, ele in enumerate(indexs):
            if not isinstance(ele, Slice):
                return False
        return True
    return False

@constexpr
def tuple_element_is_int(indexs):
    """Judges tuple element type."""
    if not indexs:
        raise IndexError("Tensor's index cannot be empty.")
    if isinstance(indexs, tuple):
        for _, ele in enumerate(indexs):
            if not isinstance(ele, int):
                return False
        return True
    return False
