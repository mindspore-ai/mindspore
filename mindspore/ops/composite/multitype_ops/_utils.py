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
from ....ops import _utils as op_utils
from ...composite import base
from .... import log as logger
from ... import functional as F
from ... import operations as P

hyper_map = base.HyperMap()
pack = P.Pack(axis=-1)

ALL_TENSOR = 0
NO_TENSOR = 1
CONTAIN_TENSOR = 2
ALL_SCALAR = 3

INT_ = 0
BOOL_ = 1
UNSUPPORTED_DTYPE = 2

TENSOR_SETITEM = "tensor setitem"
TENSOR_GETITEM = "tensor getitem"

SET_ITEM_BY_ONE_TENSOR = 0
SET_ITEM_BY_TUPLE_OF_TENSOR = 1


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
    if isinstance(index, mstype.tensor_type):
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


@constexpr
def tuple_elements_type(types):
    """Judges the type of all elements of the tuple."""
    tensors_number = 0
    for ele in types:
        if isinstance(ele, mstype.tensor_type):
            tensors_number += 1
    if tensors_number == len(types):
        return ALL_TENSOR
    if tensors_number == 0:
        return NO_TENSOR
    return CONTAIN_TENSOR


@constexpr
def check_value_elements(data_dtype, types):
    """Judges the type of all elements of the tuple."""
    tensors_number = 0
    scalars_number = 0
    for i, ele in enumerate(types):
        if isinstance(ele, mstype.tensor_type):
            ele_dtype = ele.element_type()
            if data_dtype == ele_dtype:
                tensors_number += 1
            else:
                raise TypeError(f"For '{TENSOR_SETITEM}', the data type of {i}th tensor '{ele_dtype}' "
                                f"in value tuple is not consistent with origin tensor data type '{data_dtype}'.")
        elif mstype.issubclass_(ele, data_dtype):
            scalars_number += 1
        else:
            raise TypeError(f"For '{TENSOR_SETITEM}', the {i}th element type '{ele}' in "
                            f"value tuple is not consistent with origin tensor data type '{data_dtype}'.")
    if tensors_number == len(types):
        return ALL_TENSOR
    if scalars_number == len(types):
        return ALL_SCALAR
    raise TypeError(f"For '{TENSOR_SETITEM}', the value does not support scalar and tensor mixing, but got {types}.")


@constexpr
def get_index_tensor_dtype(dtype):
    """Check a tuple of tensor data type."""
    if dtype == mstype.int32:
        return INT_
    if dtype == mstype.bool_:
        return BOOL_
    raise TypeError(f"For '{TENSOR_SETITEM}', the index tensor data type '{dtype}' is not supported.")


@constexpr
def check_index_tensors_dtype(dtypes, op_name):
    """Check a tuple of tensor data type."""
    if op_name == TENSOR_GETITEM:
        valid_dtypes = (mstype.int32, mstype.int64)
    elif op_name == TENSOR_SETITEM:
        valid_dtypes = (mstype.int32,)
    else:
        raise ValueError("Unsupported operation.")
    for ele in dtypes:
        if ele in valid_dtypes and ele == dtypes[0]:
            continue
        raise TypeError(f"For '{op_name}', the index tensors data type must be same, "
                        f"and should be one of the following: {valid_dtypes}, but got {dtypes}.")
    return True


@constexpr
def check_tensor_dtype_valid(dtype, valid_dtypes):
    """Check a tensor data type."""
    if dtype in valid_dtypes:
        return True
    raise TypeError(f"The index tensor data type must be one of "
                    f"the following: {valid_dtypes}, but got {dtype}.")


@constexpr
def check_tensors_dtype_same(x_dtype, y_dtype, op_name):
    """Check tensors data type same."""
    if x_dtype == y_dtype:
        return True
    raise TypeError(f"For '{op_name}', the value data type '{y_dtype}' "
                    f"is not consistent with origin tensor data type {x_dtype}.")


@constexpr
def broadcast_shapes(shapes, op_name):
    """Broadcasts a tuple of tensor."""
    broadcast_shape = shapes[0]
    for i, shape in enumerate(shapes):
        logger.debug(f"Broadcasts the {i}th tensor, the shape is {shape}.")
        broadcast_shape = op_utils.get_broadcast_shape(broadcast_shape, shape, op_name)
    return tuple(broadcast_shape)


@constexpr
def check_two_shapes_need_broadcast(shape_x, shape_y):
    """Check two shapes need broadcast."""
    error = ValueError(f"For 'tensor setitem with tensor', the value tensor shape "
                       f"{shape_y} could not broadcast the required updates shape {shape_x}.")
    if len(shape_y) > len(shape_x):
        raise error
    for i in range(-len(shape_y), 0):
        if shape_y[i] > shape_x[i]:
            raise error
        if shape_y[i] < shape_x[i] and shape_y[i] != 1:
            raise error
    if shape_y == shape_x:
        return False
    return True


@constexpr
def compute_multiples(origin_shape, broadcast_shape):
    """Compute multiples between broadcast_shape with origin_shape."""
    len_gap = len(broadcast_shape) - len(origin_shape)
    return broadcast_shape[0:len_gap] + tuple(map(lambda x, y: x // y, broadcast_shape[len_gap:], origin_shape))


def tile(broadcast_shape, x):
    multiples = compute_multiples(F.shape(x), broadcast_shape)
    return F.tile(x, multiples)


@constexpr
def check_shapes_same(value_shapes, op_name):
    """Check if the shapes in the tuple are consistent."""
    for i, shape in enumerate(value_shapes):
        if shape != value_shapes[0]:
            raise ValueError(f"For '{op_name}', the {i}th tensor shape in value tuple "
                             f"is not same as the first tensor shape.")
    return True


@constexpr
def convert_scalar_to_tensor(data_shape, data_dtype, indices_shape, value, op_type):
    """Convert a scalar to a tensor."""
    if op_type == SET_ITEM_BY_ONE_TENSOR:
        updates_shape = indices_shape + data_shape[1:]
    else:
        updates_shape = indices_shape[:-1] + data_shape[indices_shape[-1]:]
    if isinstance(value, mstype.dtype_to_pytype(data_dtype)):
        return Tensor(np.full(updates_shape, value), dtype=data_dtype)
    raise TypeError(f"For '{TENSOR_SETITEM}', the value type '{value.__class__.__name__}'"
                    f" is not consistent with tensor data type {data_dtype}.")


@constexpr
def convert_tuple_of_scalar_to_tensor(data_shape, data_dtype, index_shape, value, op_type):
    """Convert a tuple of scalar to a tensor."""
    updates_shape = generate_updates_shape(data_shape, index_shape, op_type)
    if len(value) != updates_shape[-1]:
        raise ValueError(f"For '{TENSOR_SETITEM}', the number of elements : {len(value)} in the updates tuple "
                         f"does not meet the requirements: {updates_shape[-1]}.")
    array = np.array(value, dtype=mstype.dtype_to_nptype(data_dtype))
    reps = compute_multiples(updates_shape[-1:], updates_shape)
    return Tensor(np.tile(array, reps))


@constexpr
def generate_updates_shape(data_shape, index_shape, op_type):
    """Generate updates shape for 'tensor setitem'."""
    if op_type == SET_ITEM_BY_ONE_TENSOR:
        updates_shape = index_shape + data_shape[1:]
    else:
        updates_shape = index_shape[:-1] + data_shape[index_shape[-1]:]
    return updates_shape


@constexpr
def check_number_of_index_tensor(data_shape, tuple_len, op_name):
    """Check if the number of index tensor exceeds the dimension of the operated tensor."""
    if tuple_len <= len(data_shape):
        return True
    raise IndexError(f"For '{op_name}', the number {tuple_len} of index tensor "
                     f"is greater than the dimension  {len(data_shape)} of the operated tensor.")


def generate_indeices_from_tuple_of_tensor(data, tuple_index, op_name):
    """Generate an indices tensor from a tuple of tensor."""
    indices = None
    check_index_tensor_number = check_number_of_index_tensor(F.shape(data), len(tuple_index), op_name)
    if check_index_tensor_number:
        dtype_tuple = hyper_map(F.dtype, tuple_index)
        check_dtypes = check_index_tensors_dtype(dtype_tuple, op_name)
        if check_dtypes:
            shape_tuple = hyper_map(F.shape, tuple_index)
            broadcast_shape = broadcast_shapes(shape_tuple, op_name)
            broadcast_tensors = hyper_map(F.partial(tile, broadcast_shape), tuple_index)
            indices = pack(broadcast_tensors)
    return indices


def generate_updates_from_scalar(data, indices, value, op_type):
    """Generate an updates tensor from a scalar."""
    data_shape = F.shape(data)
    indices_shape = F.shape(indices)
    data_dtype = F.dtype(data)
    return convert_scalar_to_tensor(data_shape, data_dtype, indices_shape, value, op_type)


def generate_updates_from_tuple(data, index, value, op_type):
    """Generate an updates tensor from a tuple."""
    value_types = hyper_map(F.typeof, value)
    data_dtype = F.dtype(data)
    value_elements_type = check_value_elements(data_dtype, value_types)
    if value_elements_type == ALL_TENSOR:
        value_shapes = hyper_map(F.shape, value)
        shapes_same = check_shapes_same(value_shapes, TENSOR_SETITEM)
        if shapes_same:
            value = F.pack(value)
        return generate_updates_from_tensor(data, index, value, op_type)

    data_shape = F.shape(data)
    index_shape = F.shape(index)
    return convert_tuple_of_scalar_to_tensor(data_shape, data_dtype, index_shape, value, op_type)


def generate_updates_from_tensor(data, index, value, op_type):
    """Generate an updates tensor from a tensor."""
    data_shape = F.shape(data)
    index_shape = F.shape(index)
    value_shape = F.shape(value)
    data_dtype = F.dtype(data)
    value_dtype = F.dtype(value)
    updates_shape = value_shape
    check_dtype_same = check_tensors_dtype_same(data_dtype, value_dtype, TENSOR_SETITEM)
    if check_dtype_same:
        updates_shape = generate_updates_shape(data_shape, index_shape, op_type)
    need_broadcast = check_two_shapes_need_broadcast(updates_shape, value_shape)
    if need_broadcast:
        return tile(updates_shape, value)
    return value
