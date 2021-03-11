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
from .... import log as logger
from ....common import dtype as mstype
from ....common.tensor import Tensor
from ....ops import _utils as op_utils

ALL_TENSOR = 0
NO_TENSOR = 1
CONTAIN_TENSOR = 2
ALL_SCALAR = 3
ALL_INT = 4
NO_INT = 5
CONTAIN_INT = 6
ALL_BASIC = 7
MIXED = 8

INT_ = 0
BOOL_ = 1
UNSUPPORTED_DTYPE = 2

TENSOR_SETITEM = "tensor setitem"
TENSOR_GETITEM = "tensor getitem"

SET_ITEM_BY_ONE_TENSOR = 0
SET_ITEM_BY_TUPLE_OF_TENSOR = 1


@constexpr
def raise_value_error(msg):
    raise ValueError(msg)


@constexpr
def raise_index_error(msg):
    raise IndexError(msg)


@constexpr
def raise_type_error(msg):
    raise TypeError(msg)


@constexpr
def check_equal(param1, param2, msg="{},{}"):
    """Checks whether the two parameters are equal or not."""
    if param1 != param2:
        raise ValueError(msg.format(param1, param2))
    return param1


@constexpr
def make_empty_slice():
    return slice(None, None, None)


@constexpr
def make_tensor(data, data_type=mstype.int64, data_shape=None):
    if data_shape:
        return Tensor(np.zeros(data_shape), data_type)
    return Tensor(data, data_type)


@constexpr
def judge_data_rank(data_rank, min_data_rank=0, max_data_rank=8):
    if data_rank < min_data_rank or data_rank > max_data_rank:
        raise ValueError(f"The input data's rank should in the range of[{min_data_rank}, "
                         f"{max_data_rank}], bug actually is '{data_rank}'")


@constexpr
def check_ellipsis_shape_size(data_shape, value_shape, data_size, value_size):
    """Checks the shape and size of the sensor and value."""
    if data_shape == value_shape or data_size == value_size or value_size == 1:
        return True
    raise ValueError("The value(shape={}), can not assign to tensor(shape={}).".format(
        value_shape, data_shape))


@constexpr
def check_tensor_setitem_index(index, element_type=None):
    """Checks tuple index type of tensor assignment."""
    if index is None:
        raise IndexError("Tensor's index cannot be None.")
    if isinstance(index, slice):
        return True
    if isinstance(index, tuple):
        if not index:
            raise IndexError("Tensor's index cannot be empty.")
        for item in index:
            if not isinstance(item, (slice, type(...), int)):
                raise IndexError(
                    "Index of type '{}' is not supported yet.".format(type(item)))
        return True
    if isinstance(index, mstype.tensor_type):
        if element_type is None or element_type != mstype.bool_:
            raise TypeError(
                "The index of tensor should be a bool type tensor. "
                "{} type is not supported yet.".format(element_type))
        return True

    raise IndexError(
        "Index of type '{}' is not supported yet.".format(type(index)))


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


@constexpr
def check_valid_dim(dim, name):
    if dim not in (1, 2):
        raise ValueError(
            f"For {name}, inputs dim must be 1d or 2d")


@constexpr
def judge_index_type(index_type, target_type):
    if index_type == target_type or (isinstance(target_type, (list, tuple)) and index_type in target_type):
        return True
    return False


@constexpr
def judge_indexes_types(dtypes, target_type):
    """Check a tuple of tensor data type."""
    for dtype in dtypes:
        if dtype != target_type and (isinstance(target_type, (list, tuple)) and dtype not in target_type):
            return False
    return True


@constexpr
def check_type_valid(dtype, target_type, op_name):
    if dtype != target_type and (isinstance(target_type, (list, tuple)) and dtype not in target_type):
        if op_name in (TENSOR_GETITEM, TENSOR_SETITEM):
            raise IndexError(
                f"The '{op_name}' doesn't supoort {dtype}' and expect to receive {target_type}.")
        raise TypeError(
            f"The '{op_name}' doesn't supoort {dtype}' and expect to receive {target_type}.")


@constexpr
def check_types_valid(dtypes, target_type, op_name):
    """Check a tuple of tensor data type."""
    for dtype in dtypes:
        check_type_valid(dtype, target_type, op_name)


@constexpr
def get_pos_of_indexes_types(indexes_types, op_name):
    """Separate the position information of tensor and slice and ellipsis from the mixed tensors index."""
    slice_positions, ellipsis_positions, none_positions, int_positions, bool_positions, tensor_positions, \
        sequence_positions = (), (), (), (), (), (), ()
    for i, index_type in enumerate(indexes_types):
        if isinstance(index_type, mstype.Slice):
            slice_positions += (i,)
        elif isinstance(index_type, mstype.Ellipsis_):
            ellipsis_positions += (i,)
        elif isinstance(index_type, mstype.none_type):
            none_positions += (i,)
        elif isinstance(index_type, mstype.Int):
            int_positions += (i,)
        elif isinstance(index_type, mstype.Bool):
            bool_positions += (i,)
        elif isinstance(index_type, mstype.tensor_type):
            tensor_positions += (i,)
        elif isinstance(index_type, (list, tuple)):
            sequence_positions += (i,)
        else:
            raise IndexError(f"For '{op_name}', the index elements only support 'Slice', 'Ellipsis', 'None', "
                             f"'Tensor', 'int',  'List', 'Tuple', 'bool' but got {index_type}.")
    if len(ellipsis_positions) > 1:
        raise IndexError(
            f"For '{op_name}, an index can only have a single ellipsis('...')")

    return slice_positions, ellipsis_positions, none_positions, int_positions, bool_positions, \
        tensor_positions, sequence_positions


def slice_expand(input_slices, shape):
    """
    Converts slice to indices.

    Inputs:
        slices (Union[Slice, tuple[Slice]]): Slice tuple or slice.
        shape (tuple): The shape of a sensor is an integer element tuple.

    Outputs:
        tuple[list], This is expressed as (begins, ends, strides).
    """
    begin, end, strides = [], [], []
    index = 0
    slices = None
    # Slice or tuple(Slice...)
    if isinstance(input_slices, slice):
        slices = (input_slices,)
    elif isinstance(input_slices, (tuple, list)) and input_slices and isinstance(input_slices[0], (slice, type(...))):
        is_have_ellipsis = False
        for _, element in enumerate(input_slices):
            if isinstance(element, type(...)):
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
        stop = shape[index] if (s.stop is None) else s.stop
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
    if isinstance(input_, type(...)):
        input_slice = (input_,)
    ell_count = 0
    for _, element in enumerate(input_slice):
        if not isinstance(element, type(...)):
            result.append(element)
            continue
        ell_count += 1
        if ell_count > 1:
            raise IndexError("There cannot be more than one ellisis (...) in the index of the tensor, "
                             "but it is currently {}".format(input_slice))
        for _ in range(len(shape) - len(input_slice) + 1):
            result.append(slice(None, None, None))
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
        raise IndexError(
            "The tensor's index is unreasonable. index:{}".format(index))
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
def tuple_element_is_int(indexes):
    """Judges tuple element type."""
    if not indexes:
        raise IndexError("Tensor's index cannot be empty.")
    if isinstance(indexes, tuple):
        for _, ele in enumerate(indexes):
            if not isinstance(ele, int):
                return False
        return True
    return False


@constexpr
def tuple_index_int_cnt(types, op_name):
    """count the int type of types which contains the tuple elements' type."""
    int_cnt = sum(isinstance(ele, mstype.Int) for ele in types)
    return ALL_INT if int_cnt == len(types) else NO_INT if int_cnt == 0 else CONTAIN_INT


@constexpr
def tuple_index_type_cnt(types, op_name):
    """count the tensor type of types which contains the tuple elements' type."""
    tensor_cnt = sum(isinstance(ele, mstype.tensor_type) for ele in types)
    basic_cnt = sum(isinstance(
        ele, (mstype.Int, mstype.Ellipsis_, mstype.Slice)) for ele in types)
    if tensor_cnt == len(types):
        return ALL_TENSOR
    if basic_cnt == len(types):
        return ALL_BASIC
    return MIXED


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
                                f"in value tuple is not consistent with assigned tensor data type '{data_dtype}'.")
        elif mstype.dtype_to_pytype(ele) == mstype.dtype_to_pytype(data_dtype):
            scalars_number += 1
        else:
            raise TypeError(f"For '{TENSOR_SETITEM}', the {i}th element type '{ele}' in "
                            f"value tuple is not consistent with assigned tensor data type '{data_dtype}'.")
    if tensors_number == len(types):
        return ALL_TENSOR
    if scalars_number == len(types):
        return ALL_SCALAR
    raise TypeError(
        f"For '{TENSOR_SETITEM}', the value does not support scalar and tensor mixing, but got {types}.")


@constexpr
def get_index_tensor_dtype(dtype):
    """Check a tuple of tensor data type."""
    if dtype in mstype.int_type:
        return INT_
    if dtype == mstype.bool_:
        return BOOL_
    raise IndexError(
        f"For '{TENSOR_SETITEM}', the index tensor data type '{dtype}' is not supported.")


@constexpr
def check_tensors_dtype_same(data_dtype, value_dtype, op_name):
    """Check tensors data type same."""
    if value_dtype == data_dtype:
        return True
    raise TypeError(f"For '{op_name}', the value data type '{value_dtype}' "
                    f"is not consistent with assigned tensor data type {data_dtype}.")


@constexpr
def generate_broadcast_shape(shapes, op_name):
    """Generate broadcast shape for a tuple of shape."""
    if not shapes:
        return ()
    broadcast_shape = shapes[0]
    for i, shape in enumerate(shapes):
        logger.debug(f"Broadcasts the {i}th tensor, the shape is {shape}.")
        try:
            broadcast_shape = op_utils.get_broadcast_shape(
                broadcast_shape, shape, op_name)
        except ValueError as ex:
            raise IndexError(ex)
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
    """Compute multiples between origin shape with broadcast shape."""
    len_gap = len(broadcast_shape) - len(origin_shape)
    return broadcast_shape[0:len_gap] + tuple(map(lambda x, y: x // y, broadcast_shape[len_gap:], origin_shape))


@constexpr
def compute_new_shape(origin_shape, indexes_shapes_info):
    """Compute new shape between origin shape with final shape."""
    new_shape = []
    for i in indexes_shapes_info:
        if i == origin_shape:
            new_shape.extend(origin_shape)
        else:
            new_shape.append(1)
    return tuple(new_shape)


@constexpr
def convert_int_to_slice(tuple_index):
    tuple_index_new = tuple(slice(i, i+1, 1) for i in tuple_index)
    return tuple_index_new


@constexpr
def check_and_transform_int_index(index, shape, op_name):
    if index < -shape or index >= shape:
        raise IndexError(f"In the \"{op_name}\", the index should in the range [-{shape}, {shape-1}] to fit "
                         f"the corresponding dim length, but get {index}.")
    if index < 0:
        index += shape
    return index


@constexpr
def transform_sequence_index(sequence_index, shape, op_name):
    """transform list or tuple with integer and boolean to tuple with integer index"""
    bool_count = len(list(filter(lambda index: isinstance(index, bool), sequence_index)))
    int_count = len(list(filter(lambda index: isinstance(index, int), sequence_index)))-bool_count
    if int_count == 0 and bool_count != 0:
        if bool_count == shape:
            list_index = list(filter(lambda i: sequence_index[i], range(bool_count)))
        else:
            raise IndexError("The boolean array should have the same length with the corresponding dimension")
    else:
        list_index = [int(index) for index in sequence_index]

    for i, index in enumerate(list_index):
        list_index[i] = check_and_transform_int_index(index, shape, op_name)
    sub_tuple_index = tuple(list_index)
    return sub_tuple_index


@constexpr
def convert_slice_to_tensor(index, final_shape, slice_cnt, broadcast_shape, slice_shapes, fancy_position):
    """Convert a slice to a tensor."""

    shape = [1] * len(slice_shapes)
    shape[slice_cnt] = slice_shapes[slice_cnt]
    shape = shape[:fancy_position] + [1] * len(broadcast_shape) + shape[fancy_position:]

    array = np.array(index, np.int64)
    array = np.reshape(array, shape)
    reps = compute_multiples(shape, final_shape)
    slice_index_tensor = Tensor(np.tile(array, reps), mstype.int64)
    return slice_index_tensor


@constexpr
def check_shapes_same(value_shapes, op_name):
    """Check if the shapes in the tuple are consistent."""
    for i, shape in enumerate(value_shapes):
        if shape != value_shapes[0]:
            raise ValueError(f"For '{op_name}', the {i}th tensor shape in "
                             f"value tuple is not same as the first tensor shape.")
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
                    f" is not consistent with the assigned tensor data type {data_dtype}.")


@constexpr
def convert_tuple_of_scalar_to_tensor(data_shape, data_dtype, index_shape, value, op_type):
    """Convert a tuple of scalar to a tensor."""
    updates_shape = generate_updates_shape(data_shape, index_shape, op_type)
    if len(value) != updates_shape[-1]:
        raise ValueError(f"For '{TENSOR_SETITEM}', the number of elements : {len(value)} "
                         f"in the updates tuple does not meet the requirements: {updates_shape[-1]}.")
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
def transform_slice_to_ele_list(slice_index, dim_len):
    slice_obj = slice(slice_index.start, slice_index.stop, slice_index.step)
    slice_ele_list = list(range(dim_len))[slice_obj]
    if not slice_ele_list:
        raise IndexError(f"An empty slice is not supported, got {slice_obj}")
    return slice_ele_list


@constexpr
def check_tuple_index_len(data_rank, tuple_index_len, op_name):
    """Check if the number of index tensor exceeds the dimension of the operated tensor."""
    if tuple_index_len <= data_rank:
        return True
    raise IndexError(f"For '{op_name}', the number {tuple_index_len} of tuple_index size"
                     f"is greater than the dimension  {data_rank} of the operated tensor.")


@constexpr
def generate_index_info_from_tuple_of_mixed_tensors(tensor_positions, tensor_indexes_shapes,
                                                    slice_shapes, op_name):
    """
    Generate index info which contain broadcast shape, final shape,
    indexes shapes info, ellipsis size from a tuple of mixed tensors.
    """
    tensor_positions = tuple(sorted(tensor_positions))
    tensor_index_continue_tag = _judge_order_continuous(tensor_positions)
    fancy_position = tensor_positions[0] if tensor_index_continue_tag else 0
    broadcast_shape = generate_broadcast_shape(tensor_indexes_shapes, op_name)
    index_tensor_new_shape, final_shape = [], []

    if tensor_index_continue_tag:
        final_shape = slice_shapes[:fancy_position] + broadcast_shape + slice_shapes[fancy_position:]
        index_tensor_new_shape = (1,) * len(slice_shapes[:fancy_position]) + \
            broadcast_shape + (1,) * len(slice_shapes[fancy_position:])

    else:
        final_shape = broadcast_shape + slice_shapes
        index_tensor_new_shape = broadcast_shape + (1,) * len(slice_shapes)

    return broadcast_shape, index_tensor_new_shape, final_shape, fancy_position

def _judge_order_continuous(order_sequence):
    if not order_sequence:
        return False
    for idx1, idx2 in zip(order_sequence[:-1], order_sequence[1:]):
        if idx1 + 1 != idx2:
            return False
    return True


@constexpr
def scalar_in_sequence(x, y):
    """Determine whether the scalar in the sequence."""
    if x is None:
        raise ValueError("Judge scalar in tuple or list require scalar and sequence should be constant, "
                         "but the scalar is not.")
    if y is None:
        raise ValueError("Judge scalar in tuple or list require scalar and sequence should be constant, "
                         "but the sequence is not.")
    if x in y:
        return True
    return False


@constexpr
def get_np_eps(input_dtype):
    nptype = mstype.dtype_to_nptype(input_dtype)
    eps = np.finfo(nptype).eps
    return float(eps)


@constexpr
def check_number_index_type(number):
    """Check if it is int or bool number"""
    if isinstance(number, bool):
        return BOOL_
    if isinstance(number, int):
        return INT_
    raise IndexError("Only support integers, slices(`:`), ellipsis(`...`), None and bool, got {0} type is {1} "
                     .format(number, type(number)))


@constexpr
def get_stride_info_from_integer(data_shape, number):
    """Get stride info from a integer"""
    begin_strides = [number]
    end_strides = [number + 1]
    step_strides = [1]
    for end in data_shape[1:]:
        begin_strides.append(0)
        end_strides.append(end)
        step_strides.append(1)
    return tuple(begin_strides), tuple(end_strides), tuple(step_strides)


def get_slice_stride(dim_size, index_slice):
    """Get slice stride info"""
    step = 1 if index_slice.step is None else index_slice.step
    start_default = 0
    stop_default = dim_size
    if step < 0:
        start_default = -1
        stop_default = -(dim_size + 1)
    start = start_default if index_slice.start is None else index_slice.start
    stop = stop_default if index_slice.stop is None else index_slice.stop
    return start, stop, step


@constexpr
def get_stride_info_from_tuple(data_shape, tuple_index):
    """Get stride info from a tuple"""
    begin_strides, end_strides, step_strides = [], [], []
    tuple_index_len = len(tuple_index)
    data_rank = len(data_shape)
    shrink_axis, index_count, ellipsis_count = 0, 0, 0
    for idx, item in enumerate(tuple_index):
        if isinstance(item, slice):
            start, stop, step = get_slice_stride(data_shape[idx], item)
            begin_strides.append(start)
            end_strides.append(stop)
            step_strides.append(step)
            index_count = index_count + 1
        elif isinstance(item, int):
            begin_strides.append(item)
            end_strides.append(item + 1)
            step_strides.append(1)
            shrink_axis = shrink_axis + (1 << index_count)
            index_count = index_count + 1
        elif item is ...:
            ellipsis_count = ellipsis_count + 1
            if ellipsis_count > 1:
                raise IndexError("An index can have only one ellipsis (...)")
            ellipsis_range_size = data_rank - tuple_index_len + 1
            begin_strides.extend([0] * (ellipsis_range_size))
            end_strides.extend(
                [shape for shape in data_shape[index_count: index_count + ellipsis_range_size]])
            step_strides.extend([1] * (ellipsis_range_size))
            index_count = index_count + ellipsis_range_size
        else:
            raise IndexError("Not supported index data type, got ",
                             item, " type is ", type(item))
    for item in range(index_count, data_rank):
        begin_strides.append(0)
        end_strides.append(data_shape[item])
        step_strides.append(1)
    return tuple(begin_strides), tuple(end_strides), tuple(step_strides), shrink_axis


@constexpr
def mstype_eq(x, y):
    if x == y:
        return True
    return False


@constexpr
def scalar_to_tensor(x):
    """Convert a scalar to a tensor"""
    return Tensor(x)
