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

from itertools import compress, zip_longest
from functools import partial
from collections import deque
import operator

import numpy as np

from mindspore.ops.primitive import constexpr
from mindspore import log as logger
from mindspore.common import dtype as mstype
from mindspore.common.tensor import Tensor
from mindspore.common._register_for_tensor import tensor_operator_registry
from mindspore.ops import _utils as op_utils
from mindspore._checkparam import Validator as validator
from mindspore.ops import operations as P
from mindspore import context

ALL_TENSOR = 0
NO_TENSOR = 1
CONTAIN_TENSOR = 2
ALL_SCALAR = 3
ALL_BASIC = 7
MIXED = 8

INT_ = 0
BOOL_ = 1
UNSUPPORTED_DTYPE = 2

TENSOR_SETITEM = "tensor setitem"
TENSOR_GETITEM = "tensor getitem"

SET_ITEM_BY_ONE_TENSOR = 0
SET_ITEM_BY_TUPLE_OF_TENSOR = 1
SET_ITEM_BY_NON_TENSOR = 2


@constexpr
def raise_value_error(msg):
    """Constexpr for raise_value_error."""
    raise ValueError(msg)


@constexpr
def raise_index_error(msg):
    """Constexpr for raise_index_error."""
    raise IndexError(msg)


@constexpr
def raise_type_error(msg):
    """Constexpr for raise_type_error."""
    raise TypeError(msg)


@constexpr
def raise_unimplemented_error(msg):
    raise NotImplementedError(msg)


@constexpr
def log_warning(msg):
    """Print warning logs"""
    logger.warning(msg)


@constexpr
def check_equal(param1, param2, msg="{},{}"):
    """Checks whether the two parameters are equal or not."""
    if param1 != param2:
        raise ValueError(msg.format(param1, param2))
    return param1


@constexpr
def make_empty_slice():
    """Creates a empty slice."""
    return slice(None, None, None)


@constexpr
def _deep_list(array_like, dim_size=-1):
    """convert nested tuple/list mixtures to pure nested list"""
    if dim_size != -1:
        array_like = check_range(array_like, dim_size)
    if isinstance(array_like, (list, tuple)):
        return list(map(lambda x: _deep_list(x, dim_size), array_like))
    return array_like


@constexpr
def deep_tuple(array_like):
    """convert nested tuple/list mixtures to pure nested tuple"""
    if isinstance(array_like, (list, tuple)):
        return tuple(map(deep_tuple, array_like))
    return array_like


def _deep_tensor_to_nparray(array_like):
    """
    convert a nested list of tensor to nested list of np_array.

    Args:
        array_like(list(tensor)): In any format of nested lists that may contain
        tensors.

    Returns:
        array_like(list(np_array)): Formatted array that can be directly processed
            by numpy.array(), with all tensor elements converted to numpy_array.
    """
    # Recursively check whether each element is a tensor or not, if is tensor,
    # convert it to a numpy array in place
    if isinstance(array_like, Tensor):
        return array_like.asnumpy()

    if isinstance(array_like, list):
        for idx, value in enumerate(array_like):
            array_like[idx] = _deep_tensor_to_nparray(value)

    return array_like


@constexpr
def check_range(x, dim_size):
    if dim_size == -1:
        return x
    if isinstance(x, int) and not isinstance(x, bool):
        if x >= dim_size or x < -dim_size:
            raise IndexError(f'index {x} is out of bounds for dimension with size {dim_size}')
        x = x % dim_size
    return x


@constexpr
def make_tensor(a, dtype=mstype.int64, data_shape=None, dim_size=-1):
    """
    Converts the input to tensor.

    This function converts tensors from an array-like object.

    Args:
        a (Union[int, float, bool, list, tuple]): Input data, in any form that can
            be converted to a `Tensor`.
        dtype (:class:`mindspore.dtype`): Designated tensor dtype.

    Returns:
        Tensor, generated tensor with the specified dtype.

    Raises:
        TypeError: If input arguments have types not specified above.
        ValueError: If input `a` has different sizes at different dimensions.
    """
    if data_shape:
        return Tensor(np.zeros(data_shape), dtype)

    if not isinstance(a, (list, tuple, int, float, bool)):
        raise TypeError("input data must be `int`, `float`, `bool`, `list` or `tuple`")

    if dim_size != -1:
        a = check_range(a, dim_size)

    if isinstance(a, (list, tuple)):
        # Convert all tuple/nested tuples to lists
        a = _deep_list(a, dim_size)
        # Convert all tensor sub-elements to numpy arrays
        a = _deep_tensor_to_nparray(a)
        a = np.asarray(a)
        if a.dtype is np.dtype('object'):
            raise ValueError('Input array must have the same size across all dimensions.')

    if isinstance(a, np.ndarray):
        if a.dtype is np.dtype('object'):
            raise TypeError(f"For Tensor conversion, the input_data is {a} that contains unsupported element.")

    return Tensor(a, dtype)

tensor_operator_registry.register('make_tensor', make_tensor)


@constexpr
def judge_data_dim(data_dim, min_data_dim=0, max_data_dim=8):
    """Judges whether the data dim is valid."""
    if data_dim < min_data_dim or data_dim > max_data_dim:
        raise ValueError(f"The input data's dim must in the range of [{min_data_dim}, "
                         f"{max_data_dim}], but got '{data_dim}'.")


@constexpr
def get_source_shape(data_shape, value_shape):
    """Returns the shape of value that will be used to broadcast against data."""
    cannot_broadcast = False
    source_shape = value_shape
    for i, j in zip(reversed(data_shape), reversed(value_shape)):
        if j not in (1, i):
            cannot_broadcast = True
    for i in range(len(value_shape) - len(data_shape)):
        source_shape = data_shape
        if value_shape[i] != 1:
            cannot_broadcast = True
    if cannot_broadcast:
        raise ValueError(f'could not broadcast input array from shape {value_shape} to {data_shape}')
    return source_shape


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
    """Checks whether the dim is valid."""
    if dim not in (1, 2):
        raise ValueError(f"For '{name}', the dimension of inputs must be 1d or 2d, but got {dim}.")


@constexpr
def judge_index_type(index_type, target_type):
    """Judges whether the index type is valid."""
    if index_type == target_type or (isinstance(target_type, (list, tuple)) and index_type in target_type):
        return True
    return False


@constexpr
def judge_indexes_types(dtypes, target_type):
    """Check a tuple of tensor data type."""
    for dtype in dtypes:
        if isinstance(target_type, (list, tuple)):
            if dtype not in target_type:
                return False
        else:
            if dtype != target_type:
                return False
    return True


@constexpr
def check_type_isinstance(dtype, target_type):
    """Checks whether the dtype is instance of target type."""
    if isinstance(dtype, (list, tuple)):
        return all(isinstance(ele, target_type) for ele in dtype)
    return isinstance(dtype, target_type)


@constexpr
def check_type_invalid(dtype, target_type):
    """Checks whether the dtype is valid."""
    return dtype != target_type and (isinstance(target_type, (list, tuple)) and dtype not in target_type)


@constexpr
def check_type_valid(dtype, target_type, op_name):
    """Checks whether the dtype is valid."""
    if dtype != target_type and (isinstance(target_type, (list, tuple)) and dtype not in target_type):
        if op_name in (TENSOR_GETITEM, TENSOR_SETITEM):
            raise IndexError(
                f"The '{op_name}' doesn't support '{dtype}' and expect to receive {target_type}.")
        raise TypeError(
            f"The '{op_name}' doesn't support '{dtype}' and expect to receive {target_type}.")


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
            raise TypeError(f"For '{op_name}', the types only support 'Slice', 'Ellipsis', 'None', 'Tensor', 'int', "
                            f"'List', 'Tuple', 'bool', but got {index_type}.")
    if len(ellipsis_positions) > 1:
        raise IndexError(
            f"For '{op_name}, an index can only have a single ellipsis('...')")

    return slice_positions, ellipsis_positions, none_positions, int_positions, bool_positions, \
        tensor_positions, sequence_positions


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
def slice2indices(input_slice, shape):
    """
    Converts slice to indices.

    Inputs:
        input_slice (Union[Slice, tuple[Slice]]): Slice tuple or slice.
        shape (tuple): The shape of a tensor is an integer element tuple.

    Outputs:
        Tensor, the shape is (n, 1).
    """
    start, stop, step = normalize_slice(input_slice, shape[0])
    if check_slice_empty(start, stop, step):
        return False
    grids = ([np.array(list(range(start, stop, step)), dtype=np.int64)] +
             [np.array(list(range(dim_size)), dtype=np.int64) for dim_size in shape[1:]])
    mesh = np.ix_(*grids)
    return Tensor(np.stack(np.broadcast_arrays(*mesh), axis=-1))


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
def tuple_index_type_cnt(types, op_name):
    """count the tensor type of types which contains the tuple elements' type."""
    if all(isinstance(ele, mstype.tensor_type) for ele in types):
        return ALL_TENSOR
    if all(isinstance(ele, (mstype.Int, mstype.Ellipsis_, mstype.Slice)) for ele in types):
        return ALL_BASIC
    return MIXED


@constexpr
def check_value_elements(types):
    """Judges the type of all elements of the tuple."""
    tensor_number = 0
    for ele in types:
        if isinstance(ele, mstype.tensor_type):
            tensor_number += 1
    if tensor_number == 0:
        return NO_TENSOR
    if tensor_number == len(types):
        return ALL_TENSOR
    return CONTAIN_TENSOR


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
    """Check shape_y needs to be broadcast to shape_x."""
    if any(j not in (i, 1) for i, j in zip(reversed(shape_x), reversed(shape_y))):
        raise ValueError(f"{shape_y} could not broadcast with {shape_x}.")
    return shape_y != shape_x


@constexpr
def compute_multiples(origin_shape, broadcast_shape):
    """Compute multiples between origin shape with broadcast shape."""
    len_gap = len(broadcast_shape) - len(origin_shape)
    return broadcast_shape[0:len_gap] + tuple(map(lambda x, y: x // y, broadcast_shape[len_gap:], origin_shape))


@constexpr
def convert_scalar_to_tensor(data_shape, data_dtype, indices_shape, value, op_type):
    """Convert a scalar to a tensor."""
    if op_type == SET_ITEM_BY_ONE_TENSOR:
        updates_shape = indices_shape + data_shape[1:]
    else:
        updates_shape = indices_shape[:-1] + data_shape[indices_shape[-1]:]
    return Tensor(np.full(updates_shape, value), dtype=data_dtype)


def generate_updates_shape(data_shape, index_shape, op_type, is_dynamic):
    """Generate updates shape for 'tensor setitem'."""
    if op_type == SET_ITEM_BY_ONE_TENSOR:
        if is_dynamic:
            updates_shape = P.Concat(-1)((index_shape, data_shape[1:]))
        else:
            updates_shape = index_shape + data_shape[1:]
    else:
        if is_dynamic:
            updates_shape = P.Concat(-1)((index_shape[:-1], data_shape[index_shape[-1]:]))
        else:
            updates_shape = index_shape[:-1] + data_shape[index_shape[-1]:]
    return updates_shape


@constexpr
def transform_slice_to_ele_list(slice_index, dim_len):
    """Transforms slice to element list."""
    slice_obj = slice(slice_index.start, slice_index.stop, slice_index.step)
    start, stop, end = normalize_slice(slice_obj, dim_len)
    slice_ele_list = list(range(start, stop, end))
    if not slice_ele_list:
        raise IndexError(f"An empty slice is not supported, got {slice_obj}")
    return slice_ele_list


@constexpr
def generate_index_info_from_tuple_of_mixed_tensors(tensor_positions, tensor_indexes_shapes,
                                                    slice_shapes, op_name, fancy_position=None):
    """
    Generate index info which contain broadcast shape, final shape,
    indexes shapes info, ellipsis size from a tuple of mixed tensors.
    """
    tensor_positions = tuple(sorted(tensor_positions))
    if fancy_position is None:
        tensor_index_continue_tag = _judge_order_continuous(tensor_positions)
        fancy_position = tensor_positions[0] if tensor_index_continue_tag else 0
    broadcast_shape = generate_broadcast_shape(tensor_indexes_shapes, op_name)

    final_shape = slice_shapes[:fancy_position] + broadcast_shape + slice_shapes[fancy_position:]
    index_tensor_new_shape = (1,) * len(slice_shapes[:fancy_position]) + \
        broadcast_shape + (1,) * len(slice_shapes[fancy_position:])

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
        raise ValueError("Judge scalar in tuple or list require scalar and sequence must be constant, "
                         "but the scalar is not.")
    if y is None:
        raise ValueError("Judge scalar in tuple or list require scalar and sequence must be constant, "
                         "but the sequence is not.")
    return x in y


@constexpr
def get_np_eps(input_dtype):
    """Get numpy eps."""
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
def get_stride_info_from_slice(data_shape, slice_index):
    """Get stride info from a python slice"""
    begin, end, step = get_slice_stride(slice_index, data_shape[0])
    begin_strides = [begin]
    end_strides = [end]
    step_strides = [step]
    for end in data_shape[1:]:
        begin_strides.append(0)
        end_strides.append(end)
        step_strides.append(1)
    return tuple(begin_strides), tuple(end_strides), tuple(step_strides)


@constexpr
def get_stride_info_from_integer(data_shape, number):
    """Get stride info from an integer"""
    begin_strides = [number]
    end_strides = [number + 1]
    step_strides = [1]
    for end in data_shape[1:]:
        begin_strides.append(0)
        end_strides.append(end)
        step_strides.append(1)
    return tuple(begin_strides), tuple(end_strides), tuple(step_strides)


@constexpr
def get_slice_stride(index_slice, dim_size):
    """Get slice stride info"""
    step = 1 if index_slice.step is None else index_slice.step
    if step < 0:
        start_default = -1
        stop_default = -(dim_size + 1)
        stop = stop_default if index_slice.stop is None else max(stop_default, index_slice.stop)
    else:
        start_default = 0
        stop_default = dim_size
        stop = stop_default if index_slice.stop is None else min(stop_default, index_slice.stop)
    start = start_default if index_slice.start is None else index_slice.start
    return start, stop, step


@constexpr
def get_stride_info_from_tuple(data_shape, tuple_index):
    """Get stride info from a tuple"""
    begin_strides, end_strides, step_strides = [], [], []
    tuple_index_len = len(tuple_index)
    data_dim = len(data_shape)
    shrink_axis, index_count, ellipsis_count = 0, 0, 0
    for index, dim_size in zip(tuple_index, data_shape):
        if isinstance(index, slice):
            start, stop, step = get_slice_stride(index, dim_size)
            begin_strides.append(start)
            end_strides.append(stop)
            step_strides.append(step)
            index_count = index_count + 1
        elif isinstance(index, int):
            begin_strides.append(index)
            end_strides.append(index + 1)
            step_strides.append(1)
            shrink_axis = shrink_axis + (1 << index_count)
            index_count = index_count + 1
        elif index is ...:
            ellipsis_count = ellipsis_count + 1
            if ellipsis_count > 1:
                raise IndexError("An index can have only one ellipsis (...)")
            ellipsis_range_size = data_dim - tuple_index_len + 1
            begin_strides.extend([0] * ellipsis_range_size)
            end_strides.extend(
                [shape for shape in data_shape[index_count: index_count + ellipsis_range_size]])
            step_strides.extend([1] * ellipsis_range_size)
            index_count = index_count + ellipsis_range_size
        else:
            raise IndexError("Not supported index data type, got ",
                             index, " type is ", type(index))
    for index in range(index_count, data_dim):
        begin_strides.append(0)
        end_strides.append(data_shape[index])
        step_strides.append(1)
    strides_v = {
        'begin': tuple(begin_strides),
        'end': tuple(end_strides),
        'step': tuple(step_strides)
    }
    return strides_v, shrink_axis


@constexpr
def scalar_to_tensor(x):
    """Convert a scalar to a tensor"""
    return Tensor(x)


@constexpr
def unpack(x):
    if isinstance(x, (tuple, list)) and len(x) == 1:
        return unpack(x[0])
    return x


@constexpr
def normalize_start(start, dim_size):
    """
    Normalize `start` according to the number of dimensions (`dim_size`).
    If the number of dimensions is not given, return the original input directly.
    """
    if start is None:
        return 0
    if dim_size == -1:
        return start
    if start < 0:
        return 0 if start < -dim_size else start % dim_size
    return start if start < dim_size else dim_size


@constexpr
def normalize_stop(stop, dim_size):
    """
    Normalize `stop` according to the number of dimensions (`dim_size`).
    If the number of dimensions is not given, return the original input directly.
    """
    if stop is None and dim_size == -1:
        raise IndexError("Not Support stop is None when dim is dynamic")
    if stop is None:
        return dim_size
    if dim_size == -1:
        return stop
    if stop < 0:
        return 0 if stop < -dim_size else stop % dim_size
    return stop if stop < dim_size else dim_size


@constexpr
def get_step_from_slice(input_slice):
    """get step in a slice."""
    step = input_slice.step
    if step is None:
        step = 1
    return step


@constexpr
def normalize_slice(input_slice, dim_size):
    """Normalizes start, stop, step in a slice."""
    start = normalize_start(input_slice.start, dim_size)
    stop = normalize_stop(input_slice.stop, dim_size)
    step = input_slice.step
    if step is None:
        step = 1
    if step >= 0:
        start = normalize_start(input_slice.start, dim_size)
        stop = normalize_stop(input_slice.stop, dim_size)
    else:
        start = normalize_stop(input_slice.start, dim_size)
        stop = normalize_start(input_slice.stop, dim_size)
    return start, stop, step


@constexpr
def tuple_slice(tup, start, end):
    """get sliced tuple from start and end."""
    return tup[start:end]


@constexpr
def expanded_shape(shape, expand_size):
    return (1,)*expand_size + shape


@constexpr
def sequence_mul_int(seq, number):
    """
    Make a new list with native python syntax.

    Args:
        seq (Union[list, tuple]): Input sequence.
        y (int): Input number.

    Returns:
        New sequence, has the same type as `seq`.
    """
    if not isinstance(number, int):
        raise TypeError(f"can't multiply sequence by non-int of type {type(number)}")
    return seq * number


@constexpr
def check_in_sequence(x, y):
    """Determine whether the input `x` is in the sequence `y`."""
    return x in y


@constexpr
def is_slice(x):
    return isinstance(x, slice)


@constexpr
def filter_expanded_dims(shape, not_expanded_dim):
    diff = len(not_expanded_dim) - len(shape)
    if diff < 0:
        raise ValueError(f'unable to broadcast {shape}')
    return tuple(compress(shape, not_expanded_dim[diff:]))


@constexpr
def sequence_to_index(sequence, dim_size):
    """Transforms sequence to tensor index."""
    if not sequence:
        return False
    if all(isinstance(i, bool) for i in sequence):
        if dim_size == -1:
            raise IndexError("Not supported to take the subscript of dynamic shape tensor using Boolean type")
        seq_size = len(sequence)
        if seq_size != dim_size:
            raise IndexError(f'dimension is {dim_size} but corresponding boolean dimension is {seq_size}')
        sequence = tuple(compress(range(dim_size), sequence))
        if not sequence:
            return False
    return make_tensor(sequence, mstype.int64, None, dim_size)


@constexpr
def int_to_index(i, shape):
    """Converts integer to tensor indices."""
    dim_size = shape[0]
    if i < -dim_size or i >= dim_size:
        raise IndexError(f'index {i} is out of bounds for axis 0 with size {dim_size}')
    i = i % dim_size
    if len(shape) == 1:
        return Tensor([[i]])
    grids = [np.array(list(range(size)), dtype=np.int64) for size in shape[1:]]
    mesh = np.ix_(*grids)
    index = np.stack(np.broadcast_arrays(*mesh), -1)
    return Tensor(np.insert(index, 0, i, -1))


@constexpr
def rem_not_expanded_dims(idx_advanced, expand_true, tensor_index_ndim, rem_ndim, not_expanded_dim):
    """Adds remaining dimensions not indexed to not_expanded_dim"""
    if idx_advanced != -1:
        if expand_true:
            # tensor indices generate only one dimension with size 1
            tensor_dims = (False,)
        else:
            tensor_dims = (True,)*tensor_index_ndim
        not_expanded_dim = not_expanded_dim[:idx_advanced] + tensor_dims + not_expanded_dim[idx_advanced:]
    not_expanded_dim = not_expanded_dim + (True,)*rem_ndim

    count_leading_false = 0
    while count_leading_false < len(not_expanded_dim) and not not_expanded_dim[count_leading_false]:
        count_leading_false += 1
    idx_advanced = max(0, idx_advanced - count_leading_false)
    return not_expanded_dim, idx_advanced


@constexpr
def check_slice_empty(start, stop, step):
    return (start - stop)*step >= 0


@constexpr
def real_axes(ndim_orig, ndim_out, axes_orig):
    """Returns the real axes to be reduced after performing broadcast"""
    _diff = ndim_out - ndim_orig
    axes = tuple(range(_diff))
    axes_orig = map(partial(operator.add, _diff), axes_orig)
    return axes + tuple(axes_orig)


check_axis_valid_const = constexpr(validator.check_axis_valid)


@constexpr
def compute_slice_shape(slice_shape, broadcast_shape_len, slice_cnt, fancy_position):
    """Computes slice tensor shapes"""
    shape = [1] * len(slice_shape)
    shape[slice_cnt] = slice_shape[slice_cnt]
    shape = shape[:fancy_position] + [1] * broadcast_shape_len + shape[fancy_position:]
    return shape


@constexpr
def infer_out_shape(*shapes):
    """
    Returns shape of output after broadcasting. Raises ValueError if shapes cannot be broadcast.
    """
    shape_out = deque()
    reversed_shapes = map(reversed, shapes)
    for items in zip_longest(*reversed_shapes, fillvalue=1):
        max_size = 0 if 0 in items else max(items)
        if any(item not in (1, max_size) for item in items):
            raise ValueError(f'operands could not be broadcast together with shapes {*shapes,}')
        shape_out.appendleft(max_size)
    return tuple(shape_out)


@constexpr
def use_copy_slice(tuple_index):
    if tuple_index is not None and len(tuple_index) >= 2:
        return (isinstance(tuple_index[0], int) and
                isinstance(tuple_index[1], slice) and tuple_index[1].step in (1, None) and
                all(x == slice(None, None, None) for x in tuple_index[2:]))
    return False


@constexpr
def is_ascend():
    """
    Returns device is Ascend or not
    """
    return context.get_context('device_target') == "Ascend"


@constexpr
def gen_exception_msg(msg_format, *args):
    return msg_format.format(*args)
