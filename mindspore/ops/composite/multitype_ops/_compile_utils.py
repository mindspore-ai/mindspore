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
from . import _constexpr_utils as const_utils
from ... import functional as F
from ... import operations as P
from ...composite import base
from ....common.tensor import Tensor
from ....common import dtype as mstype
from ....common._register_for_tensor import tensor_operator_registry

hyper_map = base.HyperMap()
stack = P.Stack(axis=-1)


def _tensor_getitem(self, index):
    """Handle tensor getitem"""
    if isinstance(index, Tensor):
        return tensor_index_by_tensor(self, index)
    if isinstance(index, list):
        return tensor_index_by_list(self, index)
    if isinstance(index, tuple):
        return tensor_index_by_tuple(self, index)
    if isinstance(index, bool):
        return _tensor_index_by_bool(self, index)
    if isinstance(index, int):
        return _tensor_index_by_integer(self, index)
    if isinstance(index, slice):
        return tensor_index_by_slice(self, index)
    if index is None:
        return F.expand_dims(self, 0)
    if index is ...:
        return self
    raise IndexError(f"Only support integers, slices(`:`), ellipsis(`...`), None, bool, tensor with int, "
                     f"list and tuple ,but got {index} with type {type(index)}.")


def _tensor_setitem(self, index, value):
    """Handle tensor setitem"""
    if not isinstance(value, (int, float, bool, list, tuple, Tensor)):
        raise ValueError(f"only support numbers, Tensor, tuple, list as value,"
                         f"but got {value} with type {type(value)}.")
    if isinstance(index, list):
        index = format_list_indices(index, self.shape[0])
    if isinstance(index, Tensor):
        return tensor_setitem_by_tensor(self, index, value)
    if isinstance(index, tuple):
        return tensor_setitem_by_tuple(self, index, value)
    if isinstance(index, bool):
        return tensor_setitem_by_bool(self, index, value)
    if isinstance(index, int):
        return tensor_setitem_by_number(self, index, value)
    if isinstance(index, slice):
        return tensor_setitem_by_slice(self, index, value)
    if index in (None, ...):
        return tensor_setitem_by_ellipsis(self, index, value)

    raise IndexError("Tensor setitem index only support integers, slices(`:`), ellipsis(`...`), bool, tensor, \
        list and tuple, but got {index} with type{type(index)}")


tensor_operator_registry.register("__getitem__", _tensor_getitem)
tensor_operator_registry.register("__setitem__", _tensor_setitem)


def _tensor_add(self, other):
    if isinstance(other, (tuple, list)):
        other = sequence_to_tensor(other, F.dtype(self))
    return F.add(self, other)


def _tensor_sub(self, other):
    if isinstance(self, (tuple, list)):
        self = sequence_to_tensor(self, F.dtype(other))
    if isinstance(other, (tuple, list)):
        other = sequence_to_tensor(other, F.dtype(self))
    return F.sub(self, other)


def _tensor_mul(self, other):
    if isinstance(other, (tuple, list)):
        other = sequence_to_tensor(other, F.dtype(self))
    return F.mul(self, other)


def _tensor_div(self, other):
    if isinstance(self, (tuple, list)):
        self = sequence_to_tensor(self, F.dtype(other))
    if isinstance(other, (tuple, list)):
        other = sequence_to_tensor(other, F.dtype(self))
    return F.div(self, other)


def _tensor_mod(self, other):
    if isinstance(self, (tuple, list)):
        self = sequence_to_tensor(self, F.dtype(other))
    if isinstance(other, (tuple, list)):
        other = sequence_to_tensor(other, F.dtype(self))
    return F.floormod(self, other)


def _tensor_pow(self, other):
    if isinstance(self, (tuple, list)):
        self = sequence_to_tensor(self, F.dtype(other))
    if isinstance(other, (tuple, list)):
        other = sequence_to_tensor(other, F.dtype(self))
    return F.tensor_pow(self, other)


def _tensor_floordiv(self, other):
    if isinstance(self, (tuple, list)):
        self = sequence_to_tensor(self, F.dtype(other))
    if isinstance(other, (tuple, list)):
        other = sequence_to_tensor(other, F.dtype(self))
    return F.floordiv(self, other)


tensor_operator_registry.register('__add__', _tensor_add)
tensor_operator_registry.register('__sub__', _tensor_sub)
tensor_operator_registry.register('__mul__', _tensor_mul)
tensor_operator_registry.register('__truediv__', _tensor_div)
tensor_operator_registry.register('__mod__', _tensor_mod)
tensor_operator_registry.register('__pow__', _tensor_pow)
tensor_operator_registry.register('__floordiv__', _tensor_floordiv)


def _broadcast(broadcast_shape, x):
    """Broadcast tensor to the required shape."""
    if not const_utils.check_two_shapes_need_broadcast(broadcast_shape, F.shape(x)):
        return x
    multiples = const_utils.compute_multiples(F.shape(x), broadcast_shape)
    if multiples:
        x = F.reshape(x, const_utils.expanded_shape(F.shape(x), len(multiples) - F.rank(x)))
        return F.tile(x, multiples)
    return x


def _transform_indexing_tensor(broadcast_shape, final_shape, new_shape, x):
    """Transform indexing tensor to the required."""
    x = _broadcast(broadcast_shape, x)
    return _broadcast(final_shape, F.reshape(x, new_shape))


def _transform_ellipsis_to_slice(data, tuple_index, op_name):
    """Check if the tuple index len is longer than the data's dims and transform ellipsis in the indices
    to several slice"""
    data_shape = F.shape(data)
    data_rank = len(data_shape)
    indexes_types = hyper_map(F.typeof, tuple_index)
    slice_positions, ellipsis_positions, _, int_positions, _, tensor_positions, sequence_positions = \
        const_utils.get_pos_of_indexes_types(indexes_types, op_name)

    ellipsis_occupy_dims = data_rank - (len(slice_positions) + len(int_positions) +
                                        len(tensor_positions) + len(sequence_positions))
    ellipsis_cnt = len(ellipsis_positions)
    # pylint: disable=chained-comparison
    if ellipsis_occupy_dims < 0 and ellipsis_cnt >= 0:
        const_utils.raise_index_error("For the 'getitem Operator', the data_shape should be no less than the "
                                      "tuple index dims")

    tuple_index_new = ()
    for i, index in enumerate(tuple_index):
        if i in ellipsis_positions:
            for _ in range(ellipsis_occupy_dims):
                empty_slice = const_utils.make_empty_slice()
                tuple_index_new += (empty_slice,)
        else:
            tuple_index_new += (index,)
    return tuple_index_new


def _expand_data_dims(data, tuple_index):
    """expand the data's dim with 'None' and 'Boolean' in tuple_index"""
    indexes_types = hyper_map(F.typeof, tuple_index)
    expand_positions, tuple_index_new = (), ()
    for i, (index, index_type) in enumerate(zip(tuple_index, indexes_types)):
        if const_utils.judge_index_type(index_type, mstype.type_none):
            tuple_index_new += (const_utils.make_empty_slice(),)
            expand_positions += (i,)
        elif const_utils.judge_index_type(index_type, mstype.bool_):
            if not index:
                const_utils.raise_index_error("Dose not support 'False'.")
            tuple_index_new += (const_utils.make_tensor([0], mstype.int64),)
            expand_positions += (i,)
        else:
            tuple_index_new += (index,)

    for dim in expand_positions:
        data = F.expand_dims(data, dim)

    return data, tuple_index_new


def tensor_index_by_slice(data, slice_index):
    """Tensor getitem by a slice."""
    min_data_dim, max_data_dim = 1, 8
    const_utils.judge_data_dim(data.ndim, min_data_dim, max_data_dim)
    data_shape = F.shape(data)
    begin_strides, end_strides, step_strides = const_utils.get_stride_info_from_slice(data_shape, slice_index)
    return F.strided_slice(data, begin_strides, end_strides, step_strides)


def tensor_index_by_number(data, number_index):
    """Tensor getitem by a Number which may be integer/float/bool value"""
    if isinstance(number_index, bool):
        return _tensor_index_by_bool(data, number_index)
    if isinstance(number_index, int):
        return _tensor_index_by_integer(data, number_index)
    return const_utils.raise_index_error("Only support integers, slices(`:`), ellipsis(`...`), None and bool.")


def _tensor_index_by_bool(data, bool_value):
    """Tensor getitem by a single bool value"""
    min_data_dim, max_data_dim = 0, 7
    const_utils.judge_data_dim(data.ndim, min_data_dim, max_data_dim)
    if bool_value:
        return F.expand_dims(data, 0)
    return const_utils.raise_index_error("When tensor is indexed by a bool object, the value only support 'True'.")


def _tensor_index_by_integer(data, int_index):
    """Tensor getitem by a single integer number"""
    if data.ndim < 1 or data.ndim > 8:
        const_utils.raise_value_error("Expect Tensor to have dimension between 1 and 8.")

    data_shape = F.shape(data)
    transformed_number = const_utils.check_range(int_index, data_shape[0])
    begin_strides, end_strides, step_strides = const_utils.get_stride_info_from_integer(data_shape, transformed_number)
    shrink_axis_mask = 1
    return P.StridedSlice(0, 0, 0, 0, shrink_axis_mask)(data, begin_strides, end_strides, step_strides)


def tensor_index_by_tensor(data, tensor_index):
    """Tensor getitem by a single tensor"""
    min_data_dim, max_data_dim = 0, 7
    const_utils.judge_data_dim(data.ndim, min_data_dim, max_data_dim)
    const_utils.check_type_valid(F.dtype(tensor_index), mstype.int_type, const_utils.TENSOR_GETITEM)
    return F.gather(data, tensor_index, 0)


def tensor_index_by_list(data, list_index):
    """Tensor getitem by list of int and bool"""
    min_data_dim, max_data_dim = 1, 8
    const_utils.judge_data_dim(data.ndim, min_data_dim, max_data_dim)

    data_shape = F.shape(data)
    indexes_types = hyper_map(F.typeof, list_index)
    if const_utils.judge_indexes_types(indexes_types, mstype.int_type + (mstype.bool_,)):
        tensor_index = const_utils.sequence_to_index(list_index, data_shape[0])
        if tensor_index is False:
            const_utils.raise_index_error("Getitem does not support empty list, this will reference shape '0'.")
        return F.gather(data, tensor_index, 0)

    tuple_index_new = ()
    for index in list_index:
        tuple_index_new += (index,)
    return tensor_index_by_tuple(data, tuple_index_new)


def tensor_index_by_tuple(data, tuple_index):
    """Tensor getitem by tuple of various types with None"""
    if not tuple_index:
        return data

    op_name = const_utils.TENSOR_GETITEM
    tuple_index = _transform_ellipsis_to_slice(data, tuple_index, op_name)
    data, tuple_index = _expand_data_dims(data, tuple_index)

    min_data_dim, max_data_dim = 1, 8
    const_utils.judge_data_dim(data.ndim, min_data_dim, max_data_dim)

    indexes_types = hyper_map(F.typeof, tuple_index)
    contain_type = const_utils.tuple_index_type_cnt(indexes_types, op_name)
    if contain_type == const_utils.ALL_BASIC:
        return _tensor_getitem_by_tuple_slice(data, tuple_index)
    return _tensor_getitem_by_tuple(data, tuple_index, op_name)


def _tensor_getitem_by_tuple_of_tensor(data, tuple_index, op_name):
    """Tensor getitem by a tuple of tensor."""
    data_shape = F.shape(data)
    tuple_index_len = len(tuple_index)

    indexes_types = hyper_map(F.dtype, tuple_index)
    const_utils.check_indexes_types_valid(indexes_types, mstype.int_type, op_name)
    tensor_index_shape = hyper_map(F.shape, tuple_index)
    broadcast_shape = const_utils.generate_broadcast_shape(tensor_index_shape, op_name)
    if 0 in broadcast_shape:
        res_shape = broadcast_shape
        if tuple_index_len < len(data_shape):
            res_shape += data_shape[tuple_index_len:]
        res = const_utils.make_tensor([], data.dtype, res_shape)
        return res

    broadcast_tensors = hyper_map(F.partial(_broadcast, broadcast_shape), tuple_index)
    new_broadcast_tensors = ()
    for tensor in broadcast_tensors:
        new_broadcast_tensors += (F.cast(tensor, mstype.int64),)
    indices = stack(new_broadcast_tensors)
    result = F.gather_nd(data, indices)
    return result


def _tensor_getitem_by_tuple_slice(data, tuple_index):
    """Tensor getitem by a tuple of slice"""
    data_shape = F.shape(data)
    begin_strides, end_strides, step_strides, shrink_axis_mask = const_utils.get_stride_info_from_tuple(
        data_shape, tuple_index)
    return P.StridedSlice(0, 0, 0, 0, shrink_axis_mask)(data, begin_strides, end_strides, step_strides)


def _tensor_getitem_by_tuple(data, tuple_index, op_name):
    """Tensor getitem by a tuple of mixed tensor."""
    data_shape = F.shape(data)
    data_rank = len(data_shape)
    tuple_index_len = len(tuple_index)
    tensor_indexes, slice_indexes = [], []
    indexes_types = hyper_map(F.typeof, tuple_index)
    slice_positions, _, _, int_positions, _, tensor_positions, sequence_positions = \
        const_utils.get_pos_of_indexes_types(indexes_types, op_name)
    tuple_index_new, slice_shapes = (), ()

    for i, (index, dim_size) in enumerate(zip(tuple_index, data_shape)):
        if i in int_positions:
            int_index = const_utils.check_range(index, dim_size)
            tensor_index = F.scalar_to_tensor(int_index, mstype.int64)
            tuple_index_new += (tensor_index,)
            tensor_indexes.append(tensor_index)
            tensor_positions += (i,)
        elif i in sequence_positions:
            tensor_index = const_utils.sequence_to_index(index, dim_size)
            tuple_index_new += (tensor_index,)
            tensor_indexes.append(tensor_index)
            tensor_positions += (i,)
        elif i in tensor_positions:
            const_utils.check_type_valid(F.dtype(index), mstype.int_type, op_name)
            tensor_index = F.cast(index, mstype.int64)
            tuple_index_new += (tensor_index,)
            tensor_indexes.append(tensor_index)
        elif i in slice_positions:
            slice_ele_list_index = const_utils.transform_slice_to_ele_list(index, dim_size)
            slice_shapes += (len(slice_ele_list_index),)
            tuple_index_new += (slice_ele_list_index,)
            slice_indexes.append(slice_ele_list_index)

    tensor_indexes_shapes = hyper_map(F.shape, tensor_indexes)
    broadcast_shape, index_tensor_new_shape, final_shape, fancy_position = \
        const_utils.generate_index_info_from_tuple_of_mixed_tensors(tensor_positions, tensor_indexes_shapes,
                                                                    slice_shapes, op_name)

    if 0 in final_shape + data_shape:
        if tuple_index_len < data_rank:
            final_shape = final_shape + data_shape[tuple_index_len:]
        return const_utils.make_tensor([], data.dtype, final_shape)

    final_index_tensors = []
    slice_cnt = 0
    for i, index in enumerate(tuple_index_new):
        if i in tensor_positions:
            transform_tensor = _transform_indexing_tensor(broadcast_shape, final_shape, index_tensor_new_shape, index)
            final_index_tensors.append(transform_tensor)
        elif i in slice_positions:
            slice_index_tensor = const_utils.convert_slice_to_tensor(index, final_shape, slice_cnt, broadcast_shape,
                                                                     slice_shapes, fancy_position)
            final_index_tensors.append(slice_index_tensor)
            slice_cnt += 1

    indices = stack(final_index_tensors)
    result = F.gather_nd(data, indices)
    return result


def _generate_indices_from_tuple_of_tensor(tuple_index, op_name):
    """Generate an indices tensor from a tuple of tensor."""
    indices = None
    indexes_types = hyper_map(F.dtype, tuple_index)
    const_utils.check_types_valid(indexes_types, mstype.int_type, op_name)
    tensor_index_shape = hyper_map(F.shape, tuple_index)
    broadcast_shape = const_utils.generate_broadcast_shape(tensor_index_shape, op_name)
    if len(broadcast_shape) < 2:
        broadcast_shape = (1,) + broadcast_shape
    broadcast_tensors = hyper_map(F.partial(_broadcast, broadcast_shape), tuple_index)
    new_broadcast_tensors = ()
    for tensor in broadcast_tensors:
        new_broadcast_tensors += (F.cast(tensor, mstype.int64),)
    indices = stack(new_broadcast_tensors)
    return indices


def _generate_indices_from_tuple(data, tuple_index, op_name):
    """Generate an indices tensor from a tuple that contains slice, int, ellipsis, tensor."""
    data_shape = F.shape(data)
    tensor_indexes, slice_indexes = [], []
    indexes_types = hyper_map(F.typeof, tuple_index)
    slice_positions, _, _, int_positions, _, tensor_positions, sequence_positions = \
        const_utils.get_pos_of_indexes_types(indexes_types, op_name)
    tuple_index_new, slice_shapes = (), ()

    for i, (index, dim_size) in enumerate(zip(tuple_index, data_shape)):
        if i in int_positions:
            int_index = const_utils.check_range(index, dim_size)
            tensor_index = F.scalar_to_tensor(int_index, mstype.int64)
            tuple_index_new += (tensor_index,)
            tensor_indexes.append(tensor_index)
            tensor_positions += (i,)
        elif i in sequence_positions:
            tensor_index = const_utils.sequence_to_index(index, dim_size)
            tuple_index_new += (tensor_index,)
            tensor_indexes.append(tensor_index)
            tensor_positions += (i,)
        elif i in tensor_positions:
            const_utils.check_type_valid(F.dtype(index), mstype.int_type, op_name)
            tensor_index = F.cast(index, mstype.int64)
            tuple_index_new += (tensor_index,)
            tensor_indexes.append(tensor_index)
        elif i in slice_positions:
            start, stop, step = const_utils.normalize_slice(index, dim_size)
            if const_utils.check_slice_empty(start, stop, step):
                return False
            slice_ele_list_index = const_utils.transform_slice_to_ele_list(index, dim_size)
            slice_shapes += (len(slice_ele_list_index),)
            tuple_index_new += (slice_ele_list_index,)
            slice_indexes.append(slice_ele_list_index)

    tensor_indexes_shapes = hyper_map(F.shape, tensor_indexes)
    broadcast_shape, index_tensor_new_shape, final_shape, fancy_position = \
        const_utils.generate_index_info_from_tuple_of_mixed_tensors(tensor_positions, tensor_indexes_shapes,
                                                                    slice_shapes, op_name)

    final_index_tensors = []
    slice_cnt = 0
    for i, index in enumerate(tuple_index_new):
        if i in tensor_positions:
            transform_tensor = _transform_indexing_tensor(broadcast_shape, final_shape, index_tensor_new_shape, index)
            final_index_tensors.append(transform_tensor)
        elif i in slice_positions:
            slice_index_tensor = const_utils.convert_slice_to_tensor(index, final_shape, slice_cnt, broadcast_shape,
                                                                     slice_shapes, fancy_position)
            final_index_tensors.append(slice_index_tensor)
            slice_cnt += 1

    indices = stack(final_index_tensors)
    return indices


def _generate_updates_from_scalar(data, indices, value, op_type):
    """Generate an updates tensor from a scalar."""
    data_shape = F.shape(data)
    indices_shape = F.shape(indices)
    data_dtype = F.dtype(data)
    return const_utils.convert_scalar_to_tensor(data_shape, data_dtype, indices_shape, value, op_type)


def sequence_to_tensor(value, dtype):
    """Generate an updates tensor from a tuple, can only handle 1-D tensor/non-tensor mixtures."""
    value_types = hyper_map(F.typeof, value)
    value_elements_type = const_utils.check_value_elements(value_types)

    if value_elements_type == const_utils.ALL_TENSOR:
        value = F.stack(value).astype(dtype)
    elif value_elements_type == const_utils.NO_TENSOR:
        value = const_utils.make_tensor(value, dtype)
    else:
        new_value = ()
        for ele in value:
            ele = ele if isinstance(ele, Tensor) else const_utils.make_tensor(ele)
            new_value += (ele,)
        value = F.stack(new_value).astype(dtype)
    return value


def _generate_updates_from_sequence(data, index, value, op_type):
    """Generate an updates tensor from a tuple, can only handle 1-D tensor/non-tensor mixtures."""
    value = sequence_to_tensor(value, F.dtype(data))
    if op_type == const_utils.SET_ITEM_BY_NON_TENSOR:
        return value
    return _generate_updates_from_tensor(data, index, value, op_type)


def _generate_updates_from_tensor(data, index, value, op_type):
    """Generate an updates tensor from a tensor."""
    value = value.astype(data.dtype)
    updates_shape = const_utils.generate_updates_shape(data.shape, index.shape, op_type)
    need_broadcast = const_utils.check_two_shapes_need_broadcast(updates_shape, value.shape)
    if need_broadcast:
        return _broadcast(updates_shape, value)
    return value


# Tensor getitem implementations are above this line, setitem implementations below.

def tensor_setitem_by_tensor(self, index, value):
    if isinstance(value, (int, float, bool)):
        return tensor_setitem_by_tensor_with_number(self, index, value)
    if isinstance(value, Tensor):
        return tensor_setitem_by_tensor_with_tensor(self, index, value)
    return tensor_setitem_by_tensor_with_sequence(self, index, value)


def tensor_setitem_by_tuple(self, index, value):
    if isinstance(value, (int, float, bool)):
        index = format_tuple_indices(index)
        return tensor_setitem_by_tuple_with_number(self, index, value)
    if isinstance(value, Tensor):
        return tensor_setitem_by_tuple_with_tensor(self, index, value)
    return tensor_setitem_by_tuple_with_sequence(self, index, value)


def tensor_setitem_by_number(self, index, value):
    if isinstance(value, (int, float, bool)):
        return tensor_setitem_by_number_with_number(self, index, value)
    if isinstance(value, Tensor):
        return tensor_setitem_by_number_with_tensor(self, index, value)
    return tensor_setitem_by_number_with_sequence(self, index, value)


def tensor_setitem_by_slice(self, index, value):
    if isinstance(value, (int, float, bool)):
        return tensor_setitem_by_slice_with_number(self, index, value)
    if isinstance(value, Tensor):
        return tensor_setitem_by_slice_with_tensor(self, index, value)
    return tensor_setitem_by_slice_with_sequence(self, index, value)


def tensor_setitem_by_ellipsis(self, index, value):
    if isinstance(value, (int, float, bool)):
        return tensor_setitem_by_ellipsis_with_number(self, value)
    if isinstance(value, Tensor):
        return tensor_setitem_by_ellipsis_with_tensor(self, value)
    return tensor_setitem_by_ellipsis_with_sequence(self, value)


def _tensor_setitem_by_int_tensor_with_tensor(data, index, value):
    """Set a tensor item by a int tensor with a tensor."""
    updates = _generate_updates_from_tensor(data, index, value, const_utils.SET_ITEM_BY_ONE_TENSOR)
    index = F.expand_dims(index, -1)
    return P.TensorScatterUpdate()(data, index, updates)


def _tensor_setitem_by_bool_tensor_with_tensor(data, index, value):
    """Set a tensor item by a bool tensor with a tensor."""
    index_shape = F.shape(index)
    data_shape = F.shape(data)
    data_shape = const_utils.check_equal(data_shape, index_shape,
                                         "The tensor(shape={}) and tensor index(shape={}) should be the same shape.")
    size = F.size(value)
    size = const_utils.check_equal(1, size,
                                   "When assign value is a tensor, its size should be {}, but current size is {}.")
    dtype = F.dtype(data)
    u_cast = F.cast(value, dtype)
    one_data = F.ones_like(data)
    u = F.tensor_mul(one_data, u_cast)
    result = F.select(index, u, data)
    return result


def tensor_setitem_by_tensor_with_tensor(data, index, value_tensor):
    """setitem by tensor index(dtype is int or bool) with tensor as value"""
    index_dtype = F.dtype(index)
    tensor_dtype = const_utils.get_index_tensor_dtype(index_dtype)
    if tensor_dtype == const_utils.INT_:
        return _tensor_setitem_by_int_tensor_with_tensor(data, index, value_tensor)
    return _tensor_setitem_by_bool_tensor_with_tensor(data, index, value_tensor)


def _tensor_setitem_by_bool_tensor_with_scalar(data, index, value):
    """Set a tensor item by a bool tensor with a scalar."""
    index_shape = F.shape(index)
    shape = F.shape(data)
    shape = const_utils.check_equal(
        shape, index_shape, "The tensor(shape={}) and tensor index(shape={}) should be the same shape.")
    dtype = F.dtype(data)
    u = F.fill(dtype, shape, value)
    return F.select(index, u, data)


def _tensor_setitem_by_int_tensor_with_scalar(data, index, value):
    """Set a tensor item by a int tensor with a scalar."""
    if not F.shape(index):
        index = F.expand_dims(index, 0)
    updates = _generate_updates_from_scalar(data, index, value, const_utils.SET_ITEM_BY_ONE_TENSOR)
    index = F.expand_dims(index, -1)
    return P.TensorScatterUpdate()(data, index, updates)


def tensor_setitem_by_tensor_with_number(data, index, value):
    index_dtype = F.dtype(index)
    tensor_dtype = const_utils.get_index_tensor_dtype(index_dtype)
    if tensor_dtype == const_utils.BOOL_:
        return _tensor_setitem_by_bool_tensor_with_scalar(data, index, value)
    if tensor_dtype == const_utils.INT_:
        return _tensor_setitem_by_int_tensor_with_scalar(data, index, value)
    return const_utils.raise_index_error("For tensor setitem, indexing tensor dtype only supports bool/int")


def tensor_setitem_by_tensor_with_sequence(data, index, value):
    """Assigns the tensor by tensor with tuple value."""
    index_dtype = F.dtype(index)
    const_utils.check_type_valid(index_dtype, (mstype.int32, mstype.int64), const_utils.TENSOR_SETITEM)
    return _tensor_setitem_by_tensor_with_sequence(data, index, value)


def _tensor_setitem_by_tensor_with_sequence(data, index, value):
    """Set a tensor item by a tensor with a tuple."""
    updates = _generate_updates_from_sequence(data, index, value, const_utils.SET_ITEM_BY_ONE_TENSOR)
    index = F.expand_dims(index, -1)
    return P.TensorScatterUpdate()(data, index, updates)


def tensor_setitem_by_slice_with_number(data, input_slice, value):
    """Givens a scalar assign to tensor by slice"""
    value = F.fill(F.dtype(data), const_utils.tuple_slice(F.shape(data), 1, None), value)
    return tensor_setitem_by_slice_with_tensor(data, input_slice, value)


def tensor_setitem_by_tuple_with_number(data, tuple_index, value):
    """Assigns the tensor by tuple with number value."""
    tuple_index = _transform_ellipsis_to_slice(data, tuple_index, const_utils.TENSOR_GETITEM)
    tuple_index, _ = remove_expanded_dims(tuple_index, F.shape(data))
    if tuple_index is False:
        return data

    if len(tuple_index) == 1:
        data[tuple_index[0]] = value
        return data

    indexes_types = hyper_map(F.typeof, tuple_index)
    contain_type = const_utils.tuple_index_type_cnt(indexes_types, const_utils.TENSOR_SETITEM)

    if contain_type == const_utils.ALL_TENSOR:
        indices = _generate_indices_from_tuple_of_tensor(tuple_index, const_utils.TENSOR_SETITEM)
    else:
        int_cnt = const_utils.tuple_index_int_cnt(indexes_types, const_utils.TENSOR_SETITEM)
        if int_cnt == const_utils.ALL_INT:
            tuple_index = const_utils.convert_int_to_slice(tuple_index)
        indices = _generate_indices_from_tuple(data, tuple_index, const_utils.TENSOR_SETITEM)
        if indices is False:
            return data
    updates = _generate_updates_from_scalar(data, indices, value, const_utils.SET_ITEM_BY_TUPLE_OF_TENSOR)
    return P.TensorScatterUpdate()(data, indices, updates)


def tensor_setitem_by_slice_with_tensor(data, input_slice, value):
    """Assigns a tensor value to the tensor by slice."""
    result = None
    check_result = const_utils.check_tensor_setitem_index(input_slice)
    if check_result:
        data_shape = F.shape(data)
        indices = const_utils.slice2indices(input_slice, data_shape)
        if indices is False:
            return data
        value_shape = const_utils.tuple_slice(F.shape(indices), None, -1)
        value = _broadcast(value_shape, value)
        result = P.TensorScatterUpdate()(data, indices, value.astype(F.dtype(data)))
    return result


def tensor_setitem_by_slice_with_sequence(data, input_slice, value):
    """Assigns a list/tuple value to the tensor by slice."""
    value = _generate_updates_from_sequence(data, input_slice, value, const_utils.SET_ITEM_BY_NON_TENSOR)
    return tensor_setitem_by_slice_with_tensor(data, input_slice, value)


def tensor_setitem_by_tuple_with_tensor(data, tuple_index, value):
    """Assigns the tensor by tuple with tensor value."""
    op_name = const_utils.TENSOR_SETITEM
    tuple_index = _transform_ellipsis_to_slice(data, tuple_index, op_name)
    tuple_index, not_expanded_dim = remove_expanded_dims(tuple_index, F.shape(data))
    if tuple_index is False:
        return data
    value_shape = const_utils.filter_expanded_dims(F.shape(value), not_expanded_dim)
    value = F.reshape(value, value_shape)

    if len(tuple_index) == 1:
        data[tuple_index[0]] = value
        return data

    indexes_types = hyper_map(F.typeof, tuple_index)
    contain_type = const_utils.tuple_index_type_cnt(indexes_types, const_utils.TENSOR_SETITEM)

    if contain_type == const_utils.ALL_TENSOR:
        indices = _generate_indices_from_tuple_of_tensor(tuple_index, const_utils.TENSOR_SETITEM)
    else:
        int_cnt = const_utils.tuple_index_int_cnt(indexes_types, const_utils.TENSOR_SETITEM)
        if int_cnt == const_utils.ALL_INT:
            tuple_index = const_utils.convert_int_to_slice(tuple_index)
            new_shape = ()
            for _ in tuple_index:
                new_shape += (1,)
            new_shape += value.shape
            value = F.reshape(value, new_shape)
        indices = _generate_indices_from_tuple(data, tuple_index, const_utils.TENSOR_SETITEM)
        if indices is False:
            return data
    updates = _generate_updates_from_tensor(data, indices, value, const_utils.SET_ITEM_BY_TUPLE_OF_TENSOR)
    return P.TensorScatterUpdate()(data, indices, updates)


def tensor_setitem_by_tuple_with_sequence(data, tuple_index, value):
    value = _generate_updates_from_sequence(data, tuple_index, value, const_utils.SET_ITEM_BY_NON_TENSOR)
    return tensor_setitem_by_tuple_with_tensor(data, tuple_index, value)


def tensor_setitem_by_number_with_number(data, index, value):
    """Assigns the tensor by number with number value."""
    value = F.fill(F.dtype(data), const_utils.tuple_slice(F.shape(data), 1, None), value)
    return tensor_setitem_by_number_with_tensor(data, index, value)


def tensor_setitem_by_number_with_sequence(data, index, value):
    """Assigns a list/tuple value to the tensor by slice."""
    value = _generate_updates_from_sequence(data, index, value, const_utils.SET_ITEM_BY_NON_TENSOR)
    return tensor_setitem_by_number_with_tensor(data, index, value)


def tensor_setitem_by_number_with_tensor(data, index, value):
    """Assigns the tensor by number with tensor value."""
    data_shape = F.shape(data)
    index = const_utils.int_to_index(index, data_shape)
    value_shape = const_utils.tuple_slice(F.shape(index), None, -1)
    value = _broadcast(value_shape, value)
    return P.TensorScatterUpdate()(data, index, value)


def tensor_setitem_by_ellipsis_with_number(data, value):
    """Assigns the tensor by ellipsis with number value."""
    data_shape = F.shape(data)
    data_dtype = F.dtype(data)
    return F.fill(data_dtype, data_shape, value)


def tensor_setitem_by_ellipsis_with_tensor(data, value):
    """Assigns the tensor by ellipsis with tensor value."""
    data_shape = F.shape(data)
    data_dtype = F.dtype(data)
    value = value.astype(data_dtype)
    value_shape = F.shape(value)
    source_shape = const_utils.get_source_shape(data_shape, value_shape)
    value = F.reshape(value, source_shape)
    value = _broadcast(data_shape, value)
    data = F.cast(value, data_dtype)
    return data


def tensor_setitem_by_ellipsis_with_sequence(data, value):
    """Assigns a list/tuple value to the tensor by ellipsis."""
    value = _generate_updates_from_sequence(data, None, value, const_utils.SET_ITEM_BY_NON_TENSOR)
    return tensor_setitem_by_ellipsis_with_tensor(data, value)


def tensor_setitem_by_bool(data, index, value):
    """Assigns a value to the tensor by boolean."""
    data_shape = F.shape(data)
    if not index:
        data_shape = (0,) + data_shape
    if isinstance(value, (list, tuple)):
        value = _generate_updates_from_sequence(data, index, value, const_utils.SET_ITEM_BY_NON_TENSOR)
    elif isinstance(value, (int, float, bool)):
        value = const_utils.make_tensor(value)
    value_shape = F.shape(value)
    source_shape = const_utils.get_source_shape(data_shape, value_shape)
    if index:
        value = F.reshape(value, source_shape)
        value = _broadcast(data_shape, value)
        data = value
    return data


def tensor_in_sequence(x, y):
    """Assigns whether a sequence contains the given tensor"""
    result = const_utils.scalar_to_tensor(False)
    for i in y:
        if isinstance(i, Tensor) and x.shape == i.shape and x.dtype == i.dtype:
            result = F.logical_or(F.equal(x, i).all(), result)
    return result


def format_list_indices(list_indices, length):
    """Convert list indices to tensor or tuple indices based on its contents."""
    indices_types = hyper_map(F.typeof, list_indices)
    # If eyery element in list is bool, it's treated as 1-D bool tensor.
    # If every element in list is int(not all bool), it's treated as int tensor.
    if const_utils.judge_indexes_types(indices_types, mstype.int_type+(mstype.bool_,)):
        return const_utils.sequence_to_index(list_indices, length)
    # If list contains other types(.../list/tuple/None), it's treated as a tuple
    return const_utils.deep_tuple(list_indices)


def format_tuple_indices(tuple_indices):
    """
    Format tuple indices by unpacking high-dimension tuple and removing expand
    dimension signs(Bool and None).
    """
    res = ()
    for i in tuple_indices:
        if isinstance(i, (list, tuple)):
            res += (const_utils.unpack(i),)
        else:
            res += (i,)
    return res


def remove_expanded_dims(tuple_index, data_shape):
    """Removes expanded dimensions in tuple_index and value."""
    op_name = const_utils.TENSOR_SETITEM
    not_expanded_dim = ()
    shapes = ()
    has_true = False
    has_false = False
    has_sequence = False
    indices_out = ()      # with dimension expansion indices removed
    idx_tensor = -1       # index of the previous tensor
    idx_advanced = -1     # index of the first advanced index in expanded tensor
    cur_dim = 0           # current dimension of the data to be indexed

    for i, v in enumerate(tuple_index):
        index_out = format_index(v, data_shape, cur_dim)

        if index_out is None:
            not_expanded_dim += (False,)
        elif const_utils.is_slice(index_out):
            indices_out += (index_out,)
            not_expanded_dim += (True,)
            start, stop, step = const_utils.normalize_slice(index_out, data_shape[cur_dim])
            if const_utils.check_slice_empty(start, stop, step):
                has_false = True
            cur_dim += 1
        elif isinstance(index_out, (Tensor, bool)): # advanced index
            if idx_advanced == -1:
                idx_advanced = len(not_expanded_dim)
            elif i - idx_tensor > 1:
                idx_advanced = 0
            idx_tensor = i
            if isinstance(index_out, Tensor):
                if F.rank(index_out) > 0:
                    has_sequence = True
                indices_out += (index_out,)
                shapes += (F.shape(index_out),)
                cur_dim += 1
            has_true = has_true or index_out is True
            has_false = has_false or index_out is False
        else:
            const_utils.raise_index_error('invalid index type')

    broadcast_shape = const_utils.generate_broadcast_shape(shapes, op_name)
    if has_false:
        if F.shape_mul(broadcast_shape) != 1:
            const_utils.raise_index_error('unable to broadcast indices')
        return False, not_expanded_dim

    expand_true = has_true and not(has_false or has_sequence) # whether to expand dimension at True
    tensor_index_ndim = len(broadcast_shape)                  # ndim of tensor indices
    rem_ndim = len(data_shape) - cur_dim       # number of remaining dimensions in data not indexed
    not_expanded_dim = const_utils.rem_not_expanded_dims(idx_advanced, expand_true, tensor_index_ndim,
                                                         rem_ndim, not_expanded_dim)

    if not indices_out:
        indices_out = (True,)
    return indices_out, not_expanded_dim


def format_index(idx, data_shape, cur_dim):
    """Converts advanced index into tensor."""
    if isinstance(idx, (tuple, list)):
        idx = const_utils.sequence_to_index(idx, data_shape[cur_dim])
    elif isinstance(idx, int) and not isinstance(idx, bool):
        idx = const_utils.make_tensor(idx, mstype.int64, None, data_shape[cur_dim])
    return idx
