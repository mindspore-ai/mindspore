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
pack = P.Stack(axis=-1)


def _broadcast(broadcast_shape, x):
    """Broadcast tensor to the required shape."""
    if F.shape(x) == broadcast_shape:
        return x
    multiples = const_utils.compute_multiples(F.shape(x), broadcast_shape)
    if multiples:
        return F.tile(x, multiples)
    return x


def _transform_indexing_tensor(broadcast_shape, final_shape, new_shape, x):
    """Transform indexing tensor to the required."""
    x = _broadcast(broadcast_shape, x)
    return _broadcast(final_shape, F.reshape(x, new_shape))


def _transform_ellipsis_to_slice(tuple_index, data, op_name):
    """transform ellipsis in the slice to several slice"""
    data_shape = F.shape(data)
    data_rank = len(data_shape)
    indexes_types = hyper_map(F.typeof, tuple_index)
    slice_positions, ellipsis_positions, _, int_positions, _, tensor_positions, sequence_positions = \
        const_utils.get_pos_of_indexes_types(indexes_types, op_name)
    ellipsis_occupy_dims = data_rank - (len(slice_positions) + len(int_positions) +
                                        len(tensor_positions) + len(sequence_positions))

    tuple_index_new = ()
    for i, index in enumerate(tuple_index):
        if i in ellipsis_positions:
            for _ in range(ellipsis_occupy_dims):
                empty_slice = const_utils.make_empty_slice()
                tuple_index_new += (empty_slice,)
        else:
            tuple_index_new += (index,)
    return tuple_index_new


def _generate_indices_from_tuple_of_tensor(data, tuple_index, op_name):
    """Generate an indices tensor from a tuple of tensor."""
    indices = None
    check_index_tensor_number = const_utils.check_number_of_index_tensor(F.shape(data), len(tuple_index), op_name)
    if check_index_tensor_number:
        dtype_tuple = hyper_map(F.dtype, tuple_index)
        check_dtypes = const_utils.check_index_tensors_dtype(dtype_tuple, op_name)
        if check_dtypes:
            shape_tuple = hyper_map(F.shape, tuple_index)
            broadcast_shape = const_utils.generate_broadcast_shape(shape_tuple, op_name)
            broadcast_tensors = hyper_map(F.partial(_broadcast, broadcast_shape), tuple_index)
            indices = pack(broadcast_tensors)
    return indices


def _generate_indices_from_tuple(data, tuple_index, op_name):
    """Generate an indices tensor from a tuple that contains slice, int, ellipsis, tensor."""
    data_shape = F.shape(data)
    indexes_types = hyper_map(F.typeof, tuple_index)
    int_positions, sequence_positions = const_utils.get_pos_of_int_sequence(indexes_types)
    tuple_index_new = ()
    tuple_len = len(tuple_index)

    for i in range(tuple_len):
        index = tuple_index[i]
        shape = data_shape[i]
        if i in int_positions:
            int_index = const_utils.check_and_transform_int_index(index, shape, op_name)
            tensor_index = F.scalar_to_tensor(int_index, mstype.int32)
            tuple_index_new += (tensor_index,)
        elif i in sequence_positions:
            sequence_index = const_utils.transform_sequence_index(index, shape, op_name)
            tensor_index = F.tuple_to_array(sequence_index)
            tuple_index_new += (tensor_index,)
        else:
            tuple_index_new += (index,)

    indexes_types = hyper_map(F.typeof, tuple_index_new)
    tensor_positions, slice_positions, ellipsis_position = \
        const_utils.separate_mixed_tensors_index(indexes_types, op_name)
    tensor_indexes, slice_indexes = [], []
    for i in tensor_positions:
        tensor_indexes.append(tuple_index_new[i])
    for j in slice_positions:
        slice_indexes.append(tuple_index_new[j])

    tensor_indexes_shapes = hyper_map(F.shape, tensor_indexes)
    tensor_indexes_dtypes = hyper_map(F.dtype, tensor_indexes)
    broadcast_shape, final_shape, indexes_shapes_info, ellipsis_occupied_dims = \
        const_utils.generate_index_info_from_tuple_of_mixed_tensors(data_shape,
                                                                    indexes_types,
                                                                    tensor_indexes_shapes,
                                                                    tensor_indexes_dtypes,
                                                                    slice_indexes,
                                                                    op_name)

    slice_number = 0
    final_index_tensors = []
    tuple_index_size = len(tuple_index_new)
    index_tensor_new_shape = const_utils.compute_new_shape(broadcast_shape, indexes_shapes_info)
    for i in range(tuple_index_size):
        if i in tensor_positions:
            transform_tensor = _transform_indexing_tensor(
                broadcast_shape, final_shape, index_tensor_new_shape, tuple_index_new[i])
            final_index_tensors.append(transform_tensor)
        if i in slice_positions:
            slice_tensor = const_utils.convert_slice_to_tensor(slice_number, final_shape, indexes_shapes_info, op_name)
            final_index_tensors.append(slice_tensor)
            slice_number += 1
        if i == ellipsis_position:
            ellipsis_tensors = const_utils.convert_ellipsis_to_tensors(
                slice_number, ellipsis_occupied_dims, final_shape, indexes_shapes_info, op_name)
            for ele in ellipsis_tensors:
                final_index_tensors.append(ele)
            slice_number += ellipsis_occupied_dims
    indices = pack(final_index_tensors)
    return indices


def _generate_indices_from_tuple_of_mixed_tensors(data, tuple_index, op_name):
    """Generate an indices tensor from a tuple that contains slice, int, ellipsis, tensor."""
    data_shape = F.shape(data)
    indexes_types = hyper_map(F.typeof, tuple_index)
    int_positions = const_utils.get_pos_of_int_index(indexes_types)
    tuple_index_new = ()
    tuple_len = len(tuple_index)
    for i in range(tuple_len):
        if i in int_positions:
            tuple_index_new += (F.scalar_to_tensor(tuple_index[i] if tuple_index[i] >= 0 else tuple_index[i] +
                                                   data_shape[i], mstype.int32),)
        else:
            tuple_index_new += (tuple_index[i],)
    indexes_types = hyper_map(F.typeof, tuple_index_new)
    tensor_positions, slice_positions, ellipsis_position = \
        const_utils.separate_mixed_tensors_index(indexes_types, op_name)
    tensor_indexes = []
    slice_indexes = []
    for i in tensor_positions:
        tensor_indexes.append(tuple_index_new[i])
    for j in slice_positions:
        slice_indexes.append(tuple_index_new[j])
    tensor_indexes_shapes = hyper_map(F.shape, tensor_indexes)
    tensor_indexes_dtypes = hyper_map(F.dtype, tensor_indexes)
    broadcast_shape, final_shape, indexes_shapes_info, ellipsis_occupied_dims = \
        const_utils.generate_index_info_from_tuple_of_mixed_tensors(data_shape,
                                                                    indexes_types,
                                                                    tensor_indexes_shapes,
                                                                    tensor_indexes_dtypes,
                                                                    slice_indexes,
                                                                    op_name)

    slice_number = 0
    final_index_tensors = []
    tuple_index_size = len(tuple_index_new)
    index_tensor_new_shape = const_utils.compute_new_shape(broadcast_shape, indexes_shapes_info)
    for i in range(tuple_index_size):
        if i in tensor_positions:
            transform_tensor = _transform_indexing_tensor(
                broadcast_shape, final_shape, index_tensor_new_shape, tuple_index_new[i])
            final_index_tensors.append(transform_tensor)
        if i in slice_positions:
            slice_tensor = const_utils.convert_slice_to_tensor(slice_number, final_shape, indexes_shapes_info, op_name)
            final_index_tensors.append(slice_tensor)
            slice_number += 1
        if i == ellipsis_position:
            ellipsis_tensors = const_utils.convert_ellipsis_to_tensors(
                slice_number, ellipsis_occupied_dims, final_shape, indexes_shapes_info, op_name)
            for ele in ellipsis_tensors:
                final_index_tensors.append(ele)
            slice_number += ellipsis_occupied_dims
    indices = pack(final_index_tensors)
    return indices


def _generate_updates_from_scalar(data, indices, value, op_type):
    """Generate an updates tensor from a scalar."""
    data_shape = F.shape(data)
    indices_shape = F.shape(indices)
    data_dtype = F.dtype(data)
    return const_utils.convert_scalar_to_tensor(data_shape, data_dtype, indices_shape, value, op_type)


def _generate_updates_from_tuple(data, index, value, op_type):
    """Generate an updates tensor from a tuple."""
    value_types = hyper_map(F.typeof, value)
    data_dtype = F.dtype(data)
    value_elements_type = const_utils.check_value_elements(data_dtype, value_types)
    if value_elements_type == const_utils.ALL_TENSOR:
        value_shapes = hyper_map(F.shape, value)
        shapes_same = const_utils.check_shapes_same(value_shapes, const_utils.TENSOR_SETITEM)
        if shapes_same:
            value = F.stack(value)
        return _generate_updates_from_tensor(data, index, value, op_type)

    data_shape = F.shape(data)
    index_shape = F.shape(index)
    return const_utils.convert_tuple_of_scalar_to_tensor(data_shape, data_dtype, index_shape, value, op_type)


def _generate_updates_from_tensor(data, index, value, op_type):
    """Generate an updates tensor from a tensor."""
    data_shape = F.shape(data)
    index_shape = F.shape(index)
    value_shape = F.shape(value)
    data_dtype = F.dtype(data)
    value_dtype = F.dtype(value)
    updates_shape = value_shape
    check_dtype_same = const_utils.check_tensors_dtype_same(data_dtype, value_dtype, const_utils.TENSOR_SETITEM)
    if check_dtype_same:
        updates_shape = const_utils.generate_updates_shape(data_shape, index_shape, op_type)
    need_broadcast = const_utils.check_two_shapes_need_broadcast(updates_shape, value_shape)
    if need_broadcast:
        return _broadcast(updates_shape, value)
    return value


def _tensor_getitem(self, index):
    """Handle tensor getitem"""
    if isinstance(index, Tensor):
        return tensor_index_by_tensor(self, index)
    if isinstance(index, tuple):
        return tensor_index_by_tuple(self, index)
    if isinstance(index, list):
        return tensor_index_by_list(self, index)
    # bool type should be judged before int
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
    raise IndexError(f"Only support integers, slices(`:`), ellipsis(`...`), None, bool and tensor with int32, "
                     f"got {index} with type {type(index)}.")


tensor_operator_registry.register("__getitem__", _tensor_getitem)


def _tensor_getitem_by_tuple_of_tensor(data, tuple_index):
    """Tensor getitem by a tuple of tensor."""
    indices = _generate_indices_from_tuple_of_tensor(data,
                                                     tuple_index,
                                                     const_utils.TENSOR_GETITEM)
    result = F.gather_nd(data, indices)
    return result


def _tensor_getitem_by_tuple(data, tuple_index):
    """Tensor getitem by a tuple of mixed tensor."""
    indices = _generate_indices_from_tuple(data, tuple_index, const_utils.TENSOR_GETITEM)
    result = F.gather_nd(data, indices)
    return result


def _tensor_getitem_by_tuple_of_mixed_tensors(data, tuple_index):
    """Tensor getitem by a tuple of mixed tensor."""
    indices = _generate_indices_from_tuple_of_mixed_tensors(data,
                                                            tuple_index,
                                                            const_utils.TENSOR_GETITEM)
    result = F.gather_nd(data, indices)
    return result


def tensor_index_by_slice(data, slice_index):
    """Tensor getitem by a single slice"""
    shape = F.shape(data)
    if not shape:
        const_utils.raise_index_error("When tensor is indexed by a slice, the dimension of the tensor"
                                      "cannot be 0.")
    begin_strides, end_strides, step_strides = const_utils.get_stride_info_from_slice(shape, slice_index)
    return F.strided_slice(data, begin_strides, end_strides, step_strides)


def _tensor_index_by_integer(data, number):
    """Tensor getitem by a single integer number"""
    shape = F.shape(data)
    if not shape:
        return const_utils.raise_type_error("When tensor is indexed by an integer,"
                                            "the dimension of the tensor cannot be 0.")
    if number >= shape[0]:
        return const_utils.raise_index_error("index {} is out of bounds for axis 0 with size {}".format(
            number, shape[0]))
    begin_strides, end_strides, step_strides = const_utils.get_stride_info_from_integer(shape, number)
    shrink_axis_mask = 1
    return P.StridedSlice(0, 0, 0, 0, shrink_axis_mask)(data, begin_strides, end_strides, step_strides)


def _tensor_index_by_bool(data, bool_value):
    """Tensor getitem by a single bool value"""
    if bool_value:
        return F.expand_dims(data, 0)
    return const_utils.raise_index_error("When tensor is indexed by a bool object, the value only support 'True'.")


def tensor_index_by_number(data, number):
    """Tensor getitem by a Number which may be integer/float/bool value"""
    number_type = const_utils.check_number_index_type(number)
    if number_type == const_utils.BOOL_:
        return _tensor_index_by_bool(data, number)
    if number_type == const_utils.INT_:
        return _tensor_index_by_integer(data, number)
    return const_utils.raise_index_error("Only support integers, slices(`:`), ellipsis(`...`), None and bool.")


def tensor_index_by_tensor(data, tensor_index):
    """Tensor getitem by a single tensor"""
    dtype_valid = const_utils.check_index_tensor_dtype(F.dtype(tensor_index),
                                                       const_utils.TENSOR_GETITEM)
    if dtype_valid:
        return F.gather(data, tensor_index, 0)
    return const_utils.raise_index_error("For 'tensor getitem', "
                                         "the index tensor data type only support mstype.int32.")


def _tensor_index_by_tuple_slice(data, tuple_index):
    """Tensor getitem by a tuple of slice"""
    data_shape = F.shape(data)
    if len(tuple_index) > len(data_shape):
        const_utils.raise_index_error("When tensor is indexed by a tuple, the length of the tuple cannot "
                                      "be greater than the dimension of the tensor.")
    begin_strides, end_strides, step_strides, shrink_axis_mask = \
        const_utils.get_stride_info_from_tuple(data_shape, tuple_index)
    return P.StridedSlice(0, 0, 0, 0, shrink_axis_mask)(data, begin_strides, end_strides, step_strides)


def tensor_index_by_list(data, list_index):
    """Tensor getitem by list of int and bool"""
    data_shape = F.shape(data)
    const_utils.check_list_index_type(list_index)
    list_index = const_utils.transform_list(list_index, data_shape[0])
    tensor_index = const_utils.convert_list_to_tensor(list_index)
    return F.gather(data, tensor_index, 0)


def tensor_index_by_tuple(data, tuple_index):
    """Tensor getitem by tuple of various types with None"""
    if len(tuple_index) == 1:
        return data[tuple_index[0]]
    tuple_index = _transform_ellipsis_to_slice(tuple_index, data, const_utils.TENSOR_GETITEM)
    indexes_types = hyper_map(F.typeof, tuple_index)
    contain_type = const_utils.tuple_index_type_cnt(indexes_types, const_utils.TENSOR_GETITEM)
    if contain_type == const_utils.ALL_TENSOR:
        return _tensor_getitem_by_tuple_of_tensor(data, tuple_index)
    if contain_type == const_utils.ALL_BASIC:
        return _tensor_index_by_tuple_slice(data, tuple_index)
    return _tensor_getitem_by_tuple(data, tuple_index)


def _tensor_setitem(self, index, value):
    """Handle tensor getitem"""
    if isinstance(index, Tensor):
        if isinstance(value, (int, float, bool)):
            return tensor_setitem_by_tensor_with_number(self, index, value)
        if isinstance(value, Tensor):
            return tensor_setitem_by_tensor_with_tensor(self, index, value)
        if isinstance(value, tuple):
            return tensor_setitem_by_tensor_with_tuple(self, index, value)
    if isinstance(index, tuple):
        if isinstance(value, (int, float, bool)):
            return tensor_setitem_by_tuple_with_number(self, index, value)
        if isinstance(value, Tensor):
            return tensor_setitem_by_tuple_with_tensor(self, index, value)
        if isinstance(value, tuple):
            return tensor_setitem_by_tuple_with_tuple(self, index, value)
    if isinstance(index, int):
        if isinstance(value, (int, float, bool)):
            return tensor_setitem_by_number_with_number(self, index, value)
        if isinstance(value, Tensor):
            return tensor_setitem_by_number_with_tensor(self, index, value)
    if isinstance(index, slice):
        if isinstance(value, (int, float, bool)):
            return tensor_setitem_by_slice_with_number(self, index, value)
        if isinstance(value, Tensor):
            return tensor_setitem_by_slice_with_tensor(self, index, value)
    if isinstance(index, bool):
        return _tensor_index_by_bool(self, index)
    if index is ...:
        if isinstance(value, (int, float, bool)):
            return tensor_setitem_by_ellipsis_with_number(self, index, value)
        if isinstance(value, Tensor):
            return tensor_setitem_by_ellipsis_with_tensor(self, index, value)
    raise IndexError("Tensor setitem index only support integers, slices(`:`), ellipsis(`...`), None, bool\
                         and tensor with int32, got {} with type{}".format(index, type(index)))


tensor_operator_registry.register("__setitem__", _tensor_setitem)


def _tensor_setitem_by_int_tensor_with_tensor(data, index, value):
    """Set a tensor item by a int tensor with a tensor."""
    updates = _generate_updates_from_tensor(data, index, value,
                                            const_utils.SET_ITEM_BY_ONE_TENSOR)
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
    updates = _generate_updates_from_scalar(data, index, value,
                                            const_utils.SET_ITEM_BY_ONE_TENSOR)
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


def tensor_setitem_by_tensor_with_tuple(data, index, value):
    """Assigns the tensor by tensor with tuple value."""
    index_dtype = F.dtype(index)
    check_dtype = const_utils.check_index_tensor_dtype(index_dtype, const_utils.TENSOR_SETITEM)
    result = None
    if check_dtype:
        result = _tensor_setitem_by_tensor_with_tuple(data, index, value)
    return result


def _tensor_indices_number(data, data_shape, index, indices, value):
    """Assigns a scalar value to the tensor."""
    data_size = F.size(data)
    data_dtype = F.dtype(data)
    indices_size = F.size(indices)
    indices_size = const_utils.check_indices(indices_size, index)
    update = F.fill(mstype.int32, (indices_size,), 1)
    condition_1d = F.scatter_nd(indices, update, (data_size,))
    condition = F.reshape(condition_1d, data_shape)
    condition = F.cast(condition, mstype.bool_)
    value_fill = F.fill(data_dtype, (indices_size,), value)
    value_1d = F.scatter_nd(indices, value_fill, (data_size,))
    u = F.reshape(value_1d, data_shape)
    return F.select(condition, u, data)


def _tensor_setitem_by_tensor_with_tuple(data, index, value):
    """Set a tensor item by a tensor with a tuple."""
    updates = _generate_updates_from_tuple(data, index, value,
                                           const_utils.SET_ITEM_BY_ONE_TENSOR)
    index = F.expand_dims(index, -1)
    result = P.TensorScatterUpdate()(data, index, updates)
    return result


def tensor_setitem_by_slice_with_number(data, input_slice, value):
    """Givens a scalar assign to tensor by slice"""
    check_result = const_utils.check_tensor_setitem_index(input_slice)
    result = None
    if check_result:
        data_shape = F.shape(data)
        indices = const_utils.slice2indices(input_slice, data_shape)
        is_tuple_int = const_utils.tuple_element_is_int(input_slice)
        if is_tuple_int:
            indices = const_utils.integer_to_indices(input_slice, data_shape)
        result = _tensor_indices_number(data, data_shape, input_slice, indices, value)
    return result


def tensor_setitem_by_tuple_with_number(data, tuple_index, value):
    """Assigns the tensor by tuple with number value."""
    if len(tuple_index) == 1:
        data[tuple_index[0]] = value
        return data
    indexes_types = hyper_map(F.typeof, tuple_index)
    index_elements_type = const_utils.tuple_index_tensor_cnt(indexes_types, const_utils.TENSOR_SETITEM)

    if index_elements_type == const_utils.ALL_TENSOR:
        indices = _generate_indices_from_tuple_of_tensor(data,
                                                         tuple_index,
                                                         const_utils.TENSOR_SETITEM)
    else:
        int_cnt = const_utils.tuple_index_int_cnt(indexes_types, const_utils.TENSOR_SETITEM)
        if int_cnt == const_utils.ALL_INT:
            tuple_index = const_utils.convert_int_to_slice(tuple_index)
        indices = _generate_indices_from_tuple_of_mixed_tensors(data,
                                                                tuple_index,
                                                                const_utils.TENSOR_SETITEM)
    updates = _generate_updates_from_scalar(data,
                                            indices,
                                            value,
                                            const_utils.SET_ITEM_BY_TUPLE_OF_TENSOR)
    return P.TensorScatterUpdate()(data, indices, updates)


def _tensor_indices_tensor(data, data_shape, index, indices, value):
    """Assigns a tensor value to the tensor."""
    data_size = F.size(data)
    data_dtype = F.dtype(data)
    indices_size = F.size(indices)
    indices_size = const_utils.check_indices(indices_size, index)
    update = F.fill(mstype.int32, (indices_size,), 1)
    condition_1d = F.scatter_nd(indices, update, (data_size,))
    condition = F.reshape(condition_1d, data_shape)
    condition = F.cast(condition, mstype.bool_)
    value_fill = None
    value_size = F.size(value)

    value_size = const_utils.check_indices_value_size(indices_size, value_size)
    if value_size == 1:
        value_fill = F.fill(data_dtype, (indices_size,), 1)
        value = F.cast(value, data_dtype)
        value_fill = F.tensor_mul(value_fill, value)
    elif value_size > 1:
        value_fill = F.reshape(value, (indices_size,))
    value_1d = F.scatter_nd(indices, value_fill, (data_size,))
    u = F.reshape(value_1d, data_shape)
    return F.select(condition, u, data)


def tensor_setitem_by_slice_with_tensor(data, input_slice, value):
    """Assigns a tensor value to the tensor by slice."""
    result = None
    check_result = const_utils.check_tensor_setitem_index(input_slice)
    if check_result:
        data_shape = F.shape(data)
        indices = const_utils.slice2indices(input_slice, data_shape)
        is_tuple_int = const_utils.tuple_element_is_int(input_slice)
        if is_tuple_int:
            indices = const_utils.integer_to_indices(input_slice, data_shape)
        result = _tensor_indices_tensor(data, data_shape, input_slice, indices, value)
    return result


def tensor_setitem_by_tuple_with_tensor(data, tuple_index, value):
    """Assigns the tensor by tuple with tensor value."""
    if len(tuple_index) == 1:
        data[tuple_index[0]] = value
        return data
    data_shape = data.shape
    tuple_index_new = ()
    for i, index in enumerate(tuple_index):
        if isinstance(index, mstype.Int):
            if index < -data_shape[i] or index >= data_shape[i]:
                const_utils.raise_index_error("The index is out of the data's special dimension range.")
            elif index < 0:
                tuple_index_new += (tuple_index[i]+data_shape[i],)
            else:
                tuple_index_new += (tuple_index[i],)
        else:
            tuple_index_new += (tuple_index[i],)

    indexes_types = hyper_map(F.typeof, tuple_index_new)
    index_elements_type = const_utils.tuple_index_tensor_cnt(indexes_types, const_utils.TENSOR_SETITEM)

    if index_elements_type == const_utils.ALL_TENSOR:
        indices = _generate_indices_from_tuple_of_tensor(data,
                                                         tuple_index_new,
                                                         const_utils.TENSOR_SETITEM)
    else:
        int_cnt = const_utils.tuple_index_int_cnt(indexes_types, const_utils.TENSOR_SETITEM)
        if int_cnt == const_utils.ALL_INT:
            tuple_index_new = const_utils.convert_int_to_slice(tuple_index_new)
            new_shape = ()
            for _ in tuple_index_new:
                new_shape += (1,)
            new_shape += value.shape
            value = F.reshape(value, new_shape)
        indices = _generate_indices_from_tuple_of_mixed_tensors(data,
                                                                tuple_index_new,
                                                                const_utils.TENSOR_SETITEM)
    updates = _generate_updates_from_tensor(data,
                                            indices,
                                            value,
                                            const_utils.SET_ITEM_BY_TUPLE_OF_TENSOR)
    return P.TensorScatterUpdate()(data, indices, updates)


def tensor_setitem_by_tuple_with_tuple(data, tuple_index, value):
    """Assigns the tensor by tuple with tuple of  value."""
    if len(tuple_index) == 1:
        data[tuple_index[0]] = value
        return data
    indexes_types = hyper_map(F.typeof, tuple_index)
    index_elements_type = const_utils.tuple_index_tensor_cnt(indexes_types, const_utils.TENSOR_SETITEM)

    if index_elements_type == const_utils.ALL_TENSOR:
        indices = _generate_indices_from_tuple_of_tensor(data,
                                                         tuple_index,
                                                         const_utils.TENSOR_SETITEM)
    else:
        int_cnt = const_utils.tuple_index_int_cnt(indexes_types, const_utils.TENSOR_SETITEM)
        if int_cnt == const_utils.ALL_INT:
            tuple_index = const_utils.convert_int_to_slice(tuple_index)
        indices = _generate_indices_from_tuple_of_mixed_tensors(data,
                                                                tuple_index,
                                                                const_utils.TENSOR_SETITEM)
    updates = _generate_updates_from_tuple(data,
                                           indices,
                                           value,
                                           const_utils.SET_ITEM_BY_TUPLE_OF_TENSOR)
    return P.TensorScatterUpdate()(data, indices, updates)


def tensor_setitem_by_number_with_number(data, index, value):
    """Assigns the tensor by number with number value."""
    data_shape = F.shape(data)
    indices = const_utils.integer_to_indices(index, data_shape)
    return _tensor_indices_number(data, data_shape, index, indices, value)


def tensor_setitem_by_number_with_tensor(data, index, value):
    """Assigns the tensor by number with tensor value."""
    data_shape = F.shape(data)
    indices = const_utils.integer_to_indices(index, data_shape)
    return _tensor_indices_tensor(data, data_shape, index, indices, value)


def tensor_setitem_by_ellipsis_with_number(data, index, value):
    """Assigns the tensor by ellipsis with number value."""
    data_shape = F.shape(data)
    data_dtype = F.dtype(data)
    return F.fill(data_dtype, data_shape, value)


def tensor_setitem_by_ellipsis_with_tensor(data, index, value):
    """Assigns the tensor by ellipsis with tensor value."""
    result = None
    data_shape = F.shape(data)
    data_dtype = F.dtype(data)
    data_size = F.size(data)
    value_shape = F.shape(value)
    value_size = F.size(value)
    check_result = const_utils.check_ellipsis_shape_size(data_shape, value_shape, data_size, value_size)
    if check_result:
        if data_size == value_size:
            result = F.reshape(value, data_shape)
            result = F.cast(result, data_dtype)
        elif value_size == 1:
            param1 = F.fill(data_dtype, data_shape, 1)
            param2 = F.cast(value, data_dtype)
            result = F.tensor_mul(param1, param2)
    return result


def tensor_in_sequence(x, y):
    """Assigns whether a sequence contains the given tensor"""
    result = const_utils.scalar_to_tensor(False)
    for i in y:
        if isinstance(i, mstype.tensor) and x.shape == i.shape and x.dtype == i.dtype:
            result = F.logical_or(F.equal(x, i).all(), result)
    return result
