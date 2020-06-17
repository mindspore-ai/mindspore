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
pack = P.Pack(axis=-1)


def broadcast(broadcast_shape, x):
    """Broadcast tensor to the required shape."""
    if F.shape(x) == broadcast_shape:
        return x
    multiples = const_utils.compute_multiples(F.shape(x), broadcast_shape)
    if multiples:
        return F.tile(x, multiples)
    return x


def transform_indexing_tensor(broadcast_shape, final_shape, new_shape, x):
    """Transform indexing tensor to the required."""
    x = broadcast(broadcast_shape, x)
    return broadcast(final_shape, F.reshape(x, new_shape))


def generate_indices_from_tuple_of_tensor(data, tuple_index, op_name):
    """Generate an indices tensor from a tuple of tensor."""
    indices = None
    check_index_tensor_number = const_utils.check_number_of_index_tensor(F.shape(data), len(tuple_index), op_name)
    if check_index_tensor_number:
        dtype_tuple = hyper_map(F.dtype, tuple_index)
        check_dtypes = const_utils.check_index_tensors_dtype(dtype_tuple, op_name)
        if check_dtypes:
            shape_tuple = hyper_map(F.shape, tuple_index)
            broadcast_shape = const_utils.generate_broadcast_shape(shape_tuple, op_name)
            broadcast_tensors = hyper_map(F.partial(broadcast, broadcast_shape), tuple_index)
            indices = pack(broadcast_tensors)
    return indices


def generate_indices_from_tuple_of_mixed_tensors(data, tuple_index, op_name):
    """Generate an indices tensor from a tuple that contains slice, int, ellipsis, tensor."""
    indexes_types = hyper_map(F.typeof, tuple_index)
    int_positions = const_utils.get_pos_of_int_index(indexes_types)
    for i in int_positions:
        tuple_index = F.tuple_setitem(tuple_index, i, F.scalar_to_tensor(tuple_index[i], mstype.int32))
    indexes_types = hyper_map(F.typeof, tuple_index)
    tensor_positions, slice_positions, ellipsis_position = \
        const_utils.separate_mixed_tensors_index(indexes_types, op_name)
    tensor_indexes = []
    slice_indexes = []
    for i in tensor_positions:
        tensor_indexes.append(tuple_index[i])
    for j in slice_positions:
        slice_indexes.append(tuple_index[j])
    data_shape = F.shape(data)
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
    tuple_index_size = len(tuple_index)
    index_tensor_new_shape = const_utils.compute_new_shape(broadcast_shape, indexes_shapes_info)
    for i in range(tuple_index_size):
        if i in tensor_positions:
            transform_tensor = transform_indexing_tensor(broadcast_shape,
                                                         final_shape,
                                                         index_tensor_new_shape,
                                                         tuple_index[i])
            final_index_tensors.append(transform_tensor)
        if i in slice_positions:
            slice_tensor = const_utils.convert_slice_to_tensor(slice_number,
                                                               final_shape,
                                                               indexes_shapes_info,
                                                               op_name)
            final_index_tensors.append(slice_tensor)
            slice_number += 1
        if i == ellipsis_position:
            ellipsis_tensors = const_utils.convert_ellipsis_to_tensors(slice_number,
                                                                       ellipsis_occupied_dims,
                                                                       final_shape,
                                                                       indexes_shapes_info,
                                                                       op_name)
            for ele in ellipsis_tensors:
                final_index_tensors.append(ele)
            slice_number += ellipsis_occupied_dims
    indices = pack(final_index_tensors)
    return indices


def generate_updates_from_scalar(data, indices, value, op_type):
    """Generate an updates tensor from a scalar."""
    data_shape = F.shape(data)
    indices_shape = F.shape(indices)
    data_dtype = F.dtype(data)
    return const_utils.convert_scalar_to_tensor(data_shape, data_dtype, indices_shape, value, op_type)


def generate_updates_from_tuple(data, index, value, op_type):
    """Generate an updates tensor from a tuple."""
    value_types = hyper_map(F.typeof, value)
    data_dtype = F.dtype(data)
    value_elements_type = const_utils.check_value_elements(data_dtype, value_types)
    if value_elements_type == const_utils.ALL_TENSOR:
        value_shapes = hyper_map(F.shape, value)
        shapes_same = const_utils.check_shapes_same(value_shapes, const_utils.TENSOR_SETITEM)
        if shapes_same:
            value = F.pack(value)
        return generate_updates_from_tensor(data, index, value, op_type)

    data_shape = F.shape(data)
    index_shape = F.shape(index)
    return const_utils.convert_tuple_of_scalar_to_tensor(data_shape, data_dtype, index_shape, value, op_type)


def generate_updates_from_tensor(data, index, value, op_type):
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
        return broadcast(updates_shape, value)
    return value


def tensor_getitem(self, index):
    """Handle tensor getitem"""
    if isinstance(index, Tensor):
        return tensor_index_by_tensor(self, index)
    if isinstance(index, tuple):
        return tensor_index_by_tuple(self, index)
    if isinstance(index, int):
        return tensor_index_by_integer(self, index)
    if isinstance(index, slice):
        return tensor_index_by_slice(self, index)
    if isinstance(index, bool):
        return tensor_index_by_bool(self, index)
    if index is ...:
        return self
    raise IndexError(f"Only support integers, slices(`:`), ellipsis(`...`), None, bool and tensor with int32, "
                     f"got {index} with type {type(index)}.")


tensor_operator_registry.register("__getitem__", tensor_getitem)


def tensor_getitem_by_tuple_of_tensor(data, tuple_index):
    """Tensor getitem by a tuple of tensor."""
    indices = generate_indices_from_tuple_of_tensor(data,
                                                    tuple_index,
                                                    const_utils.TENSOR_GETITEM)
    result = F.gather_nd(data, indices)
    return result


def tensor_getitem_by_tuple_of_mixed_tensors(data, tuple_index):
    """Tensor getitem by a tuple of mixed tensor."""
    indices = generate_indices_from_tuple_of_mixed_tensors(data,
                                                           tuple_index,
                                                           const_utils.TENSOR_GETITEM)
    result = F.gather_nd(data, indices)
    return result


def tensor_index_by_slice(data, slice_index):
    """Tensor getitem by a single slice"""
    shape = F.shape(data)
    if not shape:
        const_utils.raise_index_error("When tensor is indexed by a slice, the dimension of the tensor cannot be 0.")
    begin_strides, end_strides, step_strides = const_utils.get_stride_info_from_slice(shape, slice_index)
    return F.strided_slice(data, begin_strides, end_strides, step_strides)


def tensor_index_by_integer(data, number):
    """Tensor getitem by a single integer number"""
    shape = F.shape(data)
    if not shape:
        const_utils.raise_index_error("When tensor is indexed by an integer, the dimension of the tensor cannot be 0.")
    begin_strides, end_strides, step_strides = const_utils.get_stride_info_from_integer(shape, number)
    shrink_axis_mask = 1
    return P.StridedSlice(0, 0, 0, 0, shrink_axis_mask)(data, begin_strides, end_strides, step_strides)


def tensor_index_by_bool(data, bool_value):
    """Tensor getitem by a single bool value"""
    if bool_value:
        return F.expand_dims(data, 0)
    return const_utils.raise_index_error("When tensor is indexed by a bool object, the value only support 'True'.")


def tensor_index_by_number(data, number):
    """Tensor getitem by a Number which may be integer/float/bool value"""
    number_type = const_utils.check_number_index_type(number)
    if number_type == const_utils.BOOL_:
        return tensor_index_by_bool(data, number)
    if number_type == const_utils.INT_:
        return tensor_index_by_integer(data, number)
    return const_utils.raise_index_error("Only support integers, slices(`:`), ellipsis(`...`), None and bool.")


def tensor_index_by_tensor(data, tensor_index):
    """Tensor getitem by a single tensor"""
    dtype_valid = const_utils.check_index_tensor_dtype(F.dtype(tensor_index),
                                                       const_utils.TENSOR_GETITEM)
    if dtype_valid:
        return F.gather(data, tensor_index, 0)
    return const_utils.raise_index_error("For 'tensor getitem', "
                                         "the index tensor data type only support mstype.int32.")


def tensor_index_by_tuple_slice(data, t):
    """Tensor getitem by a tuple of slice"""
    shape = F.shape(data)
    if len(t) > len(shape):
        const_utils.raise_index_error("When tensor is indexed by a tuple, "
                                      "the length of the tuple cannot be greater than the dimension of the tensor.")
    begin_strides, end_strides, step_strides, shrink_axis_mask = \
        const_utils.get_stride_info_from_tuple(shape, t)
    return P.StridedSlice(0, 0, 0, 0, shrink_axis_mask)(data, begin_strides, end_strides, step_strides)


def tensor_index_by_tuple(data, tuple_index):
    """Tensor getitem by tuple of various types"""
    indexes_types = hyper_map(F.typeof, tuple_index)
    index_elements_type = const_utils.tuple_index_elements_type(indexes_types, const_utils.TENSOR_GETITEM)
    if index_elements_type == const_utils.NO_TENSOR:
        return tensor_index_by_tuple_slice(data, tuple_index)
    if index_elements_type == const_utils.ALL_TENSOR:
        return tensor_getitem_by_tuple_of_tensor(data, tuple_index)
    return tensor_getitem_by_tuple_of_mixed_tensors(data, tuple_index)
