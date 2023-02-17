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

"""constexpr util"""
from __future__ import absolute_import

from mindspore.ops.composite.multitype_ops import _constexpr_utils as const_utils
from mindspore.ops import functional as F
from mindspore.ops import operations as P
from mindspore.ops.composite import base
from mindspore.ops._primitive_cache import _get_cache_prim
from mindspore.ops.operations._inner_ops import TensorCopySlices, SliceGetItem, \
    TopTypeof, issubclass_, IsParameter
from mindspore.common import dtype as mstype
from mindspore.common._register_for_tensor import tensor_operator_registry
from mindspore.common import Tensor, CSRTensor, COOTensor
from mindspore.common import mutable
from mindspore import ops

slice_get_item = SliceGetItem()
hyper_map = base.HyperMap()
stack = P.Stack(axis=-1)
copy_slice = TensorCopySlices()
toptypeof = TopTypeof()
is_parameter = IsParameter()


def strided_slice(data, begin_strides, end_strides, step_strides, begin_mask=0, end_mask=0, ellipsis_mask=0,
                  new_axis_mask=0, shrink_axis_mask=0):
    """strided_slice primitive cache"""
    strided_slice_ = _get_cache_prim(P.StridedSlice)(begin_mask, end_mask, ellipsis_mask, new_axis_mask,
                                                     shrink_axis_mask)
    return strided_slice_(data, begin_strides, end_strides, step_strides)


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
        index = format_list_indices(index, F.shape(self)[0])
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
    if isinstance(other, COOTensor):
        return other + self
    return F.add(self, other)


def _tensor_sub(self, other):
    if isinstance(self, (tuple, list)):
        self = sequence_to_tensor(self, F.dtype(other))
    if isinstance(other, (tuple, list)):
        other = sequence_to_tensor(other, F.dtype(self))
    if isinstance(other, COOTensor):
        return F.tensor_scatter_sub(self, other.indices, other.values)
    return F.sub(self, other)


def _tensor_mul(self, other):
    if isinstance(other, (tuple, list)):
        other = sequence_to_tensor(other, F.dtype(self))
    elif isinstance(other, (CSRTensor, COOTensor)):
        return other * self
    return F.mul(self, other)


def _tensor_matmul(self, other):
    return F.matmul(self, other)


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
    if isinstance(other, (tuple, list)):
        other = sequence_to_tensor(other, F.dtype(self))
    return F.tensor_pow(self, other)


def _tensor_rpow(self, other):
    if isinstance(other, (tuple, list)):
        other = sequence_to_tensor(other, F.dtype(self))
    return F.tensor_pow(other, self)


def _tensor_floordiv(self, other):
    if isinstance(self, (tuple, list)):
        self = sequence_to_tensor(self, F.dtype(other))
    if isinstance(other, (tuple, list)):
        other = sequence_to_tensor(other, F.dtype(self))
    return F.floordiv(self, other)


tensor_operator_registry.register('__add__', _tensor_add)
tensor_operator_registry.register('__sub__', _tensor_sub)
tensor_operator_registry.register('__mul__', _tensor_mul)
tensor_operator_registry.register('__matmul__', _tensor_matmul)
tensor_operator_registry.register('__truediv__', _tensor_div)
tensor_operator_registry.register('__mod__', _tensor_mod)
tensor_operator_registry.register('__pow__', _tensor_pow)
tensor_operator_registry.register('__rpow__', _tensor_rpow)
tensor_operator_registry.register('__floordiv__', _tensor_floordiv)


def _scalar_to_tensor(input_x):
    if ops.isconstant(input_x):
        return P.ScalarToTensor()(input_x, ops.dtype(input_x))
    # use add Tensor([0]) cast scalar to tensor.
    return ops.add(input_x, mutable(Tensor(0)))


def tensor_item(data, *args):
    """Tensor getitem by index whose dtype is int or tuple with int."""
    # transform a.item(tuple(int)) -> a.item(int1,int2...intN)
    if len(args) == 1 and isinstance(args[0], tuple):
        args = args[0]

    args_types = hyper_map(F.typeof, args)
    if not args or const_utils.judge_index_type(args_types[0], mstype.type_none):
        if data.shape == (1,):
            return data[0]
        const_utils.raise_value_error("Can only convert an array of size 1 to a Tensor scalar")

    if not const_utils.judge_indexes_types(args_types, mstype.int64):
        const_utils.raise_type_error("The index object cannot be interpreted as an integer")

    if len(args) == data.ndim:
        return _tensor_getitem_by_tuple_slice(data, args)
    if len(args) > 1:
        const_utils.raise_value_error("Incorrect number of indices for array")
    return _tensor_index_by_integer(F.reshape(data, (-1,)), args[0])


def tensor_itemset(data, *args):
    """Tensor setitem by index and value."""
    if not args:
        const_utils.raise_value_error("'Tensor.itemset()' must have at least one argument, but got None.")
    if len(args) == 2:
        if const_utils.judge_index_type(F.typeof(args[0]), mstype.int64):
            return tensor_itemset_by_number_with_number(data, args[0], args[1])
        if isinstance(args[0], tuple):
            return tensor_itemset_by_tuple_with_number(data, args[0], args[1])
        const_utils.raise_type_error("The index object cannot be interpreted as an integer")
    if len(args) > 2:
        exp_msg = const_utils.gen_exception_msg("'Tensor.itemset()' must have at most 2 argument, but got {}.",
                                                len(args))
        const_utils.raise_value_error(exp_msg)
    return tensor_itemset_with_number(data, args[0])


tensor_operator_registry.register("item", tensor_item)
tensor_operator_registry.register("itemset", tensor_itemset)


def tensor_itemset_with_number(data, number_value):
    """set value of tensor whose shape is (1,)"""
    if not const_utils.judge_index_type(F.typeof(number_value), mstype.number_type):
        exp_msg = const_utils.gen_exception_msg(
            "'Tensor.itemset()' only support number input, but got {}", number_value)
        const_utils.raise_index_error(exp_msg)
    if data.shape != (1,):
        exp_msg = const_utils.gen_exception_msg(
            "Only tensor which shape is (1,) support 1 arg that means omit index, "
            "but the tensor shape is {} and got 1 input.", data.shape)
        const_utils.raise_index_error(exp_msg)
    return const_utils.make_tensor((number_value,), F.dtype(data))


def tensor_itemset_by_number_with_number(data, int_index, number_value):
    flatten_data = F.reshape(data, (-1,))
    itemset_data = tensor_setitem_by_number_with_number(flatten_data, int_index, number_value)
    res_data = F.reshape(itemset_data, F.shape(data))
    return res_data


def tensor_itemset_by_tuple_with_number(data, tuple_index, nubmer_value):
    if len(tuple_index) != data.ndim:
        exp_msg = const_utils.gen_exception_msg(
            "Tuple index len({}) is not same to tensor dimension({})", len(tuple_index), data.ndim)
        const_utils.raise_index_error(exp_msg)
    return tensor_setitem_by_tuple_with_number(data, tuple_index, nubmer_value)


def _broadcast(broadcast_shape, x):
    """Broadcast tensor to the required shape."""
    if F.shape(x) == broadcast_shape:
        return x
    return F.broadcast_to(x, broadcast_shape)


def _transform_indexing_tensor(broadcast_shape, final_shape, new_shape, item):
    """Transform indexing tensor to the required."""
    item = _broadcast(broadcast_shape, item)
    return _broadcast(final_shape, F.reshape(item, new_shape))


def _transform_ellipsis_to_slice(data, tuple_index, op_name):
    """
    Check if the tuple index len is longer than the data's dims and transform ellipsis in the indices
    to several slice.
    """
    data_shape = F.shape(data)
    indexes_types = hyper_map(toptypeof, tuple_index)
    slice_positions, ellipsis_positions, _, int_positions, _, tensor_positions, sequence_positions = \
        const_utils.get_pos_of_indexes_types(indexes_types, op_name)

    ellipsis_occupy_dims = data.ndim - (len(slice_positions) + len(int_positions) +
                                        len(tensor_positions) + len(sequence_positions))
    ellipsis_cnt = len(ellipsis_positions)

    if ellipsis_occupy_dims < 0:
        if ellipsis_cnt >= 0:
            exp_msg = const_utils.gen_exception_msg(
                "Tuple index {} out rang of tensor shape {}.", tuple_index, data_shape)
            const_utils.raise_index_error(exp_msg)

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
    indexes_types = hyper_map(toptypeof, tuple_index)
    expand_positions, tuple_index_new = (), ()
    for i, (index, index_type) in enumerate(zip(tuple_index, indexes_types)):
        if isinstance(index_type, mstype.none_type):
            tuple_index_new += (const_utils.make_empty_slice(),)
            expand_positions += (i,)
        elif isinstance(index_type, mstype.Bool):
            if not index:
                const_utils.raise_index_error("Bool element of tuple index must be 'True', but got 'False'.")
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
    is_dynamic = (F.is_sequence_value_unknown(data_shape)
                  or isinstance(slice_get_item(slice_index, "start"), Tensor)
                  or isinstance(slice_get_item(slice_index, "stop"), Tensor)
                  or isinstance(slice_get_item(slice_index, "step"), Tensor))
    if is_dynamic:
        begin_strides, end_strides, step_strides = get_stride_info_from_slice(data, slice_index)
    else:
        begin_strides, end_strides, step_strides = const_utils.get_stride_info_from_slice(data_shape, slice_index)
    begin_mask = 1 if slice_get_item(slice_index, "start") is None else 0
    end_mask = 1 if slice_get_item(slice_index, "stop") is None else 0
    for i in range(1, len(data_shape)):
        begin_mask += 2 ** i
        end_mask += 2 ** i
    if begin_mask or end_mask:
        return strided_slice(data, begin_strides, end_strides, step_strides, begin_mask, end_mask, 0, 0, 0)
    return F.strided_slice(data, begin_strides, end_strides, step_strides)


def get_stride_info_from_slice(data, slice_index):
    """get the stride info from slice index"""
    data_shape = F.dyn_shape(data)
    begin_strides, end_strides, step_strides = [], [], []
    start, stop, step = get_slice_stride(slice_index, data_shape[0])
    if start.ndim > 0:
        start = start.item()
    if stop.ndim > 0:
        stop = stop.item()
    if step.ndim > 0:
        step = step.item()
    begin_strides.append(start)
    end_strides.append(stop)
    step_strides.append(step)
    begin_tensor = stack(begin_strides)
    end_tensor = stack(end_strides)
    step_tensor = stack(step_strides)
    return begin_tensor, end_tensor, step_tensor


def tensor_index_by_number(data, number_index):
    """Tensor getitem by a Number which may be integer/float/bool value"""
    if isinstance(number_index, bool):
        return _tensor_index_by_bool(data, number_index)
    if isinstance(number_index, int):
        return _tensor_index_by_integer(data, number_index)
    exp_msg = const_utils.gen_exception_msg(
        "Number index of tensor must be int or bool, but got {}.", number_index)
    return const_utils.raise_index_error(exp_msg)


def _tensor_index_by_bool(data, bool_value):
    """Tensor getitem by a single bool value"""
    min_data_dim, max_data_dim = 0, 7
    const_utils.judge_data_dim(data.ndim, min_data_dim, max_data_dim)
    if bool_value:
        return F.expand_dims(data, 0)
    return const_utils.raise_index_error("When tensor is indexed by a bool object, the value only support 'True'.")


def get_stride_info_from_integer(tensor_int):
    """Convert integer to slice"""
    begin_strides = [tensor_int]
    end_strides = [tensor_int + 1]
    step_strides = [const_utils.make_tensor(1)]
    begin_tensor = stack(begin_strides)
    end_tensor = stack(end_strides)
    step_tensor = stack(step_strides)
    return begin_tensor, end_tensor, step_tensor


def _tensor_index_by_integer(data, int_index):
    """Tensor getitem by a single integer number"""
    data_shape = F.shape(data)
    if not data_shape:
        const_utils.raise_type_error("Cannot iterate over a scalar tensor.")
    if data.ndim < 1 or data.ndim > 8:
        const_utils.raise_value_error("Expect Tensor to have dimension between 1 and 8.")

    if F.is_sequence_value_unknown(data_shape) or not F.isconstant(int_index):
        tensor_index = _scalar_to_tensor(int_index)
        begin_strides, end_strides, step_strides = get_stride_info_from_integer(tensor_index)
    else:
        transformed_number = const_utils.check_range(int_index, data_shape[0])
        begin_strides, end_strides, step_strides = \
            const_utils.get_stride_info_from_integer(data_shape, transformed_number)
    shrink_axis_mask = 1
    begin_mask = 0
    end_mask = 0
    for i in range(2, 8):
        begin_mask += 2 ** i
        end_mask += 2 ** i
    return strided_slice(data, begin_strides, end_strides, step_strides, begin_mask, end_mask, 0, 0, shrink_axis_mask)


def tensor_index_by_tensor(data, tensor_index):
    """Tensor getitem by a single tensor"""
    min_data_dim, max_data_dim = 0, 7
    const_utils.judge_data_dim(data.ndim, min_data_dim, max_data_dim)
    valid = const_utils.check_type_isinstance(F.dtype(tensor_index), mstype.Int)
    if valid is False:
        exp_msg = const_utils.gen_exception_msg(
            "The tensor index must be int type, but got {}.", F.dtype(tensor_index))
        const_utils.raise_index_error(exp_msg)
    return F.gather(data, tensor_index, 0)


def tensor_index_by_list(data, list_index):
    """Tensor getitem by list of int and bool"""
    min_data_dim, max_data_dim = 1, 8
    const_utils.judge_data_dim(data.ndim, min_data_dim, max_data_dim)

    data_shape = F.shape(data)
    indexes_types = hyper_map(toptypeof, list_index)
    if const_utils.check_type_isinstance(indexes_types, (mstype.Bool, mstype.Int)):
        if not F.isconstant(data_shape[0]):
            if all(isinstance(i, bool) for i in list_index):
                const_utils.raise_unimplemented_error(
                    "Not supported to the dynamic shape tensor slice by using list of Boolean type")
            tensor_index = const_utils.sequence_to_index(list_index, None)
        else:
            tensor_index = const_utils.sequence_to_index(list_index, data_shape[0])
        if tensor_index is False:
            const_utils.raise_index_error("When tensor is indexed by list, the list can't be empty.")
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
    indexes_types = hyper_map(toptypeof, tuple_index)
    contain_type = const_utils.tuple_index_type_cnt(indexes_types, op_name)
    if contain_type == const_utils.ALL_BASIC:
        return _tensor_getitem_by_tuple_slice(data, tuple_index)
    return _tensor_getitem_by_tuple(data, tuple_index, op_name)


def get_slice_stride(slice_index, dim_size):
    """Get slice stride info"""
    start = slice_get_item(slice_index, "start")
    stop = slice_get_item(slice_index, "stop")
    step = slice_get_item(slice_index, "step")

    if start is None:
        start = const_utils.make_tensor(0)
    if stop is None:
        stop = dim_size
    if step is None:
        step = const_utils.make_tensor(1)

    if issubclass_(F.typeof(start), mstype.number):
        start = const_utils.make_tensor(start)

    if issubclass_(F.typeof(stop), mstype.number):
        stop = const_utils.make_tensor(stop)

    if issubclass_(F.typeof(step), mstype.number):
        step = const_utils.make_tensor(step)

    return start, stop, step


def cal_tuple_slice_mask(data_shape, tuple_index):
    """calculate the strided_slice begin and end mask"""
    begin_mask = 0
    end_mask = 0
    for i, slice_index in enumerate(tuple_index):
        if isinstance(slice_index, slice):
            begin_mask += 2 ** i if slice_get_item(slice_index, "start") is None else 0
            end_mask += 2 ** i if slice_get_item(slice_index, "stop") is None else 0
    for i in range(len(tuple_index), len(data_shape)):
        begin_mask += 2 ** i
        end_mask += 2 ** i
    return begin_mask, end_mask


def _get_stride_info_from_tuple(data, tuple_index):
    """get the stride info from tuple"""
    data_shape = F.dyn_shape(data)
    begin_strides, end_strides, step_strides = [], [], []
    tuple_index_len = len(tuple_index)
    data_dim = data.ndim
    shrink_axis, index_count, ellipsis_count = 0, 0, 0
    for item in range(data_dim):
        if item >= tuple_index_len or item >= data_dim:
            break
        index = tuple_index[item]
        dim_size = data_shape[item]
        if isinstance(index, slice):
            start, stop, step = get_slice_stride(index, dim_size)
            begin_strides.append(start)
            end_strides.append(stop)
            step_strides.append(step)
            index_count = index_count + 1
        elif isinstance(index, int):
            int_tensor = _scalar_to_tensor(index)
            begin_strides.append(int_tensor)
            end_strides.append(int_tensor + const_utils.make_tensor(1))
            step_strides.append(const_utils.make_tensor(1))
            shrink_axis = shrink_axis + (2 ** index_count)
            index_count = index_count + 1
        elif index is ...:
            ellipsis_count = ellipsis_count + 1
            if ellipsis_count > 1:
                const_utils.raise_value_error("An index can have only one ellipsis (...)")
            ellipsis_range_size = data_dim - tuple_index_len + 1
            begin_strides.extend([const_utils.make_tensor(0)] * ellipsis_range_size)
            end_strides.extend(
                [shape for shape in data_shape[index_count: index_count + ellipsis_range_size]])
            step_strides.extend([const_utils.make_tensor(1)] * ellipsis_range_size)
            index_count = index_count + ellipsis_range_size
        else:
            exp_msg = const_utils.gen_exception_msg("Not supported index data type, got {},  type is {}", index,
                                                    type(index))
            const_utils.raise_index_error(exp_msg)
    begin_tensor = stack(begin_strides)
    end_tensor = stack(end_strides)
    step_tensor = stack(step_strides)
    strides_v = {
        'begin': begin_tensor,
        'end': end_tensor,
        'step': step_tensor
    }
    return strides_v, shrink_axis


def _tensor_getitem_by_tuple_slice(data, tuple_index):
    """Tensor getitem by a tuple of slice"""
    data_shape = F.shape(data)
    is_dynamic = F.is_sequence_value_unknown(data_shape)
    for item in tuple_index:
        if isinstance(item, slice):
            is_dynamic = is_dynamic or isinstance(slice_get_item(item, "start"), Tensor) \
                         or isinstance(slice_get_item(item, "stop"), Tensor) \
                         or isinstance(slice_get_item(item, "step"), Tensor)

    strides_v = {}
    shrink_axis_mask = 0
    if not is_dynamic:
        strides_v, shrink_axis_mask = const_utils.get_stride_info_from_tuple(
            data_shape, tuple_index)
    else:
        strides_v, shrink_axis_mask = _get_stride_info_from_tuple(
            data, tuple_index)
    begin_mask, end_mask = cal_tuple_slice_mask(data_shape, tuple_index)
    begin_v = strides_v['begin']
    end_v = strides_v['end']
    step_v = strides_v['step']
    return strided_slice(data, begin_v, end_v, step_v, begin_mask, end_mask, 0, 0, shrink_axis_mask)


def _tensor_getitem_by_tuple(data, tuple_index, op_name):
    """Tensor getitem by a tuple of mixed tensor."""
    slice_is_tensor = False
    for item in tuple_index:
        if isinstance(item, slice):
            slice_is_tensor = isinstance(slice_get_item(item, "start"), Tensor) \
                              or isinstance(slice_get_item(item, "stop"), Tensor) \
                              or isinstance(slice_get_item(item, "step"), Tensor)
    if slice_is_tensor:
        const_utils.raise_index_error("Not supported when slice has tensor")
    tuple_index_len = len(tuple_index)
    tensor_indexes, slice_indexes = [], []
    indexes_types = hyper_map(toptypeof, tuple_index)
    slice_positions, _, _, int_positions, _, tensor_positions, sequence_positions = \
        const_utils.get_pos_of_indexes_types(indexes_types, op_name)
    tuple_index_new, slice_shapes = (), ()
    data_shape = F.shape(data)
    for i, (index, dim_size) in enumerate(zip(tuple_index, data_shape)):
        if i in int_positions:
            int_index = const_utils.check_range(index, dim_size)
            tensor_index = F.scalar_to_tensor(int_index, mstype.int64)
            if F.is_sequence_value_unknown(data_shape):
                tensor_index = _scalar_to_tensor(int_index)
                tensor_index = F.cast(tensor_index, mstype.int64)
            tuple_index_new += (tensor_index,)
            tensor_indexes.append(tensor_index)
            tensor_positions += (i,)
        elif i in sequence_positions:
            tensor_index = const_utils.sequence_to_index(index, dim_size)
            if tensor_index is False:
                const_utils.raise_index_error("The sequence element(tuple/list) in tuple index can't be empty.")
            tuple_index_new += (tensor_index,)
            tensor_indexes.append(tensor_index)
            tensor_positions += (i,)
        elif i in tensor_positions:
            invalid = const_utils.check_type_invalid(F.dtype(index), mstype.int_type)
            if invalid:
                exp_msg = const_utils.gen_exception_msg(
                    "The tensor element in tuple index must be int type, but got {}.", F.dtype(index))
                const_utils.raise_index_error(exp_msg)
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
        if tuple_index_len < len(data_shape):
            final_shape = final_shape + data_shape[tuple_index_len:]
        return const_utils.make_tensor([], data.dtype, final_shape)

    final_index_tensors = []
    slice_cnt = 0
    for i, index in enumerate(tuple_index_new):
        if i in tensor_positions:
            transform_tensor = _transform_indexing_tensor(broadcast_shape, final_shape, index_tensor_new_shape,
                                                          index)
            final_index_tensors.append(transform_tensor)
        elif i in slice_positions:
            slice_index_tensor = convert_slice_to_tensor(index, final_shape, slice_cnt, broadcast_shape,
                                                         slice_shapes, fancy_position)
            final_index_tensors.append(slice_index_tensor)
            slice_cnt += 1

    indices = stack(final_index_tensors)
    result = F.gather_nd(data, indices)
    return result


def _generate_indices_from_tuple_of_tensor(tuple_index, op_name):
    """Generate an indices tensor from a tuple of tensor."""
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


def _generate_indices_from_tuple(data, tuple_index, op_name, fancy_position):
    """Generate an indices tensor from a tuple that contains slice, int, ellipsis, tensor."""
    data_shape = F.shape(data)
    tensor_indexes, slice_indexes = [], []
    indexes_types = hyper_map(toptypeof, tuple_index)
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
            invalid = const_utils.check_type_invalid(F.dtype(index), mstype.int_type)
            if invalid:
                exp_msg = const_utils.gen_exception_msg(
                    "The tensor element in tuple index must be int type, but got {}.", F.dtype(index))
                const_utils.raise_index_error(exp_msg)
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
                                                                    slice_shapes, op_name, fancy_position)

    final_index_tensors = []
    slice_cnt = 0
    for i, index in enumerate(tuple_index_new):
        if i in tensor_positions:
            transform_tensor = _transform_indexing_tensor(broadcast_shape, final_shape, index_tensor_new_shape,
                                                          index)
            final_index_tensors.append(transform_tensor)
        elif i in slice_positions:
            slice_index_tensor = convert_slice_to_tensor(index, final_shape, slice_cnt, broadcast_shape,
                                                         slice_shapes, fancy_position)
            final_index_tensors.append(slice_index_tensor)
            slice_cnt += 1

    indices = stack(final_index_tensors)
    return indices


def sequence_to_tensor(value, dtype):
    """Generate an updates tensor from a tuple, can only handle 1-D tensor/non-tensor mixtures."""
    value_types = hyper_map(toptypeof, value)
    value_elements_type = const_utils.check_value_elements(value_types)

    if value_elements_type == const_utils.ALL_TENSOR:
        value = F.stack(value).astype(dtype)
    elif value_elements_type == const_utils.NO_TENSOR:
        value = const_utils.make_tensor(value, dtype)
    else:
        new_value = ()
        for ele in value:
            ele = ele if isinstance(ele, Tensor) else const_utils.make_tensor(ele, dtype)
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
    if F.is_sequence_value_unknown(F.shape(data)):
        data_shape = F.dyn_shape(data)
        index_shape = F.dyn_shape(index)
        updates_shape = const_utils.generate_updates_shape(data_shape, index_shape, op_type, True)
        updates = ops.broadcast_to(value, updates_shape)
        return updates
    updates_shape = const_utils.generate_updates_shape(data.shape, index.shape, op_type, False)
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
    """Set a tensor item by an int tensor with a tensor."""
    if F.rank(index) == 0:
        index = F.expand_dims(index, -1)
    updates = _generate_updates_from_tensor(data, index, value, const_utils.SET_ITEM_BY_ONE_TENSOR)
    data_shape = F.shape(data)
    first_val = data_shape[0]
    if not F.isconstant(first_val):
        first_val = -1
    index = F.select(index < 0, index + first_val, index)
    index = F.expand_dims(index, -1)
    if F.rank(index) < 2:
        index = F.expand_dims(index, 0)
        updates = F.expand_dims(updates, 0)
    if is_parameter(data):
        return F.scatter_nd_update(data, index, updates)
    return F.tensor_scatter_update(data, index, updates)


def _tensor_setitem_by_bool_tensor_with_tensor(data, index, value):
    """Set a tensor item by a bool tensor with a tensor."""
    index = index.reshape(const_utils.generate_padding_shape(index.shape, len(data.shape)))
    index = F.broadcast_to(index, data.shape)
    value = F.cast(value, F.dtype(data))
    value = value.reshape(const_utils.generate_padding_shape(value.shape, len(data.shape)))
    value = F.broadcast_to(value, data.shape)
    result = F.select(index, value, data)
    return result


def tensor_setitem_by_tensor_with_tensor(data, index, value_tensor):
    """setitem by tensor index(dtype is int or bool) with tensor as value"""
    index_dtype = F.dtype(index)
    tensor_dtype = const_utils.get_index_tensor_dtype(index_dtype)
    if tensor_dtype == const_utils.INT_:
        return _tensor_setitem_by_int_tensor_with_tensor(data, index, value_tensor)

    if F.is_sequence_value_unknown(F.shape(data)):
        const_utils.raise_unimplemented_error(
            "Not supported to the dynamic shape tensor slice by using tensor of Boolean type")
    return _tensor_setitem_by_bool_tensor_with_tensor(data, index, value_tensor)


def tensor_setitem_by_tensor_with_number(data, index, value):
    value = F.fill(F.dtype(data), (), value)
    return tensor_setitem_by_tensor_with_tensor(data, index, value)


def tensor_setitem_by_tensor_with_sequence(data, index, value):
    """Assigns the tensor by tensor with tuple value."""
    index_dtype = F.dtype(index)
    invalid = const_utils.check_type_invalid(index_dtype, (mstype.int32, mstype.int64))
    if invalid:
        exp_msg = const_utils.gen_exception_msg("The tensor index must be int type, but got {}.", index_dtype)
        const_utils.raise_index_error(exp_msg)
    return _tensor_setitem_by_tensor_with_sequence(data, index, value)


def _tensor_setitem_by_tensor_with_sequence(data, index, value):
    """Set a tensor item by a tensor with a tuple."""
    updates = _generate_updates_from_sequence(data, index, value, const_utils.SET_ITEM_BY_ONE_TENSOR)
    index = F.expand_dims(index, -1)
    return F.tensor_scatter_update(data, index, updates)


def tensor_setitem_by_slice_with_number(data, input_slice, value):
    """Givens a scalar assign to tensor by slice"""
    value = F.fill(F.dtype(data), (), value)
    return tensor_setitem_by_slice_with_tensor(data, input_slice, value)


def tensor_setitem_by_tuple_with_number(data, tuple_index, value):
    """Assigns the tensor by tuple with number value."""
    value = F.fill(F.dtype(data), (), value)
    return tensor_setitem_by_tuple_with_tensor(data, tuple_index, value)


def tensor_copy_slice_from_slice(data, input_slice, value):
    """using TensorCopySlices by slice."""
    data_shape = F.dyn_shape(data)
    start, stop, step = get_slice_stride(input_slice, data_shape[0])
    start_tensor = stack((start,))
    stop_tensor = stack((stop,))
    step_tensor = stack((step,))
    dim0_size = stop_tensor - start_tensor
    if dim0_size <= 0:
        return data
    if dim0_size >= data_shape[0]:
        dim0_size = data_shape[0:1]
    value_shape = P.Concat(-1)((dim0_size, data_shape[1:]))
    value = ops.broadcast_to(value, value_shape)
    return copy_slice(data, value.astype(data.dtype), start_tensor, stop_tensor, step_tensor)


def tensor_setitem_by_slice_with_tensor(data, input_slice, value):
    """Assigns a tensor value to the tensor by slice."""
    result = None
    check_result = const_utils.check_tensor_setitem_index(input_slice)
    if check_result:
        data_shape = F.shape(data)
        step = const_utils.get_step_from_slice(input_slice)
        if step == 1 and not const_utils.is_ascend():
            if F.is_sequence_value_unknown(data_shape):
                return tensor_copy_slice_from_slice(data, input_slice, value)
            start, stop, step = const_utils.normalize_slice(input_slice, data.shape[0])
            dim0_size = stop - start
            if dim0_size <= 0:
                return data
            value_shape = (dim0_size,) + const_utils.tuple_slice(data.shape, 1, None)
            value = _broadcast(value_shape, value)
            return copy_slice(data, value.astype(data.dtype), (start,), (stop,), (step,))
        if F.is_sequence_value_unknown(data_shape):
            const_utils.raise_unimplemented_error(
                "Not supported to take the subscript of dynamic shape tensor slice setitem")
        indices = const_utils.slice2indices(input_slice, data_shape)
        if indices is False:
            return data
        value_shape = const_utils.tuple_slice(F.shape(indices), None, -1)
        value = _broadcast(value_shape, value)
        result = F.tensor_scatter_update(data, indices, value.astype(F.dtype(data)))
    return result


def tensor_setitem_by_slice_with_sequence(data, input_slice, value):
    """Assigns a list/tuple value to the tensor by slice."""
    value = _generate_updates_from_sequence(data, input_slice, value, const_utils.SET_ITEM_BY_NON_TENSOR)
    return tensor_setitem_by_slice_with_tensor(data, input_slice, value)


def tensor_copy_slice_from_tuple(data, tuple_index, value):
    """using TensorCopySlices by fixed model tuple."""
    data_shape = F.dyn_shape(data)
    dim1_start, dim1_stop, _ = get_slice_stride(tuple_index[1], data_shape[1])
    if dim1_stop - dim1_start <= 0:
        return data
    dim0_start = _scalar_to_tensor(tuple_index[0])
    dim0_stop = dim0_start + const_utils.make_tensor(1)
    start = (dim0_start, dim1_start)
    stop = (dim0_stop, dim1_stop)
    step = (const_utils.make_tensor(1), const_utils.make_tensor(1))
    start_tensor = stack(start)
    stop_tensor = stack(stop)
    step_tensor = stack(step)
    dim1_size = stack((dim1_stop - dim1_start,))
    if dim1_size > data_shape[1]:
        dim1_size = data_shape[1:2]
    value_shape = P.Concat(-1)((dim1_size, data_shape[2:]))
    value = ops.broadcast_to(value, value_shape)
    return copy_slice(data, value.astype(data.dtype), start_tensor, stop_tensor, step_tensor)


def tensor_setitem_by_tuple_with_tensor(data, tuple_index, value):
    """Assigns the tensor by tuple with tensor value."""
    op_name = const_utils.TENSOR_SETITEM
    tuple_index = _transform_ellipsis_to_slice(data, tuple_index, op_name)

    if const_utils.use_copy_slice(tuple_index) and not const_utils.is_ascend():
        if F.is_sequence_value_unknown(F.shape(data)):
            return tensor_copy_slice_from_tuple(data, tuple_index, value)
        dim1_start, dim1_stop, _ = const_utils.normalize_slice(tuple_index[1], data.shape[1])
        if dim1_stop - dim1_start <= 0:
            return data
        dim0_start = tuple_index[0] if tuple_index[0] >= 0 else tuple_index[0] + data.shape[0]
        start = (dim0_start, dim1_start)
        stop = (dim0_start + 1, dim1_stop)
        step = (1, 1)
        value_shape = (dim1_stop - dim1_start,) + const_utils.tuple_slice(data.shape, 2, None)
        value = _broadcast(value_shape, value)
        return copy_slice(data, value.astype(data.dtype), start, stop, step)

    tuple_index, value, idx_advanced = remove_expanded_dims(tuple_index, F.shape(data), value)

    if tuple_index is False:
        return data
    if len(tuple_index) == 1:
        data[tuple_index[0]] = value
        return data
    indexes_types = hyper_map(toptypeof, tuple_index)
    contain_type = const_utils.tuple_index_type_cnt(indexes_types, op_name)

    if contain_type == const_utils.ALL_TENSOR:
        indices = _generate_indices_from_tuple_of_tensor(tuple_index, op_name)
    else:
        indices = _generate_indices_from_tuple(data, tuple_index, op_name, idx_advanced)
        if indices is False:
            return data
    updates = _generate_updates_from_tensor(data, indices, value, const_utils.SET_ITEM_BY_TUPLE_OF_TENSOR)
    return F.tensor_scatter_update(data, indices, updates)


def tensor_setitem_by_tuple_with_sequence(data, tuple_index, value):
    value = _generate_updates_from_sequence(data, tuple_index, value, const_utils.SET_ITEM_BY_NON_TENSOR)
    return tensor_setitem_by_tuple_with_tensor(data, tuple_index, value)


def tensor_setitem_by_number_with_number(data, index, value):
    """Assigns the tensor by number with number value."""
    value = F.fill(F.dtype(data), (), value)
    return tensor_setitem_by_number_with_tensor(data, index, value)


def tensor_setitem_by_number_with_sequence(data, index, value):
    """Assigns a list/tuple value to the tensor by slice."""
    value = _generate_updates_from_sequence(data, index, value, const_utils.SET_ITEM_BY_NON_TENSOR)
    return tensor_setitem_by_number_with_tensor(data, index, value)


def tensor_setitem_by_number_with_tensor(data, index, value):
    """Assigns the tensor by number with tensor value."""
    data_shape = F.shape(data)
    if F.is_sequence_value_unknown(data_shape):
        index = _scalar_to_tensor(index)
        index = F.expand_dims(index, -1)
        return _tensor_setitem_by_int_tensor_with_tensor(data, index, value)

    dim_size = data_shape[0]
    if index < -dim_size or index >= dim_size:
        raise IndexError(f'index {index} is out of bounds for axis 0 with size {dim_size}')
    index = const_utils.int_to_index(index, data_shape)
    value_shape = const_utils.tuple_slice(F.shape(index), None, -1)
    value = _broadcast(value_shape, value.astype(F.dtype(data)))
    if is_parameter(data):
        return F.scatter_nd_update(data, index, value)
    return F.tensor_scatter_update(data, index, value)


def tensor_setitem_by_ellipsis_with_number(data, value):
    """Assigns the tensor by ellipsis with number value."""
    data_shape = F.shape(data)
    data_dtype = F.dtype(data)
    if F.is_sequence_value_unknown(data_shape):
        value = F.fill(F.dtype(data), (), value)
        return tensor_setitem_by_ellipsis_with_tensor(data, value)
    return F.fill(data_dtype, data_shape, value)


def tensor_setitem_by_ellipsis_with_tensor(data, value):
    """Assigns the tensor by ellipsis with tensor value."""
    data_shape = F.shape(data)
    data_dtype = F.dtype(data)
    value = value.astype(data_dtype)

    if F.is_sequence_value_unknown(data_shape):
        data_shape = F.dyn_shape(data)
        data = ops.broadcast_to(value, data_shape)
        return data
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
    elif isinstance(value, (int, bool)):
        value = const_utils.make_tensor(value, mstype.int32)
    elif isinstance(value, float):
        value = const_utils.make_tensor(value, mstype.float32)

    if F.is_sequence_value_unknown(data_shape) and index:
        data_shape = F.dyn_shape(data)
        data = ops.broadcast_to(value, data_shape)
        return data
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
    if const_utils.judge_indexes_types(indices_types, mstype.int_type + (mstype.bool_,)):
        if not F.isconstant(length):
            return const_utils.sequence_to_index(list_indices, None)
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


def remove_expanded_dims(tuple_index, data_shape, value):
    """Removes expanded dimensions in tuple_index and value."""
    not_expanded_dim = ()
    shapes = ()
    has_true = False
    has_false = False
    has_sequence = False
    indices_out = ()  # with dimension expansion indices removed
    idx_tensor = -1  # index of the previous tensor
    idx_advanced = -1  # index of the first advanced index in expanded tensor
    cur_dim = 0  # current dimension of the data to be indexed

    for i, v in enumerate(tuple_index):
        index_out = format_index(v, data_shape, cur_dim)

        if index_out is None:
            not_expanded_dim += (False,)
        elif const_utils.is_slice(index_out):
            indices_out += (index_out,)
            not_expanded_dim += (True,)
            start, stop, step = const_utils.normalize_slice(index_out, data_shape[cur_dim])
            has_false = has_false or const_utils.check_slice_empty(start, stop, step)
            cur_dim += 1
        elif isinstance(index_out, (Tensor, bool)):  # advanced index
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

    broadcast_shape = const_utils.generate_broadcast_shape(shapes, const_utils.TENSOR_SETITEM)
    if has_false:
        if F.shape_mul(broadcast_shape) != 1:
            const_utils.raise_index_error('unable to broadcast indices')
        indices_out = False
    else:
        expand_true = has_true and not (has_false or has_sequence)  # whether to expand dimension at True
        tensor_index_ndim = len(broadcast_shape)  # ndim of tensor indices
        rem_ndim = len(data_shape) - cur_dim  # number of remaining dimensions in data not indexed
        not_expanded_dim, idx_advanced = const_utils.rem_not_expanded_dims(idx_advanced, expand_true,
                                                                           tensor_index_ndim,
                                                                           rem_ndim, not_expanded_dim)
        if not indices_out:
            indices_out = (True,)

        value_shape = const_utils.filter_expanded_dims(F.shape(value), not_expanded_dim)
        value = F.reshape(value, value_shape)
    return indices_out, value, idx_advanced


def format_index(idx, data_shape, cur_dim):
    """Converts advanced index into tensor."""
    if isinstance(idx, (tuple, list)):
        idx = const_utils.sequence_to_index(idx, data_shape[cur_dim])
    elif isinstance(idx, int) and not isinstance(idx, bool):
        idx = const_utils.make_tensor(idx, mstype.int64, None, data_shape[cur_dim])
    elif isinstance(idx, Tensor):
        # does not take bool tensor into account since it's currently not supported
        idx = F.select(idx < 0, idx + data_shape[cur_dim], idx)
    return idx


def reduce_(a, reduce_fn, cmp_fn=None, axis=None, keepdims=False, initial=None, where=True, dtype=None):
    """
    Applies comparison based on cmp_fn and reduction based on reduce_fn.
    If cmp_fn is None, only reduction is performed.
    """

    shape = F.shape(a)
    ndim = F.rank(a)
    if dtype is None:
        dtype = F.dtype(a)
    axes = const_utils.check_axis_valid_const(axis, ndim)
    if initial is not None:
        if ((isinstance(initial, Tensor) and F.rank(initial) > 0) or
                not isinstance(initial, (int, float, bool, Tensor))):
            const_utils.raise_type_error('initial must be scalar')

    if F.shape_mul(shape) == 0:
        const_utils.raise_value_error('zero-size tensors are not supported.')

    if initial is not None:
        if isinstance(initial, Tensor):
            initial = F.tile(initial, shape).astype(dtype)
        else:
            initial = F.fill(dtype, shape, initial)
        a = cmp_fn(a, initial)

    if isinstance(where, Tensor):
        if initial is None:
            const_utils.raise_value_error('initial value must be provided for where masks')
        ndim_orig = F.rank(a)
        # broadcasts input tensors
        shape_out = const_utils.infer_out_shape(F.shape(where), F.shape(a), F.shape(initial))
        broadcast_to = P.BroadcastTo(shape_out)
        where = where.astype(mstype.float32)
        where = broadcast_to(where)
        where = where.astype(mstype.bool_)
        a = broadcast_to(a)
        initial = broadcast_to(initial)
        a = F.select(where, a, initial)
        axes = const_utils.real_axes(ndim_orig, F.rank(a), axes)

    return reduce_fn(a, axes).astype(dtype)


tensor_operator_registry.register("reduce", reduce_)


def check_indices(dims, indices, mode, allow_negative_index=True):
    """Checks whether indices are out of bounds."""
    shape = F.shape(indices)
    dtype = F.dtype(indices)
    if not allow_negative_index:
        lowerbounds = F.fill(dtype, shape, 0)
    else:
        lowerbounds = F.fill(dtype, shape, -dims)
    upperbounds = F.fill(dtype, shape, dims - 1)
    out_of_lowerbounds = F.tensor_lt(indices, lowerbounds)
    out_of_upperbounds = F.tensor_gt(indices, upperbounds)
    if mode == 'raise':
        const_utils.raise_unimplemented_error('"raise" mode is not implemented')
    if mode == 'wrap':
        bounds = F.fill(dtype, shape, dims)
        quotient = F.tensor_floordiv(indices, bounds)
        prod = F.tensor_mul(bounds, quotient)
        return F.tensor_sub(indices, prod)
    zeros = F.fill(dtype, shape, 0)
    clipped = F.select(out_of_lowerbounds, zeros, indices)
    clipped = F.select(out_of_upperbounds, upperbounds, clipped)
    return clipped


tensor_operator_registry.register('check_indices', check_indices)


def convert_slice_to_tensor(index, final_shape, slice_cnt, broadcast_shape, slice_shapes, fancy_position):
    """Convert a slice to a tensor."""
    shape = const_utils.compute_slice_shape(slice_shapes, len(broadcast_shape), slice_cnt, fancy_position)
    array = const_utils.make_tensor(index, mstype.int64).reshape(shape)
    reps = const_utils.compute_multiples(shape, final_shape)
    slice_index_tensor = F.tile(array, reps)
    return slice_index_tensor


def check_coo_tensor_input_length(coo_tuple):
    """Check length of coo tensor."""
    coo_length = 3
    if len(coo_tuple) != coo_length:
        raise ValueError(f"Expect coo_tuple have 3 inputs (indices, values, shape), but got {len(coo_tuple)}.")
    return coo_tuple


def check_csr_tensor_input_length(csr_tuple):
    """Check length of csr tensor."""
    csr_length = 4
    if len(csr_tuple) != csr_length:
        raise ValueError(f"Expect csr_tuple have 4 inputs (indptr, indices, values, shape), but got {len(csr_tuple)}.")
    return csr_tuple
