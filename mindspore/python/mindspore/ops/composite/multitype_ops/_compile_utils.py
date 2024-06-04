# Copyright 2020-2023 Huawei Technologies Co., Ltd
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
from enum import IntEnum


from mindspore.ops.composite.multitype_ops import _constexpr_utils as const_utils
from mindspore.ops import functional as F
from mindspore.ops import operations as P
from mindspore.ops.composite import base
from mindspore.ops._primitive_cache import _get_cache_prim
from mindspore.ops.operations._inner_ops import TensorCopySlices, SliceGetItem, \
    TopTypeof, IsParameter, GetitemTensorIndexInfo, SetitemTensorIndexInfo, \
    SelectView, CopyWithSlice
from mindspore.ops.operations._sequence_ops import TensorToTuple, TensorToScalar, TupleToTensor
from mindspore.common import dtype as mstype
from mindspore.common._register_for_tensor import tensor_operator_registry
from mindspore.common.initializer import Zero
from mindspore.common import Tensor, CSRTensor, COOTensor, mutable
from mindspore import ops
from mindspore.ops.primitive import _primexpr
from mindspore import _checkparam as validator
from mindspore.common._stub_tensor import _convert_stub

slice_get_item = SliceGetItem()
hyper_map = base.HyperMap()
stack = P.Stack(axis=-1)
copy_slice = TensorCopySlices()
toptypeof = TopTypeof()
is_parameter = IsParameter()
getitem_tensor_index_info = GetitemTensorIndexInfo(const_utils.is_ascend())
setitem_tensor_index_info = SetitemTensorIndexInfo(const_utils.is_ascend())

selevt_view = SelectView()
copy_with_slice = CopyWithSlice()

def strided_slice(data, begin_strides, end_strides, step_strides, begin_mask=0, end_mask=0, ellipsis_mask=0,
                  new_axis_mask=0, shrink_axis_mask=0):
    """strided_slice primitive cache"""
    strided_slice_ = _get_cache_prim(P.StridedSlice)(begin_mask, end_mask, ellipsis_mask, new_axis_mask,
                                                     shrink_axis_mask)
    return strided_slice_(data, begin_strides, end_strides, step_strides)


class ValueTransferType(IntEnum):
    """Transfer op types of handling tensor getitem/setitem"""
    kUnknown = 0
    kTensorScatterUpdate = 1
    kExpandDims = 2
    kBroadCast = 3
    kCast = 4
    kSelect = 5
    kGather = 6
    kStrideSlice = 7
    kStrideSliceWithMask = 8
    kGatherND = 9
    kScatterNdUpdate = 10
    kReshape = 11
    kSelectView = 12
    kUnsqueeze = 13
    kCopyView = 14
    kScatterND = 15
    kNumberToTensor = 16
    kHandleSequenceValue = 17
    kByPass = 18
    kReSetItemByIndex = 19
    kCopySlice = 20
    kSetItemByBool = 21
    kEmptyTensor = 22
    kSetItemByEllipsis = 23
    kFormatIndexTensor = 24
    kGetitemByBoolTensor = 25
    kSetitemByBoolTensor = 26
    kJustReturn = 27
    kRaiseIndexError = 28


def data_update(transfer_types, args, data, new_index, value=None):
    """
    We finally generate a new tensor when handling tensor getitem/setitem
    by transfer data and value with index.
    """
    origin_data = data
    for transfer_type, arg in zip(transfer_types, args):
        if transfer_type == ValueTransferType.kUnknown:
            raise IndexError(f"Inlvaid transfer type {transfer_type}.")
        if transfer_type <= ValueTransferType.kScatterND:
            data = data_update_by_ops(transfer_type, arg, data, new_index, origin_data, value)
        if transfer_type == ValueTransferType.kJustReturn:
            return _convert_stub(arg)
        if transfer_type == ValueTransferType.kSetItemByBool:
            return tensor_setitem_by_bool(data, new_index, value)
        if transfer_type == ValueTransferType.kCopySlice:
            return copy_slice(data, value.astype(data.dtype), arg[0], arg[1], arg[2])
        if transfer_type == ValueTransferType.kSetItemByEllipsis:
            return tensor_setitem_by_ellipsis(data, new_index, value)
        if transfer_type == ValueTransferType.kReSetItemByIndex:
            data[new_index] = value
            return data
        if transfer_type == ValueTransferType.kEmptyTensor:
            return handle_empty_tensor(arg, data)
        if transfer_type == ValueTransferType.kFormatIndexTensor:
            new_index = format_index_tensor(new_index, arg)
        if transfer_type == ValueTransferType.kGetitemByBoolTensor:
            return F.gather_nd(data, new_index.nonzero())
        if transfer_type == ValueTransferType.kSetitemByBoolTensor:
            return handle_setitem_by_bool_tensor(data, new_index, value)
        if transfer_type == ValueTransferType.kRaiseIndexError:
            raise IndexError(
                f'index {arg[0]} is out of bounds for dimension with size {arg[1]}')
    return data


def data_update_by_ops(transfer_type, arg, data, new_index, origin_data, value=None):
    """
    Generate a new tensor when handling tensor getitem/setitem
    by ops.
    """
    if transfer_type == ValueTransferType.kStrideSliceWithMask:
        stride_info, mask_index = arg[0], arg[1]
        data = strided_slice(data, stride_info[0], stride_info[1], stride_info[2],
                             mask_index[0], mask_index[1], 0, 0, mask_index[2])
    elif transfer_type == ValueTransferType.kGatherND:
        if isinstance(new_index, list):
            new_index = handle_multi_dim_index_tensor(new_index, arg)
        new_index = format_index_tensor(new_index, (None, F.shape(data)[:F.shape(new_index)[-1]]))
        data = F.gather_nd(data, new_index)
    elif transfer_type == ValueTransferType.kTensorScatterUpdate:
        if isinstance(new_index, list):
            new_index = handle_multi_dim_index_tensor(new_index, arg)
        data = F.tensor_scatter_update(data, new_index, value)
    elif transfer_type == ValueTransferType.kScatterNdUpdate:
        F.scatter_nd_update(data, new_index, value)
    elif transfer_type == ValueTransferType.kSelect:
        data = F.select(Tensor(new_index), value, data)
    elif transfer_type == ValueTransferType.kSelectView:
        data = selevt_view(data, arg[0], arg[1])
    elif transfer_type == ValueTransferType.kCopyView:
        value = _broadcast(F.shape(data), F.cast(value, F.dtype(data)))
        data = copy_with_slice(data, value)
        return origin_data
    elif transfer_type == ValueTransferType.kReshape:
        data = F.reshape(data, arg)
    elif transfer_type == ValueTransferType.kGather:
        data = F.gather(data, new_index, 0)
    elif transfer_type == ValueTransferType.kExpandDims:
        data = F.expand_dims(data, 0)
    elif transfer_type == ValueTransferType.kUnsqueeze:
        data = F.unsqueeze(data, arg)
    elif transfer_type == ValueTransferType.kStrideSlice:
        data = strided_slice(data, arg[0], arg[1], arg[2])
    else:
        raise IndexError(f"Inlvaid transfer type {transfer_type}.")
    return data


def value_update(transfer_types, args, data, value):
    """Transfer value before set value to tensor when handling tensor setitem"""
    for transfer_type, arg in zip(transfer_types, args):
        if transfer_type == ValueTransferType.kByPass:
            continue
        if transfer_type == ValueTransferType.kNumberToTensor:
            value = F.cast(value, F.dtype(data))
        elif transfer_type == ValueTransferType.kHandleSequenceValue:
            op_type, index = arg
            if op_type == const_utils.SET_ITEM_BY_ONE_TENSOR:
                index = Tensor(index)
            value = _generate_updates_from_sequence(
                data, index, value, op_type)
        elif transfer_type == ValueTransferType.kExpandDims:
            value = F.expand_dims(value, arg)
        elif transfer_type == ValueTransferType.kBroadCast:
            value = _broadcast(arg, value.astype(F.dtype(data)))
        elif transfer_type == ValueTransferType.kCast:
            value = F.cast(value, F.dtype(data))
        elif transfer_type == ValueTransferType.kReshape:
            value = F.reshape(value, arg)
        elif transfer_type == ValueTransferType.kScatterND:
            value = F.scatter_nd(arg[0], value, arg[1])
        else:
            raise IndexError(f"Inlvaid transfer type {transfer_type}.")
    return value


def _tensor_getitem(self, index):
    """Handle tensor getitem"""
    new_index, tensor_update_types, tensor_update_args = getitem_tensor_index_info(
        self, index)
    return data_update(tensor_update_types, tensor_update_args, self, new_index)


def _tensor_setitem(self, index, value):
    """Handle tensor setitem"""
    setitem_info = setitem_tensor_index_info(self, index, value)
    new_index = setitem_info[0]
    v_transfer_types = setitem_info[1]
    v_transfer_args = setitem_info[2]
    data_update_types = setitem_info[3]
    data_update_args = setitem_info[4]
    value = value_update(v_transfer_types, v_transfer_args, self, value)
    output = data_update(data_update_types, data_update_args, self, new_index, value)
    if new_index == "view":
        return (self,)
    return output


setattr(tensor_operator_registry, "__getitem__", _tensor_getitem)
setattr(tensor_operator_registry, "__setitem__", _tensor_setitem)


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


setattr(tensor_operator_registry, '__add__', _tensor_add)
setattr(tensor_operator_registry, '__sub__', _tensor_sub)
setattr(tensor_operator_registry, '__mul__', _tensor_mul)
setattr(tensor_operator_registry, '__matmul__', _tensor_matmul)
setattr(tensor_operator_registry, '__truediv__', _tensor_div)
setattr(tensor_operator_registry, '__mod__', _tensor_mod)
setattr(tensor_operator_registry, '__pow__', _tensor_pow)
setattr(tensor_operator_registry, '__rpow__', _tensor_rpow)
setattr(tensor_operator_registry, '__floordiv__', _tensor_floordiv)


def _scalar_to_tensor(input_x):
    if ops.isconstant(input_x):
        return P.ScalarToTensor()(input_x, ops.dtype(input_x))
    # use add Tensor([0]) cast scalar to tensor.
    return ops.add(input_x, mutable(Tensor(0)))


@_primexpr
def _check_scalar_tensor_args(args):
    """For the item, check that the index of the scalar tensor is set."""
    if args not in ((None,), ()):
        const_utils.raise_value_error("For item, the index of scalar Tensor should not be set.")


def tensor_item(data, *args):
    """Tensor getitem by index whose dtype is int or tuple with int."""
    # transform a.item(tuple(int)) -> a.item(int1,int2...intN)
    if data.ndim == 0:
        _check_scalar_tensor_args(args)
        return TensorToScalar()(data)
    if len(args) == 1 and isinstance(args[0], tuple):
        args = args[0]

    args_types = hyper_map(F.typeof, args)
    if not args or const_utils.judge_index_type(args_types[0], mstype.type_none):
        if data.shape == (1,):
            return TensorToScalar()(data[0])
        const_utils.raise_value_error("Can only convert an array of size 1 to a Python scalar")

    if not const_utils.judge_indexes_types(args_types, mstype.int64):
        const_utils.raise_type_error("The index object cannot be interpreted as an integer")

    if len(args) == data.ndim:
        return tensor_index_by_tuple(data, args)
    if len(args) > 1:
        const_utils.raise_value_error("Incorrect number of indices for array")
    output = _tensor_index_by_integer(F.reshape(data, (-1,)), args[0])
    return TensorToScalar()(output)


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


setattr(tensor_operator_registry, "item", tensor_item)
setattr(tensor_operator_registry, "itemset", tensor_itemset)


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
    nubmer_value = F.cast(nubmer_value, F.dtype(data))
    return tensor_itemset_by_tuple_with_tensor(data, tuple_index, nubmer_value)


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


def handle_empty_tensor(arg, data):
    """handle data update with empty tensor"""
    if 0 in arg:
        init_func = Zero()
        init_func.__enable_zero_dim__ = True
        return Tensor(shape=arg, dtype=data.dtype, init=init_func)
    return const_utils.make_tensor([], data.dtype, arg)


def handle_multi_dim_index_tensor(new_index, arg):
    """handle data update with multi dim index tensor"""
    slice_cnt = 0
    new_indies_tensor = []
    if len(arg) == 1:
        broadcast_shape = arg[0]
        new_index = hyper_map(F.partial(Tensor), new_index)
        broadcast_tensors = hyper_map(
            F.partial(_broadcast, broadcast_shape), new_index)
        new_broadcast_tensors = ()
        for tensor in broadcast_tensors:
            new_broadcast_tensors += (F.cast(tensor, mstype.int64),)
        new_index = stack(new_broadcast_tensors)
        return new_index
    broadcast_shape, final_shape, index_tensor_new_shape, slice_shapes, tensor_positions, fancy_position = arg
    for i, index in enumerate(new_index):
        if i in tensor_positions:
            transform_tensor = _transform_indexing_tensor(broadcast_shape, final_shape, index_tensor_new_shape,
                                                          Tensor(index))
            new_indies_tensor.append(F.cast(transform_tensor, mstype.int64))
        else:
            shape = const_utils.compute_slice_shape(
                slice_shapes, len(broadcast_shape), slice_cnt, fancy_position)
            array = Tensor(index).reshape(shape)
            slice_index_tensor = _broadcast(final_shape, array)
            new_indies_tensor.append(F.cast(slice_index_tensor, mstype.int64))
            slice_cnt += 1
    new_index = stack(new_indies_tensor)
    return new_index


def format_index_tensor(index, arg):
    """Format index tensor when tensor less than 0"""
    format_indices, format_dims = arg
    if isinstance(index, list):
        for format_idx, format_dim in zip(format_indices, format_dims):
            index_tensor = index[format_idx]
            index[format_idx] = F.select(index_tensor < 0, index_tensor + format_dim, index_tensor)
        return index
    index = Tensor(index)
    return F.select(index < 0, index + format_dims, index)


def handle_setitem_by_bool_tensor(data, index, value):
    """Set a tensor item by a bool tensor with a tensor."""
    value = F.cast(value, F.dtype(data))
    indices = index.nonzero()
    if indices.shape[0] == 0:
        return data
    value_shape = (indices.shape[0],) + data.shape[index.ndim:]
    value = _broadcast(value_shape, value)
    value = F.scatter_nd(indices, value, data.shape)
    index = index.reshape(const_utils.generate_padding_shape(index.shape, len(data.shape)))
    index = _broadcast(data.shape, index)
    result = F.select(index, value, data)
    return result


def _expand_data_dims(data, tuple_index):
    """expand the data's dim with 'None' and 'Boolean' in tuple_index"""
    indexes_types = hyper_map(toptypeof, tuple_index)
    expand_positions, tuple_index_new = (), ()
    for i, (index, index_type) in enumerate(zip(tuple_index, indexes_types)):
        if isinstance(index_type, mstype.NoneType):
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


def _convert_list_index_to_tensor(list_index):
    """convert list to tensor"""
    has_bool = False
    has_int = False
    has_no_bool_int = False
    for idx in list_index:
        if isinstance(idx, bool):
            has_bool = True
        elif isinstance(idx, int):
            has_int = True
        else:
            has_no_bool_int = True

    all_bool = has_bool and not has_int and not has_no_bool_int
    all_int = has_int and not has_bool and not has_no_bool_int
    all_bool_or_int = not has_no_bool_int

    if all_int:
        index_tensor = TupleToTensor()(tuple(list_index), mstype.int64)
        return index_tensor


    if all_bool:
        index_tensor = TupleToTensor()(tuple(list_index), mstype.bool_)
        return index_tensor

    # convert bool to int if index is mixture of (bool, int)
    if all_bool_or_int:
        new_index = []
        for idx in list_index:
            if isinstance(idx, bool):
                new_idx = int(idx)
                new_index.append(new_idx)
            else:
                new_index.append(idx)
        index_tensor = TupleToTensor()(tuple(new_index), mstype.int64)
        return index_tensor

    return None


class _TensorIndexGetitem(base.TensorIndexGetitem_):
    """
    Getting item of Tensor.

    Args:
        data (Tensor): A tuple to be sliced.
        index: Index of tensor.

    Returns:
        Type is the same as the element type of data.
    """

    def __call__(self, *args):
        pass

_tensor_index_getitem = _TensorIndexGetitem('tensor_index_getitem')


def tensor_index_by_slice(data, slice_index):
    """Tensor getitem by a slice."""
    return _tensor_index_getitem(data, slice_index)


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
    output = data
    if bool_value:
        output = F.expand_dims(data, 0)
    elif not F.is_sequence_value_unknown(F.shape(data)):
        return const_utils.raise_index_error("When tensor is indexed by a bool object, the value only support 'True'.")
    return output


def get_stride_info_from_integer(int_index):
    """Convert integer to slice"""
    begin_strides = (int_index,)
    end_strides = (int_index + 1,)
    step_strides = (1,)
    return begin_strides, end_strides, step_strides


def _tensor_index_by_integer(data, int_index):
    """Tensor getitem by a single integer number"""
    begin_strides, end_strides, step_strides = get_stride_info_from_integer(int_index)

    shrink_axis_mask = 1
    begin_mask = 0
    end_mask = 0
    for i in range(2, 8):
        begin_mask += 2 ** i
        end_mask += 2 ** i
    return strided_slice(data, begin_strides, end_strides, step_strides, begin_mask, end_mask, 0, 0, shrink_axis_mask)

def _check_dim_shape_valid(data, tensor_index):
    """check dim and shape of tensor_index for tensor(bool) indexing"""
    if data.ndim < tensor_index.ndim:
        raise IndexError(f"The dim of index cannot be greater than indexed data, but got "
                         f"dim of index:{tensor_index.ndim}, dim of data:{data.ndim}")
    if data.shape[:tensor_index.ndim] != tensor_index.shape[:]:
        raise IndexError(f"The shape of index {tensor_index.shape} does not match the shape "
                         f"of the indexed data {data.shape}")


def tensor_index_by_bool_tensor(data, tensor_index):
    """Tensor getitem by a bool tensor"""
    if not F.is_sequence_value_unknown(F.shape(data)):
        _check_dim_shape_valid(data, tensor_index)
    tensor_index = tensor_index.nonzero()
    return F.gather_nd(data, tensor_index)


def tensor_index_by_tensor(data, tensor_index):
    """Tensor getitem by a single tensor"""
    min_data_dim, max_data_dim = 0, 7
    if not F.is_sequence_value_unknown(F.shape(data)):
        const_utils.judge_data_dim(data.ndim, min_data_dim, max_data_dim)
    if const_utils.check_type_isinstance(F.dtype(tensor_index), mstype.Int):
        tensor_index = F.select(tensor_index < 0, tensor_index + F.shape(data)[0], tensor_index)
        return F.gather(data, tensor_index, 0)
    if const_utils.check_type_isinstance(F.dtype(tensor_index), mstype.Bool):
        return tensor_index_by_bool_tensor(data, tensor_index)
    exp_msg = const_utils.gen_exception_msg(
        "The tensor index must be int or bool type, but got {}.", F.dtype(tensor_index))
    const_utils.raise_index_error(exp_msg)
    return data


def tensor_index_by_list(data, list_index):
    """Tensor getitem by list of int and bool"""
    min_data_dim, max_data_dim = 1, 8
    if F.isconstant(data.ndim):
        const_utils.judge_data_dim(data.ndim, min_data_dim, max_data_dim)

    data_shape = F.shape(data)
    if F.isconstant(data_shape[0]) and all(isinstance(i, bool) for i in list_index):
        if data_shape[0] != len(list_index):
            raise IndexError(
                f'dimension is {data_shape[0]} but corresponding boolean dimension is {len(list_index)}')
        tensor_index = Tensor(list_index).nonzero()
        return F.gather_nd(data, tensor_index)

    if not list_index:
        const_utils.raise_index_error("When tensor is indexed by list, the list can't be empty.")

    index_tensor = _convert_list_index_to_tensor(list_index)
    if index_tensor is not None:
        return tensor_index_by_tensor(data, index_tensor)

    tuple_index_new = ()
    for index in list_index:
        tuple_index_new += (index,)
    return tensor_index_by_tuple(data, tuple_index_new)


def judge_tuple_index_dim_check_error(index_dim, data_dim):
    """raise IndexError when tuple_index's dim is invalid"""
    if index_dim > data_dim:
        raise IndexError(f"The dim of index cannot be greater than indexed data, but got "
                         f"dim of index:{index_dim}, dim of data:{data_dim}")


def judge_tuple_index_dim(data, tuple_index):
    """Judge whether tuple_index's dim is valid"""
    data_dim = data.ndim
    index_dim = 0
    for index in tuple_index:
        if isinstance(toptypeof(index), mstype.TensorType) and index.dtype == mstype.bool_:
            index_dim += index.ndim
        elif not isinstance(toptypeof(index), (mstype.NoneType, mstype.Ellipsis_, mstype.Bool)):
            index_dim += 1
    judge_tuple_index_dim_check_error(index_dim, data_dim)


def tensor_index_by_tuple(data, tuple_index):
    """Tensor getitem by tuple of various types with None"""
    if not tuple_index:
        return data

    if not F.is_sequence_value_unknown(F.shape(data)):
        judge_tuple_index_dim(data, tuple_index)
    tuple_index, zero_index, non_zero_shapes = _handle_bool_tensor(tuple_index)
    for non_zero_shape in non_zero_shapes:
        if 0 in non_zero_shape:
            tuple_index = zero_index
            break

    return _tensor_index_getitem(data, tuple_index)


def get_slice_stride(slice_index, dim_size):
    """Get slice stride info"""
    start = slice_get_item(slice_index, "start")
    stop = slice_get_item(slice_index, "stop")
    step = slice_get_item(slice_index, "step")

    if start is None:
        start = 0
    if stop is None:
        stop = dim_size
    if step is None:
        step = 1

    if isinstance(start, Tensor):
        start = int(start)

    if isinstance(stop, Tensor):
        stop = int(stop)

    if isinstance(step, Tensor):
        step = int(step)

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


def parse_check_slice_index(index_out, dim_size):
    """ Parse and check slice index """
    has_false = False
    start, stop, step = const_utils.normalize_slice(index_out, dim_size)
    if F.isconstant(start) and F.isconstant(stop) and F.isconstant(step):
        has_false = const_utils.check_slice_empty(start, stop, step)
    return has_false


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
                    "The tensor element in tuple index must be int or bool type, but got {}.", F.dtype(index))
                const_utils.raise_index_error(exp_msg)
            tensor_index = F.cast(index, mstype.int64)
            tuple_index_new += (tensor_index,)
            tensor_indexes.append(tensor_index)
        elif i in slice_positions:
            if parse_check_slice_index(index, dim_size):
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
        if isinstance(value, list):
            value = tuple(value)

        if dtype == mstype.float16:
            value = TupleToTensor()(value, mstype.float32)
            value = F.cast(value, dtype)
        else:
            value = TupleToTensor()(value, dtype)
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
    updates_shape = const_utils.generate_updates_shape(data.shape, index.shape, op_type)
    updates = ops.broadcast_to(value, updates_shape)
    return updates


# Tensor getitem implementations are above this line, setitem implementations below.

def _tensor_index_transfer(index, broadcast_shape, final_shape, new_shape):
    """Transform tuple index tensor to the required."""
    if 0 in final_shape:
        return F.fill(index.dtype, final_shape, 0)

    if broadcast_shape == ():
        # broadcast_to () is not support on Ascend
        item = index
    else:
        item = F.broadcast_to(index, broadcast_shape)
    item = F.reshape(item, new_shape)
    return F.broadcast_to(item, final_shape)


def reshape_with_check(x, new_shape):
    if isinstance(new_shape, Tensor):
        new_shape = TensorToTuple()(new_shape)
    return F.reshape(x, new_shape)


class _TensorIndexSetitem(base.TensorIndexSetitem_):
    """
    Getting item of Tensor.

    Args:
        data (Tensor): A tuple to be sliced.
        index: Index of tensor.

    Returns:
        Type is the same as the element type of data.
    """

    def __call__(self, *args):
        pass


_tensor_index_setitem = _TensorIndexSetitem('tensor_index_setitem')


def tensor_setitem_by_slice(self, index, value):
    """Set a tensor item by slice."""
    indices, value_shape, start, stop, step, value = _tensor_index_setitem(
        self, index, value)
    if start == stop:
        return self
    value = F.broadcast_to(value, value_shape)
    if not const_utils.is_ascend() and step == 1:
        start = (start,)
        stop = (stop,)
        step = (step,)
        return copy_slice(self, value, start, stop, step)
    return F.tensor_scatter_update(self, indices, value)


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

    data_shape = F.shape(data)
    updates_shape = index.shape + data_shape[1:]
    value = F.cast(value, F.dtype(data))
    updates = ops.broadcast_to(value, updates_shape)
    first_val = data_shape[0]
    index = F.select(index < 0, index + first_val, index)
    index = F.expand_dims(index, -1)
    if is_parameter(data):
        F.scatter_nd_update(data, index, updates)
        return data
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

    return _tensor_setitem_by_bool_tensor_with_tensor(data, index, value_tensor)


def tensor_setitem_by_tensor_with_number(data, index, value):
    value = F.cast(value, F.dtype(data))
    return tensor_setitem_by_tensor_with_tensor(data, index, value)


def tensor_setitem_by_tensor_with_sequence(data, index, value):
    """Assigns the tensor by tensor with tuple value."""
    value = sequence_to_tensor(value, F.dtype(data))
    return tensor_setitem_by_tensor_with_tensor(data, index, value)


def tensor_setitem_by_tuple_with_number(data, tuple_index, value):
    """Assigns the tensor by tuple with number value."""
    value = F.cast(value, F.dtype(data))
    return tensor_setitem_by_tuple_with_tensor(data, tuple_index, value)


def tensor_setitem_by_list(data, index, value):
    """list indices will be converted to tuple or tensor based on its contents."""
    index_tensor = _convert_list_index_to_tensor(index)
    if index_tensor is not None:
        return tensor_setitem_by_tensor_with_tensor(data, index_tensor, value)

    return tensor_setitem_by_tuple_with_tensor(data, tuple(index), value)



class _PreSetitemByTuple(base.PreSetitemByTuple_):
    """
    Getting item of Tensor.

    Args:
        data (Tensor): A tuple to be sliced.
        index: Index of tensor.

    Returns:
        Type is the same as the element type of data.
    """

    def __init__(self, name):
        """Initialize _PreSetitemByTuple."""
        base.PreSetitemByTuple_.__init__(self, name)

    def __call__(self, *args):
        pass


_pre_setitem_by_tuple = _PreSetitemByTuple('pre_setitem_by_tuple')


class _HandleBoolTensor(base.HandleBoolTensor_):
    """
    Getting item of Tensor.

    Args:
        data (Tensor): A tuple to be sliced.
        index: Index of tensor.

    Returns:
        Type is the same as the element type of data.
    """

    def __init__(self, name):
        """Initialize _HandleBoolTensor."""
        base.HandleBoolTensor_.__init__(self, name)

    def __call__(self, *args):
        pass


_handle_bool_tensor = _HandleBoolTensor('handle_bool_tensor')


def tensor_setitem_by_tuple_with_tensor(data, tuple_index, value):
    """Assigns the tensor by tuple with tensor value."""
    if const_utils.use_copy_slice(tuple_index) and not const_utils.is_ascend():
        dim1_start, dim1_stop, _ = const_utils.normalize_slice(
            tuple_index[1], data.shape[1])
        if isinstance(dim1_start, Tensor):
            dim1_start = int(dim1_start)
        if isinstance(dim1_stop, Tensor):
            dim1_stop = int(dim1_stop)
        if dim1_stop - dim1_start <= 0:
            return data
        dim0_start = tuple_index[0] if tuple_index[0] >= 0 else tuple_index[0] + data.shape[0]
        start = (dim0_start, dim1_start)
        stop = (dim0_start + 1, dim1_stop)
        step = (1, 1)
        value_shape = (dim1_stop - dim1_start,) + data.shape[2:]
        value = F.broadcast_to(value, value_shape)
        return copy_slice(data, value.astype(data.dtype), start, stop, step)
    tuple_index, _, non_zero_shapes = _handle_bool_tensor(tuple_index)

    for non_zero_shape in non_zero_shapes:
        if 0 in non_zero_shape:
            return data
    value = value.astype(data.dtype)
    special_index, tuple_index, new_value_shape, idx_advanced, _broadcast_data_shape \
        = _pre_setitem_by_tuple(data, tuple_index, value)
    if special_index == 0:
        return data
    value = F.reshape(value, new_value_shape)
    if not tuple_index or special_index == 1:
        data[True] = value
        return data

    empty_broadcast_data_shape = False
    if isinstance(_broadcast_data_shape, Tensor) and _broadcast_data_shape == Tensor([0]):
        empty_broadcast_data_shape = True
    if isinstance(_broadcast_data_shape, tuple) and not _broadcast_data_shape:
        empty_broadcast_data_shape = True
    indices = _tensor_index_setitem(
        data, tuple_index, value, idx_advanced, empty_broadcast_data_shape)

    updates = _generate_updates_from_tensor(
        data, indices, value, const_utils.SET_ITEM_BY_TUPLE_OF_TENSOR)
    if is_parameter(data):
        F.scatter_nd_update(data, indices, updates)
        return data
    return F.tensor_scatter_update(data, indices, updates)

def tensor_itemset_by_tuple_with_tensor(data, tuple_index, value):
    """Assigns the tensor by tuple with tensor value."""
    op_name = const_utils.TENSOR_SETITEM
    tuple_index = _transform_ellipsis_to_slice(data, tuple_index, op_name)

    if const_utils.use_copy_slice(tuple_index) and not const_utils.is_ascend():
        dim1_start, dim1_stop, _ = const_utils.normalize_slice(tuple_index[1], data.shape[1])
        if isinstance(dim1_start, Tensor):
            dim1_start = int(dim1_start)
        if isinstance(dim1_stop, Tensor):
            dim1_stop = int(dim1_stop)
        if dim1_stop - dim1_start <= 0:
            return data
        dim0_start = tuple_index[0] if tuple_index[0] >= 0 else tuple_index[0] + data.shape[0]
        start = (dim0_start, dim1_start)
        stop = (dim0_start + 1, dim1_stop)
        step = (1, 1)
        value_shape = (dim1_stop - dim1_start,) + data.shape[2:]
        value = F.broadcast_to(value, value_shape)
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
    value = sequence_to_tensor(value, F.dtype(data))
    return tensor_setitem_by_tuple_with_tensor(data, tuple_index, value)


def tensor_setitem_by_number_with_number(data, index, value):
    """Assigns the tensor by number with number value."""
    data_shape = F.shape(data)
    dim_size = data_shape[0]
    if index < 0:
        index += dim_size
    if index < -dim_size or index >= dim_size:
        raise IndexError(f'index {index} is out of bounds for axis 0 with size {dim_size}')
    index = F.cast(index, mstype.int64)
    index = F.reshape(index, (1, 1))

    updates = F.cast(value, data.dtype)
    updates_shape = (1,) + data_shape[1:]
    updates = ops.broadcast_to(updates, updates_shape)

    if is_parameter(data):
        F.scatter_nd_update(data, index, updates)
        return data
    return F.tensor_scatter_update(data, index, updates)


def tensor_setitem_by_number_with_sequence(data, index, value):
    """Assigns a list/tuple value to the tensor by slice."""
    value = sequence_to_tensor(value, F.dtype(data))
    return tensor_setitem_by_number_with_tensor(data, index, value)


def tensor_setitem_by_number_with_tensor(data, index, value):
    return tensor_setitem_by_number_with_number(data, index, value)


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

    if len(value_shape) > len(data_shape):
        source_shape = data_shape
    else:
        source_shape = value_shape
    value = F.reshape(value, source_shape)
    data = F.broadcast_to(value, data_shape)
    return data


def tensor_setitem_by_ellipsis_with_sequence(data, value):
    """Assigns a list/tuple value to the tensor by ellipsis."""
    value = sequence_to_tensor(value, F.dtype(data))
    return tensor_setitem_by_ellipsis_with_tensor(data, value)


def tensor_setitem_by_bool(data, index, value):
    """Assigns a value to the tensor by boolean."""
    data_shape = F.shape(data)
    data_dtype = F.dtype(data)
    if not index:
        data_shape = (0,) + data_shape
    if isinstance(value, (list, tuple)):
        value = sequence_to_tensor(value, data_dtype)
    else:
        value = F.cast(value, data_dtype)

    if index:
        value_shape = F.shape(value)
        if len(value_shape) > len(data_shape):
            source_shape = data_shape
        else:
            source_shape = value_shape
        value = F.reshape(value, source_shape)
        data = F.broadcast_to(value, data_shape)
    return data


def tensor_in_sequence(x, y):
    """Assigns whether a sequence contains the given tensor"""
    result = const_utils.scalar_to_tensor(False)
    for i in y:
        if isinstance(i, Tensor) and x.shape == i.shape and x.dtype == i.dtype:
            result = F.logical_or(F.equal(x, i).all(), result)
    return result


@_primexpr
def remove_expanded_dims_parse_bool_tensor_index(index_out, indices_out, shapes, cur_dim):
    """ Parse bool tensor index """
    index_out = index_out.nonzero()
    if index_out.shape[0] == 0:
        return None, shapes, cur_dim
    for i in range(index_out.shape[1]):
        out = index_out[:, i]
        indices_out += (out,)
        shapes.append(F.shape(out))
        cur_dim += 1
    return indices_out, shapes, cur_dim


def remove_expanded_dims_parse_tensor_index(index_out, indices_out, shapes, cur_dim):
    """ Parse tensor index """
    if index_out.dtype == mstype.bool_:
        return remove_expanded_dims_parse_bool_tensor_index(index_out, indices_out, shapes, cur_dim)
    indices_out += (index_out,)
    shapes.append(F.shape(index_out))
    cur_dim += 1
    return indices_out, shapes, cur_dim


def remove_expanded_dims(tuple_index, data_shape, value):
    """Removes expanded dimensions in tuple_index and value."""
    not_expanded_dim = ()
    shapes = []
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
            has_false = has_false or parse_check_slice_index(
                index_out, data_shape[cur_dim])
            cur_dim += 1
        elif isinstance(index_out, (Tensor, bool)):  # advanced index
            if idx_advanced == -1:
                idx_advanced = len(not_expanded_dim)
            elif i - idx_tensor > 1:
                idx_advanced = 0
            idx_tensor = i
            if isinstance(index_out, Tensor):
                indices_out, shapes, cur_dim = \
                    remove_expanded_dims_parse_tensor_index(index_out, indices_out, shapes, cur_dim)
                if indices_out is None:
                    return False, value, 0
                if index_out.dtype != mstype.bool_ and F.rank(index_out) > 0:
                    has_sequence = True
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
        tensor_dtype = const_utils.get_index_tensor_dtype(idx.dtype)
        if tensor_dtype == const_utils.INT_:
            idx = F.select(idx < 0, idx + data_shape[cur_dim], idx)
        elif tensor_dtype == const_utils.BOOL_:
            # index with tensor(bool) type is processed in remove_expanded_dims()
            pass
    return idx


@_primexpr
def _check_shape_mul(shape):
    if F.shape_mul(shape) == 0:
        raise ValueError('zero-size tensors are not supported.')


def reduce_(a, reduce_fn, cmp_fn=None, axis=None, keepdims=False, initial=None, where=True, dtype=None):
    """
    Applies comparison based on cmp_fn and reduction based on reduce_fn.
    If cmp_fn is None, only reduction is performed.
    """

    shape = F.shape(a)
    ndim = F.rank(a)
    if dtype is None:
        dtype = F.dtype(a)
    axes = validator.check_axis_valid(axis, ndim)
    if initial is not None:
        if ((isinstance(initial, Tensor) and F.rank(initial) > 0) or
                not isinstance(initial, (int, float, bool, Tensor))):
            const_utils.raise_type_error('initial must be scalar')

    _check_shape_mul(shape)

    if initial is not None:
        if isinstance(initial, Tensor):
            initial = F.tile(initial, shape).astype(dtype)
        else:
            initial = F.fill(dtype, shape, initial)
        a = cmp_fn(a, initial)

    if where is not None and not isinstance(where, Tensor):
        where = Tensor(where, dtype=mstype.bool_)

    if where is not None and (where.shape or not where):
        if initial is None:
            const_utils.raise_value_error('initial value must be provided for where masks')
        ndim_orig = F.rank(a)
        # broadcasts input tensors
        shape_out = const_utils.infer_out_shape(F.shape(where), F.shape(a), F.shape(initial))
        where = where.astype(mstype.float32)
        where = F.broadcast_to(where, shape_out)
        where = where.astype(mstype.bool_)
        a = F.broadcast_to(a, shape_out)
        initial = F.broadcast_to(initial, shape_out)
        a = F.select(where, a, initial)
        axes = const_utils.real_axes(ndim_orig, F.rank(a), axes)

    return reduce_fn(a, axes).astype(dtype)


setattr(tensor_operator_registry, "reduce", reduce_)


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


setattr(tensor_operator_registry, 'check_indices', check_indices)


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
