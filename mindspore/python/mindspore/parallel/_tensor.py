# Copyright 2023 Huawei Technologies Co., Ltd
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
"""load tensor and combine tensor"""
from __future__ import division
from __future__ import absolute_import

import copy
import numpy as np
from mindspore.common.tensor import Tensor
from mindspore.communication.management import get_rank, get_group_size
from mindspore._c_expression import TensorTransform
from mindspore import log as logger

_tensor_transform = TensorTransform.get_instance()
COMM_TENSOR_CELL_CACHE = {}
RESHARD_OP_MAP_CACHE = {}

def _get_tensor_strategy(dev_mat, tensor_map):
    """
    Get split strategy by device arrangement and tensor map.

    Args:
        dev_mat (list): The device matrix.
        tensor_map (list): The map relation between tensor and devices.

    Returns:
        List, the split strategy with the same size of np_tensor.
    """
    tensor_strategy = []
    for dim in tensor_map:
        if isinstance(dim, (tuple, list)):
            acc_stra = 1
            for i in dim:
                if i != -1:
                    acc_stra *= dev_mat[len(dev_mat) - i - 1]
            tensor_strategy.append(acc_stra)
        else:
            if dim == -1:
                tensor_strategy.append(1)
            else:
                tensor_strategy.append(dev_mat[-dim - 1])
    return tensor_strategy


def _get_tensor_slice_index(device_arrangement, tensor_strategy, tensor_map, rank_index):
    """
    Get the tensor slice index for the local device.

    Args:
        device_arrangement (list): The device matrix.
        tensor_strategy (list): The split strategy with the same size of np_tensor.
        tensor_map (list): The map relation between tensor and devices.
        rank_index (int): The rank of local device.

    Returns:
        Integer, the index of the local device for tensor slices.
    """
    device_coordinate = _rank_to_coordinate(rank_index, device_arrangement)
    device_coordinate_new = _convert_to_new_device_coordinate(device_coordinate, tensor_map)
    tensor_slice_index = _coordinate_to_rank(device_coordinate_new, tensor_strategy)
    return tensor_slice_index


def _rank_to_coordinate(rank_index, device_arrangement):
    """
    Convert rank index to device coordinate.

    Args:
        rank_index (int): The index of the local device.
        device_arrangement (list): The device matrix.

    Returns:
        List, the coordinate for local device in the device matrix
    """
    dim_len = len(device_arrangement)
    device_coordinate = np.zeros(dim_len)
    for i in range(dim_len):
        size = device_arrangement[dim_len - 1 - i]
        device_coordinate[dim_len - 1 - i] = rank_index % size
        rank_index = int(rank_index / size)
    return device_coordinate


def _coordinate_to_rank(device_coordinate, device_arrangement):
    """
    Convert device coordinate to rank index.

    Args:
        device_coordinate (list): The coordinate for local device in the device matrix.
        device_arrangement (list): The device matrix.

    Returns:
        Integer, the index of the local device for tensor slices.
    """
    rank_index = 0
    size = 1
    for i in range(len(device_coordinate)):
        rank_index += size * device_coordinate[len(device_coordinate) - 1 - i]
        size *= device_arrangement[len(device_coordinate) - 1 - i]
    return rank_index


def _convert_to_new_device_coordinate(device_coordinate, tensor_map):
    """
    Convert device_coordinate according to the tensor map.

    Args:
        device_coordinate (list): The coordinate for local device in the device matrix.
        tensor_map (list): The map relation between tensor and devices.

    Returns:
        List, the converted coordinate.
    """
    device_coordinate_new = []
    for i in range(len(tensor_map)):
        if tensor_map[len(tensor_map) - 1 - i] != -1:
            device_coordinate_new.insert(0, device_coordinate[len(device_coordinate) - 1 -
                                                              tensor_map[len(tensor_map) - 1 - i]])
        else:
            device_coordinate_new.insert(0, 0)
    return device_coordinate_new


def _chunk_tensor(np_tensor, strategy, depth):
    """
    Recursive function to chunk tensor.

    Args:
        np_tensor (NDarray): The matrix to be split.
        strategy (list): The split strategy with the same size of np_tensor.
        depth (int): Recursion depth.

    Returns:
        NDarray, the splited matrix.

    Raises:
        ValueError: If np_tensor can not be split by strategy.
    """
    output = []
    axis = len(np_tensor.shape) - depth
    if np_tensor.shape[axis] % strategy[0] != 0:
        raise ValueError("np_tensor can not be split by strategy!")
    ret = list(np.split(np_tensor, strategy[0], axis))
    if depth == 1:
        return ret
    for ret_ in ret:
        output.extend(
            _chunk_tensor(ret_, strategy[len(strategy) - depth + 1:len(strategy)], depth - 1))

    return output


def _chunk_tensor_by_strategy(np_tensor, strategy):
    """
    Split the input by strategy.

    Args:
        np_tensor (NDarray): The matrix to be split.
        strategy (list): The split strategy with the same size of np_tensor.

    Returns:
        NDarray, the splited matrix.

    Raises:
        TypeError: If np_tensor is not ndarray
        ValueError: If the length of np_tensor does not match the length of strategy.
    """
    if not isinstance(np_tensor, np.ndarray):
        raise TypeError("np_tensor should be ndarray!")
    if len(strategy) != len(np_tensor.shape):
        raise ValueError("The length of np_tensor does not match the length of strategy!")
    return _chunk_tensor(np_tensor, strategy, len(strategy))


def _get_slice_index(dev_mat, tensor_map, opt_shard_group):
    """
    Get the slice index for current slice.

    Args:
        dev_mat (list): The device matrix of devices.
        tensor_map (list): The split strategy of tensor.
        opt_shard_group(str): The group of optimizer shard

    Returns:
        Integer, the slice index for slice on this device.
    """
    rank = get_rank()
    dev_num = get_group_size()
    tensor_strategy = _get_tensor_strategy(dev_mat, tensor_map)
    tensor_slice_index = _get_tensor_slice_index(dev_mat, tensor_strategy, tensor_map, rank)
    if opt_shard_group:
        tensor_slice_index += dev_num
        opt_rank = get_rank(opt_shard_group)
        tensor_slice_index += opt_rank
    return tensor_slice_index


def _load_tensor(tensor, dev_mat, tensor_map, full_shape=None, rank_id=-1):
    """
    Get the tensor slice of the local device by the device matrix and the tensor map

    Args:
        tensor (Tensor): The tensor to be split.
        dev_mat (list): The device matrix of devices.
        tensor_map (list): The split strategy of tensor.

    Returns:
        numpy.array, the sliced array.

    Examples:
        >>> tensor = Tensor(np.ones([32, 32]))
        >>> dev_mat = [2, 4]
        >>> tensor_map = [1, -1]
        >>> full_shape = [32, 32]
        >>> tensor_slice = _load_tensor(tensor, dev_mat, tensor_map, full_shape)
    """
    if rank_id == -1:
        rank = get_rank()
    else:
        rank = rank_id
    tensor_strategy = _get_tensor_strategy(dev_mat, tensor_map)
    tensor_slice_index = _get_tensor_slice_index(dev_mat, tensor_strategy, tensor_map, rank)
    np_tensor = tensor.asnumpy()
    if full_shape:
        np_tensor = np_tensor.reshape(full_shape)
    np_tensor_list = _chunk_tensor_by_strategy(np_tensor, tensor_strategy)
    np_tensor_slice = np_tensor_list[int(tensor_slice_index)]
    return np_tensor_slice


def _load_tensor_by_layout(tensor, layout, rank_id):
    """
    Load tensor by layout.

    Args:
        tensor (Tensor): The input tensor.
        layout (list): The tensor layout in auto parallel.

    Returns:
        Tensor, the sliced tensor.

    Raises:
        TypeError: If layout is not list.
        ValueError: If the length of layout is not 3.
    """
    if not isinstance(layout, tuple):
        raise TypeError("The layout should be tuple! layout is {}".format(layout))
    if len(layout) < 7:
        raise ValueError("The length of layout must be larger than 6! layout is {}".format(layout))
    dev_mat = layout[0]
    tensor_map = layout[1]
    slice_shape = layout[2]
    if not tensor_map:
        return tensor
    uniform_split = layout[4]
    group = layout[5]
    full_shape = layout[6]
    if uniform_split == 0:
        raise RuntimeError("The load tensor only support uniform split now")
    tensor_slice = _load_tensor(tensor, dev_mat, tensor_map, full_shape, rank_id)
    if tensor_slice.shape != slice_shape and not group:
        tensor_slice = tensor_slice.reshape(slice_shape)
    if group:
        # get a totally shard tensor slice for parallel optimizer
        rank = get_rank(group)
        size = get_group_size(group)
        if tensor_slice.shape != tuple(slice_shape) and slice_shape:
            slice_shape_extend = copy.deepcopy(slice_shape)
            slice_shape_extend[0] = slice_shape[0] * size
            tensor_slice = tensor_slice.reshape(slice_shape_extend)
        tensor_slice = np.split(tensor_slice, size)[rank]
    return Tensor(tensor_slice, tensor.dtype)


def _reshape_param_data(param_data, dev_mat, tensor_map):
    """
    Combine param slice by the device matrix and the tensor map, used in model parallel scenario.

    Args:
        param_data (Tensor): The tensor to be reshaped, generated from all the device from AllGatherParamNet.
        dev_mat (list): The device matrix of devices.
        tensor_map (list): The split strategy of tensor.

    Returns:
        Tensor, the combined tensor which with the whole data value.

    Examples:
        >>> param_data = _allgather_param_net(param_data)
        >>> dev_mat = [2, 2]
        >>> tensor_map = [1, 0]
        >>> tensor = _reshape_param_data(tensor_slices, dev_mat, tensor_map)
    """

    device_count = 1
    for dim in dev_mat:
        device_count *= dim

    tensor_slices = np.split(param_data.asnumpy(), device_count, axis=0)
    tensor_strategy = _get_tensor_strategy(dev_mat, tensor_map)

    # get the actual number of slices,as: different devices may load the same slice
    slice_count = 1
    for dim in tensor_strategy:
        slice_count *= dim

    # reorder slices and remove duplicates based on device matrix and tensor_map
    tensor_slices_new = list(range(slice_count))
    for i in range(device_count):
        slice_index = _get_tensor_slice_index(dev_mat, tensor_strategy, tensor_map, i)
        tensor_slices_new[int(slice_index)] = np.array(tensor_slices[i])

    # combine slices to generate complete parameter
    dim_len = len(tensor_strategy)
    for i in range(dim_len):
        ele_count = int(len(tensor_slices_new) / tensor_strategy[dim_len - 1 - i])
        tensor_slices_new_inner = []
        for j in range(ele_count):
            new_tensor = tensor_slices_new[j * tensor_strategy[dim_len - 1 - i]]
            for k in range(j * tensor_strategy[dim_len - 1 - i] + 1,
                           (j + 1) * tensor_strategy[dim_len - 1 - i]):
                new_tensor = np.concatenate((new_tensor, tensor_slices_new[k]), axis=dim_len - 1 - i)

            tensor_slices_new_inner.insert(len(tensor_slices_new_inner), np.array(new_tensor))
        tensor_slices_new = tensor_slices_new_inner

    return Tensor(tensor_slices_new[0])


def _extract_layout_item(layout_item):
    dev_matrix = layout_item[0]
    tensor_map = layout_item[1]
    opt_shard_step = layout_item[4]
    opt_shard_size = layout_item[5]
    tensor_strategy = _get_tensor_strategy(dev_matrix, tensor_map)
    model_parallel_shard_size = np.prod(tensor_strategy)
    if opt_shard_size == -1:
        opt_shard_size = np.prod(dev_matrix) // model_parallel_shard_size
    return dev_matrix, tensor_map, opt_shard_step, opt_shard_size


def _transform_tensor_by_layout(from_layout, to_layout, device_list, rank_id, enable_redist_opt=False):
    """
    Transform tensor from source layout to the destination layout.

    Args:
        from_layout (tuple(tuple)): Source tensor layout
        to_layout (tuple(tuple)): Destination tensor layout
        device_list (tuple): The rank list of the tensor distributed.
        rank_id (number): The tensor slice in which rank.
    Returns:
        transform operator list.
    """
    if not isinstance(from_layout, tuple) or not isinstance(to_layout, tuple):
        raise TypeError("The layout should be tuple! layout is {} and {}".format(from_layout, to_layout))
    return _tensor_transform.transform_tensor_sharding(from_layout, to_layout, device_list, enable_redist_opt, rank_id)


def _construct_from_to_tensor_layout(from_full_tensor_shape, from_dev_matrix,
                                     from_tensor_map, to_full_tensor_shape,
                                     to_dev_matrix, to_tensor_map, param_name=''):
    """construct from_layout and to_layout to the same device num"""
    from_full_tensor_shape = list(from_full_tensor_shape)
    to_full_tensor_shape = list(to_full_tensor_shape)
    from_dev_matrix = list(from_dev_matrix)
    from_tensor_map = list(from_tensor_map)
    to_dev_matrix = list(to_dev_matrix)
    to_tensor_map = list(to_tensor_map)
    from_dev_prod = np.prod(from_dev_matrix)
    to_dev_prod = np.prod(to_dev_matrix)
    if len(from_full_tensor_shape) != len(from_tensor_map) or len(to_full_tensor_shape) != len(to_tensor_map):
        raise ValueError(
            f"For param '{param_name}', the tensor map dimensions must match the tensor shape dimensions.\n"
            f"Should be:\n"
            f"  - len(tensor_map) == len(tensor_shape)\n"
            f"But got:\n"
            f"  from_full_tensor_shape = {from_full_tensor_shape}\n"
            f"  from_dev_matrix        = {from_dev_matrix}\n"
            f"  from_tensor_map        = {from_tensor_map}\n"
            f"  to_full_tensor_shape   = {to_full_tensor_shape}\n"
            f"  to_dev_matrix          = {to_dev_matrix}\n"
            f"  to_tensor_map          = {to_tensor_map}\n"
            f"Please check your strategy file."
        )
    if from_dev_prod > to_dev_prod:
        if from_dev_prod % to_dev_prod != 0:
            raise ValueError("Cannot transform device_num from {} to {}".format(from_dev_prod, to_dev_prod))
        repeat_dim_size = from_dev_prod // to_dev_prod
        to_dev_matrix.insert(0, repeat_dim_size)
    elif from_dev_prod < to_dev_prod:
        if to_dev_prod % from_dev_prod != 0:
            raise ValueError("Cannot transform device_num from {} to {}".format(from_dev_prod, to_dev_prod))
        repeat_dim_size = to_dev_prod // from_dev_prod
        from_dev_matrix.insert(0, repeat_dim_size)
    from_tensor_layout = (from_dev_matrix, from_tensor_map, from_full_tensor_shape)
    to_tensor_layout = (to_dev_matrix, to_tensor_map, to_full_tensor_shape)
    return from_tensor_layout, to_tensor_layout


def _expand_layout(dev_matrix, tensor_map, tensor_shape):
    """
    expand nested tensor_map and reshape tensor shape according to tensor_map
    dev_matrix = [4, 2, 2]
    tensor_map = [[2, 1], 0]
    tensor_shape = [8, 8]
    =>
    expanded_tensor_map = [2, 1, 0]
    expanded_tensor_map = [4, 8/4, 8]
    """
    new_tensor_map = []
    new_tensor_shape = []
    for index, dim in enumerate(tensor_map):
        if isinstance(dim, (tuple, list)):
            accu_shape = 1
            for i in range(len(dim) - 1):
                new_tensor_map.append(dim[i])
                new_tensor_shape.append(dev_matrix[len(dev_matrix) - 1 - dim[i]])
                accu_shape *= dev_matrix[len(dev_matrix) - 1 - dim[i]]
            new_tensor_map.append(dim[-1])
            new_tensor_shape.append(tensor_shape[index] // accu_shape)
        else:
            new_tensor_map.append(dim)
            new_tensor_shape.append(tensor_shape[index])
    return dev_matrix, new_tensor_map, new_tensor_shape


def _construct_tensor_layout_for_opt_shard_by_layout(dev_matrix, tensor_map, opt_shard_step, opt_shard_size,
                                                     origin_full_tensor_shape):
    """
    Construct tensor layout for optimizer parallel when using layout.
    For example, For Tensor with shape (4,2)
    dev_matrix = [2, 2, 2, 2]
    tensor_map = [[1, 0], -1]
    opt_shard_size = 2
    ==>
    dev_matrix = [2, 2, 2, 2]
    tensor_map = [[1, 0], 2, -1]
    the new strategy is [4, 2, 1]
    the tensor_shape should reshape to (model_parallel_size, -1, xx, xx)
    first 4 means the model parallel sharding of data_dim
    second 2 means the opt sharding of data_dim.
    """
    if opt_shard_step == 0 or opt_shard_size == 0:
        return dev_matrix, tensor_map, list(origin_full_tensor_shape)
    tensor_strategy = _get_tensor_strategy(dev_matrix, tensor_map)
    repeated_dim = []
    dev_sharded_index = []
    dev_matrix, expanded_tensor_map, _ = _expand_layout(dev_matrix, tensor_map, origin_full_tensor_shape)
    for dim in expanded_tensor_map:
        if dim != -1:
            dev_sharded_index.append(len(dev_matrix) - dim - 1)
    for index, value in enumerate(dev_matrix):
        if index not in dev_sharded_index and value > 1:
            repeated_dim.append(index)
    if not repeated_dim:
        raise ValueError("The device_matrix {} and tensor_map {} cannot sharding opt_shard".
                         format(dev_matrix, tensor_map))
    return _construct_tensor_layout_helper(dev_matrix, tensor_map, opt_shard_size, origin_full_tensor_shape,
                                           tensor_strategy, repeated_dim)


def _construct_tensor_layout_helper(dev_matrix, tensor_map, opt_shard_size, origin_full_tensor_shape,
                                    tensor_strategy, repeated_dim):
    """
    helper function to assign repeated device_matrix dim for opt shard.
    """
    new_dev_matrix = list(copy.deepcopy(dev_matrix))
    new_dev_matrix_map = list(range(len(dev_matrix)))
    opt_shard_dim = []
    remained_opt_shard_size = opt_shard_size if opt_shard_size != -1 else \
        int(np.prod([dev_matrix[i] for i in repeated_dim]))
    for dim in repeated_dim[::-1]:
        opt_sharding_size = dev_matrix[dim]
        if remained_opt_shard_size // opt_sharding_size == 0:
            if opt_sharding_size % remained_opt_shard_size != 0:
                raise ValueError("dev_matrix value {} at dim {} cannot be divided by needed opt sharding "
                                 "size {}".format(dev_matrix[dim], len(dev_matrix) - dim - 1,
                                                  remained_opt_shard_size))
            opt_sharding_size = remained_opt_shard_size
            # update dev_matrix
            new_dev_matrix[dim] = dev_matrix[dim] // opt_sharding_size
            new_dev_matrix.insert(dim + 1, opt_sharding_size)
            for i in range(len(dev_matrix) - dim - 1, len(dev_matrix)):
                new_dev_matrix_map[i] += 1
        if remained_opt_shard_size % opt_sharding_size != 0:
            raise ValueError("Remained opt_shard_size {} cannot be divided by current sharding size {}, "
                             "the repeat dim is {} with dev_matrix value {}".
                             format(remained_opt_shard_size, opt_sharding_size,
                                    len(dev_matrix) - dim - 1, dev_matrix[dim]))
        remained_opt_shard_size //= opt_sharding_size
        opt_shard_dim.insert(0, dim)
        if remained_opt_shard_size == 1:
            break
    tensor_map_new = list(copy.deepcopy(tensor_map))
    if len(new_dev_matrix) != len(dev_matrix):
        opt_shard_dim = list(map(lambda x: x + 1, opt_shard_dim))
        for index, item in enumerate(tensor_map_new):
            if isinstance(item, (tuple, list)):
                item = list(map(lambda x: new_dev_matrix_map[x] if x >= 0 else x, item))
                tensor_map_new[index] = item
            else:
                if item >= 0:
                    tensor_map_new[index] = new_dev_matrix_map[item]
    tensor_shape_new = list(copy.deepcopy(origin_full_tensor_shape))
    tensor_shape_new[0] = tensor_strategy[0]
    first_dim_no_sharding_size = origin_full_tensor_shape[0] // tensor_strategy[0]
    accu_shape = 1
    for i in range(len(opt_shard_dim) - 1):
        opt_sharding_size = new_dev_matrix[opt_shard_dim[i]]
        tensor_shape_new.insert(i + 1, opt_sharding_size)
        accu_shape = accu_shape * opt_sharding_size
    tensor_shape_new.insert(len(opt_shard_dim), first_dim_no_sharding_size // accu_shape)
    for index, r_dim in enumerate(opt_shard_dim):
        tensor_map_new.insert(index + 1, len(new_dev_matrix) - r_dim - 1)
    return list(new_dev_matrix), tensor_map_new, tensor_shape_new


def _construct_tensor_layout_for_opt_shard(dev_matrix, tensor_map, opt_shard_step, opt_shard_size,
                                           origin_full_tensor_shape):
    """
    dev_mat = [4, 2, 2]
    tensor_map = [2, 1, 0]
    opt_size = 2
    =>
    dev_mat = [opt_size, 4, 2, 2] = [2, 4, 2, 2]
    tensor_map = [2, 3, 1, 0]
    thus new_strategy = [4, 2, 2, 2]
    the tensor_shape should reshape to (model_parallel_size, -1, xx, xx)
    first 4 means the model parallel sharding of data_dim
    second 2 means the opt sharding of data_dim
    And the model parallel sharding dim is the right of opt sharding dim, so it would be 0-1-2-3 model parallel sharding
    then 0-4 optimizer sharding.
    """
    has_layout = any(isinstance(i, (list, tuple)) for i in tensor_map)
    if has_layout:
        output = _construct_tensor_layout_for_opt_shard_by_layout(dev_matrix, tensor_map, opt_shard_step,
                                                                  opt_shard_size, origin_full_tensor_shape)
        return _expand_layout(*output)

    if opt_shard_step == 0 or opt_shard_size == 0:
        return dev_matrix, tensor_map, list(origin_full_tensor_shape)
    tensor_strategy = _get_tensor_strategy(dev_matrix, tensor_map)
    repeated_dim = []
    dev_sharded_index = []
    for dim in tensor_map:
        if dim != -1:
            dev_sharded_index.append(len(dev_matrix) - dim - 1)
    for index, value in enumerate(dev_matrix):
        if index not in dev_sharded_index and value > 1:
            repeated_dim.append(index)
    if not repeated_dim:
        raise ValueError("The device_matrix {} and tensor_map {} cannot sharding opt_shard".
                         format(dev_matrix, tensor_map))
    if len(repeated_dim) == 1 and np.prod(dev_matrix[repeated_dim[0] + 1:]) != opt_shard_step:
        raise ValueError("The optimizer sharding step {} is not equal to the model parallel sharding size {}.".
                         format(opt_shard_step, np.prod(dev_matrix[repeated_dim[0] + 1:])))
    first_dim_no_sharding_size = origin_full_tensor_shape[0] // tensor_strategy[0]
    if (len(repeated_dim) < len(dev_matrix) and len(repeated_dim) > 1) or repeated_dim[0] > 0:
        return _construct_tensor_layout_helper(dev_matrix, tensor_map, opt_shard_size, origin_full_tensor_shape,
                                               tensor_strategy, repeated_dim)

    full_tensor_shape = list(origin_full_tensor_shape)
    full_tensor_shape[0] = tensor_strategy[0]
    full_tensor_shape.insert(1, first_dim_no_sharding_size)
    new_dev_matrix = tensor_strategy
    repeat_dim = np.prod(dev_matrix) // (opt_shard_step * opt_shard_size)

    new_tensor_map = []
    for idx, val in enumerate(tensor_strategy):
        if val == 1:
            new_tensor_map.append(-1)
        else:
            new_tensor_map.append(len(tensor_strategy) - 1 - idx)
    new_tensor_map.insert(1, len(tensor_strategy))
    new_dev_matrix.insert(0, opt_shard_size)
    if repeat_dim > 1:
        new_dev_matrix.insert(0, repeat_dim)
    return new_dev_matrix, new_tensor_map, full_tensor_shape


def _get_needed_rank_list_by_layouts(from_tensor_layout, to_tensor_layout, device_list, self_rank):
    """
    AllGather op: {op_name, group_ranks + axis}
    """
    result_map = _get_needed_rank_transform_operator_map_by_layouts(from_tensor_layout, to_tensor_layout, device_list,
                                                                    self_rank)
    result_list = list(result_map.keys())
    result_list.sort()
    return result_list


def _get_needed_rank_transform_operator_map_by_layouts(from_tensor_layout, to_tensor_layout, device_list, self_rank,
                                                       enable_redist_opt=False):
    """
    AllGather op: {op_name, group_ranks + axis}
    """
    stack = []
    index = 0
    transform_operators = _transform_tensor_by_layout(from_tensor_layout, to_tensor_layout, device_list, self_rank,
                                                      enable_redist_opt)
    result_map = {self_rank: transform_operators}
    for operators in transform_operators:
        op_name = operators[0]
        if op_name == "AllConcat":
            groups = operators[1][:-1]
            stack.append((index, groups))
            index += 1
    while stack:
        group_info = stack.pop()
        for rank in group_info[1]:
            if rank not in result_map:
                new_transform_operators = _transform_tensor_by_layout(from_tensor_layout, to_tensor_layout,
                                                                      device_list, rank, enable_redist_opt)
                result_map[rank] = new_transform_operators
                index = 0
                for operators in new_transform_operators:
                    op_name = operators[0]
                    if op_name == "AllConcat" and index < group_info[0]:
                        groups = operators[1][:-1]
                        stack.insert(0, (index, groups))
                        index += 1
    return result_map


def _generate_transform_operator_stack(transform_operators_map, self_rank):
    """
    return (rank_id, index, operator)
    """
    if self_rank not in transform_operators_map:
        raise ValueError("The transform operators of rank id {} is required.".format(self_rank))
    if not transform_operators_map[self_rank]:
        return []
    init_level = len(transform_operators_map[self_rank]) - 1
    handle_queue = [(self_rank, init_level, transform_operators_map[self_rank][init_level])]
    result_queue = []
    while handle_queue:
        queue_front = handle_queue.pop(0)
        result_queue.append(queue_front)
        current_rank_id = queue_front[0]
        level = queue_front[1]
        current_operator = queue_front[2]
        if level >= 1:
            if current_operator[0] == "AllConcat":
                current_group = current_operator[1][:-1]
                for rank_id in current_group:
                    handle_queue.append((rank_id, level - 1, transform_operators_map[rank_id][level - 1]))
            else:
                handle_queue.append((current_rank_id, level - 1, transform_operators_map[current_rank_id][level - 1]))
    return result_queue


def _apply_tensor_transform_operators(transform_operator_stack, tensor_dict, device_num):
    """
    transform_operator_stack: [...(rank_id, index, operator)]
    """
    if not transform_operator_stack:
        return
    level = transform_operator_stack[-1][1]
    level_operators = []
    while True:
        if not transform_operator_stack or (level != transform_operator_stack[-1][1]):
            tmp_tensor_dict = {}
            if not level_operators:
                continue
            op_name = level_operators[0][2][0]
            for operator_pair in level_operators:
                rank_id = operator_pair[0]
                if rank_id % device_num not in tensor_dict:
                    raise ValueError("The checkpoint file of rank {} is missing.".format(rank_id % device_num))
                cur_level = operator_pair[1]
                operator = operator_pair[2]
                if operator[0] != op_name:
                    raise ValueError("The operator in the same level should be equal in the transform tensor operator "
                                     "list, but the find {} and {} in level {}".format(op_name, operator[0], cur_level))
                if operator[0] != "AllConcat":
                    tensor_dict[rank_id % device_num] = _apply_operator(operator[0])(tensor_dict[rank_id % device_num],
                                                                                     operator)
                    continue
                for rank in operator[1][:-1]:
                    if rank % device_num not in tensor_dict:
                        raise ValueError("The checkpoint file of rank {} is missing.".format(rank % device_num))
                allgather_list = [tensor_dict[rank % device_num] for rank in operator[1][:-1]]
                tmp_tensor_dict[rank_id % device_num] = _apply_operator(operator[0])(allgather_list, operator)
            if op_name == "AllConcat":
                for rank, value in tmp_tensor_dict.items():
                    tensor_dict[rank % device_num] = value
            level_operators.clear()
        if not transform_operator_stack:
            break
        operator_pair = transform_operator_stack.pop()
        level = operator_pair[1]
        level_operators.append(operator_pair)


def _check_operator(operator):
    if not isinstance(operator, tuple):
        raise TypeError("The operator should be a list.")
    if len(operator) != 2:
        raise TypeError("The operator should contains 2 item.")
    if not isinstance(operator[1], list):
        raise TypeError("The operator[1] should be list.")


def _apply_operator(operator_name):
    """apply transform operator"""

    def _apply_reshape_operator(numpy_data, reshape_op):
        """
        Apply reshape operator.

        Args:
            numpy_data (numpy.ndarray): The data of tensor to apply operator.
            reshape_op (tuple): reshape operator information, the second item is the destination shape.
        Returns:
            The data of tensor after apply operator.
        """
        if not isinstance(numpy_data, np.ndarray):
            raise TypeError("The data should be a numpy.ndarray.")
        _check_operator(reshape_op)
        return np.reshape(numpy_data, reshape_op[1])

    def _apply_allconcat_operator(numpy_data_list, allgather_op):
        """
        Apply allconcat operator.

        Args:
            numpy_data (numpy.ndarray): The data of tensor to apply operator.
            allgather_op (tuple): allgather operator information.
              the second item is the allgather info, contains group and axis.
        Returns:
            The data of tensor after apply operator.
        """
        if not isinstance(numpy_data_list, list):
            raise TypeError("The data_list should be a list.")
        new_numpy_data_list = []
        for numpy_data in numpy_data_list:
            new_numpy_data_list.append(numpy_data)
        numpy_data_list = new_numpy_data_list
        _check_operator(allgather_op)
        concat_group = allgather_op[1][:-1]
        if len(concat_group) != len(numpy_data_list):
            raise ValueError("The length of data_list {} should be equal to concat_group size {}".
                             format(len(numpy_data_list), len(concat_group)))
        concat_axis = allgather_op[1][-1]
        return np.concatenate(numpy_data_list, concat_axis)

    def _apply_slice_operator(numpy_data, slice_op):
        """
        Apply reshape operator.

        Args:
            numpy_data (numpy.ndarray): The data of tensor to apply operator.
            slice_op (tuple): slice operator information, the second item is the slice information.
        Returns:
            The data of tensor after apply operator.
        """
        _check_operator(slice_op)
        if len(slice_op[1]) % 3 != 0:
            raise ValueError("The slice operator information is wrong.")
        shape_size = len(slice_op[1]) // 3
        begin = slice_op[1][:shape_size]
        end = slice_op[1][shape_size:shape_size * 2]
        stride = slice_op[1][shape_size * 2:]
        slice_index = []
        for begin_i, end_i, strides_i in zip(begin, end, stride):
            s = slice(begin_i, end_i, strides_i)
            slice_index.append(s)
        slice_index = tuple(slice_index)
        return numpy_data[slice_index]

    _apply_operator_map = {"Reshape": _apply_reshape_operator, "StridedSlice": _apply_slice_operator,
                           "AllConcat": _apply_allconcat_operator}
    return _apply_operator_map.get(operator_name)


def _reshape_param_data_with_weight(param_data, dev_mat, field_size):
    """
    Combine param slice by the device matrix, used in model parallel scenario.

    Args:
        param_data (Tensor): The tensor to be reshaped and rearrangement,
        generated from all the device from AllGatherParamNet.
        dev_mat (list): The device matrix of devices.
    Returns:
        Tensor, the combined tensor which with the whole data value.

    Examples:
        >>> param_data = _allgather_param_net(param_data)
        >>> dev_mat = [2, 2]
        >>> field_size = [39]
        >>> tensor = _reshape_param_data_with_weight(param_data, dev_mat, field_size)
    """
    device_count = 1
    for dim in dev_mat:
        device_count *= dim

    tensor_slices = np.split(param_data.asnumpy(), device_count, axis=0)
    tensor_slices_col = []
    for i in range(len(tensor_slices[0][0])):
        tensor_slices_new = np.array(tensor_slices[0][:, i]).reshape(field_size, -1)
        for j in range(1, device_count):
            tensor_slices_new = np.concatenate((tensor_slices_new, \
                                                np.array(tensor_slices[j][:, i]).reshape(field_size, -1)), axis=1)
        tensor_slices_col.append(tensor_slices_new)
    new_tensor = np.array(tensor_slices_col[0]).reshape(-1, 1)
    for i in range(1, len(tensor_slices_col)):
        new_tensor = np.concatenate((new_tensor, np.array(tensor_slices_col[i]).reshape(-1, 1)), axis=1)
    return Tensor(new_tensor)


def _load_tensor_shape(dev_mat, tensor_map, full_shape=None, rank_id=-1):
    """get tensor shape by slice"""
    if rank_id == -1:
        rank = get_rank()
    else:
        rank = rank_id
    tensor_strategy = _get_tensor_strategy(dev_mat, tensor_map)
    tensor_slice_index = _get_tensor_slice_index(dev_mat, tensor_strategy, tensor_map, rank)
    np_tensor_list = _chunk_shape_by_strategy(full_shape, tensor_strategy)
    np_tensor_slice_index = np_tensor_list[int(tensor_slice_index)]
    res = []
    for index in np_tensor_slice_index:
        res.append(slice(index[0], index[1]))
    return tuple(res)


def _count_tensor_shape(dev_mat, tensor_map, full_shape=None, rank_id=-1):
    """get tensor shape"""
    if rank_id == -1:
        rank = get_rank()
    else:
        rank = rank_id
    tensor_strategy = _get_tensor_strategy(dev_mat, tensor_map)
    tensor_slice_index = _get_tensor_slice_index(dev_mat, tensor_strategy, tensor_map, rank)
    np_tensor_list = _chunk_shape_by_strategy(full_shape, tensor_strategy)
    np_tensor_slice_index = np_tensor_list[int(tensor_slice_index)]
    res = []
    for index in np_tensor_slice_index:
        res.append(index[1] - index[0])
    return res


def _load_tensor_shape_by_layout(tensor, layout, rank_id):
    """get tensor shape by layout"""
    if not isinstance(layout, tuple):
        raise TypeError("The layout should be tuple! layout is {}".format(layout))
    if len(layout) < 7:
        raise ValueError("The length of layout must be larger than 6! layout is {}".format(layout))
    slice_shape = layout[2]
    if slice_shape:
        return slice_shape
    tensor_map = layout[1]
    if not tensor_map:
        return tensor.shape
    dev_mat = layout[0]
    uniform_split = layout[4]
    group = layout[5]
    full_shape = layout[6]
    if not full_shape:
        full_shape = tensor.shape
    if uniform_split == 0:
        raise RuntimeError("The load tensor only support uniform split now")
    tensor_slice_shape = _count_tensor_shape(dev_mat, tensor_map, full_shape, rank_id)
    if group:
        # get a totally shard tensor slice for parallel optimizer
        size = get_group_size(group)
        tensor_slice_shape[0] //= size
    return tensor_slice_shape


def _chunk_shape_by_strategy(full_shape, strategy):
    """chunk shape by strategy"""
    shape = []
    for i in full_shape:
        shape.append([0, i])
    return _chunk_shape(shape, strategy, len(strategy))


def _chunk_shape(np_tensor, strategy, depth):
    """_chunk shape"""
    output = []
    axis = len(np_tensor) - depth
    left, right = np_tensor[axis]
    num = strategy[0]
    chunk_size = (right - left) / num
    append = [[i, int(i + chunk_size)] for i in range(left, right) if i % chunk_size == 0]
    np_tensor_new = []
    for i in append:
        np_tensor_tmp = copy.deepcopy(np_tensor)
        np_tensor_tmp[axis] = i
        np_tensor_new.append(np_tensor_tmp)
    if depth == 1:
        return np_tensor_new
    for ret_ in np_tensor_new:
        output.extend(
            _chunk_shape(ret_, strategy[len(strategy) - depth + 1:len(strategy)], depth - 1))
    return output


def _infer_pp_op_map(from_layout, to_layout, self_rank):
    """
    get the ops map for merging pp stages
    """
    from_rank_list = from_layout[3]
    to_rank_list = to_layout[3]
    from_dev_num_in_stage = len(from_rank_list)
    current_rank_stage_id = self_rank // from_dev_num_in_stage
    diff_rank_id = [
        rank_id for rank_id in to_rank_list if rank_id not in from_rank_list]
    end_stage = from_dev_num_in_stage * (current_rank_stage_id + 1)
    start_stage = from_dev_num_in_stage * current_rank_stage_id
    rank_pos_in_stage = list(range(start_stage, end_stage)).index(self_rank)
    root_idx = from_rank_list[rank_pos_in_stage]
    broadcast_rank_list = [root_idx]
    while rank_pos_in_stage < len(diff_rank_id):
        broadcast_rank_list.append(diff_rank_id[rank_pos_in_stage])
        rank_pos_in_stage += from_dev_num_in_stage
    broadcast_rank_list.sort()
    broadcast_map = {rank_id: [('Broadcast', root_idx, broadcast_rank_list)] for rank_id in broadcast_rank_list}
    return broadcast_map


def _get_pipeline_operator_map(from_layout, to_layout, self_rank):
    """
    If src_pp_stages is greater than dst_pp_stages, the weights of the corresponding cards need to
    be communicated via broadcast to swap. Need to communicate src rank0's 01 to src rank2,
    so that rank2 holds param0's data. Similarly, communicate rank1's 02 to rank3
    rank0 01           01 11
    rank1 02           02 12
    pp2 ------->  pp1
    rank2 11           03 13
    rank3 12           04 14

    Args:
        from_layout (tuple): Use tuple to present layout
          (device_matrix(list), tensor_map(list), global_shape(list), rank_list(list))
        to_layout (tuple): Use tuple to present layout
          (device_matrix(list), tensor_map(list), global_shape(list), rank_list(list))
        self_rank (int): rank_id
    """
    if len(from_layout[3]) < len(to_layout[3]):
        logger.debug(f"from {from_layout} to {to_layout} need to broadcast data across pp stages")
        comm_tensor_cache_key = (
            f"{from_layout[0]}, {from_layout[1]}, {from_layout[2]}, {from_layout[3]}"
            f" -> "
            f"{to_layout[0]}, {to_layout[1]}, {from_layout[2]}, {to_layout[3]}")

        if comm_tensor_cache_key not in COMM_TENSOR_CELL_CACHE:
            logger.debug(f"comm_tensor_cache_key is {comm_tensor_cache_key}, not match cache")
            broadcast_map = _infer_pp_op_map(from_layout, to_layout, self_rank)
            broadcast_op_map_dict = {rank_id: broadcast_map for rank_id in broadcast_map.keys()}
            COMM_TENSOR_CELL_CACHE[comm_tensor_cache_key] = broadcast_op_map_dict
        else:
            comm_tensor_cache_key_rank_list = COMM_TENSOR_CELL_CACHE[comm_tensor_cache_key]
            if self_rank in comm_tensor_cache_key_rank_list:
                logger.debug(f"comm_tensor_cache_key is {comm_tensor_cache_key}, match cache")
                broadcast_map = comm_tensor_cache_key_rank_list[self_rank]
            else:
                logger.debug(f"comm_tensor_cache_key is {comm_tensor_cache_key}, but rank {self_rank} not match cache")
                broadcast_map = _infer_pp_op_map(from_layout, to_layout, self_rank)
                for rank_id in broadcast_map.keys():
                    COMM_TENSOR_CELL_CACHE[comm_tensor_cache_key][rank_id] = broadcast_map
        return broadcast_map
    logger.debug(f"from {from_layout} to {to_layout} no need to broadcast data across pp stages")
    return {}


def _is_multi_shard(in_tensor_map):
    """
    whether the input tensor map is in multi shard
    """
    for tensor_map in in_tensor_map:
        if isinstance(tensor_map, (list, tuple)) and len(tensor_map) > 1:
            return True
    return False


def _insert_expand_layout_reshape(param_rank_map, from_info_tuple, to_info_tuple,
                                  insert_from_reshape, insert_to_reshape):
    """ insert layout expand op reshape """
    from_dev_matrix = from_info_tuple[0]
    from_tensor_map = from_info_tuple[1]
    from_full_tensor_shape = from_info_tuple[2]
    to_dev_matrix_origin = to_info_tuple[0]
    to_tensor_map_origin = to_info_tuple[1]
    origin_tensor_shape = to_info_tuple[2]
    for param_rank, _ in param_rank_map.items():
        if insert_from_reshape:
            from_slice_tensor_shape = ()
            from_tensor_strategy = _get_tensor_strategy(from_dev_matrix, from_tensor_map)
            for i, item in enumerate(from_full_tensor_shape):
                from_slice_tensor_shape += (item // from_tensor_strategy[i],)
            param_rank_map.get(param_rank).insert(0, ('Reshape', list(from_slice_tensor_shape)))
        if insert_to_reshape:
            to_tensor_strategy = _get_tensor_strategy(to_dev_matrix_origin, to_tensor_map_origin)
            to_slice_tensor_shape = ()
            for i, item in enumerate(origin_tensor_shape):
                to_slice_tensor_shape += (item // to_tensor_strategy[i],)
            param_rank_map.get(param_rank).append(('Reshape', list(to_slice_tensor_shape)))


def _infer_reshard_op_map(from_layout, to_layout, self_rank):
    """infer reshard op map"""
    from_layout_without_rank_list = from_layout[:-1]
    to_layout_without_rank_list = to_layout[:-1]
    if _is_multi_shard(from_layout[1]):
        # ((2, 1), 1) --> (2, 1, 1) expand tensormap
        new_layout = _expand_layout(from_layout[0], from_layout[1], from_layout[2])
        from_layout_without_rank_list = (new_layout[0], new_layout[1], new_layout[2])
    if _is_multi_shard(to_layout[1]):
        new_layout = _expand_layout(to_layout[0], to_layout[1], to_layout[2])
        to_layout_without_rank_list = (new_layout[0], new_layout[1], new_layout[2])
    operator_map = _get_needed_rank_transform_operator_map_by_layouts(from_layout_without_rank_list,
                                                                      to_layout_without_rank_list,
                                                                      from_layout[3], self_rank,
                                                                      True)
    new_to_layout_info = to_layout[:-1]
    _insert_expand_layout_reshape(operator_map, from_layout_without_rank_list, new_to_layout_info,
                                  _is_multi_shard(from_layout[1]), _is_multi_shard(to_layout[1]))
    return operator_map


def _get_resharding_operator_map(from_layout, to_layout, self_rank):
    """
        Args:
        from_layout (tuple): Use tuple to present layout
          (device_matrix(list), tensor_map(list), global_shape(list), rank_list(list))
        to_layout (tuple): Use tuple to present layout
          (device_matrix(list), tensor_map(list), global_shape(list), rank_list(list))
        self_rank (int): rank_id
    """
    reshard_op_cache_key = (
        f"{from_layout[0]}, {from_layout[1]}, {from_layout[2]}, {from_layout[3]}"
        f" -> "
        f"{to_layout[0]}, {to_layout[1]}, {from_layout[2]}, {to_layout[3]}")
    if reshard_op_cache_key not in RESHARD_OP_MAP_CACHE:
        operator_map = _infer_reshard_op_map(from_layout, to_layout, self_rank)
        op_map_dict = {rank_id: operator_map for rank_id in operator_map}
        RESHARD_OP_MAP_CACHE[reshard_op_cache_key] = op_map_dict
        logger.debug(f"reshard_op_cache_key is {reshard_op_cache_key}, not match cache")
    else:
        cache_rank_list_dict = RESHARD_OP_MAP_CACHE[reshard_op_cache_key]
        if self_rank in cache_rank_list_dict:
            operator_map = cache_rank_list_dict[self_rank]
            logger.debug(f"reshard_op_cache_key is {reshard_op_cache_key}, match cache")
        else:
            logger.debug(f"reshard_op_cache_key is {reshard_op_cache_key}, "
                         f"but rank {self_rank} is not match cache")
            operator_map = _infer_reshard_op_map(from_layout, to_layout, self_rank)
            for rank_id in operator_map:
                RESHARD_OP_MAP_CACHE[reshard_op_cache_key][rank_id] = operator_map
    return operator_map
