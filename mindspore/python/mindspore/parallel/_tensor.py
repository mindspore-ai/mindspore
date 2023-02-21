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
"""load tensor and combine tensor"""
from __future__ import division
from __future__ import absolute_import

import numpy as np

from mindspore.common.tensor import Tensor
from mindspore.communication.management import get_rank, get_group_size
from mindspore._c_expression import TensorTransform

_tensor_transform = TensorTransform.get_instance()


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
        if dim == -1:
            tensor_strategy.append(1)
        else:
            tensor_strategy.append(dev_mat[-dim-1])
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
        opt_shard_group(string): The group of optimizer shard

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


def _load_tensor(tensor, dev_mat, tensor_map):
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
        >>> tensor_slice = _load_tensor(tensor, dev_mat, tensor_map)
    """
    rank = get_rank()
    tensor_strategy = _get_tensor_strategy(dev_mat, tensor_map)
    tensor_slice_index = _get_tensor_slice_index(dev_mat, tensor_strategy, tensor_map, rank)
    np_tensor = tensor.asnumpy()
    np_tensor_list = _chunk_tensor_by_strategy(np_tensor, tensor_strategy)
    np_tensor_slice = np_tensor_list[int(tensor_slice_index)]
    return np_tensor_slice


def _load_tensor_by_layout(tensor, layout):
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
    if len(layout) < 6:
        raise ValueError("The length of layout must be larger than 5! layout is {}".format(layout))
    dev_mat = layout[0]
    tensor_map = layout[1]
    uniform_split = layout[4]
    group = layout[5]
    if uniform_split == 0:
        raise RuntimeError("The load tensor only support uniform split now")
    if tensor.size == 1:
        return tensor
    tensor_slice = _load_tensor(tensor, dev_mat, tensor_map)
    if group:
        # get a totally shard tensor slice for parallel optimizer
        rank = get_rank(group)
        size = get_group_size(group)
        tensor_slice = np.split(tensor_slice, size)[rank]
    return Tensor(tensor_slice)


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
    if opt_shard_size == -1:
        opt_shard_size = np.prod(dev_matrix) // opt_shard_step
    return dev_matrix, tensor_map, opt_shard_step, opt_shard_size


def _transform_tensor_by_layout(from_layout, to_layout, device_list, rank_id):
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
    return _tensor_transform.transform_tensor_sharding(from_layout, to_layout, device_list, rank_id)


def _construct_from_to_tensor_layout(from_full_tensor_shape, from_dev_matrix,
                                     from_tensor_map, to_full_tensor_shape,
                                     to_dev_matrix, to_tensor_map):
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
        raise ValueError("The tensor map dimensions should be equal to tensor shape dimensions, "
                         "please check strategy file.")
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

    if opt_shard_step == 0 or opt_shard_size == 0:
        return dev_matrix, tensor_map, list(origin_full_tensor_shape)
    tensor_strategy = _get_tensor_strategy(dev_matrix, tensor_map)
    model_parallel_shard_size = np.prod(tensor_strategy)
    if model_parallel_shard_size != opt_shard_step:
        raise ValueError("The optimizer sharding step {} is not equal to the model parallel sharding size {}.".
                         format(opt_shard_step, model_parallel_shard_size))

    first_dim_no_sharding_size = origin_full_tensor_shape[0] // tensor_strategy[0]
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


def _get_needed_rank_transform_operator_map_by_layouts(from_tensor_layout, to_tensor_layout, device_list, self_rank):
    """
    AllGather op: {op_name, group_ranks + axis}
    """
    stack = []
    index = 0
    transform_operators = _transform_tensor_by_layout(from_tensor_layout, to_tensor_layout, device_list, self_rank)
    result_map = {self_rank: transform_operators}
    for operators in transform_operators:
        op_name = operators[0]
        if op_name == "AllGather":
            groups = operators[1][:-1]
            stack.append((index, groups))
            index += 1
    while stack:
        group_info = stack.pop()
        for rank in group_info[1]:
            if rank not in result_map:
                new_transform_operators = _transform_tensor_by_layout(from_tensor_layout, to_tensor_layout,
                                                                      device_list, rank)
                result_map[rank] = new_transform_operators
                index = 0
                for operators in new_transform_operators:
                    op_name = operators[0]
                    if op_name == "AllGather" and index < group_info[0]:
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
            if current_operator[0] == "AllGather":
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
                if operator[0] != "AllGather":
                    tensor_dict[rank_id % device_num] = _apply_operator(operator[0])(tensor_dict[rank_id % device_num],
                                                                                     operator)
                    continue
                for rank in operator[1][:-1]:
                    if rank % device_num not in tensor_dict:
                        raise ValueError("The checkpoint file of rank {} is missing.".format(rank % device_num))
                allgather_list = [tensor_dict[rank % device_num] for rank in operator[1][:-1]]
                tmp_tensor_dict[rank_id % device_num] = _apply_operator(operator[0])(allgather_list, operator)
            if op_name == "AllGather":
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
        for numpy_data in numpy_data_list:
            if not isinstance(numpy_data, np.ndarray):
                raise TypeError("The data should be a numpy.ndarray.")
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
        if not isinstance(numpy_data, np.ndarray):
            raise TypeError("The data should be a numpy.ndarray.")
        _check_operator(slice_op)
        if len(slice_op[1]) % 3 != 0:
            raise ValueError("The slice operator information is wrong.")
        shape_size = len(slice_op[1]) // 3
        begin = slice_op[1][:shape_size]
        end = slice_op[1][shape_size:shape_size*2]
        stride = slice_op[1][shape_size*2:]
        slice_index = []
        for begin_i, end_i, strides_i in zip(begin, end, stride):
            s = slice(begin_i, end_i, strides_i)
            slice_index.append(s)
        slice_index = tuple(slice_index)
        return numpy_data[slice_index]

    _apply_operator_map = {"Reshape": _apply_reshape_operator, "StridedSlice": _apply_slice_operator,
                           "AllGather": _apply_allconcat_operator}
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
            tensor_slices_new = np.concatenate((tensor_slices_new,\
                                   np.array(tensor_slices[j][:, i]).reshape(field_size, -1)), axis=1)
        tensor_slices_col.append(tensor_slices_new)
    new_tensor = np.array(tensor_slices_col[0]).reshape(-1, 1)
    for i in range(1, len(tensor_slices_col)):
        new_tensor = np.concatenate((new_tensor, np.array(tensor_slices_col[i]).reshape(-1, 1)), axis=1)
    return Tensor(new_tensor)
