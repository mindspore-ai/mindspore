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
import numpy as np

from mindspore.common.tensor import Tensor
from ..communication.management import get_rank, get_group_size


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


def _get_slice_index(dev_mat, tensor_map):
    """
    Get the slice index for current slice.

    Args:
        dev_mat (list): The device matrix of devices.
        tensor_map (list): The split strategy of tensor.

    Returns:
        Integer, the slice index for slice on this device.
    """
    rank = get_rank()
    tensor_strategy = _get_tensor_strategy(dev_mat, tensor_map)
    tensor_slice_index = _get_tensor_slice_index(dev_mat, tensor_strategy, tensor_map, rank)
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
            for l in range(j * tensor_strategy[dim_len - 1 - i] + 1,
                           (j + 1) * tensor_strategy[dim_len - 1 - i]):
                new_tensor = np.concatenate((new_tensor, tensor_slices_new[l]), axis=dim_len - 1 - i)

            tensor_slices_new_inner.insert(len(tensor_slices_new_inner), np.array(new_tensor))
        tensor_slices_new = tensor_slices_new_inner

    return Tensor(tensor_slices_new[0])


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
