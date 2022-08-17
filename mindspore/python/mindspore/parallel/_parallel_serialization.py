# Copyright 2022 Huawei Technologies Co., Ltd
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
import os
import numpy as np
from mindspore.common.tensor import Tensor
from mindspore.common.parameter import Parameter
from mindspore.train.node_strategy_pb2 import ParallelStrategyMap, ParallelLayouts, ParallelGroupMap
from mindspore import log as logger
from mindspore.parallel._tensor import _get_tensor_strategy, _construct_from_to_tensor_layout, \
    _get_needed_rank_list_by_layouts, _get_needed_rank_transform_operator_map_by_layouts, \
    _generate_transform_operator_stack, _apply_tensor_transform_operators, _construct_tensor_layout_for_opt_shard, \
    _extract_layout_item


def _convert_to_list(strategy):
    """Convert ParallelLayouts object to specified list."""
    train_map = {}
    for param_name in strategy.keys():
        try:
            layout = strategy.get(param_name)
            dev_mat = list(layout.dev_matrix[0].dim)
            tensor_map = list(layout.tensor_map[0].dim)
            param_split_shape = list(layout.param_split_shape[0].dim)
            field_size = int(layout.field)
            shard_stride = int(layout.opt_weight_shard_step)
            shard_size = int(layout.opt_weight_shard_size)
            train_map[param_name] = [dev_mat, tensor_map, param_split_shape, field_size, shard_stride, shard_size]
        except BaseException as e:
            raise ValueError(f"{e.__str__()}. Convert layout strategy to list "
                             f"failed, please make sure that strategy matches the node_strategy.proto, you can "
                             f"check whether 'train_strategy_filename' is correct.") from e
    return train_map


def _convert_to_layout(param_name, tensor_layout):
    """Convert list to ParallelLayouts object."""
    strategy = {}
    try:
        layout = ParallelLayouts()
        layout.field = tensor_layout[3]

        dev_matrix = layout.dev_matrix.add()
        for item in tensor_layout[0]:
            dev_matrix.dim.append(item)

        tensor_map = layout.tensor_map.add()
        for item in tensor_layout[1]:
            tensor_map.dim.append(item)

        param_split_shape = layout.param_split_shape.add()
        for item in tensor_layout[2]:
            param_split_shape.dim.append(item)
    except BaseException as e:
        raise ValueError(f"{e.__str__()}. For 'load_distributed_checkpoint', convert list to layout strategy failed, "
                         f"you can check whether your input list is correct.") from e

    strategy[param_name] = layout
    return strategy


def _build_searched_strategy(strategy_filename):
    """build searched strategy"""
    if not isinstance(strategy_filename, str):
        raise TypeError(f"For 'build_searched_strategy', the argument 'strategy_filename' should be string, "
                        f"but got {type(strategy_filename)}.")

    if not os.path.isfile(strategy_filename):
        raise ValueError(f"For 'build_searched_strategy', no such strategy file: {strategy_filename}. "
                         f"Please check whether the 'strategy_filename' exists.")

    if os.path.getsize(strategy_filename) == 0:
        raise ValueError(f"For 'build_searched_strategy', the strategy file {strategy_filename} should not "
                         f"be empty. Please check whether the 'strategy_filename' is correct.")
    parallel_strategy_map = ParallelStrategyMap()

    with open(strategy_filename, 'rb') as f:
        pb_content = f.read()
    parallel_strategy_map.ParseFromString(pb_content)

    layout_items = parallel_strategy_map.parallel_layout_item
    if not layout_items:
        raise ValueError(f"For 'build_searched_strategy', the strategy file {strategy_filename} has no sliced "
                         f"parameter, please check whether the 'strategy_filename' is correct.")

    strategy = {}
    for layout_item in layout_items:
        parameter_name = layout_item.param_name
        layout = layout_item.parallel_layouts
        strategy[parameter_name] = layout

    return strategy


def _restore_group_info_list(group_info_file_name):
    """restore group info"""
    parallel_group_map = ParallelGroupMap()

    with open(group_info_file_name, 'rb') as f:
        pb_content = f.read()
    parallel_group_map.ParseFromString(pb_content)

    restore_list = parallel_group_map.ckpt_restore_rank_list
    if not restore_list:
        raise ValueError("For 'restore_group_info_list', the group information file has no restore rank list.")

    restore_rank_list = [rank for rank in restore_list.dim]
    return restore_rank_list


def _get_device_num_from_strategy(strategy_file=None):
    if strategy_file is None:
        return 1
    src_strategy = _build_searched_strategy(strategy_file)
    strategy_list = _convert_to_list(src_strategy)
    device_mat = list(strategy_list.values())[0][0]
    return np.prod(device_mat)


def _rank_list_for_transform_parallel_checkpoint(rank_id, src_strategy_file=None, dst_strategy_file=None):
    """
    Get the needed rank list for transform model parallel dim of checkpoint.
    """
    if src_strategy_file is None:
        return [rank_id]
    src_strategy = _build_searched_strategy(src_strategy_file)
    src_strategy_list = _convert_to_list(src_strategy)
    if not src_strategy_list:
        raise ValueError("The src_strategy_file is empty.")
    if dst_strategy_file is not None:
        dst_strategy = _build_searched_strategy(dst_strategy_file)
        dst_strategy_list = _convert_to_list(dst_strategy)
    result_list = set()
    handled_layout = []
    for param_name, src_strategy in src_strategy_list.items():
        if dst_strategy_file is not None and param_name not in dst_strategy_list:
            continue
        from_dev_matrix, from_tensor_map, from_opt_shard_step, from_opt_shard_size = _extract_layout_item(
            src_strategy_list.get(param_name))
        from_device_num = np.prod(from_dev_matrix)
        fake_tensor_shape = [8] * len(from_tensor_map)
        to_dev_matrix = [1]
        to_tensor_map = [-1] * len(fake_tensor_shape)
        to_opt_shard_step = 0
        to_opt_shard_size = 0
        if dst_strategy_file is not None:
            to_dev_matrix, to_tensor_map, to_opt_shard_step, to_opt_shard_size = _extract_layout_item(
                dst_strategy_list.get(param_name))
        handled_key = (from_dev_matrix, from_tensor_map, from_opt_shard_step, from_opt_shard_size,
                       to_dev_matrix, to_tensor_map, to_opt_shard_step, to_opt_shard_size)
        if handled_key in handled_layout:
            continue
        handled_layout.append(handled_key)
        param_strategy = _get_tensor_strategy(from_dev_matrix, from_tensor_map)
        origin_tensor_shape = ()
        for i, item in enumerate(fake_tensor_shape):
            if i == 0 and from_opt_shard_size > 0:
                origin_tensor_shape += (item * param_strategy[i] * from_opt_shard_size,)
                continue
            origin_tensor_shape += (item * param_strategy[i],)

        from_dev_matrix, from_tensor_map, from_full_tensor_shape = _construct_tensor_layout_for_opt_shard(
            from_dev_matrix, from_tensor_map, from_opt_shard_step, from_opt_shard_size, origin_tensor_shape)
        to_dev_matrix, to_tensor_map, to_full_tensor_shape = _construct_tensor_layout_for_opt_shard(
            to_dev_matrix, to_tensor_map, to_opt_shard_step, to_opt_shard_size, origin_tensor_shape)
        # Convert tensor layout to same device num
        from_tensor_layout, to_tensor_layout = _construct_from_to_tensor_layout(from_full_tensor_shape, from_dev_matrix,
                                                                                from_tensor_map, to_full_tensor_shape,
                                                                                to_dev_matrix, to_tensor_map)
        device_list = list(range(0, np.prod(from_tensor_layout[0])))
        param_rank_list = _get_needed_rank_list_by_layouts(from_tensor_layout, to_tensor_layout, device_list, rank_id)
        param_rank_list_new = [rank % from_device_num for rank in param_rank_list]
        param_rank_list_new = set(param_rank_list_new)
        result_list.update(param_rank_list_new)
    return list(result_list)


def _transform_parallel_checkpoint(rank_id, param_total_dict, param_attr_dict, src_strategy_file=None,
                                   dst_strategy_file=None):
    """
    Transform model parallel dimension for distributed checkpoint files.
    """
    device_num = rank_id + 1
    if src_strategy_file is not None:
        src_strategy = _build_searched_strategy(src_strategy_file)
        src_strategy_list = _convert_to_list(src_strategy)
    if dst_strategy_file is not None:
        dst_strategy = _build_searched_strategy(dst_strategy_file)
        dst_strategy_list = _convert_to_list(dst_strategy)
    transform_param_dict = {}
    for param_name, _ in param_total_dict.items():
        tensor_shape = list(param_total_dict[param_name].values())[0].shape
        from_dev_matrix = [1]
        from_tensor_map = [-1] * len(tensor_shape)
        from_opt_shard_step = 0
        from_opt_shard_size = 0
        if src_strategy_file is not None:
            if param_name not in src_strategy_list:
                logger.warning("The parameter {} is not in src_strategy.".format(param_name))
                continue
            from_dev_matrix, from_tensor_map, from_opt_shard_step, from_opt_shard_size = _extract_layout_item(
                src_strategy_list.get(param_name))
        to_dev_matrix_origin = [1]
        to_tensor_map_origin = [-1] * len(tensor_shape)
        to_opt_shard_step = 0
        to_opt_shard_size = 0
        if dst_strategy_file is not None:
            if param_name not in dst_strategy_list:
                logger.warning("The parameter {} is not in dst_strategy.".format(param_name))
                continue
            to_dev_matrix_origin, to_tensor_map_origin, to_opt_shard_step, to_opt_shard_size = _extract_layout_item(
                dst_strategy_list.get(param_name))
        # Add optimizer sharding dim for tensor layout
        device_num = np.prod(from_dev_matrix)
        param_strategy = _get_tensor_strategy(from_dev_matrix, from_tensor_map)
        origin_tensor_shape = ()
        for i, item in enumerate(tensor_shape):
            if i == 0 and from_opt_shard_size > 0:
                origin_tensor_shape += (item * param_strategy[i] * from_opt_shard_size,)
                continue
            origin_tensor_shape += (item * param_strategy[i],)

        from_dev_matrix, from_tensor_map, from_full_tensor_shape = _construct_tensor_layout_for_opt_shard(
            from_dev_matrix, from_tensor_map, from_opt_shard_step, from_opt_shard_size, origin_tensor_shape)
        to_dev_matrix, to_tensor_map, to_full_tensor_shape = _construct_tensor_layout_for_opt_shard(
            to_dev_matrix_origin, to_tensor_map_origin, to_opt_shard_step, to_opt_shard_size, origin_tensor_shape)
        # Convert tensor layout to same device num
        from_tensor_layout, to_tensor_layout = _construct_from_to_tensor_layout(from_full_tensor_shape, from_dev_matrix,
                                                                                from_tensor_map, to_full_tensor_shape,
                                                                                to_dev_matrix, to_tensor_map)

        # when the from_layout is less devices, the checkpoint_map for map[device_num] should using map[0]
        device_list = list(range(0, np.prod(from_tensor_layout[0])))
        if rank_id % device_num not in param_attr_dict[param_name]:
            raise ValueError("The checkpoint of rank {} is missing.".format(rank_id % device_num))
        param_rank_map = _get_needed_rank_transform_operator_map_by_layouts(from_tensor_layout, to_tensor_layout,
                                                                            device_list, rank_id)
        for param_rank, _ in param_rank_map.items():
            if from_opt_shard_size > 0:
                from_tensor_strategy = _get_tensor_strategy(from_dev_matrix, from_tensor_map)
                from_slice_tensor_shape = ()
                for i, item in enumerate(from_full_tensor_shape):
                    from_slice_tensor_shape += (item // from_tensor_strategy[i],)
                param_rank_map.get(param_rank).insert(0, ('Reshape', list(from_slice_tensor_shape)))
            if to_opt_shard_size > 0:
                to_tensor_strategy = _get_tensor_strategy(to_dev_matrix_origin, to_tensor_map_origin)
                to_slice_tensor_shape = ()
                for i, item in enumerate(origin_tensor_shape):
                    if i == 0 and to_opt_shard_size > 0:
                        to_slice_tensor_shape += (item // (to_tensor_strategy[i] * to_opt_shard_size),)
                        continue
                    to_slice_tensor_shape += (item // to_tensor_strategy[i],)
                param_rank_map.get(param_rank).append(('Reshape', list(to_slice_tensor_shape)))

        transform_operator_stack = _generate_transform_operator_stack(param_rank_map, rank_id)
        _apply_tensor_transform_operators(transform_operator_stack, param_total_dict[param_name], device_num)
        transform_tensor = Tensor(param_total_dict[param_name][rank_id % device_num])
        requires_grad = param_attr_dict[param_name][rank_id % device_num][0]
        layerwise_parallel = param_attr_dict[param_name][rank_id % device_num][1]
        transform_param_dict[param_name] = Parameter(transform_tensor, param_name, requires_grad, layerwise_parallel)

    # Handle those parameter like learning_rate, global_step which not in strategy_file.
    for param_name, _ in param_total_dict.items():
        if param_name not in transform_param_dict:
            transform_param_dict[param_name] = Parameter(Tensor(param_total_dict[param_name][rank_id % device_num]),
                                                         param_name,
                                                         param_attr_dict[param_name][rank_id % device_num][0],
                                                         param_attr_dict[param_name][rank_id % device_num][1])

    transform_param_list = [{"name": param_name, "data": param_data}
                            for param_name, param_data in transform_param_dict.items()]
    return transform_param_list
