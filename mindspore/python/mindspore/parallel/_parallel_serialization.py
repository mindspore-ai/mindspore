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
"""parallel serialization"""
from __future__ import absolute_import

import os
import json
import numpy as np
import mindspore as ms
from mindspore.parallel._tensor import _get_tensor_strategy, _construct_from_to_tensor_layout, \
    _get_needed_rank_list_by_layouts, _get_needed_rank_transform_operator_map_by_layouts, \
    _generate_transform_operator_stack, _apply_tensor_transform_operators, _construct_tensor_layout_for_opt_shard, \
    _extract_layout_item


MAX_PATH_LENGTH = 1024


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
            pipeline_stage = 0
            origin_param_name = param_name
            if "-" in param_name:
                pipeline_stage, origin_param_name = param_name.split("-")
            if origin_param_name not in train_map:
                train_map[origin_param_name] = [dev_mat, tensor_map, param_split_shape, field_size, shard_stride,
                                                shard_size, [int(pipeline_stage)]]
            else:
                train_map.get(origin_param_name)[6].append(int(pipeline_stage))
        except BaseException as e:
            raise ValueError(f"{e.__str__()}. Convert layout strategy to list "
                             f"failed, please make sure that strategy matches the node_strategy.proto, you can "
                             f"check whether 'train_strategy_filename' is correct.") from e
    return train_map


def _convert_to_layout(param_name, tensor_layout):
    """Convert list to ParallelLayouts object."""
    strategy = {}
    try:
        layout = ms.train.node_strategy_pb2.ParallelLayouts()
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


def _check_strategy_file(strategy_filename):
    """load parallel strategy file"""
    if not isinstance(strategy_filename, str):
        raise TypeError(f"For 'build_searched_strategy', the argument 'strategy_filename' should be string, "
                        f"but got {type(strategy_filename)}.")

    if not os.path.isfile(strategy_filename):
        raise ValueError(f"For 'build_searched_strategy', no such strategy file: {strategy_filename}. "
                         f"Please check whether the 'strategy_filename' exists.")

    if os.path.getsize(strategy_filename) == 0:
        raise ValueError(f"For 'build_searched_strategy', the strategy file {strategy_filename} should not "
                         f"be empty. Please check whether the 'strategy_filename' is correct.")


def _load_protobuf_strategy(strategy_filename):
    """load strategy from protobuf file"""
    parallel_strategy_map = ms.train.node_strategy_pb2.ParallelStrategyMap()
    with open(strategy_filename, 'rb') as f:
        pb_content = f.read()
    try:
        parallel_strategy_map.ParseFromString(pb_content)
    except BaseException as e:
        raise TypeError("The strategy file type should be one of json or protobuf. "
                        "When the file name extension is not '.json', "
                        "the file is considered as a protobuf file.") from e
    return parallel_strategy_map


def _build_protobuf_strategy(strategy_filename):
    """build strategy from protobuf file"""
    parallel_strategy_map = _load_protobuf_strategy(strategy_filename)
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


def _build_json_strategy(strategy_filename):
    """build strategy from json file"""
    with open(strategy_filename, 'r') as f:
        json_content = json.load(f)
    layout_items = json_content.get("parallel_layout_item")
    strategy = {}
    for parameter_name, layout_item in layout_items.items():
        layout = ms.train.node_strategy_pb2.ParallelLayouts()
        layout.field = layout_item.get("field")
        layout.opt_weight_shard_size = layout_item.get("opt_weight_shard_size")
        layout.opt_weight_shard_step = layout_item.get("opt_weight_shard_step")
        dev_matrix = layout.dev_matrix.add()
        for item in layout_item.get("dev_matrix"):
            dev_matrix.dim.append(item)
        tensor_map = layout.tensor_map.add()
        for item in layout_item.get("tensor_map"):
            tensor_map.dim.append(item)
        param_split_shape = layout.param_split_shape.add()
        if "param_split_shape" in layout_item:
            for item in layout_item.get("param_split_shape"):
                param_split_shape.dim.append(item)
        indices_offset = layout.indices_offset.add()
        if "indices_offset" in layout_item:
            for item in layout_item.get("indices_offset"):
                indices_offset.dim.append(item)
        strategy[parameter_name] = layout
    return strategy


def _build_searched_strategy(strategy_filename):
    """build searched strategy"""
    _check_strategy_file(strategy_filename)
    if strategy_filename[-5:] != ".json":
        return _build_protobuf_strategy(strategy_filename)
    return _build_json_strategy(strategy_filename)


def _merge_protobuf_strategy(src_strategy_files, dst_strategy_file):
    """merge protobuf strategy"""
    dst_parallel_strategy_map = ms.train.node_strategy_pb2.ParallelStrategyMap()
    merged_stage = []
    for src_strategy_file in src_strategy_files:
        src_parallel_strategy_map = _load_protobuf_strategy(src_strategy_file)
        strategy_items = src_parallel_strategy_map.parallel_strategy_item
        layout_items = src_parallel_strategy_map.parallel_layout_item
        if not strategy_items or not layout_items:
            raise ValueError("The strategy file {} is empty".format(src_strategy_file))
        pipeline_stage = strategy_items[0].parallel_strategys.stage
        if pipeline_stage in merged_stage:
            continue
        for layout_item in layout_items:
            layout_item.param_name = "-".join([str(pipeline_stage), layout_item.param_name])
        dst_parallel_strategy_map.parallel_strategy_item.extend(strategy_items)
        dst_parallel_strategy_map.parallel_layout_item.extend(layout_items)
        merged_stage.append(pipeline_stage)
    dst_parallel_strategy_map.current_stage = 1
    with open(dst_strategy_file, "wb") as f:
        f.write(dst_parallel_strategy_map.SerializeToString())


def _merge_json_strategy(src_strategy_files, dst_strategy_file):
    """merge protobuf strategy"""
    dst_parallel_strategy_map = {"current_stage": 1, "parallel_strategy_item": {}, "parallel_layout_item": {}}
    merged_stage = []
    for src_strategy_file in src_strategy_files:
        with open(src_strategy_file, 'r') as f:
            json_content = json.load(f)
        layout_items = json_content.get("parallel_layout_item")
        strategy_items = json_content.get("parallel_strategy_item")
        if not strategy_items or not layout_items:
            raise ValueError("The strategy file {} is empty".format(src_strategy_file))
        pipeline_stage = strategy_items.get(list(strategy_items.keys())[0]).get('stage')
        if pipeline_stage in merged_stage:
            continue
        for param_name, layout_item in layout_items.items():
            new_layout_item = {}
            new_param_name = "-".join([str(pipeline_stage), param_name])
            new_layout_item[new_param_name] = layout_item
            dst_parallel_strategy_map.get("parallel_layout_item").update(new_layout_item)
        dst_parallel_strategy_map.get("parallel_strategy_item").update(strategy_items)
        merged_stage.append(pipeline_stage)
    with open(dst_strategy_file, "w") as f:
        json.dump(dst_parallel_strategy_map, f)


def _parameter_not_in_local_stage(param_name, origin_strategy_list, strategy_list):
    """parameter whether in the local stage"""
    if origin_strategy_list is None or strategy_list is None:
        return True
    return param_name in origin_strategy_list and param_name not in strategy_list


def _extract_layout_map(strategy_file):
    """Extract layout map"""
    layout_map = None
    if strategy_file is not None:
        src_strategy = _build_searched_strategy(strategy_file)
        layout_map = _convert_to_list(src_strategy)
    return layout_map


def _extract_pipeline_stage_num(strategy_file):
    """extract pipeline stage num"""
    pipeline_stage_num = 1
    if strategy_file is not None:
        src_strategy = _build_searched_strategy(strategy_file)
        layout_map = _convert_to_list(src_strategy)
        pipeline_stage_set = set()
        for _, layout in layout_map.items():
            pipeline_stage_set.update(layout[6])
        pipeline_stage_num = len(pipeline_stage_set)
        if list(pipeline_stage_set) != list(range(pipeline_stage_num)):
            raise ValueError("The strategy file for pipeline parallel dose not contains all stages.")
    return pipeline_stage_num


def _extract_src_dst_layout_map(rank_id, src_strategy_file=None, dst_strategy_file=None):
    """Extract strategy list"""
    src_layout_map = _extract_layout_map(src_strategy_file)
    dst_layout_map = _extract_layout_map(dst_strategy_file)
    if dst_layout_map is None:
        return src_layout_map, dst_layout_map
    dst_stage_device_num = np.prod(dst_layout_map.get(list(dst_layout_map.keys())[0])[0])
    dst_stage_id = rank_id // dst_stage_device_num
    # cut the source and destination layout, remain the parameter in the dst_stage
    for param_name in list(dst_layout_map.keys()):
        if dst_stage_id in dst_layout_map.get(param_name)[6]:
            continue
        dst_layout_map.pop(param_name)
        if src_layout_map is not None and param_name in src_layout_map:
            src_layout_map.pop(param_name)
    return src_layout_map, dst_layout_map


def _restore_group_info_list(group_info_file_name):
    """restore group info"""
    parallel_group_map = ms.train.node_strategy_pb2.ParallelGroupMap()

    with open(group_info_file_name, 'rb') as f:
        pb_content = f.read()
    parallel_group_map.ParseFromString(pb_content)

    restore_list = parallel_group_map.ckpt_restore_rank_list
    if not restore_list:
        raise ValueError("For 'restore_group_info_list', the group information file has no restore rank list.")

    return [rank for rank in restore_list.dim]


def _get_device_num_from_strategy(strategy_file=None):
    """Get device num from strategy file"""
    if strategy_file is None:
        return 1
    src_strategy = _build_searched_strategy(strategy_file)
    strategy_list = _convert_to_list(src_strategy)
    device_mat = list(strategy_list.values())[0][0]
    return np.prod(device_mat)


def _rank_list_for_transform_parallel_checkpoint(rank_id, src_strategy_list, dst_strategy_list):
    """
    Get the needed rank list for transform model parallel dim of checkpoint.
    """
    result_list = set()
    handled_layout = []
    for param_name, _ in src_strategy_list.items():
        if dst_strategy_list is not None and param_name not in dst_strategy_list:
            continue
        from_dev_matrix, from_tensor_map, from_opt_shard_step, from_opt_shard_size = _extract_layout_item(
            src_strategy_list.get(param_name))
        from_device_num = np.prod(from_dev_matrix)
        fake_tensor_shape = [8] * len(from_tensor_map)
        to_dev_matrix = [1]
        to_tensor_map = [-1] * len(fake_tensor_shape)
        to_opt_shard_step = 0
        to_opt_shard_size = 0
        if dst_strategy_list is not None:
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


def _transform_parallel_checkpoint(rank_id, param_total_dict, param_attr_dict, src_strategy_list, dst_strategy_list):
    """
    Transform model parallel dimension for distributed checkpoint files.
    """
    transform_param_dict = {}
    for param_name, _ in param_total_dict.items():
        tensor_shape = list(param_total_dict[param_name].values())[0].shape
        from_dev_matrix = [1]
        from_tensor_map = [-1] * len(tensor_shape)
        from_opt_shard_step = 0
        from_opt_shard_size = 0
        if src_strategy_list is not None:
            if param_name not in src_strategy_list:
                ms.log.warning("The parameter {} is not in src_strategy.".format(param_name))
                continue
            from_dev_matrix, from_tensor_map, from_opt_shard_step, from_opt_shard_size = _extract_layout_item(
                src_strategy_list.get(param_name))
        to_dev_matrix_origin = [1]
        to_tensor_map_origin = [-1] * len(tensor_shape)
        to_opt_shard_step = 0
        to_opt_shard_size = 0
        if dst_strategy_list is not None:
            if param_name not in dst_strategy_list:
                ms.log.warning("The parameter {} is not in dst_strategy.".format(param_name))
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


        from_info_tuple = (from_opt_shard_size, from_dev_matrix, from_tensor_map, from_full_tensor_shape)
        to_info_tuple = (to_opt_shard_size, to_dev_matrix_origin, to_tensor_map_origin, origin_tensor_shape)
        _insert_opt_shard_reshape(param_rank_map, from_info_tuple, to_info_tuple)
        transform_operator_stack = _generate_transform_operator_stack(param_rank_map, rank_id)
        _apply_tensor_transform_operators(transform_operator_stack, param_total_dict[param_name], device_num)
        transform_tensor = ms.Tensor(param_total_dict[param_name][rank_id % device_num])
        requires_grad = param_attr_dict[param_name][rank_id % device_num][0]
        layerwise_parallel = param_attr_dict[param_name][rank_id % device_num][1]
        transform_param_dict[param_name] = ms.Parameter(transform_tensor, param_name, requires_grad, layerwise_parallel)

    # Handle those parameter like learning_rate, global_step which not in strategy_file.
    for param_name, _ in param_total_dict.items():
        if param_name not in transform_param_dict:
            transform_param_dict[param_name] = ms.Parameter(
                ms.Tensor(param_total_dict[param_name][rank_id % device_num]), param_name,
                param_attr_dict[param_name][rank_id % device_num][0],
                param_attr_dict[param_name][rank_id % device_num][1])

    transform_param_list = [{"name": param_name, "data": param_data}
                            for param_name, param_data in transform_param_dict.items()]
    return transform_param_list


def _make_dir(path, arg_name):
    """Make directory."""
    if not isinstance(path, str):
        ms.log.critical("The %s is invalid, the type should be string.", arg_name)
        raise TypeError("The {} is invalid, the type should be string.".format(arg_name))
    if path.strip() == "":
        ms.log.critical("The %s is invalid, it should be non-blank.", arg_name)
        raise ValueError("The {} is invalid, it should be non-blank.".format(arg_name))

    path = os.path.realpath(path)

    if len(path) > MAX_PATH_LENGTH:
        ms.log.critical("The %s length is too long, it should be limited in %s.", arg_name, MAX_PATH_LENGTH)
        raise ValueError("The {} length is too long, it should be limited in {}.".format(arg_name, MAX_PATH_LENGTH))

    ms.log.debug("The abs path is %r", path)

    if os.path.exists(path):
        if not os.path.isdir(path):
            ms.log.critical("The path(%r) is a file path, it should be a directory path.", path)
            raise NotADirectoryError("The path({}) is a file path, it should be a directory path.".format(path))
        real_path = path
    else:
        ms.log.debug("The directory(%s) doesn't exist, will create it", path)
        try:
            permissions = os.R_OK | os.W_OK | os.X_OK
            os.umask(permissions << 3 | permissions)
            mode = permissions << 6
            os.makedirs(path, mode=mode, exist_ok=True)
            real_path = path
        except PermissionError as e:
            ms.log.critical("No write permission on the directory(%r), error = %r", path, e)
            raise TypeError("No write permission on the directory.") from e
        finally:
            pass
    return real_path


def _insert_opt_shard_reshape(param_rank_map, from_info_tuple, to_info_tuple):
    """insert opt_shard op reshape"""
    from_opt_shard_size = from_info_tuple[0]
    from_dev_matrix = from_info_tuple[1]
    from_tensor_map = from_info_tuple[2]
    from_full_tensor_shape = from_info_tuple[3]
    to_opt_shard_size = to_info_tuple[0]
    to_dev_matrix_origin = to_info_tuple[1]
    to_tensor_map_origin = to_info_tuple[2]
    origin_tensor_shape = to_info_tuple[3]
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
