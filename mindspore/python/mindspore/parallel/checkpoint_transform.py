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
"""Transform distributed checkpoint"""
from __future__ import absolute_import

import os
import glob
import copy
from collections import defaultdict
import numpy as np
import mindspore as ms
from mindspore.parallel._parallel_serialization import _rank_list_for_transform_parallel_checkpoint, \
    _transform_parallel_checkpoint, _get_device_num_from_strategy, _make_dir, _load_strategy_file, \
    _extract_layout_map, _extract_src_dst_layout_map, _parameter_not_in_local_stage, _extract_pipeline_stage_num


__all__ = ["merge_pipeline_strategys", "rank_list_for_transform", "transform_checkpoint_by_rank",
           "transform_checkpoints"]


def merge_pipeline_strategys(src_strategy_dirs, dst_strategy_file):
    """
    Merge parallel strategy between all pipeline stages in pipeline parallel mode.

    Note:
        Strategy file of each pipeline stage should be included in src_strategy_dirs.

    Args:
        src_strategy_dirs (str): The directory of strategy files including all pipeline stage which is saved by
                                 'mindspore.set_auto_parallel_context(strategy_ckpt_save_file)'
        dst_strategy_file (str): The file merged strategy to save.

    Raises:
        NotADirectoryError: `src_strategy_dirs` is not a directory.

    Examples:
        >>> # src_strategy_dir/stra0.ckpt, src_strategy_dir/stra1.ckpt ... src_strategy_dir/stra127.ckpt
        >>> merge_pipeline_strategys("./src_strategy_dir", "./dst_strategy.ckpt")

    """
    dst_strategy_dir, _ = os.path.split(dst_strategy_file)
    if not os.path.exists(dst_strategy_dir):
        _make_dir(dst_strategy_dir, "path")
    if not os.path.isdir(src_strategy_dirs):
        raise NotADirectoryError("src_strategy_dirs {} is not a directory.".format(src_strategy_dirs))
    src_strategy_files = os.path.join(src_strategy_dirs, "*.ckpt")
    dst_parallel_strategy_map = ms.train.node_strategy_pb2.ParallelStrategyMap()
    merged_stage = []
    for src_strategy_file in glob.glob(src_strategy_files):
        src_parallel_strategy_map = _load_strategy_file(src_strategy_file)
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


def rank_list_for_transform(rank_id, src_strategy_file=None, dst_strategy_file=None):
    """
    List of original distributed checkpoint rank index for obtaining the target checkpoint of a rank_id
    during the distributed checkpoint conversion.

    Args:
        rank_id (int): The rank of which distributed checkpoint needs to be obtained after conversion.
        src_strategy_file (str): Name of source sharding strategy file which saved by
                                 'mindspore.set_auto_parallel_context(strategy_ckpt_save_file)'.
                                 when the 'src_strategy_file' is None, it means that the source sharding strategy is
                                 without any sharing for each parameter. Default:None.
        dst_strategy_file (str): Name of destination sharding strategy file which saved by
                                 'mindspore.set_auto_parallel_context(strategy_ckpt_save_file)'.
                                 when the 'dst_strategy_file' is None, it means that the destination sharding strategy
                                 is without any sharing for each parameter. Default:None.

    Returns:
        List, the rank list required for converting the distributed checkpoint of rank_id.

    Raises:
        ValueError: `src_strategy_file` or dst_strategy_file is incorrect.
        TypeError: `src_strategy_file` or dst_strategy_file is not a string.
        TypeError: `rank_id` is not a int.

    Examples:
        >>> rank_id = 0
        >>> rank_list = rank_list_for_transform(rank_id, "./src_strategy.ckpt", "./dst_strategy.ckpt")
        >>> checkpoint_files_map = {}
        >>> for rank in rank_list:
        >>>     checkpoint_files_map[rank] = "./pangu{}-100_2.ckpt".format(rank)

    """
    if not isinstance(rank_id, int):
        raise TypeError("The rank_id should be a int.")
    if src_strategy_file is None:
        return [0]
    src_strategy_list, dst_strategy_list = _extract_src_dst_layout_map(rank_id, src_strategy_file, dst_strategy_file)
    src_stage_device_num = np.prod(src_strategy_list.get(list(src_strategy_list.keys())[0])[0]) if src_strategy_list \
                                                                                                   is not None else 1
    dst_stage_device_num = np.prod(dst_strategy_list.get(list(dst_strategy_list.keys())[0])[0]) if dst_strategy_list \
                                                                                                   is not None else 1

    if not src_strategy_list:
        raise ValueError("The src_strategy_file is empty.")
    local_rank_id = rank_id % dst_stage_device_num if dst_stage_device_num > 1 else rank_id
    needed_rank_list_in_local_stage = _rank_list_for_transform_parallel_checkpoint(local_rank_id,
                                                                                   src_strategy_list, dst_strategy_list)
    result_set = set()
    handled_pipeline_stage = []
    for _, layout in src_strategy_list.items():
        for src_pipeline_stage_id in layout[6]:
            if src_pipeline_stage_id in handled_pipeline_stage:
                continue
            src_rank_id_start = src_pipeline_stage_id * src_stage_device_num
            result_set.update([src_rank_id_start + rank for rank in needed_rank_list_in_local_stage])
            handled_pipeline_stage.append(src_pipeline_stage_id)
    return list(result_set)


def transform_checkpoint_by_rank(rank_id, checkpoint_files_map, save_checkpoint_file_name,
                                 src_strategy_file=None, dst_strategy_file=None):
    """
    Transform distributed checkpoint from source sharding strategy to destination sharding strategy by rank
    for a network.

    Args:
        rank_id (int): The rank of which distributed checkpoint needs to be obtained after conversion.
        checkpoint_files_map (dict): The checkpoint files map whose key is the rank id and the value is
                                     the checkpoint file name.
        save_checkpoint_file_name (str): The file name to save the converted checkpoint.
        src_strategy_file (str): Name of source sharding strategy file which saved by
                                 'mindspore.set_auto_parallel_context(strategy_ckpt_save_file)'.
                                 when the 'src_strategy_file' is None, it means that the source sharding strategy is
                                 without any sharing for each parameter. Default:None.
        dst_strategy_file (str): Name of destination sharding strategy file which saved by
                                 'mindspore.set_auto_parallel_context(strategy_ckpt_save_file)'.
                                 when the 'dst_strategy_file' is None, it means that the destination sharding strategy
                                 is without any sharing for each parameter. Default:None.

    Raises:
        ValueError: `src_strategy_file` or `dst_strategy_file` is incorrect.
        ValueError: item in `checkpoint_files_map` is incorrect.
        ValueError: `save_checkpoint_file_name` is not end with ".ckpt".
        TypeError: `checkpoint_files_map` is not a dict.
        TypeError: `src_strategy_file` or `dst_strategy_file` is not a string.
        TypeError: `rank_id` is not a int.
        TypeError: `save_checkpoint_file_name` is not a string.

    Examples:
        >>> dst_device_num = 8
        >>> for rank_id in range(dst_device_num)
        >>>     rank_list = rank_list_for_transform(rank_id, "./src_strategy.ckpt", "./dst_strategy.ckpt")
        >>>     checkpoint_files_map = {}
        >>>     for rank in rank_list:
        >>>         checkpoint_files_map[rank] = "./origin_checkpoint_rank{}/pangu{}-100_2.ckpt".format(rank)
        >>>     save_checkpoint_file_name = "./new_checkpoint_rank{}/pangu{}-100_2.ckpt".format(rank_id)
        >>>     transform_checkpoint_by_rank(rank_id, checkpoint_files_map, save_checkpoint_file_name,
        ...                                  "./src_strategy.ckpt", "./dst_strategy.ckpt")

    """
    if not isinstance(checkpoint_files_map, dict):
        raise TypeError("The checkpoint_files_map should be a dict.")
    if not isinstance(rank_id, int):
        raise TypeError("The rank_id should be a int.")
    if not isinstance(save_checkpoint_file_name, str):
        raise TypeError("The save_checkpoint_file_name should be a str.")
    if save_checkpoint_file_name[-5:] != ".ckpt":
        raise ValueError("The save_checkpoint_file_name {} should end with .ckpt".format(save_checkpoint_file_name))
    if dst_strategy_file and os.path.dirname(dst_strategy_file) and not os.path.exists(
            os.path.dirname(dst_strategy_file)):
        raise ValueError("The director of dst_strategy_file: {} is not exists.".
                         format(os.path.dirname(dst_strategy_file)))
    for rank, local_file in checkpoint_files_map.items():
        if not os.path.exists(local_file):
            raise ValueError("Checkpoint file {} in rank {} not exits: ".format(local_file, rank))
    param_total_dict = defaultdict(dict)
    param_attr_dict = defaultdict(dict)
    src_strategy_list, dst_strategy_list = _extract_src_dst_layout_map(rank_id, src_strategy_file, dst_strategy_file)
    # src rank => local rank inside pipeline stage
    src_stage_device_num = np.prod(src_strategy_list.get(list(src_strategy_list.keys())[0])[0]) if src_strategy_list \
                                                                                                   is not None else 1
    dst_stage_device_num = np.prod(dst_strategy_list.get(list(dst_strategy_list.keys())[0])[0]) if dst_strategy_list \
                                                                                                   is not None else 1
    origin_dst_strategy_list = _extract_layout_map(dst_strategy_file)
    origin_src_strategy_list = _extract_layout_map(src_strategy_file)
    for rank, file_name in checkpoint_files_map.items():
        ckpt_dict = ms.load_checkpoint(file_name)
        for param_name, param in ckpt_dict.items():
            # cut the parameter not in the pipeline stage.
            if _parameter_not_in_local_stage(param_name, origin_src_strategy_list, src_strategy_list) \
                    and _parameter_not_in_local_stage(param_name, origin_dst_strategy_list, dst_strategy_list):
                continue
            src_rank = rank % src_stage_device_num
            param_total_dict[param_name][src_rank] = param.data.asnumpy()
            param_attr_dict[param_name][src_rank] = (param.requires_grad, param.layerwise_parallel)
    local_rank_id = rank_id % dst_stage_device_num
    transform_param_list = _transform_parallel_checkpoint(local_rank_id, param_total_dict,
                                                          param_attr_dict, src_strategy_list, dst_strategy_list)
    ms.save_checkpoint(transform_param_list, save_checkpoint_file_name)


def transform_checkpoints(src_checkpoints_dir, dst_checkpoints_dir, ckpt_prefix, src_strategy_file=None,
                          dst_strategy_file=None):
    """
    Transform distributed checkpoint from source sharding strategy to destination sharding strategy for a rank.

    Note:
        The src_checkpoints_dir directory structure should be organized like "src_checkpoints_dir/rank_0/a.ckpt", the
        rank number should be set to a subdirectory and the checkpoint file is stored in this subdirectory. If multiple
        files exist in a rank directory, the last file in the lexicgraphic order would be selected.

    Args:
        src_checkpoints_dir (str): The source checkpoints directory.
        dst_checkpoints_dir (str): The destination checkpoints directory to save the converted checkpoints.
        ckpt_prefix (str): The destination checkpoint name prefix.
        src_strategy_file (str): Name of source sharding strategy file which saved by
                                 'mindspore.set_auto_parallel_context(strategy_ckpt_save_file)'.
                                 when the 'src_strategy_file' is None, it means that the source sharding strategy is
                                 without any sharing for each parameter. Default:None.
        dst_strategy_file (str): Name of destination sharding strategy file which saved by
                                 'mindspore.set_auto_parallel_context(strategy_ckpt_save_file)'.
                                 when the 'dst_strategy_file' is None, it means that the destination sharding strategy
                                 is without any sharing for each parameter. Default:None.

    Raises:
        ValueError: `src_strategy_file` or `dst_strategy_file` is incorrect.
        NotADirectoryError: `src_checkpoints_dir` or `dst_checkpoints_dir` is not a directory.
        ValueError: The checkpoint file is missing in `src_checkpoints_dir`.
        TypeError: `src_strategy_file` or `dst_strategy_file` is not a string.

    Examples:
        >>> transform_checkpoints(src_checkpoints_dir, dst_checkpoints_dir, "dst_checkpoint",
        ...                       "./src_strategy.ckpt", "./dst_strategy.ckpt")

    """
    if not os.path.isdir(src_checkpoints_dir):
        raise NotADirectoryError("src_checkpoints_dir {} is not a directory.".format(src_checkpoints_dir))
    _make_dir(dst_checkpoints_dir, "path")
    if not isinstance(ckpt_prefix, str):
        raise TypeError("The ckpt_prefix should be a str.")
    checkpoints_rank_dir_list = os.path.join(src_checkpoints_dir, "rank_[0-9]*")
    all_checkpoint_files_map = {}
    for checkpoint_dir in glob.glob(checkpoints_rank_dir_list):
        if not os.path.isdir(checkpoint_dir):
            ms.log.warning("{} is not a directory.".format(checkpoint_dir))
            continue
        rank_id_str = checkpoint_dir.split('rank_')[-1]
        if not rank_id_str.isdigit():
            ms.log.warning("{} is not a expected directory, the directory should end with rank_0/rank_1.....".
                           format(checkpoint_dir))
            continue
        rank_id = int(rank_id_str)
        checkpoint_file_name = os.path.join(checkpoint_dir, "*.ckpt")
        rank_ckpts = glob.glob(checkpoint_file_name)
        rank_ckpts.sort()
        for checkpoint_file in rank_ckpts:
            if not os.path.isfile(checkpoint_file):
                ms.log.warning("{} is not a checkpoint file.".format(checkpoint_file))
                continue
            all_checkpoint_files_map[rank_id] = checkpoint_file

    needed_rank_list_map = defaultdict(list)
    dst_stage_device_num = _get_device_num_from_strategy(dst_strategy_file)
    src_stage_device_num = _get_device_num_from_strategy(src_strategy_file)
    dst_stage_num = _extract_pipeline_stage_num(dst_strategy_file)
    dst_device_num = dst_stage_device_num * dst_stage_num
    origin_src_strategy_list = _extract_layout_map(src_strategy_file)
    origin_dst_strategy_list = _extract_layout_map(dst_strategy_file)
    for rank in range(dst_device_num):
        needed_rank_list = rank_list_for_transform(rank, src_strategy_file, dst_strategy_file)
        for needed_rank in needed_rank_list:
            if needed_rank not in all_checkpoint_files_map:
                raise ValueError("The checkpoint file of rank{} is needed for converting rank{}'s checkpoint, "
                                 "but it is missing.".format(needed_rank, rank))
        needed_rank_list_key = "-".join([str(r) for r in needed_rank_list])
        needed_rank_list_map[needed_rank_list_key].append(rank)
    for needed_rank_list_key, transform_rank_list in needed_rank_list_map.items():
        param_total_dict = defaultdict(dict)
        param_attr_dict = defaultdict(dict)
        needed_rank_list = needed_rank_list_key.split("-")
        for needed_rank in needed_rank_list:
            ckpt_dict = ms.load_checkpoint(all_checkpoint_files_map.get(int(needed_rank)))
            for param_name, param in ckpt_dict.items():
                src_rank = int(needed_rank) % src_stage_device_num
                param_total_dict[param_name][src_rank] = param.data.asnumpy()
                param_attr_dict[param_name][src_rank] = (param.requires_grad, param.layerwise_parallel)
        for transform_rank in transform_rank_list:
            param_total_dict_copy = copy.deepcopy(param_total_dict)
            src_strategy_list, dst_strategy_list = _extract_src_dst_layout_map(transform_rank, src_strategy_file,
                                                                               dst_strategy_file)
            # cut the parameter not in the pipeline stage.
            for param in list(param_total_dict_copy.keys()):
                if _parameter_not_in_local_stage(param, origin_src_strategy_list, src_strategy_list) \
                        and _parameter_not_in_local_stage(param, origin_dst_strategy_list, dst_strategy_list):
                    param_total_dict_copy.pop(param)

            local_rank_id = transform_rank % dst_stage_device_num
            transform_param_list = _transform_parallel_checkpoint(local_rank_id, param_total_dict_copy,
                                                                  param_attr_dict, src_strategy_list, dst_strategy_list)
            save_checkpoint_file = "{}{}.ckpt".format(ckpt_prefix, transform_rank)
            save_checkpoint_file_dir = os.path.join(dst_checkpoints_dir, "rank_{}".format(transform_rank))
            if not os.path.exists(save_checkpoint_file_dir):
                _make_dir(save_checkpoint_file_dir, "path")
            save_checkpoint_file_name = os.path.join(save_checkpoint_file_dir, save_checkpoint_file)
            ms.save_checkpoint(transform_param_list, save_checkpoint_file_name)
            del param_total_dict_copy
        del param_total_dict
