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
import mindspore as ms
from mindspore.parallel._parallel_serialization import _rank_list_for_transform_parallel_checkpoint, \
    _transform_parallel_checkpoint, _get_device_num_from_strategy, _make_dir


__all__ = ["rank_list_for_transform", "transform_checkpoint_by_rank", "transform_checkpoints"]


def rank_list_for_transform(rank_id, src_strategy_file=None, dst_strategy_file=None):
    """
    List of original distributed checkpoint rank index for obtaining the target checkpoint of a rank_id
    during the distributed checkpoint conversion.

    Note:
        Cannot transform pipeline parallel dimensions currently.

    Args:
        rank_id (int): The rank of which distributed checkpoint needs to be obtained after conversion.
        src_strategy_file (str): Name of source sharding strategy file, when the 'src_strategy_file' is None,
                                 it means that the source sharding strategy is without any sharing for each parameter.
                                 Default:None.
        dst_strategy_file (str): Name of destination sharding strategy file. when the 'dst_strategy_file' is None,
                                 it means that the source sharding strategy is without any sharing for each parameter.
                                 Default:None.

    Returns:
        List, the rank list required for converting the distributed checkpoint of rank_id.

    Raises:
        ValueError: src_strategy_file or dst_strategy_file is incorrect.
        TypeError: src_strategy_file or dst_strategy_file is not a string.
        TypeError: rank_id is not a int.

    Examples:
        >>> rank_id = 0
        >>> rank_list = rank_list_for_transform(rank_id, "./src_strategy.ckpt",
        >>> "./dst_strategy.ckpt")
        >>> checkpoint_files_map = {}
        >>> for rank in rank_list:
        >>>     checkpoint_files_map[rank] = "./pangu{}-100_2.ckpt".format(rank)

    """
    if not isinstance(rank_id, int):
        raise TypeError("The rank_id should be a int.")
    return _rank_list_for_transform_parallel_checkpoint(rank_id, src_strategy_file, dst_strategy_file)


def transform_checkpoint_by_rank(rank_id, checkpoint_files_map, save_checkpoint_file_name,
                                 src_strategy_file=None, dst_strategy_file=None):
    """
    Transform distributed checkpoint from source sharding strategy to destination sharding strategy by rank.

    Note:
        Cannot transform pipeline parallel dimensions currently.

    Args:
        rank_id (int): The rank of which distributed checkpoint needs to be obtained after conversion.
        checkpoint_files_map (dict): The checkpoint files map whose key is the rank id and the value is
                                     the checkpoint file name.
        save_checkpoint_file_name (str): The file name to save the converted checkpoint.
        src_strategy_file (str): Name of source sharding strategy file, when the 'src_strategy_file' is None,
                                 it means that the source sharding strategy is without any sharding for each parameter.
                                 Default:None.
        dst_strategy_file (str): Name of destination sharding strategy file. when the 'dst_strategy_file' is None,
                                 it means that the source sharding strategy is without any sharding for each parameter.
                                 Default:None.

    Raises:
        ValueError: src_strategy_file or dst_strategy_file is incorrect.
        ValueError: item in checkpoint_files_map is incorrect.
        TypeError: checkpoint_files_map is not a dict.
        TypeError: src_strategy_file or dst_strategy_file is not a string.
        TypeError: rank_id is not a int.

    Examples:
        >>> dst_device_num = 8
        >>> for rank_id in range(dst_device_num)
        >>>     rank_list = rank_list_for_transform(rank_id, "./src_strategy.ckpt",
        >>>                 "./dst_strategy.ckpt")
        >>>     checkpoint_files_map = {}
        >>>     for rank in rank_list:
        >>>         checkpoint_files_map[rank] = "./origin_checkpoint_rank{}/pangu{}-100_2.ckpt".format(rank)
        >>>     save_checkpoint_file_name = "./new_checkpoint_rank{}/pangu{}-100_2.ckpt".format(rank_id)
        >>>     transform_checkpoint_by_rank(rank_id, checkpoint_files_map, save_checkpoint_file_name,
        >>>                                           "./src_strategy.ckpt", "./dst_strategy.ckpt")

    """
    if not isinstance(checkpoint_files_map, dict):
        raise TypeError("The checkpoint_files_map should be a dict.")
    if not isinstance(rank_id, int):
        raise TypeError("The rank_id should be a int.")
    for rank, local_file in checkpoint_files_map.items():
        if not os.path.exists(local_file):
            raise ValueError("Checkpoint file {} in rank {} not exits: ".format(local_file, rank))
    param_total_dict = defaultdict(dict)
    param_attr_dict = defaultdict(dict)
    for rank, file_name in checkpoint_files_map.items():
        ckpt_dict = ms.load_checkpoint(file_name)
        for param_name, param in ckpt_dict.items():
            param_total_dict[param_name][rank] = param.data.asnumpy()
            param_attr_dict[param_name][rank] = (param.requires_grad, param.layerwise_parallel)
    transform_param_list = _transform_parallel_checkpoint(rank_id, param_total_dict, param_attr_dict, src_strategy_file,
                                                          dst_strategy_file)
    ms.save_checkpoint(transform_param_list, save_checkpoint_file_name)


def transform_checkpoints(src_checkpoints_dir, dst_checkpoints_dir, ckpt_prefix, src_strategy_file=None,
                          dst_strategy_file=None):
    """
    Transform distributed checkpoint from source sharding strategy to destination sharding strategy.

    Note:
        The src_checkpoints_dir directory structure should be organized like "src_checkpoints_dir/rank_0/a.ckpt", the
        rank number should be set to a subdirectory and the checkpoint file is stored in this subdirectory. If multiple
        files exist in a rank directory, the last file in the lexicgraphic order would be selected.
        Cannot transform pipeline parallel dimensions currently.

    Args:
        src_checkpoints_dir (str): The source checkpoints directory.
        dst_checkpoints_dir (str): The destination checkpoints directory to save the converted checkpoints.
        src_strategy_file (str): Name of source sharding strategy file, when the 'src_strategy_file' is None,
                                 it means that the source sharding strategy is without any sharding for each parameter.
                                 Default:None.
        dst_strategy_file (str): Name of destination sharding strategy file. when the 'dst_strategy_file' is None,
                                 it means that the source sharding strategy is without any sharding for each parameter.
                                 Default:None.

    Raises:
        ValueError: src_strategy_file or dst_strategy_file is incorrect.
        NotADirectoryError: src_checkpoints_dir or dst_checkpoints_dir is not a directory.
        ValueError: The checkpoint file is missing in src_checkpoints_dir.
        TypeError: src_strategy_file or dst_strategy_file is not a string.

    Examples:
        >>> transform_checkpoints(src_checkpoints_dir, dst_checkpoints_dir,
        >>>                       "./src_strategy.ckpt", "./dst_strategy.ckpt")

    """
    if not os.path.isdir(src_checkpoints_dir):
        raise NotADirectoryError("src_checkpoints_dir {} is not a directory.".format(src_checkpoints_dir))
    if not os.path.isdir(dst_checkpoints_dir):
        raise NotADirectoryError("dst_checkpoints_dir {} is not a directory.".format(dst_checkpoints_dir))
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
        for checkpoint_file in glob.glob(checkpoint_file_name):
            if not os.path.isfile(checkpoint_file):
                ms.log.warning("{} is not a checkpoint file.".format(checkpoint_file))
                continue
            all_checkpoint_files_map[rank_id] = checkpoint_file

    needed_rank_list_map = defaultdict(list)
    dst_device_num = _get_device_num_from_strategy(dst_strategy_file)
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
                param_total_dict[param_name][int(needed_rank)] = param.data.asnumpy()
                param_attr_dict[param_name][int(needed_rank)] = (param.requires_grad, param.layerwise_parallel)
        for transform_rank in transform_rank_list:
            param_total_dict_copy = copy.deepcopy(param_total_dict)
            transform_param_list = _transform_parallel_checkpoint(transform_rank, param_total_dict_copy,
                                                                  param_attr_dict, src_strategy_file, dst_strategy_file)
            save_checkpoint_file = os.path.join(ckpt_prefix, str(transform_rank), ".ckpt")
            save_checkpoint_file_dir = os.path.join(dst_checkpoints_dir, "rank_{}".format(transform_rank))
            if not os.path.exists(save_checkpoint_file_dir):
                _make_dir(save_checkpoint_file_dir, "path")
            save_checkpoint_file_name = os.path.join(save_checkpoint_file_dir, save_checkpoint_file)
            ms.save_checkpoint(transform_param_list, save_checkpoint_file_name)
            del param_total_dict_copy
        del param_total_dict
