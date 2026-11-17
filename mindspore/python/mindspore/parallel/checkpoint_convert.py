# Copyright 2025 Huawei Technologies Co., Ltd
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
"""Convert distributed checkpoint"""
from __future__ import absolute_import

__all__ = ["rank_list_for_convert", "convert_checkpoint_by_rank", "convert_checkpoints"]

from mindspore.parallel.checkpoint_transform import rank_list_for_transform, transform_checkpoint_by_rank, \
    transform_checkpoints


def rank_list_for_convert(rank_id, src_strategy_file=None, dst_strategy_file=None):
    """
    List of original distributed checkpoint rank index for obtaining the target checkpoint of a rank_id during the
    distributed checkpoint conversion.

    Args:
        rank_id (int): The rank of which distributed checkpoint needs to be obtained after conversion.
        src_strategy_file (str): Name of source sharding strategy file which is saved by
                        `mindspore.parallel.auto_parallel.AutoParallel(cell).save_param_strategy_file(file_path)`.
                        when the `src_strategy_file` is ``None``, it means that the source sharding strategy is
                        without any sharing for each parameter. Default: ``None``.
        dst_strategy_file (str): Name of destination sharding strategy file which is saved by
                        `mindspore.parallel.auto_parallel.AutoParallel(cell).save_param_strategy_file(file_path)`.
                        when the `dst_strategy_file` is ``None``,
                        it means that the destination sharding strategy
                        is without any sharing for each parameter. Default: ``None``.

    Returns:
        List, the rank list required for converting the distributed checkpoint of rank_id.

    Raises:
        ValueError: `src_strategy_file` or `dst_strategy_file` is incorrect.
        TypeError: `src_strategy_file` or `dst_strategy_file` is not a string.
        TypeError: `rank_id` is not an int.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> from mindspore.parallel import rank_list_for_convert
        >>> rank_id = 0
        >>> rank_list = rank_list_for_convert(rank_id, "./src_strategy.ckpt", "./dst_strategy.ckpt")
        >>> checkpoint_files_map = {}
        >>> for rank in rank_list:
        ...     checkpoint_files_map[rank] = "./{}-100_2.ckpt".format(rank)

    """
    return rank_list_for_transform(rank_id, src_strategy_file, dst_strategy_file)


def convert_checkpoint_by_rank(rank_id, checkpoint_files_map, save_checkpoint_file_name,
                               src_strategy_file=None, dst_strategy_file=None):
    """
    Convert distributed checkpoint from source sharding strategy to destination sharding strategy by rank
    for a network.

    Args:
        rank_id (int): The rank of which distributed checkpoint needs to be obtained after conversion.
        checkpoint_files_map (dict): The checkpoint files map whose key is the rank id and the value is
                                     the checkpoint file name.
        save_checkpoint_file_name (str): The file name to save the converted checkpoint.
        src_strategy_file (str): Name of source sharding strategy file which is saved by
                        :func:`mindspore.parallel.auto_parallel.AutoParallel.save_param_strategy_file`.
                        When the `src_strategy_file` is None, it means that the source sharding strategy is
                        without any sharing for each parameter. Default: ``None``.
        dst_strategy_file (str): Name of destination sharding strategy file which is saved by
                        :func:`mindspore.parallel.auto_parallel.AutoParallel.save_param_strategy_file`.
                        When the `dst_strategy_file` is ``None``,
                        it means that the destination sharding strategy
                        is without any sharing for each parameter. Default: ``None``.

    Raises:
        ValueError: `src_strategy_file` or `dst_strategy_file` is incorrect.
        ValueError: item in `checkpoint_files_map` is incorrect.
        ValueError: `save_checkpoint_file_name` does not end with ".ckpt".
        TypeError: `checkpoint_files_map` is not a dict.
        TypeError: `src_strategy_file` or `dst_strategy_file` is not a string.
        TypeError: `rank_id` is not an int.
        TypeError: `save_checkpoint_file_name` is not a string.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> from mindspore.parallel import rank_list_for_convert, convert_checkpoint_by_rank
        >>> dst_device_num = 8
        >>> for rank_id in range(dst_device_num):
        ...     rank_list = rank_list_for_convert(rank_id, "./src_strategy.ckpt", "./dst_strategy.ckpt")
        ...     checkpoint_files_map = {}
        ...     for rank in rank_list:
        ...         checkpoint_files_map[rank] = "./origin_checkpoint_rank{}/{}-100_2.ckpt".format(rank, rank)
        ...     save_checkpoint_file_name = "./new_checkpoint_rank{}/{}-100_2.ckpt".format(rank_id, rank_id)
        ...     convert_checkpoint_by_rank(rank_id, checkpoint_files_map, save_checkpoint_file_name,
        ...                                  "./src_strategy.ckpt", "./dst_strategy.ckpt")

    """
    transform_checkpoint_by_rank(rank_id, checkpoint_files_map, save_checkpoint_file_name,
                                 src_strategy_file, dst_strategy_file)


def convert_checkpoints(src_checkpoints_dir, dst_checkpoints_dir, ckpt_prefix, src_strategy_file=None,
                        dst_strategy_file=None, process_num=1, output_format="ckpt"):
    """
    Convert distributed checkpoint from source sharding strategy to destination sharding strategy for a rank.

    Note:
        The `src_checkpoints_dir` directory structure should be organized like "src_checkpoints_dir/rank_0/a.ckpt", the
        rank number should be set to a subdirectory and the checkpoint file is stored in this subdirectory. If multiple
        files exist in a rank directory, the last file in the lexicographic order would be selected.

        The number of multiprocess settings is related to the size of the host, and it is not recommended to set it
        too large, otherwise it may cause freezing.

    Args:
        src_checkpoints_dir (str): The source checkpoints directory.
        dst_checkpoints_dir (str): The destination checkpoints directory to save the converted checkpoints.
        ckpt_prefix (str): The destination checkpoint name prefix.
        src_strategy_file (str, optional): Name of source sharding strategy file which is saved by
                        'mindspore.parallel.auto_parallel.AutoParallel.save_param_strategy_file'.
                        When the 'src_strategy_file' is None, it means that the source sharding strategy is
                        without any sharing for each parameter. Default: ``None``.
        dst_strategy_file (str, optional): Name of destination sharding strategy file which is saved by
                        'mindspore.parallel.auto_parallel.AutoParallel.save_param_strategy_file'.
                        When the 'dst_strategy_file' is None, it means that the destination sharding strategy
                        is without any sharing for each parameter. Default: ``None``.
        process_num (int, optional): Number of processes to use for parallel processing. Defaults: 1.
        output_format (str, optional): Control the format of the output checkpoint after conversion.
            It can be set to either "ckpt" or "safetensors". Default: "ckpt".

    Raises:
        ValueError: `src_strategy_file` or `dst_strategy_file` is incorrect.
        NotADirectoryError: `src_checkpoints_dir` or `dst_checkpoints_dir` is not a directory.
        ValueError: The checkpoint file is missing in `src_checkpoints_dir`.
        TypeError: `src_strategy_file` or `dst_strategy_file` is not a string.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> from mindspore.parallel import convert_checkpoints
        >>> convert_checkpoints(src_checkpoints_dir, dst_checkpoints_dir, "dst_checkpoint",
        ...                       "./src_strategy.ckpt", "./dst_strategy.ckpt")

    """
    transform_checkpoints(src_checkpoints_dir, dst_checkpoints_dir, ckpt_prefix, src_strategy_file,
                          dst_strategy_file, process_num, output_format)
