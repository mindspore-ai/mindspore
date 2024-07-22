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
"""Transform distributed checkpoint"""
from __future__ import absolute_import

import os
import glob
import copy
from collections import defaultdict
import numpy as np
import mindspore as ms
from mindspore.common import dtype as mstype
from mindspore.parallel._utils import _is_in_auto_parallel_mode
from mindspore.parallel._parallel_serialization import _rank_list_for_transform_parallel_checkpoint, \
    _transform_parallel_checkpoint, _get_device_num_from_strategy, _make_dir, \
    _extract_layout_map, _extract_src_dst_layout_map, _parameter_not_in_local_stage, _extract_pipeline_stage_num, \
    _merge_protobuf_strategy, _merge_json_strategy, _extract_src_dst_layout_map_by_src


__all__ = ["merge_pipeline_strategys", "rank_list_for_transform", "transform_checkpoint_by_rank",
           "transform_checkpoints", "sync_pipeline_shared_parameters", "load_segmented_checkpoints"]


def merge_pipeline_strategys(src_strategy_dirs, dst_strategy_file):
    """
    Merge parallel strategy between all pipeline stages in pipeline parallel mode.
    For more details about converting distributed Checkpoint, please refer to
    `Model Transformation <https://www.mindspore.cn/tutorials/experts/en/master/parallel/model_transformation.html>`_.

    Note:
        Strategy file of each pipeline stage should be included in src_strategy_dirs.

    Args:
        src_strategy_dirs (str): The directory of strategy files including all pipeline stage which is saved by
                                 'mindspore.set_auto_parallel_context(strategy_ckpt_save_file)'.
        dst_strategy_file (str): The file merged strategy to save.

    Raises:
        NotADirectoryError: `src_strategy_dirs` is not a directory.

    Examples:
        >>> import mindspore as ms
        >>> # src_strategy_dir/stra0.ckpt, src_strategy_dir/stra1.ckpt ... src_strategy_dir/stra127.ckpt
        >>> ms.merge_pipeline_strategys("./src_strategy_dir", "./dst_strategy.ckpt")

    """
    dst_strategy_dir, _ = os.path.split(dst_strategy_file)
    if not os.path.exists(dst_strategy_dir):
        _make_dir(dst_strategy_dir, "path")
    if not os.path.isdir(src_strategy_dirs):
        raise NotADirectoryError("src_strategy_dirs {} is not a directory.".format(src_strategy_dirs))
    src_strategy_files_protobuf = glob.glob(os.path.join(src_strategy_dirs, "*.ckpt"))
    src_strategy_files_json = glob.glob(os.path.join(src_strategy_dirs, "*.json"))
    if src_strategy_files_protobuf and src_strategy_files_json:
        raise ValueError("The strategys format should be all '.ckpt' or all '.json'")
    is_protobuf = len(src_strategy_files_protobuf) > 0
    if is_protobuf:
        _merge_protobuf_strategy(src_strategy_files_protobuf, dst_strategy_file)
    else:
        _merge_json_strategy(src_strategy_files_json, dst_strategy_file)



def rank_list_for_transform(rank_id, src_strategy_file=None, dst_strategy_file=None):
    """
    List of original distributed checkpoint rank index for obtaining the target checkpoint of a rank_id during the
    distributed checkpoint conversion. For more details about converting distributed Checkpoint, please refer to
    `Model Transformation <https://www.mindspore.cn/tutorials/experts/en/master/parallel/model_transformation.html>`_.

    Args:
        rank_id (int): The rank of which distributed checkpoint needs to be obtained after conversion.
        src_strategy_file (str): Name of source sharding strategy file which saved by
                                 `mindspore.set_auto_parallel_context(strategy_ckpt_save_file)`.
                                 when the `src_strategy_file` is ``None``, it means that the source sharding strategy is
                                 without any sharing for each parameter. Default: ``None``.
        dst_strategy_file (str): Name of destination sharding strategy file which saved by
                                 `mindspore.set_auto_parallel_context(strategy_ckpt_save_file)`.
                                 when the `dst_strategy_file` is ``None``,
                                 it means that the destination sharding strategy
                                 is without any sharing for each parameter. Default: ``None``.

    Returns:
        List, the rank list required for converting the distributed checkpoint of rank_id.

    Raises:
        ValueError: `src_strategy_file` or `dst_strategy_file` is incorrect.
        TypeError: `src_strategy_file` or `dst_strategy_file` is not a string.
        TypeError: `rank_id` is not an int.

    Examples:
        >>> import mindspore as ms
        >>> rank_id = 0
        >>> rank_list = ms.rank_list_for_transform(rank_id, "./src_strategy.ckpt", "./dst_strategy.ckpt")
        >>> checkpoint_files_map = {}
        >>> for rank in rank_list:
        ...     checkpoint_files_map[rank] = "./pangu{}-100_2.ckpt".format(rank)

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
    result_list = list(result_set)
    result_list.sort(reverse=True)
    return list(result_list)


def transform_checkpoint_by_rank(rank_id, checkpoint_files_map, save_checkpoint_file_name,
                                 src_strategy_file=None, dst_strategy_file=None):
    """
    Transform distributed checkpoint from source sharding strategy to destination sharding strategy by rank
    for a network. For more details about converting distributed Checkpoint, please refer to
    `Model Transformation <https://www.mindspore.cn/tutorials/experts/en/master/parallel/model_transformation.html>`_.

    Args:
        rank_id (int): The rank of which distributed checkpoint needs to be obtained after conversion.
        checkpoint_files_map (dict): The checkpoint files map whose key is the rank id and the value is
                                     the checkpoint file name.
        save_checkpoint_file_name (str): The file name to save the converted checkpoint.
        src_strategy_file (str): Name of source sharding strategy file which saved by
                                 'mindspore.set_auto_parallel_context(strategy_ckpt_save_file)'.
                                 when the `src_strategy_file` is None, it means that the source sharding strategy is
                                 without any sharing for each parameter. Default: ``None``.
        dst_strategy_file (str): Name of destination sharding strategy file which saved by
                                 'mindspore.set_auto_parallel_context(strategy_ckpt_save_file)'.
                                 when the `dst_strategy_file` is ``None``,
                                 it means that the destination sharding strategy
                                 is without any sharing for each parameter. Default: ``None``.

    Raises:
        ValueError: `src_strategy_file` or `dst_strategy_file` is incorrect.
        ValueError: item in `checkpoint_files_map` is incorrect.
        ValueError: `save_checkpoint_file_name` is not end with ".ckpt".
        TypeError: `checkpoint_files_map` is not a dict.
        TypeError: `src_strategy_file` or `dst_strategy_file` is not a string.
        TypeError: `rank_id` is not an int.
        TypeError: `save_checkpoint_file_name` is not a string.

    Examples:
        >>> import mindspore as ms
        >>> dst_device_num = 8
        >>> for rank_id in range(dst_device_num):
        ...     rank_list = ms.rank_list_for_transform(rank_id, "./src_strategy.ckpt", "./dst_strategy.ckpt")
        ...     checkpoint_files_map = {}
        ...     for rank in rank_list:
        ...         checkpoint_files_map[rank] = "./origin_checkpoint_rank{}/pangu{}-100_2.ckpt".format(rank)
        ...     save_checkpoint_file_name = "./new_checkpoint_rank{}/pangu{}-100_2.ckpt".format(rank_id)
        ...     ms.transform_checkpoint_by_rank(rank_id, checkpoint_files_map, save_checkpoint_file_name,
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
    param_type_dict = defaultdict(dict)
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
            param_type_dict[param_name][src_rank] = str(param.data.dtype)
            if param.data.dtype == mstype.bfloat16:
                param.set_dtype(mstype.float32)
            param_total_dict[param_name][src_rank] = param.data.asnumpy()
            param_attr_dict[param_name][src_rank] = (param.requires_grad, param.layerwise_parallel)
    local_rank_id = rank_id % dst_stage_device_num
    transform_param_list = _transform_parallel_checkpoint(local_rank_id, param_total_dict,
                                                          param_attr_dict, src_strategy_list, dst_strategy_list,
                                                          param_type_dict)
    ms.save_checkpoint(transform_param_list, save_checkpoint_file_name)


def _transform_checkpoint_by_stage(src_checkpoints_dir, dst_checkpoints_dir, ckpt_prefix, src_strategy_file,
                                   dst_strategy_file=None):
    """Transform checkpoint for stage in src_strategy_file"""
    param_total_dict = defaultdict(dict)
    param_attr_dict = defaultdict(dict)
    param_type_dict = defaultdict(dict)
    src_strategy_list, dst_strategy_list, stage_id = _extract_src_dst_layout_map_by_src(src_strategy_file, \
                                                                                             dst_strategy_file)
    src_stage_device_num = np.prod(src_strategy_list.get(list(src_strategy_list.keys())[0])[0]) if src_strategy_list \
                                                                                                   is not None else 1
    dst_stage_device_num = np.prod(dst_strategy_list.get(list(dst_strategy_list.keys())[0])[0]) if dst_strategy_list \
                                                                                                   is not None else 1
    origin_dst_strategy_list = _extract_layout_map(dst_strategy_file)
    origin_src_strategy_list = _extract_layout_map(src_strategy_file)
    checkpoint_files_map = {}
    src_rank_id_start = stage_id * src_stage_device_num
    for local_rank in range(src_stage_device_num):
        rank_id = src_rank_id_start + local_rank
        checkpoint_file_name = os.path.join(src_checkpoints_dir, "rank_{}".format(rank_id), "*.ckpt")
        rank_ckpts = glob.glob(checkpoint_file_name)
        rank_ckpts.sort()
        for checkpoint_file in rank_ckpts:
            if not os.path.isfile(checkpoint_file):
                ms.log.warning("{} is not a checkpoint file.".format(checkpoint_file))
                continue
            checkpoint_files_map[rank_id] = checkpoint_file
    for rank, local_file in checkpoint_files_map.items():
        if not os.path.exists(local_file):
            raise ValueError("Checkpoint file {} in rank {} not exits: ".format(local_file, rank))
    for rank, file_name in checkpoint_files_map.items():
        ckpt_dict = ms.load_checkpoint(file_name)
        for param_name, param in ckpt_dict.items():
            # cut the parameter not in the pipeline stage.
            if _parameter_not_in_local_stage(param_name, origin_src_strategy_list, src_strategy_list) \
                    and _parameter_not_in_local_stage(param_name, origin_dst_strategy_list, dst_strategy_list):
                continue
            src_rank = rank % src_stage_device_num
            param_type_dict[param_name][src_rank] = str(param.data.dtype)
            if param.data.dtype == mstype.bfloat16:
                param.set_dtype(mstype.float32)
            param_total_dict[param_name][src_rank] = param.data.asnumpy()
            param_attr_dict[param_name][src_rank] = (param.requires_grad, param.layerwise_parallel)
    for local_rank_id in range(dst_stage_device_num):
        transform_param_list = _transform_parallel_checkpoint(local_rank_id, param_total_dict,
                                                              param_attr_dict, src_strategy_list, dst_strategy_list,
                                                              param_type_dict)
        save_checkpoint_file = "{}{}_part{}.ckpt".format(ckpt_prefix, local_rank_id, stage_id)
        save_checkpoint_file_dir = os.path.join(dst_checkpoints_dir, "rank_{}".format(local_rank_id))
        if not os.path.exists(save_checkpoint_file_dir):
            _make_dir(save_checkpoint_file_dir, "path")
        save_checkpoint_file_name = os.path.join(save_checkpoint_file_dir, save_checkpoint_file)
        ms.save_checkpoint(transform_param_list, save_checkpoint_file_name)


def _transform_checkpoints(src_checkpoints_dir, dst_checkpoints_dir, ckpt_prefix, src_strategy_file=None,
                           dst_strategy_file=None):
    """Transform checkpoints for all stages in src_strategy_file"""
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
        param_type_dict = defaultdict(dict)
        needed_rank_list = needed_rank_list_key.split("-")
        for needed_rank in needed_rank_list:
            ckpt_dict = ms.load_checkpoint(all_checkpoint_files_map.get(int(needed_rank)))
            for param_name, param in ckpt_dict.items():
                src_rank = int(needed_rank) % src_stage_device_num
                param_type_dict[param_name][src_rank] = str(param.data.dtype)
                if param.data.dtype == mstype.bfloat16:
                    param.set_dtype(mstype.float32)
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
                                                                  param_attr_dict, src_strategy_list, dst_strategy_list,
                                                                  param_type_dict)
            save_checkpoint_file = "{}{}.ckpt".format(ckpt_prefix, transform_rank)
            save_checkpoint_file_dir = os.path.join(dst_checkpoints_dir, "rank_{}".format(transform_rank))
            if not os.path.exists(save_checkpoint_file_dir):
                _make_dir(save_checkpoint_file_dir, "path")
            save_checkpoint_file_name = os.path.join(save_checkpoint_file_dir, save_checkpoint_file)
            ms.save_checkpoint(transform_param_list, save_checkpoint_file_name)
            del param_total_dict_copy
        del param_total_dict


def transform_checkpoints(src_checkpoints_dir, dst_checkpoints_dir, ckpt_prefix, src_strategy_file=None,
                          dst_strategy_file=None):
    """
    Transform distributed checkpoint from source sharding strategy to destination sharding strategy for a rank.
    For more details about converting distributed Checkpoint, please refer to
    `Model Transformation <https://www.mindspore.cn/tutorials/experts/en/master/parallel/model_transformation.html>`_.

    Note:
        The `src_checkpoints_dir` directory structure should be organized like "src_checkpoints_dir/rank_0/a.ckpt", the
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
        >>> import mindspore as ms
        >>> ms.transform_checkpoints(src_checkpoints_dir, dst_checkpoints_dir, "dst_checkpoint",
        ...                       "./src_strategy.ckpt", "./dst_strategy.ckpt")

    """
    if not os.path.isdir(src_checkpoints_dir):
        raise NotADirectoryError("src_checkpoints_dir {} is not a directory.".format(src_checkpoints_dir))
    _make_dir(dst_checkpoints_dir, "path")
    if not isinstance(ckpt_prefix, str):
        raise TypeError("The ckpt_prefix should be a str.")
    if src_strategy_file and os.path.dirname(src_strategy_file) and not os.path.exists(
            os.path.dirname(src_strategy_file)):
        raise ValueError("The director of src_strategy_file: {} is not exists.".
                         format(os.path.dirname(src_strategy_file)))
    if dst_strategy_file and os.path.dirname(dst_strategy_file) and not os.path.exists(
            os.path.dirname(dst_strategy_file)):
        raise ValueError("The director of dst_strategy_file: {} is not exists.".
                         format(os.path.dirname(dst_strategy_file)))
    src_layout_map = _extract_layout_map(src_strategy_file)
    dst_layout_map = _extract_layout_map(dst_strategy_file)
    pipeline_stage_num = _extract_pipeline_stage_num(src_strategy_file)
    dst_stage_num = _extract_pipeline_stage_num(dst_strategy_file)
    if src_layout_map:
        src_param_keys = {param_name for param_name in src_layout_map if
                          not param_name.startswith(("accu_grads", "adam_v", "adam_m"))}
    if dst_layout_map:
        dst_param_keys = {param_name for param_name in dst_layout_map if
                          not param_name.startswith(("accu_grads", "adam_v", "adam_m"))}
    layout_is_passed = src_layout_map and dst_layout_map

    if layout_is_passed and pipeline_stage_num == 1 and dst_stage_num == 1 and \
        src_param_keys.issubset(dst_param_keys) and len(src_param_keys) < len(dst_param_keys):
        ms.log.info("Transform checkpoint by every pipeline stage.")
        _transform_checkpoint_by_stage(src_checkpoints_dir, dst_checkpoints_dir, ckpt_prefix,
                                       src_strategy_file, dst_strategy_file)
    else:
        ms.log.info("Transform checkpoints by all pipeline stage.")
        _transform_checkpoints(src_checkpoints_dir, dst_checkpoints_dir, ckpt_prefix,
                               src_strategy_file, dst_strategy_file)


def _sync_params(name, param, layout):
    """synchronize single parameter"""
    if len(layout) < 10:
        ms.log.warning("The layout dict does not contain the pipeline_shared_param info %s", name)
        return

    pipeline_shared = layout[8]
    if not pipeline_shared:
        return

    is_send = layout[9]
    peer_rank = layout[10]
    sr_tag = layout[11]

    class SharedParameterSyncCell(ms.nn.Cell):
        """synchronize cell"""
        def __init__(self, param, is_send, peer_rank, sr_tag):
            super().__init__()
            self.param = param
            self.is_send = is_send
            self.ret = ms.Tensor([0])

            from mindspore.ops import Send, Receive
            if self.is_send:
                self.send = Send(sr_tag=sr_tag, dest_rank=peer_rank)
            else:
                self.receive = Receive(sr_tag=sr_tag, src_rank=peer_rank, shape=param.shape, dtype=param.dtype)

        def construct(self):
            if self.is_send:
                out = self.send(self.param)
                return ms.ops.functional.depend(self.ret, out)

            self.param = self.receive(self.ret)
            return ms.ops.functional.depend(self.ret, self.param)

    sync_net = SharedParameterSyncCell(param, is_send, peer_rank, sr_tag)
    sync_net()


def sync_pipeline_shared_parameters(net):
    """synchronize pipeline parallel stage shared parameters.
    Parameters may be shared between different stages. For example, `embedding table` is
    shared by `WordEmbedding` layer and `LMHead` layer, which are usually split into different stages. It is necessary
    to perform synchronization after `embedding table` changes.

    Note:
        The network should be compiled before synchronize pipeline parallel stage shared parameters.

    Args:
        net (nn.Cell): the inference network.

    Supported Platforms:
        ``Ascend``

    Examples:
        .. note::
            Before running the following examples, you need to configure the communication environment variables.

            For the Ascend device, users need to write a dynamic cluster startup script, please see the `Dynamic Cluster
            Startup <https://www.mindspore.cn/tutorials/experts/en/master/parallel/dynamic_cluster.html>`_ .

        >>> import numpy as np
        >>> import mindspore as ms
        >>> import mindspore.communication.management as D
        >>> from mindspore import lazy_inline, context, nn, ops, Parameter, Tensor
        >>> context.set_context(mode=context.GRAPH_MODE)
        >>> class Embedding(nn.Cell):
        ...     def __init__(self, shape):
        ...         super().__init__()
        ...         self.w = Parameter(Tensor(np.ones(shape), ms.float32), name='w')
        ...         self.matmul = ops.MatMul().shard(((1, 1), (1, 1)))
        ...     def construct(self, x):
        ...         return self.matmul(x, self.w), self.w
        ...
        >>> class LMHead(nn.Cell):
        ...     def __init__(self):
        ...         super().__init__()
        ...         self.matmul = ops.MatMul(transpose_b=True).shard(((1, 1), (1, 1)))
        ...     def construct(self, x, w):
        ...         return self.matmul(x, w)
        ...
        >>> class Network(nn.Cell):
        ...     @lazy_inline
        ...     def __init__(self):
        ...         super().__init__()
        ...         shape = (4, 4)
        ...         self.word_embedding = Embedding(shape)
        ...         self.lm_head = LMHead()
        ...         self.word_embedding.pipeline_stage = 0
        ...         self.lm_head.pipeline_stage = 1
        ...     def construct(self, x):
        ...         x, embed = self.word_embedding(x)
        ...         return self.lm_head(x, embed)
        ...
        >>> class PipelineCellInference(nn.Cell):
        ...     def __init__(self, network, micro_batch_num):
        ...         super().__init__()
        ...         self.network = network
        ...         self.micro_batch_num = micro_batch_num
        ...         self.concat = ops.Concat()
        ...     def construct(self, x):
        ...         ret = ()
        ...         for i in range(self.micro_batch_num):
        ...             micro_batch_size = x.shape[0] // self.micro_batch_num
        ...             start = micro_batch_size * i
        ...             end = micro_batch_size * (i + 1)
        ...             micro_input = x[start:end]
        ...             y = self.network(micro_input)
        ...             ret = ret + (y,)
        ...         ret = self.concat(ret)
        ...         return ret
        >>> D.init()
        >>> context.set_auto_parallel_context(parallel_mode='semi_auto_parallel', full_batch=True, pipeline_stages=2)
        >>> net = Network()
        >>> net = PipelineCellInference(net, 2)
        >>> net.set_train(False)
        >>> x = Tensor(np.ones((2, 4)), ms.float32)
        >>> net.compile(x)
        >>> ms.sync_pipeline_shared_parameters(net)
        >>> print(net.network.word_embedding.w.asnumpy())
        [[1. 1. 1. 1.]
         [1. 1. 1. 1.]
         [1. 1. 1. 1.]
         [1. 1. 1. 1.]]
    """

    if not isinstance(net, ms.nn.Cell):
        ms.log.critical("Failed to synchronize pipeline shared parameters.")
        msg = ("For 'sync_pipeline_shared_parameters', the argument 'net' should be a Cell, "
               "but got {}.".format(type(net)))
        raise TypeError(msg)

    layout_dict = net.parameter_layout_dict
    if _is_in_auto_parallel_mode() and not layout_dict:
        from mindspore.common.api import _get_parameter_layout
        layout_dict = _get_parameter_layout()

    # switch to standalone mode
    parallel_mode = ms.context.get_auto_parallel_context("parallel_mode")
    full_batch = ms.context.get_auto_parallel_context("full_batch")
    ms.context.set_auto_parallel_context(parallel_mode="stand_alone", full_batch=False)

    # synchronize shared parameter
    for name, param in net.parameters_and_names():
        if name in layout_dict:
            _sync_params(name, param, layout_dict[name])

    # restore parallel context
    ms.context.set_auto_parallel_context(parallel_mode=parallel_mode, full_batch=full_batch)


def load_segmented_checkpoints(ckpt_file_dir, net=None, strict_load=False, filter_prefix=None,
                               dec_key=None, dec_mode="AES-GCM", specify_prefix=None, choice_func=None):
    """
    Load checkpoint info from a specified file. If the specified ckpt_file_dir path contains multiple
    checkpoint files, all checkpoint files will be loaded one by one and the combined dictionary will be return.

    Note:
        - `specify_prefix` and `filter_prefix` do not affect each other.
        - If none of the parameters are loaded from checkpoint file, it will throw ValueError.
        - `specify_prefix` and `filter_prefix` are in the process of being deprecated,
          `choice_func` is recommended instead.
          And using either of those two args will override `choice_func` at the same time.

    Args:
        ckpt_file_dir (str): Checkpoint file directory.
        net (Cell): The network where the parameters will be loaded. Default: ``None`` .
        strict_load (bool): Whether to strict load the parameter into net. If ``False`` , it will load parameter
                            into net when parameter name's suffix in checkpoint file is the same as the
                            parameter in the network. When the types are inconsistent perform type conversion
                            on the parameters of the same type, such as float32 to float16. Default: ``False`` .
        filter_prefix (Union[str, list[str], tuple[str]]): Deprecated(see `choice_func`). Parameters starting with the
            filter_prefix will not be loaded. Default: ``None`` .
        dec_key (Union[None, bytes]): Byte type key used for decryption. If the value is ``None`` , the decryption
                                      is not required. Default: ``None`` .
        dec_mode (str): This parameter is valid only when dec_key is not set to ``None`` . Specifies the decryption
                        mode, currently supports ``"AES-GCM"`` and ``"AES-CBC"`` and ``"SM4-CBC"`` .
                        Default: ``"AES-GCM"`` .
        specify_prefix (Union[str, list[str], tuple[str]]): Deprecated(see `choice_func`). Parameters starting with the
            specify_prefix will be loaded. Default: ``None`` .
        choice_func (Union[None, function]) : Input value of the function is a Parameter name of type string,
            and the return value is a bool. If returns ``True`` , the Parameter
            that matches the custom condition will be loaded. If returns ``False`` , the Parameter that
            matches the custom condition will be removed. Default: ``None`` .

    Returns:
        Dict, key is parameter name, value is a Parameter or string. When the `append_dict` parameter of
        :func:`mindspore.save_checkpoint` and the `append_info` parameter of :class:`mindspore.train.CheckpointConfig`
        are used to save the checkpoint, `append_dict` and `append_info` are dict types, and their value are string,
        then the return value obtained by loading checkpoint is string, and in other cases the return value is
        Parameter.

    Raises:
        TypeError: Input ckpt_file_dir is not a string.
        ValueError: Checkpoint file directory doesn't exist. Or it's not a directory
        ValueError: Checkpoint file's format is incorrect.
        ValueError: Parameter's dict is None after load checkpoint file.
        TypeError: The type of `specify_prefix` or `filter_prefix` is incorrect.
    """
    if not isinstance(ckpt_file_dir, str):
        raise TypeError("The ckpt_file_dir should be a str.")
    if not os.path.isdir(ckpt_file_dir):
        raise ValueError("The dst_strategy_file: {} doesn't exist. Or it's not a directory".
                         format(ckpt_file_dir))
    checkpoint_file_name = os.path.join(ckpt_file_dir, "*.ckpt")
    rank_ckpts = glob.glob(checkpoint_file_name)
    parameter_dict = {}
    for checkpoint_file in rank_ckpts:
        parameter_dict.update(ms.load_checkpoint(checkpoint_file, net, strict_load, filter_prefix, dec_key,
                                                 dec_mode, specify_prefix, choice_func))
    return parameter_dict
