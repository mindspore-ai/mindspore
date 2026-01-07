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
from multiprocessing import Pool
from collections import defaultdict
import numpy as np
import mindspore as ms
from mindspore import log as logger
from mindspore import _checkparam as Validator
from mindspore.common import dtype as mstype
from mindspore.common.parameter import Parameter
from mindspore.common.tensor import Tensor
from mindspore.communication.management import get_rank, get_group_size
from mindspore.parallel._tensor import _load_tensor, _reshape_param_data, _reshape_param_data_with_weight, \
    _get_tensor_slice_index, _get_tensor_strategy
from mindspore.parallel._utils import _is_in_auto_parallel_mode, _get_pipeline_stages, _infer_rank_list, \
    _remove_repeated_slices, _get_auto_parallel_net, _check_path_safe, _check_path_writable, _mstx_range_decorator
from mindspore.parallel._parallel_serialization import _rank_list_for_transform_parallel_checkpoint, \
    _transform_parallel_checkpoint, _get_device_num_from_strategy, _make_dir, _build_searched_strategy, \
    _extract_layout_map, _extract_src_dst_layout_map, _parameter_not_in_local_stage, _extract_pipeline_stage_num, \
    _merge_protobuf_strategy, _merge_json_strategy, _extract_src_dst_layout_map_by_src, _convert_to_list, \
    _check_checkpoint_file, _check_predict_strategy, _gather_tasks_load_dis, _get_param_list_when_first_dim_sharded, \
    _convert_to_layout, _restore_group_info_list
from mindspore._c_expression import AutoParallelContext
from mindspore.parallel.transform_safetensors import _transform_safetensors, _collect_safetensor_files, \
    _load_parallel_checkpoint

__all__ = ["merge_pipeline_strategys", "rank_list_for_transform", "transform_checkpoint_by_rank",
           "transform_checkpoints", "sync_pipeline_shared_parameters", "load_segmented_checkpoints",
           "load_distributed_checkpoint", "merge_sliced_parameter", "restore_group_info_list",
           "build_searched_strategy"]


def merge_pipeline_strategys(src_strategy_dirs, dst_strategy_file):
    """
    Aggregate the sharding strategy files of all pipeline parallel subgraphs to the destination file.

    Note:
        Strategy file of each pipeline stage should be included in src_strategy_dirs.

    Args:
        src_strategy_dirs (str): The directory of strategy files including all pipeline stage which is saved by
                                 :func:`mindspore.parallel.auto_parallel.AutoParallel.save_param_strategy_file`.
        dst_strategy_file (str): The file merged strategy to save.

    Raises:
        NotADirectoryError: `src_strategy_dirs` is not a directory.

    Examples:
        >>> import mindspore as ms
        >>> # src_strategy_dir/stra0.ckpt, src_strategy_dir/stra1.ckpt ... src_strategy_dir/stra127.ckpt
        >>> ms.parallel.merge_pipeline_strategys("./src_strategy_dir", "./dst_strategy.ckpt")

    """
    dst_strategy_file = os.path.normpath(dst_strategy_file)
    dst_strategy_file = os.path.abspath(dst_strategy_file)
    dst_strategy_dir = os.path.dirname(dst_strategy_file)
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


def merge_sliced_parameter(sliced_parameters, strategy=None):
    """
    Merge parameter slices into one parameter. Used in the case of distributed inference.

    Args:
        sliced_parameters (list[Parameter]): Parameter slices in order of rank id.
        strategy (Optional[dict], optional): Parameter slice strategy, whose key is parameter name and
            value is slice strategy of this parameter. If strategy is None, just merge
            parameter slices in 0 axis order. Default: ``None``.

    Returns:
        Parameter, the merged parameter which has the whole data.

    Raises:
        ValueError: Failed to merge.
        TypeError: The sliced_parameters is incorrect or strategy is not dict.
        KeyError: The parameter name is not in keys of strategy.

    Examples:
        >>> import numpy as np
        >>> import mindspore as ms
        >>> from mindspore import Tensor, Parameter
        >>>
        >>> sliced_parameters = [
        ...                      Parameter(Tensor(np.array([0.00023915, 0.00013939, -0.00098059])),
        ...                                "network.embedding_table"),
        ...                      Parameter(Tensor(np.array([0.00015815, 0.00015458, -0.00012125])),
        ...                                "network.embedding_table"),
        ...                      Parameter(Tensor(np.array([0.00042165, 0.00029692, -0.00007941])),
        ...                                "network.embedding_table"),
        ...                      Parameter(Tensor(np.array([0.00084451, 0.00089960, -0.00010431])),
        ...                                "network.embedding_table")]
        >>> merged_parameter = ms.merge_sliced_parameter(sliced_parameters)
        >>> print(merged_parameter)
        Parameter (name=network.embedding_table, shape=(12,), dtype=Float64, requires_grad=True)
    """
    if not isinstance(sliced_parameters, list):
        raise TypeError(f"For 'merge_sliced_parameter', the argument 'sliced_parameters' should be list, "
                        f"but got {type(sliced_parameters)}.")

    if not sliced_parameters:
        raise ValueError("For 'merge_sliced_parameter', the argument 'sliced_parameters' should not be empty.")

    if strategy and not isinstance(strategy, dict):
        raise TypeError(f"For 'merge_sliced_parameter', the argument 'strategy' should be dict, "
                        f"but got {type(strategy)}.")

    try:
        parameter_name = sliced_parameters[0].name
        parameter_shape = sliced_parameters[0].data.shape
        parameter_shape_length = len(parameter_shape)
    except BaseException as e:
        raise TypeError(e.__str__() + f" For 'merge_sliced_parameter', the element in 'sliced_parameters' should be "
                                      f"'Parameter', but got {type(sliced_parameters[0])} at index 0.") from e

    is_even = True
    for index, parameter in enumerate(sliced_parameters):
        if not isinstance(parameter, Parameter):
            raise TypeError(f"For 'merge_sliced_parameter', the element in 'sliced_parameters' should be 'Parameter', "
                            f"but got {type(parameter)} at index {index}.")

        if parameter.name != parameter_name \
                or len(parameter.data.shape) != parameter_shape_length \
                or parameter.data.shape[1:] != parameter_shape[1:]:
            raise ValueError(f"For 'merge_sliced_parameter', please make sure that the elements in 'slice_parameters'"
                             f" have the same name, dimension length and shape except 0 axis. The name, dimension "
                             f"length, shape except 0 axis should be {parameter_name}, {parameter_shape_length}, "
                             f"{parameter_shape[1:]}, but got name: {parameter.name}, dimension length: "
                             f"{len(parameter.data.shape)}, shape except 0 axis: {parameter.data.shape[1:]} "
                             f"at index {index}.")

        if parameter.data.shape != parameter_shape:
            is_even = False

    layerwise_parallel = sliced_parameters[0].layerwise_parallel
    requires_grad = sliced_parameters[0].requires_grad
    sliced_data = []
    for parameter in sliced_parameters:
        if parameter.data.dtype == mstype.bfloat16:
            from mindspore.ops import Cast
            cpu_cast = Cast().set_device("CPU")
            sliced_data.append(cpu_cast(parameter.data, mstype.float32).asnumpy())
        else:
            sliced_data.append(parameter.data.asnumpy())

    if not strategy:
        merged_tensor = Tensor(np.concatenate(sliced_data))
        merged_parameter = Parameter(merged_tensor, parameter_name, requires_grad, layerwise_parallel)

    else:
        if parameter_name not in strategy.keys():
            raise KeyError(f"For 'merge_sliced_parameter', the parameter name {parameter_name} should be a key in "
                           f"the 'strategy'. Please check 'sliced_parameter' and 'strategy'.")
        merged_tensor = _merge_param_with_strategy(sliced_data, parameter_name, strategy, is_even)
        merged_parameter = Parameter(merged_tensor, parameter_name, requires_grad, layerwise_parallel)

    return merged_parameter


def _merge_and_split(sliced_params, train_strategy, predict_strategy):
    """Merge sliced parameter and split it according to the predict strategy."""
    merged_param = merge_sliced_parameter(sliced_params, train_strategy)
    if not predict_strategy:
        return merged_param
    param_name = merged_param.name
    tensor_layout = predict_strategy[param_name]
    rank = get_rank()
    split_tensor = _load_tensor(merged_param.data, tensor_layout[0], tensor_layout[1], rank_id=rank)
    requires_grad = merged_param.requires_grad
    layerwise_parallel = merged_param.layerwise_parallel
    if merged_param.data.dtype == mstype.bfloat16:
        split_param = Parameter(Tensor(split_tensor, mstype.bfloat16), param_name, requires_grad, layerwise_parallel)
    else:
        split_param = Parameter(split_tensor, param_name, requires_grad, layerwise_parallel)
    return split_param


def _merge_param_with_strategy(sliced_data, parameter_name, strategy, is_even):
    """
    Merge data slices to one tensor with whole data when strategy is not None.

    Args:
        sliced_data (list[numpy.ndarray]): Data slices in order of rank_id.
        parameter_name (str): Name of parameter.
        strategy (dict): Parameter slice strategy.
        is_even (bool): Slice manner that True represents slicing evenly and False represents slicing unevenly.

    Returns:
        Tensor, the merged Tensor which has the whole data.

    Raises:
        ValueError: Failed to merge.
    """
    layout = strategy.get(parameter_name)
    try:
        dev_mat = list(layout.dev_matrix[0].dim)
        tensor_map = list(layout.tensor_map[0].dim)
        param_split_shape = list(layout.param_split_shape[0].dim)
        field_size = int(layout.field)
    except BaseException as e:
        raise ValueError(f"{e.__str__()}. For 'merge_sliced_parameter'"
                         f", please make sure that 'strategy' is correct.") from e

    device_count = 1
    for dim in dev_mat:
        device_count *= dim

    if len(sliced_data) != device_count:
        raise ValueError(f"For 'merge_sliced_parameter', the length of 'sliced_parameters' should be equal to "
                         f"device_count. The length of 'sliced_parameters' is {len(sliced_data)}, but "
                         f"device_count is {device_count}.")

    if not param_split_shape:
        if not is_even:
            raise ValueError("For 'merge_sliced_parameter', the shape of every parameter in 'sliced_parameters' "
                             "should be the same when slice manner is even.")

        all_gather_tensor = Tensor(np.concatenate(sliced_data))

        if field_size > 0:
            merged_tensor = _reshape_param_data_with_weight(all_gather_tensor, dev_mat, field_size)
        else:
            merged_tensor = _reshape_param_data(all_gather_tensor, dev_mat, tensor_map)

    else:
        tensor_strategy = _get_tensor_strategy(dev_mat, tensor_map)

        slice_count = 1
        for dim in tensor_strategy:
            slice_count *= dim

        if len(param_split_shape) != slice_count:
            raise ValueError(f"For 'merge_sliced_parameter', the param_split_shape length in 'strategy' should be "
                             f"{slice_count}, but got {len(param_split_shape)}.")

        tensor_slices_new = list(range(slice_count))
        tensor_slices = sliced_data
        for i in range(device_count):
            slice_index = int(_get_tensor_slice_index(dev_mat, tensor_strategy, tensor_map, i))
            if tensor_slices[i].shape[0] != param_split_shape[slice_index]:
                raise ValueError(f"For 'merge_sliced_parameter', the slice {slice_index} should be "
                                 f"{param_split_shape[slice_index]} in 0 axis, but got "
                                 f"{tensor_slices[i].shape[0]}.")
            tensor_slices_new[slice_index] = np.array(tensor_slices[i])

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
        merged_tensor = Tensor(tensor_slices_new[0])

    return merged_tensor


def rank_list_for_transform(rank_id, src_strategy_file=None, dst_strategy_file=None):
    """
    List of original distributed checkpoint rank index for obtaining the target checkpoint of a rank_id during the
    distributed checkpoint conversion.

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
        >>> rank_list = ms.parallel.rank_list_for_transform(rank_id, "./src_strategy.ckpt", "./dst_strategy.ckpt")
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
    for a network.

    Args:
        rank_id (int): The rank of which distributed checkpoint needs to be obtained after conversion.
        checkpoint_files_map (dict): The checkpoint files map whose key is the rank id and the value is
                                     the checkpoint file name.
        save_checkpoint_file_name (str): The file name to save the converted checkpoint.
        src_strategy_file (str): Name of source sharding strategy file which saved by
                                 `mindspore.set_auto_parallel_context(strategy_ckpt_save_file)`.
                                 when the `src_strategy_file` is None, it means that the source sharding strategy is
                                 without any sharing for each parameter. Default: ``None``.
        dst_strategy_file (str): Name of destination sharding strategy file which saved by
                                 `mindspore.set_auto_parallel_context(strategy_ckpt_save_file)`.
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
    _check_path_safe(dst_checkpoints_dir, "dst_checkpoints_dir")
    dst_checkpoints_dir = os.path.realpath(dst_checkpoints_dir)
    _check_path_safe(ckpt_prefix, "ckpt_prefix")
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
            _check_path_writable(save_checkpoint_file_dir)
            save_checkpoint_file_name = os.path.join(save_checkpoint_file_dir, save_checkpoint_file)
            ms.save_checkpoint(transform_param_list, save_checkpoint_file_name)
            del param_total_dict_copy
        del param_total_dict


def transform_checkpoints(src_checkpoints_dir, dst_checkpoints_dir, ckpt_prefix, src_strategy_file=None,
                          dst_strategy_file=None, process_num=1, output_format="ckpt"):
    """
    Transform distributed checkpoint from source sharding strategy to destination sharding strategy for a rank.

    Note:
        The `src_checkpoints_dir` directory structure should be organized like "src_checkpoints_dir/rank_0/a.ckpt", the
        rank number should be set to a subdirectory and the checkpoint file is stored in this subdirectory. If multiple
        files exist in a rank directory, the last file in the lexicgraphic order would be selected.

        The number of multiprocess settings is related to the size of the host, and it is not recommended to set it
        too large, otherwise it may cause freezing.

        This function does not support converting remove_redundancy's checkpoint file.

    Args:
        src_checkpoints_dir (str): The source checkpoints directory.
        dst_checkpoints_dir (str): The destination checkpoints directory to save the converted checkpoints.
        ckpt_prefix (str): The destination checkpoint name prefix.
        src_strategy_file (str, optional): Name of source sharding strategy file which saved by
                                 'mindspore.set_auto_parallel_context(strategy_ckpt_save_file)'.
                                 when the 'src_strategy_file' is None, it means that the source sharding strategy is
                                 without any sharing for each parameter. Default:None.
        dst_strategy_file (str, optional): Name of destination sharding strategy file which saved by
                                 'mindspore.set_auto_parallel_context(strategy_ckpt_save_file)'.
                                 when the 'dst_strategy_file' is None, it means that the destination sharding strategy
                                 is without any sharing for each parameter. Default:None.
        process_num (int, optional): Number of processes to use for parallel processing. Defaults: 1.
        output_format (str, optional): Control the format of the output checkpoint after conversion.
            It can be set to either ``"ckpt"`` or ``"safetensors"``. Default: ``"ckpt"``.

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
    all_safetensor_files_map = _collect_safetensor_files(src_checkpoints_dir)
    all_ckpt_files_map = _collect_safetensor_files(src_checkpoints_dir, format='ckpt')
    if all_safetensor_files_map and all_ckpt_files_map:
        raise ValueError("For 'transform_checkpoints', the 'src_checkpoints_dir' cannot contain "
                         "both ckpt file and safetensors file simultaneously")
    if all_safetensor_files_map and not all_ckpt_files_map:
        _transform_safetensors(src_checkpoints_dir, dst_checkpoints_dir, ckpt_prefix, src_strategy_file,
                               dst_strategy_file, process_num, output_format)
        return
    if not all_safetensor_files_map and not all_ckpt_files_map:
        raise ValueError("For 'transform_checkpoints', the 'src_checkpoints_dir' can not be empty.")
    if all_ckpt_files_map and not all_safetensor_files_map and output_format == 'safetensors':
        raise ValueError("For 'transform_checkpoints', 'output_format' can not be 'safetensors' "
                         "when 'src_checkpoints_dir' only contains ckpt file.")

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
    if is_send:
        ms.ops.Send(sr_tag=sr_tag, dest_rank=peer_rank)(param)
    else:
        param.assign_value(ms.ops.Receive(sr_tag=sr_tag,
                                          src_rank=peer_rank,
                                          shape=param.shape,
                                          dtype=param.dtype)(param))

# pylint: disable=W0212
def sync_pipeline_shared_parameters(net):
    """Synchronization of shared weights between stages for pipeline parallel inference scenarios.
    For example, `embedding table` is
    shared by `WordEmbedding` layer and `LMHead` layer, which are usually split into different stages. It is necessary
    to perform synchronization after `embedding table` changes.

    Note:
        The network should be compiled before shared parameters are synchronized in the pipeline parallel stage.

    Args:
        net (Cell): the inference network.

    Raises:
        TypeError: `net` is not in Cell type.

    Supported Platforms:
        ``Ascend``

    Examples:
        .. note::
            Before running the following examples, you need to configure the communication environment variables.

            For the Ascend device, users need to write a dynamic cluster startup script, please see the `Dynamic Cluster
            Startup <https://www.mindspore.cn/tutorials/en/master/parallel/dynamic_cluster.html>`_ .

        >>> import numpy as np
        >>> import mindspore as ms
        >>> import mindspore.communication.management as D
        >>> from mindspore import lazy_inline, context, nn, ops, Parameter, Tensor
        >>> from mindspore.parallel.auto_parallel import AutoParallel
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
        >>> net = Network()
        >>> net = PipelineCellInference(net, 2)
        >>> net.set_train(False)
        >>> x = Tensor(np.ones((2, 4)), ms.float32)
        >>> net.compile(x)
        >>> pp_net = AutoParallel(net, parallel_mode="semi_auto")
        >>> pp_net.full_batch = True
        >>> pp_net.pipeline(stages=2, scheduler="1f1b")
        >>> ms.parallel.sync_pipeline_shared_parameters(pp_net)
        >>> print(pp_net.network.network.word_embedding.w.asnumpy())
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

    parallel_net = _get_auto_parallel_net(net)
    pipeline_stages = 1
    if type(parallel_net).__name__ != 'AutoParallel':
        pipeline_stages = _get_pipeline_stages()
    else:
        pipeline_stages = parallel_net._pipeline_stages
    if pipeline_stages < 2:
        return

    layout_dict = net.parameter_layout_dict
    if (_is_in_auto_parallel_mode() or (type(parallel_net).__name__ == 'AutoParallel')) and not layout_dict:
        from mindspore.common.api import _get_parameter_layout
        layout_dict = _get_parameter_layout()

    # switch to standalone mode
    if type(parallel_net).__name__ != 'AutoParallel':
        parallel_mode = ms.context.get_auto_parallel_context("parallel_mode")
        full_batch = ms.context.get_auto_parallel_context("full_batch")
        ms.context.set_auto_parallel_context(parallel_mode="stand_alone", full_batch=False)

    # synchronize shared parameter
    for name, param in net.parameters_and_names():
        if name in layout_dict:
            _sync_params(name, param, layout_dict[name])

    # restore parallel context
    if type(parallel_net).__name__ != 'AutoParallel':
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

    Supported Platforms:
        ``Ascend``
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


def set_op_strategy_config(mode="SAVE", path=""):
    """
    Set strategy json configuration when using sharding propagation.

    .. warning::
        - This is an experimental interface, may be changed or canceled in the future, please use the api
          :func:`mindspore.parallel.auto_parallel.AutoParallel.load_operator_strategy_file` or
          :func:`mindspore.parallel.auto_parallel.AutoParallel.save_operator_strategy_file` instead;
        - This interface currently doesn't support saving or loading strategies using layout.

    Note:
        - It only works when `parallel_mode=ParallelMode.AUTO_PARALLEL` and `search_mode='sharding_propagation'`.
        - It only supports saving and reloading with the same configuration for the same network. If the network
          or training hyperparameters are modified after using the `SAVE` mode to save the strategies of operator
          to the setting json file, which may lead to the failure of using the `LOAD` mode to load operator
          strategies from json.
        - When performing distributed training, users can first save the strategy using dryrun on a single device
          and then load strategy to perform distributed training.

    Args:
        mode (str): The parameter for choosing save or load .json file. Default value: ``"SAVE"`` .
        path (str): Path to save or load parallel strategy json, must be an absolute path. Default value: ``""`` .

    Raises:
        KeyError: When type is not ``"SAVE"`` or ``"LOAD"`` .
        KeyError: When path does not end in ``".json"`` .
        KeyError: When path is not an absolute path.
    """
    if not os.path.isabs(path):
        raise KeyError("File path must be an absolute path")
    _, file_type = os.path.splitext(path)
    if file_type != ".json":
        raise KeyError("File type must be .json")
    dir_path = os.path.dirname(path)

    normalized_path = os.path.abspath(os.path.realpath(path))
    dangerous_paths = ['/etc', '/usr', '/bin', '/sbin', '/boot', '/proc', '/sys']
    for dangerous_path in dangerous_paths:
        if normalized_path.startswith(dangerous_path):
            raise PermissionError(
                f"Writing to system directory '{dangerous_path}' is not allowed"
            )

    if dir_path and not os.path.exists(dir_path):
        os.makedirs(dir_path, mode=0o700, exist_ok=True)
    check_mode_type = ["SAVE", "LOAD"]
    if mode in check_mode_type:
        if AutoParallelContext.get_instance() is None:
            raise ValueError("Get AutoParallelContext instance failed!!!")
        AutoParallelContext.get_instance().set_ops_strategy_json_config(mode, path, "all")
    else:
        raise KeyError("Type must be 'SAVE' or 'LOAD'")


def build_searched_strategy(strategy_filename):
    """
    Extract the sharding strategy for each parameter in the network from the strategy file
    for distributed inference scenarios.

    Args:
        strategy_filename (str): Name of strategy file.

    Returns:
        Dict, whose key is parameter name and value is slice strategy of this parameter.

    Raises:
        ValueError: Strategy file is incorrect.
        TypeError: `strategy_filename` is not a string.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> from mindspore.parallel import build_searched_strategy
        >>> strategy = build_searched_strategy("./strategy_train.ckpt")
    """
    return _build_searched_strategy(strategy_filename)


# disable pylint too broad Exception
# pylint: disable=W0212
@_mstx_range_decorator("load_distributed_checkpoint", domain="model_preparation")
def load_distributed_checkpoint(network, checkpoint_filenames=None, predict_strategy=None,
                                train_strategy_filename=None, strict_load=False, dec_key=None, dec_mode='AES-GCM',
                                format='ckpt', unified_safetensors_dir=None, dst_safetensors_dir=None, rank_id=None,
                                output_format='safetensors', name_map=None, max_process_num=64,
                                return_param_dict=False):
    """
    Load checkpoint into net for distributed predication. Used in the case of distributed inference.

    Note:
        `output_format` will only take effect when `format` is set to `safetensors` and `network` is set to `None`.

    Args:
        network (Cell): Network for distributed predication, When the format is `safetensors`, the network parameter
                        can be left blank or passed as None, and the interface will execute save mode.
        checkpoint_filenames (list[str]): The name of Checkpoint files in order of rank id. Default: ``None`` .
        predict_strategy (Union[dict, str]): Strategy of predication process. It means that using one device to predict
                                 when setting predict_strategy as None. Default: ``None`` .
        train_strategy_filename (str): The filename of training strategy protocol buffer file.
                                       When train_strategy_filename is None, the training strategy file will be
                                       obtained from context.get_auto_parallel_context("strategy_ckpt_load_file").
                                       Therefore, the training strategy file needs to be specified
                                       in at least one of them. Default: ``None`` .
        strict_load (bool): Whether to strict load the parameter into net. If ``False`` , it will load parameter
                            into net when parameter name's suffix in checkpoint file is the same as the
                            parameter in the network. When the types are inconsistent, perform type conversion
                            on the parameters of the same type, such as float32 to float16. Default: ``False`` .
        dec_key (Union[None, bytes]): Byte type key used for decryption. If the value is ``None`` , the decryption
                                      is not required. Default: ``None`` .
        dec_mode (str): Specifies the decryption
                        mode, currently supports ``'AES-GCM'`` , ``'AES-CBC'``  and ``'SM4-CBC'`` .
                        This parameter is valid only when dec_key is not set to ``None`` .
                        Default: ``'AES-GCM'`` .
        format (str): Input weight format to be loaded into the network.
                      It can be set to either "ckpt" or "safetensors". Default: ``"ckpt"``.
        unified_safetensors_dir (str): Directory of input weight files to be loaded into the network.
                                       Default: ``None`` .
        dst_safetensors_dir (str): In the save mode scenario, the save directory for weights.
        rank_id (int): The logical sequence number of the card. In non save mode, it is automatically obtained
                       globally by initializing the network; In save mode, save the file according to the input
                       sequence number. If it is not input, save the entire file.
        output_format (str, optional): Control the format of the output checkpoint after conversion.
            It can be set to either "ckpt" or "safetensors". Default: ``"safetensors"``.
        name_map (dict): The weight mapping dictionary will modify the weight names according to the mapping
            dictionary before loading or saving the segmented weights into the network. Default: ``None``.
        max_process_num (int): Maximum number of processes. Default: ``64``.
        return_param_dict (bool): Whether to return the param_dict. Default: ``False``.

    Raises:
        TypeError: The type of inputs do not match the requirements.
        ValueError: Failed to load checkpoint into net.

    Supported Platforms:
        ``Ascend``

    Examples:
        .. note::
            Before running the following examples, you need to configure the communication environment variables.

            For Ascend/GPU/CPU devices, it is recommended to use the msrun startup method
            without any third-party or configuration file dependencies.
            Please see the `msrun startup
            <https://www.mindspore.cn/tutorials/en/master/parallel/msrun_launcher.html>`_
            for more details.

        >>> import os
        >>> import numpy as np
        >>> import mindspore as ms
        >>> import mindspore.dataset as ds
        >>> from mindspore import nn, ops, train
        >>> from mindspore.communication import init
        >>> from mindspore.parallel import load_distributed_checkpoint
        >>> from mindspore.parallel.auto_parallel import AutoParallel
        >>> from mindspore.nn.utils import no_init_parameters
        >>> from mindspore.common.initializer import initializer, One
        >>> from mindspore.communication.management import get_group_size
        >>>
        >>> step_per_epoch = 4
        >>> device_num = get_group_size()
        >>>
        >>> # Define the network structure.
        >>> class Net(nn.Cell):
        ...     def __init__(self, matmul_size, strategy=None):
        ...         super().__init__()
        ...         self.matmul_weight = ms.Parameter(initializer(One(), matmul_size, ms.float32))
        ...         self.matmul = ops.MatMul()
        ...         self.neg = ops.Neg()
        ...         if strategy is not None:
        ...             self.matmul.shard(strategy)
        ...
        ...     def construct(self, inputs):
        ...         x = self.matmul(inputs, self.matmul_weight)
        ...         x = self.neg(x)
        ...         return x
        >>>
        >>> # Create dataset.
        >>> def get_dataset(*inputs):
        ...     def generate():
        ...         for _ in range(step_per_epoch):
        ...             yield inputs
        ...     return generate
        >>>
        >>> # Train network and save distributed checkpoint.
        >>> def train_net():
        ...     ms.set_context(mode=ms.GRAPH_MODE)
        ...     init()
        ...     np.random.seed(1)
        ...     input_data = np.random.rand(16, 96).astype(np.float32)
        ...     label_data = np.random.rand(16, 16).astype(np.float32)
        ...     fake_dataset = get_dataset(input_data, label_data)
        ...     dataset = ds.GeneratorDataset(fake_dataset, ["input", "label"])
        ...
        ...     # Set parallel strategy.
        ...     strategy = ((1, 4), (4, 1))
        ...     with no_init_parameters():
        ...         network = Net(matmul_size=(96, 16), strategy=strategy)
        ...         net_opt = nn.Momentum(network.trainable_params(), 0.01, 0.9)
        ...
        ...     net_loss = nn.SoftmaxCrossEntropyWithLogits(reduction="mean")
        ...     network = AutoParallel(network, parallel_mode="semi_auto")
        ...     network.save_param_strategy_file(file_path="./train_strategy.ckpt")
        ...     model = ms.Model(network=network, loss_fn=net_loss, optimizer=net_opt)
        ...     ckpt_config = train.CheckpointConfig(keep_checkpoint_max=1, integrated_save=True)
        ...     global_rank_id = int(os.getenv("RANK_ID"))
        ...     ckpt_path = "./rank_{}_ckpt".format(global_rank_id)
        ...     ckpt_callback = train.ModelCheckpoint(prefix="parallel", directory=ckpt_path, config=ckpt_config)
        ...     model.train(epoch=2, train_dataset=dataset, callbacks=[ckpt_callback], dataset_sink_mode=False)
        >>>
        >>> # Load distributed checkpoint and test.
        >>> def load_model():
        ...     ms.set_context(mode=ms.GRAPH_MODE)
        ...     init()
        ...     predict_data = ms.Tensor(np.random.randn(128, 96).astype(np.float32))
        ...     with no_init_parameters():
        ...         network = Net(matmul_size=(96, 16))
        ...         network = AutoParallel(network, parallel_mode="semi_auto")
        ...     network.dataset_strategy(config="full_batch")
        ...     train_strategy_file = "./train_strategy.ckpt"
        ...     network.save_param_strategy_file(file_path=train_strategy_file)
        ...     model = ms.Model(network)
        ...     predict_layout = model.infer_predict_layout(ms.Tensor(predict_data))
        ...     ckpt_file_list = ["./rank_{}_ckpt/parallel-2_4.ckpt".format(i) for i in range(0, device_num)]
        ...     load_distributed_checkpoint(network, ckpt_file_list, predict_layout, None)
        ...     predict_result = model.predict(predict_data)
        ...     print(predict_result)
        >>>
        >>> train_net()
        >>> load_model()
        [[-9.62929535e+00, -9.76258755e+00, -9.70192051e+00 ... -9.67151260e+00, -9.71998310e+00, -9.64571190e+00],
        [-4.63218540e-01, -4.07317460e-01, -3.78161550e-01 ... -3.95918339e-01, -2.87363172e-01, -3.48693460e-01],
        ...
        [-4.28075647e+00, -4.36630344e+00, -4.25664043e+00 ... -4.32012939e+00, -4.30337954e+00, -4.27571440e+00]]
    """
    if format not in ['safetensors', 'ckpt'] or output_format not in ['safetensors', 'ckpt']:
        raise ValueError(
            f"For 'load_distributed_checkpoint', 'format' and 'output_format' "
            f"must be 'ckpt' or 'safetensors', but got {format}.")

    if format == 'safetensors':
        if unified_safetensors_dir is None:
            raise ValueError(f"For 'load_distributed_checkpoint', 'unified_safetensors_dir' can not be None "
                             f"when format is 'safetensors'.")
        unsupport_param = [checkpoint_filenames, train_strategy_filename, dec_key]
        for param in unsupport_param:
            if param is not None:
                raise ValueError(f"For 'load_distributed_checkpoint', {param} must be None "
                                 f"when format is 'safetensors'.")
        if strict_load or dec_mode != 'AES-GCM':
            raise ValueError(f"For 'load_distributed_checkpoint', strict_load and dec_mode must be default "
                             f"when format is 'safetensors'.")
        if network is not None:
            try:
                rank_id = get_rank()
            except RuntimeError:
                rank_id = 0
                logger.warning(f"Get rank failed, default loading weight for rank 0.")
            param_dict = _load_parallel_checkpoint(
                (unified_safetensors_dir, predict_strategy, network, None, rank_id, output_format, name_map,
                 return_param_dict))
            return param_dict
        if dst_safetensors_dir is None:
            raise ValueError(f"For 'load_distributed_checkpoint', 'dst_safetensors_dir' can not be None "
                             f"when network is None.")
        if rank_id is not None:
            _load_parallel_checkpoint(
                (unified_safetensors_dir, predict_strategy, network, dst_safetensors_dir,
                 rank_id, output_format, name_map, return_param_dict))
        else:
            dst_strategy_dict = _build_searched_strategy(predict_strategy)
            dst_stage_device_num = _get_device_num_from_strategy(dst_strategy_dict)
            dst_stage_num = _extract_pipeline_stage_num(dst_strategy_dict)
            dst_device_num = dst_stage_device_num * dst_stage_num
            tasks = _gather_tasks_load_dis(unified_safetensors_dir, predict_strategy, network, dst_safetensors_dir,
                                           dst_device_num, output_format, name_map, return_param_dict)
            with Pool(processes=max_process_num) as pool:
                list(pool.imap(_load_parallel_checkpoint, tasks))
        return True

    network = Validator.check_isinstance("network", network, ms.nn.Cell)
    _check_checkpoint_file(checkpoint_filenames)
    _check_predict_strategy(predict_strategy)

    dec_key = Validator.check_isinstance('dec_key', dec_key, (type(None), bytes))
    dec_mode = Validator.check_isinstance('dec_mode', dec_mode, str)

    if train_strategy_filename is None:
        parallel_net = _get_auto_parallel_net(network)
        if parallel_net.__class__.__name__ == "AutoParallel":
            train_strategy_filename = parallel_net._save_strategy_file_path
        else:
            train_strategy_filename = ms.context.get_auto_parallel_context("strategy_ckpt_load_file")

    _train_strategy = build_searched_strategy(train_strategy_filename)
    if not _train_strategy:
        return True
    train_strategy = _convert_to_list(_train_strategy)

    train_dev_count = 1
    ckpt_file_len = len(checkpoint_filenames)
    for dim in train_strategy[list(train_strategy.keys())[0]][0]:
        train_dev_count *= dim
    if train_dev_count != ckpt_file_len:
        raise ValueError(f"For 'Load_distributed_checkpoint', the length of 'checkpoint_filenames' should be "
                         f"equal to the device count of training process. "
                         f"But got the length of 'checkpoint_filenames'"
                         f" is {ckpt_file_len} and the device count is {train_dev_count}.")
    rank_list = _infer_rank_list(train_strategy, predict_strategy)

    param_total_dict = defaultdict(dict)
    for file_index, file_name in enumerate(checkpoint_filenames):
        file_name = os.path.abspath(file_name)
        ckpt_dict = ms.load_checkpoint(file_name, dec_key=dec_key, dec_mode=dec_mode)
        for param_name, param in ckpt_dict.items():
            param_total_dict[param_name][file_index] = param

    param_dict = {}
    param_not_in_strategy = []
    param_not_in_ckpt = []
    for _, param in network.parameters_and_names():
        sliced_params = []
        if param.name not in rank_list:
            param_not_in_strategy.append(param.name)
            continue
        if param.name not in param_total_dict:
            param_not_in_ckpt.append(param.name)
            continue

        param_rank = rank_list.get(param.name)[0]
        skip_merge_split = rank_list.get(param.name)[1]
        shard_stride = train_strategy.get(param.name)[4]
        tensor_map = train_strategy.get(param.name)[1]
        first_dim_shard_idx = tensor_map[0] if tensor_map else -1
        device_arrangement = train_strategy.get(param.name)[0]
        first_dim_shard_size = 1
        if first_dim_shard_idx >= 0:
            first_dim_shard_size = device_arrangement[-1 - first_dim_shard_idx]
        if train_strategy.get(param.name)[5]:
            repeat_size = int(ckpt_file_len / shard_stride / train_strategy.get(param.name)[5] / first_dim_shard_size)
        else:
            repeat_size = 0
        for rank in param_rank:
            param_total_list = list(range(0, ckpt_file_len))
            if first_dim_shard_size != 1:
                param_total_list = _get_param_list_when_first_dim_sharded(device_arrangement, first_dim_shard_idx, rank)
            if repeat_size > 0:
                shard_size = shard_stride * train_strategy.get(param.name)[5]
                rank_index = param_total_list.index(rank)
                start = rank_index // shard_size * shard_size
                param_total_list = param_total_list[start:start + shard_size]
            if shard_stride > 0:
                param_stride = []
                # merge pre parameter
                param_index = param_total_list[0:param_total_list.index(rank) + 1][::-1][::shard_stride]
                param_index.extend(param_total_list[param_total_list.index(rank):][::shard_stride])
                param_index = list(set(param_index))
                param_index.sort()
                for rank_num in param_index:
                    if param_total_dict[param.name][rank_num].data.dtype == mstype.bfloat16:
                        from mindspore.ops import Cast
                        cpu_cast = Cast().set_device("CPU")
                        param_stride.append(
                            cpu_cast(param_total_dict[param.name][rank_num].data, mstype.float32).asnumpy())
                    else:
                        param_stride.append(param_total_dict[param.name][rank_num].data.asnumpy())

                sliced_param = Parameter(Tensor(np.concatenate(param_stride)), name=param.name)
            else:
                sliced_param = param_total_dict[param.name][rank]

            sliced_params.append(sliced_param)
        if skip_merge_split:
            split_param = sliced_params[0]
        else:
            param_unique_strategy = _remove_repeated_slices(train_strategy[param.name])
            _param_unique_strategy = _convert_to_layout(param.name, param_unique_strategy)
            split_param = _merge_and_split(sliced_params, _param_unique_strategy, predict_strategy)
        opt_shard_group = predict_strategy[param.name][5] if predict_strategy else None
        if opt_shard_group:
            if split_param.data.dtype == mstype.bfloat16:
                from mindspore.ops import Cast
                cpu_cast = Cast().set_device("CPU")
                data = cpu_cast(split_param.data, mstype.float32).asnumpy()
            else:
                data = split_param.data.asnumpy()
            rank = get_rank(opt_shard_group)
            size = get_group_size(opt_shard_group)
            try:
                data_slice = np.split(data, size)[rank]
            except BaseException as e:
                logger.critical("Failed to load opt shard slice in load distributed checkpoint for {}. Data shape is {}"
                                " and group is {}".format(param.name, split_param.data.shape, opt_shard_group))
                raise RuntimeError(e.__str__() + f"\nFor 'load_distributed_checkpoint', failed to load opt shard slice"
                                                 f" in load distributed checkpoint for {param.name}. Data shape is "
                                                 f"{split_param.data.shape} and group is {opt_shard_group}.") from e
            split_param = Parameter(Tensor(data_slice), param.name,
                                    split_param.requires_grad, split_param.layerwise_parallel)
        param_dict[param.name] = split_param

    if param_not_in_strategy:
        logger.warning("For 'load_distributed_checkpoint', {} parameters in network are not in the slice strategy, "
                       "you can check whether 'predict_strategy' or 'train_strategy_filename' is correct."
                       .format(param_not_in_strategy))
    if param_not_in_ckpt:
        logger.warning("For 'load_distributed_checkpoint', {} parameters in network and slice strategy but not in "
                       "the checkpoint file, please check whether 'checkpoint_filenames' is correct."
                       .format(param_not_in_ckpt))

    ms.load_param_into_net(network, param_dict, strict_load=strict_load)
    return True


def restore_group_info_list(group_info_file_name):
    """
    Extract rank list information from communication domain files. To save the group info file,
    please export GROUP_INFO_FIL
    environment variables like "export GROUP_INFO_FILE=/data/group_info.pb".

    Args:
        group_info_file_name (str): Name of group information file.

    Returns:
        List, the rank list.

    Raises:
        ValueError: group information file is incorrect.
        TypeError: `group_info_file_name` is not str.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> import mindspore as ms
        >>> from mindspore.parallel import restore_group_info_list
        >>> ms.restore_list = restore_group_info_list("./group_info.pb")
    """
    if not isinstance(group_info_file_name, str):
        raise TypeError(f"For 'restore_group_info_list', the argument 'group_info_file_name' should be str, "
                        f"but got {type(group_info_file_name)}.")

    if not os.path.isfile(group_info_file_name):
        raise ValueError(f"For 'restore_group_info_list', no such group information file: {group_info_file_name}.")

    if os.path.getsize(group_info_file_name) == 0:
        raise ValueError("For 'restore_group_info_list', the group information file should not be empty.")

    return _restore_group_info_list(group_info_file_name)
