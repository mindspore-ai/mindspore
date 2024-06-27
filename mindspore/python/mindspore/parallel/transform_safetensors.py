# Copyright 2024 Huawei Technologies Co., Ltd
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
"""Transform distributed safetensors"""
from __future__ import absolute_import

import os
import glob
import re
from collections import defaultdict

import multiprocessing as mp
import numpy as np
import mindspore as ms
from mindspore.parallel._parallel_serialization import _get_device_num_from_strategy, _make_dir, \
    _extract_layout_map, _extract_src_dst_layout_map, _parameter_not_in_local_stage, _extract_pipeline_stage_num, \
    _insert_opt_shard_reshape, _extract_src_dst_layout_map_by_src
from mindspore.parallel._tensor import _get_tensor_strategy, _construct_from_to_tensor_layout, \
    _get_needed_rank_transform_operator_map_by_layouts, \
    _generate_transform_operator_stack, _apply_tensor_transform_operators, _construct_tensor_layout_for_opt_shard, \
    _extract_layout_item
from mindspore.parallel._parallel_serialization import _build_searched_strategy, _load_protobuf_strategy

from tqdm import tqdm
from safetensors.numpy import save_file, load_file
from safetensors import safe_open


def _load_and_transform(path, name_map, load_func, transform_func):
    if load_func is not None:
        param_dict = load_func(path)
    else:
        param_dict = path
    transform_dict = {}
    for k, v in param_dict.items():
        new_name = name_map.get(k, k) if name_map is not None else k
        transform_dict[new_name] = transform_func(v, new_name)
    return transform_dict


def _transform_tensor_to_numpy(path, name_map=None):
    return _load_and_transform(path, name_map, ms.load_checkpoint, lambda v, new_name: v.asnumpy())


def _transform_numpy_to_tensor(path, name_map=None):
    return _load_and_transform(path, name_map, load_file, lambda v, new_name: ms.Parameter(v, name=new_name))


def _process_file(file_info):
    cur_ckpt_path, name_map, save_path, file = file_info
    param_dict_numpy = _transform_tensor_to_numpy(cur_ckpt_path, name_map)
    safetensors_filename = file.replace(".ckpt", ".safetensors")
    dst_file = os.path.join(save_path, safetensors_filename)
    save_file(param_dict_numpy, dst_file)


def _process_file_safetensors(file_info):
    cur_safe_path, name_map, save_path, file = file_info
    param_dict_tensor = _transform_numpy_to_tensor(cur_safe_path, name_map)
    ckpt_filename = file.replace(".safetensors", ".ckpt")
    dst_file = os.path.join(save_path, ckpt_filename)
    ms.save_checkpoint(param_dict_tensor, dst_file)


def _gather_tasks(file_path, save_path, file_name_regex, name_map):
    """gather transform rank together"""
    tasks = []
    for root, dirs, _ in os.walk(file_path):
        if root != file_path:
            continue

        rank_dirs = [d for d in dirs if d.startswith('rank')]
        if not rank_dirs:
            raise ValueError(
                f"For 'ckpt_to_safetensors', no directories starting with 'rank' found in {file_path}")

        for rank_dir in rank_dirs:
            rank_dir_path = os.path.join(root, rank_dir)
            dst_root = os.path.join(save_path,
                                    os.path.relpath(rank_dir_path, file_path)) if save_path else rank_dir_path
            os.makedirs(dst_root, exist_ok=True)
            tasks.extend(
                (os.path.join(rank_dir_path, file), name_map, dst_root, file)
                for file in os.listdir(rank_dir_path)
                if file.endswith(".ckpt") and (file_name_regex is None or re.findall(file_name_regex, file))
            )
    return tasks


def ckpt_to_safetensors(file_path, save_path=None, name_map=None, file_name_regex=None, processes_num=1):
    """
    Converts MindSpore checkpoint files into safetensors format and saves them to `save_path`.

    Note:
        The number of multiprocess settings is related to the size of the host, and it is not recommended to set it
        too large, otherwise it may cause freezing.

    Args:
        file_path (str): Path to the directory containing checkpoint files or a single checkpoint file (.ckpt).
        save_path (str, optional): Directory path where safetensors files will be saved. Defaults: ``None``.
        name_map (dict, optional): Dictionary mapping original parameter names to new names. Defaults: ``None``.
        file_name_regex (str, optional): Regular expression used to match the file that needs to be converted.
                                   Defaults: ``None``.
        processes_num (int, optional): Number of processes to use for parallel processing. Defaults: 1.
    Raises:
        ValueError: If the input path is invalid or the save_path is not a directory,
                    or the file_path does not end with '.ckpt'.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore as ms
        >>> ms.ckpt_to_safetensors("./ckpt_save_path")
        >>> ms.ckpt_to_safetensors("./ckpt_save_path/rank0/checkpoint_0.ckpt")
        >>> ms.ckpt_to_safetensors(file_path="./ckpt_save_path/rank0/checkpoint_0.ckpt", save_path="./new_path/")
        >>> namemap = {"lin.weight":"new_name"}
        >>> ms.ckpt_to_safetensors("./ckpt_save_path/rank0/checkpoint_0.ckpt", "./new_path/", namemap)
    """
    is_dir = os.path.isdir(file_path)
    is_file = os.path.isfile(file_path)
    if not is_dir and not is_file:
        raise ValueError(f"For 'ckpt_to_safetensors', the input path must be a valid path or file, but got {file_path}")
    if save_path and os.path.splitext(save_path)[1]:
        raise ValueError(f"For 'ckpt_to_safetensors', the save_path must be a directory, but got '{save_path}'")
    if name_map is not None and not isinstance(name_map, dict):
        raise ValueError(
            f"For 'ckpt_to_safetensors', the type of 'name_map' must be a directory, but got '{type(name_map)}'")

    if is_dir:
        tasks = _gather_tasks(file_path, save_path, file_name_regex, name_map)
        with mp.Pool(processes=processes_num) as pool:
            list(tqdm(pool.imap(_process_file, tasks), total=len(tasks)))
    elif is_file:
        if not file_path.endswith(".ckpt"):
            raise ValueError(f"For 'ckpt_to_safetensors', the input file must be a .ckpt file, but got {file_path}")
        if file_name_regex is not None and not re.findall(file_name_regex, file_path):
            raise ValueError(f"For 'ckpt_to_safetensors', the input file does not match the regular expression.")
        if save_path and not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)

        param_dict_numpy = _transform_tensor_to_numpy(file_path, name_map)
        safetensors_filename = os.path.basename(file_path).replace(".ckpt", ".safetensors")
        dst_file = os.path.join(save_path if save_path else os.path.dirname(file_path), safetensors_filename)
        save_file(param_dict_numpy, dst_file)


def _gather_safetensors_tasks(file_path, save_path, file_name_regex, name_map):
    """gather transform rank together"""
    tasks = []
    for root, dirs, _ in os.walk(file_path):
        if root != file_path:
            continue

        rank_dirs = [d for d in dirs if d.startswith('rank')]
        if not rank_dirs:
            raise ValueError(
                f"For 'safetensors_to_ckpt', no directories starting with 'rank' found in {file_path}")

        for rank_dir in rank_dirs:
            rank_dir_path = os.path.join(root, rank_dir)
            dst_root = os.path.join(save_path,
                                    os.path.relpath(rank_dir_path, file_path)) if save_path else rank_dir_path
            os.makedirs(dst_root, exist_ok=True)
            tasks.extend(
                (os.path.join(rank_dir_path, file), name_map, dst_root, file)
                for file in os.listdir(rank_dir_path)
                if file.endswith(".safetensors") and (file_name_regex is None or re.findall(file_name_regex, file))
            )
    return tasks


def safetensors_to_ckpt(file_path, save_path=None, name_map=None, file_name_regex=None, processes_num=1):
    """
    Converts safetensors files into MindSpore checkpoint format and saves them to `save_path`.

    Note:
        The number of multiprocess settings is related to the size of the host, and it is not recommended to set it
        too large, otherwise it may cause freezing.

    Args:
        file_path (str): Path to the directory containing safetensors files or a single safetensors file (.safetensors).
        save_path (str, optional): Directory path where checkpoint files will be saved. Defaults: ``None``.
        name_map (dict, optional): Dictionary mapping original parameter names to new names. Defaults: ``None``.
        file_name_regex (str, optional): Regular expression used to match the file that needs to be converted.
                                   Defaults: ``None``.
        processes_num (int, optional): Number of processes to use for parallel processing. Defaults: 1.

    Raises:
        ValueError: If the input path is invalid, the save_path is not a directory,
                    or the file_path does not end with '.safetensors'.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore as ms
        >>> ms.safetensors_to_ckpt("./safetensors_save_path")
        >>> ms.safetensors_to_ckpt("./safetensors_save_path/rank0/checkpoint_0.safetensors")
        >>> ms.safetensors_to_ckpt("./safetensors_save_path/rank0/checkpoint_0.safetensors", "./new_path/")
        >>> namemap = {"lin.weight":"new_name"}
        >>> ms.safetensors_to_ckpt("./safetensors_save_path/rank0/checkpoint_0.safetensors", "./new_path/", namemap)
    """
    is_dir = os.path.isdir(file_path)
    is_file = os.path.isfile(file_path)
    if not is_dir and not is_file:
        raise ValueError(f"For 'safetensors_to_ckpt', the input path must be a valid path or file, but got {file_path}")
    if save_path and os.path.splitext(save_path)[1]:
        raise ValueError(f"For 'safetensors_to_ckpt', the save_path must be a directory, but got '{save_path}'")
    if name_map is not None and not isinstance(name_map, dict):
        raise ValueError(
            f"For 'safetensors_to_ckpt', the type of 'name_map' must be a directory, but got '{type(name_map)}'")

    if is_dir:
        tasks = _gather_safetensors_tasks(file_path, save_path, file_name_regex, name_map)
        with mp.Pool(processes=processes_num) as pool:
            list(tqdm(pool.imap(_process_file_safetensors, tasks), total=len(tasks)))
    elif is_file:
        if not file_path.endswith(".safetensors"):
            raise ValueError(
                f"For 'safetensors_to_ckpt', the input file must be a .safetensors file, but got {file_path}")
        if file_name_regex is not None and not re.findall(file_name_regex, file_path):
            raise ValueError(f"For 'safetensors_to_ckpt', the input file does not match the regular expression.")
        if save_path and not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)

        param_dict_tensor = _transform_numpy_to_tensor(file_path, name_map)
        ckpt_filename = os.path.basename(file_path).replace(".safetensors", ".ckpt")
        dst_file = os.path.join(save_path if save_path else os.path.dirname(file_path), ckpt_filename)
        ms.save_checkpoint(param_dict_tensor, dst_file)


def _check_transform_safetensors(src_safetensors_dir, ckpt_prefix, src_strategy_file, dst_strategy_file):
    """check _transform_safetensors input"""
    if not os.path.isdir(src_safetensors_dir):
        raise NotADirectoryError("src_safetensors_dir {} is not a directory.".format(src_safetensors_dir))
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


def _check_output_format(output_format):
    if output_format not in ["safetensors", "ckpt"]:
        raise ValueError(f"For 'transform_safetensors', the output_format must be "
                         f"'safetensors' or 'ckpt', but got {output_format}.")


def _split_protobuf_strategy(merged_strategy_file):
    """split src_strategy_file by pp"""
    dst_parallel_strategy_map = _load_protobuf_strategy(merged_strategy_file)
    if not dst_parallel_strategy_map.parallel_strategy_item or not dst_parallel_strategy_map.parallel_layout_item:
        raise ValueError(f"The merged strategy file {merged_strategy_file} is empty")

    src_dict = {}
    for layout_item in dst_parallel_strategy_map.parallel_layout_item:
        stage, _ = layout_item.param_name.split('-', 1)
        stage = int(stage)
        if stage not in src_dict:
            src_dict[stage] = {}
        parameter_name = layout_item.param_name
        layout = layout_item.parallel_layouts
        src_dict[stage][parameter_name] = layout
    return src_dict


def _transform_safetensors(src_safetensors_dir, dst_safetensors_dir, ckpt_prefix, src_strategy_file=None,
                           dst_strategy_file=None, process_num=1, output_format="safetensors"):
    """Transform distributed safetensors from source sharding strategy to destination sharding strategy for a rank."""
    _check_transform_safetensors(src_safetensors_dir, ckpt_prefix, src_strategy_file, dst_strategy_file)
    _check_output_format(output_format)
    _make_dir(dst_safetensors_dir, "path")
    all_safetensor_files_map = _collect_safetensor_files(src_safetensors_dir)

    dst_strategy_dict = _build_searched_strategy(dst_strategy_file)
    pipeline_stage_num = _extract_pipeline_stage_num(src_strategy_file)
    dst_stage_num = _extract_pipeline_stage_num(dst_strategy_file)

    if pipeline_stage_num > 1 and dst_stage_num == 1:
        stage_dict = _split_protobuf_strategy(src_strategy_file)

        processes = []
        manager = mp.Manager()
        transform_param_dict_with_name = manager.dict()
        for _, src_strategy_dict in stage_dict.items():
            p = mp.Process(target=_transform_stage_safetensors,
                           args=(src_strategy_dict, dst_strategy_dict, ckpt_prefix,
                                 dst_safetensors_dir, output_format, all_safetensor_files_map, process_num,
                                 transform_param_dict_with_name))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()

        _save_final_safetensors(transform_param_dict_with_name, output_format)
    else:
        src_strategy_dict = _build_searched_strategy(src_strategy_file)
        _transform_stage_safetensors(src_strategy_dict, dst_strategy_dict, ckpt_prefix,
                                     dst_safetensors_dir, output_format, all_safetensor_files_map, process_num,
                                     transform_param_dict_with_name=None)


def _transform_stage_safetensors(src_strategy_dict, dst_strategy_dict, ckpt_prefix,
                                 dst_safetensors_dir, output_format, all_safetensor_files_map, process_num,
                                 transform_param_dict_with_name):
    """Transform distributed safetensors by stage"""
    src_stage_device_num = _get_device_num_from_strategy(src_strategy_dict)
    dst_stage_device_num = _get_device_num_from_strategy(dst_strategy_dict)

    origin_src_strategy_list = _extract_layout_map(src_strategy_dict)
    origin_dst_strategy_list = _extract_layout_map(dst_strategy_dict)

    needed_rank_list_map = _find_needed_ranks(src_strategy_dict, dst_strategy_dict)
    for needed_rank_list, rank in needed_rank_list_map.items():
        for needed_rank in needed_rank_list.split("-"):
            if int(needed_rank) not in all_safetensor_files_map:
                raise ValueError("The safetensor file of rank{} is needed for converting rank{}'s safetensor, "
                                 "but it is missing.".format(needed_rank, rank))
    if process_num > len(needed_rank_list_map):
        ms.log.warning("The value of process_num cannot be greater than that of needed_rank_list_map.")
        process_num = len(needed_rank_list_map)
    if process_num <= 1:
        _transform_safetensors_single(needed_rank_list_map, all_safetensor_files_map, src_stage_device_num,
                                      dst_stage_device_num, src_strategy_dict, dst_strategy_dict,
                                      origin_src_strategy_list,
                                      origin_dst_strategy_list, ckpt_prefix, dst_safetensors_dir, output_format,
                                      transform_param_dict_with_name)
    else:
        _transform_safetensors_with_parallel(needed_rank_list_map, all_safetensor_files_map, src_stage_device_num,
                                             dst_stage_device_num, src_strategy_dict, dst_strategy_dict,
                                             origin_src_strategy_list, origin_dst_strategy_list, ckpt_prefix,
                                             dst_safetensors_dir, process_num, output_format,
                                             transform_param_dict_with_name)


def _distribute_files_by_size(all_safetensor_files_map, needed_rank_list_map, process_num):
    """
    Distributes files across multiple processes based on file size to balance the processing load.
    """
    # Calculate the size of each file.
    rank_size = dict()
    for rank_id, file_name in all_safetensor_files_map.items():
        tmp_size = os.path.getsize(file_name) / 1024 / 1024
        rank_size[rank_id] = tmp_size
    # Obtain the rank and size required by all parts.
    part_total = []
    for index, (k, v) in enumerate(needed_rank_list_map.items()):
        tmp_part = []
        key_ele = k.split("-")
        tmp_size = 0
        for ele in key_ele:
            tmp_size += rank_size[ele]
        tmp_part.append(index)
        tmp_part.append(tmp_size)
        part_total.append(tmp_part)
    # Sort each part by size.
    part_total = sorted(part_total, key=lambda x: x[1], reverse=True)
    part_list = [[] for _ in range(process_num)]
    part_size = [[] for _ in range(process_num)]
    for [index, size] in part_total:
        min_sum = float('inf')
        min_idx = -1
        for ele in range(process_num):
            if sum(part_size[ele]) < min_sum:
                min_sum = sum(part_size[ele])
                min_idx = ele
        part_list[min_idx].append(index)
        part_size[min_idx].append(size)

    part_list_dict = [dict() for _ in range(process_num)]
    for index, (k, v) in enumerate(needed_rank_list_map.items()):
        for idd, ele in enumerate(part_list):
            if index in ele:
                part_list_dict[idd][k] = v
                break
    return part_list_dict


def _transform_safetensors_with_parallel(needed_rank_list_map, all_safetensor_files_map, src_stage_device_num,
                                         dst_stage_device_num, src_strategy_dict, dst_strategy_dict,
                                         origin_src_strategy_list, origin_dst_strategy_list, ckpt_prefix,
                                         dst_safetensors_dir, process_num, output_format,
                                         transform_param_dict_with_name):
    """
    Transforms safetensors files to a specified format using parallel processing.
    """
    part_list_dict = _distribute_files_by_size(all_safetensor_files_map, needed_rank_list_map, process_num)
    processes = []
    for i in range(process_num):
        p = mp.Process(target=_transform_safetensors_single, args=(
            part_list_dict[i], all_safetensor_files_map, src_stage_device_num, dst_stage_device_num,
            src_strategy_dict, dst_strategy_dict, origin_src_strategy_list, origin_dst_strategy_list,
            ckpt_prefix, dst_safetensors_dir, output_format, transform_param_dict_with_name))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()


def _transform_safetensors_single(needed_rank_list_map, all_safetensor_files_map, src_stage_device_num,
                                  dst_stage_device_num,
                                  src_strategy_dict, dst_strategy_dict, origin_src_strategy_list,
                                  origin_dst_strategy_list,
                                  ckpt_prefix, dst_safetensors_dir, output_format, transform_param_dict_with_name):
    """
    Transforms safetensors files to a specified format without using parallel processing.
    """
    for needed_rank_list_key, transform_rank_list in needed_rank_list_map.items():
        param_total_dict = defaultdict(dict)
        param_attr_dict = defaultdict(dict)
        needed_rank_list = needed_rank_list_key.split("-")
        for needed_rank in needed_rank_list:
            print(f"loading safetensors from rank {needed_rank}...")
            saftensor_dict = load_file(all_safetensor_files_map.get(int(needed_rank)))
            for param_name, param in saftensor_dict.items():
                src_rank = int(needed_rank) % src_stage_device_num
                param_total_dict[param_name][src_rank] = param
                param_attr_dict[param_name][src_rank] = (True, False)

        for transform_rank in transform_rank_list:
            param_total_dict_keys = list(param_total_dict.keys())
            src_strategy_list, dst_strategy_list = _extract_src_dst_layout_map(transform_rank, src_strategy_dict,
                                                                               dst_strategy_dict)
            # cut the parameter not in the pipeline stage.
            for param in list(param_total_dict.keys()):
                if _parameter_not_in_local_stage(param, origin_src_strategy_list, src_strategy_list) \
                        and _parameter_not_in_local_stage(param, origin_dst_strategy_list, dst_strategy_list):
                    param_total_dict_keys.remove(param)

            local_rank_id = transform_rank % dst_stage_device_num
            transform_param_dict = _transform_parallel_safetensor(local_rank_id, param_total_dict,
                                                                  param_attr_dict, src_strategy_list, dst_strategy_list,
                                                                  param_total_dict_keys)
            save_safetensor_file = f"{ckpt_prefix}{transform_rank}.{output_format}"
            save_safetensor_file_dir = os.path.join(dst_safetensors_dir, "rank_{}".format(transform_rank))
            if not os.path.exists(save_safetensor_file_dir):
                _make_dir(save_safetensor_file_dir, "path")
            save_file_name = os.path.join(save_safetensor_file_dir, save_safetensor_file)
            if transform_param_dict_with_name is not None:
                if save_file_name not in transform_param_dict_with_name:
                    transform_param_dict_with_name[save_file_name] = transform_param_dict
                else:
                    transform_param_dict_with_name[save_file_name].update(transform_param_dict)
            else:
                if output_format == "safetensors":
                    save_file(transform_param_dict, save_file_name)
                else:
                    transform_param_dict = _load_and_transform(transform_param_dict, None, None,
                                                               transform_func=lambda v, name: ms.Parameter(v,
                                                                                                           name=name))
                    ms.save_checkpoint(transform_param_dict, save_file_name)
            del param_total_dict_keys
        del param_total_dict


def _save_final_safetensors(transform_param_dict_with_name, output_format):
    for save_file_name, transform_param_dict in transform_param_dict_with_name.items():
        if output_format == "safetensors":
            save_file(transform_param_dict, save_file_name)
        else:
            transform_param_dict = _load_and_transform(transform_param_dict, None, None,
                                                       transform_func=lambda v, name: ms.Parameter(v, name=name))
            ms.save_checkpoint(transform_param_dict, save_file_name)


def transform_safetensors_by_stage(src_safetensors_dir, dst_safetensors_dir, ckpt_prefix,
                                   src_strategy_file,
                                   dst_strategy_file=None):
    """Transform safetensor for stage in src_strategy_file"""
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
    safetensor_files_map = {}
    src_rank_id_start = stage_id * src_stage_device_num
    for local_rank in range(src_stage_device_num):
        rank_id = src_rank_id_start + local_rank
        safetensor_file_name = os.path.join(src_safetensors_dir, "rank_{}".format(rank_id), "*.safetensors")
        rank_ckpts = glob.glob(safetensor_file_name)
        rank_ckpts.sort()
        for safetensor_file in rank_ckpts:
            if not os.path.isfile(safetensor_file):
                continue
            safetensor_files_map[rank_id] = safetensor_file
    for rank, local_file in safetensor_files_map.items():
        if not os.path.exists(local_file):
            raise ValueError("safetensor file {} in rank {} not exits: ".format(local_file, rank))
    for rank, file_name in safetensor_files_map.items():
        saftensor_dict = load_file(file_name)
        for param_name, param in saftensor_dict.items():
            # cut the parameter not in the pipeline stage.
            if _parameter_not_in_local_stage(param_name, origin_src_strategy_list, src_strategy_list) \
                    and _parameter_not_in_local_stage(param_name, origin_dst_strategy_list, dst_strategy_list):
                continue
            src_rank = rank % src_stage_device_num
            param_type_dict[param_name][src_rank] = str(param.data.dtype)
            param_total_dict[param_name][src_rank] = param
            param_attr_dict[param_name][src_rank] = (True, False)
    for local_rank_id in range(dst_stage_device_num):
        transform_param_dict = _transform_parallel_safetensor(local_rank_id, param_total_dict,
                                                              param_attr_dict, src_strategy_list, dst_strategy_list,
                                                              param_type_dict)
        save_safetensor_file = "{}{}_part{}.safetensors".format(ckpt_prefix, local_rank_id, stage_id)
        save_safetensor_file_dir = os.path.join(dst_safetensors_dir, "rank_{}".format(local_rank_id))
        if not os.path.exists(save_safetensor_file_dir):
            _make_dir(save_safetensor_file_dir, "path")
        save_safetensor_file_name = os.path.join(save_safetensor_file_dir, save_safetensor_file)
        save_file(transform_param_dict, save_safetensor_file_name)


def transform_safetensors_by_rank(rank_id, safetensor_files_map, save_safetensor_file_name,
                                  src_strategy_file=None, dst_strategy_file=None):
    """
    Transform distributed checkpoint from source sharding strategy to destination sharding strategy by rank.
    """
    if not isinstance(safetensor_files_map, dict):
        raise TypeError("The safetensor_files_map should be a dict.")
    if not isinstance(rank_id, int):
        raise TypeError("The rank_id should be a int.")
    if not isinstance(save_safetensor_file_name, str):
        raise TypeError("The save_safetensor_file_name should be a str.")
    if not save_safetensor_file_name.endswith(".safetensors"):
        raise ValueError(
            "The save_safetensor_file_name {} should end with .safetensors".format(save_safetensor_file_name))
    if dst_strategy_file and os.path.dirname(dst_strategy_file) and not os.path.exists(
            os.path.dirname(dst_strategy_file)):
        raise ValueError("The director of dst_strategy_file: {} is not exists.".
                         format(os.path.dirname(dst_strategy_file)))
    for rank, local_file in safetensor_files_map.items():
        if not os.path.exists(local_file):
            raise ValueError("safetensor file {} in rank {} not exits: ".format(local_file, rank))
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
    for rank, file_name in safetensor_files_map.items():
        saftensor_dict = load_file(file_name)
        for param_name, param in saftensor_dict.items():
            # cut the parameter not in the pipeline stage.
            if _parameter_not_in_local_stage(param_name, origin_src_strategy_list, src_strategy_list) \
                    and _parameter_not_in_local_stage(param_name, origin_dst_strategy_list, dst_strategy_list):
                continue
            src_rank = rank % src_stage_device_num
            param_type_dict[param_name][src_rank] = str(param.data.dtype)
            # if param.data.dtype == mstype.bfloat16:
            #     param.set_dtype(mstype.float32)
            param_total_dict[param_name][src_rank] = param
            param_attr_dict[param_name][src_rank] = (True, False)
    local_rank_id = rank_id % dst_stage_device_num
    transform_param_dict = _transform_parallel_safetensor(local_rank_id, param_total_dict,
                                                          param_attr_dict, src_strategy_list, dst_strategy_list,
                                                          param_type_dict)
    save_file(transform_param_dict, save_safetensor_file_name)


def _collect_safetensor_files(src_safetensors_dir, format='safetensors'):
    """
    Collects all safetensors files from the specified directory and its subdirectories.
    """
    safetensors_rank_dir_list = os.path.join(src_safetensors_dir, "rank_[0-9]*")
    all_safetensor_files_map = {}
    for safetensor_dir in glob.glob(safetensors_rank_dir_list):
        if not os.path.isdir(safetensor_dir):
            ms.log.warning("{} is not a directory.".format(safetensor_dir))
            continue
        rank_id_str = safetensor_dir.split('rank_')[-1]
        if not rank_id_str.isdigit():
            ms.log.warning("{} is not a expected directory, the directory should end with rank_0/rank_1.....".
                           format(safetensor_dir))
            continue
        rank_id = int(rank_id_str)
        safetensor_file_name = os.path.join(safetensor_dir, f"*.{format}")
        rank_ckpts = glob.glob(safetensor_file_name)
        rank_ckpts.sort()
        for safetensor_file in rank_ckpts:
            if not os.path.isfile(safetensor_file):
                ms.log.warning("{} is not a safetensor file.".format(safetensor_file))
                continue
            all_safetensor_files_map[rank_id] = safetensor_file
    return all_safetensor_files_map


def _find_needed_ranks(src_strategy_dict, dst_strategy_dict):
    """
    Identifies the ranks needed for transformation based on source and destination strategies.
    """
    needed_rank_list_map = defaultdict(list)
    dst_stage_device_num = _get_device_num_from_strategy(dst_strategy_dict)
    dst_stage_num = _extract_pipeline_stage_num(dst_strategy_dict)
    dst_device_num = dst_stage_device_num * dst_stage_num
    for rank in tqdm(range(dst_device_num)):
        needed_rank_list = ms.rank_list_for_transform(rank, src_strategy_dict, dst_strategy_dict)
        needed_rank_list_key = "-".join([str(r) for r in needed_rank_list])
        needed_rank_list_map[needed_rank_list_key].append(rank)
    return needed_rank_list_map


def load_file_by_param_name(filename, parme_name_list):
    result = {}
    with safe_open(filename, framework="np") as f:
        for k in parme_name_list:
            result[k] = f.get_tensor(k)
    return result


def _transform_parallel_safetensor(rank_id, param_total_dict, param_attr_dict, src_strategy_list,
                                   dst_strategy_list, param_total_dict_keys=None):
    """
    Transform model parallel dimension for distributed safetensor files.
    """
    transform_param_dict = {}
    device_num = -1
    param_total_dict_keys = list(param_total_dict.keys()) if param_total_dict_keys is None else param_total_dict_keys
    for param_name in param_total_dict_keys:
        tensor_shape = list(param_total_dict[param_name].values())[0].shape
        from_dev_matrix = [1]
        from_tensor_map = [-1] * len(tensor_shape)
        from_opt_shard_step = 0
        from_opt_shard_size = 0
        if src_strategy_list is not None:
            if param_name not in src_strategy_list:
                continue
            from_dev_matrix, from_tensor_map, from_opt_shard_step, from_opt_shard_size = _extract_layout_item(
                src_strategy_list.get(param_name))
        to_dev_matrix_origin = [1]
        to_tensor_map_origin = [-1] * len(tensor_shape)
        to_opt_shard_step = 0
        to_opt_shard_size = 0
        if dst_strategy_list is not None:
            if param_name not in dst_strategy_list:
                continue
            to_dev_matrix_origin, to_tensor_map_origin, to_opt_shard_step, to_opt_shard_size = _extract_layout_item(
                dst_strategy_list.get(param_name))
        # Add optimizer sharding dim for tensor layout
        device_num = np.prod(from_dev_matrix)
        if device_num < 1:
            raise ValueError("None of the parameters in safetensor file are in either src strategy or "
                             "dst strategy. Please check correctness of strategy files. "
                             "Param name is: {}, rank_id is {}.".format(param_name, rank_id))
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

        # when the from_layout is less devices, the safetensor_map for map[device_num] should using map[0]
        device_list = list(range(0, np.prod(from_tensor_layout[0])))
        if rank_id % device_num not in param_attr_dict[param_name]:
            raise ValueError("The safetensor of rank {} is missing.".format(rank_id % device_num))
        param_rank_map = _get_needed_rank_transform_operator_map_by_layouts(from_tensor_layout, to_tensor_layout,
                                                                            device_list, rank_id)

        from_info_tuple = (from_opt_shard_size, from_dev_matrix, from_tensor_map, from_full_tensor_shape)
        to_info_tuple = (to_opt_shard_size, to_dev_matrix_origin, to_tensor_map_origin, origin_tensor_shape)
        _insert_opt_shard_reshape(param_rank_map, from_info_tuple, to_info_tuple)
        transform_operator_stack = _generate_transform_operator_stack(param_rank_map, rank_id)
        param_total_dict_copy = param_total_dict[param_name].copy()
        _apply_tensor_transform_operators(transform_operator_stack, param_total_dict_copy, device_num)

        transform_param_dict[param_name] = param_total_dict_copy[rank_id % device_num]

    # Handle those parameter like learning_rate, global_step which not in strategy_file.
    for param_name in param_total_dict_keys:
        if param_name not in transform_param_dict:
            transform_para = param_total_dict[param_name][rank_id % device_num]
            transform_param_dict[param_name] = transform_para
    return transform_param_dict


__all__ = ["_transform_safetensors", "transform_safetensors_by_stage",
           "transform_safetensors_by_rank", "ckpt_to_safetensors", "safetensors_to_ckpt"]
