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

import copy
import os
import sys
import glob
import math
import json
import re
import mmap
import stat
from collections import defaultdict, OrderedDict

import time
import multiprocessing as mp

from safetensors.numpy import save_file, load_file
import psutil
import numpy as np

import mindspore as ms
from mindspore import log as logger
from mindspore.log import vlog_print
from mindspore.common.parameter import Parameter
from mindspore.common.tensor import Tensor
from mindspore.parallel._parallel_serialization import _get_device_num_from_strategy, _make_dir, \
    _extract_layout_map, _extract_src_dst_layout_map, _parameter_not_in_local_stage, _extract_pipeline_stage_num, \
    _insert_opt_shard_reshape, _extract_src_dst_layout_map_by_src, _insert_expand_layout_reshape
from mindspore.parallel._tensor import _get_tensor_strategy, _construct_from_to_tensor_layout, \
    _get_needed_rank_transform_operator_map_by_layouts, \
    _generate_transform_operator_stack, _apply_tensor_transform_operators, _construct_tensor_layout_for_opt_shard, \
    _extract_layout_item, _apply_operator
from mindspore.parallel._parallel_serialization import _build_searched_strategy, _load_protobuf_strategy, \
    _convert_to_list
from mindspore.parallel._utils import _mstx_range_decorator
from mindspore.common import dtype as mstype

safetensors_to_mstype = {'Int4': mstype.qint4x2}

MAX_HEADER_SIZE = 100 * 1000 * 1000

dtype_size = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "I16": 2,
    "U16": 2,
    "I32": 4,
    "U32": 4,
    "I64": 8,
    "U64": 8,
    "F16": 2,
    "BF16": 2,
    "F32": 4,
    "F64": 8,
}
np_dtype_size = {
    "bool": 1,
    "bool_": 1,
    "uint8": 1,
    "int8": 1,
    "int16": 2,
    "uint16": 2,
    "int32": 4,
    "uint32": 4,
    "int64": 8,
    "uint64": 8,
    "float16": 2,
    "bfloat16": 2,
    "float32": 4,
    "float64": 8,
}
numpy_dtype = {
    "BOOL": np.bool_,
    "U8": np.uint8,
    "I8": np.int8,
    "I16": np.int16,
    "U16": np.uint16,
    "I32": np.int32,
    "U32": np.uint32,
    "I64": np.int64,
    "U64": np.uint64,
    "F16": np.float16,
    "F32": np.float32,
    "F64": np.float64,
}


def getSize(fileobject):
    fileobject.seek(0, 2)  # move the cursor to the end of the file
    size = fileobject.tell()
    fileobject.seek(0)  # move the cursor to the start of the file
    return size


def _save_file_atomically(transform_param_dict, save_file_name, metadata=None):
    """Atomically save file using temporary name and rename."""
    if metadata is None:
        metadata = {"format": "ms"}
    file_name_list = list(os.path.splitext(save_file_name))
    file_name_list[1] = file_name_list[1].replace('.safetensors', '.tmp')
    tmp_name = ''.join(file_name_list)
    try:
        if os.path.exists(save_file_name):
            os.chmod(save_file_name, stat.S_IWUSR)
            os.remove(save_file_name)
        if os.path.exists(tmp_name):
            os.chmod(tmp_name, stat.S_IWUSR)
            os.remove(tmp_name)
        save_file(transform_param_dict, tmp_name, metadata=metadata)
        os.rename(tmp_name, save_file_name)
        os.chmod(save_file_name, stat.S_IRUSR)
    except Exception as e:
        if not os.path.exists(save_file_name):
            logger.warning(f"Save failed, {save_file_name} not found. "
                           f"This may indicate multiple processes modifying the same file "
                           f"or insufficient disk space.")
        raise e


def metadata_validate(metadata):
    """validation metadata"""
    start = 0
    for key, info in metadata.items():
        s, e = info["data_offsets"]
        if s != start or e < s:
            raise ValueError(f"SafeTensorError::InvalidOffset({key})")
        start = e
        nelements = np.prod(info["shape"])
        nbytes = nelements * dtype_size[info["dtype"]]
        if (e - s) != nbytes:
            raise ValueError("SafeTensorError::TensorInvalidInfo")
    return start


def read_metadata(buffer):
    """read metadata by buffer"""
    buffer_len = getSize(buffer)
    if buffer_len < 8:
        raise ValueError("SafeTensorError::HeaderTooSmall")

    n = np.frombuffer(buffer.read(8), dtype=np.uint64).item()
    if n > MAX_HEADER_SIZE:
        raise ValueError("SafeTensorError::HeaderTooLarge")

    stop = n + 8
    if stop > buffer_len:
        raise ValueError("SafeTensorError::InvalidHeaderLength")

    tensors = json.loads(buffer.read(n), object_pairs_hook=OrderedDict)
    metadata = tensors.pop("__metadata__", None)
    buffer_end = metadata_validate(tensors)

    if buffer_end + 8 + n != buffer_len:
        raise ValueError("SafeTensorError::MetadataIncompleteBuffer")

    return stop, tensors, metadata


class PySafeSlice:
    """Create PySafeSlice by file"""

    def __init__(self, info, bufferfile, base_ptr, buffermmap):
        self.info = info
        self.bufferfile = bufferfile
        self.buffermmap = buffermmap
        self.base_ptr = base_ptr

        self.start = [0 for dim in self.shape]
        self.stop = [dim for dim in self.shape]
        self.step = [1 for dim in self.shape]

    @property
    def ndim(self):
        return len(self.shape)

    def get(self, *args, **kwargs):
        """Get tensor from buffer by data_offset"""
        nbytes = int(np.prod(self.shape)) * np.dtype(self.dtype).itemsize
        offset = self.start_offset
        tensor = np.frombuffer(self.buffermmap, dtype=self.dtype, offset=offset,
                               count=nbytes // np.dtype(self.dtype).itemsize)
        tensor = tensor.reshape(self.shape)
        if not tensor.flags["ALIGNED"]:
            logger.info("This safetensors file is not aligned.")
            tensor = tensor.copy()
        return tensor

    @property
    def start_offset(self):
        return self.base_ptr + self.info["data_offsets"][0]

    def get_shape(self):
        return self.shape

    @property
    def shape(self):
        return self.info["shape"]

    @property
    def dtype(self):
        """Get dtype by numpy_dtype"""
        if self.info["dtype"] == "BF16":
            from mindspore.common import np_dtype
            if not np_dtype.np_dtype_valid(True):
                raise TypeError(
                    "The Numpy bfloat16 data type is not supported now, please ensure that the current "
                    "Numpy version is not less than the version when the mindspore is compiled, "
                    "and the major versions are same."
                )
            return np_dtype.bfloat16
        return numpy_dtype[self.info["dtype"]]

    @property
    def nelements(self):
        return np.prod(self.info["shape"])

    @property
    def bits(self):
        return dtype_size[self.info["dtype"]]

    @property
    def nbytes(self):
        return self.nelements * dtype_size[self.info["dtype"]]


class _fast_safe_open:
    """
    Open a safetensors file and access its metadata and tensors efficiently.

    This function is designed to work similarly to `safetensors.safe_open`,
    providing a fast way to open and interact with safetensors files.
    """

    def __init__(self, filename, framework=None, device="cpu"):
        self.filename = filename
        self.framework = framework
        self.file = open(self.filename, "rb")
        self.file_mmap = mmap.mmap(self.file.fileno(), 0, access=mmap.ACCESS_COPY)
        try:
            self.base, self.tensors_decs, self.__metadata__ = read_metadata(self.file)
        except ValueError:
            raise ValueError(f"Fail to parse the input safetensors file: '{self.filename}'. "
                             f"Please check the correctness of the file.")
        self.tensors = OrderedDict()
        for key, info in self.tensors_decs.items():
            self.tensors[key] = PySafeSlice(info, self.file, self.base, self.file_mmap)
            self.tensors[key].key = key

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.file.close()

    def metadata(self):
        return self.__metadata__

    def keys(self):
        return list(self.tensors.keys())

    def get_tensor(self, name):
        return self.tensors[name].get()


def _fast_load_file(filename):
    """
    Load safetensors info from a specified file.
    """
    result = {}
    with _fast_safe_open(filename, framework="np") as f:
        for k in f.keys():
            result[k] = f.get_tensor(k)
    return result


def _progress_bar(iterable, total=None):
    """
    Decorate an iterable object, returning an iterator which acts exactly
    like the original iterable, but prints a dynamically updating
    progressbar every time a value is requested.
    """
    if total is None:
        total = len(iterable)

    start_time = time.time()

    def print_progress_bar(iteration):
        percent = f"{100 * (iteration / float(total)):.1f}"
        bar_length = 40
        filled_length = int(bar_length * iteration // total)
        bar = '█' * filled_length + '-' * (bar_length - filled_length)

        elapsed_time = time.time() - start_time
        estimated_total_time = elapsed_time / iteration * total
        remaining_time = estimated_total_time - elapsed_time

        elapsed_time_str = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
        remaining_time_str = time.strftime("%H:%M:%S", time.gmtime(remaining_time))

        sys.stdout.reconfigure(encoding="utf-8")
        print(f'\r{percent}%|{bar}|[{elapsed_time_str}<{remaining_time_str}]', end='')
        if iteration == total:
            print()

    for i, item in enumerate(iterable, start=1):
        yield item
        print_progress_bar(i)


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


def _check_transform_safetensors(src_safetensors_dir, ckpt_prefix, src_strategy_file, dst_strategy_file):
    """check _transform_safetensors input"""
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
        _transform_param_list = manager.list()
        for _, src_strategy_dict in stage_dict.items():
            p = mp.Process(target=_transform_stage_safetensors,
                           args=(src_strategy_dict, dst_strategy_dict, ckpt_prefix,
                                 dst_safetensors_dir, output_format, all_safetensor_files_map, process_num,
                                 _transform_param_list))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()

        _save_final_safetensors(_transform_param_list, output_format)
    else:
        src_strategy_dict = _build_searched_strategy(src_strategy_file)
        _transform_stage_safetensors(src_strategy_dict, dst_strategy_dict, ckpt_prefix,
                                     dst_safetensors_dir, output_format, all_safetensor_files_map, process_num,
                                     _transform_param_list=None)


def _transform_stage_safetensors(src_strategy_dict, dst_strategy_dict, ckpt_prefix,
                                 dst_safetensors_dir, output_format, all_safetensor_files_map, process_num,
                                 _transform_param_list):
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
    dst_stage_num = _extract_pipeline_stage_num(dst_strategy_dict)
    if not (len(needed_rank_list_map) == 1 and dst_stage_num > 1) and process_num > len(needed_rank_list_map):
        ms.log.warning("The value of process_num cannot be greater than that of needed_rank_list_map.")
        process_num = len(needed_rank_list_map)
    _transform_safetensors_with_parallel(needed_rank_list_map, all_safetensor_files_map, src_stage_device_num,
                                         dst_stage_device_num, src_strategy_dict, dst_strategy_dict,
                                         origin_src_strategy_list, origin_dst_strategy_list, ckpt_prefix,
                                         dst_safetensors_dir, process_num, output_format,
                                         _transform_param_list)


def _distribute_files_by_size(all_safetensor_files_map, needed_rank_list_map, process_num):
    """
    Distributes files across multiple processes based on file size to balance the processing load.
    """
    if process_num == 1:
        return [needed_rank_list_map]
    # Calculate the size of each file.
    # if src==1, dst pp>1, split for pp number.
    if len(needed_rank_list_map) == 1:
        src_rank = next(iter(needed_rank_list_map.keys()))
        dst_list = next(iter(needed_rank_list_map.values()))
        size = len(dst_list) // process_num
        split_list = [dst_list[i:i + size] for i in range(0, len(dst_list), size)]
        part_list_dict = [dict() for _ in range(process_num)]
        for index in range(process_num):
            part_list_dict[index][src_rank] = split_list[index]
        return part_list_dict

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
            tmp_size += rank_size[int(ele)]
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
                                         _transform_param_list):
    """
    Transforms safetensors files to a specified format using parallel processing.
    """
    # cal param name for every pipeline, save in pipe_param_list.
    pipe_num = _extract_pipeline_stage_num(dst_strategy_dict)
    pipe_param_list = [None for _ in range(max(pipe_num, process_num))]
    if len(needed_rank_list_map) == 1 and pipe_num > 1:
        process_num = pipe_num
        pipe_param_list = [[] for _ in range(pipe_num)]
        layout_map = _convert_to_list(dst_strategy_dict)

        for name, layout in layout_map.items():
            pipe_param_list[layout[6][0]].append(name)
    part_list_dict = _distribute_files_by_size(all_safetensor_files_map, needed_rank_list_map, process_num)
    processes = []
    if process_num > 1:
        for i in range(process_num):
            p = mp.Process(target=_transform_safetensors_single, args=(
                part_list_dict[i], all_safetensor_files_map, src_stage_device_num, dst_stage_device_num,
                src_strategy_dict, dst_strategy_dict, origin_src_strategy_list, origin_dst_strategy_list,
                ckpt_prefix, dst_safetensors_dir, output_format, _transform_param_list, pipe_param_list[i]))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
    else:
        _transform_safetensors_single(part_list_dict[0], all_safetensor_files_map, src_stage_device_num,
                                      dst_stage_device_num, src_strategy_dict, dst_strategy_dict,
                                      origin_src_strategy_list, origin_dst_strategy_list, ckpt_prefix,
                                      dst_safetensors_dir, output_format, _transform_param_list,
                                      pipe_param_list[0])


def _count_redundancy_list(rank_num, param_name, redundancy_dict, device_num):
    """Obtain the specified redundant group."""
    redundancy_tuple = redundancy_dict.get(param_name)
    for rank_list in redundancy_tuple:
        for rank in rank_list:
            if rank_num % device_num == rank % device_num:
                return set(rank_list)
    return set()


def _find_remove_redundancy_rank_id(pipe_param_list, single_param_dict, file_dict, safetensor_dict, redundancy_dict,
                                    needed_rank, device_num, choice_func):
    """Find the rank_id under redundant groups."""
    io_time = 0
    for param_name in pipe_param_list:
        rank_num = int(needed_rank)
        redundancy_ranks = _count_redundancy_list(rank_num, param_name, redundancy_dict, device_num)
        open_file_id = None
        if single_param_dict.get(param_name) is None:
            continue
        for real_rank in single_param_dict[param_name]:
            for redundancy_rank in redundancy_ranks:
                if real_rank % device_num == redundancy_rank % device_num:
                    open_file_id = real_rank
                    break
        if open_file_id is not None:
            start_time = time.time()
            output = file_dict[open_file_id].get_tensor(param_name)
            end_time = time.time()
            cost_time = end_time - start_time
            io_time += cost_time
            if choice_func is not None:
                choice_out = choice_func(param_name)
                if isinstance(choice_out, bool) and not choice_out:
                    continue
                if not isinstance(choice_out, (bool, str)):
                    raise ValueError("For 'unified_safetensors', the return value type of the function "
                                     f"'choice_func' must be bool or str, but got {type(choice_out)}.")
            safetensor_dict[param_name] = output
        else:
            raise ValueError(f"For _transform_safetensors_single, {param_name} should be in "
                             f"{redundancy_ranks}, but in {single_param_dict[param_name]}.")
    return io_time


def _transform_safetensors_single(needed_rank_list_map, all_safetensor_files_map, src_stage_device_num,
                                  dst_stage_device_num,
                                  src_strategy_dict, dst_strategy_dict, origin_src_strategy_list,
                                  origin_dst_strategy_list,
                                  ckpt_prefix, dst_safetensors_dir, output_format,
                                  _transform_param_list, pipe_param_list=None, file_index=None, unified_flag=False,
                                  src_strategy_file=None, choice_func=None):
    """
    Transforms safetensors files to a specified format without using parallel processing.
    """
    io_cost_time = 0
    meta_data = {"format": "ms"}
    if src_strategy_file is not None:
        from mindspore.train._utils import get_parameter_redundancy
        redundancy_dict_tmp = get_parameter_redundancy(src_strategy_file, initial_rank=0)
        redundancy_dict = {}
        device_num = 0
        for param_name, redundancy in redundancy_dict_tmp.items():
            if device_num == 0:
                device_num = max(max(redundancy)) + 1
            origin_param_name = param_name
            pipeline_stage = 0
            if "-" in param_name:
                pipeline_stage, origin_param_name = param_name.split("-")
                pipeline_stage = int(pipeline_stage)
            redundancy_new = tuple(
                (tuple(x + pipeline_stage * device_num for x in subtuple)) for subtuple in redundancy)
            redundancy_dict[origin_param_name] = redundancy_new
        file_dict = {}
        single_param_dict = {}
        for file_id, _ in all_safetensor_files_map.items():
            f = _fast_safe_open(all_safetensor_files_map.get(file_id), framework="np")
            file_dict[file_id] = f
            for param_name in f.keys():
                if param_name not in single_param_dict.keys():
                    single_param_dict[param_name] = {file_id}
                else:
                    single_param_dict[param_name].add(file_id)
            if f.metadata() is not None:
                meta_data.update(f.metadata())
    src_strategy_list_keys = _convert_to_list(src_strategy_dict).keys() if src_strategy_dict else []
    dst_strategy_list_keys = _convert_to_list(dst_strategy_dict).keys() if dst_strategy_dict else []
    for needed_rank_list_key, transform_rank_list in needed_rank_list_map.items():
        param_total_dict = defaultdict(dict)
        param_attr_dict = defaultdict(dict)
        needed_rank_list = needed_rank_list_key.split("-")
        for needed_rank in needed_rank_list:
            if pipe_param_list:
                safetensor_dict = dict()
                if src_strategy_file is not None:
                    io_time = _find_remove_redundancy_rank_id(pipe_param_list, single_param_dict, file_dict,
                                                              safetensor_dict, redundancy_dict, needed_rank,
                                                              device_num, choice_func)
                    io_cost_time += io_time
                else:
                    with _fast_safe_open(all_safetensor_files_map.get(int(needed_rank)), framework="np") as f:
                        if not unified_flag:
                            all_param_name_set = set(f.keys())
                            src_param_name_set = set(src_strategy_list_keys)
                            dst_param_name_set = set(dst_strategy_list_keys)
                            hyper_param_set = all_param_name_set - (src_param_name_set & dst_param_name_set)
                            pipe_param_list.extend(list(hyper_param_set))
                        if f.metadata() is not None:
                            meta_data.update(f.metadata())
                        io_time = 0
                        for param_name in pipe_param_list:
                            if param_name not in f.keys():
                                # param not in ckpt file, check reason
                                continue
                            start_time = time.time()
                            output = f.get_tensor(param_name)
                            end_time = time.time()
                            cost_time = end_time - start_time
                            io_time += cost_time
                            io_cost_time += io_time
                            if choice_func is not None:
                                choice_out = choice_func(param_name)
                                if isinstance(choice_out, bool) and not choice_out:
                                    continue
                                if not isinstance(choice_out, (bool, str)):
                                    raise ValueError("For 'unified_safetensors', the return value type of the function "
                                                     f"'choice_func' must be bool or str, but got {type(choice_out)}.")
                            safetensor_dict[param_name] = output
            else:
                start_time = time.time()
                safetensor_dict = load_file(all_safetensor_files_map.get(int(needed_rank)))
                end_time = time.time()
                cost_time = end_time - start_time
                io_cost_time += cost_time

            for param_name, param in safetensor_dict.items():
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
                                                                  param_total_dict_keys, src_strategy_file, choice_func)
            if file_index is not None:
                save_safetensor_file = f"part{file_index}.{output_format}"
                save_safetensor_file_dir = dst_safetensors_dir
            else:
                save_safetensor_file = f"{ckpt_prefix}{transform_rank}.{output_format}"
                save_safetensor_file_dir = os.path.join(dst_safetensors_dir, "rank_{}".format(transform_rank))

            if not os.path.exists(save_safetensor_file_dir):
                _make_dir(save_safetensor_file_dir, "path")
            save_file_name = os.path.join(save_safetensor_file_dir, save_safetensor_file)
            if _transform_param_list is not None:
                _transform_param_list.append({save_file_name: transform_param_dict})
            else:
                if transform_param_dict:
                    if output_format == "safetensors":
                        if meta_data and "remove_redundancy" in meta_data:
                            meta_data["remove_redundancy"] = "False"
                        _save_file_atomically(transform_param_dict, save_file_name, metadata=meta_data)
                    else:
                        transform_param_dict = _load_and_transform(transform_param_dict, None, None,
                                                                   transform_func=lambda v, name: Parameter(v,
                                                                                                            name=name))
                        ms.save_checkpoint(transform_param_dict, save_file_name)
            del param_total_dict_keys
        del param_total_dict
    return io_cost_time


def _save_final_safetensors(_transform_param_list, output_format):
    """save file with list"""
    new_transform_dict = {}
    for transform_dict in _transform_param_list:
        for save_file_name, transform_param_dict in transform_dict.items():
            if save_file_name not in new_transform_dict:
                new_transform_dict[save_file_name] = transform_param_dict
            else:
                new_transform_dict[save_file_name].update(transform_param_dict)
    for save_file_name, transform_param_dict in new_transform_dict.items():
        if output_format == "safetensors":
            _save_file_atomically(transform_param_dict, save_file_name, metadata={"format": "ms"})
        else:
            transform_param_dict = _load_and_transform(transform_param_dict, None, None,
                                                       transform_func=lambda v, name: Parameter(v, name=name))
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
        safetensor_dict = load_file(file_name)
        for param_name, param in safetensor_dict.items():
            # cut the parameter not in the pipeline stage.
            if _parameter_not_in_local_stage(param_name, origin_src_strategy_list, src_strategy_list) \
                    and _parameter_not_in_local_stage(param_name, origin_dst_strategy_list, dst_strategy_list):
                continue
            src_rank = rank % src_stage_device_num
            param_type_dict[param_name][src_rank] = str(param.data.dtype)
            param_total_dict[param_name][src_rank] = param
            param_attr_dict[param_name][src_rank] = (True, False)

    ckpt_prefix = os.path.basename(ckpt_prefix)
    if '..' in ckpt_prefix or '/' in ckpt_prefix or '\\' in ckpt_prefix:
        raise ValueError(f"Invalid ckpt_prefix: {ckpt_prefix}. Must not contain path traversal characters.")

    for local_rank_id in range(dst_stage_device_num):
        transform_param_dict = _transform_parallel_safetensor(local_rank_id, param_total_dict,
                                                              param_attr_dict, src_strategy_list, dst_strategy_list,
                                                              param_type_dict)
        save_safetensor_file = "{}{}_part{}.safetensors".format(ckpt_prefix, local_rank_id, stage_id)
        save_safetensor_file_dir = os.path.join(dst_safetensors_dir, "rank_{}".format(local_rank_id))
        if not os.path.exists(save_safetensor_file_dir):
            _make_dir(save_safetensor_file_dir, "path")
        save_safetensor_file_name = os.path.join(save_safetensor_file_dir, save_safetensor_file)
        _save_file_atomically(transform_param_dict, save_safetensor_file_name, metadata={"format": "ms"})


def transform_safetensors_by_rank(rank_id, safetensor_files_map, save_safetensor_file_name,
                                  src_strategy_file=None, dst_strategy_file=None):
    """
    Transform distributed checkpoint from source sharding strategy to destination sharding strategy by rank.
    """
    save_safetensor_file_name = os.path.abspath(save_safetensor_file_name)
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
        safetensor_dict = load_file(file_name)
        for param_name, param in safetensor_dict.items():
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
    _save_file_atomically(transform_param_dict, save_safetensor_file_name, metadata={"format": "ms"})


def _extract_numbers(s):
    """Extract all numbers from a string and convert them to integers."""
    return [int(num) for num in re.findall(r'\d+', s)]


def _extract_last_two_numbers(file_name):
    """Get the last two numbers from a filename."""
    all_numbers = _extract_numbers(file_name)
    return all_numbers[-2:]


def _find_shortest_file(matched_files, rank_ckpts, new_file_suffix, file_suffix):
    """Find the shortest file from a list of matched files."""
    min_length = min(len(os.path.basename(ckpt)) for ckpt in matched_files)
    shortest_files = [ckpt for ckpt in matched_files if len(os.path.basename(ckpt)) == min_length]
    if len(shortest_files) == 1:
        return shortest_files[0]
    raise ValueError(f"Multiple files with suffix '{file_suffix}' found in {rank_ckpts}. Following MindSpore naming "
                     f"rules, searched for files ending with '{new_file_suffix}' but found multiple "
                     f"files {matched_files}. Then searched for the shortest filename, but found multiple shortest "
                     f"files {shortest_files}. Please set file_suffix to the longest common suffix of all files.")


def _get_matched_file(matched, rank_ckpts, new_file_suffix, file_suffix):
    """Get the file from a list of matched files."""
    if len(matched) == 1:
        return matched[0]
    if len(matched) > 1:
        return _find_shortest_file(matched, rank_ckpts, new_file_suffix, file_suffix)
    raise ValueError(f"Multiple files with suffix '{file_suffix}' found in {rank_ckpts}. Following MindSpore naming "
                     f"rules, searched for files ending with '{new_file_suffix}' but found zero files. "
                     f"Please set file_suffix to the longest common suffix of all files.")


def _find_most_matching_file(rank_ckpts, file_suffix, format):
    """Finds the most matching checkpoint file based on the file_suffix."""
    if file_suffix is None:
        rank_ckpts.sort(key=_extract_last_two_numbers)
        return rank_ckpts[-1]

    new_file_suffix = file_suffix
    pattern1 = rf'^_(\d+)-(\d+)_(\d+)$'
    matches1 = re.search(pattern1, file_suffix)
    pattern2 = rf'^(\d+)-(\d+)_(\d+)$'
    matches2 = re.search(pattern2, file_suffix)
    # Pattern matching for _{task_id}-{epoch}_{step} format (e.g., _1-10_100 or 1-10_100)
    if matches1 is not None or matches2 is not None:
        if matches2 is not None:
            new_file_suffix = "_" + new_file_suffix
        matched = [ckpt for ckpt in rank_ckpts if ckpt.endswith(f"{new_file_suffix}.{format}") and
                   not ckpt.endswith(f"rank{new_file_suffix}.{format}")]
        return _get_matched_file(matched, rank_ckpts, new_file_suffix, file_suffix)

    pattern3 = rf'^-(\d+)_(\d+)$'
    matches3 = re.search(pattern3, file_suffix)
    pattern4 = rf'^(\d+)_(\d+)$'
    matches4 = re.search(pattern4, file_suffix)
    # Pattern matching for -{epoch}_{step} format (e.g., -10_100 or 10_100)
    if matches3 is not None or matches4 is not None:
        if matches4 is not None:
            new_file_suffix = "-" + new_file_suffix
        matched = [ckpt for ckpt in rank_ckpts if ckpt.endswith(f"{new_file_suffix}.{format}")]
        return _get_matched_file(matched, rank_ckpts, new_file_suffix, file_suffix)

    pattern5 = rf'^_(\d+)$'
    matches5 = re.search(pattern5, file_suffix)
    pattern6 = rf'^(\d+)$'
    matches6 = re.search(pattern6, file_suffix)
    # Pattern matching for _{step} format (e.g., _100 or 100)
    if matches5 is not None or matches6 is not None:
        if matches6 is not None:
            new_file_suffix = "_" + new_file_suffix
        matched = [ckpt for ckpt in rank_ckpts if ckpt.endswith(f"{new_file_suffix}.{format}")]
        return _get_matched_file(matched, rank_ckpts, new_file_suffix, file_suffix)

    raise ValueError(f"Multiple {format} files ending with '{file_suffix}' found in {rank_ckpts}. "
                     f"Cannot determine which file is the intended one. "
                     f"Please set file_suffix to the longest common suffix.")


def _collect_safetensor_files(src_safetensors_dir, format='safetensors', file_suffix=None):
    """
    Collects all safetensors files from the specified directory and its subdirectories.
    """
    if os.path.isfile(src_safetensors_dir) and format == 'safetensors' and src_safetensors_dir.endswith('safetensors'):
        return {0: src_safetensors_dir}
    safetensors_rank_dir_list = os.path.join(src_safetensors_dir, "rank_[0-9]*")
    all_safetensor_files_map = {}
    multiple_files_found_flag = False
    multiple_files_list = None
    chosen_file = None
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
        if file_suffix is None:
            safetensor_file_name = os.path.join(safetensor_dir, f"*.{format}")
        else:
            safetensor_file_name = os.path.join(safetensor_dir, f"*{file_suffix}.{format}")
        rank_ckpts = glob.glob(safetensor_file_name)
        if len(rank_ckpts) > 1:
            all_safetensor_files_map[rank_id] = _find_most_matching_file(rank_ckpts, file_suffix, format)
            if not multiple_files_found_flag:
                multiple_files_found_flag = True
                multiple_files_list = copy.deepcopy(rank_ckpts)
                chosen_file = all_safetensor_files_map[rank_id]
        elif rank_ckpts:
            all_safetensor_files_map[rank_id] = rank_ckpts[0]
        elif file_suffix is not None:
            raise ValueError(f"No safetensors files found in directory '{safetensor_dir}' "
                             f"with suffix '{file_suffix}' and format '{format}'. "
                             f"Please verify the directory contains the expected files. "
                             f"Recommend setting file_suffix to the longest common suffix.")
    if file_suffix is not None and multiple_files_found_flag:
        logger.warning(f"When unified_safetensors files with file_suffix `{file_suffix}`, multiple files were found. "
                       f"Showing one list: {multiple_files_list}; selected `{chosen_file}` from it. "
                       f"Please check whether the file_suffix is set correctly.")
    return all_safetensor_files_map


def _find_needed_ranks(src_strategy_dict, dst_strategy_dict):
    """
    Identifies the ranks needed for transformation based on source and destination strategies.
    """
    needed_rank_list_map = defaultdict(list)
    dst_stage_device_num = _get_device_num_from_strategy(dst_strategy_dict)
    dst_stage_num = _extract_pipeline_stage_num(dst_strategy_dict)
    dst_device_num = dst_stage_device_num * dst_stage_num
    for rank in range(dst_device_num):
        needed_rank_list = ms.rank_list_for_transform(rank, src_strategy_dict, dst_strategy_dict)
        needed_rank_list_key = "-".join([str(r) for r in needed_rank_list])
        needed_rank_list_map[needed_rank_list_key].append(rank)
    return needed_rank_list_map


def load_file_by_param_name(filename, parme_name_list):
    result = {}
    with _fast_safe_open(filename, framework="np") as f:
        for k in parme_name_list:
            result[k] = f.get_tensor(k)
    return result


def _transform_parallel_safetensor(rank_id, param_total_dict, param_attr_dict, src_strategy_list,
                                   dst_strategy_list, param_total_dict_keys=None, src_strategy_file=None,
                                   choice_func=None):
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

        has_layout_from = any(isinstance(i, (list, tuple)) for i in from_tensor_map)
        has_layout_to = any(isinstance(i, (list, tuple)) for i in to_tensor_map_origin)

        from_dev_matrix, from_tensor_map, from_full_tensor_shape = _construct_tensor_layout_for_opt_shard(
            from_dev_matrix, from_tensor_map, from_opt_shard_step, from_opt_shard_size, origin_tensor_shape)
        to_dev_matrix, to_tensor_map, to_full_tensor_shape = _construct_tensor_layout_for_opt_shard(
            to_dev_matrix_origin, to_tensor_map_origin, to_opt_shard_step, to_opt_shard_size, origin_tensor_shape)
        # Convert tensor layout to same device num
        from_tensor_layout, to_tensor_layout = _construct_from_to_tensor_layout(from_full_tensor_shape,
                                                                                from_dev_matrix,
                                                                                from_tensor_map,
                                                                                to_full_tensor_shape,
                                                                                to_dev_matrix,
                                                                                to_tensor_map,
                                                                                param_name)

        # when the from_layout is less devices, the safetensor_map for map[device_num] should using map[0]
        device_list = list(range(0, np.prod(from_tensor_layout[0])))
        if rank_id % device_num not in param_attr_dict[param_name] and src_strategy_file is None:
            raise ValueError("The param: {} in rank {} is missing.".format(param_name, rank_id % device_num))
        param_rank_map = _get_needed_rank_transform_operator_map_by_layouts(from_tensor_layout, to_tensor_layout,
                                                                            device_list, rank_id)

        from_info_tuple = (from_opt_shard_size, from_dev_matrix, from_tensor_map, from_full_tensor_shape)
        to_info_tuple = (to_opt_shard_size, to_dev_matrix_origin, to_tensor_map_origin, origin_tensor_shape)
        _insert_opt_shard_reshape(param_rank_map, from_info_tuple, to_info_tuple)
        _insert_expand_layout_reshape(param_rank_map, from_info_tuple, to_info_tuple, has_layout_from, has_layout_to)
        transform_operator_stack = _generate_transform_operator_stack(param_rank_map, rank_id)
        param_total_dict_copy = param_total_dict[param_name].copy()
        _apply_tensor_transform_operators(transform_operator_stack, param_total_dict_copy, device_num)
        if choice_func is not None:
            choice_out = choice_func(param_name)
            if isinstance(choice_out, str):
                param_name = choice_out
        transform_param_dict[param_name] = param_total_dict_copy[rank_id % device_num]

    # Handle those parameter like learning_rate, global_step which not in strategy_file.
    for param_name in param_total_dict_keys:
        if choice_func is not None:
            choice_out = choice_func(param_name)
            if isinstance(choice_out, str):
                continue
        if param_name not in transform_param_dict:
            transform_para = param_total_dict[param_name][rank_id % device_num]
            transform_param_dict[param_name] = transform_para
    return transform_param_dict


def _cal_param_size(shape, dtype):
    """cal param size by dtype and shape"""
    num_elements = math.prod(shape)
    element_size = np_dtype_size.get(str(dtype), 4)
    total_bytes = num_elements * element_size
    return total_bytes


def _split_weight_dict(weights, num_groups):
    """split weights by num"""
    sorted_items = sorted(weights.items(), key=lambda x: -x[1])
    groups = [[] for _ in range(num_groups)]
    total_bytes = [0] * num_groups
    for weight_name, byte_size in sorted_items:
        min_index = total_bytes.index(min(total_bytes))
        groups[min_index].append(weight_name)
        total_bytes[min_index] += byte_size

    return groups


def _save_hyper_param(split_dst_file, all_safetensor_files_map, name_list, dst_dir):
    """save hyper param"""
    if not split_dst_file or (split_dst_file and split_dst_file[0] == 1):
        with _fast_safe_open(all_safetensor_files_map.get(0), framework="np") as f:
            all_key = f.keys()
            hyper_parameter = set(all_key) - set(name_list)
            if hyper_parameter:
                hyper_dict = {}
                for key in hyper_parameter:
                    hyper_dict[key] = f.get_tensor(key)
                _save_file_atomically(hyper_dict, os.path.join(dst_dir, "hyper_param.safetensors"),
                                      metadata={"format": "ms"})


def _save_parameter_map_json(split_list, choice_func, split_dst_file, dst_dir, param_total_size):
    """save parameter map json file"""
    param_name_dict = dict()
    for index, part_list in enumerate(split_list):
        for name in part_list:
            save_param_name = name
            if choice_func is not None:
                choice_out = choice_func(name)
                if isinstance(choice_out, str):
                    save_param_name = choice_out
            if save_param_name == -1:
                break
            param_name_dict[save_param_name] = f"part{index}.safetensors"
    output_dict = {"metadata": {"total_size": param_total_size}, "weight_map": param_name_dict}
    if not split_dst_file or (split_dst_file and split_dst_file[0] == 1):
        json_str = json.dumps(output_dict, indent=4)
        map_file = os.path.join(dst_dir, "param_name_map.json")
        with open(map_file, 'w') as f:
            f.write(json_str)


def _get_dst_shape(param_name, param_shape, src_strategy_list):
    """get dst shape by strategy"""
    from_dev_matrix = [1]
    from_tensor_map = [-1] * len(param_shape)
    from_opt_shard_size = 0
    if src_strategy_list is not None:
        from_dev_matrix, from_tensor_map, _, from_opt_shard_size = _extract_layout_item(
            src_strategy_list.get(param_name))
    to_dev_matrix_origin = [1]
    to_tensor_map_origin = [-1] * len(param_shape)
    to_opt_shard_step = 0
    to_opt_shard_size = 0

    param_strategy = _get_tensor_strategy(from_dev_matrix, from_tensor_map)
    origin_tensor_shape = ()
    for i, item in enumerate(param_shape):
        if i == 0 and from_opt_shard_size > 0:
            origin_tensor_shape += (item * param_strategy[i] * from_opt_shard_size,)
            continue
        origin_tensor_shape += (item * param_strategy[i],)

    _, _, to_full_tensor_shape = _construct_tensor_layout_for_opt_shard(
        to_dev_matrix_origin, to_tensor_map_origin, to_opt_shard_step, to_opt_shard_size, origin_tensor_shape)
    return to_full_tensor_shape


def _check_remove_redundancy(merge_with_redundancy, f):
    """Check whether remove_redundancy is consistent with the safetensors file."""
    if f.metadata() is not None and "remove_redundancy" in f.metadata().keys():
        if f.metadata()["remove_redundancy"] == "True" and merge_with_redundancy:
            logger.warning("For 'unified_safetensors', the safetensors file is deduplicated, "
                           "but merge_with_redundancy is set to True.")
            return False
        if f.metadata()["remove_redundancy"] == "False" and not merge_with_redundancy:
            logger.warning("For 'unified_safetensors', the safetensors file is non-deduplicated, "
                           "but merge_with_redundancy is set to False.")
            return True
    return merge_with_redundancy


def set_affinity_pid():
    """Set CPU affinity pid"""
    pid = os.getpid()
    total_cores = os.cpu_count()
    all_cores = set(range(total_cores))
    os.sched_setaffinity(pid, all_cores)


def _validate_safetensors_files(target_directory, expected_file_ids):
    """Validate whether safetensors files are completely generated in the target directory."""
    missing_file_ids = []
    for file_id in expected_file_ids:
        safetensors_file = os.path.join(target_directory, f"part{file_id}.safetensors")
        if os.path.exists(safetensors_file):
            continue
        missing_file_ids.append(file_id)

    if missing_file_ids:
        logger.warning(
            f"For unified_safetensors, target file part {missing_file_ids} does not exist. "
            f"Possible causes: file rename failed, insufficient permissions, or disk space shortage."
        )


@_mstx_range_decorator("unified_safetensors", domain="model_preparation")
def unified_safetensors(src_dir, src_strategy_file, dst_dir, merge_with_redundancy=True, file_suffix=None,
                        max_process_num=64, choice_func=None, split_dst_file=()):
    """
    Merge multiple safetensor files into a unified safetensor file.

    Note:
        When merging weights, it will verify whether the `merge_with_redundancy` parameter differs from
        the deduplication flag in the merged safetensors files. If they are the same, the merging will be performed
        according to the deduplication flag in the files.

    Args:
        src_dir (str): Source weight saving directory.
        src_strategy_file (str): Source weight segmentation strategy file with the file extension `.ckpt` .
        dst_dir (str): Target save directory.
        merge_with_redundancy (bool, optional): Whether the merged source weight files are de-duplicated and
            saved safetensors files. Default: ``True``, indicating that the merged source weight files are complete.
        file_suffix (str, optional): Specify the filename suffix for merging safetensors files. Default: ``None``,
            meaning all safetensors files in the source weight directory will be merged.
        max_process_num (int, optional): Maximum number of processes. Default: ``64``.
        choice_func (callable, optional): A callable function used to filter parameters or modify parameter names.
            The return value of the function must be of type str (string) or bool (boolean). Default: ``None``.
        split_dst_file (tuple, optional) - A parameter used to manually split a task into multiple subtasks for
            execution, represented as a tuple containing two elements. The first element indicates the number of
            the current subtask, and the second element indicates the total number of tasks. This parameter supports
            splitting and executing tasks multiple times on a single machine, and also supports executing different
            subtasks on multiple machines respectively. Default: ``()``.

    Raises:
        ValueError: If the safetensors file of rank is missing.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore as ms
        >>> src_dir = "/usr/safetensors/llama31B/4p_safetensors/"
        >>> src_strategy_file = "/usr/safetensors/llama31B/strategy_4p.ckpt"
        >>> dst_dir = "/usr/safetensors/llama31B/merge_llama31B_4p/"
        >>> ms.parallel.unified_safetensors(src_dir, src_strategy_file, dst_dir)
    """
    set_affinity_pid()
    _check_transform_safetensors(src_dir, "", src_strategy_file, None)
    _make_dir(dst_dir, "path")
    if os.path.isfile(src_dir):
        raise ValueError("For 'unified_safetensors', the 'src_dir' can not be a file.")
    all_safetensor_files_map = _collect_safetensor_files(src_dir, format="safetensors", file_suffix=file_suffix)
    all_ckpt_files_map = _collect_safetensor_files(src_dir, format="ckpt")
    if all_safetensor_files_map and all_ckpt_files_map:
        raise ValueError("For 'unified_safetensors', the 'src_dir' cannot contain "
                         "both ckpt file and safetensors file simultaneously")
    src_strategy_dict = _build_searched_strategy(src_strategy_file)
    src_stage_device_num = _get_device_num_from_strategy(src_strategy_dict)
    dst_stage_device_num = 1
    origin_src_strategy_list = _extract_layout_map(src_strategy_dict)
    origin_dst_strategy_list = None

    needed_rank_list_map = _find_needed_ranks(src_strategy_dict, dst_strategy_dict=None)
    for needed_rank_list, rank in needed_rank_list_map.items():
        for needed_rank in needed_rank_list.split("-"):
            if int(needed_rank) not in all_safetensor_files_map:
                raise ValueError("The safetensor file of rank{} is needed for converting rank{}'s safetensor, "
                                 "but it is missing.".format(needed_rank, rank))
    layout_map = _convert_to_list(src_strategy_dict)

    actual_params = set()
    for _, file_name in all_safetensor_files_map.items():
        with _fast_safe_open(file_name, framework="np") as f:
            actual_params.update(f.keys())
            merge_with_redundancy = _check_remove_redundancy(merge_with_redundancy, f)

    params_to_store = actual_params & set(layout_map.keys())

    name_list = []
    for name in list(params_to_store):
        if name.startswith("accu_grads"):
            continue
        name_list.append(name)

    param_size_dict = {}
    param_total_size = 0
    for _, file_name in all_safetensor_files_map.items():
        with _fast_safe_open(file_name, framework="np") as f:
            for k in f.keys():
                if k in name_list:
                    if choice_func is not None:
                        choice_out = choice_func(k)
                        if isinstance(choice_out, bool):
                            if not choice_out:
                                name_list.remove(k)
                                continue
                    if k not in param_size_dict:
                        py_slice = f.get_tensor(k)
                        param_dst_shape = _get_dst_shape(k, py_slice.shape, origin_src_strategy_list)
                        # Convert the shape of np.int32 type to int type to prevent overflow in subsequent calculations.
                        param_dst_shape = [int(item) for item in param_dst_shape]
                        param_size = _cal_param_size(param_dst_shape, py_slice.dtype)
                        param_total_size += param_size
                        param_size_dict[k] = param_size
    split_num = math.ceil(sum(param_size_dict.values()) / 1024 / 1024 / 1024 / 3)
    split_num = min(split_num, len(name_list))
    split_list = _split_weight_dict(param_size_dict, split_num)

    if split_dst_file:
        current_machine_num = split_dst_file[0]
        total_machine_num = split_dst_file[1]
        n = len(split_list)
        avg_length = n // total_machine_num
        remainder = n % total_machine_num
        start_index = (avg_length * (current_machine_num - 1)) + min(current_machine_num - 1, remainder)
        end_index = start_index + avg_length + (1 if current_machine_num <= remainder else 0)
        sub_list = []
        for i, item in enumerate(split_list):
            if start_index <= i < end_index:
                sub_list.append(item)
            else:
                sub_list.append([-1])
        split_num = end_index - start_index
        res = list(range(start_index, end_index))
    else:
        sub_list = split_list
        res = [i for i in range(split_num)]

    _save_hyper_param(split_dst_file, all_safetensor_files_map, name_list, dst_dir)
    _save_parameter_map_json(split_list, choice_func, split_dst_file, dst_dir, param_total_size)

    max_process = min(split_num, max_process_num)
    file_ids = res[:]
    res = _split_list(res, max_process)
    processes = []
    src_strategy_name = None
    if not merge_with_redundancy:
        src_strategy_name = src_strategy_file
    if max_process > 1:
        for i in range(max_process):
            p = mp.Process(target=_transform_safetensors_single_semaphore, args=(
                needed_rank_list_map, all_safetensor_files_map, src_stage_device_num, dst_stage_device_num,
                src_strategy_dict, None, origin_src_strategy_list, origin_dst_strategy_list,
                "", dst_dir, "safetensors", None, sub_list, res[i], True, src_strategy_name, choice_func))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
    else:
        _transform_safetensors_single_semaphore(needed_rank_list_map, all_safetensor_files_map, src_stage_device_num,
                                                dst_stage_device_num, src_strategy_dict, None,
                                                origin_src_strategy_list, origin_dst_strategy_list, "",
                                                dst_dir, "safetensors", None, sub_list,
                                                res[0], True, src_strategy_name, choice_func)
    _validate_safetensors_files(dst_dir, file_ids)


def _transform_safetensors_single_semaphore(needed_rank_list_map, all_safetensor_files_map,
                                            src_stage_device_num,
                                            dst_stage_device_num,
                                            src_strategy_dict, dst_strategy_dict, origin_src_strategy_list,
                                            origin_dst_strategy_list,
                                            ckpt_prefix, dst_safetensors_dir, output_format,
                                            _transform_param_list, pipe_param_list=None, file_index=None,
                                            unified_flag=False, src_strategy_file=None, choice_func=None):
    """transform safetensors single semaphore"""
    total_io_cost_time = 0
    for i in file_index:
        io_cost_time = _transform_safetensors_single(needed_rank_list_map, all_safetensor_files_map,
                                                     src_stage_device_num, dst_stage_device_num, src_strategy_dict,
                                                     dst_strategy_dict, origin_src_strategy_list,
                                                     origin_dst_strategy_list, ckpt_prefix, dst_safetensors_dir,
                                                     output_format, _transform_param_list, pipe_param_list[i], i,
                                                     unified_flag, src_strategy_file, choice_func)
        while psutil.virtual_memory().percent > 50:
            time.sleep(1)
        total_io_cost_time += io_cost_time
    vlog_print("1", "ME", __file__, sys._getframe().f_lineno,
               f"Unified safetensors io cost time:{total_io_cost_time}.")


def _split_list(split_list, split_num):
    split_array = np.array_split(split_list, split_num)
    return [array.tolist() for array in split_array]


def _apply_sf_obj_transform_operators(transform_operator_stack, sf_obj, device_num):
    """apply safetensors object operators"""
    if not transform_operator_stack:
        return sf_obj
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
                cur_level = operator_pair[1]
                operator = operator_pair[2]
                if operator[0] != op_name:
                    raise ValueError("The operator in the same level should be equal in the transform tensor operator "
                                     "list, but the find {} and {} in level {}".format(op_name, operator[0], cur_level))
                if operator[0] != "AllConcat":
                    sf_obj = _apply_operator(operator[0])(sf_obj, operator)
                    continue
                for rank in operator[1][:-1]:
                    if rank % device_num not in sf_obj:
                        raise ValueError("The checkpoint file of rank {} is missing.".format(rank % device_num))
                allgather_list = [sf_obj for _ in operator[1][:-1]]
                tmp_tensor_dict[rank_id % device_num] = _apply_operator(operator[0])(allgather_list, operator)
            if op_name == "AllConcat":
                for _, value in tmp_tensor_dict.items():
                    sf_obj = value
            level_operators.clear()
        if not transform_operator_stack:
            break
        operator_pair = transform_operator_stack.pop()
        level = operator_pair[1]
        level_operators.append(operator_pair)
    return sf_obj


def _process_hyper_params(file_list, total_safetensors_dir, total_param):
    """process hyper params"""
    if 'hyper_param.safetensors' in file_list:
        hyper_parameter_file_name = os.path.join(total_safetensors_dir, "hyper_param.safetensors")
        with _fast_safe_open(hyper_parameter_file_name, framework="np") as f:
            for key in f.keys():
                total_param[key] = Parameter(Tensor.from_numpy(f.get_tensor(key)))
    return total_param


def _get_param_name_map_by_file(file_name, file_list, name_map):
    """get param_name_map by file"""
    with _fast_safe_open(file_name, framework="np") as f:
        keys = f.keys()
        values = len(keys) * [file_list[0]]
        if name_map:
            flipped_name_map = {value: key for key, value in name_map.items()}
            keys = [flipped_name_map.get(key, key) for key in keys]
        param_name_map = dict(zip(keys, values))
    return param_name_map


def _cal_param_name_map_and_param_list(file_list, total_safetensors_dir, json_files,
                                       dst_strategy_file, rank_id, name_map=None):
    """calculate param_name_map and param_list"""
    if len(file_list) == 1:
        logger.info("There is only one weight file in the directory, which will be automatically mapped.")
        file_name = os.path.join(total_safetensors_dir, file_list[0])
        is_file = os.path.isfile(file_name)
        if not is_file:
            raise ValueError(f"For 'load_parallel_checkpoint', weight files must be included "
                             f"in the `unified_safetensors_dir`.")
        param_name_map = _get_param_name_map_by_file(file_name, file_list, name_map)
    else:
        if not json_files:
            raise ValueError(
                f"For 'load_parallel_checkpoint', there must be a JSON file named 'param_name_map.json' in "
                f"the 'total_safetensors_dir'.")
        param_name_json = os.path.join(total_safetensors_dir, json_files[0])
        with open(param_name_json, 'r') as f:
            param_name_map = json.load(f)
            if "weight_map" in param_name_map:
                param_name_map = param_name_map["weight_map"]

    if dst_strategy_file is not None:
        _, dst_strategy_list = _extract_src_dst_layout_map(rank_id, None, dst_strategy_file)
        param_list = dst_strategy_list.keys()
    else:
        dst_strategy_list = None
        param_list = param_name_map.keys()
    return param_name_map, param_list, dst_strategy_list


def _cal_transform_operator_stack_and_device_num(from_dev_matrix, from_tensor_map, from_opt_shard_step,
                                                 from_opt_shard_size, param_name, dst_strategy_list, tensor_shape,
                                                 local_rank_id):
    """cal transform_operator_stack and device_num"""
    to_dev_matrix_origin, to_tensor_map_origin, to_opt_shard_step, to_opt_shard_size = _extract_layout_item(
        dst_strategy_list.get(param_name))

    device_num = np.prod(from_dev_matrix)
    param_strategy = _get_tensor_strategy(from_dev_matrix, from_tensor_map)
    origin_tensor_shape = ()
    for i, item in enumerate(tensor_shape):
        if i == 0 and from_opt_shard_size > 0:
            origin_tensor_shape += (item * param_strategy[i] * from_opt_shard_size,)
            continue
        origin_tensor_shape += (item * param_strategy[i],)

    has_layout_from = any(isinstance(i, (list, tuple)) for i in from_tensor_map)
    has_layout_to = any(isinstance(i, (list, tuple)) for i in to_tensor_map_origin)

    from_dev_matrix, from_tensor_map, from_full_tensor_shape = _construct_tensor_layout_for_opt_shard(
        from_dev_matrix, from_tensor_map, from_opt_shard_step, from_opt_shard_size, origin_tensor_shape)
    to_dev_matrix, to_tensor_map, to_full_tensor_shape = _construct_tensor_layout_for_opt_shard(
        to_dev_matrix_origin, to_tensor_map_origin, to_opt_shard_step, to_opt_shard_size, origin_tensor_shape)
    # Convert tensor layout to same device num
    from_tensor_layout, to_tensor_layout = _construct_from_to_tensor_layout(from_full_tensor_shape,
                                                                            from_dev_matrix,
                                                                            from_tensor_map,
                                                                            to_full_tensor_shape,
                                                                            to_dev_matrix,
                                                                            to_tensor_map,
                                                                            param_name)

    # when the from_layout is less devices, the safetensor_map for map[device_num] should using map[0]
    device_list = list(range(0, np.prod(from_tensor_layout[0])))
    param_rank_map = _get_needed_rank_transform_operator_map_by_layouts(from_tensor_layout, to_tensor_layout,
                                                                        device_list, local_rank_id)

    from_info_tuple = (from_opt_shard_size, from_dev_matrix, from_tensor_map, from_full_tensor_shape)
    to_info_tuple = (to_opt_shard_size, to_dev_matrix_origin, to_tensor_map_origin, origin_tensor_shape)
    _insert_opt_shard_reshape(param_rank_map, from_info_tuple, to_info_tuple)
    _insert_expand_layout_reshape(param_rank_map, from_info_tuple, to_info_tuple,
                                  has_layout_from, has_layout_to)
    transform_operator_stack = _generate_transform_operator_stack(param_rank_map, local_rank_id)
    return transform_operator_stack, device_num


def check_param_dtype(file, param_name):
    dtype_need_changed = False
    changed_dtype = None
    if file.metadata() is not None and param_name in file.metadata().keys():
        dtype_need_changed = True
        sf_dtype = file.metadata()[param_name]
        changed_dtype = safetensors_to_mstype[sf_dtype]
    return dtype_need_changed, changed_dtype


def _load_parallel_checkpoint(file_info):
    """load parallel safetensors by merged file."""
    total_safetensors_dir, dst_strategy_file, net, dst_safetensors_dir, \
        rank_id, output_format, name_map, return_param_dict = file_info
    set_affinity_pid()
    file_list = os.listdir(total_safetensors_dir)
    json_files = [file for file in file_list if file == "param_name_map.json"]
    sf_files = [file for file in file_list if file.endswith('.safetensors')]
    param_name_map, param_list, dst_strategy_list = _cal_param_name_map_and_param_list(sf_files, total_safetensors_dir,
                                                                                       json_files, dst_strategy_file,
                                                                                       rank_id, name_map)
    total_param = dict()
    dst_stage_device_num = np.prod(dst_strategy_list.get(list(dst_strategy_list.keys())[0])[0]) if dst_strategy_list \
                                                                                                   is not None else 1
    local_rank_id = rank_id % dst_stage_device_num
    total_io_cost_time = 0
    for param_name in _progress_bar(param_list):
        if param_name not in param_name_map:
            continue
        file_name = os.path.join(total_safetensors_dir, param_name_map[param_name])
        with _fast_safe_open(file_name, framework="np") as f:
            cur_param_name = name_map.get(param_name) if name_map is not None and param_name in name_map else param_name
            if cur_param_name not in f.keys():
                continue
            sf_obj = f.get_tensor(cur_param_name)
            dtype_need_changed, changed_dtype = check_param_dtype(f, param_name)

        tensor_shape = sf_obj.shape
        from_dev_matrix = [1]
        from_tensor_map = [-1] * len(tensor_shape)
        from_opt_shard_step = 0
        from_opt_shard_size = 0
        if dst_strategy_list is not None:
            if param_name not in dst_strategy_list:
                continue
            transform_operator_stack, device_num = _cal_transform_operator_stack_and_device_num(from_dev_matrix,
                                                                                                from_tensor_map,
                                                                                                from_opt_shard_step,
                                                                                                from_opt_shard_size,
                                                                                                param_name,
                                                                                                dst_strategy_list,
                                                                                                tensor_shape,
                                                                                                local_rank_id)
            start_time = time.time()
            slice_param = _apply_sf_obj_transform_operators(transform_operator_stack, sf_obj, device_num)
            end_time = time.time()
            cost_time = end_time - start_time
            total_io_cost_time += cost_time
        else:
            start_time = time.time()
            slice_param = sf_obj
            end_time = time.time()
            cost_time = end_time - start_time
            total_io_cost_time += cost_time
        slice_param_copy = np.copy(slice_param)
        if dtype_need_changed:
            total_param[param_name] = Parameter(Tensor(slice_param_copy, dtype=changed_dtype))
        else:
            total_param[param_name] = Parameter(Tensor.from_numpy(slice_param_copy))
    vlog_print("1", "ME", __file__, sys._getframe().f_lineno,
               f"load distributed safetensors io cost time:{total_io_cost_time}.")
    total_param = _process_hyper_params(file_list, total_safetensors_dir, total_param)
    if net is not None:
        if not return_param_dict:
            logger.info("start load param into net...")
            param_not_load, ckpt_not_load = ms.load_param_into_net(net, total_param)
            logger.info("load param into net is end...")
            return param_not_load, ckpt_not_load
        return total_param
    _make_dir(os.path.join(dst_safetensors_dir, f"rank_{rank_id}"), "path")
    ms.save_checkpoint(total_param, os.path.join(dst_safetensors_dir, f"rank_{rank_id}", f"net.{output_format}"),
                       format=output_format)
    return None


__all__ = ["_transform_safetensors", "transform_safetensors_by_stage",
           "transform_safetensors_by_rank", "unified_safetensors"]
