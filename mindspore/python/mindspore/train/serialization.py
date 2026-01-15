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

"""Model and parameters serialization."""
from __future__ import absolute_import
from __future__ import division

import binascii
import copy
import json
import os
import re
import shutil
import stat
import atexit
import threading
from threading import Thread, RLock
from multiprocessing import active_children
import multiprocessing as mp
from collections import OrderedDict
from io import BytesIO
from functools import partial

import math
import sys
import time
from safetensors.numpy import save_file
import numpy as np
import google

from mindspore.train.checkpoint_pb2 import Checkpoint
from mindspore.train.mind_ir_pb2 import ModelProto as mindir_model

import mindspore
from mindspore import nn
from mindspore import context
from mindspore import log as logger
from mindspore.log import vlog_print
from mindspore._checkparam import check_input_data, check_input_dataset
from mindspore import _checkparam as Validator
from mindspore.common import dtype as mstype
from mindspore.common.api import _cell_graph_executor as _executor
from mindspore.common.api import _JitExecutor
from mindspore.common.api import _get_parameter_layout
from mindspore.common.initializer import initializer
from mindspore.common.parameter import Parameter, _offload_if_config
from mindspore.common.tensor import Tensor
from mindspore._c_expression import TensorPy as Tensor_
from mindspore.common.file_system import FileSystem, _register_basic_file_system, _register_mindio_file_system
from mindspore.communication.management import get_rank, get_group_size
from mindspore.experimental import MapParameter
from mindspore.ops import Cast
from mindspore.parallel._cell_wrapper import get_allgather_cell, _single_parameter_broadcast
from mindspore.parallel._tensor import _reshape_param_data
from mindspore.parallel._utils import _is_in_auto_parallel_mode
from mindspore.parallel._ps_context import _set_checkpoint_load_status, _store_warm_up_ptr_by_tensor, \
    _store_warm_up_ptr_by_tensor_list
from mindspore.parallel.checkpoint_transform import sync_pipeline_shared_parameters
from mindspore.parallel.checkpoint_transform import restore_group_info_list as new_restore_group_info_list
from mindspore.parallel.checkpoint_transform import load_distributed_checkpoint as new_load_distributed_checkpoint
from mindspore.parallel.checkpoint_transform import merge_sliced_parameter as new_merge_sliced_parameter
from mindspore.parallel.checkpoint_transform import build_searched_strategy as new_build_searched_strategy
from mindspore.parallel.transform_safetensors import _fast_safe_open
from mindspore.train._utils import get_parameter_redundancy, _progress_bar, _load_and_transform, _mstx_range_decorator
from mindspore._c_expression import load_mindir, _encrypt, _decrypt, _is_cipher_file, \
    split_mindir, split_dynamic_mindir, _get_snapshot_params
from mindspore.common.generator import Generator

tensor_to_ms_type = {"Int8": mstype.int8, "UInt8": mstype.uint8, "Int16": mstype.int16, "UInt16": mstype.uint16,
                     "Int32": mstype.int32, "UInt32": mstype.uint32, "Int64": mstype.int64, "UInt64": mstype.uint64,
                     "Float16": mstype.float16, "Float32": mstype.float32, "Float64": mstype.float64,
                     "Bool": mstype.bool_, "str": mstype.string, "BFloat16": mstype.bfloat16, "Int4": mstype.qint4x2}

_tensor_to_np_type = {"Int8": np.int8, "UInt8": np.uint8, "Int16": np.int16, "UInt16": np.uint16,
                      "Int32": np.int32, "UInt32": np.uint32, "Int64": np.int64, "UInt64": np.uint64,
                      "Float16": np.float16, "Float32": np.float32, "Float64": np.float64, "Bool": np.bool_, "str": "U"}

np_type_convert = {"int32": np.int32, "float32": np.float32, "float16": np.float16, "float64": np.float64}

mindir_to_tensor_type = {1: mstype.float32, 2: mstype.uint8, 3: mstype.int8, 4: mstype.uint16,
                         5: mstype.int16, 6: mstype.int32, 7: mstype.int64, 10: mstype.float16,
                         11: mstype.float64, 12: mstype.uint32, 13: mstype.uint64}

safetensors_to_mstype = {'Int4': mstype.qint4x2}

_ckpt_mutex = RLock()

# unit is KB
SLICE_SIZE = 512 * 1024
PROTO_LIMIT_SIZE = 1024 * 1024 * 2
TOTAL_SAVE = 1024 * 1024
PARAMETER_SPLIT_SIZE = 1024 * 1024 * 1024
ENCRYPT_BLOCK_SIZE = 64 * 1024
INT_64_MAX = 9223372036854775807

cpu_cast = Cast().set_device("CPU")

_ckpt_fs = FileSystem()
_ckpt_fs_initialized = False


def tensor_to_np_type(tensor_type_str):
    """tensor to numpy type"""
    if tensor_type_str == "BFloat16":
        from mindspore.common import np_dtype
        if not np_dtype.np_dtype_valid(True):
            raise TypeError(
                "The Numpy bfloat16 data type is not supported now, please ensure that the current "
                "Numpy version is not less than the version when the mindspore is compiled, "
                "and the major versions are same."
            )
        return np_dtype.bfloat16
    return _tensor_to_np_type.get(tensor_type_str)


def init_ckpt_file_system(fs: FileSystem):
    """Initialize checkpoint file system"""
    if _register_mindio_file_system(fs):
        return
    _register_basic_file_system(fs)


def _ensure_ckpt_fs_initialized():
    """Ensure checkpoint file system is initialized"""
    global _ckpt_fs_initialized
    if not _ckpt_fs_initialized:
        init_ckpt_file_system(_ckpt_fs)
        _ckpt_fs_initialized = True


def _wait_async_process_save_ckpt():
    """Waiting for asynchronous saving process of ckpt to complete"""
    for process in active_children():
        if process.name == "asyn_save_ckpt":
            process.join()


def _wait_async_thread_save_ckpt():
    """Waiting for asynchronous saving thread of ckpt to complete"""
    thread_list = threading.enumerate()
    for thread in thread_list:
        if thread.getName() == "asyn_save_ckpt":
            thread.join()


def _async_save_close():
    """Waiting for asynchronous saving of ckpt to complete"""
    _wait_async_process_save_ckpt()
    _wait_async_thread_save_ckpt()


# Registering atexit handles asynchronous save
atexit.register(_async_save_close)


def _get_cur_rank_dp(parameter_layout_dict):
    """ Get dp and tp from layout dict. """
    global_rank = get_rank()
    parameter_redundancy_dict = get_parameter_redundancy(parameter_layout_dict)
    value_len = sys.maxsize
    min_value = ()
    min_value_set = set()
    for key, value in parameter_redundancy_dict.items():
        if key.startswith("accu_grads") or key.startswith("inputs"):
            continue
        for item in value:
            if global_rank not in item:
                continue
            # if item is subset of min_value_set, update min_value_set and min_value
            if len(item) < value_len:
                if min_value_set and not set(item).issubset(min_value_set):
                    return (global_rank,)
                value_len = len(item)
                min_value_set = set(item)
                min_value = item
            # if value is not smaller than len of min_value len,
            # check if min_value_set is subset of current item
            elif not min_value_set.issubset(set(item)):
                return (global_rank,)
    return min_value


def get_ckpt_path_with_strategy(cur_ckpt_path, cur_strategy_path):
    """
    Find available checkpoint file path from all backup checkpoint files of current rank.
    It suppose that checkpoint path contains substring 'rank_{rank_id}' which is used to
    distinguish between different path.If cur_ckpt_path doesn't have 'rank_{rank_id}' substring, will return
    cur_ckpt_path itself when cur_ckpt_path is exist, otherwise return None.

    Note:
       This API must be called after the communication is initialized because the cluster information
       needs to be obtained internally.

    Args:
        cur_ckpt_path (str): the checkpoint file path which cur rank needs.
        cur_strategy_path (str): strategy file path for current rank.

    Returns:
        - new_ckpt_file (str), if found available checkpoint file , return it.
        - None, if not found available checkpoint, return None.

    Examples:
        >>> import mindspore as ms
        >>> from mindspore.communication import init
        >>> from mindspore import get_ckpt_path_with_strategy
        >>> ms.set_context(mode=ms.GRAPH_MODE)
        >>> ms.set_auto_parallel_context(parallel_mode=ms.ParallelMode.DATA_PARALLEL, gradients_mean=True)
        >>> init()
        >>> ckpt_file= "./rank_5/iteration-1_40.ckpt"
        >>> strategy_file = "./src_pipeline_strategys/src_strategy_5.ckpt"
        >>> ckpt_file_new = get_ckpt_path_with_strategy(ckpt_file, strategy_file)
        >>> print(ckpt_file_new)
    """
    cur_rank = get_rank()
    if f"rank_{str(cur_rank)}" in cur_ckpt_path and os.path.isfile(cur_ckpt_path):
        return cur_ckpt_path
    dp = _get_cur_rank_dp(cur_strategy_path)
    pattern = r'rank_\d+'
    for i in dp:
        new_ckpt_path = re.sub(pattern, f"rank_{str(i)}", cur_ckpt_path)
        if not os.path.isfile(new_ckpt_path):
            continue
        return new_ckpt_path
    return None


class ParamDictFuture:
    def __init__(self, executor, param_dict_future):
        self.executor = executor
        self.param_dict_future = param_dict_future

    def result(self):
        param_dict = self.param_dict_future.result()
        self.executor.shutdown()
        return param_dict


def _special_process_par(par, new_par):
    """
    Processes the special condition.

    Like (12,2048,1,1)->(12,2048), this case is caused by GE 4 dimensions tensor.
    """
    par_shape_len = len(par.data.shape)
    new_par_shape_len = len(new_par.data.shape)
    if new_par_shape_len <= par_shape_len:
        return False

    for i in range(new_par_shape_len - par_shape_len):
        if new_par.data.shape[par_shape_len + i] != 1:
            return False

    if new_par.data.dtype == mstype.bfloat16:
        new_val = cpu_cast(new_par.data, mstype.float32).asnumpy()
    else:
        new_val = new_par.data.asnumpy()

    new_val = new_val.reshape(par.data.shape)
    par.set_data(Tensor(new_val, par.data.dtype))
    return True


def _update_param(param, new_param, strict_load):
    """Updates param's data from new_param's data."""
    if isinstance(param.data, Tensor) and isinstance(new_param.data, Tensor):
        if param.data.shape != new_param.data.shape:
            if not _special_process_par(param, new_param):
                logger.critical("Failed to combine the net and the parameters for param %s.", param.name)
                msg = (f"For 'load_param_into_net', {param.name} in the argument 'net' should have the same shape "
                       f"as {param.name} in the argument 'parameter_dict'. But got its shape {param.data.shape} in"
                       f" the argument 'net' and shape {new_param.data.shape} in the argument 'parameter_dict'."
                       f"May you need to check whether the checkpoint you loaded is correct or the batch size and "
                       f"so on in the 'net' and 'parameter_dict' are same.")
                raise RuntimeError(msg)

        if param.data.dtype != new_param.data.dtype:
            if _type_convert(param, new_param, strict_load):
                new_tensor = Tensor(new_param.data.asnumpy(), param.data.dtype)
                param.set_data(new_tensor, param.sliced)
                return

            logger.critical("Failed to combine the net and the parameters for param %s.", param.name)
            msg = (f"For 'load_param_into_net', {param.name} in the argument 'net' should have the same type as "
                   f"{param.name} in the argument 'parameter_dict'. but got its type {param.data.dtype} in the "
                   f"argument 'net' and type {new_param.data.dtype} in the argument 'parameter_dict'."
                   f"May you need to check whether the checkpoint you loaded is correct.")
            raise RuntimeError(msg)

        param.set_data(new_param.data, param.sliced)
        return

    if isinstance(param.data, Tensor) and not isinstance(new_param.data, Tensor):
        if param.data.shape != (1,) and param.data.shape != ():
            logger.critical("Failed to combine the net and the parameters for param %s.", param.name)
            msg = (f"For 'load_param_into_net', {param.name} in the argument 'parameter_dict' is "
                   f"scalar, then the shape of {param.name} in the argument 'net' should be "
                   f"(1,) or (), but got shape {param.data.shape}."
                   f"May you need to check whether the checkpoint you loaded is correct.")
            raise RuntimeError(msg)
        param.set_data(initializer(new_param.data, param.data.shape, param.data.dtype))

    elif isinstance(new_param.data, Tensor) and not isinstance(param.data, Tensor):
        logger.critical("Failed to combine the net and the parameters for param %s.", param.name)
        msg = (f"For 'load_param_into_net', {param.name} in the argument 'parameter_dict' is Tensor, "
               f"then {param.name} in the argument 'net' also should be Tensor, but got {type(param.data)}."
               f"May you need to check whether the checkpoint you loaded is correct.")
        raise RuntimeError(msg)

    else:
        param.set_data(type(param.data)(new_param.data))


def _type_convert(param, new_param, strict_load):
    """Whether to convert parameter's type during load checkpoint into network."""
    float_type = (mstype.float16, mstype.float32, mstype.float64, mstype.bfloat16)
    int_type = (mstype.int8, mstype.int16, mstype.int32, mstype.int64, mstype.qint4x2)
    if not strict_load and ({param.data.dtype, new_param.data.dtype}.issubset(float_type) or
                            {param.data.dtype, new_param.data.dtype}.issubset(int_type)):
        logger.warning(f"The type of {new_param.name}:{new_param.data.dtype} in 'parameter_dict' is different from "
                       f"the type of it in 'net':{param.data.dtype}, then the type convert from "
                       f"{new_param.data.dtype} to {param.data.dtype} in the network. May consume additional memory "
                       f"and time")
        return True
    return False


def _save_weight(checkpoint_dir, model_name, iteration, params):
    """Save model weight into checkpoint."""
    logger.debug(f"Checkpoint dir is: '{checkpoint_dir}'")
    exist_ckpt_file_list = []
    if os.path.exists(checkpoint_dir):
        for exist_ckpt_name in os.listdir(checkpoint_dir):
            file_prefix = os.path.join(model_name, "_iteration_")
            if exist_ckpt_name.startswith(file_prefix):
                exist_ckpt_file_list.append(exist_ckpt_name)

        param_dict = OrderedDict()
        for key in params.keys():
            value = params[key]
            weight_type = value[0]
            weight_shape = value[1]
            weight_data = value[2]
            weight_size = value[3]
            weight_np = np.array(weight_data, dtype=weight_type.lower())
            logger.debug(f"weight_type: '{weight_type}', weight_shape: '{weight_shape}', weight_size: "
                         f"'{weight_size}', weight_np.nbytes: '{weight_np.nbytes}'")

            param_dict[key] = [weight_shape, weight_type, weight_np]
        ckpt_file_save_name = model_name + "_iteration_" + iteration + ".ckpt"
        ckpt_file_save_path = os.path.join(checkpoint_dir, ckpt_file_save_name)

        _exec_save(ckpt_file_save_path, param_dict)

        for exist_ckpt_name in exist_ckpt_file_list:
            os.remove(os.path.join(checkpoint_dir, exist_ckpt_name))
        logger.info(f"Save weight to checkpoint file path '{ckpt_file_save_path}' success.")
    else:
        logger.warning(f"Checkpoint dir: '{checkpoint_dir}' is not existed.")


@_mstx_range_decorator("_exec_save", domain="model_preparation")
def _exec_save(ckpt_file_name, data_list, enc_key=None, enc_mode="AES-GCM", map_param_inc=False, crc_check=False,
               format="ckpt", remove_redundancy=None):
    """Execute the process of saving checkpoint into file."""
    try:
        with _ckpt_mutex:
            file_name_list = list(os.path.splitext(ckpt_file_name))
            file_name_list[1] = file_name_list[1].replace(f".{format}", ".tmp")
            tmp_name = ''.join(file_name_list)
            if _ckpt_fs.backend == "mindio":
                tmp_name = ckpt_file_name
            if os.path.exists(ckpt_file_name):
                os.chmod(ckpt_file_name, stat.S_IWUSR)
                os.remove(ckpt_file_name)
            if os.path.exists(tmp_name):
                os.chmod(tmp_name, stat.S_IWUSR)
                os.remove(tmp_name)
            if format == "ckpt":
                ckpt_total_io_time = 0
                with _ckpt_fs.create(tmp_name, *_ckpt_fs.create_args) as f:
                    plain_data = None
                    if enc_key is not None:
                        plain_data = BytesIO()

                    crc_num = 0
                    for name, value in data_list.items():
                        if value[0] == "mapparameter":
                            _write_mapparameter(name, value, f, map_param_inc)
                            continue
                        if value[0] == "offload_parameter":
                            new_value = value[1:]
                            new_value[2] = value[3]
                            _write_parameter_bytes_data(name, new_value, f, enc_key, plain_data, ckpt_total_io_time)
                            _offload_if_config(value[3])
                            continue
                        if value[1] == "str":
                            crc_num, ckpt_total_io_time = _write_parameter_data(name, value, f, enc_key, plain_data,
                                                                                crc_num, crc_check,
                                                                                ckpt_total_io_time)
                            continue
                        if isinstance(value[2], np.ndarray):
                            crc_num, ckpt_total_io_time = _write_parameter_data(name, value, f, enc_key, plain_data,
                                                                                crc_num, crc_check,
                                                                                ckpt_total_io_time)
                            continue

                        crc_num, ckpt_total_io_time = _write_parameter_bytes_data(name, value, f, enc_key, plain_data,
                                                                                  crc_num, crc_check,
                                                                                  ckpt_total_io_time)

                    if enc_key is not None:
                        plain_data.seek(0)
                        max_block_size = ENCRYPT_BLOCK_SIZE * 1024
                        block_data = plain_data.read(max_block_size)
                        while block_data:
                            f.write(_encrypt(block_data, len(block_data), enc_key, len(enc_key), enc_mode))
                            block_data = plain_data.read(max_block_size)
                    if crc_check:
                        f.write('crc_num'.encode() + crc_num.to_bytes(10, byteorder='big'))
                vlog_print("1", "ME", __file__, sys._getframe().f_lineno,
                           f"Save ckpt io cost time:{ckpt_total_io_time}.")

            elif format == "safetensors":
                save_dict = {}
                crc_num = 0
                meta_data = {"format": "ms"}
                if remove_redundancy is not None and isinstance(remove_redundancy, bool):
                    meta_data["remove_redundancy"] = str(remove_redundancy)
                for name in sorted(data_list.keys()):
                    value = data_list[name]
                    if isinstance(value[2], np.ndarray):
                        if value[1] == str(mstype.qint4x2):
                            meta_data[name] = str(mstype.qint4x2)
                        save_dict[name] = value[2]
                    else:
                        if value[2].dtype == mstype.qint4x2:
                            meta_data[name] = str(mstype.qint4x2)
                        save_dict[name] = value[2].asnumpy()

                    if crc_check:
                        crc_num = binascii.crc32(bytes(name, encoding='utf-8'), crc_num)
                        crc_num = binascii.crc32(
                            bytes(save_dict[name]), crc_num)
                safetensors_save_time_start = time.time()
                if crc_check:
                    meta_data.update({"crc_num": str(crc_num)})
                if save_dict:
                    save_file(save_dict, tmp_name, metadata=meta_data)
                else:
                    save_file(save_dict, tmp_name)

                safetensors_save_time_end = time.time()
                cost_time = safetensors_save_time_end - safetensors_save_time_start
                vlog_print("1", "ME", __file__, sys._getframe().f_lineno, f"Save safetensors io cost time:{cost_time}.")
            if not os.path.exists(tmp_name):
                logger.warning(f"Rename failed, can't find {tmp_name}, it is possible that multiple processes have "
                               f"simultaneously modified a file.")
            elif _ckpt_fs.backend != "mindio":
                os.rename(tmp_name, ckpt_file_name)
                os.chmod(ckpt_file_name, stat.S_IRUSR)
    except BaseException as e:
        logger.critical("Failed to save the checkpoint file %s. Maybe don't have the permission to write files, "
                        "or the disk space is insufficient and so on.", ckpt_file_name)
        raise e


def _write_parameter_data(name, value, f, enc_key, plain_data, crc_num=0, crc_check=False, ckpt_total_io_time=0):
    """Write parameter data into protobuf file."""
    data_size = value[2].nbytes / 1024
    if data_size > SLICE_SIZE:
        slice_count = math.ceil(data_size / SLICE_SIZE)
        param_slice_list = np.array_split(value[2], slice_count)
    else:
        param_slice_list = [value[2]]

    for param_slice in param_slice_list:
        checkpoint_list = Checkpoint()
        param_value = checkpoint_list.value.add()
        param_value.tag = name
        param_tensor = param_value.tensor
        param_tensor.dims.extend(value[0])
        param_tensor.tensor_type = value[1]
        param_tensor.tensor_content = param_slice.tobytes()

        if enc_key is None:
            output_data = checkpoint_list.SerializeToString()
            if crc_check:
                crc_num = binascii.crc32(output_data, crc_num)
            io_start_time = time.time()
            f.write(output_data)
            io_end_time = time.time()
            io_cost_time = io_end_time - io_start_time
            ckpt_total_io_time += io_cost_time
        else:
            plain_data.write(checkpoint_list.SerializeToString())

    return crc_num, ckpt_total_io_time


def _write_parameter_bytes_data(name, value, f, enc_key, plain_data, crc_num=0, crc_check=False, ckpt_total_io_time=0):
    """Write parameter bytes data into protobuf file."""
    bytes_value = value[2].get_bytes()
    chunk_size = 1024 * SLICE_SIZE

    for i in range(0, len(bytes_value), chunk_size):
        checkpoint_list = Checkpoint()
        param_value = checkpoint_list.value.add()
        param_value.tag = name
        param_tensor = param_value.tensor
        param_tensor.dims.extend(value[0])
        param_tensor.tensor_type = value[1]
        param_tensor.tensor_content = bytes_value[i:i + chunk_size]

        if enc_key is None:
            output_data = checkpoint_list.SerializeToString()
            if crc_check:
                crc_num = binascii.crc32(output_data, crc_num)
            io_start_time = time.time()
            f.write(output_data)
            io_end_time = time.time()
            io_cost_time = io_end_time - io_start_time
            ckpt_total_io_time += io_cost_time
        else:
            plain_data.write(checkpoint_list.SerializeToString())

    return crc_num, ckpt_total_io_time


def _write_mapparameter(name, value, f, map_param_inc=False):
    """Write map parameter into protobuf file."""
    while True:
        logger.info("Checkpoint save map_parameter.")
        data_map_slice = value[1].export_slice_data(map_param_inc)
        checkpoint_list = Checkpoint()
        param_value = checkpoint_list.value.add()
        param_value.tag = name
        map_tensor = param_value.maptensor
        for numpy_data in data_map_slice[:3]:
            tensor_pro = map_tensor.tensor.add()
            tensor_pro.dims.extend(numpy_data.shape)
            tensor_pro.tensor_type = str(numpy_data.dtype)
            tensor_pro.tensor_content = numpy_data.reshape(-1).tobytes()
        f.write(checkpoint_list.SerializeToString())
        if data_map_slice[3]:
            break


def _check_save_obj_and_ckpt_file_name(save_obj, ckpt_file_name, format):
    """Check save_obj and ckpt_file_name for save_checkpoint."""
    if format not in ["safetensors", "ckpt"]:
        raise ValueError(f"For 'save_checkpoint', the format must be "
                         f"'safetensors' or 'ckpt', but got {format}.")
    if not isinstance(save_obj, (nn.Cell, list, dict)):
        raise TypeError("For 'save_checkpoint', the parameter 'save_obj' must be nn.Cell, list or dict, "
                        "but got {}.".format(type(save_obj)))
    if not isinstance(ckpt_file_name, str):
        raise TypeError("For 'save_checkpoint', the parameter {} for checkpoint file name is invalid,"
                        "'ckpt_file_name' must be "
                        "string, but got {}.".format(ckpt_file_name, type(ckpt_file_name)))
    ckpt_file_name = os.path.realpath(ckpt_file_name)
    if os.path.isdir(ckpt_file_name):
        raise IsADirectoryError("For 'save_checkpoint', the parameter `ckpt_file_name`: {} is a directory, "
                                "it must be a file name.".format(ckpt_file_name))
    if not ckpt_file_name.endswith(format):
        ckpt_file_name += f".{format}"
    return ckpt_file_name


def _check_load_checkpoint_unsupported_param(format, dec_key, dec_mode):
    """check load checkpoint unsupported param"""
    if format != "safetensors":
        return
    default_params = {
        "dec_key": None,
        "dec_mode": "AES-GCM",
    }
    for param_name, default_value in default_params.items():
        current_value = locals()[param_name]
        if current_value != default_value:
            raise ValueError(f"For 'load_checkpoint', when format is 'safetensors', the parameter '{param_name}' must "
                             f"be set to default value '{default_value}', but got '{current_value}'.")


def _check_save_checkpoint_unsupported_param(format, enc_key, enc_mode, map_param_inc=False, global_step_num=None):
    """check save checkpoint unsupported param"""
    if format != "safetensors":
        return
    default_params = {
        "enc_key": None,
        "enc_mode": "AES-GCM",
        "map_param_inc": False,
        "global_step_num": None
    }
    for param_name, default_value in default_params.items():
        current_value = locals()[param_name]
        if current_value != default_value:
            raise ValueError(f"For 'save_checkpoint', when format is 'safetensors', the parameter '{param_name}' must "
                             f"be set to default value '{default_value}', but got '{current_value}'.")


def _check_async_save(async_save):
    """Check async_save for save_checkpoint."""
    if not isinstance(async_save, (bool, str)):
        raise TypeError("For 'save_checkpoint', the parameter 'async_save' must be bool or str, "
                        "but got {}.".format(type(async_save)))
    if isinstance(async_save, str):
        if async_save not in ("process", "thread"):
            raise ValueError("For 'save_checkpoint', the argument 'async_save' can only be 'process' or 'thread',"
                             "but got {}.".format(async_save))
    return async_save


def _async_process_save(ckpt_file_name, data_list, enc_key=None, enc_mode="AES-GCM", map_param_inc=False,
                        crc_check=False, format="ckpt", cond=None, remove_redundancy=None):
    """Check whether the process is pulled up successfully, execute the process of saving checkpoint into file."""
    with cond:
        cond.notify()
    _exec_save(ckpt_file_name, data_list, enc_key, enc_mode, map_param_inc, crc_check, format, remove_redundancy)


@_mstx_range_decorator("save_checkpoint", domain="model_preparation")
def save_checkpoint(save_obj, ckpt_file_name, integrated_save=True,
                    async_save=False, append_dict=None, enc_key=None, enc_mode="AES-GCM", choice_func=None,
                    crc_check=False, format="ckpt", **kwargs):
    r"""
    Save checkpoint to a specified file.

    Note:
        The `enc_mode` and `crc_check` parameters are mutually exclusive and cannot be configured simultaneously.

    Args:
        save_obj (Union[Cell, list, dict]): The object to be saved. The data type can be :class:`mindspore.nn.Cell`,
            list, or dict.

            - If a list, it can be the returned value of `Cell.trainable_params()`, or a list of dict
              elements(each element is a dictionary, like [{"name": param_name, "data": param_data},...], the type of
              `param_name` must be string, and the type of `param_data` must be parameter or Tensor).
            - If dict, it can be the returned value of :func:`mindspore.load_checkpoint`.

        ckpt_file_name (str): Checkpoint file name. If the file name already exists, it will be overwritten.
        integrated_save (bool): Whether to integrated save in automatic model parallel scene. Default: ``True`` .
        async_save (Union[bool, str], optional): Whether to use asynchronous saving of the checkpoint file or
                                    safetensors file, if True, the asynchronous thread is used by default. If the type
                                    is string, the method of asynchronous saving, it can be "process" or "thread".
                                    Default: ``False`` .
        append_dict (dict): Additional information that needs to be saved. The key of dict must be str, the value
                            of dict must be one of int, float, bool, string, Parameter or Tensor. Default: ``None`` .
        enc_key (Union[None, bytes]): Byte type key used for encryption. If the value is ``None`` , the encryption
                                      is not required. Default: ``None`` .
        enc_mode (str): This parameter is valid only when enc_key is not set to ``None`` . Specifies the encryption
                        mode, currently supports ``"AES-GCM"`` and ``"AES-CBC"`` and ``"SM4-CBC"`` .
                        Default: ``"AES-GCM"`` .
        choice_func (function) : A function for saving custom selected parameters. The input value of `choice_func` is
                                 a parameter name in string type, and the returned value is a bool.
                                 Default: ``None`` .

                                 - If returns ``True`` , the Parameter that matching the custom condition will be saved.
                                 - If returns ``False`` , the Parameter that not matching the custom condition will not
                                   be saved.

        crc_check (bool) : Whether to perform crc32 calculation when saving checkpoint and save the calculation
            result to the file. Default: ``False`` .
        format (str): Format of the output file, can be "ckpt" or "safetensors". Default: "ckpt".
        kwargs (dict): Configuration options dictionary.

    Raises:
        TypeError: If the parameter `save_obj` is not :class:`mindspore.nn.Cell` , list or dict type.
        TypeError: If the parameter `integrated_save` is not bool type.
        TypeError: If the parameter `ckpt_file_name` is not string type.
        TypeError: If the parameter `async_save` is not bool or string type.
        ValueError: If the parameter `async_save` is string type but not in ["process", "thread"].

    Examples:
        >>> import mindspore as ms
        >>>
        >>> # Define the network structure of LeNet5. Refer to
        >>> # https://gitee.com/mindspore/docs/blob/master/docs/mindspore/code/lenet.py
        >>> net = LeNet5()
        >>> ms.save_checkpoint(net, "./lenet.ckpt",
        ...                    choice_func=lambda x: x.startswith("conv") and not x.startswith("conv1"))
        >>> param_dict1 = ms.load_checkpoint("./lenet.ckpt")
        >>> print(param_dict1)
        {'conv2.weight': Parameter (name=conv2.weight, shape=(16, 6, 5, 5), dtype=Float32, requires_grad=True)}
        >>> params_list = net.trainable_params()
        >>> ms.save_checkpoint(params_list, "./lenet_list.ckpt",
        ...                    choice_func=lambda x: x.startswith("conv") and not x.startswith("conv2"))
        >>> param_dict2 = ms.load_checkpoint("./lenet_list.ckpt")
        >>> print(param_dict2)
        {'conv1.weight': Parameter (name=conv1.weight, shape=(6, 1, 5, 5), dtype=Float32, requires_grad=True)}
        >>> ms.save_checkpoint(param_dict2, "./lenet_dict.ckpt")
        >>> param_dict3 = ms.load_checkpoint("./lenet_dict.ckpt")
        >>> print(param_dict3)
        {'conv1.weight': Parameter (name=conv1.weight, shape=(6, 1, 5, 5), dtype=Float32, requires_grad=True)}

    Tutorial Examples:
        - `Saving and Loading the Model - Saving and Loading the Model Weight
          <https://mindspore.cn/tutorials/en/master/beginner/save_load.html#saving-and-loading-the-model-weight>`_
    """
    start_save_time = time.time()
    _ensure_ckpt_fs_initialized()
    ckpt_file_name = _check_save_obj_and_ckpt_file_name(save_obj, ckpt_file_name, format)
    integrated_save = Validator.check_bool(integrated_save)
    async_save = _check_async_save(async_save)
    append_dict = _check_append_dict(append_dict)
    enc_key = Validator.check_isinstance('enc_key', enc_key, (type(None), bytes))
    enc_mode = Validator.check_isinstance('enc_mode', enc_mode, str)
    crc_check = Validator.check_isinstance('crc_check', crc_check, bool)
    map_param_inc = kwargs.get('incremental', False)
    logger.info("Execute the process of saving checkpoint files.")
    global_step_num = kwargs.get('global_step_num', None)
    remove_redundancy = kwargs.get('remove_redundancy', None)
    remove_redundancy = Validator.check_isinstance("remove_redundancy", remove_redundancy, (type(None), bool))
    _check_save_checkpoint_unsupported_param(format, enc_key, enc_mode, map_param_inc, global_step_num)

    if append_dict and "__exception_save__" in append_dict:
        s1 = mindspore.hal.Stream()
        with mindspore.hal.StreamCtx(s1):
            save_obj = _convert_save_obj_to_param_list(save_obj, integrated_save, append_dict, choice_func)
            for k_name, value in append_dict.items():
                if isinstance(value, (Tensor, Parameter)):
                    append_dict[k_name] = Tensor(Tensor_.move_to(value, "CPU", False))
        s1.synchronize()
    else:
        save_obj = _convert_save_obj_to_param_list(save_obj, integrated_save, append_dict, choice_func)

    if append_dict:
        if "__exception_save__" in append_dict:
            del append_dict["__exception_save__"]
        append_info_list = []
        for k_name, value in append_dict.items():
            if isinstance(value, Generator):
                value = value.get_state()
            elif not isinstance(value, str):
                value = Tensor(value)
            append_info_list.append({"name": k_name, "data": value})
        save_obj.extend(append_info_list)

    data_list = OrderedDict()
    data_list_np = OrderedDict()
    with _ckpt_mutex:
        for param in save_obj:
            key = param["name"]
            data_list[key] = []
            data_list_np[key] = []
            if isinstance(param["data"], MapParameter):
                data_list[param["name"]].append("mapparameter")
                data_list[param["name"]].append(param["data"])
                continue
            if isinstance(param["data"], list):
                if param["data"][0] == "offload_parameter":
                    data_list[key].append("offload_parameter")
                    _save_param_list_data(data_list, key, param)

            if isinstance(param["data"], str):
                if os.getenv("AITURBO") == "1":
                    data_list_np[key].append(np.array(param["data"]))
                    if crc_check:
                        bytes_value = data_list_np[key][0].tobytes()
                        data_list_np[key].append(binascii.crc32(bytes_value))
                else:
                    data_list[key].append([0])
                    data_list[key].append('str')
                    data = np.array(param["data"])
                    data_list[key].append(data)
            else:
                if isinstance(param["data"], Parameter):
                    param["data"].init_data()
                if os.getenv("AITURBO") == "1":
                    data_list_np[key].append(param["data"].asnumpy())
                    if crc_check:
                        bytes_value = data_list_np[key][0].tobytes()
                        data_list_np[key].append(binascii.crc32(bytes_value))
                else:
                    dims = []
                    for dim in param['data'].shape:
                        dims.append(dim)
                    data_list[key].append(dims)
                    tensor_type = str(param["data"].dtype)
                    data_list[key].append(tensor_type)
                    data = param["data"] if async_save is False else param["data"].asnumpy()
                    data_list[key].append(data)

    from mindspore.profiler import mstx
    range_id = mstx.range_start('save_checkpoint', None)
    if os.getenv("AITURBO") == "1":
        from aiturbo.checkpoint import aiturbo_mindspore as aiturbo
        ckpt_name = os.path.basename(ckpt_file_name)
        aiturbo.save_ckpt(ckpt_name, global_step_num, data_list_np, crc_check)
    elif async_save:
        if async_save == "process":
            if sys.platform.startswith("win"):
                logger.warining("The Win platform currently does not support asynchronous process saving of ckpt, "
                                "so serial saving of ckpt is used now.")
                _exec_save(ckpt_file_name, data_list, enc_key, enc_mode, map_param_inc, crc_check, format)
            else:
                _wait_async_process_save_ckpt()
                ctx = mp.get_context("fork")
                cond = ctx.Condition()
                process_flag = True
                while process_flag:
                    process = ctx.Process(target=_async_process_save,
                                          args=(ckpt_file_name, data_list, enc_key, enc_mode, map_param_inc, crc_check,
                                                format, cond, remove_redundancy), daemon=True, name="asyn_save_ckpt")
                    process.start()
                    with cond:
                        wait_flag = cond.wait(timeout=5)
                        if not wait_flag:
                            logger.warning("Async save process fails to create. will kill and recreate")
                            process.kill()
                        else:
                            process_flag = False
        else:
            data_copy = copy.deepcopy(data_list)
            _wait_async_thread_save_ckpt()
            thr = Thread(target=_exec_save,
                         args=(ckpt_file_name, data_copy, enc_key, enc_mode, map_param_inc, crc_check, format,
                               remove_redundancy),
                         name="asyn_save_ckpt")
            thr.start()
    else:
        _exec_save(ckpt_file_name, data_list, enc_key, enc_mode, map_param_inc, crc_check, format, remove_redundancy)

    mstx.range_end(range_id)
    logger.info("Saving checkpoint process is finished.")
    end_save_time = time.time()
    save_checkpoint_cost_time = end_save_time - start_save_time
    vlog_print("1", "ME", __file__, sys._getframe().f_lineno, f"Save checkpoint cost time {save_checkpoint_cost_time}.")


def _handle_shared_param_for_pipeline_parallel(save_obj):
    """ Remove shared param for save_obj """
    filtered_save_obj = []
    for param_dict in save_obj:
        cur_param = param_dict['data']
        if isinstance(cur_param, Parameter):
            if not cur_param.param_info.is_pipeline_shared_param:
                filtered_save_obj.append(param_dict)
        else:
            filtered_save_obj.append(param_dict)
    return filtered_save_obj


def _is_auto_parallel_mode(save_obj):
    """Check if in auto parallel mode by verifying parameter initialization."""
    for _, param in save_obj.parameters_and_names():
        if param.param_info.is_param_init:
            return True
    return False


def _convert_list_to_param_list(save_obj, choice_func):
    """Convert a list of Parameter to param_list."""
    param_list = []
    if not save_obj:
        return param_list
    if isinstance(save_obj[0], dict):
        for param in save_obj:
            if isinstance(param, dict) and "name" in param and "data" in param:
                if not isinstance(param["name"], str):
                    raise TypeError(f"For save_checkpoint, when save_obj is a list of dict items, the name in dict "
                                    f"should be string, but got {type(param['name'])}.")
                if not isinstance(param["data"], Tensor):
                    raise TypeError(f"For save_checkpoint, when save_obj is a list of dict items, the data in dict "
                                    f"should be parameter, but got {type(param['data'])}.")
                if choice_func is not None and not choice_func(param["name"]):
                    continue
                each_param = {"name": param["name"], "data": param["data"]}
                param_list.append(each_param)
            else:
                raise TypeError(f"For save_checkpoint, save_obj should be a list of dict items, and the dict should "
                                f"have key values 'name' and 'value', but got {type(param)} and {param}.")
    else:
        for param in save_obj:
            if isinstance(param, Parameter):
                if choice_func is not None and not choice_func(param.name):
                    continue
                each_param = {"name": param.name, "data": param}
                param_list.append(each_param)
            else:
                raise TypeError(f"For save_checkpoint, when save_obj is made up by list of Parameter,"
                                f"the param should be parameter, but got {type(param)}")
    return param_list


def _convert_dict_to_param_dict(save_obj, choice_func):
    """Convert a dict of Parameter to param_list."""
    param_list = []
    for (key, value) in save_obj.items():
        if isinstance(key, str):
            if choice_func is not None and not choice_func(key):
                continue
            if isinstance(value, np.ndarray):
                each_param = {"name": key, "data": Parameter(Tensor.from_numpy(value))}
            if isinstance(value, (Parameter, str)) or _is_buffer_type(value):
                each_param = {"name": key, "data": value}
            param_list.append(each_param)
        else:
            raise TypeError(f"For save_checkpoint, when save_obj is made up by dict, the key should be str and"
                            f"value should be Parameter, but got the type of key is {type(key)} and"
                            f"the type of value is {type(value)}")
    return param_list


def _convert_cell_param_and_names_to_dict(save_obj, choice_func, is_parallel_mode):
    """Convert cell.parameters_and_names to OrderedDict."""
    param_dict = OrderedDict()
    is_graph_mode = context.get_context('mode') == context.GRAPH_MODE
    for _, param in save_obj.parameters_and_names():
        # All parameters are initialized immediately under PyNative mode, skip this judgement.
        if param.param_info.is_pipeline_shared_param:
            continue
        if is_parallel_mode and is_graph_mode and (not param.sliced or param.has_init):
            continue
        if choice_func is not None and not choice_func(param.name):
            continue
        # Add suffix for cache_enabled parameter, and then parameter can carry key info.
        # Notice that suffix needs be removed when loading into net.
        if param.cache_enable:
            param_dict[param.name + ".__param_key__" + str(param.key)] = param
        else:
            param_dict[param.name] = param
    return param_dict


def _convert_cell_to_param_list(save_obj, integrated_save, append_dict, choice_func):
    """Convert nn.Cell to param_list."""
    sync_pipeline_shared_parameters(save_obj)
    param_list = []
    parameter_layout_dict = save_obj.parameter_layout_dict
    is_parallel_mode = _is_auto_parallel_mode(save_obj)
    if is_parallel_mode and not parameter_layout_dict:
        parameter_layout_dict = _get_parameter_layout()
    if not is_parallel_mode:
        save_obj.init_parameters_data()
    param_dict = _convert_cell_param_and_names_to_dict(save_obj, choice_func, is_parallel_mode)
    enable_ckpt_d2h_sync = os.getenv('MS_ENABLE_D2H_ASYNC') == '1'
    param_snapshot = _get_snapshot_params() if enable_ckpt_d2h_sync else {}
    for (key, value) in param_dict.items():
        each_param = {"name": key}
        if isinstance(value, MapParameter):
            each_param["data"] = value
            param_list.append(each_param)
            continue

        if value.data.offload_file_path() != "":
            # list save offload data: [Param, shape, type, param.key]
            param_data = ["offload_parameter"]
            param_tensor = value.data
            if key in parameter_layout_dict:
                param_tensor = _get_merged_param_data(save_obj, parameter_layout_dict, key, param_tensor,
                                                      integrated_save)
            param_data.append(param_tensor)
            param_data.append(param_tensor.shape)
            param_data.append(str(param_tensor.dtype))
            param_data.append(value.key)
        else:
            if append_dict and "__exception_save__" in append_dict:
                param_data = Tensor(Tensor_.move_to(value, "CPU", False))
            else:
                # when enable MS_ENABLE_D2H_ASYNC=1, fetch param from sanpshot in priority
                param_data = param_snapshot.get(key, Tensor(value.data))

            # in automatic model parallel scenario, some parameters were split to all the devices,
            # which should be combined before saving
            if key in parameter_layout_dict:
                param_data = _get_merged_param_data(save_obj, parameter_layout_dict, key, param_data,
                                                    integrated_save)

        each_param["data"] = param_data
        param_list.append(each_param)
    return param_list


def _convert_save_obj_to_param_list(save_obj, integrated_save, append_dict, choice_func):
    """Convert a save_obj to param_list."""
    if isinstance(save_obj, (list, dict)):
        if isinstance(save_obj, list):
            save_obj = _convert_list_to_param_list(save_obj, choice_func)

        if isinstance(save_obj, dict):
            save_obj = _convert_dict_to_param_dict(save_obj, choice_func)

        return _handle_shared_param_for_pipeline_parallel(save_obj)

    if isinstance(save_obj, nn.Cell):
        return _convert_cell_to_param_list(save_obj, integrated_save, append_dict, choice_func)

    raise TypeError("For 'save_checkpoint', the argument 'save_obj' must be list、dict or nn.cell, "
                    "but got {}.".format(type(save_obj)))


def _save_param_list_data(data_list, key, param):
    """Save persistent data into save_obj."""
    dims = []
    for dim in param['data'][2]:
        dims.append(dim)
    data_list[key].append(dims)
    data_list[key].append(param['data'][3])
    data_list[key].append(param['data'][1])
    data_list[key].append(param['data'][4])


def _check_append_dict(append_dict):
    """Check the argument append_dict for save_checkpoint."""
    if append_dict is None:
        return append_dict
    if not isinstance(append_dict, dict):
        raise TypeError("For 'save_checkpoint', the argument 'append_dict' must be dict, but got "
                        "{}.".format(type(append_dict)))
    for key, value in append_dict.items():
        if not isinstance(key, str) or not isinstance(value, (int, float, bool, str, Parameter, Tensor, Generator)):
            raise TypeError(f"For 'save_checkpoint', the type of dict 'append_info' must be key: string, "
                            f"value: int, float, bool or Generator, but got key: {type(key)}, value: {type(value)}")
    return append_dict


def _is_buffer_type(value):
    if isinstance(value, Tensor) and getattr(value, "_is_buffer", False):
        return True
    return False


def load(file_name, **kwargs):
    """
    Load MindIR.

    The returned object can be executed by a `GraphCell`, see class :class:`mindspore.nn.GraphCell` for more details.

    Args:
        file_name (str): MindIR file name.

        kwargs (dict): Configuration options dictionary.

            - dec_key (bytes): Byte-type key used for decryption. The valid length is 16, 24, or 32.
            - dec_mode (Union[str, function], optional):
              Specifies the decryption mode, to take effect when dec_key is set.

              - Option: 'AES-GCM', 'AES-CBC', 'SM4-CBC' or customized decryption. Default: ``'AES-GCM'``.
              - For details of using the customized decryption, please check the `tutorial
                <https://mindspore.cn/mindarmour/docs/en/master/model_encrypt_protection.html>`_.

    Returns:
        GraphCell, a compiled graph that can executed by `GraphCell`.

    Raises:
        NotImplementedError: Dynamic model structure obfuscation is no longer supported.
        ValueError: MindIR file does not exist or `file_name` is not a string.
        RuntimeError: Failed to parse MindIR file.

    Examples:
        >>> import numpy as np
        >>> import mindspore as ms
        >>> import mindspore.nn as nn
        >>> from mindspore import Tensor
        >>> from mindspore import context
        >>> context.set_context(mode=context.GRAPH_MODE)
        >>>
        >>> net = nn.Conv2d(1, 1, kernel_size=3, weight_init="ones")
        >>> input_tensor = Tensor(np.ones([1, 1, 3, 3]).astype(np.float32))
        >>> ms.export(net, input_tensor, file_name="net", file_format="MINDIR")
        >>> graph = ms.load("net.mindir")
        >>> net = nn.GraphCell(graph)
        >>> output = net(input_tensor)
        >>> print(output)
        [[[[4. 6. 4.]
           [6. 9. 6.]
           [4. 6. 4.]]]]

    Tutorial Examples:
        - `Saving and Loading the Model - Saving and Loading MindIR
          <https://mindspore.cn/tutorials/en/master/beginner/save_load.html#saving-and-loading-mindir>`_
    """
    if 'obf_func' in kwargs.keys():
        raise NotImplementedError("Dynamic model structure obfuscation is no longer supported.")
    if not isinstance(file_name, str):
        raise ValueError("For 'load', the argument 'file_name' must be string, but "
                         "got {}.".format(type(file_name)))
    if not file_name.endswith(".mindir"):
        raise ValueError("For 'load', the argument 'file_name'(MindIR file) should end with '.mindir', "
                         "please input the correct 'file_name'.")
    if not os.path.exists(file_name):
        raise ValueError("For 'load', the argument 'file_name'(MindIR file) does not exist, "
                         "please check whether the 'file_name' is correct.")
    file_name = os.path.realpath(file_name)

    logger.info("Execute the process of loading mindir.")
    if 'dec_key' in kwargs.keys():
        dec_key = Validator.check_isinstance('dec_key', kwargs.get('dec_key'), bytes)
        dec_mode = "AES-GCM"
        dec_func = None
        if 'dec_mode' in kwargs.keys():
            if callable(kwargs.get('dec_mode')):
                dec_mode = "Customized"
                dec_func = kwargs.get('dec_mode')
            else:
                dec_mode = Validator.check_isinstance('dec_mode', kwargs.get('dec_mode'), str)
        graph = load_mindir(file_name, dec_key=dec_key, key_len=len(dec_key), dec_mode=dec_mode,
                            decrypt=dec_func)
    else:
        graph = load_mindir(file_name)

    if graph is None:
        if _is_cipher_file(file_name):
            raise RuntimeError("Load MindIR failed. The file may be encrypted and decrypt failed, you "
                               "can check whether the values of the arguments 'dec_key' and 'dec_mode'"
                               " are the same as when exported MindIR file, or check the file integrity.")
        raise RuntimeError("Load MindIR failed.")
    return graph


def export_split_mindir(file_name, device_num=8, rank_id=0, dynamic=True, sapp=True):
    """
    Auto Split MindIR.

    The returned object can be executed by a `GraphCell`, see class :class:`mindspore.nn.GraphCell` for more details.

    Args:
        file_name (str): MindIR file name.
        device_num (int): device number. Default: '8'.
        rank_id (int): rank id. Default: '0'.
        dynamic (bool): Indicates whether the model is a dynamic shape mindir model. Default: 'True'.
        sapp (bool): Indicates whether to automatically generate split strategy through SAPP. Default: 'True'.

    Raises:
        ValueError: MindIR file does not exist or `file_name` is not a string.
        RuntimeError: Failed to split MindIR file.

    Examples:
        >>> import mindspore as ms
        >>> context.set_context(mode=context.GRAPH_MODE)
        >>>
        >>> ms.export_split_mindir("net.mindir", device_num=8, rank_id=0)

    """
    if not isinstance(file_name, str):
        raise ValueError("For 'Split MindIR', the argument 'file_name' must be string, but "
                         "got {}.".format(type(file_name)))
    if not file_name.endswith(".mindir"):
        raise ValueError("For 'Split MindIR', the argument 'file_name'(MindIR file) should end with '.mindir', "
                         "please input the correct 'file_name'.")
    if not os.path.exists(file_name):
        raise ValueError("For 'Split MindIR', the argument 'file_name'(MindIR file) does not exist, "
                         "please check whether the 'file_name' is correct.")
    file_name = os.path.realpath(file_name)

    logger.info("Execute the process of export and split mindir.")
    dynamic = True
    if dynamic:
        graph = split_dynamic_mindir(file_name, device_num, rank_id, sapp)
    else:
        graph = split_mindir(file_name)

    if graph is None:
        if _is_cipher_file(file_name):
            raise RuntimeError("Export and split MindIR failed. The file may be encrypted and decrypt failed, you "
                               "can check whether the values of the arguments 'dec_key' and 'dec_mode'"
                               " are the same as when exported MindIR file, or check the file integrity.")
        raise RuntimeError("Export and split MindIR failed.")
    return graph


def _check_param_type(param_config, key, target_type, requested):
    """check type of parameters"""
    if key in param_config:
        if not isinstance(param_config[key], target_type):
            raise TypeError("The type of {} must be {}, but got {}.".format(key, target_type, type(param_config[key])))
        return param_config[key]
    if requested:
        raise ValueError("The parameter {} is requested, but not got.".format(key))
    return None


def _check_remove_redundancy(remove_redundancy, f):
    """Check whether remove_redundancy is consistent with the safetensors file."""
    if f.metadata() is not None and "remove_redundancy" in f.metadata().keys():
        if f.metadata()["remove_redundancy"] == "True" and not remove_redundancy:
            logger.warning("For 'load_checkpoint', the safetensors file is deduplicated, "
                           "but remove_redundancy is set to False.")
            return True
        if f.metadata()["remove_redundancy"] == "False" and remove_redundancy:
            logger.warning("For 'load_checkpoint', the safetensors file is non-deduplicated, "
                           "but remove_redundancy is set to True.")
            return False
    return remove_redundancy


def _load_into_param_dict(ckpt_file_name, parameter_dict, specify_prefix, filter_prefix, choice_func, dec_key,
                          dec_mode, crc_check, format, remove_redundancy):
    """load parameter into parameter_dict"""
    ckpt_file_name = _check_ckpt_file_name(ckpt_file_name, format)
    if format == "safetensors":
        with _fast_safe_open(ckpt_file_name, framework='np') as f:
            cal_crc_num = 0
            total_io_cost_time = 0
            for k in sorted(f.keys()):
                if crc_check:
                    cal_crc_num = binascii.crc32(bytes(k, encoding='utf-8'), cal_crc_num)
                    cal_crc_num = binascii.crc32(bytes(f.get_tensor(k)), cal_crc_num)
                if choice_func is not None and not choice_func(k):
                    continue
                io_start_time = time.time()
                value = f.get_tensor(k)
                io_end_time = time.time()
                io_cost_time = io_end_time - io_start_time
                total_io_cost_time += io_cost_time
                if f.metadata() is not None and k in f.metadata().keys():
                    sf_dtype = f.metadata()[k]
                    ms_dtype = safetensors_to_mstype[sf_dtype]
                    parameter_dict[k] = Parameter(Tensor(value, dtype=ms_dtype))
                else:
                    parameter_dict[k] = Parameter(Tensor.from_numpy(value))
            remove_redundancy = _check_remove_redundancy(remove_redundancy, f)
            vlog_print("1", "ME", __file__, sys._getframe().f_lineno,
                       f"Load safetensors io cost time:{total_io_cost_time}.")
            if crc_check:
                if f.metadata() is None or f.metadata().get("crc_num") is None:
                    logger.warning(
                        "For 'load_checkpoint', the safetensors file do not contain the crc code, "
                        "please check the file.")
                else:
                    crc_num = int(f.metadata()["crc_num"])
                    if cal_crc_num != crc_num:
                        raise ValueError("For 'load_checkpoint', the crc check has failed. "
                                         "Please check whether the ckpt file is damaged.")
        return remove_redundancy
    checkpoint_list = _parse_ckpt_proto(ckpt_file_name, dec_key, dec_mode, crc_check)
    try:
        param_data_list = []
        map_data_list = [[], [], []]
        map_shape_list = [0, 0, 0]
        if specify_prefix:
            logger.warning("For load_checkpoint, this parameter `specity_prefix` will be deprecated, "
                           "please use `choice_func` instead.")
        if filter_prefix:
            logger.warning("For load_checkpoint, this parameter `filter_prefix` will be deprecated, "
                           "please use `choice_func` instead.")
        for element_id, element in enumerate(checkpoint_list.value):
            if not _whether_load_param(specify_prefix, filter_prefix, element.tag):
                continue
            if specify_prefix is None and filter_prefix is None and \
                    choice_func is not None and not choice_func(element.tag):
                continue
            if element.tensor.ByteSize() == 0:
                _load_map_parameter(checkpoint_list, element, element_id, map_data_list, map_shape_list,
                                    parameter_dict)
                if element.tag in parameter_dict:
                    map_data_list = [[], [], []]
                    map_shape_list = [0, 0, 0]
                continue
            data = element.tensor.tensor_content
            data_type = element.tensor.tensor_type
            ms_type = tensor_to_ms_type[data_type]
            param_data_list.append(data)
            if (element_id == len(checkpoint_list.value) - 1) or \
                    (element.tag != checkpoint_list.value[element_id + 1].tag):
                new_data = b"".join(param_data_list)
                param_data_list.clear()
                dims = element.tensor.dims
                if data_type == 'str':
                    str_length = int(len(data) / 4)
                    np_type = "U" + str(str_length)
                    str_value = np.frombuffer(new_data, np_type)
                    parameter_dict[element.tag] = str(str_value[0])
                else:
                    if dims == [0]:
                        dims = []
                    param_data = Tensor_.convert_bytes_to_tensor(new_data, tuple(dims), ms_type)
                    parameter = Parameter(param_data, name=element.tag)
                    parameter_dict[element.tag] = parameter

        logger.info("Loading checkpoint files process is finished.")
        return remove_redundancy

    except BaseException as e:
        logger.critical("Failed to load the checkpoint file '%s'.", ckpt_file_name)
        raise ValueError(e.__str__() + "\nFor 'load_checkpoint', "
                                       "failed to load the checkpoint file {}.".format(ckpt_file_name)) from e


@_mstx_range_decorator("load_checkpoint", domain="model_preparation")
def load_checkpoint(ckpt_file_name, net=None, strict_load=False, filter_prefix=None,
                    dec_key=None, dec_mode="AES-GCM", specify_prefix=None, choice_func=None,
                    crc_check=False, remove_redundancy=False, format="ckpt"):
    """
    Load checkpoint info from a specified file.

    Note:
        - `specify_prefix` and `filter_prefix` are in the process of being deprecated,
          `choice_func` is recommended instead. `specify_prefix` and `filter_prefix` do not affect each other.
          And using either of those two args will override `choice_func` at the same time.
        - If none of the parameters are loaded from checkpoint file, it will throw ValueError.
        - When loading a checkpoint that has removed redundancy, the network should be compiled.
        - When `net` is not None, it will verify whether the `remove_redundancy` parameter matches the
          deduplication flag in the loaded safetensors file. If they are different, load the file according to
          the deduplication flag in the file.

    Args:
        ckpt_file_name (str): Checkpoint file name.
        net (Cell, optional): The network where the parameters will be loaded. Default: ``None`` .
        strict_load (bool, optional): Whether to strict load the parameter into net. If ``False`` , it will load
                            parameter into net when parameter name's suffix in checkpoint file is the same as the
                            parameter in the network. When the types are inconsistent perform type conversion
                            on the parameters of the same type, such as float32 to float16. Default: ``False`` .
        filter_prefix (Union[str, list[str], tuple[str]], optional): Deprecated(see `choice_func`).
            Parameters starting with the filter_prefix will not be loaded. Default: ``None`` .
        dec_key (Union[None, bytes], optional): Byte type key used for decryption. If the value is ``None`` ,
                                      the decryption is not required. Default: ``None`` .
        dec_mode (str, optional): This parameter is valid only when dec_key is not set to ``None`` . Specifies the
                        decryption mode, currently supports ``"AES-GCM"`` and ``"AES-CBC"`` and ``"SM4-CBC"`` .
                        Default: ``"AES-GCM"`` .
        specify_prefix (Union[str, list[str], tuple[str]], optional): Deprecated(see `choice_func`).
            Parameters starting with the specify_prefix will be loaded. Default: ``None`` .
        choice_func (Union[None, function], optional) : Input value of the function is a Parameter name of type string,
            and the return value is a bool. If returns ``True`` , the Parameter
            that matches the custom condition will be loaded. If returns ``False`` , the Parameter that
            matches the custom condition will be removed. Default: ``None`` .
        crc_check (bool, optional) : Whether to perform crc32 validation when loading checkpoint. Default: ``False`` .
        remove_redundancy (bool, optional): Whether to enable loading of checkpoint saved with redundancy removal.
            Redundancy removal refers to eliminating redundant data in data parallelism mode. Default: ``False`` , means
            redundant-free loading is not enabled.
        format (str, optional): Format of the input file, can be "ckpt" or "safetensors". Default: "ckpt".

    Returns:
        Dict, key is parameter name, value is a Parameter or string. When the `append_dict` parameter of
        :func:`mindspore.save_checkpoint` and the `append_info` parameter of :class:`mindspore.train.CheckpointConfig`
        are used to save the checkpoint, `append_dict` and `append_info` are dict types, and their value are string,
        then the return value obtained by loading checkpoint is string, and in other cases the return value is
        Parameter.

    Raises:
        ValueError: Checkpoint file's format is incorrect.
        ValueError: Parameter's dict is None after load checkpoint file.
        TypeError: The type of `specify_prefix` or `filter_prefix` is incorrect.

    Examples:
        >>> import mindspore as ms
        >>>
        >>> ckpt_file_name = "./checkpoint/LeNet5-1_32.ckpt"
        >>> param_dict = ms.load_checkpoint(ckpt_file_name,
        ...                                 choice_func=lambda x: x.startswith("conv") and not x.startswith("conv1"))
        >>> print(param_dict["conv2.weight"])
        Parameter (name=conv2.weight, shape=(16, 6, 5, 5), dtype=Float32, requires_grad=True)
        >>> def func(param_name):
        ...     whether_load = False
        ...     if param_name.startswith("conv"):
        ...         whether_load = True
        ...     if param_name.startswith("conv1"):
        ...         whether_load = False
        ...     return whether_load
        >>> param_dict1 = ms.load_checkpoint(ckpt_file_name, choice_func=func)
        >>> print(param_dict1["conv2.weight"])
        Parameter (name=conv2.weight, shape=(16, 6, 5, 5), dtype=Float32, requires_grad=True)
        >>> def func(param_name):
        ...     whether_load = False
        ...     if param_name.startswith("conv1"):
        ...         whether_load = True
        ...     return whether_load
        >>> param_dict2 = ms.load_checkpoint(ckpt_file_name, choice_func=func)
        >>> print(param_dict2)
        {'conv1.weight': Parameter (name=conv1.weight, shape=(6, 1, 5, 5), dtype=Float32, requires_grad=True)}

    Tutorial Examples:
        - `Saving and Loading the Model - Saving and Loading the Model Weight
          <https://mindspore.cn/tutorials/en/master/beginner/save_load.html#saving-and-loading-the-model-weight>`_
    """
    start_load_time = time.time()
    vlog_print("1", "ME", __file__, sys._getframe().f_lineno, "Begin load checkpoint.")
    _ensure_ckpt_fs_initialized()
    specify_prefix = _check_prefix(specify_prefix)
    filter_prefix = _check_prefix(filter_prefix)
    dec_key = Validator.check_isinstance('dec_key', dec_key, (type(None), bytes))
    dec_mode = Validator.check_isinstance('dec_mode', dec_mode, str)
    crc_check = Validator.check_isinstance('crc_check', crc_check, bool)
    remove_redundancy = Validator.check_isinstance('remove_redundancy', remove_redundancy, bool)
    _check_load_checkpoint_unsupported_param(format, dec_key, dec_mode)
    logger.info("Execute the process of loading checkpoint files.")

    parameter_dict = {}

    if os.getenv("AITURBO") == "1":
        rank_id = get_rank()
        from aiturbo.checkpoint import aiturbo_mindspore as aiturbo
        ckpt_path = os.path.dirname(ckpt_file_name)
        ckpt_name = os.path.basename(ckpt_file_name)
        np_dict = aiturbo.load_ckpt(ckpt_path, ckpt_name, rank_id, crc_check)
        for key, value in np_dict.items():
            if crc_check and len(value) != 2:
                raise ValueError(f"When loading a checkpoint from AITurbo, if CRC check is enabled, "
                                 f"the length of the value must be 2, but got {len(value)}.")
            if isinstance(value, str):
                if crc_check and value[1] != binascii.crc32(np.array(value[0]).tobytes()):
                    raise ValueError("When loading a checkpoint from AITurbo, the value of the string has not "
                                     "passed the CRC check and has been corrupted.")
                parameter_dict[key] = value[0]
            else:
                if crc_check and value[1] != binascii.crc32(value[0].tobytes()):
                    raise ValueError("When loading a checkpoint from AITurbo, the value of the parameter has not "
                                     "passed the CRC check and has been corrupted.")
                parameter_dict[key] = Parameter(Tensor(value[0]), name=key)
    else:
        remove_redundancy = _load_into_param_dict(ckpt_file_name, parameter_dict, specify_prefix, filter_prefix,
                                                  choice_func, dec_key, dec_mode, crc_check, format, remove_redundancy)

    if not parameter_dict:
        raise ValueError("The loaded parameter dict is empty after filter or specify, please check whether "
                         "'filter_prefix' or 'specify_prefix' are set correctly.")

    if _warm_up_host_cache_enabled(parameter_dict):
        (is_worker, net_dict, warm_up_dict) = _warm_up_host_cache(parameter_dict, net)
    if net is not None:
        load_param_into_net(net, parameter_dict, strict_load, remove_redundancy)
    if _warm_up_host_cache_enabled(parameter_dict):
        _warm_up_host_cache_post_process(is_worker, net_dict, warm_up_dict)

    vlog_print("1", "ME", __file__, sys._getframe().f_lineno, "Load checkpoint is finished.")
    end_load_time = time.time()
    load_checkpoint_cost_time = end_load_time - start_load_time
    vlog_print("1", "ME", __file__, sys._getframe().f_lineno, f"Load checkpoint cost time {load_checkpoint_cost_time}.")
    return parameter_dict


@_mstx_range_decorator("load_checkpoint_async", domain="model_preparation")
def load_checkpoint_async(ckpt_file_name, net=None, strict_load=False, filter_prefix=None, dec_key=None,
                          dec_mode="AES-GCM", specify_prefix=None, choice_func=None):
    """
    Load checkpoint info from a specified file asyncly.

    .. warning::
        This is an experimental API that is subject to change or deletion.

    Note:
        - `specify_prefix` and `filter_prefix` do not affect each other.
        - If none of the parameters are loaded from checkpoint file, it will throw ValueError.
        - `specify_prefix` and `filter_prefix` are in the process of being deprecated,
          `choice_func` is recommended instead.
          And using either of those two args will override `choice_func` at the same time.

    Args:
        ckpt_file_name (str): Checkpoint file name. The file extension must be `ckpt` or `safetensors` .
        net (Cell, optional): The network where the parameters will be loaded. Default: ``None`` .
        strict_load (bool, optional): Whether to strict load the parameter into net. If ``False`` , it will load
                                      parameter into net when parameter name's suffix in checkpoint file is the
                                      same as the parameter in the network. When the types are inconsistent
                                      perform type conversion on the parameters of the same type, such as float32
                                      to float16. Default: ``False`` .
        filter_prefix (Union[str, list[str], tuple[str]], optional): Deprecated(see `choice_func`). Parameters
            starting with the `filter_prefix` will not be loaded. Default: ``None`` .
        dec_key (Union[None, bytes], optional): Byte type key used for decryption. If the value is ``None`` ,
                                                the decryption is not required. Default: ``None`` .
        dec_mode (str, optional): This parameter is valid only when dec_key is not set to ``None`` . Specifies
                                  the decryption mode, currently supports ``"AES-GCM"`` and ``"AES-CBC"``
                                  and ``"SM4-CBC"`` . Default: ``"AES-GCM"`` .
        specify_prefix (Union[str, list[str], tuple[str]], optional): Deprecated(see `choice_func`). Parameters
            starting with the specify_prefix will be loaded. Default: ``None`` .
        choice_func (Union[None, function], optional): Input value of the function is a Parameter name of type
            string, and the return value is a bool. If returns ``True`` , the Parameter
            that matches the custom condition will be loaded. If returns ``False`` , the Parameter that
            matches the custom condition will be removed. Default: ``None`` .

    Returns:
        A custom inner class, calling its `result` method yields the :func:`mindspore.load_checkpoint` result.

    Raises:
        ValueError: Checkpoint file's format is incorrect.
        ValueError: Parameter's dict is None after load checkpoint file.
        TypeError: The type of `specify_prefix` or `filter_prefix` is incorrect.

    Examples:
        >>> import mindspore
        >>> from mindspore import nn
        >>> from mindspore.train import Model
        >>> from mindspore.amp import FixedLossScaleManager
        >>> from mindspore import context
        >>> from mindspore import load_checkpoint_async
        >>> from mindspore import load_param_into_net
        >>> mindspore.set_device(device_target="Ascend")
        >>> context.set_context(mode=context.GRAPH_MODE)
        >>> # Create the dataset taking MNIST as an example. Refer to
        >>> # https://gitee.com/mindspore/docs/blob/master/docs/mindspore/code/mnist.py
        >>> dataset = create_dataset()
        >>> # Define the network structure of LeNet5. Refer to
        >>> # https://gitee.com/mindspore/docs/blob/master/docs/mindspore/code/lenet.py
        >>> ckpt_file = "./checkpoint/LeNet5-1_32.ckpt"
        >>> net = LeNet5()
        >>> loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")
        >>> loss_scale_manager = FixedLossScaleManager()
        >>> optim = nn.Momentum(params=net.trainable_params(), learning_rate=0.1, momentum=0.9)
        >>> model = Model(net, loss_fn=loss, optimizer=optim, metrics=None,
        ...               loss_scale_manager=loss_scale_manager)
        >>> pd_future = load_checkpoint_async(ckpt_file)
        >>> model.build(train_dataset=dataset, epoch=2)
        >>> param_dict = pd_future.result()
        >>> load_param_into_net(net, param_dict)
        >>> model.train(2, dataset)
        >>> print("param dict len: ", len(param_dict), flush=True)
    """
    format = "safetensors" if ckpt_file_name.endswith(".safetensors") else "ckpt"
    from concurrent.futures import ThreadPoolExecutor
    executor = ThreadPoolExecutor(max_workers=2)
    param_dict_future = executor.submit(load_checkpoint, ckpt_file_name, net, strict_load, filter_prefix,
                                        dec_key, dec_mode, specify_prefix, choice_func, format=format)
    return ParamDictFuture(executor, param_dict_future)


def _load_map_parameter(checkpoint_list, element, element_id, map_data_list,
                        map_shape_list, parameter_dict):
    """load map parameter."""
    logger.info("Checkpoint load map_parameter.")
    if (element_id != len(checkpoint_list.value) - 1) and \
            element.tag == checkpoint_list.value[element_id + 1].tag:
        for index, tensor in enumerate(element.maptensor.tensor):
            data = tensor.tensor_content
            data_type = tensor.tensor_type
            np_type = np_type_convert.get(data_type)
            element_data = np.frombuffer(data, np_type)
            map_data_list[index].append(element_data)
            map_shape_list[index] += tensor.dims[0]
    else:
        map_array = []
        for index, tensor in enumerate(element.maptensor.tensor):
            data = tensor.tensor_content
            data_type = tensor.tensor_type
            np_type = np_type_convert.get(data_type)
            element_data = np.frombuffer(data, np_type)
            map_data_list[index].append(element_data)
            new_data = b"".join(map_data_list[index])
            param_data = np.frombuffer(new_data, np_type)
            dims = tensor.dims
            dims[0] += map_shape_list[index]
            param_data = param_data.reshape(list(dims))
            map_array.append(param_data)
        parameter_dict[element.tag] = map_array


def _check_ckpt_file_name(ckpt_file_name, format):
    """Check function load_checkpoint's ckpt_file_name."""
    if not isinstance(ckpt_file_name, str):
        raise TypeError("For 'load_checkpoint', the argument 'ckpt_file_name' must be string, "
                        "but got {}.".format(type(ckpt_file_name)))

    if format not in ['ckpt', 'safetensors']:
        raise ValueError("For 'load_checkpoint', the checkpoint file should end with '.ckpt' or '.safetensors', please "
                         "input the correct 'ckpt_file_name'.")
    if not ckpt_file_name.endswith(format):
        raise ValueError(f"For 'load_checkpoint', the checkpoint file format must same with 'format', but got "
                         f"file_name:'{ckpt_file_name}', format:'{format}'")

    ckpt_file_name = os.path.realpath(ckpt_file_name)
    if not os.path.exists(ckpt_file_name):
        raise ValueError("For 'load_checkpoint', the checkpoint file: {} does not exist, please check "
                         "whether the 'ckpt_file_name' is correct.".format(ckpt_file_name))

    return ckpt_file_name


def _check_prefix(prefix):
    """Check the correctness of the parameters."""
    if prefix is None:
        return prefix
    if not isinstance(prefix, (str, list, tuple)):
        raise TypeError("For 'load_checkpoint', the type of 'specify_prefix' or 'filter_prefix' must be string, "
                        "list[string] or tuple[string], but got {}.".format(str(type(prefix))))
    if isinstance(prefix, str):
        prefix = (prefix,)
    if not prefix:
        raise ValueError("For 'load_checkpoint', the argument 'specify_prefix' or 'filter_prefix' can't be empty when"
                         " 'specify_prefix' or 'filter_prefix' is list or tuple.")
    for index, pre in enumerate(prefix):
        if not isinstance(pre, str):
            raise TypeError("For 'load_checkpoint', when 'specify_prefix' or 'filter_prefix' is list or tuple, "
                            "the element in it must be string, but got "
                            f"{str(type(pre))} at index {index}.")
        if pre == "":
            raise ValueError("For 'load_checkpoint', the value of 'specify_prefix' or 'filter_prefix' "
                             "can't include ''.")
    return prefix


def _parse_ckpt_proto(ckpt_file_name, dec_key, dec_mode, crc_check):
    """Parse checkpoint protobuf."""
    checkpoint_list = Checkpoint()
    try:
        if dec_key is None:
            with _ckpt_fs.open(ckpt_file_name, *_ckpt_fs.open_args) as f:
                ckpt_load_time_start = time.time()
                pb_content = f.read()
                ckpt_load_time_end = time.time()
                cost_time = ckpt_load_time_end - ckpt_load_time_start
                vlog_print("1", "ME", __file__, sys._getframe().f_lineno, f"Load ckpt io cost time:{cost_time}.")

        else:
            pb_content = _decrypt(ckpt_file_name, dec_key, len(dec_key), dec_mode)
            if pb_content is None:
                raise ValueError("For 'load_checkpoint', failed to decrypt the checkpoint file.")
        if crc_check and pb_content[-17:-10] != b"crc_num":
            logger.warning("For 'load_checkpoint', the ckpt file do not contain the crc code, please check the file.")
        if pb_content[-17:-10] == b"crc_num":
            crc_num_bytes = pb_content[-10:]
            pb_content = pb_content[:-17]
            if crc_check:
                crc_num = int.from_bytes(crc_num_bytes, byteorder='big')
                cal_crc_num = binascii.crc32(pb_content, 0)
                if cal_crc_num != crc_num:
                    raise ValueError("For 'load_checkpoint', the crc check is failed, "
                                     "please check whether the ckpt file is damaged.")
        checkpoint_list.ParseFromString(pb_content)
    except google.protobuf.message.DecodeError as e:
        raise ValueError(f"Failed to read the checkpoint file {ckpt_file_name}. "
                         f"The file may be corrupted, and the content cannot be parsed.") from e
    except BaseException as e:
        if _is_cipher_file(ckpt_file_name):
            err_info = "Failed to read the checkpoint file {}. The file may be encrypted or tempered with, " \
                       "please pass in the correct 'dec_key' or check the file integrity.".format(ckpt_file_name)
        else:
            err_info = "Failed to read the checkpoint file {}. May not have permission to read it, please check" \
                       " the correct of the file.".format(ckpt_file_name)
        logger.error(err_info)
        raise ValueError(err_info) from e
    return checkpoint_list


def _whether_load_param(specify_prefix, filter_prefix, param_name):
    """Checks whether the load the parameter after `specify_prefix` or `filter_prefix`."""
    whether_load = True
    if specify_prefix:
        whether_load = False
        for prefix in specify_prefix:
            if param_name.startswith(prefix):
                whether_load = True
                break
    if filter_prefix:
        for prefix in filter_prefix:
            if param_name.startswith(prefix):
                whether_load = False
                break
    return whether_load


def _check_load_param_into_net(net, parameter_dict):
    """check load_param_into_net"""
    if not isinstance(net, nn.Cell):
        logger.critical("Failed to combine the net and the parameters.")
        msg = ("For 'load_param_into_net', the argument 'net' should be a Cell, but got {}.".format(type(net)))
        raise TypeError(msg)
    if not isinstance(parameter_dict, dict):
        logger.critical("Failed to combine the net and the parameters.")
        msg = ("For 'load_param_into_net', the argument 'parameter_dict' should be a dict, "
               "but got {}.".format(type(parameter_dict)))
        raise TypeError(msg)
    for key, value in parameter_dict.items():
        if not isinstance(key, str) or not isinstance(value, (Parameter, str, list)):
            logger.critical("Load parameters into net failed.")
            msg = ("For 'parameter_dict', the element in the argument 'parameter_dict' should be a "
                   "'str' and 'Parameter' , but got {} and {}.".format(type(key), type(value)))
            raise TypeError(msg)


def _check_remove_redundancy_net(net):
    """Check whether the network is compiled with the remove_redundancy feature."""
    if get_group_size() == 1:
        raise TypeError("The deduplication feature for loading checkpoint can only be used "
                        "in parallel scenarios, but got stand_alone.")
    if not net.compile_cache and not net.parameter_layout_dict:
        raise ValueError("When loading a parameter dict that has removed redundancy, "
                         "the network should be compiled.")


@_mstx_range_decorator("load_param_into_net", domain="model_preparation")
def load_param_into_net(net, parameter_dict, strict_load=False, remove_redundancy=False):
    """
    Load parameters into network, return parameter list that are not loaded in the network.

    Note:
        When loading a parameter dict that has removed redundancy, the network should be compiled.

    Args:
        net (Cell): The network where the parameters will be loaded.
        parameter_dict (dict): The dictionary generated by load checkpoint file,
                               it is a dictionary consisting of key: parameters's name, value: parameter.
        strict_load (bool, optional): Whether to strict load the parameter into net. If ``False`` ,
                            it will load parameter
                            into net when parameter name's suffix in checkpoint file is the same as the
                            parameter in the network. When the types are inconsistent perform type conversion
                            on the parameters of the same type, such as float32 to float16. Default: ``False`` .
        remove_redundancy (bool, optional): Whether to enable loading of checkpoint saved with redundancy removal.
            Redundancy removal refers to eliminating redundant data in data parallelism mode. Default: ``False`` , means
            redundant-free loading is not enabled.

    Returns:
        - param_not_load (List), the parameter name in model which are not loaded into the network.
        - ckpt_not_load (List), the parameter name in checkpoint file which are not loaded into the network.

    Raises:
        TypeError: Argument is not a Cell, or parameter_dict is not a Parameter dictionary.

    Examples:
        >>> import mindspore as ms
        >>>
        >>> # Define the network structure of LeNet5. Refer to
        >>> # https://gitee.com/mindspore/docs/blob/master/docs/mindspore/code/lenet.py
        >>> net = LeNet5()
        >>> ckpt_file_name = "./checkpoint/LeNet5-1_32.ckpt"
        >>> param_dict = ms.load_checkpoint(ckpt_file_name, filter_prefix="conv1")
        >>> param_not_load, _ = ms.load_param_into_net(net, param_dict)
        >>> print(param_not_load)
        ['conv1.weight']

    Tutorial Examples:
        - `Saving and Loading the Model - Saving and Loading the Model Weight
          <https://mindspore.cn/tutorials/en/master/beginner/save_load.html#saving-and-loading-the-model-weight>`_
    """
    _check_load_param_into_net(net, parameter_dict)

    strict_load = Validator.check_bool(strict_load)
    remove_redundancy = Validator.check_isinstance('remove_redundancy', remove_redundancy, bool)
    logger.info("Execute the process of loading parameters into net.")
    param_not_load = []
    param_loaded = set()
    ckpt_not_load = list(parameter_dict.keys())
    is_parallel_mode = _is_auto_parallel_mode(net)
    for _, param in net.parameters_and_names():
        if param.param_info.is_pipeline_shared_param:
            continue
        if param.name in parameter_dict:
            if isinstance(param, MapParameter):
                param.import_data(parameter_dict[param.name])
                continue
            # Add has attr protection when load server checkpoint file on worker.
            if not hasattr(parameter_dict[param.name], "data"):
                continue
            new_param = parameter_dict[param.name]
            _update_param(param, new_param, strict_load)
            if hasattr(param, "init_param") and not param.init_param:
                param.init_param = True
            ckpt_not_load.remove(param.name)
            param_loaded.add(param.name)
        else:
            if param.name.startswith("accu_grads"):
                continue
            if param.param_info.is_pipeline_shared_param:
                continue
            if is_parallel_mode and not param.sliced:
                continue
            param_not_load.append(param.name)

    if param_not_load and not strict_load:
        _load_dismatch_prefix_params(net, parameter_dict, param_not_load, strict_load)

    if remove_redundancy:
        _check_remove_redundancy_net(net)
        param_layout = net.parameter_layout_dict
        _single_parameter_broadcast(net, param_layout, param_not_load, param_loaded)

    logger.info("Loading parameters into net is finished.")
    if param_not_load:
        logger.warning("For 'load_param_into_net', "
                       "{} parameters in the 'net' are not loaded, because they are not in the "
                       "'parameter_dict', please check whether the network structure is consistent "
                       "when training and loading checkpoint.".format(len(param_not_load)))
        logger.warning("{} are not loaded.".format(param_not_load))
    return param_not_load, ckpt_not_load


def _warm_up_host_cache_enabled(parameter_dict):
    """Warm up host cache enabled."""
    for key in parameter_dict.keys():
        if key.find(".__param_key__") != -1:
            return True
    return False


def _warm_up_host_cache(parameter_dict, net):
    """Warm up host cache."""
    ms_role = os.getenv("MS_ROLE")
    is_worker = ms_role == "MS_WORKER"
    param_key_dict = {}
    # Traverse key, value in parameter_dict, warm up param key and record param key into param_key_dict.
    if is_worker:
        net.init_parameters_data()
        net_dict = {}
        for name, value in net.parameters_and_names():
            net_dict[name] = value
        for param_name, value in parameter_dict.items():
            pos = param_name.find(".__param_key__")
            if pos != -1:
                net_param_name = param_name[:pos]
                param_key_dict[param_name] = net_param_name
                net_value = None
                if net_param_name not in net_dict:
                    logger.warning("net param name : %s is not in net", net_param_name)
                else:
                    net_value = net_dict.get(net_param_name, None)
                pos += len(".__param_key__")
                param_key = int(param_name[pos:])
                value_is_map_parameter = isinstance(value, list) and len(value) == 3
                if value_is_map_parameter and (net_value is None or isinstance(net_value, Parameter)):
                    key_tensor = Tensor.from_numpy(value[0])
                    value_tensor = Tensor.from_numpy(value[1])
                    status_tensor = Tensor.from_numpy(value[2])
                    _store_warm_up_ptr_by_tensor_list(param_key, key_tensor, value_tensor, status_tensor)
                elif not isinstance(value, list) and isinstance(net_value, Parameter):
                    _store_warm_up_ptr_by_tensor(param_key, value)
                else:
                    logger.warning("Unknown matches parameter type %s and net_value %s", type(value), type(net_value))
    else:
        for param_name, value in parameter_dict.items():
            pos = param_name.find(".__param_key__")
            if pos != -1:
                net_param_name = param_name[:pos]
                param_key_dict[param_name] = net_param_name
    # Split param key from parameter_dict since worker cannot load param key.
    warm_up_dict = {}
    for key, value in param_key_dict.items():
        if is_worker:
            warm_up_dict[value] = parameter_dict.pop(key)
        else:
            parameter_dict[value] = parameter_dict.pop(key)
    return (is_worker, parameter_dict, warm_up_dict)


def _warm_up_host_cache_post_process(is_worker, net_dict, warm_up_dict):
    """Warm up host cache post process."""
    if is_worker:
        net_dict.update(warm_up_dict)
    _set_checkpoint_load_status(True)


def _load_dismatch_prefix_params(net, parameter_dict, param_not_load, strict_load):
    """When some net parameter did not load, try to continue loading."""
    prefix_name = ""
    longest_name = param_not_load[0]
    while prefix_name != longest_name and param_not_load:
        logger.debug("Count: {} parameters has not been loaded, try to continue loading.".format(len(param_not_load)))
        prefix_name = longest_name
        for net_param_name in param_not_load:
            for dict_name in parameter_dict:
                if dict_name.endswith(net_param_name):
                    prefix_name = dict_name[:-len(net_param_name)]
                    break
            if prefix_name != longest_name:
                break

        if prefix_name != longest_name:
            logger.warning(f"For 'load_param_into_net', remove parameter prefix name: {prefix_name},"
                           f" continue to load.")
            for _, param in net.parameters_and_names():
                new_param_name = prefix_name + param.name
                if param.name in param_not_load and new_param_name in parameter_dict:
                    new_param = parameter_dict[new_param_name]
                    _update_param(param, new_param, strict_load)
                    if hasattr(param, "init_param") and not param.init_param:
                        param.init_param = True
                    param_not_load.remove(param.name)


def _save_graph(network, file_name):
    """
    Saves the graph of network to a file.

    Args:
        network (Cell): Obtain a pipeline through network for saving graph.
        file_name (str): Graph file name into which the graph will be saved.
    """
    logger.info("Execute the process of saving graph.")
    file_name = os.path.realpath(file_name)
    graph_pb = network.get_func_graph_proto()
    if os.path.isfile(file_name) and graph_pb:
        os.remove(file_name)
    if graph_pb:
        with open(file_name, "wb") as f:
            os.chmod(file_name, stat.S_IRUSR | stat.S_IWUSR)
            f.write(graph_pb)


def _reshape_tensor(tensor, dst_shape):
    """reshape tensor to dst shape"""
    np_tensor = tensor.asnumpy()
    np_tensor = np_tensor.reshape(dst_shape)
    return Tensor(np_tensor, tensor.dtype)


def _check_param_for_integrate_save(pipeline_stages, uniform_split):
    """check whether current settings and parameters are supported in integrated save checkpoint mode"""
    if pipeline_stages > 1:
        raise RuntimeError("Pipeline Parallel don't support Integrated save checkpoint now.")
    if uniform_split == 0:
        raise RuntimeError("For 'save_checkpoint' and in automatic model parallel scene, when set "
                           "'integrated_save' to True, the checkpoint will be integrated save, it "
                           "is only supports uniform split tensor now.")


def _get_merged_param_data(net, parameter_layout_dict, param_name, param_data, integrated_save):
    """
    Gets the merged data(tensor) from tensor slice, by device arrangement and tensor map.

    Args:
        net (Cell): MindSpore network.
        param_name (str): The parameter name, which to be combined.
        param_data (Tensor): The parameter data on the local device, which was a slice of the whole parameter data.
        integrated_save (bool): Whether to integrated save in automatic model parallel scene.
    Returns:
        Tensor, the combined tensor which with the whole data value.
    """
    layout = parameter_layout_dict[param_name]
    if len(layout) < 8:
        logger.info("The layout dict does not contain the key %s", param_name)
        return param_data

    dev_mat = layout[0]
    tensor_map = layout[1]
    uniform_split = layout[4]
    opt_shard_group = layout[5]
    before_reshape_slice_shape = layout[2]
    before_reshape_full_shape = layout[6]
    after_reshape_slice_shape = layout[7]
    do_reshape = False
    if before_reshape_full_shape and after_reshape_slice_shape \
            and after_reshape_slice_shape != before_reshape_slice_shape:
        do_reshape = True

    allgather_net = None
    mp_weight = False
    for dim in tensor_map:
        if dim != -1:
            mp_weight = True
            break
    if param_name in net.parallel_parameter_merge_net_dict:
        allgather_net = net.parallel_parameter_merge_net_dict[param_name]
    else:
        logger.info("Need to create allgather net for %s", param_name)
        if integrated_save:
            _check_param_for_integrate_save(context.get_auto_parallel_context("pipeline_stages"), uniform_split)
            # while any dim is not equal to -1, means param is split and needs to be merged
            # pipeline parallel need to be supported here later
            if mp_weight:
                allgather_net = get_allgather_cell(opt_shard_group, bool(opt_shard_group), do_reshape,
                                                   tuple(after_reshape_slice_shape))
                object.__setattr__(allgather_net, "keep_input_unchanged", True)
            elif opt_shard_group:
                allgather_net = get_allgather_cell(opt_shard_group, False, do_reshape,
                                                   tuple(after_reshape_slice_shape))
        net.parallel_parameter_merge_net_dict[param_name] = allgather_net
    if allgather_net:
        param_data = allgather_net(param_data)
    if mp_weight and integrated_save:
        param_data = _reshape_param_data(param_data, dev_mat, tensor_map)
        if do_reshape:
            param_data = _reshape_tensor(param_data, before_reshape_full_shape)
    return param_data


def export(net, *inputs, file_name, file_format, **kwargs):
    """
    Export the MindSpore network into an offline model in the specified format.

    Note:
        1. When exporting AIR, ONNX format, the size of a single tensor can not exceed 2GB.
        2. When `file_name` does not have a suffix, the system will automatically add one
           according to the `file_format`.
        3. Exporting functions decorated with :func:`mindspore.jit` to mindir format is supported.
        4. When exporting a function decorated with :func:`mindspore.jit`, the function should not involve
           class properties in calculations.
        5. AIR format is deprecated, and will be removed in a future version, please use other format or use
           MindSpore Lite to do offline inference.

    Args:
        net (Union[Cell, function]): MindSpore network.
        inputs (Union[Tensor, Dataset, List, Tuple, Number, Bool]): It represents the inputs
             of the `net`, if the network has multiple inputs, set them together. While its type is Dataset,
             it represents the preprocess behavior of the `net`, data preprocess operations will be serialized.
             In second situation, you should adjust batch size of dataset script manually which will impact on
             the batch size of 'net' input. Only supports parse "image" column from dataset currently.
        file_name (str): File name of the model to be exported.
        file_format (str): MindSpore currently supports 'AIR', 'ONNX' and 'MINDIR' format for exported model.

            - AIR: Ascend Intermediate Representation. An intermediate representation format of Ascend model.
            - ONNX: Open Neural Network eXchange. An open format built to represent machine learning models.
            - MINDIR: MindSpore Native Intermediate Representation for Anf. An intermediate representation format
              for MindSpore models. MINDIR does not support operators which have dictionary attribute.

        kwargs (dict): Configuration options dictionary.

            - enc_key (byte): Byte-type key used for encryption. The valid length is 16, 24, or 32.
            - enc_mode (Union[str, function]): Specifies the encryption mode, to take effect when enc_key is set.

              - For 'AIR' and 'ONNX' models, only customized encryption is supported.
              - For 'MINDIR', all options are supported. Option: 'AES-GCM', 'AES-CBC', 'SM4-CBC'
                or Customized encryption.
                Default: ``'AES-GCM'``.
              - For details of using the customized encryption, please check the `tutorial
                <https://mindspore.cn/mindarmour/docs/en/master/model_encrypt_protection.html>`_.

            - dataset (Dataset): Specifies the preprocessing method of the dataset, which is used to import the
              preprocessing of the dataset into MindIR.
            - incremental (bool): export MindIR incrementally.

            - custom_func (function): Functions for custom defined export policies. This function will be used to
              customize the model during network export. Currently only support for files with mindir format. The
              function only accepts one input representing the proto object of the mindir file. When modifying a model,
              it is necessary to ensure the correctness of the `custom_func` , otherwise it may lead to model loading
              failure or functional errors. Default: ``None`` .

    Examples:
        >>> import mindspore as ms
        >>> import numpy as np
        >>> from mindspore import Tensor
        >>>
        >>> # Define the network structure of LeNet5. Refer to
        >>> # https://gitee.com/mindspore/docs/blob/master/docs/mindspore/code/lenet.py
        >>> net = LeNet5()
        >>> input_tensor = Tensor(np.ones([1, 1, 32, 32]).astype(np.float32))
        >>> ms.export(net, input_tensor, file_name='lenet', file_format='MINDIR')
        >>>
        >>> # Export model in MindIR format and modified the model info using custom_func
        >>> # The custom_func only support one input representing the Proto object of the model
        >>> # And custom_func does not support return value
        >>> def _custom_func(mindir_model):
        ...     mindir_model.producer_name = "test11111"
        ...     mindir_model.producer_version = "11.0"
        ...     mindir_model.user_info["version"] = "11.0"
        >>> ms.export(net, input_tensor, file_name="lenet", file_format='MINDIR', custom_func=_custom_func)


    Tutorial Examples:
        - `Saving and Loading the Model - Saving and Loading MindIR
          <https://mindspore.cn/tutorials/en/master/beginner/save_load.html#saving-and-loading-mindir>`_
    """
    if 'obf_func' in kwargs.keys():
        raise NotImplementedError("Dynamic model structure obfuscation is no longer supported.")
    old_ms_jit_value = context.get_context("jit_syntax_level")
    context.set_context(jit_syntax_level=mindspore.STRICT)

    supported_formats = ['AIR', 'ONNX', 'MINDIR']
    if file_format not in supported_formats:
        raise ValueError(f"For 'export', 'file_format' must be one of {supported_formats}, but got {file_format}.")
    if file_format == 'AIR':
        logger.warning("AIR format is deprecated, and will be removed in a future version, please use other format or "
                       "use MindSpore Lite to do offline inference")
    Validator.check_file_name_by_regular(file_name)
    logger.info("exporting model file:%s format:%s.", file_name, file_format)

    if check_input_dataset(*inputs, dataset_type=mindspore.dataset.Dataset):
        if len(inputs) != 1:
            raise RuntimeError("You can only serialize one dataset into MindIR, got " + str(len(inputs)) + " datasets")
        shapes, types, columns = inputs[0].output_shapes(), inputs[0].output_types(), inputs[0].get_col_names()
        kwargs['dataset'] = inputs[0]
        only_support_col = "image"

        inputs_col = []
        for c, s, t in zip(columns, shapes, types):
            if only_support_col != c:
                continue
            inputs_col.append(Tensor(np.random.uniform(-1.0, 1.0, size=s).astype(t)))
        if not inputs_col:
            raise RuntimeError("Only supports parse \"image\" column from dataset now, given dataset has columns: "
                               + str(columns))
        inputs = tuple(inputs_col)

    file_name = os.path.realpath(file_name)
    if 'enc_key' in kwargs.keys():
        kwargs['enc_key'], kwargs['enc_mode'] = _check_key_mode_type(file_format, **kwargs)
    _export(net, file_name, file_format, *inputs, **kwargs)

    context.set_context(jit_syntax_level=old_ms_jit_value)


def _get_funcgraph(net, *inputs):
    """
    Compile the MindSpore network and get FuncGraph.

    Arg:
        net (Union[Cell, function]): MindSpore network.
        inputs (Union[Tensor, Dataset, List, Tuple, Number, Bool]): It represents the inputs
             of the `net`, if the network has multiple inputs, set them together. While its type is Dataset,
             it represents the preprocess behavior of the `net`, data preprocess operations will be serialized.
             In second situation, you should adjust batch size of dataset script manually which will impact on
             the batch size of 'net' input. Only supports parse "image" column from dataset currently.

    Returns:
        FuncGraph, a mindspore._c_expression.FuncGraph obj.

    Raises:
        ValueError: input `net` is not a nn.Cell.

    Examples:
        >>> import mindspore as ms
        >>> import numpy as np
        >>> from mindspore import Tensor
        >>>
        >>> # Define the network structure of LeNet5. Refer to
        >>> # https://gitee.com/mindspore/docs/blob/master/docs/mindspore/code/lenet.py
        >>> net = LeNet5()
        >>> input_tensor = Tensor(np.ones([1, 1, 32, 32]).astype(np.float32))
        >>> ms.get_funcgraph(net, input_tensor)

    """
    if not isinstance(net, nn.Cell):
        raise ValueError("For get_funcgraph's parameter 'net', currently only support Cell right now.")
    phase_name = "lite_infer_predict" if _is_in_auto_parallel_mode() else "lite_infer_get_func_graph"
    graph_id, _ = _executor.compile(net, *inputs, phase=phase_name, do_convert=False)
    # pylint: disable=protected-access
    func_graph = _executor._get_func_graph(net, graph_id)
    return func_graph


def _export(net, file_name, file_format, *inputs, **kwargs):
    """
    It is an internal conversion function. Export the MindSpore prediction model to a file in the specified format.
    """
    logger.info("exporting model file:%s format:%s.", file_name, file_format)
    if "custom_func" in kwargs and file_format != "MINDIR" and kwargs["custom_func"] is not None:
        raise ValueError(f"Currently only support custom_func for MindIR format, but got {file_format} format.")
    if file_format == 'AIR':
        _save_air(net, file_name, *inputs, **kwargs)
    elif file_format == 'ONNX':
        logger.warning("mindspore.export(file_format='ONNX') will be deleted, please use mindspore.onnx.export()")
        _save_onnx(net, file_name, *inputs, **kwargs)
    elif file_format == 'MINDIR':
        _save_mindir(net, file_name, *inputs, **kwargs)


def _check_key_mode_type(file_format, **kwargs):
    """check enc_key and enc_mode are valid"""
    enc_key = Validator.check_isinstance('enc_key', kwargs.get('enc_key'), bytes)
    enc_mode = kwargs.get('enc_mode')

    if callable(enc_mode):
        return enc_key, enc_mode

    enc_mode = 'AES-GCM'
    if 'enc_mode' in kwargs.keys():
        enc_mode = Validator.check_isinstance('enc_mode', kwargs.get('enc_mode'), str)

    if file_format in ('AIR', 'ONNX'):
        raise ValueError(f"AIR/ONNX only support customized encryption, but got {enc_mode}.")

    if enc_mode in ('AES-CBC', 'AES-GCM', 'SM4-CBC'):
        return enc_key, enc_mode
    raise ValueError(f"MindIR only support AES-GCM/AES-CBC/SM4-CBC encryption, but got {enc_mode}")


def _save_air(net, file_name, *inputs, **kwargs):
    """Save AIR format file."""
    phase_name = 'export.air'
    graph_id, _ = _executor.compile(net, *inputs, phase=phase_name)
    if not file_name.endswith('.air'):
        file_name += ".air"
    if os.path.exists(file_name):
        os.chmod(file_name, stat.S_IWUSR)
    if "/" in file_name:
        real_path = os.path.realpath(file_name[:file_name.rfind("/")])
        os.makedirs(real_path, mode=0o700, exist_ok=True)
    if 'enc_key' in kwargs.keys() and 'enc_mode' in kwargs.keys():
        _executor.export(file_name, graph_id, enc_key=kwargs.get('enc_key'), encrypt_func=kwargs.get('enc_mode'))
    else:
        _executor.export(file_name, graph_id)
    os.chmod(file_name, stat.S_IRUSR)


def _save_onnx(net, file_name, *inputs, **kwargs):
    """Save ONNX format file."""
    # When dumping ONNX file, switch network mode to infer when it is training(NOTE: ONNX only designed for prediction)
    if not isinstance(net, nn.Cell):
        raise ValueError(f"Export ONNX format model only support nn.Cell object, but got {type(net)}.")
    _check_dynamic_input(inputs)
    cell_mode = net.training
    net.set_train(mode=False)
    total_size = _calculation_net_size(net)
    if total_size > PROTO_LIMIT_SIZE:
        raise RuntimeError('Export onnx model failed. Network size is: {}G, it exceeded the protobuf: {}G limit.'
                           .format(total_size / 1024 / 1024, PROTO_LIMIT_SIZE / 1024 / 1024))
    phase_name = 'export.onnx'
    graph_id, _ = _executor.compile(net, *inputs, phase=phase_name, do_convert=False)
    onnx_stream = _executor._get_func_graph_proto(net, graph_id)
    if 'enc_key' in kwargs.keys() and 'enc_mode' in kwargs.keys():
        enc_mode = kwargs.get('enc_mode')
        onnx_stream = enc_mode(onnx_stream, kwargs.get('enc_key'))
    if not file_name.endswith('.onnx'):
        file_name += ".onnx"
    if os.path.exists(file_name):
        os.chmod(file_name, stat.S_IWUSR)
    else:
        dir_path = os.path.dirname(file_name)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path, mode=0o700, exist_ok=True)
        os.chmod(dir_path, 0o700)
    with open(file_name, 'wb') as f:
        f.write(onnx_stream)
        os.chmod(file_name, stat.S_IRUSR)
    net.set_train(mode=cell_mode)


def _check_dynamic_input(inputs):
    for ele in inputs:
        if isinstance(ele, Tensor) and -1 in ele.shape:
            raise ValueError("Export ONNX format model not support dynamic shape mode.")


def _generate_front_info_for_param_data_file(is_encrypt, kwargs):
    front_info = bytes()
    check_code = sys.byteorder == "little"
    front_info += check_code.to_bytes(1, byteorder=sys.byteorder)
    front_info += bytes(63)
    if is_encrypt():
        front_info = _encrypt(front_info, len(front_info), kwargs.get('enc_key'),
                              len(kwargs.get('enc_key')), kwargs.get('enc_mode'))
    return front_info


def _change_file(f, dirname, external_local, is_encrypt, kwargs):
    """Change to another file to write parameter data."""
    # The parameter has been not written in the file
    front_info = _generate_front_info_for_param_data_file(is_encrypt, kwargs)
    f.seek(0, 0)
    f.write(front_info)
    f.close()
    ori_data_file_name = f.name
    os.chmod(ori_data_file_name, stat.S_IRUSR)
    if os.path.getsize(ori_data_file_name) == 64:
        raise RuntimeError("The parameter size is exceed 1T,cannot export to the file")
    data_file_name = os.path.join(dirname, external_local)
    return _get_data_file(is_encrypt, kwargs, data_file_name)


def _get_data_file(is_encrypt, kwargs, data_file_name):
    """Get Data File to write parameter data."""
    # Reserves 64 bytes as spare information such as check data
    offset = 64
    if os.path.exists(data_file_name):
        os.chmod(data_file_name, stat.S_IWUSR)

    place_holder_data = bytes(offset)
    if is_encrypt():
        place_holder_data = _encrypt(place_holder_data, len(place_holder_data), kwargs["enc_key"],
                                     len(kwargs["enc_key"]), kwargs["enc_mode"])
    parameter_size = offset / 1024
    try:
        f = open(data_file_name, "wb")
        f.write(place_holder_data)
    except IOError:
        f.close()

    return f, parameter_size, offset


def _encrypt_data(is_encrypt, write_data, kwargs):
    """Encrypt parameter data."""
    if is_encrypt():
        if callable(kwargs.get('enc_mode')):
            enc_func = kwargs.get('enc_mode')
            write_data = enc_func(write_data, kwargs.get('enc_key'))
        else:
            write_data = _encrypt(write_data, len(write_data), kwargs.get('enc_key'),
                                  len(kwargs.get('enc_key')), kwargs.get('enc_mode'))
    return write_data


def _split_save(net_dict, model, file_name, is_encrypt, **kwargs):
    """The function to save parameter data."""
    logger.warning("Parameters in the net capacity exceeds 1G, save MindIR model and parameters separately.")
    # save parameter
    if model.graph.map_parameter:
        raise ValueError("MapParameter not support save in split MindIR file now.")
    file_prefix = file_name.split("/")[-1]
    if file_prefix.endswith(".mindir"):
        file_prefix = file_prefix[:-7]
    current_path = os.path.realpath(file_name)
    dirname = os.path.dirname(current_path)
    data_path = os.path.join(dirname, file_prefix + "_variables")
    if os.path.exists(data_path):
        shutil.rmtree(data_path)
    os.makedirs(data_path, mode=0o700, exist_ok=True)
    os.chmod(data_path, stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR)
    index = 0
    external_local = os.path.join(file_prefix + "_variables", "data_" + str(index))
    data_file_name = os.path.join(dirname, external_local)
    f, parameter_size, offset = _get_data_file(is_encrypt, kwargs, data_file_name)

    round = 0
    names = []

    try:
        for param_proto in model.graph.parameter:
            name = param_proto.name[param_proto.name.find(":") + 1:]
            names.append((name, param_proto))
            names.sort(key=lambda x: x[0])
        for pairs in names:
            name = pairs[0]
            param_proto = pairs[1]
            param = net_dict[name]
            raw_data = param.data.get_bytes()
            data_length = len(raw_data)
            append_size = 0
            if data_length % 64 != 0:
                append_size = 64 - (data_length % 64)
                parameter_size += ((append_size + data_length) / 1024)
            if parameter_size > PARAMETER_SPLIT_SIZE:
                index += 1
                external_local = os.path.join(file_prefix + "_variables", "data_" + str(index))
                f, parameter_size, offset = _change_file(f, dirname, external_local, is_encrypt, kwargs)
                parameter_size += ((append_size + data_length) / 1024)
            param_proto.external_data.location = external_local
            param_proto.external_data.length = data_length
            param_proto.external_data.offset = offset
            write_data = raw_data + bytes(append_size)
            offset += (data_length + append_size)
            write_data = _encrypt_data(is_encrypt, write_data, kwargs)
            f.write(write_data)
            round += 1
            logger.debug(f"writing {round}th split data, name:{name}")

        graph_file_name = os.path.join(dirname, file_prefix + "_graph.mindir")
        if os.path.exists(graph_file_name):
            os.chmod(graph_file_name, stat.S_IWUSR)
        with open(graph_file_name, 'wb') as model_file:
            os.chmod(graph_file_name, stat.S_IRUSR | stat.S_IWUSR)
            model_string = model.SerializeToString()
            if is_encrypt():
                model_string = _encrypt(model_string, len(model_string), kwargs.get('enc_key'),
                                        len(kwargs.get('enc_key')), kwargs.get('enc_mode'))
            model_file.write(model_string)
            os.chmod(graph_file_name, stat.S_IRUSR)

        front_info = _generate_front_info_for_param_data_file(is_encrypt, kwargs)
        f.seek(0, 0)
        f.write(front_info)
    finally:
        f.close()
        os.chmod(data_file_name, stat.S_IRUSR)


def _msfunc_info(net, jit_executor, *inputs):
    """Get mindir stream and parameter dict of ms_function"""
    # pylint: disable=protected-access
    net_dict = OrderedDict()
    phase_name = "export.mindir"
    graph_id = jit_executor.compile(net.__name__, phase=phase_name, *inputs)
    mindir_stream = jit_executor._get_func_graph_proto(net, graph_id, 'mind_ir')
    params = jit_executor._graph_executor.get_params(graph_id)
    for name, value in params.items():
        net_dict[name] = Parameter(value, name=name)
    return mindir_stream, net_dict


def _cell_info(net, incremental, *inputs):
    """Get mindir stream and net dict of cell"""
    phase_name = "export.mindir"
    graph_id, _ = _executor.compile(net, *inputs, phase=phase_name, do_convert=False)
    # pylint: disable=protected-access
    mindir_stream = _executor._get_func_graph_proto(net, graph_id, 'mind_ir', incremental=incremental)
    net_dict = net.parameters_dict()
    return mindir_stream, net_dict


def _save_mindir(net, file_name, *inputs, **kwargs):
    """Save MindIR format file."""
    executor = _executor
    if not isinstance(net, nn.Cell):
        executor = _JitExecutor(net, time.time() * 1e9)

    incremental = kwargs.get('incremental', False)

    model = mindir_model()
    if not isinstance(net, nn.Cell):
        mindir_stream, net_dict = _msfunc_info(net, executor, *inputs)
    else:
        mindir_stream, net_dict = _cell_info(net, incremental, *inputs)
    model.ParseFromString(mindir_stream)

    if kwargs.get('dataset'):
        check_input_data(kwargs.get('dataset'), data_class=mindspore.dataset.Dataset)
        dataset = kwargs.get('dataset')
        _save_dataset_to_mindir(model, dataset)

    custom_func = kwargs.get('custom_func', None)
    if custom_func is not None:
        custom_func(model)

    save_together = _save_together(net_dict, model)
    is_encrypt = lambda: 'enc_key' in kwargs.keys() and 'enc_mode' in kwargs.keys()
    if save_together:
        _save_mindir_together(net_dict, model, file_name, is_encrypt, **kwargs)
    else:
        _split_save(net_dict, model, file_name, is_encrypt, **kwargs)


def _save_mindir_together(net_dict, model, file_name, is_encrypt, **kwargs):
    """Save graph and parameter together."""
    for param_proto in model.graph.parameter:
        param_name = param_proto.name[param_proto.name.find(":") + 1:]
        if param_name in net_dict.keys():
            param_data = net_dict[param_name].data.get_bytes()
            param_proto.raw_data = param_data
        else:
            raise ValueError("The parameter '{}' is not belongs to any cell,"
                             "the data of parameter cannot be exported.".format(param_proto.name))
    incremental = kwargs.get('incremental', False)
    for map_param_proto in model.graph.map_parameter:
        map_param_name = map_param_proto.name[map_param_proto.name.find(":") + 1:]
        if map_param_name in net_dict.keys():
            map_parameter = net_dict[map_param_name]
            key_bytes, value_bytes, status_bytes = map_parameter.export_bytes(incremental)
            map_param_proto.key_tensor.raw_data = key_bytes
            map_param_proto.value_tensor.raw_data = value_bytes
            map_param_proto.status_tensor.raw_data = status_bytes
        else:
            raise ValueError("The map_parameter '{}' is not belongs to any cell,"
                             "the data of parameter cannot be exported.".format(map_param_proto.name))
    if not file_name.endswith('.mindir'):
        file_name += ".mindir"
    current_path = os.path.realpath(file_name)
    dirname = os.path.dirname(current_path)
    os.makedirs(dirname, mode=0o700, exist_ok=True)
    if os.path.exists(file_name):
        os.chmod(file_name, stat.S_IWUSR)
    with open(file_name, 'wb') as f:
        os.chmod(file_name, stat.S_IRUSR | stat.S_IWUSR)
        model_string = model.SerializeToString()
        if is_encrypt():
            if callable(kwargs.get('enc_mode')):
                enc_func = kwargs.get('enc_mode')
                model_string = enc_func(model_string, kwargs.get('enc_key'))
            else:
                model_string = _encrypt(model_string, len(model_string), kwargs.get('enc_key'),
                                        len(kwargs.get('enc_key')), kwargs.get('enc_mode'))
        f.write(model_string)
        os.chmod(file_name, stat.S_IRUSR)


def _save_together(net_dict, model):
    """Whether graph and parameter save together during save mindir model."""
    data_total = 0
    for param_proto in model.graph.parameter:
        name = param_proto.name[param_proto.name.find(":") + 1:]
        if name in net_dict.keys():
            data_total += sys.getsizeof(net_dict[name].data.get_bytes()) / 1024
        else:
            raise ValueError("There's a mindspore.Parameter that wasn't created in nn.Cell, and mindspore.export() "
                             f"does not support exporting such Parameters. The parameter name is: {name}.\n"
                             "You can find the supported syntax range for mindspore.export() at the following link:\n"
                             "https://www.mindspore.cn/tutorials/zh-CN/master/beginner/save_load.html")
        if data_total > TOTAL_SAVE:
            return False
    return True


def _save_dataset_to_mindir(model, dataset):
    """Save dataset preprocess operations into mindir model."""
    dataset_json = dataset.to_json()
    reverse_dataset = []
    while dataset_json:
        reverse_dataset = [dataset_json] + reverse_dataset
        if len(dataset_json['children']) > 1:
            logger.warning("Need to support dataset_node with more than one child, using child 0 as default.")
        dataset_json = dataset_json['children'][0] if dataset_json['children'] else []

    for op in reverse_dataset:
        if op['op_type'] == 'Map':
            model.preprocessor.op.add()
            model.preprocessor.op[-1].input_columns = json.dumps(op['input_columns'])
            model.preprocessor.op[-1].output_columns = json.dumps(op['output_columns'])
            model.preprocessor.op[-1].op_type = json.dumps(op['op_type'])
            model.preprocessor.op[-1].operations = json.dumps(op['operations'])
            model.preprocessor.op[-1].offload = op['offload'] if 'offload' in op.keys() else False


def async_ckpt_thread_status():
    """
    Get the status of asynchronous save checkpoint thread.

    Note:
        The interface is deprecated from version 2.5 and will be removed in a future version.

    When performing asynchronous save checkpoint, you can determine whether the asynchronous thread is completed.

    Returns:
        bool. ``True`` if asynchronous save checkpoint thread is running,
        ``False`` if asynchronous save checkpoint thread is not executing.

    Examples:
        >>> import mindspore as ms
        >>> ms.async_ckpt_thread_status()
        False
    """
    logger.warning("The interface 'mindspore.async_ckpt_thread_status' is deprecated from version 2.5 "
                   "and will be removed in a future version.")
    thr_list = threading.enumerate()
    return True in [ele.getName() == "asyn_save_ckpt" for ele in thr_list]


def _calculation_net_size(net):
    """Calculate the size of parameters in the network."""
    data_total = 0
    net_dict = net.parameters_dict()
    for name in net_dict:
        data_total += sys.getsizeof(net_dict[name].data.get_bytes()) / 1024

    return data_total


def _load_file_and_convert_name(path, name_map=None, format="ckpt"):
    """
    Load file, during load convert name by name_map.

    Args:
        path (str): The file path.
        name_map (dict): Convert the name of parameter by name_map.
        format (str): The format of the file. Option: 'ckpt', 'safetensors'

    Returns:
        Dict, key is parameter name, value is a Parameter or string.
    """
    if name_map is not None:
        load_func = partial(mindspore.load_checkpoint, format=format)
        return _load_and_transform(path, name_map, load_func)

    return mindspore.load_checkpoint(path, format=format)


def _process_file(file_info):
    """Rrocess file."""
    cur_path, name_map, save_path, file, dst_format = file_info
    if dst_format == "safetensors":
        param_dict = _load_file_and_convert_name(cur_path, name_map, format="ckpt")
        safetensors_filename = file.replace(".ckpt", ".safetensors")
        dst_file = os.path.join(save_path, safetensors_filename)
        mindspore.save_checkpoint(param_dict, dst_file, format='safetensors')
    else:
        param_dict = _load_file_and_convert_name(cur_path, name_map, format="safetensors")
        ckpt_filename = file.replace(".safetensors", ".ckpt")
        dst_file = os.path.join(save_path, ckpt_filename)
        mindspore.save_checkpoint(param_dict, dst_file)


def _gather_all_tasks(file_path, save_path, file_name_regex, name_map, dst_format="ckpt"):
    """gather transform rank together"""
    if dst_format == "ckpt":
        cur_file_suffix = ".safetensors"
    else:
        cur_file_suffix = ".ckpt"

    tasks_list = []
    for root, dirs, _ in os.walk(file_path):
        if root != file_path:
            continue

        rank_dirs = [d for d in dirs if d.startswith('rank')]
        if not rank_dirs:
            if dst_format == "safetensors":
                raise ValueError(
                    f"For 'ckpt_to_safetensors', no directories starting with 'rank' found in {file_path}.")
            if dst_format == "ckpt":
                raise ValueError(
                    f"For 'safetensors_to_ckpt', no directories starting with 'rank' found in {file_path}.")

            raise ValueError(f"For '_gather_all_tasks', error args 'format': {dst_format}.")

        for rank_dir in rank_dirs:
            rank_dir_path = os.path.join(root, rank_dir)
            if save_path is not None:
                dst_root = os.path.join(save_path, os.path.relpath(rank_dir_path, file_path))
            else:
                dst_root = rank_dir_path

            os.makedirs(dst_root, exist_ok=True)

            for file in os.listdir(rank_dir_path):
                if file.endswith(cur_file_suffix) and (file_name_regex is None or re.search(file_name_regex, file)):
                    tasks_list.append((os.path.join(rank_dir_path, file), name_map, dst_root, file, dst_format))

    return tasks_list


def _convert_checkpoint_file(file_path, save_path=None, name_map=None, file_name_regex=None,
                             processes_num=1, dst_format="safetensors"):
    """
    Converts MindSpore checkpoint files format and saves them to `save_path`.
    Safetensors is a reliable and portable machine learning model storage format introduced by Huggingface,
    used for securely storing Tensors with fast speed (zero copy).

    Args:
        file_path (str): Path to the directory containing checkpoint files or a single checkpoint file (.ckpt).
        save_path (str, optional): Directory path where safetensors files will be saved. Default: ``None``.
        name_map (dict, optional): Dictionary mapping original parameter names to new names. Default: ``None``.
        file_name_regex (str, optional): Regular expression used to match the file that needs to be converted.
                                       Default: ``None``.
        processes_num (int, optional): Number of processes to use for parallel processing. Default: 1.
        dst_format (str): dst file format. Default: "safetensors".
    """
    if dst_format == "safetensors":
        src_format = "ckpt"
        src_file_suffix = ".ckpt"
        dst_file_suffix = ".safetensors"
        func_name = "ckpt_to_safetensors"
    else:
        src_format = "safetensors"
        src_file_suffix = ".safetensors"
        dst_file_suffix = ".ckpt"
        func_name = "safetensors_to_ckpt"
    is_dir = os.path.isdir(file_path)
    is_file = os.path.isfile(file_path)
    if not is_dir and not is_file:
        raise ValueError(f"For {func_name}, the input path must be a valid path or file, but got {file_path}")
    if save_path and os.path.splitext(save_path)[1]:
        raise ValueError(f"For {func_name}, the save_path must be a directory, but got '{save_path}'")
    if name_map is not None and not isinstance(name_map, dict):
        raise ValueError(
            f"For {func_name}, the type of 'name_map' must be a directory, but got '{type(name_map)}'")

    if is_dir:
        tasks_list = _gather_all_tasks(file_path, save_path, file_name_regex, name_map, dst_format=dst_format)
        with mp.Pool(processes=processes_num) as pool:
            list(_progress_bar(pool.imap(_process_file, tasks_list), total=len(tasks_list)))
    elif is_file:
        if not file_path.endswith(src_file_suffix):
            raise ValueError(f"For {func_name}, the input file must be a {src_file_suffix} file, but got {file_path}")
        if file_name_regex is not None and not re.findall(file_name_regex, file_path):
            raise ValueError(f"For {func_name}, the input file does not match the regular expression.")
        if save_path and not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)

        param_dict = _load_file_and_convert_name(file_path, name_map, format=src_format)

        file_filename = os.path.basename(file_path).replace(src_file_suffix, dst_file_suffix)
        dst_file = os.path.join(save_path if save_path else os.path.dirname(file_path), file_filename)
        mindspore.save_checkpoint(param_dict, dst_file, format=dst_format)


@_mstx_range_decorator("ckpt_to_safetensors", domain="model_preparation")
def ckpt_to_safetensors(file_path, save_path=None, name_map=None, file_name_regex=None, processes_num=1):
    """
    Converts MindSpore checkpoint files into safetensors format and saves them to `save_path`.
    Safetensors is a reliable and portable machine learning model storage format introduced by Huggingface,
    used for securely storing Tensors with fast speed (zero copy).

    Note:
        The number of multiprocess settings should be set according to the host size, and it is not recommended to set
        it too large, otherwise it may cause freezing.
        The safetensors format does not support the encryption verification function. If the checkpoint file has
        encryption verification enabled, an error will be generated when performing the conversion.
        The safetensors format currently does not support the crc verification function. If the checkpoint file
        contains crc verification information, the crc verification information will be lost after conversion to
        safetensors.

    Args:
        file_path (str): Path to the directory containing checkpoint files or a single checkpoint file (.ckpt).
        save_path (str, optional): Directory path where safetensors files will be saved. Default: ``None``.
        name_map (dict, optional): Dictionary mapping original parameter names to new names. Default: ``None``.
        file_name_regex (str, optional): Regular expression used to match the file that needs to be converted.
            Default: ``None``.
        processes_num (int, optional): Number of processes to use for parallel processing. Default: 1.
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
    _convert_checkpoint_file(file_path, save_path, name_map,
                             file_name_regex, processes_num, "safetensors")


@_mstx_range_decorator("safetensors_to_ckpt", domain="model_preparation")
def safetensors_to_ckpt(file_path, save_path=None, name_map=None, file_name_regex=None, processes_num=1):
    """
    Converts safetensors files into MindSpore checkpoint format and saves them to `save_path`.
    Safetensors is a reliable and portable machine learning model storage format introduced by Huggingface,
    used for securely storing Tensors with fast speed (zero copy).

    Note:
        The number of multiprocess settings is related to the size of the host, and it is not recommended to set it
        too large, otherwise it may cause freezing.

    Args:
        file_path (str): Path to the directory containing safetensors files or a single safetensors file (.safetensors).
        save_path (str, optional): Directory path where checkpoint files will be saved. Default: ``None``.
        name_map (dict, optional): Dictionary mapping original parameter names to new names. Default: ``None``.
        file_name_regex (str, optional): Regular expression used to match the file that needs to be converted.
                                   Default: ``None``.
        processes_num (int, optional): Number of processes to use for parallel processing. Default: 1.

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
    _convert_checkpoint_file(file_path, save_path, name_map,
                             file_name_regex, processes_num, "ckpt")


def restore_group_info_list(group_info_file_name):
    """
    Build rank list, the checkpoint of ranks in the rank list has the same contents with the local rank
    who saves the `group_info_file_name`. To save the group info file, please export GROUP_INFO_FIL
    environment variables like "export GROUP_INFO_FILE=/data/group_info.pb".
    """
    return new_restore_group_info_list(group_info_file_name)


@_mstx_range_decorator("load_distributed_checkpoint", domain="model_preparation")
def load_distributed_checkpoint(network, checkpoint_filenames=None, predict_strategy=None,
                                train_strategy_filename=None, strict_load=False, dec_key=None, dec_mode='AES-GCM',
                                format='ckpt', unified_safetensors_dir=None, dst_safetensors_dir=None, rank_id=None,
                                output_format='safetensors', name_map=None, max_process_num=64,
                                return_param_dict=False):
    """ Load checkpoint into net for distributed predication. Used in the case of distributed inference. """
    new_load_distributed_checkpoint(network, checkpoint_filenames, predict_strategy,
                                    train_strategy_filename, strict_load, dec_key, dec_mode,
                                    format, unified_safetensors_dir, dst_safetensors_dir, rank_id,
                                    output_format, name_map, max_process_num,
                                    return_param_dict)


def merge_sliced_parameter(sliced_parameters, strategy=None):
    """ Merge parameter slices into one parameter. Used in the case of distributed inference. """
    return new_merge_sliced_parameter(sliced_parameters, strategy)


def build_searched_strategy(strategy_filename):
    """ Build strategy of every parameter in network. Used in the case of distributed inference. """
    return new_build_searched_strategy(strategy_filename)
