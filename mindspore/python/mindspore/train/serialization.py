# Copyright 2020-2021 Huawei Technologies Co., Ltd
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

import copy
import json
import os
import shutil
import stat
import threading
from threading import Thread, Lock
from collections import defaultdict, OrderedDict
from functools import wraps
from io import BytesIO

import math
import sys
import time
import numpy as np

from mindspore.train.checkpoint_pb2 import Checkpoint
from mindspore.train.mind_ir_pb2 import ModelProto as mindir_model
from mindspore.train.print_pb2 import Print

import mindspore
import mindspore.nn as nn
from mindspore import context
from mindspore import log as logger
from mindspore._checkparam import check_input_data, check_input_dataset, Validator
from mindspore.common import dtype as mstype
from mindspore.common.api import _cell_graph_executor as _executor
from mindspore.common.api import _MindsporeFunctionExecutor
from mindspore.common.api import _get_parameter_layout
from mindspore.common.initializer import initializer, One
from mindspore.common.parameter import Parameter
from mindspore.common.tensor import Tensor
from mindspore.common._utils import is_shape_unknown
from mindspore.communication.management import get_rank, get_group_size
from mindspore.compression.export import quant_export
from mindspore.parallel._cell_wrapper import get_allgather_cell
from mindspore.parallel._tensor import _load_tensor, _get_tensor_strategy, _get_tensor_slice_index
from mindspore.parallel._tensor import _reshape_param_data, _reshape_param_data_with_weight
from mindspore.parallel._utils import _infer_rank_list, _remove_repeated_slices, _is_in_auto_parallel_mode
from mindspore.parallel._parallel_serialization import _convert_to_list, _convert_to_layout, _build_searched_strategy, \
    _restore_group_info_list
from mindspore.train._utils import read_proto
from mindspore._c_expression import load_mindir, _encrypt, _decrypt, _is_cipher_file, dynamic_obfuscate_mindir
from ..ops.operations._opaque_predicate_registry import add_opaque_predicate, clean_funcs

tensor_to_ms_type = {"Int8": mstype.int8, "UInt8": mstype.uint8, "Int16": mstype.int16, "UInt16": mstype.uint16,
                     "Int32": mstype.int32, "UInt32": mstype.uint32, "Int64": mstype.int64, "UInt64": mstype.uint64,
                     "Float16": mstype.float16, "Float32": mstype.float32, "Float64": mstype.float64,
                     "Bool": mstype.bool_, "str": mstype.string}

tensor_to_np_type = {"Int8": np.int8, "UInt8": np.uint8, "Int16": np.int16, "UInt16": np.uint16,
                     "Int32": np.int32, "UInt32": np.uint32, "Int64": np.int64, "UInt64": np.uint64,
                     "Float16": np.float16, "Float32": np.float32, "Float64": np.float64, "Bool": np.bool_, "str": "U"}

mindir_to_tensor_type = {1: mstype.float32, 2: mstype.uint8, 3: mstype.int8, 4: mstype.uint16,
                         5: mstype.int16, 6: mstype.int32, 7: mstype.int64, 10: mstype.float16,
                         11: mstype.float64, 12: mstype.uint32, 13: mstype.uint64}

_ckpt_mutex = Lock()

# unit is KB
SLICE_SIZE = 512 * 1024
PROTO_LIMIT_SIZE = 1024 * 1024 * 2
TOTAL_SAVE = 1024 * 1024
PARAMETER_SPLIT_SIZE = 1024 * 1024 * 1024
ENCRYPT_BLOCK_SIZE = 64 * 1024
INT_64_MAX = 9223372036854775807


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
                param.set_data(new_tensor)
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
    float_type = (mstype.float16, mstype.float32, mstype.float64)
    int_type = (mstype.int8, mstype.int16, mstype.int32, mstype.int64)
    if not strict_load and ({param.data.dtype, new_param.data.dtype}.issubset(float_type) or
                            {param.data.dtype, new_param.data.dtype}.issubset(int_type)):
        logger.warning(f"The type of {new_param.name}:{new_param.data.dtype} in 'parameter_dict' is different from "
                       f"the type of it in 'net':{param.data.dtype}, then the type convert from "
                       f"{new_param.data.dtype} to {param.data.dtype} in the network.")
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


def _exec_save(ckpt_file_name, data_list, enc_key=None, enc_mode="AES-GCM"):
    """Execute the process of saving checkpoint into file."""
    try:
        with _ckpt_mutex:
            if os.path.exists(ckpt_file_name):
                os.chmod(ckpt_file_name, stat.S_IWUSR)
                os.remove(ckpt_file_name)
            with open(ckpt_file_name, "ab") as f:
                if enc_key is not None:
                    plain_data = BytesIO()

                for name, value in data_list.items():
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
                            f.write(checkpoint_list.SerializeToString())
                        else:
                            plain_data.write(checkpoint_list.SerializeToString())

                if enc_key is not None:
                    plain_data.seek(0)
                    max_block_size = ENCRYPT_BLOCK_SIZE * 1024
                    block_data = plain_data.read(max_block_size)
                    while block_data:
                        f.write(_encrypt(block_data, len(block_data), enc_key, len(enc_key), enc_mode))
                        block_data = plain_data.read(max_block_size)

                os.chmod(ckpt_file_name, stat.S_IRUSR)

    except BaseException as e:
        logger.critical("Failed to save the checkpoint file %s. Maybe don't have the permission to write files, "
                        "or the disk space is insufficient and so on.", ckpt_file_name)
        raise e


def _check_save_obj_and_ckpt_file_name(save_obj, ckpt_file_name):
    """Check save_obj and ckpt_file_name for save_checkpoint."""
    if not isinstance(save_obj, nn.Cell) and not isinstance(save_obj, list):
        raise TypeError("For 'save_checkpoint', the parameter 'save_obj' must be nn.Cell or list, "
                        "but got {}.".format(type(save_obj)))
    if not isinstance(ckpt_file_name, str):
        raise TypeError("For 'save_checkpoint', the parameter {} for checkpoint file name is invalid,"
                        "'ckpt_file_name' must be "
                        "string, but got {}.".format(ckpt_file_name, type(ckpt_file_name)))
    ckpt_file_name = os.path.realpath(ckpt_file_name)
    if os.path.isdir(ckpt_file_name):
        raise IsADirectoryError("For 'save_checkpoint', the parameter `ckpt_file_name`: {} is a directory, "
                                "it must be a file name.".format(ckpt_file_name))
    if not ckpt_file_name.endswith('.ckpt'):
        ckpt_file_name += ".ckpt"
    return ckpt_file_name


def save_checkpoint(save_obj, ckpt_file_name, integrated_save=True,
                    async_save=False, append_dict=None, enc_key=None, enc_mode="AES-GCM"):
    """
    Save checkpoint to a specified file.

    Args:
        save_obj (Union[Cell, list]): The cell object or data list(each element is a dictionary, like
                                      [{"name": param_name, "data": param_data},...], the type of
                                      param_name would be string, and the type of param_data would
                                      be parameter or Tensor).
        ckpt_file_name (str): Checkpoint file name. If the file name already exists, it will be overwritten.
        integrated_save (bool): Whether to integrated save in automatic model parallel scene. Default: True
        async_save (bool): Whether to open an independent thread to save the checkpoint file. Default: False
        append_dict (dict): Additional information that needs to be saved. The key of dict must be str, the value
                            of dict must be one of int, float, bool, string, Parameter or Tensor. Default: None.
        enc_key (Union[None, bytes]): Byte type key used for encryption. If the value is None, the encryption
                                      is not required. Default: None.
        enc_mode (str): This parameter is valid only when enc_key is not set to None. Specifies the encryption
                        mode, currently supports 'AES-GCM' and 'AES-CBC' and 'SM4-CBC'. Default: 'AES-GCM'.

    Raises:
        TypeError: If the parameter save_obj is not `nn.Cell` or list type. And if the parameter `integrated_save`
                   and `async_save` are not bool type. If the parameter ckpt_file_name is not string type.

    Examples:
        >>> import mindspore as ms
        >>>
        >>> net = Net()
        >>> ms.save_checkpoint(net, "lenet.ckpt")
    """
    ckpt_file_name = _check_save_obj_and_ckpt_file_name(save_obj, ckpt_file_name)
    integrated_save = Validator.check_bool(integrated_save)
    async_save = Validator.check_bool(async_save)
    append_dict = _check_append_dict(append_dict)
    enc_key = Validator.check_isinstance('enc_key', enc_key, (type(None), bytes))
    enc_mode = Validator.check_isinstance('enc_mode', enc_mode, str)

    logger.info("Execute the process of saving checkpoint files.")

    if isinstance(save_obj, nn.Cell):
        parameter_layout_dict = save_obj.parameter_layout_dict
        if _is_in_auto_parallel_mode() and not parameter_layout_dict:
            parameter_layout_dict = _get_parameter_layout()
        save_obj.init_parameters_data()
        param_dict = OrderedDict()
        for _, param in save_obj.parameters_and_names():
            param_dict[param.name] = param
        param_list = []
        for (key, value) in param_dict.items():
            each_param = {"name": key}
            param_data = Tensor(value.data.asnumpy())

            # in automatic model parallel scenario, some parameters were split to all the devices,
            # which should be combined before saving
            if key in parameter_layout_dict:
                param_data = _get_merged_param_data(save_obj, parameter_layout_dict, key, param_data, integrated_save)

            each_param["data"] = param_data
            param_list.append(each_param)
        save_obj = param_list

    if append_dict:
        append_info_list = []
        for k_name, value in append_dict.items():
            if not isinstance(value, str):
                value = Tensor(value)
            append_info_list.append({"name": k_name, "data": value})
            save_obj.extend(append_info_list)

    data_list = OrderedDict()
    with _ckpt_mutex:
        for param in save_obj:
            key = param["name"]
            data_list[key] = []
            if isinstance(param["data"], str):
                data_list[key].append([0])
                data_list[key].append('str')
                data = np.array(param["data"])
                data_list[key].append(data)
            else:
                if isinstance(param["data"], Parameter):
                    param["data"].init_data()
                dims = []
                if param['data'].shape == ():
                    dims.append(0)
                else:
                    for dim in param['data'].shape:
                        dims.append(dim)
                data_list[key].append(dims)
                tensor_type = str(param["data"].dtype)
                data_list[key].append(tensor_type)
                data = param["data"].asnumpy().reshape(-1)
                data_list[key].append(data)

    if async_save:
        data_copy = copy.deepcopy(data_list)
        thr = Thread(target=_exec_save, args=(ckpt_file_name, data_copy, enc_key, enc_mode), name="asyn_save_ckpt")
        thr.start()
    else:
        _exec_save(ckpt_file_name, data_list, enc_key, enc_mode)

    logger.info("Saving checkpoint process is finished.")


def _check_append_dict(append_dict):
    """Check the argument append_dict for save_checkpoint."""
    if append_dict is None:
        return append_dict
    if not isinstance(append_dict, dict):
        raise TypeError("For 'save_checkpoint', the argument 'append_dict' must be dict, but got "
                        "{}.".format(type(append_dict)))
    for key, value in append_dict.items():
        if not isinstance(key, str) or not isinstance(value, (int, float, bool, str, Parameter, Tensor)):
            raise TypeError(f"For 'save_checkpoint', the type of dict 'append_info' must be key: string, "
                            f"value: int, float or bool, but got key: {type(key)}, value: {type(value)}")
    return append_dict


def _check_load_obfuscate(**kwargs):
    if 'obf_func' in kwargs.keys():
        customized_func = _check_customized_func(kwargs.get('obf_func'))
        clean_funcs()
        add_opaque_predicate(customized_func.__name__, customized_func)
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
            - dec_mode (Union[str, function]): Specifies the decryption mode, to take effect when dec_key is set.

              - Option: 'AES-GCM', 'AES-CBC', 'SM4-CBC' or customized decryption. Default: 'AES-GCM'.
              - For details of using the customized decryption, please check the `tutorial
                <https://mindspore.cn/mindarmour/docs/en/r1.9/model_encrypt_protection.html>`_.

            - obf_func (function): A python function used for loading obfuscated MindIR model, which can refer to
              `obfuscate_model()
              <https://www.mindspore.cn/docs/en/r2.0.0-alpha/api_python/mindspore/mindspore.obfuscate_model.html>` .

    Returns:
        GraphCell, a compiled graph that can executed by `GraphCell`.

    Raises:
        ValueError: MindIR file does not exist or `file_name` is not a string.
        RuntimeError: Failed to parse MindIR file.

    Examples:
        >>> import numpy as np
        >>> import mindspore as ms
        >>> import mindspore.nn as nn
        >>> from mindspore import Tensor
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
    """
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

    # set customized functions for dynamic obfuscation
    obfuscated = _check_load_obfuscate(**kwargs)

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
                            decrypt=dec_func, obfuscated=obfuscated)
    else:
        graph = load_mindir(file_name, obfuscated=obfuscated)

    if graph is None:
        if _is_cipher_file(file_name):
            raise RuntimeError("Load MindIR failed. The file may be encrypted and decrypt failed, you "
                               "can check whether the values of the arguments 'dec_key' and 'dec_mode'"
                               " are the same as when exported MindIR file, or check the file integrity.")
        raise RuntimeError("Load MindIR failed.")
    return graph


def _check_param_type(param_config, key, target_type, requested):
    """check type of parameters"""
    if key in param_config:
        if not isinstance(param_config[key], target_type):
            raise TypeError("The type of {} must be {}, but got {}.".format(key, target_type, type(param_config[key])))
        if key == 'obf_password':
            if param_config[key] > INT_64_MAX or param_config[key] <= 0:
                raise ValueError(
                    "'obf_password' must be in (0, INT_64_MAX({})], but got {}.".format(INT_64_MAX, param_config[key]))
        return param_config[key]
    if requested:
        raise ValueError("The parameter {} is requested, but not got.".format(key))
    if key == "obf_password":
        return 0
    return None


def _check_customized_func(customized_func):
    """ check customized function of dynamic obfuscation """
    if not callable(customized_func):
        raise TypeError(
            "'customized_func' must be a function, but not got {}.".format(type(customized_func)))
    # test customized_func
    try:
        func_result = customized_func(1.0, 1.0)
    except Exception as ex:
        raise TypeError("customized_func must be a function with two inputs, but got exception: {}".format(ex))
    else:
        if not isinstance(func_result, bool):
            raise TypeError("Return value of customized_func must be boolean, but got: {}".format(type(func_result)))
    return customized_func


def _check_obfuscate_params(obf_config):
    """check obfuscation parameters, including obf_password, obf_ratio, customized_func"""
    if 'obf_password' not in obf_config.keys() and 'customized_func' not in obf_config.keys():
        raise ValueError(
            "At least one of 'obf_password' or 'customized_func' must be set in obf_config, but got None of them.")
    obfuscate_type = _check_param_type(obf_config, "type", str, False)
    if obfuscate_type not in (None, "dynamic"):
        raise ValueError("Only 'dynamic' type is supported by now, but got {}.".format(obfuscate_type))
    if ('obf_ratio' in obf_config) and isinstance(obf_config['obf_ratio'], str):
        if obf_config['obf_ratio'] not in ["small", "medium", "large"]:
            raise ValueError("'obf_ratio' can only be 'small', 'medium', 'large' or float, but got {}.".format(
                obf_config['obf_ratio']))
        ratio_dict = {"small": 0.1, "medium": 0.3, "large": 0.6}
        obf_config['obf_ratio'] = ratio_dict.get(obf_config['obf_ratio'])
    obf_ratio = _check_param_type(obf_config, "obf_ratio", float, True)
    if (obf_ratio <= 0) or (obf_ratio > 1):
        raise ValueError("'obf_ratio' must be in (0, 1] if it is a float, but got {}.".format(obf_config['obf_ratio']))
    customized_funcs = []
    if 'customized_func' in obf_config.keys():
        customized_funcs.append(_check_customized_func(obf_config['customized_func']))
    obf_password = _check_param_type(obf_config, "obf_password", int, False)
    return obf_ratio, customized_funcs, obf_password


def obfuscate_model(obf_config, **kwargs):
    """
    Obfuscate a model of MindIR format. Obfuscation means changing the struct of a network without affecting its
    predict correctness. The obfuscated model can prevent attackers from stealing the model.

    Args:
        obf_config (dict): obfuscation config.

            - type (str): The type of obfuscation, only 'dynamic' is supported until now.
            - original_model_path (str): The path of MindIR format model that need to be obfuscated. If the original
              model is encrypted, then enc_key and enc_mode should be provided.
            - save_model_path (str): The path to save the obfuscated model.
            - model_inputs (list(Tensor)): The inputs of the original model, the values of Tensor can be random, which
              is the same as using `export()`.
            - obf_ratio (Union(float, str)): The ratio of nodes in original model that would be obfuscated. `obf_ratio`
              should be in range of (0, 1] or in ["small", "medium", "large"].
            - customized_func (function): A python function used for customized function mode, which used for control
              the switch branch of obfuscation structure. The outputs of customized_func should be boolean. This
              function needs to ensure that its result is constant for any input. Users can refer to opaque
              predicates. If customized_func is set, then it should be passed to `load()` interface when loading
              obfuscated model.
            - obf_password (int): A password used for password mode, which should be in (0, 9223372036854775807]. If
              obf_password is set, then it should be passed to `nn.GraphCell()` interface when loading obfuscated
              model. It should be noted that at least one of 'customized_func' or 'obf_password' should be set, and
              'obf_password' mode would be applied if both of them are set.

        kwargs (dict): Configuration options dictionary.

            - enc_key (bytes): Byte type key used for encryption. The valid length is 16, 24, or 32.
            - enc_mode (str): Specifies the encryption mode, to take effect when dec_key is set.
              Option: 'AES-GCM' | 'AES-CBC' | 'SM4-CBC'. Default: 'AES-GCM'.

    Raises:
        TypeError: If obf_config is not a dict.
        ValueError: If enc_key is passed and enc_mode is not in ["AES-GCM", "AES-CBC", "SM4-CBC"].
        ValueError: If original_model_path is not provided in obf_config.
        ValueError: If the model saved in original_model_path has been obfuscated.
        ValueError: If save_model_path is not provided in obf_config.
        ValueError: If obf_ratio is not provided in obf_config.
        ValueError: If both customized_func and obf_password are not provided in obf_config.
        ValueError: If both obf_password is not in (0, 9223372036854775807].
        ValueError: If file_path is not exist or file_path is not end with '.mindir'.

    Examples:
        >>> obf_config = {'original_model_path': "./net.mindir",
        ...          'save_model_path': "./obf_net",
        ...          'model_inputs': [input1, ],
        ...          'obf_ratio': 0.1, 'obf_password': 173262358423}
        >>> obfuscate_model(obf_config)
        >>> obf_func = load("obf_net.mindir")
        >>> obf_net = nn.GraphCell(obf_func, obf_password=173262358423)
        >>> print(obf_net(input1).asnumpy())
    """
    if not isinstance(obf_config, dict):
        raise TypeError("'obf_config' must be a dict, but got {}.".format(type(obf_config)))
    file_path = _check_param_type(obf_config, "original_model_path", str, True)
    if not file_path.endswith(".mindir"):
        raise ValueError("For 'obfuscate_model', the argument 'file_path'(MindIR file) should end with '.mindir', "
                         "please input the correct 'file_path'.")
    if not os.path.exists(file_path):
        raise ValueError("For 'obfuscate_model', the argument 'file_path'(MindIR file) does not exist, "
                         "please check whether the 'file_path' is correct.")
    saved_path = _check_param_type(obf_config, "save_model_path", str, True)
    model_inputs = _check_param_type(obf_config, "model_inputs", list, True)
    for item in model_inputs:
        if not isinstance(item, Tensor):
            raise TypeError("The item in 'model_inputs' must be Tensor, but got {}.".format(type(item)))
        if -1 in item.shape:
            raise ValueError(
                "Dynamic shape input is not supported now, but got the shape of inputs: {}.".format(item.shape))
    obf_ratio, customized_funcs, obf_password = _check_obfuscate_params(obf_config)
    if customized_funcs and obf_password > 0:
        logger.warning("Although 'customized_func' and 'obf_password' are set, the 'obf_password' mode would be"
                       " applied, remember to set 'obf_password' when loading obfuscated model.")

    if obf_password == 0:  # apply customized_func mode
        clean_funcs()
        for func in customized_funcs:
            add_opaque_predicate(func.__name__, func)
        append_password = 0
    else:
        seed_max = 2 ** 32 - 1
        int_max = 2 ** 31 - 1
        np.random.seed(obf_password % seed_max)
        append_password = np.random.randint(int_max)
        obf_password %= int_max

    if 'enc_key' in kwargs.keys():
        enc_key = Validator.check_isinstance('enc_key', kwargs.get('enc_key'), bytes)
        enc_mode = "AES-GCM"
        if 'enc_mode' in kwargs.keys():
            enc_mode = Validator.check_isinstance('enc_mode', kwargs.get('enc_mode'), str)
            if enc_mode not in ["AES-GCM", "AES-CBC", "SM4-CBC"]:
                raise ValueError(
                    "Only MindIR files that encrypted with 'AES-GCM', 'AES-CBC' or 'SM4-CBC' is supported for"
                    "obfuscate_model(), but got {}.".format(enc_mode))
        obf_graph = dynamic_obfuscate_mindir(file_name=file_path, obf_ratio=obf_ratio, obf_password=obf_password,
                                             append_password=append_password, dec_key=enc_key, key_len=len(enc_key),
                                             dec_mode=enc_mode)
    else:
        obf_graph = dynamic_obfuscate_mindir(file_name=file_path, obf_ratio=obf_ratio, obf_password=obf_password,
                                             append_password=append_password)

    obf_net = nn.GraphCell(obf_graph)
    if obf_password != 0:
        y_tensor = Tensor(np.ones((1, 1)).astype(np.int32))
        append_y_tensor = Tensor(np.ones((1, 1)).astype(np.int32))
        model_inputs += [y_tensor, append_y_tensor]
    export(obf_net, *model_inputs, file_name=saved_path, file_format="MINDIR", **kwargs)


def load_checkpoint(ckpt_file_name, net=None, strict_load=False, filter_prefix=None,
                    dec_key=None, dec_mode="AES-GCM", specify_prefix=None):
    """
    Load checkpoint info from a specified file.

    Note:
        1. `specify_prefix` and `filter_prefix` do not affect each other.
        2. If none of the parameters are loaded from checkpoint file, it will throw ValueError.

    Args:
        ckpt_file_name (str): Checkpoint file name.
        net (Cell): The network where the parameters will be loaded. Default: None
        strict_load (bool): Whether to strict load the parameter into net. If False, it will load parameter
                            into net when parameter name's suffix in checkpoint file is the same as the
                            parameter in the network. When the types are inconsistent perform type conversion
                            on the parameters of the same type, such as float32 to float16. Default: False.
        filter_prefix (Union[str, list[str], tuple[str]]): Parameters starting with the filter_prefix
            will not be loaded. Default: None.
        dec_key (Union[None, bytes]): Byte type key used for decryption. If the value is None, the decryption
                                      is not required. Default: None.
        dec_mode (str): This parameter is valid only when dec_key is not set to None. Specifies the decryption
                        mode, currently supports 'AES-GCM' and 'AES-CBC' and 'SM4-CBC'. Default: 'AES-GCM'.
        specify_prefix (Union[str, list[str], tuple[str]]): Parameters starting with the specify_prefix
            will be loaded. Default: None.

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
        >>> param_dict = ms.load_checkpoint(ckpt_file_name, filter_prefix="conv1", specify_prefix="conv", )
        >>> print(param_dict["conv2.weight"])
        Parameter (name=conv2.weight, shape=(16, 6, 5, 5), dtype=Float32, requires_grad=True)
    """
    ckpt_file_name = _check_ckpt_file_name(ckpt_file_name)
    specify_prefix = _check_prefix(specify_prefix)
    filter_prefix = _check_prefix(filter_prefix)
    dec_key = Validator.check_isinstance('dec_key', dec_key, (type(None), bytes))
    dec_mode = Validator.check_isinstance('dec_mode', dec_mode, str)
    logger.info("Execute the process of loading checkpoint files.")
    checkpoint_list = _parse_ckpt_proto(ckpt_file_name, dec_key, dec_mode)

    parameter_dict = {}
    try:
        param_data_list = []
        for element_id, element in enumerate(checkpoint_list.value):
            if not _whether_load_param(specify_prefix, filter_prefix, element.tag):
                continue
            data = element.tensor.tensor_content
            data_type = element.tensor.tensor_type
            np_type = tensor_to_np_type.get(data_type)
            ms_type = tensor_to_ms_type[data_type]
            if data_type == 'str':
                str_length = int(len(data) / 4)
                np_type = np_type + str(str_length)
            element_data = np.frombuffer(data, np_type)
            param_data_list.append(element_data)
            if (element_id == len(checkpoint_list.value) - 1) or \
                    (element.tag != checkpoint_list.value[element_id + 1].tag):
                param_data = np.concatenate((param_data_list), axis=0)
                param_data_list.clear()
                dims = element.tensor.dims
                if dims == [0] and data_type == 'str':
                    parameter_dict[element.tag] = str(element_data[0])
                else:
                    if dims == [0] and 'Float' in data_type:
                        param_data = float(param_data[0])
                    if dims == [0] and 'Int' in data_type:
                        param_data = int(param_data[0])
                    if dims not in ([0], [1]):
                        param_data = param_data.reshape(list(dims))
                    parameter_dict[element.tag] = Parameter(Tensor(param_data, ms_type), name=element.tag)

        logger.info("Loading checkpoint files process is finished.")

    except BaseException as e:
        logger.critical("Failed to load the checkpoint file '%s'.", ckpt_file_name)
        raise ValueError(e.__str__() + "\nFor 'load_checkpoint', "
                                       "failed to load the checkpoint file {}.".format(ckpt_file_name)) from e

    if not parameter_dict:
        raise ValueError(f"The loaded parameter dict is empty after filter or specify, please check whether "
                         f"'filter_prefix' or 'specify_prefix' are set correctly.")

    if net is not None:
        load_param_into_net(net, parameter_dict, strict_load)

    return parameter_dict


def _check_ckpt_file_name(ckpt_file_name):
    """Check function load_checkpoint's cket_file_name."""
    if not isinstance(ckpt_file_name, str):
        raise TypeError("For 'load_checkpoint', the argument 'ckpt_file_name' must be string, "
                        "but got {}.".format(type(ckpt_file_name)))

    if ckpt_file_name[-5:] != ".ckpt":
        raise ValueError("For 'load_checkpoint', the checkpoint file should end with '.ckpt', please "
                         "input the correct 'ckpt_file_name'.")

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


def _parse_ckpt_proto(ckpt_file_name, dec_key, dec_mode):
    """Parse checkpoint protobuf."""
    checkpoint_list = Checkpoint()
    try:
        if dec_key is None:
            with open(ckpt_file_name, "rb") as f:
                pb_content = f.read()
        else:
            pb_content = _decrypt(ckpt_file_name, dec_key, len(dec_key), dec_mode)
            if pb_content is None:
                raise ValueError("For 'load_checkpoint', failed to decrypt the checkpoint file.")
        checkpoint_list.ParseFromString(pb_content)
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


def load_param_into_net(net, parameter_dict, strict_load=False):
    """
    Load parameters into network, return parameter list that are not loaded in the network.

    Args:
        net (Cell): The network where the parameters will be loaded.
        parameter_dict (dict): The dictionary generated by load checkpoint file,
                               it is a dictionary consisting of key: parameters's name, value: parameter.
        strict_load (bool): Whether to strict load the parameter into net. If False, it will load parameter
                            into net when parameter name's suffix in checkpoint file is the same as the
                            parameter in the network. When the types are inconsistent perform type conversion
                            on the parameters of the same type, such as float32 to float16. Default: False.

    Returns:
        List, the parameter name which are not loaded into the network.

    Raises:
        TypeError: Argument is not a Cell, or parameter_dict is not a Parameter dictionary.

    Examples:
        >>> import mindspore as ms
        >>>
        >>> net = Net()
        >>> ckpt_file_name = "./checkpoint/LeNet5-1_32.ckpt"
        >>> param_dict = ms.load_checkpoint(ckpt_file_name, filter_prefix="conv1")
        >>> param_not_load = ms.load_param_into_net(net, param_dict)
        >>> print(param_not_load)
        ['conv1.weight']
    """
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
        if not isinstance(key, str) or not isinstance(value, (Parameter, str)):
            logger.critical("Load parameters into net failed.")
            msg = ("For 'parameter_dict', the element in the argument 'parameter_dict' should be a "
                   "'str' and 'Parameter' , but got {} and {}.".format(type(key), type(value)))
            raise TypeError(msg)

    strict_load = Validator.check_bool(strict_load)
    logger.info("Execute the process of loading parameters into net.")
    net.init_parameters_data()
    param_not_load = []
    for _, param in net.parameters_and_names():
        if param.name in parameter_dict:
            new_param = copy.deepcopy(parameter_dict[param.name])
            _update_param(param, new_param, strict_load)
        else:
            param_not_load.append(param.name)

    if param_not_load and not strict_load:
        _load_dismatch_prefix_params(net, parameter_dict, param_not_load, strict_load)

    logger.debug("Params not matched(in net but not in parameter_dict):")
    for param_name in param_not_load:
        logger.debug("%s", param_name)

    logger.info("Loading parameters into net is finished.")
    if param_not_load:
        logger.warning("For 'load_param_into_net', "
                       "{} parameters in the 'net' are not loaded, because they are not in the "
                       "'parameter_dict', please check whether the network structure is consistent "
                       "when training and loading checkpoint.".format(len(param_not_load)))
        for param_name in param_not_load:
            logger.warning("{} is not loaded.".format(param_name))
    return param_not_load


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
    if graph_pb:
        with open(file_name, "wb") as f:
            os.chmod(file_name, stat.S_IRUSR | stat.S_IWUSR)
            f.write(graph_pb)


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
    if len(layout) < 6:
        logger.info("The layout dict does not contain the key %s", param_name)
        return param_data

    dev_mat = layout[0]
    tensor_map = layout[1]
    uniform_split = layout[4]
    opt_shard_group = layout[5]

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
            if context.get_auto_parallel_context("pipeline_stages") > 1:
                raise RuntimeError("Pipeline Parallel don't support Integrated save checkpoint now.")
            if uniform_split == 0:
                raise RuntimeError("For 'save_checkpoint' and in automatic model parallel scene, when set "
                                   "'integrated_save' to True, the checkpoint will be integrated save, it "
                                   "is only supports uniform split tensor now.")
            # while any dim is not equal to -1, means param is split and needs to be merged
            # pipeline parallel need to be supported here later
            if mp_weight:
                allgather_net = get_allgather_cell(opt_shard_group, bool(opt_shard_group))
                object.__setattr__(allgather_net, "keep_input_unchanged", True)
            elif opt_shard_group:
                allgather_net = get_allgather_cell(opt_shard_group, False)
        elif opt_shard_group and context.get_auto_parallel_context("optimizer_weight_shard_aggregated_save"):
            allgather_net = get_allgather_cell(opt_shard_group, False)
        net.parallel_parameter_merge_net_dict[param_name] = allgather_net
    if allgather_net:
        param_data = allgather_net(param_data)
    if mp_weight and integrated_save:
        param_data = _reshape_param_data(param_data, dev_mat, tensor_map)
    return param_data


def _fill_param_into_net(net, parameter_list):
    """
    Fills parameter_list into net.

    Args:
        net (Cell): train network.
        parameter_list (list): parameters list from ge callback.
    """
    parameter_dict = {}
    for each_param in parameter_list:
        param_name = each_param["name"]
        if isinstance(each_param["data"], Parameter):
            each_param["data"].init_data()
        np_val = each_param["data"].asnumpy()
        if np_val.shape == (1,):
            parameter_dict[param_name] = Parameter(np_val, name=param_name)
        elif np_val.shape == ():
            parameter_dict[param_name] = Parameter(Tensor(np_val.tolist(), mstype.pytype_to_dtype(np_val.dtype)),
                                                   name=param_name)
        else:
            parameter_dict[param_name] = Parameter(Tensor(np_val), name=param_name)

    load_param_into_net(net, parameter_dict)


def export(net, *inputs, file_name, file_format, **kwargs):
    """
    Export the MindSpore network into an offline model in the specified format.

    Note:
        1. When exporting AIR, ONNX format, the size of a single tensor can not exceed 2GB.
        2. When file_name does not have a suffix, the system will automatically add one according to the file_format.
        3. Exporting functions decorated with 'jit' to mindir format is supported.
        4. When exporting a function decorated with 'jit', the function should not involve class properties in
           calculations.

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
              for MindSpore models.

        kwargs (dict): Configuration options dictionary.

            - quant_mode (str): If the network is a quantization aware training network, the quant_mode should
              be set to "QUANT", else the quant_mode should be set to "NONQUANT".
            - mean (float): The mean of input data after preprocessing, used for quantizing the first layer of network.
              Default: 127.5.
            - std_dev (float): The variance of input data after preprocessing,
              used for quantizing the first layer of the network. Default: 127.5.
            - enc_key (byte): Byte-type key used for encryption. The valid length is 16, 24, or 32.
            - enc_mode (Union[str, function]): Specifies the encryption mode, to take effect when enc_key is set.

              - For 'AIR' and 'ONNX' models, only customized encryption is supported.
              - For 'MINDIR', all options are supported. Option: 'AES-GCM', 'AES-CBC', 'SM4-CBC'
                or Customized encryption.
                Default: 'AES-GCM'.
              - For details of using the customized encryption, please check the `tutorial
                <https://mindspore.cn/mindarmour/docs/en/r1.9/model_encrypt_protection.html>`_.

            - dataset (Dataset): Specifies the preprocessing method of the dataset, which is used to import the
              preprocessing of the dataset into MindIR.

            - obf_config (dict): obfuscation config.

              - type (str): The type of obfuscation, only 'dynamic' is supported until now.
              - obf_ratio (float, str): The ratio of nodes in original model that would be obfuscated. `obf_ratio`
                should be in range of (0, 1] or in ["small", "medium", "large"].
              - customized_func (function): A python function used for customized function mode, which used for control
                the switch branch of obfuscation structure. The outputs of customized_func should be boolean. This
                function needs to ensure that its result is constant for any input. Users can refer to opaque
                predicates. If customized_func is set, then it should be passed to `load()` interface when loading
                obfuscated model.
              - obf_password (int): A password used for password mode, which should be in (0, 9223372036854775807]. If
                obf_password is set, then it should be passed to `nn.GraphCell()` interface when loading obfuscated
                model. It should be noted that at least one of 'customized_func' or 'obf_password' should be set, and
                'obf_password' mode would be applied if both of them are set.
    Examples:
        >>> import mindspore as ms
        >>> import numpy as np
        >>> from mindspore import Tensor
        >>>
        >>> net = LeNet()
        >>> input_tensor = Tensor(np.ones([1, 1, 32, 32]).astype(np.float32))
        >>> ms.export(net, input_tensor, file_name='lenet', file_format='MINDIR')
    """
    supported_formats = ['AIR', 'ONNX', 'MINDIR']
    if file_format not in supported_formats:
        raise ValueError(f"For 'export', 'file_format' must be one of {supported_formats}, but got {file_format}.")
    Validator.check_file_name_by_regular(file_name)
    logger.info("exporting model file:%s format:%s.", file_name, file_format)

    if check_input_dataset(*inputs, dataset_type=mindspore.dataset.Dataset):
        if len(inputs) != 1:
            raise RuntimeError(f"You can only serialize one dataset into MindIR, got " + str(len(inputs)) + " datasets")
        shapes, types, columns = inputs[0].output_shapes(), inputs[0].output_types(), inputs[0].get_col_names()
        kwargs['dataset'] = inputs[0]
        only_support_col = "image"

        inputs_col = list()
        for c, s, t in zip(columns, shapes, types):
            if only_support_col != c:
                continue
            inputs_col.append(Tensor(np.random.uniform(-1.0, 1.0, size=s).astype(t)))
        if not inputs_col:
            raise RuntimeError(f"Only supports parse \"image\" column from dataset now, given dataset has columns: "
                               + str(columns))
        inputs = tuple(inputs_col)

    file_name = os.path.realpath(file_name)
    net = _quant_export(net, *inputs, file_format=file_format, **kwargs)
    if 'enc_key' in kwargs.keys():
        kwargs['enc_key'], kwargs['enc_mode'] = _check_key_mode_type(file_format, **kwargs)
    _export(net, file_name, file_format, *inputs, **kwargs)


def _export(net, file_name, file_format, *inputs, **kwargs):
    """
    It is an internal conversion function. Export the MindSpore prediction model to a file in the specified format.
    """
    logger.info("exporting model file:%s format:%s.", file_name, file_format)

    if file_format == 'GEIR':
        logger.warning(f"For 'export', format 'GEIR' is deprecated, "
                       f"it would be removed in future release, use 'AIR' instead.")
        file_format = 'AIR'

    # When dumping ONNX file, switch network mode to infer when it is training(NOTE: ONNX only designed for prediction)
    is_dump_onnx_in_training = False
    if hasattr(net, 'training'):
        is_dump_onnx_in_training = net.training and file_format == 'ONNX'

    if is_dump_onnx_in_training:
        net.set_train(mode=False)

    if file_format == 'AIR':
        _save_air(net, file_name, *inputs, **kwargs)
    elif file_format == 'ONNX':
        _save_onnx(net, file_name, *inputs, **kwargs)
    elif file_format == 'MINDIR':
        _save_mindir(net, file_name, *inputs, **kwargs)

    if is_dump_onnx_in_training:
        net.set_train(mode=True)


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
        os.makedirs(real_path, exist_ok=True)
    if 'enc_key' in kwargs.keys() and 'enc_mode' in kwargs.keys():
        _executor.export(file_name, graph_id, enc_key=kwargs.get('enc_key'), encrypt_func=kwargs.get('enc_mode'))
    else:
        _executor.export(file_name, graph_id)
    os.chmod(file_name, stat.S_IRUSR)


def _save_onnx(net, file_name, *inputs, **kwargs):
    """Save ONNX format file."""
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
    with open(file_name, 'wb') as f:
        f.write(onnx_stream)
        os.chmod(file_name, stat.S_IRUSR)


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
    parameter_size = (offset / 1024)
    try:
        f = open(data_file_name, "wb")
        f.write(place_holder_data)
    except IOError:
        f.close()

    return f, parameter_size, offset


def _spilt_save(net_dict, model, file_name, is_encrypt, **kwargs):
    """The function to save parameter data."""
    logger.warning("Parameters in the net capacity exceeds 1G, save MindIR model and parameters separately.")
    # save parameter
    file_prefix = file_name.split("/")[-1]
    if file_prefix.endswith(".mindir"):
        file_prefix = file_prefix[:-7]
    current_path = os.path.abspath(file_name)
    dirname = os.path.dirname(current_path)
    data_path = os.path.join(dirname, file_prefix + "_variables")
    if os.path.exists(data_path):
        shutil.rmtree(data_path)
    os.makedirs(data_path, exist_ok=True)
    os.chmod(data_path, stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR)
    index = 0
    external_local = os.path.join(file_prefix + "_variables", "data_" + str(index))
    data_file_name = os.path.join(dirname, external_local)
    f, parameter_size, offset = _get_data_file(is_encrypt, kwargs, data_file_name)
    try:
        for param_proto in model.graph.parameter:
            name = param_proto.name[param_proto.name.find(":") + 1:]
            param = net_dict[name]
            raw_data = param.data.asnumpy().tobytes()
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
            if is_encrypt():
                if callable(kwargs.get('enc_mode')):
                    enc_func = kwargs.get('enc_mode')
                    write_data = enc_func(write_data, kwargs.get('enc_key'))
                else:
                    write_data = _encrypt(write_data, len(write_data), kwargs.get('enc_key'),
                                          len(kwargs.get('enc_key')), kwargs.get('enc_mode'))
            f.write(write_data)

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


def _msfunc_info(net, *inputs):
    """Get mindir stream and parameter dict of ms_function"""
    # pylint: disable=protected-access
    net_dict = OrderedDict()
    _ms_func_executor = _MindsporeFunctionExecutor(net, time.time() * 1e9)
    graph_id = _ms_func_executor.compile(args_list=inputs, method_name=net.__name__)
    mindir_stream = _executor._get_func_graph_proto(net, graph_id, 'mind_ir')
    params = _ms_func_executor._graph_executor.get_params(graph_id)
    for name, value in params.items():
        net_dict[name] = Parameter(value, name=name)
    return mindir_stream, net_dict


def _cell_info(net, *inputs):
    """Get mindir stream and net dict of cell"""
    phase_name = "predict" if _is_in_auto_parallel_mode() else "export.mindir"
    graph_id, _ = _executor.compile(net, *inputs, phase=phase_name, do_convert=False)
    # pylint: disable=protected-access
    mindir_stream = _executor._get_func_graph_proto(net, graph_id, 'mind_ir')
    # clean obfuscation config to prevent the next call
    _executor.obfuscate_config = None

    net_dict = net.parameters_dict()
    return mindir_stream, net_dict


def _set_obfuscate_config(**kwargs):
    """Set obfuscation config for executor."""
    logger.warning("Obfuscate model.")
    if 'enc_mode' in kwargs.keys():
        enc_mode = Validator.check_isinstance('enc_mode', kwargs.get('enc_mode'), str)
        if enc_mode not in ["AES-GCM", "AES-CBC", "SM4-CBC"]:
            raise ValueError(
                "Only MindIR files that encrypted with 'AES-GCM', 'AES-CBC' or 'SM4-CBC' is supported for"
                "obfuscation, but got {}.".format(enc_mode))
    obf_ratio, customized_funcs, obf_password = _check_obfuscate_params(kwargs.get('obf_config'))
    if customized_funcs and obf_password > 0:
        logger.warning("Although 'customized_func' and 'obf_password' are set, the 'obf_password' mode would be"
                       " applied, remember to set 'obf_password' when loading obfuscated model.")

    if obf_password == 0:  # apply customized_func mode
        clean_funcs()
        for func in customized_funcs:
            add_opaque_predicate(func.__name__, func)
    _executor.obfuscate_config = {'obf_ratio': obf_ratio, 'obf_password': obf_password}


def _save_mindir(net, file_name, *inputs, **kwargs):
    """Save MindIR format file."""
    # set obfuscate configs
    if 'obf_config' in kwargs.keys():
        _set_obfuscate_config(**kwargs)
        for item in inputs:
            if -1 in item.shape:
                raise ValueError(
                    "Dynamic shape input is not supported now, but got the shape of inputs: {}.".format(item.shape))

    model = mindir_model()
    if not isinstance(net, nn.Cell):
        mindir_stream, net_dict = _msfunc_info(net, *inputs)
    else:
        mindir_stream, net_dict = _cell_info(net, *inputs)
    model.ParseFromString(mindir_stream)

    if kwargs.get('dataset'):
        check_input_data(kwargs.get('dataset'), data_class=mindspore.dataset.Dataset)
        dataset = kwargs.get('dataset')
        _save_dataset_to_mindir(model, dataset)

    save_together = _save_together(net_dict, model)
    is_encrypt = lambda: 'enc_key' in kwargs.keys() and 'enc_mode' in kwargs.keys()
    if save_together:
        _save_mindir_together(net_dict, model, file_name, is_encrypt, **kwargs)
    else:
        _spilt_save(net_dict, model, file_name, is_encrypt, **kwargs)


def _save_mindir_together(net_dict, model, file_name, is_encrypt, **kwargs):
    """Save graph and parameter together."""
    for param_proto in model.graph.parameter:
        param_name = param_proto.name[param_proto.name.find(":") + 1:]
        if param_name in net_dict.keys():
            param_data = net_dict[param_name].data.asnumpy().tobytes()
            param_proto.raw_data = param_data
        else:
            logger.warning("The parameter '{}' is not belongs to any cell,the data of parameter cannot be exported."
                           .format(param_proto.name))
    for map_param_proto in model.graph.map_parameter:
        map_param_name = map_param_proto.name[map_param_proto.name.find(":") + 1:]
        if map_param_name in net_dict.keys():
            map_parameter = net_dict[map_param_name]
            key_nparr, value_nparr, status_nparr = map_parameter.export_data()
            map_param_proto.key_tensor.raw_data = key_nparr.tobytes()
            map_param_proto.value_tensor.raw_data = value_nparr.tobytes()
            map_param_proto.status_tensor.raw_data = status_nparr.tobytes()
        else:
            logger.warning("The map_parameter '{}' is not belongs to any cell,the data of parameter cannot be exported."
                           .format(map_param_proto.name))
    if not file_name.endswith('.mindir'):
        file_name += ".mindir"
    current_path = os.path.abspath(file_name)
    dirname = os.path.dirname(current_path)
    os.makedirs(dirname, exist_ok=True)
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
            data_total += sys.getsizeof(net_dict[name].data.asnumpy().tobytes()) / 1024
        else:
            logger.info("The parameter '{}' is not belongs to any cell,the data of parameter cannot be exported."
                        .format(param_proto.name))
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


def quant_mode_manage(func):
    """Inherit the quant_mode in old version."""

    @wraps(func)
    def wrapper(network, *inputs, file_format, **kwargs):
        if 'quant_mode' not in kwargs:
            return network
        quant_mode = kwargs.get('quant_mode')
        if not isinstance(quant_mode, str):
            raise TypeError("For 'export', the type of 'quant_mode' should be string, "
                            "but got {}.".format(type(quant_mode)))
        if quant_mode in ('AUTO', 'MANUAL'):
            kwargs['quant_mode'] = 'QUANT'
        return func(network, *inputs, file_format=file_format, **kwargs)

    return wrapper


@quant_mode_manage
def _quant_export(network, *inputs, file_format, **kwargs):
    """Exports MindSpore quantization predict model to deploy with AIR and MINDIR."""
    supported_device = ["Ascend", "GPU"]
    supported_formats = ['AIR', 'MINDIR']
    quant_mode_formats = ['QUANT', 'NONQUANT']

    quant_mode = kwargs['quant_mode']
    if quant_mode not in quant_mode_formats:
        raise KeyError(f"For 'export', the argument 'quant_mode' must be one of {quant_mode_formats}, "
                       f"but got {quant_mode}.")
    if quant_mode == 'NONQUANT':
        return network
    quant_net = copy.deepcopy(network)
    quant_net._create_time = int(time.time() * 1e9)

    mean = 127.5 if kwargs.get('mean', None) is None else kwargs.get('mean')
    std_dev = 127.5 if kwargs.get('std_dev', None) is None else kwargs.get('std_dev')
    mean = Validator.check_value_type("mean", mean, (int, float))
    std_dev = Validator.check_value_type("std_dev", std_dev, (int, float))

    if context.get_context('device_target') not in supported_device:
        raise KeyError(f"For 'export', quant export only support {supported_device} device target now, "
                       f"but got {context.get_context('device_target')}")

    if file_format not in supported_formats:
        raise ValueError(f"For 'export', quant export only support 'file_format' {supported_formats}, "
                         f"but got {file_format}.")

    quant_net.set_train(False)
    if file_format == "MINDIR":
        exporter = quant_export.ExportToQuantInferNetwork(quant_net, mean, std_dev, *inputs, is_mindir=True)
    else:
        exporter = quant_export.ExportToQuantInferNetwork(quant_net, mean, std_dev, *inputs)
    deploy_net = exporter.run()
    return deploy_net


def parse_print(print_file_name):
    """
    Parse data file generated by mindspore.ops.Print.

    Args:
        print_file_name (str): The file name needs to be parsed.

    Returns:
        List, element of list is Tensor.

    Raises:
        ValueError: The print file does not exist or is empty.
        RuntimeError: Failed to parse the file.

    Examples:
        >>> import numpy as np
        >>> import mindspore as ms
        >>> import mindspore.ops as ops
        >>> from mindspore import nn
        >>> from mindspore import Tensor
        >>> ms.set_context(mode=ms.GRAPH_MODE, print_file_path='log.data')
        >>> class PrintInputTensor(nn.Cell):
        ...         def __init__(self):
        ...             super().__init__()
        ...             self.print = ops.Print()
        ...
        ...         def construct(self, input_pra):
        ...             self.print('print:', input_pra)
        ...             return input_pra
        >>> x = np.array([[1, 2, 3, 4], [5, 6, 7, 8]]).astype(np.float32)
        >>> input_pra = Tensor(x)
        >>> net = PrintInputTensor()
        >>> net(input_pra)
        >>>
        >>> import mindspore
        >>> data = mindspore.parse_print('./log.data')
        >>> print(data)
        ['print:', Tensor(shape=[2, 4], dtype=Float32, value=
        [[ 1.00000000e+00,  2.00000000e+00,  3.00000000e+00,  4.00000000e+00],
        [ 5.00000000e+00,  6.00000000e+00,  7.00000000e+00,  8.00000000e+00]])]
    """
    print_file_path = os.path.realpath(print_file_name)

    if os.path.getsize(print_file_path) == 0:
        raise ValueError("For 'parse_print', the print file may be empty, please make sure enter the correct "
                         "'print_file_name'.")

    logger.info("Execute load print process.")
    print_list = Print()

    try:
        with open(print_file_path, "rb") as f:
            pb_content = f.read()
        print_list.ParseFromString(pb_content)
    except BaseException as e:
        logger.critical("Failed to read the print file %s, please check whether the file is "
                        "correct.", print_file_name)
        raise ValueError(e.__str__() + "\nFailed to read the print file {}, please check whether "
                                       "the file is correct.".format(print_file_name)) from e

    tensor_list = []

    try:
        for print_ in print_list.value:
            # String type
            if print_.HasField("desc"):
                tensor_list.append(print_.desc)
            elif print_.HasField("tensor"):
                dims = print_.tensor.dims
                data_type = print_.tensor.tensor_type
                data = print_.tensor.tensor_content
                np_type = tensor_to_np_type.get(data_type)
                param_data = np.fromstring(data, np_type)
                ms_type = tensor_to_ms_type.get(data_type)
                if dims and dims != [0]:
                    param_value = param_data.reshape(dims)
                    tensor_list.append(Tensor(param_value, ms_type))
                # Scalar type
                else:
                    data_type_ = data_type.lower()
                    if 'float' in data_type_:
                        param_data = float(param_data[0])
                    elif 'int' in data_type_:
                        param_data = int(param_data[0])
                    elif 'bool' in data_type_:
                        param_data = bool(param_data[0])
                    tensor_list.append(Tensor(param_data, ms_type))

    except BaseException as e:
        logger.critical("Failed to load the print file %s.", print_list)
        raise RuntimeError(e.__str__() + "\nFailed to load the print file {}.".format(print_list)) from e

    return tensor_list


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


def restore_group_info_list(group_info_file_name):
    """
    Build rank list, the checkpoint of ranks in the rank list has the same contents with the local rank
    who saves the group_info_file_name. To save the group info file, please export GROUP_INFO_FILE environment variables
    like "export GROUP_INFO_FILE=/data/group_info.pb".

    Args:
        group_info_file_name (str): Name of group information file.

    Returns:
        List, the rank list.

    Raises:
        ValueError: group information file is incorrect.
        TypeError: group_info_file_name is not str.

    Examples:
        >>> restore_list = restore_group_info_list("./group_info.pb")
    """
    if not isinstance(group_info_file_name, str):
        raise TypeError(f"For 'restore_group_info_list', the argument 'group_info_file_name' should be str, "
                        f"but got {type(group_info_file_name)}.")

    if not os.path.isfile(group_info_file_name):
        raise ValueError(f"For 'restore_group_info_list', no such group information file: {group_info_file_name}.")

    if os.path.getsize(group_info_file_name) == 0:
        raise ValueError("For 'restore_group_info_list', the group information file should not be empty.")

    return _restore_group_info_list(group_info_file_name)


def build_searched_strategy(strategy_filename):
    """
    Build strategy of every parameter in network. Used in the case of distributed inference.
    For details of it, please check:
    `Saving and Loading Models in Hybrid Parallel Mode
    <https://www.mindspore.cn/tutorials/experts/en/r2.0.0-alpha/parallel/save_load.html>`_.

    Args:
        strategy_filename (str): Name of strategy file.

    Returns:
        Dict, whose key is parameter name and value is slice strategy of this parameter.

    Raises:
        ValueError: Strategy file is incorrect.
        TypeError: strategy_filename is not a string.

    Examples:
        >>> strategy = build_searched_strategy("./strategy_train.ckpt")
    """
    return _build_searched_strategy(strategy_filename)


def merge_sliced_parameter(sliced_parameters, strategy=None):
    """
    Merge parameter slices into one parameter. Used in the case of distributed inference.
    For details of it, please check:
    `<https://www.mindspore.cn/tutorials/experts/en/r2.0.0-alpha/parallel/save_load.html>`_.

    Args:
        sliced_parameters (list[Parameter]): Parameter slices in order of rank id.
        strategy (Optional[dict]): Parameter slice strategy, whose key is parameter name and
            value is slice strategy of this parameter. If strategy is None, just merge
            parameter slices in 0 axis order. Default: None.

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
    sliced_data = [parameter.data.asnumpy() for parameter in sliced_parameters]

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


def load_distributed_checkpoint(network, checkpoint_filenames, predict_strategy=None,
                                train_strategy_filename=None, strict_load=False, dec_key=None, dec_mode='AES-GCM'):
    """
    Load checkpoint into net for distributed predication. Used in the case of distributed inference.
    For details of distributed inference, please check:
    `Distributed Inference <https://www.mindspore.cn/tutorials/experts/en/r2.0.0-alpha/parallel/
    distributed_inference.html>`_ .

    Args:
        network (Cell): Network for distributed predication.
        checkpoint_filenames (list[str]): The name of Checkpoint files in order of rank id.
        predict_strategy (dict): Strategy of predication process. It means that using one device to predict
                                 when setting predict_strategy as None. Default: None.
        train_strategy_filename (str): The filename of training strategy protocol buffer file.
                                       When train_strategy_filename is None, the training strategy file will be
                                       obtained from context.get_auto_parallel_context("strategy_ckpt_load_file").
                                       Therefore, the training strategy file needs to be specified
                                       in at least one of them. Default: None.
        strict_load (bool): Whether to strict load the parameter into net. If False, it will load parameter
                            into net when parameter name's suffix in checkpoint file is the same as the
                            parameter in the network. When the types are inconsistent perform type conversion
                            on the parameters of the same type, such as float32 to float16. Default: False.
        dec_key (Union[None, bytes]): Byte type key used for decryption. If the value is None, the decryption
                                      is not required. Default: None.
        dec_mode (str): This parameter is valid only when dec_key is not set to None. Specifies the decryption
                        mode, currently supports 'AES-GCM', 'AES-CBC' and 'SM4-CBC'. Default: 'AES-GCM'.

    Raises:
        TypeError: The type of inputs do not match the requirements.
        ValueError: Failed to load checkpoint into net.
    """
    network = Validator.check_isinstance("network", network, nn.Cell)
    _check_checkpoint_file(checkpoint_filenames)
    _check_predict_strategy(predict_strategy)

    dec_key = Validator.check_isinstance('dec_key', dec_key, (type(None), bytes))
    dec_mode = Validator.check_isinstance('dec_mode', dec_mode, str)

    if train_strategy_filename is None:
        train_strategy_filename = context.get_auto_parallel_context("strategy_ckpt_load_file")
    _train_strategy = build_searched_strategy(train_strategy_filename)
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
        ckpt_dict = load_checkpoint(file_name, dec_key=dec_key, dec_mode=dec_mode)
        for param_name, param in ckpt_dict.items():
            param_total_dict[param_name][file_index] = param

    param_dict = {}
    param_not_in_strategy = []
    param_not_in_ckpt = []
    for _, param in network.parameters_and_names():
        sliced_params = []
        if param.name not in rank_list.keys():
            param_not_in_strategy.append(param.name)
            continue
        if param.name not in param_total_dict:
            param_not_in_ckpt.append(param.name)
            continue

        param_rank = rank_list.get(param.name)[0]
        skip_merge_split = rank_list.get(param.name)[1]
        shard_stride = train_strategy.get(param.name)[4]
        if train_strategy.get(param.name)[5]:
            shard_size = ckpt_file_len / shard_stride / train_strategy.get(param.name)[5]
        else:
            shard_size = 0
        for rank in param_rank:
            param_total_list = list(range(0, ckpt_file_len))
            if shard_size > 0:
                shard_total_list = []
                for i in range(0, ckpt_file_len, shard_size):
                    shard_total_list.append(param_total_list[i:i + shard_size])
                param_total_list = shard_total_list[rank // shard_size]
            if shard_stride > 0:
                param_stride = []
                # merge pre parameter
                param_index = param_total_list[0:param_total_list.index(rank) + 1][::-1][::shard_stride]
                param_index.extend(param_total_list[param_total_list.index(rank):][::shard_stride])
                param_index = list(set(param_index))
                param_index.sort()
                for rank_num in param_index:
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

    load_param_into_net(network, param_dict, strict_load=strict_load)


def async_ckpt_thread_status():
    """
    Get the status of asynchronous save checkpoint thread.

    When performing asynchronous save checkpoint, you can determine whether the asynchronous thread is completed.

    Returns:
        bool, True, Asynchronous save checkpoint thread is running.
        False, Asynchronous save checkpoint thread is not executing.
    """
    thr_list = threading.enumerate()
    return True in [ele.getName() == "asyn_save_ckpt" for ele in thr_list]


def _check_predict_strategy(predict_strategy):
    """Check predict strategy."""

    def _check_int_list(arg):
        if not isinstance(arg, list):
            return False
        for item in arg:
            if not isinstance(item, int):
                return False
        return True

    if predict_strategy is None:
        return

    flag = True
    predict_strategy = Validator.check_isinstance("predict_strategy", predict_strategy, dict)
    for key in predict_strategy.keys():
        if not isinstance(key, str) or not isinstance(predict_strategy[key], (list, tuple)) \
                or len(predict_strategy[key]) < 4:
            flag = False
        dev_matrix, tensor_map, param_split_shape, field_size = predict_strategy[key][:4]
        if not _check_int_list(dev_matrix) or not _check_int_list(tensor_map) or \
                not (_check_int_list(param_split_shape) or not param_split_shape) or \
                not (isinstance(field_size, int) and field_size == 0):
            flag = False

    if not flag:
        raise ValueError(f"For 'load_distributed_checkpoint', the argument 'predict_strategy' is dict, "
                         f"the key of it must be string, and the value of it must be list or tuple that "
                         f"the first four elements must be dev_matrix (list[int]), tensor_map (list[int]), "
                         f"param_split_shape (list[int]) and field_size (int, which value is 0)."
                         f"Please check whether 'predict_strategy' is correct.")


def _check_checkpoint_file(checkpoint_filenames):
    """Check checkpoint file name."""
    for index, filename in enumerate(checkpoint_filenames):
        if not isinstance(filename, str) or not os.path.exists(filename) \
                or filename[-5:] != ".ckpt" or os.path.getsize(filename) == 0:
            raise ValueError(f"For 'load_distributed_checkpoint', please check 'checkpoint_filenames', and "
                             f"make sure the {filename} at index {index} is a valid checkpoint file, it must "
                             f"be a string ending with '.ckpt', and the checkpoint file it represents must "
                             f"be exist and not empty.")


def _merge_and_split(sliced_params, train_strategy, predict_strategy):
    """Merge sliced parameter and split it according to the predict strategy."""
    merged_param = merge_sliced_parameter(sliced_params, train_strategy)
    if predict_strategy is None:
        return merged_param
    param_name = merged_param.name
    tensor_layout = predict_strategy[param_name]
    split_tensor = _load_tensor(merged_param.data, tensor_layout[0], tensor_layout[1])
    requires_grad = merged_param.requires_grad
    layerwise_parallel = merged_param.layerwise_parallel
    split_param = Parameter(split_tensor, param_name, requires_grad, layerwise_parallel)
    return split_param


def _calculation_net_size(net):
    """Calculate the size of parameters in the network."""
    data_total = 0
    net_dict = net.parameters_dict()
    for name in net_dict:
        data_total += sys.getsizeof(net_dict[name].data.asnumpy().tobytes()) / 1024

    return data_total


def _get_mindir_inputs(file_name):
    """
    Get MindIR file's inputs.

    Note:
        1. Parsing encrypted MindIR file is not supported.
        2. Parsing dynamic shape MindIR file is not supported.

    Args:
        file_name (str): MindIR file name.

    Returns:
        Tensor, list(Tensor), the input of MindIR file.

    Raises:
        TypeError: If the parameter file_name is not `str`.
        RuntimeError: MindIR's input is not tensor type or has no dims.

    Examples:
        >>> input_tensor = get_mindir_inputs("lenet.mindir")
    """
    Validator.check_file_name_by_regular(file_name)
    file_name = os.path.realpath(file_name)
    model = read_proto(file_name)
    input_tensor = []

    for ele_input in model.graph.input:
        input_shape = []
        if not hasattr(ele_input, "tensor") or not hasattr(ele_input.tensor[0], "dims"):
            raise RuntimeError("MindIR's inputs has no tensor or tensor has no dims, please check MindIR file.")

        for ele_shape in ele_input.tensor[0].dims:
            input_shape.append(ele_shape)
        if is_shape_unknown(input_shape):
            raise RuntimeError(f"MindIR input's shape is: {input_shape}, dynamic shape is not supported.")

        mindir_type = ele_input.tensor[0].data_type
        if mindir_type not in mindir_to_tensor_type:
            raise RuntimeError(f"MindIR input's type: {mindir_type} is not supported.")

        input_type = mindir_to_tensor_type.get(mindir_type)
        input_tensor.append(Tensor(shape=input_shape, dtype=input_type, init=One()))

    if not input_tensor:
        logger.warning("The MindIR model has no input, return None.")
        return None
    return input_tensor[0] if len(input_tensor) == 1 else input_tensor


def convert_model(mindir_file, convert_file, file_format):
    """
    Convert mindir model to other format model. Current version only support convert to "ONNX" format.

    Note:
        This is an experimental function that is subject to change or deletion.

    Args:
        mindir_file (str): MindIR file name.
        convert_file (str): Convert model file name.
        file_format (str): Convert model's format, current version only supports "ONNX".

    Raises:
        TypeError: If the parameter `mindir_file` is not `str`.
        TypeError: If the parameter `convert_file` is not `str`.
        ValueError: If the parameter `file_format` is not "ONNX".

    Examples:
        >>> convert_model("lenet.mindir", "lenet.onnx", "ONNX")
    """
    Validator.check_file_name_by_regular(mindir_file)
    Validator.check_file_name_by_regular(convert_file)
    if file_format != "ONNX":
        raise ValueError(f"For 'convert_model', 'file_format' must be 'ONNX', but got {file_format}.")
    net_input = _get_mindir_inputs(mindir_file)
    graph = load(mindir_file)
    net = nn.GraphCell(graph)
    if isinstance(net_input, Tensor):
        export(net, net_input, file_name=convert_file, file_format=file_format)
    else:
        export(net, *net_input, file_name=convert_file, file_format=file_format)
