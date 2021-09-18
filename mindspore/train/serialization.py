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
import os
import sys
import stat
import math
import shutil
import time
import copy
import json
import threading
from threading import Thread, Lock
from collections import defaultdict

import numpy as np

import mindspore
import mindspore.nn as nn
from mindspore import context
from mindspore import log as logger
from mindspore.train.checkpoint_pb2 import Checkpoint
from mindspore.train.print_pb2 import Print
from mindspore.train.node_strategy_pb2 import ParallelStrategyMap, ParallelLayouts
from mindspore.train.mind_ir_pb2 import ModelProto as mindir_model
from mindspore.train.mind_ir_pb2 import GraphProto as graph_proto
from mindspore.common.tensor import Tensor
from mindspore.common.initializer import initializer
from mindspore.common.parameter import Parameter
from mindspore.common.api import _cell_graph_executor as _executor
from mindspore.common import dtype as mstype
from mindspore._checkparam import check_input_data, Validator
from mindspore.compression.export import quant_export
from mindspore.parallel._tensor import _load_tensor, _get_tensor_strategy, _get_tensor_slice_index
from mindspore.parallel._utils import _infer_rank_list, _remove_repeated_slices
from mindspore.communication.management import get_rank, get_group_size
from .._c_expression import load_mindir, _encrypt, _decrypt, _is_cipher_file


tensor_to_ms_type = {"Int8": mstype.int8, "Uint8": mstype.uint8, "Int16": mstype.int16, "Uint16": mstype.uint16,
                     "Int32": mstype.int32, "Uint32": mstype.uint32, "Int64": mstype.int64, "Uint64": mstype.uint64,
                     "Float16": mstype.float16, "Float32": mstype.float32, "Float64": mstype.float64,
                     "Bool": mstype.bool_}

tensor_to_np_type = {"Int8": np.int8, "Uint8": np.uint8, "Int16": np.int16, "Uint16": np.uint16,
                     "Int32": np.int32, "Uint32": np.uint32, "Int64": np.int64, "Uint64": np.uint64,
                     "Float16": np.float16, "Float32": np.float32, "Float64": np.float64, "Bool": np.bool_}

_ckpt_mutex = Lock()

# unit is KB
SLICE_SIZE = 512 * 1024
PROTO_LIMIT_SIZE = 1024 * 1024 * 2
TOTAL_SAVE = 1024 * 1024


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
                logger.error("Failed to combine the net and the parameters for param %s.", param.name)
                msg = ("Net parameters {} shape({}) different from parameter_dict's({})"
                       .format(param.name, param.data.shape, new_param.data.shape))
                raise RuntimeError(msg)

        if param.data.dtype != new_param.data.dtype:
            if _type_convert(param, new_param, strict_load):
                new_tensor = Tensor(new_param.data.asnumpy(), param.data.dtype)
                param.set_data(new_tensor)
                return

            logger.error("Failed to combine the net and the parameters for param %s.", param.name)
            msg = ("Net parameters {} type({}) different from parameter_dict's({})"
                   .format(param.name, param.data.dtype, new_param.data.dtype))
            raise RuntimeError(msg)

        param.set_data(new_param.data, param.sliced)
        return

    if isinstance(param.data, Tensor) and not isinstance(new_param.data, Tensor):
        if param.data.shape != (1,) and param.data.shape != ():
            logger.error("Failed to combine the net and the parameters for param %s.", param.name)
            msg = ("Net parameters {} shape({}) is not (1,), inconsistent with parameter_dict's(scalar)."
                   .format(param.name, param.data.shape))
            raise RuntimeError(msg)
        param.set_data(initializer(new_param.data, param.data.shape, param.data.dtype))

    elif isinstance(new_param.data, Tensor) and not isinstance(param.data, Tensor):
        logger.error("Failed to combine the net and the parameters for param %s.", param.name)
        msg = ("Net parameters {} type({}) different from parameter_dict's({})"
               .format(param.name, type(param.data), type(new_param.data)))
        raise RuntimeError(msg)

    else:
        param.set_data(type(param.data)(new_param.data))


def _type_convert(param, new_param, strict_load):
    """Whether to convert parameter's type during load checkpoint into network."""
    float_type = (mstype.float16, mstype.float32, mstype.float64)
    int_type = (mstype.int8, mstype.int16, mstype.int32, mstype.int64)
    if not strict_load and ({param.data.dtype, new_param.data.dtype}.issubset(float_type) or
                            {param.data.dtype, new_param.data.dtype}.issubset(int_type)):
        logger.warning("ckpt_dict parameter: {}'s type is {}, convert to {} in the network."
                       .format(new_param.name, new_param.data.dtype, param.data.dtype))
        return True
    return False


def _exec_save(ckpt_file_name, data_list, enc_key=None, enc_mode="AES-GCM"):
    """Execute the process of saving checkpoint into file."""
    try:
        with _ckpt_mutex:
            if os.path.exists(ckpt_file_name):
                os.remove(ckpt_file_name)
            with open(ckpt_file_name, "ab") as f:
                if enc_key is not None:
                    plain_data = bytes(0)
                    cipher_data = bytes(0)

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
                            plain_data += checkpoint_list.SerializeToString()

                            max_block_size = SLICE_SIZE*1024
                            while len(plain_data) >= max_block_size:
                                cipher_data += _encrypt(plain_data[0: max_block_size], max_block_size, enc_key,
                                                        len(enc_key), enc_mode)
                                plain_data = plain_data[max_block_size:]

                if enc_key is not None:
                    if plain_data:
                        cipher_data += _encrypt(plain_data, len(plain_data), enc_key, len(enc_key), enc_mode)
                    f.write(cipher_data)

        os.chmod(ckpt_file_name, stat.S_IRUSR)

    except BaseException as e:
        logger.error("Failed to save the checkpoint file %s.", ckpt_file_name)
        raise e


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
        async_save (bool): Whether to open a independent thread to save the checkpoint file. Default: False
        append_dict (dict): Additional information that needs to be saved.  The key of dict must be str,
            the value of dict must be one of int float and bool. Default: None
        enc_key (Union[None, bytes]): Byte type key used for encryption. If the value is None, the encryption
                                      is not required. Default: None.
        enc_mode (str): This parameter is valid only when enc_key is not set to None. Specifies the encryption
                        mode, currently supports 'AES-GCM' and 'AES-CBC'. Default: 'AES-GCM'.

    Raises:
        TypeError: If the parameter save_obj is not `nn.Cell` or list type. And if the parameter
                   `integrated_save` and `async_save` are not bool type.

    Examples:
        >>> from mindspore import save_checkpoint
        >>>
        >>> net = Net()
        >>> save_checkpoint(net, "lenet.ckpt")
    """

    if not isinstance(save_obj, nn.Cell) and not isinstance(save_obj, list):
        raise TypeError("The parameter save_obj should be nn.Cell or list, but got {}".format(type(save_obj)))
    integrated_save = Validator.check_bool(integrated_save)
    async_save = Validator.check_bool(async_save)
    append_dict = _check_append_dict(append_dict)
    enc_key = Validator.check_isinstance('enc_key', enc_key, (type(None), bytes))
    enc_mode = Validator.check_isinstance('enc_mode', enc_mode, str)

    logger.info("Execute the process of saving checkpoint files.")

    if isinstance(save_obj, nn.Cell):
        save_obj.init_parameters_data()
        param_dict = {}
        for _, param in save_obj.parameters_and_names():
            param_dict[param.name] = param
        param_list = []
        for (key, value) in param_dict.items():
            each_param = {"name": key}
            param_data = Tensor(value.data)

            # in automatic model parallel scenario, some parameters were split to all the devices,
            # which should be combined before saving
            if key in save_obj.parameter_layout_dict:
                param_data = _get_merged_param_data(save_obj, key, param_data, integrated_save)

            each_param["data"] = param_data
            param_list.append(each_param)
        save_obj = param_list

    if append_dict:
        append_info_list = []
        for k_name, value in append_dict.items():
            append_info_list.append({"name": k_name, "data": Tensor(value)})
            save_obj.extend(append_info_list)

    data_list = {}
    with _ckpt_mutex:
        for param in save_obj:
            key = param["name"]
            data_list[key] = []
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

    ckpt_file_name = os.path.realpath(ckpt_file_name)
    if async_save:
        thr = Thread(target=_exec_save, args=(ckpt_file_name, data_list, enc_key, enc_mode), name="asyn_save_ckpt")
        thr.start()
    else:
        _exec_save(ckpt_file_name, data_list, enc_key, enc_mode)

    logger.info("Saving checkpoint process is finished.")


def _check_param_prefix(filter_prefix, param_name):
    """Checks whether the prefix of parameter name matches the given filter_prefix."""
    for prefix in filter_prefix:
        if param_name.find(prefix) == 0 \
                and (param_name == prefix or param_name[len(prefix)] == "." or (prefix and prefix[-1] == ".")):
            return True
    return False


def _check_append_dict(append_dict):
    if append_dict is None:
        return append_dict
    if not isinstance(append_dict, dict):
        raise TypeError(f"The type of append_dict must dict, but got {str(type(append_dict))}.")
    if not all(isinstance(ele, str) for ele in append_dict.keys()) or \
            not all(isinstance(ele, (int, float, bool)) for ele in append_dict.values()):
        raise TypeError(f"The type of element in append_dict must be key: str, value: int or float.")
    return append_dict


def load(file_name, **kwargs):
    """
    Load MindIR.

    The returned object can be executed by a `GraphCell`, see class :class:`mindspore.nn.GraphCell` for more details.

    Args:
        file_name (str): MindIR file name.

        kwargs (dict): Configuration options dictionary.

            - dec_key (bytes): Byte type key used for decryption. Tha valid length is 16, 24, or 32.
            - dec_mode (str): Specifies the decryption mode, take effect when dec_key is set.
              Option: 'AES-GCM' | 'AES-CBC'. Default: 'AES-GCM'.
    Returns:
        Object, a compiled graph that can executed by `GraphCell`.

    Raises:
        ValueError: MindIR file name is incorrect.
        RuntimeError: Failed to parse MindIR file.

    Examples:
        >>> import numpy as np
        >>> import mindspore.nn as nn
        >>> from mindspore import Tensor, export, load
        >>>
        >>> net = nn.Conv2d(1, 1, kernel_size=3, weight_init="ones")
        >>> input_tensor = Tensor(np.ones([1, 1, 3, 3]).astype(np.float32))
        >>> export(net, input_tensor, file_name="net", file_format="MINDIR")
        >>> graph = load("net.mindir")
        >>> net = nn.GraphCell(graph)
        >>> output = net(input_tensor)
        >>> print(output)
        [[[[4. 6. 4.]
           [6. 9. 6.]
           [4. 6. 4.]]]]
    """
    if not isinstance(file_name, str):
        raise ValueError("The file name must be string.")
    if not file_name.endswith(".mindir"):
        raise ValueError("The MindIR should end with mindir, please input the correct file name.")
    if not os.path.exists(file_name):
        raise ValueError("The file does not exist.")
    file_name = os.path.realpath(file_name)

    logger.info("Execute the process of loading mindir.")
    if 'dec_key' in kwargs.keys():
        dec_key = Validator.check_isinstance('dec_key', kwargs['dec_key'], bytes)
        dec_mode = 'AES-GCM'
        if 'dec_mode' in kwargs.keys():
            dec_mode = Validator.check_isinstance('dec_mode', kwargs['dec_mode'], str)
        graph = load_mindir(file_name, dec_key=dec_key, key_len=len(dec_key), dec_mode=dec_mode)
    else:
        graph = load_mindir(file_name)

    if graph is None:
        if _is_cipher_file(file_name):
            raise RuntimeError("Load MindIR failed. The file may be encrypted, please pass in the "
                               "correct dec_key and dec_mode.")
        raise RuntimeError("Load MindIR failed.")
    return graph


def load_checkpoint(ckpt_file_name, net=None, strict_load=False, filter_prefix=None, dec_key=None, dec_mode="AES-GCM"):
    """
    Load checkpoint info from a specified file.

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
                        mode, currently supports 'AES-GCM' and 'AES-CBC'. Default: 'AES-GCM'.

    Returns:
        Dict, key is parameter name, value is a Parameter.

    Raises:
        ValueError: Checkpoint file is incorrect.

    Examples:
        >>> from mindspore import load_checkpoint
        >>>
        >>> ckpt_file_name = "./checkpoint/LeNet5-1_32.ckpt"
        >>> param_dict = load_checkpoint(ckpt_file_name, filter_prefix="conv1")
        >>> print(param_dict["conv2.weight"])
        Parameter (name=conv2.weight, shape=(16, 6, 5, 5), dtype=Float32, requires_grad=True)
    """
    ckpt_file_name, filter_prefix = _check_checkpoint_param(ckpt_file_name, filter_prefix)
    dec_key = Validator.check_isinstance('dec_key', dec_key, (type(None), bytes))
    dec_mode = Validator.check_isinstance('dec_mode', dec_mode, str)
    logger.info("Execute the process of loading checkpoint files.")
    checkpoint_list = Checkpoint()

    try:
        if dec_key is None:
            with open(ckpt_file_name, "rb") as f:
                pb_content = f.read()
        else:
            pb_content = _decrypt(ckpt_file_name, dec_key, len(dec_key), dec_mode)
            if pb_content is None:
                raise ValueError
        checkpoint_list.ParseFromString(pb_content)
    except BaseException as e:
        if _is_cipher_file(ckpt_file_name):
            logger.error("Failed to read the checkpoint file `%s`. The file may be encrypted, please pass in the "
                         "correct dec_key.", ckpt_file_name)
        else:
            logger.error("Failed to read the checkpoint file `%s`, please check the correct of the file.", \
                         ckpt_file_name)
        raise ValueError(e.__str__())

    parameter_dict = {}
    try:
        param_data_list = []
        for element_id, element in enumerate(checkpoint_list.value):
            if filter_prefix is not None and _check_param_prefix(filter_prefix, element.tag):
                continue
            data = element.tensor.tensor_content
            data_type = element.tensor.tensor_type
            np_type = tensor_to_np_type[data_type]
            ms_type = tensor_to_ms_type[data_type]
            element_data = np.frombuffer(data, np_type)
            param_data_list.append(element_data)
            if (element_id == len(checkpoint_list.value) - 1) or \
                    (element.tag != checkpoint_list.value[element_id + 1].tag):
                param_data = np.concatenate((param_data_list), axis=0)
                param_data_list.clear()
                dims = element.tensor.dims
                if dims == [0]:
                    if 'Float' in data_type:
                        param_data = float(param_data[0])
                    elif 'Int' in data_type:
                        param_data = int(param_data[0])
                    parameter_dict[element.tag] = Parameter(Tensor(param_data, ms_type), name=element.tag)
                elif dims == [1]:
                    parameter_dict[element.tag] = Parameter(Tensor(param_data, ms_type), name=element.tag)
                else:
                    param_dim = []
                    for dim in dims:
                        param_dim.append(dim)
                    param_value = param_data.reshape(param_dim)
                    parameter_dict[element.tag] = Parameter(Tensor(param_value, ms_type), name=element.tag)

        logger.info("Loading checkpoint files process is finished.")

    except BaseException as e:
        logger.error("Failed to load the checkpoint file `%s`.", ckpt_file_name)
        raise RuntimeError(e.__str__())

    if not parameter_dict:
        raise ValueError(f"The loaded parameter dict is empty after filtering, please check filter_prefix.")

    if net is not None:
        load_param_into_net(net, parameter_dict, strict_load)

    return parameter_dict


def _check_checkpoint_param(ckpt_file_name, filter_prefix=None):
    """Check function load_checkpoint's parameter."""
    if not isinstance(ckpt_file_name, str):
        raise ValueError("The ckpt_file_name must be string.")

    if not os.path.exists(ckpt_file_name):
        raise ValueError("The checkpoint file does not exist.")

    if ckpt_file_name[-5:] != ".ckpt":
        raise ValueError("Please input the correct checkpoint file name.")
    ckpt_file_name = os.path.realpath(ckpt_file_name)

    if filter_prefix is not None:
        if not isinstance(filter_prefix, (str, list, tuple)):
            raise TypeError(f"The type of filter_prefix must be str, list[str] or tuple[str] "
                            f"when filter_prefix is not None, but got {str(type(filter_prefix))}.")
        if isinstance(filter_prefix, str):
            filter_prefix = (filter_prefix,)
        if not filter_prefix:
            raise ValueError("The filter_prefix can't be empty when filter_prefix is list or tuple.")
        for index, prefix in enumerate(filter_prefix):
            if not isinstance(prefix, str):
                raise TypeError(f"The type of filter_prefix must be str, list[str] or tuple[str], "
                                f"but got {str(type(prefix))} at index {index}.")
    return ckpt_file_name, filter_prefix


def load_param_into_net(net, parameter_dict, strict_load=False):
    """
    Load parameters into network.

    Args:
        net (Cell): The network where the parameters will be loaded.
        parameter_dict (dict): The dictionary generated by load checkpoint file.
        strict_load (bool): Whether to strict load the parameter into net. If False, it will load parameter
                            into net when parameter name's suffix in checkpoint file is the same as the
                            parameter in the network. When the types are inconsistent perform type conversion
                            on the parameters of the same type, such as float32 to float16. Default: False.

    Returns:
        List, parameter name not loaded into the network

    Raises:
        TypeError: Argument is not a Cell, or parameter_dict is not a Parameter dictionary.

    Examples:
        >>> from mindspore import load_checkpoint, load_param_into_net
        >>>
        >>> net = Net()
        >>> ckpt_file_name = "./checkpoint/LeNet5-1_32.ckpt"
        >>> param_dict = load_checkpoint(ckpt_file_name, filter_prefix="conv1")
        >>> param_not_load = load_param_into_net(net, param_dict)
        >>> print(param_not_load)
        ['conv1.weight']
    """
    if not isinstance(net, nn.Cell):
        logger.error("Failed to combine the net and the parameters.")
        msg = ("Argument net should be a Cell, but got {}.".format(type(net)))
        raise TypeError(msg)

    if not isinstance(parameter_dict, dict):
        logger.error("Failed to combine the net and the parameters.")
        msg = ("Argument parameter_dict should be a dict, but got {}.".format(type(parameter_dict)))
        raise TypeError(msg)

    strict_load = Validator.check_bool(strict_load)
    logger.info("Execute the process of loading parameters into net.")
    net.init_parameters_data()
    param_not_load = []
    for _, param in net.parameters_and_names():
        if param.name in parameter_dict:
            new_param = parameter_dict[param.name]
            if not isinstance(new_param, Parameter):
                logger.error("Failed to combine the net and the parameters.")
                msg = ("Argument parameter_dict element should be a Parameter, but got {}.".format(type(new_param)))
                raise TypeError(msg)
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
        logger.warning("{} parameters in the net are not loaded.".format(len(param_not_load)))
        for param_name in param_not_load:
            logger.warning("{} is not loaded.".format(param_name))
    return param_not_load


def _load_dismatch_prefix_params(net, parameter_dict, param_not_load, strict_load):
    """When some net parameter did not load, try to continue load."""
    prefix_name = ""
    longest_name = param_not_load[0]
    while prefix_name != longest_name and param_not_load:
        logger.debug("Count: {} parameters has not been loaded, try to load continue.".format(len(param_not_load)))
        prefix_name = longest_name
        for net_param_name in param_not_load:
            for dict_name in parameter_dict:
                if dict_name.endswith(net_param_name):
                    prefix_name = dict_name[:-len(net_param_name)]
                    break
            if prefix_name != longest_name:
                break

        if prefix_name != longest_name:
            logger.warning("Remove parameter prefix name: {}, continue to load.".format(prefix_name))
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


def _get_merged_param_data(net, param_name, param_data, integrated_save):
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
    from mindspore.parallel._cell_wrapper import get_allgather_cell
    from mindspore.parallel._tensor import _reshape_param_data
    layout = net.parameter_layout_dict[param_name]
    if len(layout) < 6:
        logger.info("layout dict does not contain the key %s", param_name)
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
        logger.info("need to create allgather net for %s", param_name)
        if integrated_save:
            if uniform_split == 0:
                raise RuntimeError("Integrated save checkpoint only support uniform split tensor now.")
            # while any dim is not equal to -1, means param is split and needs to be merged
            # pipeline parallel need to be supported here later
            if mp_weight:
                allgather_net = get_allgather_cell(opt_shard_group, bool(opt_shard_group))
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


def export(net, *inputs, file_name, file_format='AIR', **kwargs):
    """
    Export the MindSpore prediction model to a file in the specified format.

    Note:
        1. When exporting AIR, ONNX format, the size of a single tensor can not exceed 2GB.
        2. When file_name does not have a suffix, the system will automatically add one according to the file_format.

    Args:
        net (Cell): MindSpore network.
        inputs (Tensor): Inputs of the `net`, if the network has multiple inputs, incoming tuple(Tensor).
        file_name (str): File name of the model to be exported.
        file_format (str): MindSpore currently supports 'AIR', 'ONNX' and 'MINDIR' format for exported model.

            - AIR: Ascend Intermediate Representation. An intermediate representation format of Ascend model.
            - ONNX: Open Neural Network eXchange. An open format built to represent machine learning models.
            - MINDIR: MindSpore Native Intermediate Representation for Anf. An intermediate representation format
              for MindSpore models.

        kwargs (dict): Configuration options dictionary.

            - quant_mode (str): If the network is quantization aware training network, the quant_mode should
              be set to "QUANT", else the quant_mode should be set to "NONQUANT".
            - mean (float): The mean of input data after preprocessing, used for quantizing the first layer of network.
              Default: 127.5.
            - std_dev (float): The variance of input data after preprocessing,
              used for quantizing the first layer of network. Default: 127.5.
            - enc_key (byte): Byte type key used for encryption. Tha valid length is 16, 24, or 32.
            - enc_mode (str): Specifies the encryption mode, take effect when enc_key is set.
              Option: 'AES-GCM' | 'AES-CBC'. Default: 'AES-GCM'.
            - dataset (Dataset): Specifies the preprocess methods of network.

    Examples:
        >>> import numpy as np
        >>> from mindspore import export, Tensor
        >>>
        >>> net = LeNet()
        >>> input_tensor = Tensor(np.ones([1, 1, 32, 32]).astype(np.float32))
        >>> export(net, Tensor(input_tensor), file_name='lenet', file_format='MINDIR')
    """
    logger.info("exporting model file:%s format:%s.", file_name, file_format)
    check_input_data(*inputs, data_class=Tensor)
    Validator.check_file_name_by_regular(file_name)
    file_name = os.path.realpath(file_name)
    net = _quant_export(net, *inputs, file_format=file_format, **kwargs)
    if 'enc_key' in kwargs.keys():
        if file_format != 'MINDIR':
            raise ValueError(f"enc_key can be passed in only when file_format=='MINDIR', but got {file_format}")

        enc_key = Validator.check_isinstance('enc_key', kwargs['enc_key'], bytes)
        enc_mode = 'AES-GCM'
        if 'enc_mode' in kwargs.keys():
            enc_mode = Validator.check_isinstance('enc_mode', kwargs['enc_mode'], str)
        dataset = kwargs['dataset'] if 'dataset' in kwargs.keys() else None
        _export(net, file_name, file_format, *inputs, enc_key=enc_key, enc_mode=enc_mode, dataset=dataset)
    else:
        _export(net, file_name, file_format, *inputs, **kwargs)


def _export(net, file_name, file_format, *inputs, **kwargs):
    """
    It is an internal conversion function. Export the MindSpore prediction model to a file in the specified format.
    """
    logger.info("exporting model file:%s format:%s.", file_name, file_format)
    check_input_data(*inputs, data_class=Tensor)
    if 'dataset' in kwargs.keys() and kwargs['dataset'] is not None:
        check_input_data(kwargs['dataset'], data_class=mindspore.dataset.Dataset)

    if file_format == 'GEIR':
        logger.warning(f"Format 'GEIR' is deprecated, it would be removed in future release, use 'AIR' instead.")
        file_format = 'AIR'

    supported_formats = ['AIR', 'ONNX', 'MINDIR']
    if file_format not in supported_formats:
        raise ValueError(f'Illegal file format {file_format}, it must be one of {supported_formats}')
    # When dumping ONNX file, switch network mode to infer when it is training(NOTE: ONNX only designed for prediction)
    is_dump_onnx_in_training = net.training and file_format == 'ONNX'
    if is_dump_onnx_in_training:
        net.set_train(mode=False)

    if file_format == 'AIR':
        phase_name = 'export.air'
        graph_id, _ = _executor.compile(net, *inputs, phase=phase_name)
        if not file_name.endswith('.air'):
            file_name += ".air"
        if os.path.exists(file_name):
            os.chmod(file_name, stat.S_IWUSR)
        if "/" in file_name:
            real_path = os.path.realpath(file_name[:file_name.rfind("/")])
            os.makedirs(real_path, exist_ok=True)
        _executor.export(file_name, graph_id)
        os.chmod(file_name, stat.S_IRUSR)
    elif file_format == 'ONNX':
        total_size = _calculation_net_size(net)
        if total_size > PROTO_LIMIT_SIZE:
            raise RuntimeError('Export onnx model failed. Network size is: {}G, it exceeded the protobuf: {}G limit.'
                               .format(total_size/1024/1024, PROTO_LIMIT_SIZE/1024/1024))
        phase_name = 'export.onnx'
        graph_id, _ = _executor.compile(net, *inputs, phase=phase_name, do_convert=False)
        onnx_stream = _executor._get_func_graph_proto(net, graph_id)
        if not file_name.endswith('.onnx'):
            file_name += ".onnx"
        if os.path.exists(file_name):
            os.chmod(file_name, stat.S_IWUSR)
        with open(file_name, 'wb') as f:
            f.write(onnx_stream)
            os.chmod(file_name, stat.S_IRUSR)
    elif file_format == 'MINDIR':
        _save_mindir(net, file_name, *inputs, **kwargs)

    if is_dump_onnx_in_training:
        net.set_train(mode=True)


def _save_mindir(net, file_name, *inputs, **kwargs):
    """Save MindIR format file."""
    model = mindir_model()

    phase_name = "predict" if net._auto_parallel_mode else "export.mindir"

    graph_id, _ = _executor.compile(net, *inputs, phase=phase_name,
                                    do_convert=False, auto_parallel_mode=net._auto_parallel_mode)
    mindir_stream = _executor._get_func_graph_proto(net, graph_id, 'mind_ir')

    net_dict = net.parameters_dict()
    model.ParseFromString(mindir_stream)

    if 'dataset' in kwargs.keys() and kwargs['dataset'] is not None:
        dataset = kwargs['dataset']
        model.preprocessor = json.dumps(dataset.to_json(), indent=2)

    save_together = _save_together(net_dict, model)
    is_encrypt = lambda: 'enc_key' in kwargs.keys() and 'enc_mode' in kwargs.keys()
    if save_together:
        _save_mindir_together(net_dict, model, file_name, is_encrypt, **kwargs)
    else:
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
        graphproto = graph_proto()
        data_size = 0

        for name, param in net_dict.items():
            for param_proto in model.graph.parameter:
                if name == param_proto.name[param_proto.name.find(":") + 1:]:
                    parameter = graphproto.parameter.add()
                    parameter.name = param_proto.name
                    parameter.data_type = param_proto.data_type
                    for dim in param_proto.dims:
                        parameter.dims.append(dim)
                    byte_data = param.data.asnumpy().tobytes()
                    parameter.raw_data = byte_data
                    data_size += sys.getsizeof(byte_data) / 1024
                    break
            if data_size > TOTAL_SAVE:
                data_file_name = os.path.join(data_path, "data_" + str(index))
                if os.path.exists(data_file_name):
                    os.chmod(data_file_name, stat.S_IWUSR)
                with open(data_file_name, "ab") as f:
                    os.chmod(data_file_name, stat.S_IRUSR | stat.S_IWUSR)
                    graph_string = graphproto.SerializeToString()
                    if is_encrypt():
                        graph_string = _encrypt(graph_string, len(graph_string), kwargs['enc_key'],
                                                len(kwargs['enc_key']), kwargs['enc_mode'])
                    f.write(graph_string)
                    os.chmod(data_file_name, stat.S_IRUSR)
                index += 1
                data_size = 0
                del graphproto.parameter[:]

        if graphproto.parameter:
            data_file_name = os.path.join(data_path, "data_" + str(index))
            if os.path.exists(data_file_name):
                os.chmod(data_file_name, stat.S_IWUSR)
            with open(data_file_name, "ab") as f:
                os.chmod(data_file_name, stat.S_IRUSR | stat.S_IWUSR)
                graph_string = graphproto.SerializeToString()
                if is_encrypt():
                    graph_string = _encrypt(graph_string, len(graph_string), kwargs['enc_key'], len(kwargs['enc_key']),
                                            kwargs['enc_mode'])
                f.write(graph_string)
                os.chmod(data_file_name, stat.S_IRUSR)

        # save graph
        del model.graph.parameter[:]
        graph_file_name = os.path.join(dirname, file_prefix + "_graph.mindir")
        if os.path.exists(graph_file_name):
            os.chmod(graph_file_name, stat.S_IWUSR)
        with open(graph_file_name, 'wb') as f:
            os.chmod(graph_file_name, stat.S_IRUSR | stat.S_IWUSR)
            model_string = model.SerializeToString()
            if is_encrypt():
                model_string = _encrypt(model_string, len(model_string), kwargs['enc_key'], len(kwargs['enc_key']),
                                        kwargs['enc_mode'])
            f.write(model_string)
            os.chmod(graph_file_name, stat.S_IRUSR)


def _save_mindir_together(net_dict, model, file_name, is_encrypt, **kwargs):
    """Save graph and parameter together."""
    for param_proto in model.graph.parameter:
        param_name = param_proto.name[param_proto.name.find(":") + 1:]
        if param_name in net_dict.keys():
            param_data = net_dict[param_name].data.asnumpy().tobytes()
            param_proto.raw_data = param_data
        else:
            logger.error("The parameter %s in the graph are not in the network.", param_name)
            raise ValueError("The parameter in the graph must in the network.")
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
            model_string = _encrypt(model_string, len(model_string), kwargs['enc_key'], len(kwargs['enc_key']),
                                    kwargs['enc_mode'])
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
            raise RuntimeError('Graph parameter: {} Undefined in network.'.format(param_proto.name))
        if data_total > TOTAL_SAVE:
            return False
    return True


def quant_mode_manage(func):
    """
    Inherit the quant_mode in old version.
    """
    def warpper(network, *inputs, file_format, **kwargs):
        if 'quant_mode' not in kwargs:
            return network
        quant_mode = kwargs['quant_mode']
        if not isinstance(quant_mode, str):
            raise TypeError("The type of quant_mode should be str, but got {}.".format(type(quant_mode)))
        if quant_mode in ('AUTO', 'MANUAL'):
            kwargs['quant_mode'] = 'QUANT'
        return func(network, *inputs, file_format=file_format, **kwargs)
    return warpper


@quant_mode_manage
def _quant_export(network, *inputs, file_format, **kwargs):
    """
    Exports MindSpore quantization predict model to deploy with AIR and MINDIR.
    """
    supported_device = ["Ascend", "GPU"]
    supported_formats = ['AIR', 'MINDIR']
    quant_mode_formats = ['QUANT', 'NONQUANT']

    quant_mode = kwargs['quant_mode']
    if quant_mode not in quant_mode_formats:
        raise KeyError(f'Quant_mode input is wrong, Please choose the right mode of the quant_mode.')
    if quant_mode == 'NONQUANT':
        return network
    quant_net = copy.deepcopy(network)
    quant_net._create_time = int(time.time() * 1e9)

    mean = 127.5 if kwargs.get('mean', None) is None else kwargs['mean']
    std_dev = 127.5 if kwargs.get('std_dev', None) is None else kwargs['std_dev']
    mean = Validator.check_value_type("mean", mean, (int, float))
    std_dev = Validator.check_value_type("std_dev", std_dev, (int, float))

    if context.get_context('device_target') not in supported_device:
        raise KeyError("Unsupported {} device target.".format(context.get_context('device_target')))

    if file_format not in supported_formats:
        raise ValueError('Illegal file format {}.'.format(file_format))

    quant_net.set_train(False)
    if file_format == "MINDIR":
        exporter = quant_export.ExportToQuantInferNetwork(quant_net, mean, std_dev, *inputs, is_mindir=True)
    else:
        exporter = quant_export.ExportToQuantInferNetwork(quant_net, mean, std_dev, *inputs)
    deploy_net = exporter.run()
    return deploy_net


def parse_print(print_file_name):
    """
    Load Print data from a specified file.

    Args:
        print_file_name (str): The file name of saved print data.

    Returns:
        List, element of list is Tensor.

    Raises:
        ValueError: The print file may be empty, please make sure enter the correct file name.
    """
    print_file_path = os.path.realpath(print_file_name)

    if os.path.getsize(print_file_path) == 0:
        raise ValueError("The print file may be empty, please make sure enter the correct file name.")

    logger.info("Execute load print process.")
    print_list = Print()

    try:
        with open(print_file_path, "rb") as f:
            pb_content = f.read()
        print_list.ParseFromString(pb_content)
    except BaseException as e:
        logger.error("Failed to read the print file %s, please check the correct of the file.", print_file_name)
        raise ValueError(e.__str__())

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
                np_type = tensor_to_np_type[data_type]
                param_data = np.fromstring(data, np_type)
                ms_type = tensor_to_ms_type[data_type]
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
        logger.error("Failed to load the print file %s.", print_list)
        raise RuntimeError(e.__str__())

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
        raise ValueError(f"{e.__str__()}. Please make sure that strategy matches the node_strategy.proto.")

    device_count = 1
    for dim in dev_mat:
        device_count *= dim

    if len(sliced_data) != device_count:
        raise ValueError(f"The sliced_parameters length should be equal to device_count. "
                         f"the sliced_parameters length is {len(sliced_data)} but device_count is {device_count}.")

    if not param_split_shape:
        if not is_even:
            raise ValueError("The shape of every parameter in sliced_parameters should be the same "
                             "when slice manner is even.")

        all_gather_tensor = Tensor(np.concatenate(sliced_data))

        if field_size > 0:
            from mindspore.parallel._tensor import _reshape_param_data_with_weight
            merged_tensor = _reshape_param_data_with_weight(all_gather_tensor, dev_mat, field_size)

        else:
            from mindspore.parallel._tensor import _reshape_param_data
            merged_tensor = _reshape_param_data(all_gather_tensor, dev_mat, tensor_map)

    else:
        tensor_strategy = _get_tensor_strategy(dev_mat, tensor_map)

        slice_count = 1
        for dim in tensor_strategy:
            slice_count *= dim

        if len(param_split_shape) != slice_count:
            raise ValueError(f"The param_split_shape length in strategy should be {slice_count}, "
                             f"but got {len(param_split_shape)}.")

        tensor_slices_new = list(range(slice_count))
        tensor_slices = sliced_data
        for i in range(device_count):
            slice_index = int(_get_tensor_slice_index(dev_mat, tensor_strategy, tensor_map, i))
            if tensor_slices[i].shape[0] != param_split_shape[slice_index]:
                raise ValueError(f"The slice {slice_index} is {param_split_shape[slice_index]} in 0 axis, "
                                 f"but got {tensor_slices[i].shape[0]}.")
            tensor_slices_new[slice_index] = np.array(tensor_slices[i])

        dim_len = len(tensor_strategy)
        for i in range(dim_len):
            ele_count = int(len(tensor_slices_new) / tensor_strategy[dim_len - 1 - i])
            tensor_slices_new_inner = []
            for j in range(ele_count):
                new_tensor = tensor_slices_new[j * tensor_strategy[dim_len - 1 - i]]
                for l in range(j * tensor_strategy[dim_len - 1 - i] + 1,
                               (j + 1) * tensor_strategy[dim_len - 1 - i]):
                    new_tensor = np.concatenate((new_tensor, tensor_slices_new[l]), axis=dim_len - 1 - i)
                tensor_slices_new_inner.insert(len(tensor_slices_new_inner), np.array(new_tensor))
            tensor_slices_new = tensor_slices_new_inner
        merged_tensor = Tensor(tensor_slices_new[0])

    return merged_tensor


def build_searched_strategy(strategy_filename):
    """
    Build strategy of every parameter in network.

    Args:
        strategy_filename (str): Name of strategy file.

    Returns:
        Dict, whose key is parameter name and value is slice strategy of this parameter.

    Raises:
        ValueError: Strategy file is incorrect.
        TypeError: strategy_filename is not str.
    """
    if not isinstance(strategy_filename, str):
        raise TypeError(f"The strategy_filename should be str, but got {type(strategy_filename)}.")

    if not os.path.isfile(strategy_filename):
        raise ValueError(f"No such strategy file: {strategy_filename}.")

    if os.path.getsize(strategy_filename) == 0:
        raise ValueError("The strategy file should not be empty.")

    parallel_strategy_map = ParallelStrategyMap()

    with open(strategy_filename, 'rb') as f:
        pb_content = f.read()
    parallel_strategy_map.ParseFromString(pb_content)

    layout_items = parallel_strategy_map.parallel_layout_item
    if not layout_items:
        raise ValueError("The strategy file has no sliced parameter.")

    strategy = {}
    for layout_item in layout_items:
        parameter_name = layout_item.param_name
        layout = layout_item.parallel_layouts
        strategy[parameter_name] = layout

    return strategy


def merge_sliced_parameter(sliced_parameters, strategy=None):
    """
    Merge parameter slices to one whole parameter.

    Args:
        sliced_parameters (list[Parameter]): Parameter slices in order of rank_id.
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
        >>> from mindspore import Tensor, merge_sliced_parameter, Parameter
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
        >>> merged_parameter = merge_sliced_parameter(sliced_parameters)
        >>> print(merged_parameter)
        Parameter (name=network.embedding_table, shape=(12,), dtype=Float64, requires_grad=True)
    """
    if not isinstance(sliced_parameters, list):
        raise TypeError(f"The sliced_parameters should be list, but got {type(sliced_parameters)}.")

    if not sliced_parameters:
        raise ValueError("The sliced_parameters should not be empty.")

    if strategy and not isinstance(strategy, dict):
        raise TypeError(f"The strategy should be dict, but got {type(strategy)}.")

    try:
        parameter_name = sliced_parameters[0].name
        parameter_shape = sliced_parameters[0].data.shape
        parameter_shape_length = len(parameter_shape)
    except BaseException as e:
        raise TypeError(f"{e.__str__()}. the element in sliced_parameters should be Parameter.")

    is_even = True
    for index, parameter in enumerate(sliced_parameters):
        if not isinstance(parameter, Parameter):
            raise TypeError(f"The element in sliced_parameters should be Parameter, "
                            f"but got {type(parameter)} at index {index}.")

        if parameter.name != parameter_name \
                or len(parameter.data.shape) != parameter_shape_length \
                or parameter.data.shape[1:] != parameter_shape[1:]:
            raise ValueError("Please make sure that the elements in slice_parameters have the same name, "
                             "dimension length and shape except 0 axis")

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
            raise KeyError(f"The parameter name should be one key of strategy. "
                           f"the parameter name is {parameter_name}.")
        merged_tensor = _merge_param_with_strategy(sliced_data, parameter_name, strategy, is_even)
        merged_parameter = Parameter(merged_tensor, parameter_name, requires_grad, layerwise_parallel)

    return merged_parameter


def load_distributed_checkpoint(network, checkpoint_filenames, predict_strategy=None,
                                train_strategy_filename=None, strict_load=False, dec_key=None, dec_mode='AES-GCM'):
    """
    Load checkpoint into net for distributed predication.

    Args:
        network (Cell): Network for distributed predication.
        checkpoint_filenames (list[str]): The name of Checkpoint files in order of rank id.
        predict_strategy (dict): Strategy of predication process, whose key is parameter name, and value is a list or
            a tuple that the first four elements are [dev_matrix, tensor_map, param_split_shape, field]. If None,
            it means that the predication process just uses single device. Default: None.
        train_strategy_filename (str): Train strategy proto file name. Default: None.
        strict_load (bool): Whether to strict load the parameter into net. If False, it will load parameter
                            into net when parameter name's suffix in checkpoint file is the same as the
                            parameter in the network. When the types are inconsistent perform type conversion
                            on the parameters of the same type, such as float32 to float16. Default: False.
        dec_key (Union[None, bytes]): Byte type key used for decryption. If the value is None, the decryption
                                      is not required. Default: None.
        dec_mode (str): This parameter is valid only when dec_key is not set to None. Specifies the decryption
                        mode, currently supports 'AES-GCM' and 'AES-CBC'. Default: 'AES-GCM'.

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
        raise ValueError(
            f"The length of checkpoint_filenames should be equal to the device count of training process. "
            f"The length is {ckpt_file_len} but the device count is {train_dev_count}.")

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

        param_rank = rank_list[param.name][0]
        skip_merge_split = rank_list[param.name][1]
        shard_stride = train_strategy[param.name][4]
        if train_strategy[param.name][5]:
            shard_size = ckpt_file_len / shard_stride / train_strategy[param.name][5]
        else:
            shard_size = 0
        for rank in param_rank:
            param_total_list = list(range(0, ckpt_file_len))
            if shard_size > 0:
                shard_total_list = [param_total_list[i:i + shard_size] for i in
                                    range(0, ckpt_file_len, shard_size)]
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
                logger.error("Failed to load opt shard slice in load distributed checkpoint for {}. Data shape is {}"
                             " and group is {}".format(param.name, split_param.data.shape, opt_shard_group))
                raise RuntimeError(e.__str__())
            split_param = Parameter(Tensor(data_slice), param.name,
                                    split_param.requires_grad, split_param.layerwise_parallel)
        param_dict[param.name] = split_param

    if param_not_in_strategy:
        logger.warning("{} parameters in network are not in the slice strategy.".format(param_not_in_strategy))
    if param_not_in_ckpt:
        logger.warning("{} parameters in slice strategy but not in the checkpoint file.".format(param_not_in_ckpt))

    load_param_into_net(network, param_dict, strict_load=strict_load)


def async_ckpt_thread_status():
    """
    Get the status of asynchronous save checkpoint thread.

    When performing asynchronous save checkpoint, you can get the thread state through this function
    to ensure that write checkpoint file are completed.

    Returns:
        True, Asynchronous save checkpoint thread is running.
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
        raise ValueError(f"Please make sure that the key of predict_strategy is str, "
                         f"and the value is a list or a tuple that the first four elements are "
                         f"dev_matrix (list[int]), tensor_map (list[int]), "
                         f"param_split_shape (list[int]) and field_size (zero).")


def _check_checkpoint_file(checkpoint_filenames):
    """Check checkpoint file name."""
    for index, filename in enumerate(checkpoint_filenames):
        if not isinstance(filename, str) or not os.path.exists(filename) \
                or filename[-5:] != ".ckpt" or os.path.getsize(filename) == 0:
            raise ValueError(f"Please make sure that the {filename} at index {index} is a valid checkpoint file.")


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
            train_map[param_name] = [dev_mat, tensor_map, param_split_shape, field_size, shard_stride, shard_size]
        except BaseException as e:
            raise ValueError(f"{e.__str__()}. Please make sure that strategy matches the node_strategy.proto.")
    return train_map


def _convert_to_layout(param_name, tensor_layout):
    """Convert list to ParallelLayouts object."""
    strategy = {}
    try:
        layout = ParallelLayouts()
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
        raise ValueError("Convert failed. " + e.__str__())

    strategy[param_name] = layout
    return strategy


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
