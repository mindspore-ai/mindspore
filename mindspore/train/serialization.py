# Copyright 2020 Huawei Technologies Co., Ltd
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
from threading import Thread, Lock
import numpy as np

import mindspore.nn as nn
import mindspore.context as context
from mindspore import log as logger
from mindspore.train.checkpoint_pb2 import Checkpoint
from mindspore.train.print_pb2 import Print
from mindspore.train.node_strategy_pb2 import ParallelStrategyMap, ParallelLayouts
from mindspore.train.mind_ir_pb2 import ModelProto as mindir_model
from mindspore.train.mind_ir_pb2 import GraphProto as graph_proto
from mindspore.common.tensor import Tensor
from mindspore.common.initializer import initializer
from mindspore.common.parameter import Parameter
from mindspore.common.api import _executor
from mindspore.common import dtype as mstype
from mindspore._checkparam import check_input_data, Validator
from mindspore.compression.export import quant_export
from mindspore.parallel._tensor import _load_tensor
from mindspore.parallel._utils import _infer_rank_list, _remove_repeated_slices
from .._c_expression import load_mindir


tensor_to_ms_type = {"Int8": mstype.int8, "Uint8": mstype.uint8, "Int16": mstype.int16, "Uint16": mstype.uint16,
                     "Int32": mstype.int32, "Uint32": mstype.uint32, "Int64": mstype.int64, "Uint64": mstype.uint64,
                     "Float16": mstype.float16, "Float32": mstype.float32, "Float64": mstype.float64,
                     "Bool": mstype.bool_}

tensor_to_np_type = {"Int8": np.int8, "Uint8": np.uint8, "Int16": np.int16, "Uint16": np.uint16,
                     "Int32": np.int32, "Uint32": np.uint32, "Int64": np.int64, "Uint64": np.uint64,
                     "Float16": np.float16, "Float32": np.float32, "Float64": np.float64, "Bool": np.bool_}

_ckpt_mutex = Lock()
SLICE_SIZE = 512 * 1024 * 1024
TOTAL_SAVE = 1024 * 1024


def _special_process_par(par, new_par):
    """
    Processes the special condition.

    Like (12,2048,1,1)->(12,2048), this case is caused by GE 4 dimensions tensor.
    """
    par_shape_len = len(par.data.shape)
    new_par_shape_len = len(new_par.data.shape)
    delta_len = new_par_shape_len - par_shape_len
    delta_i = 0
    for delta_i in range(delta_len):
        if new_par.data.shape[par_shape_len + delta_i] != 1:
            break
    if delta_i == delta_len - 1:
        new_val = new_par.data.asnumpy()
        new_val = new_val.reshape(par.data.shape)
        par.set_data(Tensor(new_val, par.data.dtype))
        return True
    return False


def _update_param(param, new_param):
    """Updates param's data from new_param's data."""

    if isinstance(param.data, Tensor) and isinstance(new_param.data, Tensor):
        if param.data.dtype != new_param.data.dtype:
            logger.error("Failed to combine the net and the parameters for param %s.", param.name)
            msg = ("Net parameters {} type({}) different from parameter_dict's({})"
                   .format(param.name, param.data.dtype, new_param.data.dtype))
            raise RuntimeError(msg)

        if param.data.shape != new_param.data.shape:
            if not _special_process_par(param, new_param):
                logger.error("Failed to combine the net and the parameters for param %s.", param.name)
                msg = ("Net parameters {} shape({}) different from parameter_dict's({})"
                       .format(param.name, param.data.shape, new_param.data.shape))
                raise RuntimeError(msg)
            return

        param.set_data(new_param.data)
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


def _exec_save(ckpt_file_name, data_list):
    """Execute the process of saving checkpoint into file."""

    try:
        with _ckpt_mutex:
            if os.path.exists(ckpt_file_name):
                os.remove(ckpt_file_name)
            with open(ckpt_file_name, "ab") as f:
                for name, value in data_list.items():
                    data_size = value[2].nbytes
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

                        f.write(checkpoint_list.SerializeToString())

        os.chmod(ckpt_file_name, stat.S_IRUSR)

    except BaseException as e:
        logger.error("Failed to save the checkpoint file %s.", ckpt_file_name)
        raise e


def save_checkpoint(save_obj, ckpt_file_name, integrated_save=True, async_save=False):
    """
    Saves checkpoint info to a specified file.

    Args:
        save_obj (Union[Cell, list]): The cell object or data list(each element is a dictionary, like
                                      [{"name": param_name, "data": param_data},...], the type of
                                      param_name would be string, and the type of param_data would
                                      be parameter or `Tensor`).
        ckpt_file_name (str): Checkpoint file name. If the file name already exists, it will be overwritten.
        integrated_save (bool): Whether to integrated save in automatic model parallel scene. Default: True
        async_save (bool): Whether asynchronous execution saves the checkpoint to a file. Default: False

    Raises:
        TypeError: If the parameter save_obj is not `nn.Cell` or list type. And if the parameter
                   `integrated_save` and `async_save` are not bool type.
    """

    if not isinstance(save_obj, nn.Cell) and not isinstance(save_obj, list):
        raise TypeError("The parameter save_obj should be nn.Cell or list, but got {}".format(type(save_obj)))
    integrated_save = Validator.check_bool(integrated_save)
    async_save = Validator.check_bool(async_save)

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

            # in automatic model parallel scenario, some parameters were spliteds to all the devices,
            # which should be combined before saving
            if key in save_obj.parameter_layout_dict:
                param_data = _get_merged_param_data(save_obj, key, param_data, integrated_save)

            each_param["data"] = param_data
            param_list.append(each_param)
        save_obj = param_list

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

    if async_save:
        thr = Thread(target=_exec_save, args=(ckpt_file_name, data_list), name="asyn_save_ckpt")
        thr.start()
    else:
        _exec_save(ckpt_file_name, data_list)

    logger.info("Saving checkpoint process is finished.")


def _check_param_prefix(filter_prefix, param_name):
    """Checks whether the prefix of parameter name matches the given filter_prefix."""
    for prefix in filter_prefix:
        if param_name.find(prefix) == 0 \
                and (param_name == prefix or param_name[len(prefix)] == "." or (prefix and prefix[-1] == ".")):
            return True
    return False


def load(file_name):
    """
    Load MindIR.

    The returned object can be executed by a `GraphCell`. However, there are some limitations to the current use
    of `GraphCell`, see class :class:`mindspore.nn.GraphCell` for more details.

    Args:
        file_name (str): MindIR file name.

    Returns:
        Object, a compiled graph that can executed by `GraphCell`.

    Raises:
        ValueError: MindIR file is incorrect.

    Examples:
        >>> import numpy as np
        >>> import mindspore.nn as nn
        >>> from mindspore import Tensor
        >>> from mindspore.train import export, load
        >>>
        >>> net = nn.Conv2d(1, 1, kernel_size=3)
        >>> input = Tensor(np.ones([1, 1, 3, 3]).astype(np.float32))
        >>> export(net, input, file_name="net", file_format="MINDIR")
        >>> graph = load("net.mindir")
        >>> net = nn.GraphCell(graph)
        >>> output = net(input)
    """
    if not isinstance(file_name, str):
        raise ValueError("The file name must be string.")
    if not os.path.exists(file_name):
        raise ValueError("The file does not exist.")
    if not file_name.endswith(".mindir"):
        raise ValueError("The MindIR should end with mindir, please input the correct file name.")

    logger.info("Execute the process of loading mindir.")
    graph = load_mindir(file_name)
    if graph is None:
        raise RuntimeError("Load MindIR failed.")
    return graph


def load_checkpoint(ckpt_file_name, net=None, strict_load=False, filter_prefix=None):
    """
    Loads checkpoint info from a specified file.

    Args:
        ckpt_file_name (str): Checkpoint file name.
        net (Cell): Cell network. Default: None
        strict_load (bool): Whether to strict load the parameter into net. If False, it will load parameter
                           in the param_dict into net with the same suffix. Default: False
        filter_prefix (Union[str, list[str], tuple[str]]): Parameters starting with the filter_prefix
            will not be loaded. Default: None.

    Returns:
        Dict, key is parameter name, value is a Parameter.

    Raises:
        ValueError: Checkpoint file is incorrect.

    Examples:
        >>> from mindspore import load_checkpoint
        >>>
        >>> ckpt_file_name = "./checkpoint/LeNet5-1_32.ckpt"
        >>> param_dict = load_checkpoint(ckpt_file_name, filter_prefix="conv1")
    """
    ckpt_file_name, filter_prefix = _check_checkpoint_param(ckpt_file_name, filter_prefix)
    logger.info("Execute the process of loading checkpoint files.")
    checkpoint_list = Checkpoint()

    try:
        with open(ckpt_file_name, "rb") as f:
            pb_content = f.read()
        checkpoint_list.ParseFromString(pb_content)
    except BaseException as e:
        logger.error("Failed to read the checkpoint file `%s`, please check the correct of the file.", ckpt_file_name)
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
    Loads parameters into network.

    Args:
        net (Cell): Cell network.
        parameter_dict (dict): Parameter dictionary.
        strict_load (bool): Whether to strict load the parameter into net. If False, it will load parameter
                           in the param_dict into net with the same suffix. Default: False

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
            _update_param(param, new_param)
        else:
            param_not_load.append(param.name)

    if param_not_load and not strict_load:
        _load_dismatch_prefix_params(net, parameter_dict, param_not_load)

    logger.debug("Params not matched(in net but not in parameter_dict):")
    for param_name in param_not_load:
        logger.debug("%s", param_name)

    logger.info("Loading parameters into net is finished.")
    if param_not_load:
        logger.warning("{} parameters in the net are not loaded.".format(len(param_not_load)))
    return param_not_load


def _load_dismatch_prefix_params(net, parameter_dict, param_not_load):
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
                    _update_param(param, new_param)
                    param_not_load.remove(param.name)


def _save_graph(network, file_name):
    """
    Saves the graph of network to a file.

    Args:
        network (Cell): Obtain a pipeline through network for saving graph.
        file_name (str): Graph file name into which the graph will be saved.
    """
    logger.info("Execute the process of saving graph.")

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
    from mindspore.parallel._tensor import _reshape_param_data, _reshape_param_data_with_weight
    layout = net.parameter_layout_dict[param_name]
    if len(layout) < 6:
        logger.info("layout dict does not contain the key %s", param_name)
        return param_data

    dev_mat = layout[0]
    tensor_map = layout[1]
    field_size = layout[3]
    uniform_split = layout[4]
    opt_shard_group = layout[5]

    allgather_net = None
    if param_name in net.parallel_parameter_merge_net_dict:
        allgather_net = net.parallel_parameter_merge_net_dict[param_name]
    else:
        logger.info("need to create allgather net for %s", param_name)

    if integrated_save:
        if uniform_split == 0:
            raise RuntimeError("Integrated save checkpoint only support uniform split tensor now.")
        # while any dim is not equal to -1, means param is split and needs to be merged
        # pipeline parallel need to be supported here later
        for dim in tensor_map:
            if dim != -1:
                if allgather_net is None:
                    if opt_shard_group:
                        allgather_net = get_allgather_cell(opt_shard_group, True)
                    else:
                        allgather_net = get_allgather_cell(opt_shard_group, False)
                    net.parallel_parameter_merge_net_dict[param_name] = allgather_net
                param_data = allgather_net(param_data)
                if field_size:
                    return _reshape_param_data_with_weight(param_data, dev_mat, field_size)
                return _reshape_param_data(param_data, dev_mat, tensor_map)
        if opt_shard_group:
            if allgather_net is None:
                allgather_net = get_allgather_cell(opt_shard_group, False)
                net.parallel_parameter_merge_net_dict[param_name] = allgather_net
            param_data = allgather_net(param_data)
    elif opt_shard_group:
        if allgather_net is None:
            allgather_net = get_allgather_cell(opt_shard_group, False)
            net.parallel_parameter_merge_net_dict[param_name] = allgather_net
        param_data = allgather_net(param_data)
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

    Notes:
        When exporting to AIR format, the size of a single tensor can not exceed 2GB.

    Args:
        net (Cell): MindSpore network.
        inputs (Tensor): Inputs of the `net`.
        file_name (str): File name of the model to be exported.
        file_format (str): MindSpore currently supports 'AIR', 'ONNX' and 'MINDIR' format for exported model.

            - AIR: Ascend Intermediate Representation. An intermediate representation format of Ascend model.
              Recommended suffix for output file is '.air'.
            - ONNX: Open Neural Network eXchange. An open format built to represent machine learning models.
              Recommended suffix for output file is '.onnx'.
            - MINDIR: MindSpore Native Intermediate Representation for Anf. An intermediate representation format
              for MindSpore models.
              Recommended suffix for output file is '.mindir'.

        kwargs (dict): Configuration options dictionary.

            - quant_mode: The mode of quant.
            - mean: Input data mean. Default: 127.5.
            - std_dev: Input data variance. Default: 127.5.
    """
    logger.info("exporting model file:%s format:%s.", file_name, file_format)
    check_input_data(*inputs, data_class=Tensor)
    if not isinstance(file_name, str):
        raise ValueError("Args file_name {} must be string, please check it".format(file_name))

    Validator.check_file_name_by_regular(file_name)
    net = _quant_export(net, *inputs, file_format=file_format, **kwargs)
    _export(net, file_name, file_format, *inputs)


def _export(net, file_name, file_format, *inputs):
    """
    It is an internal conversion function. Export the MindSpore prediction model to a file in the specified format.
    """
    logger.info("exporting model file:%s format:%s.", file_name, file_format)
    check_input_data(*inputs, data_class=Tensor)

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
        _executor.export(file_name, graph_id)
    elif file_format == 'ONNX':
        phase_name = 'export.onnx'
        graph_id, _ = _executor.compile(net, *inputs, phase=phase_name, do_convert=False)
        onnx_stream = _executor._get_func_graph_proto(net, graph_id)
        if not file_name.endswith('.onnx'):
            file_name += ".onnx"
        with open(file_name, 'wb') as f:
            os.chmod(file_name, stat.S_IRUSR | stat.S_IWUSR)
            f.write(onnx_stream)
    elif file_format == 'MINDIR':
        _save_mindir(net, file_name, *inputs)

    if is_dump_onnx_in_training:
        net.set_train(mode=True)


def _save_mindir(net, file_name, *inputs):
    """Save MindIR format file."""
    model = mindir_model()
    if net._auto_parallel_mode:
        phase_name = "predict"
    else:
        phase_name = 'export.mindir'
    graph_id, _ = _executor.compile(net, *inputs, phase=phase_name,
                                    do_convert=False, auto_parallel_mode=net._auto_parallel_mode)
    mindir_stream = _executor._get_func_graph_proto(net, graph_id, 'mind_ir')

    net_dict = net.parameters_dict()
    data_total = 0
    save_together = True

    model.ParseFromString(mindir_stream)
    for param_proto in model.graph.parameter:
        name = param_proto.name[param_proto.name.find(":") + 1:]
        if name in net_dict.keys():
            data_total += sys.getsizeof(net_dict[name].data.asnumpy().tobytes()) / 1024
        else:
            raise RuntimeError('Graph parameter: {} Undefined in network.'.format(param_proto.name))
        if data_total > TOTAL_SAVE:
            save_together = False
            break

    if save_together:
        for param_proto in model.graph.parameter:
            param_name = param_proto.name[param_proto.name.find(":")+1:]
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
        with open(file_name, 'wb') as f:
            os.chmod(file_name, stat.S_IRUSR | stat.S_IWUSR)
            f.write(model.SerializeToString())
    else:
        logger.warning("Parameters in the net capacity exceeds 1G, save MindIR model and parameters separately.")
        # save parameter
        file_prefix = file_name.split("/")[-1]
        if file_prefix.endswith(".mindir"):
            file_prefix = file_prefix[:-7]
        current_path = os.path.abspath(file_name)
        dirname = os.path.dirname(current_path)
        data_path = dirname + "/" + file_prefix + "_variables"
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
                data_file_name = data_path + "/" + "data_" + str(index)
                with open(data_file_name, "ab") as f:
                    os.chmod(data_file_name, stat.S_IRUSR | stat.S_IWUSR)
                    f.write(graphproto.SerializeToString())
                index += 1
                data_size = 0
                del graphproto.parameter[:]

        if graphproto.parameter:
            data_file_name = data_path + "/" + "data_" + str(index)
            with open(data_file_name, "ab") as f:
                os.chmod(data_file_name, stat.S_IRUSR | stat.S_IWUSR)
                f.write(graphproto.SerializeToString())

        # save graph
        del model.graph.parameter[:]
        graph_file_name = dirname + "/" + file_prefix + "_graph.mindir"
        with open(graph_file_name, 'wb') as f:
            os.chmod(graph_file_name, stat.S_IRUSR | stat.S_IWUSR)
            f.write(model.SerializeToString())


def _quant_export(network, *inputs, file_format, **kwargs):
    """
    Exports MindSpore quantization predict model to deploy with AIR and MINDIR.
    """
    if not kwargs.get('quant_mode', None):
        return network

    supported_device = ["Ascend", "GPU"]
    supported_formats = ['AIR', 'MINDIR']
    quant_mode_formats = ['AUTO', 'MANUAL']

    mean = 127.5 if kwargs.get('mean', None) is None else kwargs['mean']
    std_dev = 127.5 if kwargs.get('std_dev', None) is None else kwargs['std_dev']

    quant_mode = kwargs['quant_mode']
    if quant_mode not in quant_mode_formats:
        raise KeyError(f'Quant_mode input is wrong, Please choose the right mode of the quant_mode.')

    mean = Validator.check_value_type("mean", mean, (int, float))
    std_dev = Validator.check_value_type("std_dev", std_dev, (int, float))

    if context.get_context('device_target') not in supported_device:
        raise KeyError("Unsupported {} device target.".format(context.get_context('device_target')))

    if file_format not in supported_formats:
        raise ValueError('Illegal file format {}.'.format(file_format))

    network.set_train(False)
    if file_format == "MINDIR":
        if quant_mode == 'MANUAL':
            exporter = quant_export.ExportManualQuantNetwork(network, mean, std_dev, *inputs, is_mindir=True)
        else:
            exporter = quant_export.ExportToQuantInferNetwork(network, mean, std_dev, *inputs, is_mindir=True)
    else:
        if quant_mode == 'MANUAL':
            exporter = quant_export.ExportManualQuantNetwork(network, mean, std_dev, *inputs)
        else:
            exporter = quant_export.ExportToQuantInferNetwork(network, mean, std_dev, *inputs)
    deploy_net = exporter.run()
    return deploy_net


def parse_print(print_file_name):
    """
    Loads Print data from a specified file.

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

    merged_tensor = None
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
        from mindspore.parallel._tensor import _get_tensor_strategy, _get_tensor_slice_index
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
        Dictionary, whose key is parameter name and value is slice strategy of this parameter.

    Raises:
        ValueError: Strategy file is incorrect.
        TypeError: Strategy_filename is not str.

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
        >>> from mindspore.common.parameter import Parameter
        >>> from mindspore.train import merge_sliced_parameter
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
    merged_parameter = None

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


def load_distributed_checkpoint(network, checkpoint_filenames, predict_strategy=None):
    """
    Load checkpoint into net for distributed predication.

    Args:
        network (Cell): Network for distributed predication.
        checkpoint_filenames (list(str)): The name of Checkpoint files
            in order of rank id.
        predict_strategy (Optional(dict)): Strategy of predication process, whose key
            is parameter name, and value is a list or a tuple that the first four
            elements are [dev_matrix, tensor_map, param_split_shape, field]. If None,
            it means that the predication process just uses single device.
            Default: None.

    Raises:
        TypeError: The type of inputs do not match the requirements.
        ValueError: Failed to load checkpoint into net.
    """
    network = Validator.check_isinstance("network", network, nn.Cell)

    for index, filename in enumerate(checkpoint_filenames):
        if not isinstance(filename, str) or not os.path.exists(filename) \
                or filename[-5:] != ".ckpt" or os.path.getsize(filename) == 0:
            raise ValueError(f"Please make sure that the {filename} at index {index} is a valid checkpoint file.")

    if not _check_predict_strategy(predict_strategy):
        raise ValueError(f"Please make sure that the key of predict_strategy is str, "
                         f"and the value is a list or a tuple that the first four elements are "
                         f"dev_matrix (list[int]), tensor_map (list[int]), "
                         f"param_split_shape (list[int]) and field_size (zero).")

    train_strategy_filename = context.get_auto_parallel_context("strategy_ckpt_load_file")
    _train_strategy = build_searched_strategy(train_strategy_filename)
    train_strategy = _convert_to_list(_train_strategy)

    train_dev_count = 1
    for dim in train_strategy[list(train_strategy.keys())[0]][0]:
        train_dev_count *= dim
    if train_dev_count != len(checkpoint_filenames):
        raise ValueError(
            f"The length of checkpoint_filenames should be equal to the device count of training process. "
            f"The length is {len(checkpoint_filenames)} but the device count is {train_dev_count}.")

    rank_list = _infer_rank_list(train_strategy, predict_strategy)

    param_dict = {}
    for _, param in network.parameters_and_names():
        sliced_params = []
        if param.name not in rank_list.keys():
            continue
        param_rank = rank_list[param.name][0]
        skip_merge_split = rank_list[param.name][1]
        for rank in param_rank:
            sliced_param = _load_single_param(checkpoint_filenames[rank], param.name)
            sliced_params.append(sliced_param)
        if skip_merge_split:
            split_param = sliced_params[0]
        else:
            param_unique_strategy = _remove_repeated_slices(train_strategy[param.name])
            _param_unique_strategy = _convert_to_layout(param.name, param_unique_strategy)
            split_param = _merge_and_split(sliced_params, _param_unique_strategy, predict_strategy)
        param_dict[param.name] = split_param

    load_param_into_net(network, param_dict)


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
        return True

    predict_strategy = Validator.check_isinstance("predict_strategy", predict_strategy, dict)
    for key in predict_strategy.keys():
        if not isinstance(key, str) or not isinstance(predict_strategy[key], (list, tuple)) \
                or len(predict_strategy[key]) < 4:
            return False
        dev_matrix, tensor_map, param_split_shape, field_size = predict_strategy[key][:4]
        if not _check_int_list(dev_matrix) or not _check_int_list(tensor_map) or \
                not (_check_int_list(param_split_shape) or not param_split_shape) or \
                not (isinstance(field_size, int) and field_size == 0):
            return False
    return True


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
            train_map[param_name] = [dev_mat, tensor_map, param_split_shape, field_size]
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


def _load_single_param(ckpt_file_name, param_name):
    """Load a parameter from checkpoint."""
    checkpoint_list = Checkpoint()

    try:
        with open(ckpt_file_name, "rb") as f:
            pb_content = f.read()
        checkpoint_list.ParseFromString(pb_content)
    except BaseException as e:
        logger.error("Failed to read the checkpoint file `%s` during load single parameter,"
                     " please check the correct of the file.", ckpt_file_name)
        raise ValueError(e.__str__())

    parameter = None
    try:
        param_data_list = []
        for element_id, element in enumerate(checkpoint_list.value):
            if element.tag != param_name:
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
                    parameter = Parameter(Tensor(param_data, ms_type), name=element.tag)
                elif dims == [1]:
                    parameter = Parameter(Tensor(param_data, ms_type), name=element.tag)
                else:
                    param_dim = []
                    for dim in dims:
                        param_dim.append(dim)
                    param_value = param_data.reshape(param_dim)
                    parameter = Parameter(Tensor(param_value, ms_type), name=element.tag)
                break

    except BaseException as e:
        logger.error("Failed to load the checkpoint file `%s`.", ckpt_file_name)
        raise RuntimeError(e.__str__())

    if parameter is None:
        raise ValueError(f"There is no parameter named {param_name} in this checkpoint file {ckpt_file_name}, "
                         f"please check parameter name or checkpoint file.")
    return parameter
