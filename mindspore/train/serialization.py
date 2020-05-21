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
import stat
import numpy as np

import mindspore.nn as nn
import mindspore.context as context
from mindspore import log as logger
from mindspore.train.checkpoint_pb2 import Checkpoint
from mindspore.common.tensor import Tensor
from mindspore.common.initializer import initializer
from mindspore.common.parameter import Parameter
from mindspore.common.api import _executor
from mindspore.common import dtype as mstype
from mindspore._checkparam import check_input_data

__all__ = ["save_checkpoint", "load_checkpoint", "load_param_into_net", "export"]

tensor_to_ms_type = {"Int8": mstype.int8, "Int16": mstype.int16, "Int32": mstype.int32, "Int64": mstype.int64,
                     "Float16": mstype.float16, "Float32": mstype.float32, "Float64": mstype.float64}

tensor_to_np_type = {"Int8": np.int8, "Int16": np.int16, "Int32": np.int32, "Int64": np.int64,
                     "Float16": np.float16, "Float32": np.float32, "Float64": np.float64}

def _special_process_par(par, new_par):
    """
    Processes the special condition.

    Like (12,2048,1,1)->(12,2048), this case is caused by GE 4 dimensions tensor.
    """
    par_shape_len = len(par.data.shape())
    new_par_shape_len = len(new_par.data.shape())
    delta_len = new_par_shape_len - par_shape_len
    delta_i = 0
    for delta_i in range(delta_len):
        if new_par.data.shape()[par_shape_len + delta_i] != 1:
            break
    if delta_i == delta_len - 1:
        new_val = new_par.data.asnumpy()
        new_val = new_val.reshape(par.data.shape())
        par.set_parameter_data(Tensor(new_val, par.data.dtype()))
        return True
    return False


def _update_param(param, new_param):
    """Updates param's data from new_param's data."""

    if isinstance(param.data, Tensor) and isinstance(new_param.data, Tensor):
        if param.data.dtype() != new_param.data.dtype():
            logger.error("Failed to combine the net and the parameters for param %s.", param.name)
            msg = ("Net parameters {} type({}) different from parameter_dict's({})"
                   .format(param.name, param.data.dtype(), new_param.data.dtype()))
            raise RuntimeError(msg)

        if param.data.shape() != new_param.data.shape():
            if not _special_process_par(param, new_param):
                logger.error("Failed to combine the net and the parameters for param %s.", param.name)
                msg = ("Net parameters {} shape({}) different from parameter_dict's({})"
                       .format(param.name, param.data.shape(), new_param.data.shape()))
                raise RuntimeError(msg)
            return

        param.set_parameter_data(new_param.data)
        return

    if isinstance(param.data, Tensor) and not isinstance(new_param.data, Tensor):
        if param.data.shape() != (1,) and param.data.shape() != ():
            logger.error("Failed to combine the net and the parameters for param %s.", param.name)
            msg = ("Net parameters {} shape({}) is not (1,), inconsitent with parameter_dict's(scalar)."
                   .format(param.name, param.data.shape()))
            raise RuntimeError(msg)
        param.set_parameter_data(initializer(new_param.data, param.data.shape(), param.data.dtype()))

    elif isinstance(new_param.data, Tensor) and not isinstance(param.data, Tensor):
        logger.error("Failed to combine the net and the parameters for param %s.", param.name)
        msg = ("Net parameters {} type({}) different from parameter_dict's({})"
               .format(param.name, type(param.data), type(new_param.data)))
        raise RuntimeError(msg)

    else:
        param.set_parameter_data(type(param.data)(new_param.data))


def save_checkpoint(parameter_list, ckpoint_file_name):
    """
    Saves checkpoint info to a specified file.

    Args:
        parameter_list (list): Parameters list, each element is a dict
                               like {"name":xx, "type":xx, "shape":xx, "data":xx}.
        ckpoint_file_name (str): Checkpoint file name.

    Raises:
        RuntimeError: Failed to save the Checkpoint file.
    """
    logger.info("Execute save checkpoint process.")
    checkpoint_list = Checkpoint()

    try:
        for param in parameter_list:
            param_value = checkpoint_list.value.add()
            param_value.tag = param["name"]
            param_tensor = param_value.tensor
            if isinstance(param["data"], Parameter):
                param["data"].init_data()
            param_data = param["data"].asnumpy().reshape(-1)
            param_tensor.tensor_content = param_data.tostring()
            param_tensor.tensor_type = str(param["data"].dtype())

            if param['data'].shape() == ():
                param_tensor.dims.append(0)
            else:
                for dim in param['data'].shape():
                    param_tensor.dims.append(dim)

        with open(ckpoint_file_name, "wb") as f:
            f.write(checkpoint_list.SerializeToString())
        os.chmod(ckpoint_file_name, stat.S_IRUSR)

    except BaseException as e:
        logger.error("Failed to save the checkpoint file %s.", ckpoint_file_name)
        raise RuntimeError(e.__str__())
    logger.info("Save checkpoint process finish.")


def load_checkpoint(ckpoint_file_name, net=None):
    """
    Loads checkpoint info from a specified file.

    Args:
        ckpoint_file_name (str): Checkpoint file name.
        net (Cell): Cell network. Default: None

    Returns:
        Dict, key is parameter name, value is a Parameter.

    Raises:
        ValueError: Checkpoint file is incorrect.
    """
    if not isinstance(ckpoint_file_name, str):
        raise ValueError("The ckpoint_file_name must be String.")

    if not os.path.exists(ckpoint_file_name) or ckpoint_file_name[-5:] != ".ckpt":
        raise ValueError("Please input the correct checkpoint file name.")

    if os.path.getsize(ckpoint_file_name) == 0:
        raise ValueError("The checkpoint file may be empty, please make sure enter the correct file name.")

    logger.info("Execute load checkpoint process.")
    checkpoint_list = Checkpoint()

    try:
        with open(ckpoint_file_name, "rb") as f:
            pb_content = f.read()
        checkpoint_list.ParseFromString(pb_content)
    except BaseException as e:
        logger.error("Failed to read the checkpoint file %s, please check the correct of the file.", ckpoint_file_name)
        raise ValueError(e.__str__())

    parameter_dict = {}

    try:
        for element in checkpoint_list.value:
            data = element.tensor.tensor_content
            data_type = element.tensor.tensor_type
            np_type = tensor_to_np_type[data_type]
            ms_type = tensor_to_ms_type[data_type]
            param_data = np.fromstring(data, np_type)
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

        logger.info("Load checkpoint process finish.")

    except BaseException as e:
        logger.error("Failed to load the checkpoint file %s.", ckpoint_file_name)
        raise RuntimeError(e.__str__())

    if net:
        load_param_into_net(net, parameter_dict)

    return parameter_dict


def load_param_into_net(net, parameter_dict):
    """
    Loads parameters into network.

    Args:
        net (Cell): Cell network.
        parameter_dict (dict): Parameter dict.

    Raises:
        TypeError: Argument is not a Cell, or parameter_dict is not a Parameter dict.
    """
    if not isinstance(net, nn.Cell):
        logger.error("Failed to combine the net and the parameters.")
        msg = ("Argument net should be a Cell, but got {}.".format(type(net)))
        raise TypeError(msg)

    if not isinstance(parameter_dict, dict):
        logger.error("Failed to combine the net and the parameters.")
        msg = ("Argument parameter_dict should be a dict, but got {}.".format(type(parameter_dict)))
        raise TypeError(msg)

    logger.info("Execute load parameter into net process.")
    net.init_parameters_data()
    param_not_load = []
    for _, param in net.parameters_and_names():
        if param.name in parameter_dict:
            new_param = parameter_dict[param.name]
            if not isinstance(new_param, Parameter):
                logger.error("Failed to combine the net and the parameters.")
                msg = ("Argument parameter_dict element should be a Parameter, but got {}.".format(type(new_param)))
                raise TypeError(msg)
            param.init_data()
            _update_param(param, new_param)
        else:
            param_not_load.append(param.name)

    if param_not_load:
        _load_dismatch_prefix_params(net, parameter_dict, param_not_load)

    logger.debug("Params not matched(in net but not in parameter_dict):")
    for param_name in param_not_load:
        logger.debug("%s", param_name)

    logger.info("Load parameter into net finish, {} parameters has not been loaded.".format(len(param_not_load)))


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
    logger.info("Execute save the graph process.")

    graph_proto = network.get_func_graph_proto()
    if graph_proto:
        with open(file_name, "wb") as f:
            f.write(graph_proto)
        os.chmod(file_name, stat.S_IWUSR | stat.S_IRUSR)


def _exec_save_checkpoint(train_network, ckpoint_file_name, integrated_save=True):
    """
    Saves checkpoint for 'ms' backend.

    Args:
        train_network (Network): The train network for training.
        ckpoint_file_name (str): The name of checkpoint file.
        integrated_save (bool): Whether to intergrated save in automatic model parallel scene.
    """

    param_dict = {}
    for _, param in train_network.parameters_and_names():
        param_dict[param.name] = param

    param_list = []
    for (key, value) in param_dict.items():
        each_param = {"name": key}
        value.init_data()
        if isinstance(value.data, Tensor):
            param_data = value.data
        else:
            param_data = Tensor(value.data)

        # in automatic model parallel scenario, some parameters were spliteds to all the devices,
        # which should be combined before saving
        if integrated_save and key in train_network.parameter_layout_dict:
            param_data = _get_merged_param_data(train_network, key, param_data)

        each_param["data"] = param_data
        param_list.append(each_param)

    save_checkpoint(param_list, ckpoint_file_name)


def _get_merged_param_data(net, param_name, param_data):
    """
    Gets the merged data(tensor) from tensor slice, by device arrangement and tensor map.

    Args:
        net (Cell): MindSpore network.
        param_name(str): The parameter name, which to be combined.
        param_data(Tensor):The parameter data on the local device,
                           It was a slice of the whole parameter data.
    Returns:
        Tensor, the combined tensor which with the whole data value.
    """
    layout = []
    layout = net.parameter_layout_dict[param_name]
    if len(layout) < 2:
        logger.info("layout dict does not contain the key %s", param_name)
        return param_data

    dev_mat = layout[0]
    tensor_map = layout[1]

    from mindspore.parallel._cell_wrapper import get_allgather_cell
    from mindspore.parallel._tensor import _reshape_param_data
    # while any dim is not equal to -1, means param is splited and needs to be merged
    for dim in tensor_map:
        if dim != -1:
            allgather_net = get_allgather_cell()
            param_data = allgather_net(param_data)
            return _reshape_param_data(param_data, dev_mat, tensor_map)

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


def export(net, *inputs, file_name, file_format='GEIR'):
    """
    Exports MindSpore predict model to file in specified format.

    Args:
        net (Cell): MindSpore network.
        inputs (Tensor): Inputs of the `net`.
        file_name (str): File name of model to export.
        file_format (str): MindSpore currently supports 'GEIR', 'ONNX' and 'LITE' format for exported model.

            - GEIR: Graph Engine Intermidiate Representation. An intermidiate representation format of
              Ascend model.
            - ONNX: Open Neural Network eXchange. An open format built to represent machine learning models.
            - LITE: Huawei model format for mobile. A lite model only for the MindSpore Lite
    """
    logger.info("exporting model file:%s format:%s.", file_name, file_format)
    check_input_data(*inputs, data_class=Tensor)

    supported_formats = ['GEIR', 'ONNX', 'LITE']
    if file_format not in supported_formats:
        raise ValueError(f'Illegal file format {file_format}, it must be one of {supported_formats}')
    # switch network mode to infer when it is training
    is_training = net.training
    if is_training:
        net.set_train(mode=False)
    # export model
    if file_format == 'GEIR':
        _executor.compile(net, *inputs, phase='export')
        _executor.export(net, file_name, file_format)
    elif file_format == 'ONNX':  # file_format is 'ONNX'
        phase_name = 'export_onnx'
        graph_id, _ = _executor.compile(net, *inputs, phase=phase_name, do_convert=False)
        onnx_stream = _executor._get_func_graph_proto(graph_id)
        with open(file_name, 'wb') as f:
            os.chmod(file_name, stat.S_IWUSR | stat.S_IRUSR)
            f.write(onnx_stream)
    elif file_format == 'LITE':  # file_format is 'LITE'
        context.set_context(save_ms_model=True, save_ms_model_path=file_name)
        net(*inputs)
    # restore network training mode
    if is_training:
        net.set_train(mode=True)
