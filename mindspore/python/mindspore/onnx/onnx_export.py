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

"""Model export to ONNX."""
from __future__ import absolute_import
from __future__ import division

import os

import mindspore.nn as nn
from mindspore import log as logger
from mindspore._checkparam import check_input_dataset
from mindspore import _checkparam as Validator
from mindspore.graph.api import _cell_graph_executor as _executor
from mindspore.train.serialization import _calculation_net_size
from mindspore.dataset.engine.datasets import Dataset

PROTO_LIMIT_SIZE = 1024 * 1024 * 2


def export(net, *inputs, file_name, input_names=None, output_names=None, export_params=True,
           keep_initializers_as_inputs=False, dynamic_axes=None):
    """
    Export the MindSpore network into an ONNX model.

    Note:
        - Support exporting network larger than 2GB. When the network exceeds 2GB,
          parameters are saved in additional binary files stored in the same directory as the ONNX file.
        - When `file_name` does not have a suffix, the system will automatically add the suffix `.onnx` .

    Args:
        net (Union[Cell, function]): MindSpore network.
        inputs (Union[Tensor, list, tuple, Number, bool]): It represents the inputs of the `net` , if the network has
            multiple inputs, set them together.
        file_name (str): File name of the model to be exported.
        input_names (list, optional): Names to assign to the input nodes of the graph, in order. Default: ``None`` .
        output_names (list, optional): Names to assign to the output nodes of the graph, in order. Default: ``None`` .
        export_params (bool, optional): If false, parameters (weights) will not be exported,
            parameters will add input nodes as input of the graph. Default: ``True`` .
        keep_initializers_as_inputs (bool, optional): If True, all the initializers (model parameters/weights) will
            add as inputs to the graph. This allows modifying any or all weights when running the exported ONNX model.
            Default: ``False`` .
        dynamic_axes (dict[str, dict[int, str]], optional): To specify axes of input tensors as dynamic (at runtime).
            Default: ``None`` .

            - Set a dict with scheme: {input_node_name: {axis_index:axis_name}},
              for example, {"input1": {0:"batch_size", 1: "seq_len"}, "input2": {0:"batch_size"}}.
            - By default, the shapes of all input tensors in the exported model exactly match those specified in
              `inputs`.

    Raises:
        ValueError: If the parameter `net` is not :class:`mindspore.nn.Cell`.
        ValueError: If the parameter `input_names` is not list type.
        ValueError: If the parameter `output_names` is not list type
        ValueError: If the parameter `dynamic_axes` is not dict type.

    Examples:
        >>> import mindspore as ms
        >>> import numpy as np
        >>> from mindspore import Tensor
        >>>
        >>> # Define the network structure of LeNet5. Refer to
        >>> # https://gitee.com/mindspore/docs/blob/master/docs/mindspore/code/lenet.py
        >>> net = LeNet5()
        >>> input_tensor = Tensor(np.ones([1, 1, 32, 32]).astype(np.float32))
        >>> ms.onnx.export(net, input_tensor, file_name='lenet.onnx', input_names=['input1'], output_names=['output1'])

    """
    Validator.check_file_name_by_regular(file_name)
    logger.info("exporting model file:%s format:%s.", file_name, "ONNX")
    Validator.check_isinstance("net", net, nn.Cell)
    input_names = input_names or []
    Validator.check_isinstance("input_names", input_names, list)
    output_names = output_names or []
    Validator.check_isinstance("output_names", output_names, list)
    dynamic_axes = dynamic_axes or {}
    Validator.check_isinstance("dynamic_axes", dynamic_axes, dict)

    if check_input_dataset(*inputs, dataset_type=Dataset):
        raise ValueError(f"Can not support dataset as inputs to export ONNX model.")

    cell_mode = net.training
    net.set_train(mode=False)

    extra_save_params = False
    total_size = _calculation_net_size(net)
    if total_size > PROTO_LIMIT_SIZE:
        logger.warning('Network size is: {}G, it exceeded the protobuf: {}G limit, now parameters in network are saved '
                       'in external data files.'.format(total_size / 1024 / 1024, PROTO_LIMIT_SIZE / 1024 / 1024))
        extra_save_params = True

    phase_name = 'export.onnx'
    graph_id, _ = _executor.compile(net, *inputs, phase=phase_name, do_convert=False)

    abs_file_name = os.path.abspath(file_name)
    if not abs_file_name.endswith('.onnx'):
        abs_file_name += ".onnx"

    dir_path = os.path.dirname(abs_file_name)
    if dir_path and not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)

    abs_file_dir = os.path.dirname(abs_file_name) if extra_save_params else ""

    onnx_stream = _executor._get_onnx_func_graph_proto(obj=net, exec_id=graph_id, input_names=input_names,
                                                       output_names=output_names, export_params=export_params,
                                                       keep_initializers_as_inputs=keep_initializers_as_inputs,
                                                       dynamic_axes=dynamic_axes, extra_save_params=extra_save_params,
                                                       save_file_dir=abs_file_dir)
    if onnx_stream is None:
        raise RuntimeError("Export onnx model failed, ensure that the model has been compiled correctly")

    try:
        with open(abs_file_name, 'wb') as f:
            f.write(onnx_stream)

        if os.path.getsize(abs_file_name) != len(onnx_stream):
            logger.warning("ONNX file size doesn't match expected value, but proceeding continue.")

    except IOError as e:
        logger.error(f"Failed to write ONNX file: {e}")
        if os.path.exists(abs_file_name):
            os.remove(abs_file_name)

    net.set_train(mode=cell_mode)
