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
"""Train utility."""
from __future__ import absolute_import

import os
from collections.abc import Iterable
import numpy as np

from mindspore.common.tensor import Tensor
from mindspore._c_expression import Tensor as Tensor_
from mindspore.common.dtype import dtype_to_nptype, pytype_to_dtype
from mindspore.common import dtype as mstype
from mindspore import log as logger
from mindspore._checkparam import Validator
from mindspore.common.api import _cell_graph_executor
from mindspore.train.mind_ir_pb2 import ModelProto as mindir_model
from mindspore.train.checkpoint_pb2 import Checkpoint
from mindspore.train.node_strategy_pb2 import ParallelStrategyMap as ckpt_strategy
from mindspore.train.lineage_pb2 import DatasetGraph, TrainLineage, EvaluationLineage, UserDefinedInfo
from mindspore.parallel._parallel_serialization import _make_dir
from mindspore.ops.operations import debug_ops


def _convert_type(types):
    """
    Convert from numpy type to tensor type.

    Args:
        types (list): Numpy type list of element in dataset.

    Returns:
        list, list of element in dataset.
    """
    ms_types = []
    for np_type in types:
        ms_type = pytype_to_dtype(np_type)
        ms_types.append(ms_type)
    return ms_types


def _get_types_and_shapes(dataset):
    """Get dataset types and shapes."""
    dataset_types = _convert_type(dataset.output_types())
    dataset_shapes = dataset.output_shapes()
    return dataset_types, dataset_shapes


def _exec_datagraph(exec_dataset, dataset_size, phase='dataset', create_data_info_queue=False):
    """Initialize and execute the dataset graph."""
    batch_size = exec_dataset.get_batch_size()
    input_indexs = exec_dataset.input_indexs

    # transform data format
    dataset_types, dataset_shapes = _get_types_and_shapes(exec_dataset)
    send_epoch_end = bool(dataset_size == -1)
    exec_dataset = exec_dataset.device_que(send_epoch_end=send_epoch_end, create_data_info_queue=create_data_info_queue)
    _cell_graph_executor.init_dataset(exec_dataset.queue_name,
                                      dataset_size,
                                      batch_size,
                                      dataset_types,
                                      dataset_shapes,
                                      input_indexs,
                                      phase=phase)
    return exec_dataset


def _make_directory(path, arg_name='path'):
    """Make directory."""
    return _make_dir(path, arg_name)


def _construct_tensor_list(types, shapes, batch_expand_num=1):
    """
    Construct list of tensors with types and shapes, used to initialize the network.

    Args:
        types: List or Tuple. The output types of element in dataset.
        shapes: List or Tuple. The output shapes of element in dataset.
        batch_expand_num (int): Batch expand number.

    Returns:
        List, list of Tensors.
    """
    if len(types) != len(shapes):
        raise ValueError("The length of dataset types must be equal to dataset shapes, "
                         "but got dataset types={} and dataset shapes={}".format(types, shapes))
    tensor_list = []
    for type_, shape in zip(types, shapes):
        new_shape = ()
        for i, item in enumerate(shape):
            if i == 0:
                new_shape += (item * batch_expand_num,)
            else:
                new_shape += (item,)
        tensor = Tensor(np.zeros(new_shape, dtype_to_nptype(type_)))
        tensor.virtual_flag = True
        tensor_list.append(tensor)
    return tensor_list


def _to_tensor(elem, scaling_sens=None):
    """Convert numpy to tensor, adapt to feed the data from host solution."""
    lst = []
    if not isinstance(elem, (tuple, list)):
        elem = [elem]
    for data in elem:
        if not isinstance(data, np.ndarray):
            if scaling_sens:
                elem_tuple = tuple(elem) + (Tensor(scaling_sens, mstype.float32),)
            else:
                elem_tuple = tuple(elem)
            return elem_tuple
        lst.append(Tensor(data))
    if scaling_sens:
        lst.append(Tensor(scaling_sens, mstype.float32))

    return lst[0] if len(lst) == 1 else tuple(lst)


def _construct_input_tensors(dataset_types, dataset_shapes, device_number=1):
    """Construct tensor list to initialize the network which implemented in dataset sink."""
    tensor_list_run = _construct_tensor_list(dataset_types, dataset_shapes, batch_expand_num=1)
    tensor_list_compile = _construct_tensor_list(dataset_types, dataset_shapes, batch_expand_num=device_number)
    return tensor_list_run, tensor_list_compile


def _check_to_numpy(plugin, tensor, prim=None):
    """Check the tensor and return a numpy.ndarray."""
    np_value = tensor.asnumpy()
    np_value = np_value.copy()
    summary_name = plugin.capitalize() + "Summary" if prim else "SummaryRecord"
    if plugin == 'scalar':
        if np_value.size == 1:
            return np_value
        raise ValueError(
            f'For "{summary_name}", the v rank must be less than or equal to 1, but got {len(np_value)}.')
    if plugin == 'image':
        if np_value.ndim == 4:
            return np_value
        raise ValueError(f'For "{summary_name}", The tensor seems not to hold a valid image.')
    if plugin in ('tensor', 'histogram'):
        if np_value.ndim > 0:
            return np_value
        raise ValueError(f'For "{summary_name}", The value should not be empty.')
    return np_value


def check_summary_param(summary_name, tag, tensor):
    """Checks the tag is valid for summary."""
    plugin = summary_name.split('Summary')[0].lower()
    try:
        if not isinstance(tag, str) or not tag:
            raise TypeError(f'For "{summary_name}", the name must be valid string, but got "{tag}".')
        if not isinstance(tensor, (Tensor, Tensor_)):
            raise TypeError(f'For "{summary_name}", the parameter "value" expect to be Tensor, '
                            f'but got {type(tensor).__name__}')
        _check_to_numpy(plugin, tensor, prim=True)
    except TypeError as err:
        raise TypeError(err)
    except ValueError as err:
        raise ValueError(err)
    finally:
        debug_ops.SUMMARY_TENSOR_CACHE = []


def _check_lineage_value(plugin, value):
    """Check the lineage value."""

    def raises(plugin, prototype):
        raise TypeError(f'Plugin {repr(plugin)} expects a {prototype.__name__} value.')

    if plugin == 'dataset_graph' and not isinstance(value, DatasetGraph):
        raises(plugin, DatasetGraph)

    if plugin == 'eval_lineage' and not isinstance(value, EvaluationLineage):
        raises(plugin, EvaluationLineage)

    if plugin == 'train_lineage' and not isinstance(value, TrainLineage):
        raises(plugin, TrainLineage)

    if plugin == 'custom_lineage_data' and not isinstance(value, UserDefinedInfo):
        raises(plugin, UserDefinedInfo)


def check_value_type(arg_name, arg_value, valid_types):
    """Checks whether a value is instance of some types."""
    valid_types = tuple(valid_types) if isinstance(valid_types, Iterable) else (valid_types,)
    is_valid = True

    # bool is subclass of int, so for a bool value, we need to extra check
    if isinstance(arg_value, int) and isinstance(arg_value, bool) and bool not in valid_types:
        is_valid = False

    if not isinstance(arg_value, valid_types):
        is_valid = False

    if not is_valid:
        raise TypeError(f'For `{arg_name}` the type should be a valid type of {[t.__name__ for t in valid_types]}, '
                        f'but got {type(arg_value).__name__}.')


def read_proto(file_name, proto_format="MINDIR", display_data=False):
    """
    Read protobuf file.

    Args:
        file_name (str): File name.
        proto_format (str): Proto format {MINDIR, CKPT, CKPT_STRATEGY}.  Default: MINDIR.
        display_data (bool): Whether display data. Default: False.

    Returns:
        Object, proto object.
    """
    Validator.check_file_name_by_regular(file_name)
    file_name = os.path.realpath(file_name)
    if proto_format == "MINDIR":
        model = mindir_model()
    elif proto_format == "CKPT":
        model = Checkpoint()
    elif proto_format == "CKPT_STRATEGY":
        model = ckpt_strategy()
    else:
        raise ValueError("Unsupported proto format.")

    try:
        with open(file_name, "rb") as f:
            pb_content = f.read()
            model.ParseFromString(pb_content)
    except BaseException as e:
        logger.critical(f"Failed to phase the file: {file_name} as format: {proto_format},"
                        f" please check the correct file and format.")
        raise ValueError(e.__str__()) from e
    finally:
        pass

    if proto_format == "MINDIR" and not display_data:
        for param_proto in model.graph.parameter:
            param_proto.raw_data = b'\0'

    if proto_format == "CKPT" and not display_data:
        for element in model.value:
            element.tensor.tensor_content = b'\0'

    return model
