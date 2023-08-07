# Copyright 2022 Huawei Technologies Co., Ltd
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
"""Data sink help for minddata dataset"""
from functools import wraps
import mindspore.ops as ops
from mindspore import context
from mindspore.common.dtype import pytype_to_dtype
from mindspore.common.api import jit
from mindspore.train._utils import _exec_datagraph, _get_types_and_shapes
from mindspore.train.dataset_helper import _has_dynamic_shape, _check_inputs
import mindspore.dataset as ds
from mindspore._c_expression import _set_dataset_mode_config
from mindspore.parallel._utils import _get_device_num, _need_to_full, _to_full_shapes, _get_pipeline_stages
from mindspore import _checkparam as Validator


def _init_sink_dataset(dataset, sink_size, input_signature, create_info):
    """
    Initialize data sinking
    """
    if hasattr(dataset, '__transfer_dataset__'):
        raise ValueError(f"The dataset has been used with network.")

    dataset_size = dataset.get_dataset_size()
    dataset_types, dataset_shapes = _get_types_and_shapes(dataset)
    dynamic_shape = _has_dynamic_shape(dataset_shapes) or ds.config.get_dynamic_shape()

    # create transfer_dataset
    is_info_queue = (create_info and sink_size == 1 and dataset_size != 1 and
                     input_signature is None and not dynamic_shape and
                     context.get_context('device_target') == 'Ascend')
    transfer_dataset = _exec_datagraph(dataset, sink_size, create_data_info_queue=is_info_queue)
    dataset.__transfer_dataset__ = transfer_dataset

    # send data
    transfer_dataset.send(-1)

    # create GetNext op
    if input_signature is not None:
        _check_inputs(input_signature, dataset_shapes, dataset_types)

    queue_name = transfer_dataset.queue_name
    if _need_to_full() and context.get_context('mode') == context.GRAPH_MODE:
        device_num = _get_device_num() // _get_pipeline_stages()
        dataset_shapes = _to_full_shapes(dataset_shapes, device_num)
    next_op = ops.GetNext(dataset_types, dataset_shapes, len(dataset_types), queue_name)

    _set_dataset_mode_config('sink')

    dataset.__transfer_dataset__ = transfer_dataset

    return next_op, is_info_queue


class _DataSinkAux:
    @staticmethod
    def __deepcopy__(memodict):
        return


def _get_next_op(dataset, ori_next_op, is_info_queue):
    """
    get the next operation.
    """

    if not is_info_queue:
        return ori_next_op, ''

    if not hasattr(dataset, '__sink_aux__'):
        dataset.__sink_aux__ = _DataSinkAux()
        dataset.__sink_aux__.next_ops = {}
        dataset.__sink_aux__.sink_funcs = {}

    queue_name = dataset.__transfer_dataset__.queue_name
    dataset_types, dataset_shapes = dataset.__transfer_dataset__.get_data_info()
    dataset_types = [pytype_to_dtype(x) for x in dataset_types]
    key = str(dataset_types) + str(dataset_shapes)
    if key in dataset.__sink_aux__.next_ops:
        next_op = dataset.__sink_aux__.next_ops[key]
    else:
        if _need_to_full() and context.get_context('mode') == context.GRAPH_MODE:
            device_num = _get_device_num() // _get_pipeline_stages()
            dataset_shapes = _to_full_shapes(dataset_shapes, device_num)
        next_op = ops.GetNext(dataset_types, dataset_shapes, len(dataset_types), queue_name)

    return next_op, (key, dataset_shapes, dataset_types)


def _get_sink_fun(sink_fun, key_info, is_info_queue, dataset, jit_config):
    """
    get the sink function.
    """
    if not is_info_queue:
        if not hasattr(dataset, '__sink_fun__'):
            if jit_config is None:
                dst_sink_fun = sink_fun
            else:
                dst_sink_fun = jit(sink_fun, jit_config=jit_config)
            dataset.__sink_fun__ = dst_sink_fun

        return dataset.__sink_fun__

    key = key_info[0]
    if key in dataset.__sink_aux__.sink_funcs:
        dst_sink_fun = dataset.__sink_aux__.sink_funcs[key]
    else:
        if jit_config is None:
            dst_sink_fun = sink_fun
        else:
            dst_sink_fun = jit(sink_fun, jit_config=jit_config)
        dataset.__sink_aux__.sink_funcs[key] = dst_sink_fun

    return dst_sink_fun


def data_sink(fn, dataset, sink_size=1, jit_config=None, input_signature=None):
    """
    A wrapper function to generate a function for the input function.

    Note:
        When using data sinking, the dataset will be automatically sent in a loop, and only the step size of sinking
        `sink_size` needs to be considered. The default value of `sink_size` is ``1``, which means that all data will
        be sunk every epoch. If `sink_size` is greater than 1, the amount of data sunk per epoch will be the dataset
        with a size of `sink_size`.

    Args:
        fn (Function): The Python function that will be run with dataset.
        dataset (Dataset): The dataset iterator. The dataset can be generated by dataset generator API in
            :class:`mindspore.dataset`, such as :class:`mindspore.dataset.ImageFolderDataset`.
        sink_size (int): Control the amount of data in each sink. `sink_size` must be positive integer. Default: ``1`` .
        jit_config (JitConfig): Controls the execution mode(Graph mode/PyNative mode) of the generated function, and Jit
            config for compile. Default: ``None`` , means running in PyNative mode.
        input_signature (Union[Tensor, List or Tuple of Tensors]): The Tensor which describes the input arguments.
            The shape and dtype of the Tensor will be supplied to this function. If input_signature is specified,
            each input to `fn` must be a `Tensor`. And the input parameters of `fn` cannot accept `**kwargs`. The shape
            and dtype of actual inputs should keep the same as input_signature. Otherwise, TypeError will be raised.
            Default: ``None`` .

    Returns:
        Function, the generated function will be executed in data sinking mode.

    Raises:
        ValueError: If `sink_size` is not positive integer.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> import numpy as np
        >>> import mindspore as ms
        >>> from mindspore import dataset as ds
        >>>
        >>> data = {"x": np.ones((1,), dtype=np.int32), "y": np.ones((1,), dtype=np.int32)}
        >>> dataset = ds.NumpySlicesDataset(data=data)
        >>>
        >>> def func_net(x, y):
        ...     out = x + y
        ...     return out
        >>>
        >>> sink_process = ms.data_sink(func_net, dataset, sink_size=1)
        >>> for _ in range(2):
        ...     out = sink_process()
        ...     print(out)
        2
        2
    """

    Validator.check_value_type("sink_size", sink_size, int, "Data sink")
    if sink_size <= 0:
        raise ValueError(
            f"The 'sink_size' must be positive, but got sink_size {sink_size}.")

    if context.get_context('device_target') not in ('Ascend', 'GPU'):
        raise ValueError(
            f"Data sinking supports ascend or gpu device target, "
            f"but device target is {context.get_context('device_target')}.")

    loop = sink_size
    create_info = True
    if jit_config is None:
        create_info = (loop == 1)
        loop = 1
    ori_next_op, is_info_queue = _init_sink_dataset(dataset, loop, input_signature, create_info)

    @wraps(fn)
    def sink_process(*args, **kwargs):
        next_op, key_info = _get_next_op(dataset, ori_next_op, is_info_queue)

        def sink_fun():
            data = next_op()
            out = fn(*data)
            return out

        real_sink_fun = _get_sink_fun(sink_fun, key_info, is_info_queue, dataset, jit_config)

        loop = sink_size
        if jit_config is not None and context.get_context('mode') == context.GRAPH_MODE:
            loop = 1

        out = None
        for _ in range(loop):
            out = real_sink_fun()

        return out

    return sink_process
