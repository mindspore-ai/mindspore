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
"""Dataset help for minddata dataset"""
from __future__ import absolute_import

import math

from mindspore import _checkparam as Validator
from mindspore import log as logger
from mindspore.common._auto_dynamic import is_auto_dynamic, convert_new_shapes
from mindspore.common.dtype import pytype_to_dtype
from mindspore.common.api import _cell_graph_executor
from mindspore.common._utils import is_shape_unknown
from mindspore.dataset.engine import offload
from mindspore import context, nn
from mindspore.train._utils import _exec_datagraph, _get_types_and_shapes, _construct_tensor_list
from mindspore.parallel._utils import _get_device_num, _get_global_rank, _need_to_full, \
    _to_full_shapes, _get_pipeline_stages
from mindspore.parallel._ps_context import _is_role_sched
from mindspore.ops import operations as P
from mindspore.common.auto_dynamic_shape import _auto_dynamic_shape


def _send_data(dataset, epoch_num):
    """Engine dataset to write data to tdt queue."""
    if not hasattr(dataset, '__has_sent__'):
        exec_dataset = dataset.__transfer_dataset__
        exec_dataset.send(epoch_num)
        dataset.__has_sent__ = True


def _send_data_no_flag(dataset, epoch_num):
    """Engine dataset to write data to tdt queue directly."""
    exec_dataset = dataset.__transfer_dataset__
    exec_dataset.send(epoch_num)


def _dynamic_sink_data(dataset, dataset_iter):
    """Special scenario for dataset with sink_size=1."""
    if hasattr(dataset_iter, "sink_size") and \
       dataset_iter.sink_size == 1 and \
       dataset.get_dataset_size() != 1 and \
       hasattr(dataset_iter, "sink_count") and \
       dataset_iter.sink_count == 1:
        return True
    return False


def _dynamic_sink_exception_scenario(dataset_iter, is_dynamic):
    """The exception scenario for dynamic data is not applicable."""
    if context.get_context("mode") != context.GRAPH_MODE or is_dynamic:
        return True
    return False


def _dynamic_sink_scenario(dataset, dataset_iter, is_dynamic):
    """Special scenario with dynamic shape and sink_size=1."""
    flag = False

    # This is used only for test
    if is_auto_dynamic():
        return False

    if _dynamic_sink_data(dataset, dataset_iter) and not _dynamic_sink_exception_scenario(dataset_iter, is_dynamic):
        flag = True

    return flag


class _DataWrapper(nn.Cell):
    """
    Wraps the input network with a dataset which automatically fetches data with 'GetNext' function from the
    dataset channel 'queue_name' and performs the forward computation.
    """

    def __init__(self, network, dataset_types, dataset_shapes, queue_name):
        super(_DataWrapper, self).__init__(
            auto_prefix=False, flags=network.get_flags())
        # Also copy the flag in `network` construct
        flags = getattr(network.__class__.construct, "_func_graph_flags", {})
        self.info = (dataset_types, dataset_shapes)
        self.add_flags(**flags)
        self.get_next = P.GetNext(
            dataset_types, dataset_shapes, len(dataset_types), queue_name)
        self.network = network
        self._get_attr_from_cell(network)

    def construct(self):
        outputs = self.get_next()
        return self.network(*outputs)


def _generate_dataset_sink_mode_net(network, dataset_shapes, dataset_types, queue_name):
    if not isinstance(network, _DataWrapper):
        network = _DataWrapper(
            network, dataset_types, dataset_shapes, queue_name)
    return network


def _has_dynamic_shape(dataset_shapes):
    for shape in dataset_shapes:
        if is_shape_unknown(shape):
            return True
    return False


def _generate_network_with_dataset(network, dataset_helper, queue_name):
    """
    Generate new network with network and dataset info.
    """
    dataset_types, dataset_shapes = dataset_helper.types_shapes()

    # This is used only for test
    if is_auto_dynamic():
        new_shapes = convert_new_shapes(dataset_shapes)
        return _generate_dataset_sink_mode_net(network, new_shapes, dataset_types, queue_name)

    if network.get_inputs() and None not in network.get_inputs():
        _check_inputs(network.get_inputs(), dataset_shapes, dataset_types)
    elif context.get_context("mode") == context.PYNATIVE_MODE:
        dataset_shapes = tuple([(-2,)] * len(dataset_shapes))
    network = _generate_dataset_sink_mode_net(
        network, dataset_shapes, dataset_types, queue_name)
    return network


def _check_inputs(network_shapes, dataset_shapes, dataset_types):
    """
    Check if set inputs are correct.
    """
    for tensor_index, ele_dataset_shape in enumerate(dataset_shapes):
        if network_shapes[tensor_index] is None:
            continue
        set_inputs_shape = list(network_shapes[tensor_index].shape)
        inputs_shape = list(ele_dataset_shape)
        if dataset_types[tensor_index] != network_shapes[tensor_index].dtype:
            raise TypeError(
                f"The {tensor_index+1}th input type of 'set_inputs' must be the same as network's input, "
                f"but got 'set_inputs': {network_shapes[tensor_index].dtype} and network's "
                f"input: {dataset_types[tensor_index]}."
            )
        if len(inputs_shape) != len(set_inputs_shape):
            raise ValueError(
                f"The {tensor_index + 1}th input dims of 'set_inputs' must be the same as network's input, "
                f"but got 'set_inputs': {len(set_inputs_shape)} and network's input: {len(inputs_shape)}.")
        for index, ele_shape in enumerate(ele_dataset_shape):
            if network_shapes[tensor_index].shape[index] != -1:
                if set_inputs_shape[index] != ele_shape:
                    raise ValueError(
                        f"The {index + 1}th input shape of 'set_inputs' must be the same as network's input, "
                        f"but got 'set_inputs': {set_inputs_shape[index]} and network's input: "
                        f"{dataset_shapes[tensor_index][index]}.")
            else:
                dataset_shapes[tensor_index][index] = -1


class _DatasetAux:
    @staticmethod
    def __deepcopy__(memodict):
        return


def _get_dataset_aux(dataset):
    if not hasattr(dataset, '__network_aux__'):
        dataset.__network_aux__ = _DatasetAux()
    return dataset.__network_aux__


def connect_network_with_dataset(network, dataset_helper):
    """
    Connect the `network` with dataset in `dataset_helper`. Only supported in `sink mode
    <https://mindspore.cn/tutorials/experts/en/master/optimize/execution_opt.html>`_, (dataset_sink_mode=True).

    Args:
        network (Cell): The training network for dataset.
        dataset_helper (DatasetHelper): A class to process the MindData dataset, it provides the type, shape and queue
            name of the dataset.

    Returns:
        Cell, a new network containing the type, shape and queue name of the dataset info.

    Raises:
        RuntimeError: If the API was not called in dataset sink mode.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> import numpy as np
        >>> import mindspore as ms
        >>> from mindspore import nn
        >>> from mindspore import dataset as ds
        >>>
        >>> data = {"x": np.float32(np.random.rand(64, 10)), "y": np.random.randint(0, 5, (64,))}
        >>> train_dataset = ds.NumpySlicesDataset(data=data).batch(32)
        >>> dataset_helper = ms.DatasetHelper(train_dataset, dataset_sink_mode=True)
        >>> net = nn.Dense(10, 5)
        >>> net_with_dataset = ms.connect_network_with_dataset(net, dataset_helper)
    """
    dataset_iter = dataset_helper.iter
    dataset = dataset_iter.dataset
    aux = _get_dataset_aux(dataset)

    if isinstance(dataset_iter, _DatasetIterNormal):
        raise RuntimeError(
            "The API 'connect_network_with_dataset' should be called in dataset sink mode.")

    if _is_role_sched():
        network.add_flags(sink_mode=True)
        return network

    if not hasattr(aux, '__network__'):
        aux.__network__ = network

    if aux.__network__ is not network:
        raise ValueError(
            "The dataset has been connected to other network, please check the code.")
    is_dynamic = bool(network.get_inputs())
    queue_name = dataset.__transfer_dataset__.queue_name
    if _dynamic_sink_scenario(dataset, dataset_iter, is_dynamic):
        dataset_types, dataset_shapes = dataset_helper.get_data_info()
        dataset_types = [pytype_to_dtype(x) for x in dataset_types]
        if not is_dynamic:
            dataset_shapes = _auto_dynamic_shape.auto_dynamic_generate_compile_args(dataset_shapes, True)
        key = str(dataset_types) + str(dataset_shapes)
        _auto_dynamic_shape.update_phase_and_compile_args(dataset_shapes, key, True, aux)
        if hasattr(aux, '__network_manage__') and key in aux.__network_manage__:
            network = aux.__network_manage__[key]
        else:
            if _need_to_full():
                device_num = _get_device_num() // _get_pipeline_stages()
                dataset_shapes = _to_full_shapes(dataset_shapes, device_num)

            network = _generate_dataset_sink_mode_net(
                network, dataset_shapes, dataset_types, queue_name)
            if hasattr(aux, '__network_manage__'):
                aux.__network_manage__ = aux.__network_manage__
            else:
                aux.__network_manage__ = dict()
            aux.__network_manage__[key] = network
        network.add_flags(sink_mode=True)
        return network

    if hasattr(aux, '__sink_network__'):
        network = aux.__sink_network__
    else:
        if context.get_context("device_target") in ("Ascend", "GPU"):
            network = offload.check_add_offload_sink_mode(
                dataset, dataset_helper, network)
            network = _generate_network_with_dataset(
                network, dataset_helper, queue_name)
            aux.__sink_network__ = network

    if _dynamic_sink_data(dataset, dataset_iter) and _dynamic_sink_exception_scenario(dataset_iter, is_dynamic):
        dataset_helper.get_data_info()
    network.add_flags(sink_mode=True)
    return network


class DatasetHelper:
    """
    DatasetHelper is a class to process the MindData dataset and provides the information of dataset.

    According to different contexts, change the iterations of dataset and use the same iteration for loop in different
    contexts.

    Note:
        The iteration of DatasetHelper will provide one epoch data.

    Args:
        dataset (Dataset): The dataset iterator. The dataset can be generated by dataset generator API in
                           :class:`mindspore.dataset`, such as :class:`mindspore.dataset.ImageFolderDataset`.
        dataset_sink_mode (bool): If the value is True, GetNext is employed to fetch the data at device through the
                                  dataset pipeline, otherwise fetch the data at host by iterating through the dataset.
                                  Default: ``True``.
        sink_size (int): Control the amount of data in each sink.
                          If sink_size=-1, sink the complete dataset for each epoch.
                          If sink_size>0, sink sink_size data for each epoch.
                          Default: -1.
        epoch_num (int): The number of passes of the entire dataset to be sent. Default: 1.

    Examples:
        >>> import numpy as np
        >>> import mindspore as ms
        >>> from mindspore import nn
        >>> from mindspore import dataset as ds
        >>>
        >>> data = {"x": np.float32(np.random.rand(64, 10)), "y": np.random.randint(0, 5, (64,))}
        >>> train_dataset = ds.NumpySlicesDataset(data=data).batch(32)
        >>> set_helper = ms.DatasetHelper(train_dataset, dataset_sink_mode=False)
        >>>
        >>> net = nn.Dense(10, 5)
        >>> # Object of DatasetHelper is iterable
        >>> for next_element in set_helper:
        ...     # `next_element` includes data and label, using data to run the net
        ...     data = next_element[0]
        ...     result = net(data)
    """

    def __init__(self, dataset, dataset_sink_mode=True, sink_size=-1, epoch_num=1):
        dataset_sink_mode = Validator.check_bool(dataset_sink_mode)
        Validator.check_is_int(sink_size)
        if sink_size < -1 or sink_size == 0:
            raise ValueError(
                "The 'sink_size' must be -1 or positive, but got sink_size {}.".format(sink_size))
        if sink_size == -1:
            sink_size = dataset.get_dataset_size()

        if dataset_sink_mode:
            if context.get_context("mode") == context.GRAPH_MODE:
                if _is_role_sched():
                    iterclass = _DatasetIterPSServer
                elif (context.get_context("device_target") == "Ascend") or \
                        (context.get_context("device_target") == "GPU"):
                    iterclass = _DatasetIterMSLoopSink
                else:
                    target = context.get_context("device_target")
                    raise RuntimeError("Currently dataset sink mode is not supported when the device "
                                       "target is {}, please set dataset_sink_mode to False "
                                       "in Model.train()".format(target))
            else:
                iterclass = _DatasetIterPyNative
            self.iter = iterclass(dataset, sink_size, epoch_num)
        else:
            iterclass = _DatasetIterNormal
            self.iter = iterclass(dataset, epoch_num=epoch_num)

    def __iter__(self):
        return self.iter.__iter__()

    # A temp solution for loop sink. Delete later
    def types_shapes(self):
        """
        Get the types and shapes from dataset on the current configuration.

        Examples:
            >>> import mindspore as ms
            >>> import numpy as np
            >>>
            >>> # Define a dataset pipeline
            >>> def generator():
            ...    for i in range(5):
            ...        yield (np.ones((32, 10)),)
            >>>
            >>> train_dataset = ms.dataset.GeneratorDataset(generator, ["data"])
            >>> dataset_helper = ms.DatasetHelper(train_dataset, dataset_sink_mode=True)
            >>>
            >>> types, shapes = dataset_helper.types_shapes()
        """
        return self.iter.types_shapes()

    def sink_size(self):
        """
        Get sink_size for each iteration.

        Examples:
            >>> import mindspore as ms
            >>> import numpy as np
            >>>
            >>> # Define a dataset pipeline
            >>> def generator():
            ...    for i in range(5):
            ...        yield (np.ones((32, 10)),)
            >>>
            >>> train_dataset = ms.dataset.GeneratorDataset(generator, ["data"])
            >>> dataset_helper = ms.DatasetHelper(train_dataset, dataset_sink_mode=True, sink_size=-1)
            >>>
            >>> # if sink_size==-1, then will return the full size of source dataset.
            >>> sink_size = dataset_helper.sink_size()
        """
        return self.iter.get_sink_size()

    def stop_send(self):
        """
        Stop send data about data sink.

        Examples:
            >>> import mindspore as ms
            >>> import numpy as np
            >>> # Define a dataset pipeline
            >>> def generator():
            ...    for i in range(5):
            ...        yield (np.ones((32, 10)),)
            >>> train_dataset = ms.dataset.GeneratorDataset(generator, ["data"])
            >>> dataset_helper = ms.DatasetHelper(train_dataset, dataset_sink_mode=True, sink_size=-1)
            >>> dataset_helper.stop_send()
        """
        self.iter.stop_send()

    def release(self):
        """
        Free up resources about data sink.

        Examples:
            >>> import numpy as np
            >>> import mindspore as ms
            >>> from mindspore import nn
            >>> from mindspore import dataset as ds
            >>>
            >>> data = {"x": np.float32(np.random.rand(64, 10)), "y": np.random.randint(0, 5, (64,))}
            >>> train_dataset = ds.NumpySlicesDataset(data=data).batch(32)
            >>> dataset_helper = ms.DatasetHelper(train_dataset, dataset_sink_mode=True)
            >>> dataset_helper.release()
        """
        self.iter.release()

    def continue_send(self):
        """
        Continue to send data to device at the beginning of epoch.

        Examples:
            >>> import numpy as np
            >>> import mindspore as ms
            >>> from mindspore import nn
            >>> from mindspore import dataset as ds
            >>>
            >>> data = {"x": np.float32(np.random.rand(64, 10)), "y": np.random.randint(0, 5, (64,))}
            >>> train_dataset = ds.NumpySlicesDataset(data=data).batch(32)
            >>> dataset_helper = ms.DatasetHelper(train_dataset, dataset_sink_mode=True)
            >>> dataset_helper.continue_send()
        """
        self.iter.continue_send()

    def _reset(self, step, dataset_size):
        """Reset the dataset to the provided step and epoch."""
        self.iter._reset(step, dataset_size)  # pylint: disable=protected-access

    # pylint: disable=missing-docstring
    def get_data_info(self):
        # In sink mode, it returns the types and shapes of the current data.
        # Generally, it works in dynamic shape scenarios.
        return self.iter.get_data_info()

    # pylint: disable=missing-docstring
    def get_send_info(self, run_context):
        # In sink mode, it returns the send information of dataset at this moment.
        # Send information includes number of send batches, time summary of fetching data on host
        # and time summary of sending data.
        class InfoViewer:
            '''
            Inner class for parsing send info.
            '''
            def __init__(self, send_info, run_context):
                self.info_ = {}
                self.sink_size = run_context.original_args()["batch_num"]
                if run_context.original_args().get("train_dataset", None) is not None:
                    self.dataset_size = run_context.original_args()["train_dataset"].get_dataset_size()
                elif run_context.original_args().get("valid_dataset", None) is not None:
                    self.dataset_size = run_context.original_args()["valid_dataset"].get_dataset_size()
                else:
                    raise RuntimeError("Could not find a proper dataset to estimate dataset size.")
                if not send_info:
                    epoch = 1
                    self.info_[epoch] = {'fetch_data_num': 0, 'fetch_data_time': 0, 'first_data_time': 0}
                else:
                    for info_per_epoch in send_info:
                        epoch, fetch_data_num, first_data_time, fetch_data_time = info_per_epoch
                        if fetch_data_num > 1:
                            fetch_data_time = (fetch_data_time - first_data_time) / (fetch_data_num - 1) * 1000.
                        self.info_[epoch] = {'fetch_data_num': fetch_data_num,
                                             'fetch_data_time': fetch_data_time,
                                             'first_data_time': first_data_time}

            def epoch(self, epoch):
                if self.sink_size == self.dataset_size:
                    return self.info_[epoch]
                global_step = epoch * self.sink_size
                data_epoch = math.ceil(global_step / self.dataset_size)
                return self.info_[data_epoch]

        # send info struct:[epoch, data_num_per_epoch, first_data_time, accumulate_data_time]
        # for example [1, 1875, 0.421, 0.362]
        send_info = self.iter.get_send_info()
        return InfoViewer(send_info, run_context)


class _DatasetIter:
    """Base iter for dataset helper"""

    def __init__(self, dataset, sink_size, epoch_num):
        self.dataset = dataset
        self.sink_size = sink_size
        self.sink_count = self.get_sink_count(dataset)
        self.dataset_types, self.dataset_shapes = _get_types_and_shapes(
            dataset)

        if dataset.get_init_step() % sink_size != 0:
            init_epoch = dataset.get_init_step() // sink_size
            init_step = init_epoch * sink_size
            logger.warning("Init global step must be the end of the epoch in sink mode, "
                           "but got: {0}. Reset it to the end of epoch {1} at step {2}."
                           .format(dataset.get_init_step(), init_epoch, init_step))
            dataset.set_init_step(init_step)

        if not hasattr(dataset, '__transfer_dataset__'):
            if hasattr(dataset, '__loop_size__'):
                self.sink_size = dataset.__loop_size__
            create_data_info_queue = (
                sink_size == 1 and self.sink_count == 1 and dataset.get_dataset_size() != 1)
            dataset.__transfer_dataset__ = _exec_datagraph(dataset, self.sink_size,
                                                           create_data_info_queue=create_data_info_queue)

            if not hasattr(dataset, '__no_send__'):
                _send_data(dataset, epoch_num)
        else:
            # if using an existed __transfer_dataset__, set the queue_name directly
            if not dataset.__transfer_dataset__.queue_name:
                _cell_graph_executor.set_queue_name(
                    dataset.__transfer_dataset__.queue_name)
            _send_data_no_flag(dataset, epoch_num)

        self.stop_send = dataset.__transfer_dataset__.stop_send
        self.release = dataset.__transfer_dataset__.release
        self.continue_send = dataset.__transfer_dataset__.continue_send
        self.get_data_info = dataset.__transfer_dataset__.get_data_info
        self.get_send_info = dataset.__transfer_dataset__.get_send_info
        if hasattr(dataset.__transfer_dataset__, "_reset"):
            self._reset = dataset.__transfer_dataset__._reset  # pylint: disable=protected-access

    def __iter__(self):
        self.index = 0
        return self

    def __next__(self):
        if self.index >= self.sink_count:
            raise StopIteration()
        self.index += 1
        return self.op()

    def types_shapes(self):
        """
        Return the types and shapes of the dataset. The type and shape of each data in the dataset
        should be consistent.
        """
        return self.dataset_types, self.dataset_shapes

    def get_sink_count(self, dataset):
        sink_count = 1
        if hasattr(dataset, '__loop_size__'):
            loop_size = dataset.__loop_size__
            if loop_size <= dataset.get_dataset_size() and dataset.get_dataset_size() % loop_size != 0:
                raise ValueError(f"Dataset size {dataset.get_dataset_size()} and 'sink_size' {loop_size} "
                                 f"are not matched, dataset size should be divisible by 'sink_size'.")
            sink_count = math.ceil(dataset.get_dataset_size() / loop_size)
        return sink_count

    def get_sink_size(self):
        """get sink_size to device"""
        sink_size = 1
        if hasattr(self.dataset, '__loop_size__'):
            sink_size = self.dataset.__loop_size__
        else:
            if context.get_context("device_target") == "Ascend" or context.get_context("device_target") == "GPU":
                if self.sink_size > 0:
                    sink_size = self.sink_size
                else:
                    sink_size = self.dataset.get_dataset_size()
        return sink_size


class _DatasetIterPyNative(_DatasetIter):
    """Iter for context (mode=PYNATIVE_MODE)."""

    def __init__(self, dataset, sink_size, epoch_num):
        super().__init__(dataset, sink_size, epoch_num)
        if sink_size > 0:
            self.sink_count = sink_size
        else:
            self.sink_count = dataset.get_dataset_size()

        def op():
            return tuple()

        self.op = op


class _DatasetIterMSLoopSink(_DatasetIter):
    """Iter for context (device_target=Ascend)"""

    def __init__(self, dataset, sink_size, epoch_num):
        super().__init__(dataset, sink_size, epoch_num)
        self.sink_count = self.get_sink_count(dataset)
        # for self._parallel_mode equal to semi_auto_parallel or auto_parallel, and not using full_batch,
        # use a complete tensor to compile, and slice tensor to run. The batch dimension of tensors for
        # compile is device_number times the batch dimension of tensors for run. Now only support LoopSink.
        if _need_to_full():
            device_num = _get_device_num() // _get_pipeline_stages()
            self.dataset_shapes = _to_full_shapes(
                self.dataset_shapes, device_num)

        def op():
            return tuple()

        self.op = op


class _DatasetIterPSServer(_DatasetIter):
    """Iter for context on MS_PSERVER or MS_SCHED"""

    def __init__(self, dataset, sink_size, epoch_num):
        super().__init__(dataset, sink_size, epoch_num)
        self.sink_count = 1
        self.sink_size = 1
        self.op = None

        def op():
            return _construct_tensor_list(self.dataset_types, self.dataset_shapes, batch_expand_num=1)

        self.op = op


class _DatasetIterNormal:
    """Iter for normal(non sink) mode, feed the data from host."""

    def __init__(self, dataset, epoch_num=-1):
        self.dataset = dataset
        self.device_num = _get_device_num()
        self.global_rank = _get_global_rank()
        self.iter = self.dataset.create_tuple_iterator(
            num_epochs=epoch_num, do_copy=True)

    def __iter__(self):
        return self

    def __next__(self):
        data = self.iter.__next__()
        return data


__all__ = ["DatasetHelper", "connect_network_with_dataset"]
