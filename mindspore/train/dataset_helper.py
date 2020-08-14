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
import math
import os

from mindspore._checkparam import check_bool, check_int
from .. import context
from ._utils import _exec_datagraph, _get_types_and_shapes, _to_tensor, \
    _construct_tensor_list
from ..nn.wrap import GetNextSingleOp
from ..parallel._utils import _get_device_num, _get_global_rank, _need_to_full, _to_full_shapes


def _send_data(dataset, epoch_num):
    """Engine dataset to write data to tdt queue."""
    if not hasattr(dataset, '__has_sent__'):
        exec_dataset = dataset.__TRANSFER_DATASET__
        exec_dataset.send(epoch_num)
        dataset.__has_sent__ = True

def _send_data_no_flag(dataset, epoch_num):
    """Engine dataset to write data to tdt queue directly."""
    exec_dataset = dataset.__TRANSFER_DATASET__
    exec_dataset.send(epoch_num)


class DatasetHelper:
    """
    Help function to use the MindData dataset.

    According to different contexts, change the iterations of dataset and use the same iteration for loop in different
    contexts.

    Note:
        The iteration of DatasetHelper will provide one epoch data.

    Args:
        dataset (DataSet): The training dataset iterator.
        dataset_sink_mode (bool): If true use GetNext to fetch the data, or else feed the data from host. Default: True.
        sink_size (int): Control the amount of data in each sink.
                             If sink_size=-1, sink the complete dataset for each epoch.
                             If sink_size>0, sink sink_size data for each epoch. Default: -1.
        epoch_num (int): Control the number of epoch data to send. Default: 1.

    Examples:
        >>> dataset_helper = DatasetHelper(dataset)
        >>> for inputs in dataset_helper:
        >>>     outputs = network(*inputs)
    """

    def __init__(self, dataset, dataset_sink_mode=True, sink_size=-1, epoch_num=1):
        check_bool(dataset_sink_mode)
        check_int(sink_size)
        if sink_size < -1 or sink_size == 0:
            raise ValueError("The sink_size must be -1 or positive, but got sink_size {}.".format(sink_size))

        if dataset_sink_mode:
            if context.get_context("enable_ge"):
                iterclass = _DatasetIterGE
            else:
                if context.get_context("device_target") == "Ascend":
                    iterclass = _DatasetIterMSLoopSink
                elif context.get_context("device_target") == "GPU":
                    ms_role = os.getenv("MS_ROLE")
                    if ms_role in ("MS_PSERVER", "MS_SCHED"):
                        iterclass = _DatasetIterPSLite
                    else:
                        iterclass = _DatasetIterMS
                elif context.get_context("device_target") == "CPU":
                    raise RuntimeError("Currently dataset sink mode is not supported when the device target is CPU.")
            self.iter = iterclass(dataset, sink_size, epoch_num)
        else:
            iterclass = _DatasetIterNormal
            self.iter = iterclass(dataset)

    def __iter__(self):
        return self.iter.__iter__()

    # A temp solution for loop sink. Delete later
    def types_shapes(self):
        """Get the types and shapes from dataset on the current configuration."""
        return self.iter.types_shapes()

    def sink_size(self):
        """Get sink_size for each iteration."""
        return self.iter.get_sink_size()

    def stop_send(self):
        """Free up resources about data sink."""
        self.iter.stop_send()


class _DatasetIter:
    """Base iter for dataset helper"""
    def __init__(self, dataset, sink_size, epoch_num):
        self.dataset = dataset
        self.sink_size = sink_size
        self.sink_count = 1

        if not hasattr(dataset, '__TRANSFER_DATASET__'):
            if hasattr(dataset, '__loop_size__'):
                self.sink_size = dataset.__loop_size__
            dataset.__TRANSFER_DATASET__ = _exec_datagraph(dataset, self.sink_size)
            dataset.__ME_INITED__ = dataset.__TRANSFER_DATASET__.queue_name

            if not hasattr(dataset, '__no_send__'):
                _send_data(dataset, epoch_num)
        else:
            _send_data_no_flag(dataset, epoch_num)

        self.stop_send = dataset.__TRANSFER_DATASET__.stop_send
        self.dataset_types, self.dataset_shapes = _get_types_and_shapes(dataset)

    def __iter__(self):
        self.index = 0
        return self

    def __next__(self):
        if self.index >= self.sink_count:
            raise StopIteration()
        self.index += 1
        return self.op()

    def types_shapes(self):
        return self.dataset_types, self.dataset_shapes

    def get_sink_count(self, dataset):
        sink_count = 1
        if hasattr(dataset, '__loop_size__'):
            loop_size = dataset.__loop_size__
            if loop_size <= dataset.get_dataset_size() and dataset.get_dataset_size() % loop_size != 0:
                raise ValueError(f'Dataset size {dataset.get_dataset_size()} and '
                                 f'sink_size {loop_size} are not matched.')
            sink_count = math.ceil(dataset.get_dataset_size() / loop_size)
        return sink_count

    def get_sink_size(self):
        """get sink_size to device"""
        sink_size = 1
        if hasattr(self.dataset, '__loop_size__'):
            sink_size = self.dataset.__loop_size__
        else:
            if context.get_context("enable_ge") or context.get_context("device_target") == "Ascend":
                if self.sink_size > 0:
                    sink_size = self.sink_size
                else:
                    sink_size = self.dataset.get_dataset_size()
        return sink_size


class _DatasetIterGE(_DatasetIter):
    """Iter for GE."""
    def __init__(self, dataset, sink_size, epoch_num):
        super().__init__(dataset, sink_size, epoch_num)
        self.sink_count = self.get_sink_count(dataset)
        batch_expand_num = 1
        if _need_to_full():
            batch_expand_num = _get_device_num()
        tensor_list_run = _construct_tensor_list(self.dataset_types, self.dataset_shapes, batch_expand_num)

        def op():
            return tensor_list_run

        self.op = op


class _DatasetIterMSLoopSink(_DatasetIter):
    """Iter for context (device_target=Ascend)"""
    def __init__(self, dataset, sink_size, epoch_num):
        super().__init__(dataset, sink_size, epoch_num)
        self.sink_count = self.get_sink_count(dataset)
        ms_role = os.getenv("MS_ROLE")
        if ms_role in ("MS_PSERVER", "MS_SCHED"):
            self.sink_count = 1
        # for self._parallel_mode equal to semi_auto_parallel or auto_parallel, and not using full_batch,
        # use a complete tensor to compile, and slice tensor to run. The batch dimension of tensors for
        # compile is device_number times the batch dimension of tensors for run. Now only support LoopSink.
        if _need_to_full():
            device_num = _get_device_num()
            self.dataset_shapes = _to_full_shapes(self.dataset_shapes, device_num)

        def op():
            return tuple()

        self.op = op


class _DatasetIterMS(_DatasetIter):
    """Iter for MS(enable_loop_sink=False)."""
    def __init__(self, dataset, sink_size, epoch_num):
        super().__init__(dataset, sink_size, epoch_num)
        if sink_size > 0:
            self.sink_count = sink_size
        else:
            self.sink_count = dataset.get_dataset_size()

        queue_name = dataset.__ME_INITED__
        self.op = GetNextSingleOp(self.dataset_types, self.dataset_shapes, queue_name)


class _DatasetIterPSLite(_DatasetIter):
    """Iter for context (device_target=GPU) on MS_PSERVER or MS_SCHED"""
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
    def __init__(self, dataset):
        self.dataset = dataset
        self.device_num = _get_device_num()
        self.global_rank = _get_global_rank()
        self.iter = self.dataset.create_tuple_iterator()

    def __iter__(self):
        return self

    def __next__(self):
        data = self.iter.__next__()
        return _to_tensor(data)
