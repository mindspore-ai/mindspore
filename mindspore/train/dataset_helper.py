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

from mindspore._checkparam import check_bool
from .. import context
from ._utils import _exec_datagraph, _get_types_and_shapes, _to_tensor, \
    _construct_tensor_list, _to_full_shapes, _to_full_tensor
from ..nn.wrap import GetNextSingleOp
from ..parallel._utils import _get_device_num, _get_global_rank, _need_to_full


class DatasetHelper:
    """
    Help function to use the Minddata dataset.

    According to different context, change the iter of dataset, to use the same for loop in different context.

    Note:
        The iter of DatasetHelper will give one epoch data.

    Args:
        dataset (DataSet): The dataset.
        dataset_sink_mode (bool): If true use GetNext to fetch the data, or else feed the data from host.
            Default: True.

    Examples:
        >>> dataset_helper = DatasetHelper(dataset)
        >>> for inputs in dataset_helper:
        >>>     outputs = network(*inputs)
    """
    def __init__(self, dataset, dataset_sink_mode=True):
        check_bool(dataset_sink_mode)

        if dataset_sink_mode:
            if context.get_context("enable_ge"):
                iterclass = _DatasetIterGE
            else:
                if context.get_context("device_target") == "Ascend":
                    iterclass = _DatasetIterMSLoopSink
                elif context.get_context("device_target") == "GPU":
                    iterclass = _DatasetIterMS
                elif context.get_context("device_target") == "CPU":
                    raise RuntimeError("Currently dataset sink mode is not supported when the device target is CPU.")
        else:
            iterclass = _DatasetIterFeed
        self.iter = iterclass(dataset)

    def __iter__(self):
        return self.iter.__iter__()

    # A temp solution for loop sink. Delete later
    def types_shapes(self):
        """Get the types and shapes from dataset on current config."""
        return self.iter.types_shapes()

    def loop_size(self):
        """Get loop_size for every iteration."""
        return self.iter.loop_size


class _DatasetIter:
    """Base iter for dataset help"""
    def __init__(self, dataset):
        self.loop_size = 1
        if not hasattr(dataset, '__ME_INITED__'):
            if not hasattr(dataset, '__loop_size__'):
                self.loop_size = dataset.get_dataset_size()
            else:
                self.loop_size = dataset.__loop_size__
            dataset.__ME_INITED__ = _exec_datagraph(dataset, self.loop_size).queue_name

        self.ind = 0
        self.dataset = dataset
        dataset_types, dataset_shapes = _get_types_and_shapes(dataset)
        self.dataset_types, self.dataset_shapes = dataset_types, dataset_shapes

    def __iter__(self):
        self.ind = 0
        return self

    def __next__(self):
        if self.ind >= self.loop_count:
            raise StopIteration()
        self.ind += 1
        return self.op()

    def types_shapes(self):
        return self.dataset_types, self.dataset_shapes

    def get_loop_count(self, dataset):
        loop_count = 1
        if hasattr(dataset, '__loop_size__'):
            loop_size = dataset.__loop_size__
            if loop_size <= dataset.get_dataset_size() and dataset.get_dataset_size() % loop_size != 0:
                raise ValueError(f'Dataset size {dataset.get_dataset_size()} and '
                                 f'loop_size {loop_size} are not matched.')
            loop_count = math.ceil(dataset.get_dataset_size() / loop_size)
        return loop_count


class _DatasetIterMSLoopSink(_DatasetIter):
    """Iter for context (device_target=Ascend)"""
    def __init__(self, dataset):
        super(_DatasetIterMSLoopSink, self).__init__(dataset)
        self.loop_count = self.get_loop_count(dataset)
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
    """Iter for context (device_target=GPU)"""
    def __init__(self, dataset):
        super(_DatasetIterMS, self).__init__(dataset)
        self.loop_count = dataset.get_dataset_size()
        self.loop_size = 1
        queue_name = dataset.__ME_INITED__
        self.op = GetNextSingleOp(self.dataset_types, self.dataset_shapes, queue_name)


class _DatasetIterGE(_DatasetIter):
    """Iter for ge"""
    def __init__(self, dataset):
        super(_DatasetIterGE, self).__init__(dataset)
        self.loop_count = self.get_loop_count(dataset)
        batch_expand_num = 1
        if _need_to_full():
            batch_expand_num = _get_device_num()
        tensor_list_run = _construct_tensor_list(self.dataset_types, self.dataset_shapes, batch_expand_num)

        def op():
            return tensor_list_run

        self.op = op


class _DatasetIterFeed:
    """Iter for normal(non sink) mode, feed the data from host."""
    def __init__(self, dataset):
        self.dataset = dataset
        self.device_num = _get_device_num()
        self.global_rank = _get_global_rank()
        self.repeat_count = dataset.get_repeat_count()
        self.repeat_ind = 0
        self.loop_count = dataset.get_dataset_size()
        self.ind = 0

    def __iter__(self):
        if self.repeat_ind % self.repeat_count == 0:
            self.iter = self.dataset.__iter__()

        self.repeat_ind += 1
        self.ind = 0
        return self

    def __next__(self):
        if self.ind >= self.loop_count:
            raise StopIteration()
        self.ind += 1
        data = self.iter.__next__()
        if _need_to_full():
            return _to_full_tensor(data, self.device_num, self.global_rank)
        return _to_tensor(data)
