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
from mindspore import context
from mindspore._checkparam import check_bool
from mindspore.nn.wrap import GetNextSingleOp
from mindspore.parallel._utils import _get_device_num, _get_global_rank, _get_parallel_mode
from mindspore.train._utils import _exec_datagraph, _get_types_and_shapes, _to_tensor, \
    _construct_tensor_list, _to_full_shapes, _to_full_tensor
from mindspore.train.parallel_utils import ParallelMode


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

    def __init__(self, dataset, first_order_iter=0, dataset_sink_mode=True):
        check_bool(dataset_sink_mode)

        iterclass = _DatasetIterGE
        if not dataset_sink_mode:
            iterclass = _DatasetIterFeed
        elif not context.get_context("enable_ge"):
            if context.get_context("enable_loop_sink"):
                iterclass = _DatasetIterMSLoopSink
            else:
                iterclass = _DatasetIterMS

        self.iter = iterclass(dataset, first_order_iter)

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
        # for self._parallel_mode equal to semi_auto_parallel or auto_parallel, use a complete tensor to
        # compile, and slice tensor to run. The batch dimension of tensors for compile is device_number
        # times the batch dimension of tensors for run
        if _get_parallel_mode() in (ParallelMode.SEMI_AUTO_PARALLEL, ParallelMode.AUTO_PARALLEL):
            device_num = _get_device_num()
            self.dataset_shapes = _to_full_shapes(dataset_shapes, device_num)

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
            loop_count = int(dataset.get_dataset_size() / loop_size)
        return loop_count


class _DatasetIterMSLoopSink(_DatasetIter):
    """Iter for context (enable_loop_sink=True)"""

    def __init__(self, dataset, first_order_iter):
        super(_DatasetIterMSLoopSink, self).__init__(dataset)
        # self.loop_count = self.get_loop_count(dataset)
        loop_size = dataset.__loop_size__ + first_order_iter
        self.loop_count = int(dataset.get_dataset_size() / loop_size) * 2

        def op():
            return tuple()

        self.op = op


class _DatasetIterMS(_DatasetIter):
    """Iter for context (enable_loop_sink=False)"""

    def __init__(self, dataset, first_order_order):
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
        parallel_mode = _get_parallel_mode()
        self.need_to_full = parallel_mode in (ParallelMode.SEMI_AUTO_PARALLEL, ParallelMode.AUTO_PARALLEL)
        batch_expand_num = 1
        if self.need_to_full:
            batch_expand_num = _get_device_num()
        tensor_list_run = _construct_tensor_list(self.dataset_types, self.dataset_shapes, batch_expand_num)

        def op():
            return tensor_list_run

        self.op = op


class _DatasetIterFeed:
    """Iter for feed data"""

    def __init__(self, dataset, first_order_order):
        self.dataset = dataset
        self.device_num = _get_device_num()
        self.global_rank = _get_global_rank()
        self.repeat_count = dataset.get_repeat_count()
        self.repeat_ind = 0
        self.loop_count = dataset.get_dataset_size()
        self.ind = 0

        parallel_mode = context.get_auto_parallel_context("parallel_mode")
        self.need_to_full = parallel_mode in (ParallelMode.SEMI_AUTO_PARALLEL, ParallelMode.AUTO_PARALLEL)

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
        if self.need_to_full:
            return _to_full_tensor(data, self.device_num, self.global_rank)
        return _to_tensor(data)
