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
from mindspore._checkparam import check_bool
from mindspore.parallel._utils import _get_device_num, _get_parallel_mode
from mindspore.train.dataset_helper import _send_data
from mindspore.train._utils import _exec_datagraph, _get_types_and_shapes, \
    _to_full_shapes
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
        iter_first_order (int): The iteration of first-order subgraph.
            Default: 1.

    Examples:
        >>> dataset_helper = DatasetHelper(dataset)
        >>> for inputs in dataset_helper:
        >>>     outputs = network(*inputs)
    """

    def __init__(self, dataset, dataset_sink_mode=True, iter_first_order=0):
        check_bool(dataset_sink_mode)
        self.iter = _DatasetIterMSLoopSink(dataset, iter_first_order)

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
            dataset.__TRANSFER_DATASET__ = _exec_datagraph(dataset, self.loop_size)
            dataset.__ME_INITED__ = dataset.__TRANSFER_DATASET__.queue_name

            if not hasattr(dataset, '__no_send__'):
                _send_data(dataset)
        else:
            _send_data(dataset)

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
            if dataset.get_dataset_size() % loop_size != 0:
                raise ValueError(f'Dataset size {dataset.get_dataset_size()} and '
                                 f'loop_size {loop_size} are not matched.')
            loop_count = int(dataset.get_dataset_size() / loop_size)
        return loop_count


class _DatasetIterMSLoopSink(_DatasetIter):
    """Iter for context (device_target=Ascend)"""

    def __init__(self, dataset, iter_first_order):
        super(_DatasetIterMSLoopSink, self).__init__(dataset)
        loop_size = dataset.__loop_size__ + iter_first_order
        self.loop_count = int(dataset.get_dataset_size() / loop_size * 2)
        # for self._parallel_mode equal to semi_auto_parallel or auto_parallel, use a complete tensor to
        # compile, and slice tensor to run. The batch dimension of tensors for compile is device_number
        # times the batch dimension of tensors for run. Now only support LoopSink.
        if _get_parallel_mode() in (ParallelMode.SEMI_AUTO_PARALLEL, ParallelMode.AUTO_PARALLEL):
            device_num = _get_device_num()
            self.dataset_shapes = _to_full_shapes(self.dataset_shapes, device_num)

        def op():
            return tuple()

        self.op = op
