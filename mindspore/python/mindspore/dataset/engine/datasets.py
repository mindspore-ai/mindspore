# Copyright 2022-2024 Huawei Technologies Co., Ltd
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
# ==============================================================================
"""
1. This file is an abstraction of the dataset loading class. It contains
some basic dataset operations(skip, filter, map, batch, ...).
2. Specific dataset loading classes can be found in datasets_vision.py, datasets_text.py,
datasets_audio.py, datasets_standard_format.py and datasets_user_defined.py files.
    datasets_vision.py: contains vision dataset loading classes.
    datasets_text.py: contains text dataset loading classes.
    datasets_audio.py: contains audio dataset loading classes.
    datasets_standard_format.py: contains standard format loading classes which
                                 any other kinds of datasets can be converted to.
    datasets_user_defined.py: contains basic classes that help users to define
                             flexible ways to load dataset.
"""
import atexit
import glob
import json
import os
import signal
import stat
import warnings

import time
import uuid
import multiprocessing
from importlib import import_module
import sys
import threading
from types import GeneratorType

import copy
import weakref
import platform
import numpy as np

import mindspore._c_dataengine as cde
from mindspore._c_expression import typing

from mindspore import log as logger
from mindspore.parallel._ps_context import _is_role_sched
from mindspore.dataset.engine.offload import GetOffloadModel
from mindspore.dataset import transforms
from mindspore.dataset.text.utils import SentencePieceModel, DE_C_INTER_SENTENCEPIECE_MODE
from mindspore.dataset.debug import DebugHook

from mindspore.dataset.engine import samplers
from mindspore.dataset.engine.samplers import Shuffle
from mindspore.common import Tensor
from mindspore.common import dtype as mstype
import mindspore as ms

from .iterators import DictIterator, TupleIterator, DummyIterator, check_iterator_cleanup, _set_iterator_cleanup, \
    ITERATORS_LIST, _unset_iterator_cleanup, _cleanup_the_iterators_if_created
from .validators import check_batch, check_shuffle, check_map, check_filter, check_repeat, check_skip, check_zip, \
    check_rename, check_device_send, check_take, check_output_shape, check_project, \
    check_sync_wait, check_zip_dataset, check_add_column, check_concat, check_split, check_bucket_batch_by_length, \
    check_save, check_tuple_iterator, check_dict_iterator, check_schema, check_to_device_send, check_padded_batch, \
    check_total_batch, check_sync_update, check_send, check_recv
from ..core.config import get_callback_timeout, _init_device_info, get_num_parallel_workers, \
    get_enable_watchdog, get_seed, set_seed, get_debug_mode, get_multiprocessing_timeout_interval, \
    _get_debug_hook_list, get_multiprocessing_start_method, get_video_backend, set_video_backend, \
    get_error_samples_mode, ErrorSamplesMode
from ..core.datatypes import mstype_to_detype
from ..core.validator_helpers import replace_none
from ..core.py_util_helpers import ExceptionHandler
from ..transforms.py_transforms_util import FuncWrapper, Implementation
from ..vision.transforms import ToNumpy
from ...mindrecord.config import _get_enc_key, _get_enc_mode, encrypt

try:
    context = import_module("mindspore.context")
except ModuleNotFoundError:
    context = None

if platform.system().lower() == "darwin" and multiprocessing.get_start_method() != "fork":
    multiprocessing.set_start_method("fork", True)

OffloadToManualOffloadMode = {
    None: cde.ManualOffloadMode.UNSPECIFIED,
    False: cde.ManualOffloadMode.DISABLED,
    True: cde.ManualOffloadMode.ENABLED
}

_train_dataset = None


def _set_training_dataset(dataset):
    """
    Set the dataset to be used when training recovery has occurred.

    Args:
        dataset: the training dataset or iterator
    """
    global _train_dataset
    _train_dataset = dataset


def _get_training_dataset():
    """
    Get the dataset to be used when training recovery has occurred.

    Returns:
        training dataset/iterator
    """
    return _train_dataset


def _reset_training_dataset(global_step, dataset_size):
    """
    Reset the training dataset to the given global step.

    Args:
        global_step (int): Number of global steps that have completed training.
            Dataset will provide data from its next step after reset.
        dataset_size (int): Number of steps per epoch.
    """
    dataset = _get_training_dataset()
    if dataset is not None:
        dataset._reset(global_step, dataset_size)  # pylint: disable=protected-access
    else:
        raise RuntimeError("Training dataset is not set.")


@check_zip
def zip(datasets):
    """
    Zip the datasets in the input tuple of datasets.

    Args:
        datasets (tuple[Dataset]): A tuple of datasets to be zipped together.
            The number of datasets must be more than 1.

    Returns:
        Dataset, a new dataset with the above operation applied.

    Raises:
        ValueError: If the number of datasets is 1.
        TypeError: If datasets is not a tuple.

    Examples:
            >>> # Create a dataset which is the combination of dataset_1 and dataset_2
            >>> import mindspore.dataset as ds
            >>>
            >>> dataset_1 = ds.GeneratorDataset([1], "column1")
            >>> dataset_2 = ds.GeneratorDataset([2], "column2")
            >>> dataset = ds.zip((dataset_1, dataset_2))
    """
    if len(datasets) <= 1:
        raise ValueError(
            "Can't zip empty or just one dataset!")
    for dataset in datasets:
        if not isinstance(dataset, Dataset):
            raise TypeError(f"Invalid dataset, expected Dataset object, but got {type(dataset)}!")
    return ZipDataset(datasets)


def _get_operator_process():
    """
    Inner implemented method, mainly for passing sub-process id in C layer

    Returns:
         dict, mapping dict of operation id and corresponding process id.
    """
    process_info = _OP_PROCESS
    op_process = {}
    keys = process_info.keys()
    fetched_all = True
    for key in keys:
        try:
            op_process[key] = list(process_info[key][1])
            item_full = len(process_info[key][1]) == process_info[key][0]
        except KeyError as err:
            raise err
        fetched_all = fetched_all and item_full
    return op_process, fetched_all


def _set_dataset_permissions(file_name, num_files):
    """
    set saved dataset files' permissions to 600
    the rule of dataset filenames should be the same as those in C++.
    """
    num_digits = len(str(num_files - 1))
    if num_files == 1:
        paths = [file_name]
    else:
        paths = []
        for x in range(num_files):
            paths.append(f"{file_name}{str(x).rjust(num_digits, '0')}")

    for item in paths:
        if os.path.exists(item):
            os.chmod(item, stat.S_IRUSR | stat.S_IWUSR)
            index_file = item + ".db"
            if os.path.exists(index_file):
                os.chmod(index_file, stat.S_IRUSR | stat.S_IWUSR)


# dict used to cast mstype to int
mstype_to_int = {mstype.bool: 0,
                 mstype.int8: 1,
                 mstype.int16: 2,
                 mstype.short: 3,
                 mstype.int32: 4,
                 mstype.int: 5,
                 mstype.int64: 6,
                 mstype.long: 7,
                 mstype.uint8: 8,
                 mstype.uint16: 9,
                 mstype.uint32: 10,
                 mstype.uint64: 11,
                 mstype.float16: 12,
                 mstype.half: 13,
                 mstype.float32: 14,
                 mstype.float: 15,
                 mstype.float64: 16,
                 mstype.double: 17,
                 mstype.bfloat16: 18}


int_to_mstype = {value: key for key, value in mstype_to_int.items()}


MAX_METADATA_LENGTH = 2048


def flatten_single_lists(nested_list):
    """
    Recursively remove list nesting of length 1
    """
    if not isinstance(nested_list, list):
        return nested_list

    if len(nested_list) == 1 and isinstance(nested_list[0], list):
        return flatten_single_lists(nested_list[0])
    return [flatten_single_lists(item) for item in nested_list]


class Dataset:
    """
    Abstract class to represent a dataset in DataEngine's data pipeline.

    This class is the base class of SourceDataset and Dataset, and represents
    a node in the data flow graph.
                                     Dataset
           -----------------------------------------------------------
           |                  |                   |                  |
    VisionBaseDataset    TextBaseDataset    AudioBaseDataset         |
           -                  -                   -                  |
           |                  |                   |                  |
           ----------------------------------------                  |
                      UnionBaseDataset                               |
                                                                     |
                                                               SourceDataset
                                                                     -
                                                                     |
                                                              MappableDataset

    DatasetOperation: MapDataset(UnionBaseDataset)
                      BatchDataset(UnionBaseDataset)
                      PaddedBatchDataset(UnionBaseDataset)
                      BucketBatchByLengthDataset(UnionBaseDataset)
                      ShuffleDataset(UnionBaseDataset)
                      FilterDataset(UnionBaseDataset)
                      RepeatDataset(UnionBaseDataset)
                      SkipDataset(UnionBaseDataset)
                      TakeDataset(UnionBaseDataset)
                      ZipDataset(UnionBaseDataset)
                      ConcatDataset(UnionBaseDataset)
                      RenameDataset(UnionBaseDataset)
                      ProjectDataset(UnionBaseDataset)
                      SyncWaitDataset(UnionBaseDataset)

    Impl Dataset - vision:       ImageFolderDataset(MappableDataset, VisionBaseDataset)
                                 USPSDataset(SourceDataset, VisionBaseDataset)
    Impl Dataset - text:         TextFileDataset(SourceDataset, TextBaseDataset)
                                 YahooAnswersDataset(SourceDataset, TextBaseDataset)
    Impl Dataset - audio:        LJSpeechDataset(MappableDataset, AudioBaseDataset)
                                 TedliumDataset(MappableDataset, AudioBaseDataset)
    Impl Dataset - standard:     MindDataset(MappableDataset, UnionBaseDataset)
                                 TFRecordDataset(SourceDataset, UnionBaseDataset)
    Impl Dataset - user defined: GeneratorDataset(MappableDataset, UnionBaseDataset)
                                 NumpySlicesDataset(GeneratorDataset)

    Args:
        num_parallel_workers (int, optional): Number of workers to process the dataset in parallel.
            Default: ``None``.
    """

    def __init__(self, children=None, num_parallel_workers=None, cache=None):
        # Note: children and parent are internal variables, not recommended for external using.
        self.children = replace_none(children, [])
        if isinstance(self.children, tuple):
            self.children = list(self.children)
        if not isinstance(self.children, list):
            self.children = [self.children]

        self.parent = []
        for child in self.children:
            child.parent.append(weakref.ref(self))
        self.num_parallel_workers = num_parallel_workers
        self.cache = cache

        self._device_iter = 0
        self._input_indexs = ()
        self.saved_output_types = None
        self.saved_output_shapes = None
        self.estimated_output_shapes = None
        self.runtime_context = None
        self._col_names = None
        self.dataset_size = None
        self._batch_size = None
        self._num_classes = None
        self._repeat_count = None
        self._class_indexing = None
        self._sync = False
        self._global_step = None
        self._dataset_iter = None  # used to send & recv

    @staticmethod
    def _get_operator_id(dataset):
        """
        Internal method to iterate the tree and obtain op_id of each operation.

        Returns:
            Dataset, the root dataset of the tree.
        """
        op_name = {}
        generator_process = {}
        op_name[str(dataset)] = 0
        op_id = 1

        def process_name(datasets, operator_id):
            if not datasets:
                return 0
            temp = []
            for item in datasets:
                for d in item.children:
                    temp.append(d)
                    op_name[str(d)] = operator_id

                    from mindspore.dataset.engine.datasets_user_defined import GeneratorDataset  \
                    # pylint: disable=import-outside-toplevel
                    if isinstance(d, GeneratorDataset) and d.sample_fn and d.sample_fn.pids:
                        generator_process[operator_id] = [d.num_parallel_workers, set(d.sample_fn.pids)]

            operator_id = operator_id + 1
            return process_name(temp, operator_id)

        process_name([dataset], op_id)
        if generator_process:
            _OP_PROCESS.update(generator_process)
        return op_name

    def create_ir_tree(self, getter_mode=False):
        """
        Internal method to build an IR tree.

        Args:
            getter_mode (bool, optional): Whether to build IR tree in pull mode. Default: ``False``.

        Returns:
            Union[DatasetNode, Dataset], the root node of the IR tree and the root dataset of the IR tree.
        """
        parent = self.parent
        self.parent = []
        dataset = copy.deepcopy(self)
        dataset = self.pre_process(dataset)
        global _OP_NAME
        _OP_NAME = Dataset._get_operator_id(dataset)
        ir_tree = dataset.parse_tree(getter_mode)
        self.parent = parent
        _init_device_info()
        return ir_tree, dataset

    def pre_process(self, dataset):
        """Insert batch operation for GeneratorDataset with batch_sampler."""
        if hasattr(dataset, "has_batch_sampler") and dataset.has_batch_sampler:
            original_parent = dataset.parent
            dataset.parent = []
            dataset = dataset.batch(batch_size=-1, num_parallel_workers=dataset.num_parallel_workers,
                                    per_batch_map=dataset.collate_fn)
            dataset.parent = original_parent
        else:
            for index, _ in enumerate(dataset.children):
                dataset.children[index] = self.pre_process(dataset.children[index])
        return dataset

    def parse_tree(self, getter_mode=False):
        """
        Internal method to parse the API tree into an IR tree.

        Args:
            getter_mode (bool, optional): Whether to build IR tree in pull mode. Default: ``False``.

        Returns:
            DatasetNode, the root node of the IR tree.
        """
        if len(self.parent) > 1:
            raise ValueError("The data pipeline is not a tree (i.e., one node has 2 consumers)")
        ir_children = [d.parse_tree(getter_mode) for d in self.children]
        # Bootstrap can only be performed on a copy of the original dataset node.
        # Bootstrap on original dataset node will make all iterators share the same process pool
        self.pre_parse(getter_mode)
        self.iterator_bootstrap()
        ir_node = self.parse(ir_children)
        ir_node = self.post_parse(ir_node)
        return ir_node

    def __safe_deepcopy__(self, memodict, exclude=()):
        if id(self) in memodict:
            return memodict[id(self)]
        cls = self.__class__
        new_op = cls.__new__(cls)
        memodict[id(self)] = new_op
        for arg, value in self.__dict__.items():
            if arg in exclude:
                setattr(new_op, arg, value)
            else:
                try:
                    setattr(new_op, arg, copy.deepcopy(value, memodict))
                except TypeError:
                    setattr(new_op, arg, value)
        return new_op

    @staticmethod
    def _noop_mode():
        if _is_role_sched():
            return True
        return False

    def iterator_bootstrap(self):
        pass

    def __add__(self, datasets):
        return self.concat(datasets)

    def to_json(self, filename=""):
        """
        Serialize a pipeline into JSON string and dump into file if filename is provided.

        Args:
            filename (str, optional): filename of JSON file to be saved as. Default: ``""``.

        Returns:
            str, JSON string of the pipeline.

        Examples:
            >>> import mindspore.dataset as ds
            >>> mnist_dataset_dir = "/path/to/mnist_dataset_directory"
            >>> dataset = ds.MnistDataset(dataset_dir=mnist_dataset_dir)
            >>> dataset_json = dataset.to_json("/path/to/mnist_dataset_pipeline.json")
        """
        ir_tree, _ = self.create_ir_tree()
        return json.loads(ir_tree.to_json(filename))

    @check_bucket_batch_by_length
    def bucket_batch_by_length(self, column_names, bucket_boundaries, bucket_batch_sizes, element_length_function=None,
                               pad_info=None, pad_to_bucket_boundary=False, drop_remainder=False):
        """
        Bucket elements according to their lengths. Each bucket will be padded and batched when
        they are full.

        A length function is called on each row in the dataset. The row is then
        bucketed based on its length and bucket boundaries. When a bucket reaches its
        corresponding size specified in bucket_batch_sizes, the entire bucket will be
        padded according to pad_info, and then form a batch.

        Refer to the following figure for the execution process:

        .. image:: bucket_batch_by_length_en.png

        Note:
            - When using Data Sinking in Graph mode, the input shape of the network should keep consistent.
              You should set `drop_remainder` to "True" to discard the last incomplete batch of data,
              or supplement/remove samples to ensure the dataset size is divisible by `batch_size`.

        Args:
            column_names (list[str]): Columns passed to element_length_function.
            bucket_boundaries (list[int]): A list consisting of the upper boundaries
                of the buckets. Must be strictly increasing. If there are n boundaries,
                n+1 buckets are created: One bucket for [0, bucket_boundaries[0]), one
                bucket for [bucket_boundaries[i], bucket_boundaries[i+1]) for each
                0<i<n-1, and the last bucket for [bucket_boundaries[n-1], inf).
            bucket_batch_sizes (list[int]): A list consisting of the batch sizes for
                each bucket. Must contain len(bucket_boundaries)+1 elements.
            element_length_function (Callable, optional): A function that takes in
                M arguments where M = len(column_names) and returns an integer. If no value
                provided, parameter M the len(column_names) must be 1. At this time, the length of the data in this
                column is determined based on its ndim. If ndim=0, the data length is 0, indicating a str, bool, int,
                or float scalar; if it is an array with ndim > 0, the length of the data is array.shape[0].
                Default: ``None`` , indicating this parameter is not specified.
            pad_info (dict, optional): The information about how to batch each column. The key
                corresponds to the column name, and the value must be a tuple of 2 elements.
                The first element corresponds to the shape to pad to, and the second
                element corresponds to the value to pad with. If a column is not
                specified, then that column will be padded to the longest in the current
                batch, and 0 will be used as the padding value. Any None dimensions will
                be padded to the longest in the current batch, unless if
                `pad_to_bucket_boundary` is ``True``. If no padding is wanted, set `pad_info`
                to ``None``. Default: ``None``.
            pad_to_bucket_boundary (bool, optional): If `pad_to_bucket_boundary` is True,
                columns in `pad_info` with a shape of None will be padded to a length of one less than
                the corresponding bucket size specified by the parameter `bucket_batch_sizes`. If there are any
                elements that fall into the last bucket, an error will occur.
                Default: ``False``.
            drop_remainder (bool, optional): If ``True``, will drop the last batch for each
                bucket if it is not a full batch. Default: ``False``.

        Returns:
            Dataset, a new dataset with the above operation applied.

        Examples:
            >>> # Create a dataset where certain counts rows are combined into a batch
            >>> # and drops the last incomplete batch if there is one.
            >>> import mindspore.dataset as ds
            >>> import numpy as np
            >>> def generate_2_columns(n):
            ...     for i in range(n):
            ...         yield (np.array([i]), np.array([j for j in range(i + 1)]))
            >>>
            >>> column_names = ["col1", "col2"]
            >>> dataset = ds.GeneratorDataset(generate_2_columns(8), column_names)
            >>> bucket_boundaries = [5, 10]
            >>> bucket_batch_sizes = [2, 1, 1]
            >>> element_length_function = (lambda col1, col2: max(len(col1), len(col2)))
            >>> # Will pad col2 to shape [bucket_boundaries[i]] where i is the
            >>> # index of the bucket that is currently being batched.
            >>> pad_info = {"col2": ([None], -1)}
            >>> pad_to_bucket_boundary = True
            >>> dataset = dataset.bucket_batch_by_length(column_names, bucket_boundaries,
            ...                                          bucket_batch_sizes,
            ...                                          element_length_function, pad_info,
            ...                                          pad_to_bucket_boundary)
        """
        return BucketBatchByLengthDataset(self, column_names, bucket_boundaries, bucket_batch_sizes,
                                          element_length_function, pad_info, pad_to_bucket_boundary, drop_remainder)

    @check_batch
    def batch(self, batch_size, drop_remainder=False, num_parallel_workers=None, **kwargs):
        """
        Combine `batch_size` number of consecutive rows into batch which apply `per_batch_map` to the samples first.

        For any column, all the elements within that column must have the same shape.

        Refer to the following figure for the execution process:

        .. image:: batch_en.png

        Note:
            - The order of using repeat and batch reflects the number of batches and per_batch_map.
              It is recommended that the repeat operation applied after the batch operation finished.
            - When using Data Sinking in Graph mode, the input shape of the network should keep consistent.
              You should set `drop_remainder` to "True" to discard the last incomplete batch of data,
              or supplement/remove samples to ensure the dataset size is divisible by `batch_size`.
            - The parameter `max_rowsize` will be deprecated in a future version.

        Args:
            batch_size (Union[int, Callable]): The number of rows each batch is created with. An
                int or callable object which takes exactly 1 parameter, BatchInfo.
            drop_remainder (bool, optional): Determines whether or not to drop the last block
                whose data row number is less than batch size. Default: ``False`` . If ``True`` ,
                and if there are less than `batch_size` rows available to make the last batch,
                then those rows will be dropped and not propagated to the child node.
            num_parallel_workers (int, optional): Number of workers(threads) to process the dataset in parallel.
                Default: ``None`` .
            **kwargs:

                - per_batch_map (Callable[[List[numpy.ndarray], ..., List[numpy.ndarray], BatchInfo], \
                  (List[numpy.ndarray], ..., List[numpy.ndarray])], optional): Per batch map callable.
                  Default: ``None``.
                  A callable which takes (List[numpy.ndarray], ..., List[numpy.ndarray], BatchInfo) as input parameters.
                  Each list[numpy.ndarray] represents a batch of numpy.ndarray on a given column. The number of lists
                  should match with the number of entries in input_columns. The last parameter of the callable should
                  always be a BatchInfo object. Per_batch_map should return
                  (list[numpy.ndarray], list[numpy.ndarray], ...). The length of each list in output should be the same
                  as the input. output_columns is required if the number of output lists is different from input.

                - input_columns (Union[str, list[str]], optional): List of names of the input columns. The size of
                  the list should match with signature of `per_batch_map` callable. Default: ``None`` .

                - output_columns (Union[str, list[str]], optional): List of names assigned to the columns
                  outputted by the last operation. This parameter is mandatory if len(input_columns) !=
                  len(output_columns). The size of this list must match the number of output
                  columns of the last operation. Default: ``None`` , output columns will have the same
                  name as the input columns, i.e., the columns will be replaced.

                - python_multiprocessing (bool, optional): Parallelize Python function `per_batch_map` with
                  multiprocessing or multithreading mode, ``True`` means multiprocessing,
                  ``False`` means multithreading. If `per_batch_map` is a I/O bound task, use
                  multithreading mode. If `per_batch_map` is a CPU bound task, it is recommended to use
                  multiprocessing mode. Default: ``False`` , use python multithreading mode.

                - max_rowsize (Union[int, list[int]], optional): Maximum size of row in MB that is used for shared
                  memory allocation to copy data between processes, the total occupied shared memory will increase as
                  ``num_parallel_workers`` and :func:`mindspore.dataset.config.set_prefetch_size` increase.
                  This is only used if ``python_multiprocessing`` is set to ``True``.
                  Default: ``None`` , allocate shared memory dynamically (deprecated in future version).

                  - If set to ``-1`` / ``None``, shared memory will be dynamically allocated with the
                    actual size of data.

                  - If it is an int value, it represents ``input_columns`` and ``output_columns`` use this value as the
                    unit to create shared memory.

                  - If it is a list, represents the ``input_columns`` use the first element as the unit to
                    create shared memory, and represents ``output_columns`` use the second element as the
                    unit to create shared memory.

        .. warning::
            `batch` uses `dill` module implicitly in multiprocessing `spawn` mode to serialize/deserialize
            `per_batch_map`, which is known to be insecure. It is possible to construct malicious pickle data
            which will execute arbitrary code during unpickling. Never load data that could have come from
            untrusted sources, or has been tampered with.

        Returns:
            Dataset, a new dataset with the above operation applied.

        Examples:
            >>> # 1) Create a dataset where every 5 rows are combined into a batch
            >>> # and drops the last incomplete batch if there is one.
            >>> import mindspore.dataset as ds
            >>> from PIL import Image
            >>>
            >>> cifar10_dataset_dir = "/path/to/cifar10_dataset_directory"
            >>> dataset = ds.Cifar10Dataset(dataset_dir=cifar10_dataset_dir, num_samples=10)
            >>> dataset = dataset.batch(5, True)
            >>>
            >>> # 2) resize image according to its batch number, if it's 5-th batch, resize to (5^2, 5^2) = (25, 25)
            >>> def np_resize(col, BatchInfo):
            ...     output = col.copy()
            ...     s = (BatchInfo.get_batch_num() + 1) ** 2
            ...     index = 0
            ...     for c in col:
            ...         img = Image.fromarray(c.astype('uint8')).convert('RGB')
            ...         img = img.resize((s, s))
            ...         output[index] = np.array(img)
            ...         index += 1
            ...     return (output,)
            >>> dataset = dataset.batch(batch_size=8, input_columns=["image"], per_batch_map=np_resize)
            >>>
            >>> # 3) Create a dataset where its batch size is dynamic
            >>> # Define a callable batch size function and let batch size increase 1 each time.
            >>> def add_one(BatchInfo):
            ...     return BatchInfo.get_batch_num() + 1
            >>> dataset = dataset.batch(batch_size=add_one, drop_remainder=True)
        """
        return BatchDataset(self, batch_size, drop_remainder, num_parallel_workers, **kwargs)

    @check_padded_batch
    def padded_batch(self, batch_size, drop_remainder=False, num_parallel_workers=None, pad_info=None):
        """
        Combine batch_size number of consecutive rows into batches which apply pad_info to the samples first.

        Refer to the following figure for the execution process:

        .. image:: padded_batch_en.png

        Note:
            - The order of using repeat and padded_batch reflects the number of batches.
              It is recommended that the repeat operation applied after the padded_batch operation finished.
            - When using Data Sinking in Graph mode, the input shape of the network should keep consistent.
              You should set `drop_remainder` to "True" to discard the last incomplete batch of data,
              or supplement/remove samples to ensure the dataset size is divisible by `batch_size`.

        Args:
            batch_size (Union[int, Callable]): The number of rows each batch is created with. An
                int or callable object which takes exactly 1 parameter, BatchInfo.
            drop_remainder (bool, optional): Determines whether or not to drop the last block
                whose data row number is less than batch size. Default: ``False``. If ``True``, and if there
                are less than batch_size rows available to make the last batch, then those rows will
                be dropped and not propagated to the child node.
            num_parallel_workers (int, optional): Number of workers(threads) to process the dataset in parallel.
                Default: ``None``.
            pad_info (dict, optional): The pad information about how to batch each column. The key
                corresponds to the column name, and the value must be a tuple of 2 elements.
                The first element corresponds to the shape to pad to, and the second
                element corresponds to the value to pad with. If a column is not
                specified, then that column will be padded to the longest in the current
                batch, and 0 will be used as the padding value. If ``pad_info={"col1": ([224, 224], 0)}``,
                expand the data column named ``col1`` to shape (224, 224), and fill in the missing values with 0.
                If ``pad_info={}``, all samples in the batch will be filled to the shape with the largest sample
                in the current batch. If ``pad_info={"col1": (None, 100)}``, all samples in the batch will be filled
                to the shape with the largest sample in the current batch, and fill in the missing values with 100.
                If no padding is wanted, set `pad_info` to ``None``. Default: ``None``.

        Returns:
            Dataset, a new dataset with the above operation applied.

        Examples:
            >>> # 1) Pad every sample to the largest sample's shape and batch the samples
            >>> import mindspore.dataset as ds
            >>> dataset = ds.NumpySlicesDataset([[1], [1, 2], [1, 2, 3], [1, 2, 3, 4]], "column1")
            >>> dataset = dataset.padded_batch(2, True, pad_info={})
            >>>
            >>> # 2) Create a dataset where every 3 rows are combined into a batch
            >>> # and drops the last incomplete batch if there is one.
            >>> dataset = ds.NumpySlicesDataset([i for i in range(10)], "column1")
            >>> dataset = dataset.padded_batch(3, True)
            >>>
            >>> # 3) Create a dataset where its batch size is dynamic
            >>> # Define a callable batch size function and let batch size increase 1 each time.
            >>> def add_one(BatchInfo):
            ...     return BatchInfo.get_batch_num() + 1
            >>> dataset = dataset.padded_batch(batch_size=add_one, drop_remainder=True)
        """
        return PaddedBatchDataset(self, batch_size, drop_remainder, num_parallel_workers, pad_info)

    @check_sync_wait
    def sync_wait(self, condition_name, num_batch=1, callback=None):
        """
        Add a blocking condition to the input Dataset and a synchronize action will be applied.

        Args:
            condition_name (str): The condition name that is used to toggle sending next row.
            num_batch (int, optional): The number of batches without blocking at the start of each epoch.
                Default: ``1``.
            callback (function, optional): The callback function that will be invoked when sync_update is called.
                Default: ``None``.

        Returns:
            Dataset, a new dataset with the above operation applied.

        Raises:
            RuntimeError: If condition name already exists.

        Examples:
            >>> import mindspore.dataset as ds
            >>> import numpy as np
            >>> def gen():
            ...     for i in range(100):
            ...         yield (np.array(i),)
            >>>
            >>> class Augment:
            ...     def __init__(self, loss):
            ...         self.loss = loss
            ...
            ...     def preprocess(self, input_):
            ...         return input_
            ...
            ...     def update(self, data):
            ...         self.loss = data["loss"]
            >>>
            >>> batch_size = 4
            >>> dataset = ds.GeneratorDataset(gen, column_names=["input"])
            >>>
            >>> aug = Augment(0)
            >>> dataset = dataset.sync_wait(condition_name="policy", callback=aug.update)
            >>> dataset = dataset.map(operations=[aug.preprocess], input_columns=["input"])
            >>> dataset = dataset.batch(batch_size)
            >>> count = 0
            >>> for data in dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
            ...     assert data["input"][0] == count
            ...     count += batch_size
            ...     data = {"loss": count}
            ...     dataset.sync_update(condition_name="policy", data=data)
        """
        return SyncWaitDataset(self, condition_name, num_batch, callback)

    @check_shuffle
    def shuffle(self, buffer_size):
        """
        Shuffle the dataset by creating a cache with the size of `buffer_size` .

        1. Make a shuffle buffer that contains the first `buffer_size` rows.
        2. Randomly select an element from the shuffle buffer to be the next row
           propagated to the child node.
        3. Get the next row (if any) from the parent node and put it in the shuffle buffer.
        4. Repeat steps 2 and 3 until there are no more rows left in the shuffle buffer.

        A random seed can be provided to be used on the first epoch via `dataset.config.set_seed` . In every subsequent
        epoch, the seed is changed to a new one, randomly generated value.

        Args:
            buffer_size (int): The size of the buffer (must be larger than 1) for
                shuffling. Setting `buffer_size` equal to the number of rows in the entire
                dataset will result in a global shuffle.

        Returns:
            Dataset, a new dataset with the above operation applied.

        Raises:
            RuntimeError: If exist sync operations before shuffle.

        Examples:
            >>> import mindspore.dataset as ds
            >>> dataset = ds.GeneratorDataset([i for i in range(10)], "column1")
            >>>
            >>> # Optionally set the seed for fixed randomness
            >>> ds.config.set_seed(58)
            >>>
            >>> # Create a shuffled dataset using a shuffle buffer of size 4
            >>> dataset = dataset.shuffle(4)
        """
        return ShuffleDataset(self, buffer_size)

    def flat_map(self, func):
        """
        Map `func` to each row in dataset and flatten the result.

        Args:
            func (function): A function that must take one `numpy.ndarray` as an argument and
                return a `Dataset` .

        Returns:
            Dataset, a new dataset with the above operation applied.

        Raises:
            TypeError: If `func` is not a function.
            TypeError: If `func` doesn't return a Dataset.

        Examples:
            >>> import mindspore.dataset as ds
            >>> # 1) flat_map on one column dataset
            >>> dataset = ds.NumpySlicesDataset([[0, 1], [2, 3]], shuffle=False)
            >>>
            >>> def repeat(array):
            ...     # create a NumpySlicesDataset with the array
            ...     data = ds.NumpySlicesDataset(array, shuffle=False)
            ...     # repeat the dataset twice
            ...     data = data.repeat(2)
            ...     return data
            >>>
            >>> dataset = dataset.flat_map(repeat)
            >>> # [0, 1, 0, 1, 2, 3, 2, 3]
            >>>
            >>> # 2) flat_map on multi column dataset
            >>> dataset = ds.NumpySlicesDataset(([[0, 1], [2, 3]], [[0, -1], [-2, -3]]), shuffle=False)
            >>>
            >>> def plus_and_minus(col1, col2):
            ...     # apply different methods on columns
            ...     data = ds.NumpySlicesDataset((col1 + 1, col2 - 1), shuffle=False)
            ...     return data
            >>>
            >>> dataset = dataset.flat_map(plus_and_minus)
            >>> # ([1, 2, 3, 4], [-1, -2, -3, -4])
        """
        dataset = None
        if not hasattr(func, '__call__'):
            logger.critical("func must be a function.")
            raise TypeError("func must be a function.")

        for row_data in self.create_tuple_iterator(num_epochs=1, output_numpy=True):
            if dataset is None:
                dataset = func(*row_data)
            else:
                dataset += func(*row_data)

        if not isinstance(dataset, Dataset):
            logger.critical("flat_map must return a Dataset object.")
            raise TypeError("flat_map must return a Dataset object.")
        return dataset

    @check_map
    def map(self, operations, input_columns=None, output_columns=None, num_parallel_workers=None, **kwargs):
        """
        Apply each operation in operations to this dataset.

        Each operation will be passed one or more columns from the dataset as input, and one or
        more columns will be outputted. The first operation will be passed the columns specified
        in input_columns as input. If there is more than one operation in operations, the outputted
        columns of the previous operation are used as the input columns for the next operation.

        The columns outputted by the very last operation will be assigned names specified by
        `output_columns` , and if not specified, the column name of output column is same as that of `input_columns` .

        - If you use transformations (
          `vision transform <https://mindspore.cn/docs/en/master/api_python/mindspore.\
          dataset.transforms.html#module-mindspore.dataset.vision>`_ ,
          `nlp transform <https://mindspore.cn/docs/en/master/api_python/mindspore.\
          dataset.transforms.html#module-mindspore.dataset.text>`_ ,
          `audio transform <https://mindspore.cn/docs/en/master/api_python/mindspore.\
          dataset.transforms.html#module-mindspore.dataset.audio>`_ )
          provided by mindspore dataset, please use the following parameters:

          .. image:: map_parameter_en.png

        - If you use user-defined transform as PyFunc (Python Func), please use the following parameters:

          .. image:: map_parameter_pyfunc_en.png

        Args:
            operations (Union[list[TensorOperation], list[functions]]): List of operations to be
                applied on the dataset. Operations are applied in the order they appear in this list.
            input_columns (Union[str, list[str]], optional): List of the names of the columns that will be passed to
                the first operation as input. The size of this list must match the number of
                input columns expected by the first operation. Default: ``None``, the first
                operation will be passed however many columns that are required, starting from
                the first column.
            output_columns (Union[str, list[str]], optional): List of names assigned to the columns outputted by
                the last operation. This parameter is mandatory if len(input_columns) !=
                len(output_columns). The size of this list must match the number of output
                columns of the last operation. Default: ``None``, output columns will have the same
                name as the input columns, i.e., the columns will be replaced.
            num_parallel_workers (int, optional): Number of threads used to process the dataset in
                parallel. Default: ``None``, the value from the configuration will be used.
            **kwargs:

                - python_multiprocessing (bool, optional): Parallelize Python operations with multiple worker processes.
                  This option could be beneficial if the Python operation is computational heavy. Default: ``False``.

                - max_rowsize (Union[int, list[int]], optional): Maximum size of row in MB that is used for shared
                  memory allocation to copy data between processes, the total occupied shared memory will increase as
                  ``num_parallel_workers`` and :func:`mindspore.dataset.config.set_prefetch_size` increase.
                  This is only used if ``python_multiprocessing`` is set to ``True``.
                  Default: ``None`` , allocate shared memory dynamically (deprecated in future version).

                  - If set to ``-1`` / ``None``, shared memory will be dynamically allocated with the
                    actual size of data.

                  - If it is an int value, it represents ``input_columns`` and ``output_columns`` use this value as the
                    unit to create shared memory.

                  - If it is a list, the first element represents the ``input_columns`` use this value as the unit to
                    create shared memory, and the second element represents ``output_columns`` use this value as the
                    unit to create shared memory.

                - cache (DatasetCache, optional): Use tensor caching service to speed up dataset processing.
                  Default: ``None``, which means no cache is used.

                - callbacks (DSCallback, list[DSCallback], optional): List of Dataset callbacks to be called.
                  Default: ``None``.

                - offload (bool, optional): Flag to indicate whether offload is used. Default: ``None``.

        .. warning::
            `map` uses `dill` module implicitly in multiprocessing `spawn` mode to serialize/deserialize `operations`,
            which is known to be insecure. It is possible to construct malicious pickle data which will
            execute arbitrary code during unpickling. Never load data that could have come from untrusted sources,
            or has been tampered with.

        Note:
            - The parameter `max_rowsize` will be deprecated in a future version.
            - Input `operations` accepts TensorOperations defined in mindspore.dataset part, plus user-defined
              Python functions (PyFuncs).
            - Setting the start method of multiprocessing to `spawn` mode by
              ds.config.set_multiprocessing_start_method("spawn") with `python_multiprocessing=True`
              and `num_parallel_workers>1` supports adding network computing operators from mindspore.nn and
              mindspore.ops or other network computing operators into this `operations` .
              Otherwise, adding to `operations` is not supported.
            - Currently only some scenarios support calling DVPP operators in Python functions passed in with the
              `operations` parameter:

              +---------------+----------------------------+----------------------------+----------------------------+
              |               |                            |                     Multiprocessing                     |
              |               |       Multithreading       +----------------------------+----------------------------+
              |               |                            |           spawn            |            fork            |
              +===============+============================+============================+============================+
              |Independent    |Data Processing: support    |Data Processing: support    |Data Processing: support    |
              |               |                            |                            |                            |
              |process mode   |Data Processing + Network   |Data Processing + Network   |Data Processing + Network   |
              |               |training: not support       |training: support           |training: not support       |
              +---------------+----------------------------+----------------------------+----------------------------+
              |Non-independent|Data Processing: support    |Data Processing: support    |Data Processing: support    |
              |               |                            |                            |                            |
              |process mode   |Data Processing + Network   |Data Processing + Network   |Data Processing + Network   |
              |               |training: support           |training: support           |training: not support       |
              +---------------+----------------------------+----------------------------+----------------------------+

        Returns:
            Dataset, a new dataset with the above operation applied.

        Examples:
            >>> import mindspore.dataset as ds
            >>> import mindspore.dataset.vision as vision
            >>> # dataset is an instance of Dataset which has 2 columns, "image" and "label".
            >>> # image is of type bytes type which can be decoded to RGB
            >>> # label is of type int32
            >>> cifar10_dataset_dir = "/path/to/cifar10_dataset_directory"
            >>> dataset = ds.Cifar10Dataset(dataset_dir=cifar10_dataset_dir)
            >>>
            >>> # Define two operations, where each operation accepts 1 input column and outputs 1 column.
            >>> decode_op = vision.Decode(to_pil=False)
            >>> random_jitter_op = vision.RandomColorAdjust(brightness=(0.8, 0.8), contrast=(1, 1),
            ...                                             saturation=(1, 1), hue=(0, 0))
            >>>
            >>> # 1) Simple map example.
            >>>
            >>> # Apply decode_op on column "image".
            >>> dataset = dataset.map(operations=[decode_op], input_columns=["image"])
            >>>
            >>> # Decode and rename column "image" to "decoded_image".
            >>> dataset = dataset.map(operations=[decode_op], input_columns=["image"], output_columns=["decoded_image"])
            >>>
            >>> # A simple example for user defined python function transform.
            >>> dataset = ds.NumpySlicesDataset(data=[[0, 1, 2]], column_names=["data"])
            >>> dataset = dataset.map(operations=[(lambda x: x - 1)], input_columns=["data"])
            >>>
            >>> # 2) Map example with more than one operation.
            >>>
            >>> # Create a dataset where the images are decoded, then randomly color jittered.
            >>> # decode_op takes column "image" as input and outputs one column. The column
            >>> # outputted by decode_op is passed as input to random_jitter_op.
            >>> # random_jitter_op will output one column. Column "image" will be replaced by
            >>> # the column outputted by random_jitter_op (the very last operation). All other
            >>> # columns are unchanged.
            >>> dataset = dataset.map(operations=[decode_op, random_jitter_op], input_columns=["image"])
            >>>
            >>> # Rename the column outputted by random_jitter_op to "image_mapped".
            >>> dataset = dataset.map(operations=[decode_op, random_jitter_op], input_columns=["image"],
            ...                       output_columns=["image_mapped"])
            >>>
            >>> # Map with multiple operations using pyfunc and rename column's name
            >>> dataset = ds.NumpySlicesDataset(data=[[0, 1, 2]], column_names=["data"])
            >>> dataset = dataset.map(operations=[(lambda x: x * x), (lambda x: x - 1)], input_columns=["data"],
            ...                                   output_columns=["data_mapped"])
            >>>
            >>> # 3) Example where number of input columns is not equal to number of output columns.
            >>>
            >>> # operations[0] is a lambda that takes 2 columns as input and outputs 3 columns.
            >>> # operations[1] is a lambda that takes 3 columns as input and outputs 1 column.
            >>> # operations[2] is a lambda that takes 1 column as input and outputs 4 columns.
            >>> #
            >>> # Note: The number of output columns of operation[i] must equal the number of
            >>> # input columns of operation[i+1]. Otherwise, this map call will also result
            >>> # in an error.
            >>> operations = [(lambda x, y: (x, x + y, x + y + 1)),
            ...               (lambda x, y, z: x * y * z),
            ...               (lambda x: (x % 2, x % 3, x % 5, x % 7))]
            >>> dataset = ds.NumpySlicesDataset(data=([[0, 1, 2]], [[3, 4, 5]]), column_names=["x", "y"])
            >>> dataset = dataset.map(operations, input_columns=["x", "y"],
            ...                       output_columns=["mod2", "mod3", "mod5", "mod7"])
        """
        if hasattr(self, 'operator_mixed') and self.operator_mixed is True:
            num_parallel_workers = 1
            logger.warning(
                "Input 'operations' of 'map' includes network computing operators like in mindspore.nn, mindspore.ops, "
                "mindspore.numpy module and etc, which do not support multithreading compiling, recommend to replace "
                "it with python implemented operator like numpy etc. Here decrease 'num_parallel_workers' into 1.")

        return MapDataset(self, operations, input_columns, output_columns, num_parallel_workers, **kwargs)

    @check_filter
    def filter(self, predicate, input_columns=None, num_parallel_workers=None):
        """
        Filter dataset by predicate.

        Args:
            predicate (callable): Python callable which returns a boolean value. If False then filter the element.
            input_columns (Union[str, list[str]], optional): List of names of the input columns. If not provided
                or provided with ``None``, the predicate will be applied on all columns in the dataset.
                Default: ``None``.
            num_parallel_workers (int, optional): Number of workers to process the dataset
                in parallel. Default: ``None``.

        Returns:
            Dataset, a new dataset with the above operation applied.

        Examples:
            >>> # generator data(0 ~ 19)
            >>> # filter the data that greater than or equal to 11
            >>> import mindspore.dataset as ds
            >>> dataset = ds.GeneratorDataset([i for i in range(20)], "data")
            >>> dataset = dataset.filter(predicate=lambda data: data < 11, input_columns = ["data"])
        """
        return FilterDataset(self, predicate, input_columns, num_parallel_workers)

    @check_repeat
    def repeat(self, count=None):
        """
        Repeat this dataset `count` times. Repeat infinitely if the `count` is ``None`` or ``-1``.

        Note:
            The order of using repeat and batch reflects the number of batches. It is recommended that
            the repeat operation is used after the batch operation.

        Args:
            count (int, optional): Number of times the dataset is going to be repeated. Default: ``None``.

        Returns:
            Dataset, a new dataset with the above operation applied.

        Examples:
            >>> import mindspore.dataset as ds
            >>>
            >>> # Create a dataset with 10 elements
            >>> dataset = ds.GeneratorDataset([i for i in range(10)], "column1")
            >>> ori_size = dataset.get_dataset_size()
            >>>
            >>> # Repeat the dataset 50 times.
            >>> dataset = dataset.repeat(50)
            >>> repeated_size = dataset.get_dataset_size()
            >>> print("ori_size", ori_size, ", repeated_size", repeated_size)
            ori_size 10 , repeated_size 500
            >>>
            >>> # Since the original dataset size is less than batch_size, thus no data is returned
            >>> dataset1 = ds.GeneratorDataset([i for i in range(10)], "column1")
            >>> dataset1 = dataset1.batch(batch_size=20, drop_remainder=True)
            >>> dataset1 = dataset1.repeat(6)
            >>>
            >>> # Repeat the original dataset to 60 elements, thus 3 batches are returned
            >>> dataset2 = ds.GeneratorDataset([i for i in range(10)], "column1")
            >>> dataset2 = dataset2.repeat(6)
            >>> dataset2 = dataset2.batch(batch_size=20, drop_remainder=True)
            >>> print("dataset1 size", dataset1.get_dataset_size(), ", dataset2 size", dataset2.get_dataset_size())
            dataset1 size 0 , dataset2 size 3
        """
        return RepeatDataset(self, count)

    @check_skip
    def skip(self, count):
        """
        Skip the first N elements of this dataset.

        Args:
            count (int): Number of elements in the dataset to be skipped.

        Returns:
            Dataset, a new dataset with the above operation applied.

        Examples:
            >>> import mindspore.dataset as ds
            >>> dataset = ds.GeneratorDataset([i for i in range(10)], "column1")
            >>> # Skip first 3 elements of dataset and retain 7 elements.
            >>> dataset = dataset.skip(3)
        """
        return SkipDataset(self, count)

    @check_take
    def take(self, count=-1):
        """
        Take the first specified number of samples from the dataset.

        Args:
            count (int, optional): The desired number of samples to take. If the value exceeds
                the total number of samples in the dataset, all data will be returned.
                Default: ``-1`` , will return all data.

        Note:
            When there are operations that will change the number of samples of the dataset in
            the data pipeline, the location of the `take` operation can change its effect.
            For example, `batch` operation will combine the successive samples of the specified
            `batch_size` into 1 sample, so `.batch(batch_size).take(1)` will be equivalent to
            `.take(batch_size).batch(batch_size)`.

        Returns:
            Dataset, a new dataset with the above operation applied.

        Examples:
            >>> import mindspore.dataset as ds
            >>> mnist_dataset_dir = "/path/to/mnist_dataset_directory"
            >>> dataset = ds.MnistDataset(dataset_dir=mnist_dataset_dir)
            >>> # Take 50 samples from MNIST dataset.
            >>> dataset = dataset.take(50)
        """
        return TakeDataset(self, count)

    def _get_absolute_split_sizes(self, sizes):
        """
        Internal method called by split to calculate absolute split sizes and to
        do some error checking after calculating absolute split sizes.

        Returns:
            int, absolute split sizes of the dataset.
        """
        # Call get_dataset_size here and check input here because
        # don't want to call this once in check_split and another time in
        # here again
        dataset_size = self.get_dataset_size()

        if dataset_size is None or dataset_size <= 0:
            raise RuntimeError("dataset_size is unknown, unable to split.")

        if not isinstance(sizes, list):
            raise RuntimeError("sizes must be a list.")

        all_int = all(isinstance(item, int) for item in sizes)
        if all_int:
            sizes_sum = sum(sizes)
            if sizes_sum != dataset_size:
                raise RuntimeError(f"Sum of split sizes {sizes_sum} is not equal to dataset size {dataset_size}.")
            return sizes

        absolute_sizes = []
        for item in sizes:
            absolute_size = int(round(item * dataset_size))
            if absolute_size == 0:
                raise RuntimeError(f"Split percentage {item} is too small.")
            absolute_sizes.append(absolute_size)

        absolute_sizes_sum = sum(absolute_sizes)

        # if we still need more rows, give them to the first split.
        # if we have too many rows, remove the extras from the first split that has
        # enough rows.
        size_difference = int(dataset_size - absolute_sizes_sum)
        if size_difference > 0:
            absolute_sizes[0] += size_difference
        else:
            for i, _ in enumerate(absolute_sizes):
                if absolute_sizes[i] + size_difference > 0:
                    absolute_sizes[i] += size_difference
                    break

        if sum(absolute_sizes) != dataset_size:
            raise RuntimeError(f"Sum of calculated split sizes {absolute_sizes_sum} is not equal to " +
                               f"dataset size {dataset_size}.")

        return absolute_sizes

    @check_split
    def split(self, sizes, randomize=True):
        """
        Split the dataset into smaller, non-overlapping datasets.

        Args:
            sizes (Union[list[int], list[float]]): If a list of integers [s1, s2, …, sn] is
                provided, the dataset will be split into n datasets of size s1, size s2, …, size sn
                respectively. If the sum of all input sizes does not equal the original dataset size, an
                error will throw.
                If a list of floats [f1, f2, …, fn] is provided, all floats must be between 0 and 1
                and must sum to 1, otherwise an error will throw. The dataset will be split into n
                datasets of size round(f1*K), round(f2*K), …, round(fn*K) where K is the size of the
                original dataset.
                If after rounding:

                - Any size equals 0, an error will occur.
                - The sum of split sizes < K, the difference of K - sigma(round(fi * k)) will be added to the first
                  split.
                - The sum of split sizes > K, the difference of sigma(round(fi * K)) - K will be removed from the first
                  large enough split such that it will have at least 1 row after removing the difference.

            randomize (bool, optional): Determines whether or not to split the data randomly. Default: ``True``.
                If True, the data will be randomly split. Otherwise, each split will be created with
                consecutive rows from the dataset.

        Note:
            1. If the dataset object for which the split operation is performed is of type MappableDataset,
               an optimized split function will be called automatically.
            2. If the split function is performed, the dataset object should not be sharded (e.g. by specifying
               num_shards or using :class:`mindspore.dataset.DistributedSampler`). Instead, 
               create a :class:`mindspore.dataset.DistributedSampler` and specify a split to shard after splitting.
               It is strongly recommended to set the same seed in each instance of execution,
               otherwise each shard may not be part of the same split (see Examples).
            3. It is strongly recommended to not shuffle the dataset, but use randomize=True instead.
               Shuffling the dataset may not be deterministic, which means the data in each split
               will be different in each epoch.

        Returns:
            Tuple[Dataset], a tuple of new datasets split from the original one.

        Raises:
            RuntimeError: If get_dataset_size returns None or is not supported for this dataset.
            RuntimeError: If `sizes` is list of integers and sum of all elements in sizes does not
                equal the dataset size.
            RuntimeError: If `sizes` is list of float and there is a split with size 0 after calculations.
            RuntimeError: If the dataset is sharded prior to calling split.
            ValueError: If `sizes` is list of float and not all floats are between 0 and 1, or if the
                floats don't sum to 1.

        Examples:
            >>> # Split the data into train part and test part.
            >>> import mindspore.dataset as ds
            >>> dataset = ds.GeneratorDataset([i for i in range(10)], "column1")
            >>> train_dataset, test_dataset = dataset.split([0.9, 0.1])
        """
        if self.is_shuffled():
            logger.warning("Dataset is shuffled before split.")

        if self.is_sharded():
            raise RuntimeError("Dataset should not be sharded before split.")

        absolute_sizes = self._get_absolute_split_sizes(sizes)
        splits = []
        rows_to_skip = 0
        for size in absolute_sizes:
            ds = copy.deepcopy(self)
            if randomize:
                # want to shuffle the same way every epoch before split
                # in alter_tree, shuffle buffer is minimum 10000, so use 10000 here
                ds = ds.shuffle(10000)
                ds.reshuffle_each_epoch = False

            if rows_to_skip > 0:
                ds = ds.skip(rows_to_skip)

            ds = ds.take(size)
            splits.append(ds)

            rows_to_skip += size

        return tuple(splits)

    @check_zip_dataset
    def zip(self, datasets):
        """
        Zip the datasets in the sense of input tuple of datasets. Columns in the input datasets must have different
        names.

        Args:
            datasets (Union[Dataset, tuple[Dataset]]): A tuple of datasets or a single class Dataset
                to be zipped together with this dataset.

        Returns:
            Dataset, a new dataset with the above operation applied.

        Raises:
            TypeError: The parameter is not dataset object or tuple of dataset objects.

        Examples:
            >>> # Create a dataset which is the combination of dataset_1 and dataset_2
            >>> import mindspore.dataset as ds
            >>> dataset_1 = ds.GeneratorDataset([1, 2, 3], "column1")
            >>> dataset_2 = ds.GeneratorDataset([1, 2, 3], "column2")
            >>> dataset = dataset_1.zip(dataset_2)
        """
        if isinstance(datasets, tuple):
            datasets = (self, *datasets)
        elif isinstance(datasets, Dataset):
            datasets = (self, datasets)
        else:
            raise TypeError(f"Invalid datasets, expected Dataset object or tuple of Dataset, but got {datasets}!")
        return ZipDataset(datasets)

    @check_concat
    def concat(self, datasets):
        """
        Concatenate the dataset objects in the input list.
        Performing "+" operation on dataset objects can achieve the same effect.

        For a dataset concatenated by many other dataset objects, it returns the data in the order of
        datasets passed in. If you want to change the data order (such as random selection from each dataset
        instead of in sequence), apply `use_sampler` method on the concatenated dataset object.
        Currently `use_sampler` supports `dataset.DistributedSampler` for sharding selection from each dataset
        or `dataset.RandomSampler` for random selection from each dataset, see examples below.

        Note:
            The column name, and rank and type of the column data must be the same in the input datasets.

        Args:
            datasets (Union[list, Dataset]): A list of datasets or a single class Dataset
                to be concatenated together with this dataset.

        Returns:
            Dataset, a new dataset with the above operation applied.

        Examples:
            >>> import mindspore.dataset as ds
            >>> dataset_1 = ds.GeneratorDataset([1, 2, 3], "column1", shuffle=False)
            >>> dataset_2 = ds.GeneratorDataset([4, 5, 6], "column1", shuffle=False)
            >>>
            >>> # Create a dataset by concatenating dataset_1 and dataset_2 with "+" operator
            >>> dataset = dataset_1 + dataset_2
            >>> # Create a dataset by concatenating dataset_1 and dataset_2 with concat operation
            >>> dataset = dataset_1.concat(dataset_2)
            >>>
            >>> # Check the data order of dataset
            >>> dataset_1 = ds.GeneratorDataset([1, 2, 3], "column1", shuffle=False)
            >>> dataset_2 = ds.GeneratorDataset([4, 5, 6], "column1", shuffle=False)
            >>> dataset = dataset_1 + dataset_2
            >>> result = list(dataset)
            >>> # [[Tensor(shape=[], dtype=Int64, value= 1)], [Tensor(shape=[], dtype=Int64, value= 2)],
            >>> #  [Tensor(shape=[], dtype=Int64, value= 3)], [Tensor(shape=[], dtype=Int64, value= 4)],
            >>> #  [Tensor(shape=[], dtype=Int64, value= 5)], [Tensor(shape=[], dtype=Int64, value= 6)]]
            >>>
            >>> # Change the data order of concatenated dataset with sharding selection
            >>> dataset_1 = ds.GeneratorDataset([1, 2, 3], "column1", shuffle=False)
            >>> dataset_2 = ds.GeneratorDataset([4, 5, 6], "column1", shuffle=False)
            >>> dataset = dataset_1.concat(dataset_2)
            >>> dataset.use_sampler(ds.DistributedSampler(num_shards=2, shard_id=1, shuffle=False))
            >>> result = list(dataset)
            >>> # [[Tensor(shape=[], dtype=Int64, value= 2)], [Tensor(shape=[], dtype=Int64, value= 4)],
            >>> #  [Tensor(shape=[], dtype=Int64, value= 6)]]
            >>>
            >>> # Change the data order of concatenated dataset with random selection
            >>> dataset_1 = ds.GeneratorDataset([1, 2, 3], "column1", shuffle=False)
            >>> dataset_2 = ds.GeneratorDataset([4, 5, 6], "column1", shuffle=False)
            >>> dataset = dataset_1.concat(dataset_2)
            >>> dataset.use_sampler(ds.RandomSampler())
            >>> result = list(dataset)
            >>> # [[Tensor(shape=[], dtype=Int64, value= 1)], [Tensor(shape=[], dtype=Int64, value= 4)],
            >>> #  [Tensor(shape=[], dtype=Int64, value= 2)], [Tensor(shape=[], dtype=Int64, value= 5)],
            >>> #  [Tensor(shape=[], dtype=Int64, value= 6)], [Tensor(shape=[], dtype=Int64, value= 3)]]
        """
        if isinstance(datasets, Dataset):
            datasets = [self] + [datasets]
        elif isinstance(datasets, list):
            datasets = [self] + datasets
        else:
            raise TypeError(f"Invalid datasets, expected Dataset object or list of Dataset, but got {datasets}!")
        return ConcatDataset(datasets)

    @check_rename
    def rename(self, input_columns, output_columns):
        """
        Rename the columns in input datasets.

        Args:
            input_columns (Union[str, list[str]]): List of names of the input columns.
            output_columns (Union[str, list[str]]): List of names of the output columns.

        Returns:
            Dataset, a new dataset with the above operation applied.

        Examples:
            >>> import mindspore.dataset as ds
            >>> input_columns = ["input_col1", "input_col2", "input_col3"]
            >>> output_columns = ["output_col1", "output_col2", "output_col3"]
            >>>
            >>> # Create a dataset with 3 columns
            >>> dataset = ds.GeneratorDataset([(1, 2, 3), (3, 4, 5), (5, 6, 7)], column_names=input_columns)
            >>>
            >>> # Rename "input_col1" to "output_col1", "input_col2" to "output_col2", "input_col3" to "output_col3"
            >>> dataset = dataset.rename(input_columns=input_columns, output_columns=output_columns)
        """

        return RenameDataset(self, input_columns, output_columns)

    @check_project
    def project(self, columns):
        """
        The specified columns will be selected from the dataset and passed into
        the pipeline with the order specified. The other columns are discarded.

        Args:
            columns (Union[str, list[str]]): List of names of the columns to project.

        Returns:
            Dataset, a new dataset with the above operation applied.

        Examples:
            >>> import mindspore.dataset as ds
            >>> # Create a dataset with 3 columns
            >>> input_columns = ["column1", "column2", "column3"]
            >>> dataset = ds.GeneratorDataset([(1, 2, 3), (3, 4, 5), (5, 6, 7)], column_names=input_columns)
            >>>
            >>> columns_to_project = ["column3", "column1", "column2"]
            >>> # in that order, regardless of the original order of columns.
            >>> dataset = dataset.project(columns=columns_to_project)
        """

        return ProjectDataset(self, columns)

    def apply(self, apply_func):
        """
        Apply a function in this dataset.

        Args:
            apply_func (function): A function that must take one `Dataset` as an argument and
                                   return a preprocessed `Dataset` .

        Returns:
            Dataset, a new dataset with the above operation applied.

        Raises:
            TypeError: If apply_func is not a function.
            TypeError: If apply_func doesn't return a Dataset.

        Examples:
            >>> import mindspore.dataset as ds
            >>> dataset = ds.GeneratorDataset([i for i in range(10)], "column1")
            >>>
            >>> # Declare an apply_func function which returns a Dataset object
            >>> def apply_func(data):
            ...     data = data.batch(2)
            ...     return data
            >>>
            >>> # Use apply to call apply_func
            >>> dataset = dataset.apply(apply_func)
        """

        if not hasattr(apply_func, '__call__'):
            raise TypeError("apply_func must be a function.")

        dataset = apply_func(self)
        if not isinstance(dataset, Dataset):
            raise TypeError("apply_func must return a dataset.")
        return dataset

    @check_device_send
    def device_que(self, send_epoch_end=True, create_data_info_queue=False, queue_name=""):
        """
        Return a transferred Dataset that transfers data through a device.

        Args:
            send_epoch_end (bool, optional): Whether to send end of sequence to device or not.
                Default: ``True``.
            create_data_info_queue (bool, optional): Whether to create queue which stores
                types and shapes of data or not. Default: ``False``.
            queue_name (str, optional): Name of queue which connects dataset processing and model
                computing. Default: ``""``.

        Note:
            If device is Ascend, features of data will be transferred one by one. The limitation
            of data transmission per time is 256M.

        Returns:
            Dataset, a new dataset with the above operation applied.

        Examples:
            >>> import mindspore.dataset as ds
            >>> import time
            >>>
            >>> data = ds.TFRecordDataset('/path/to/TF_FILES', '/path/to/TF_SCHEMA_FILE', shuffle=ds.Shuffle.FILES)
            >>> data = data.device_que()
            >>> data.send()
            >>> time.sleep(0.1)
            >>> data.stop_send()
        """
        return TransferDataset(self, send_epoch_end, create_data_info_queue, queue_name)

    @check_save
    def save(self, file_name, num_files=1, file_type='mindrecord'):
        """
        Save the dynamic data processed by the dataset pipeline in common dataset format.
        Supported dataset formats: ``'mindrecord'`` only. And you can use
        :class:`mindspore.dataset.MindDataset` API to read the saved file(s).

        Implicit type casting exists when saving data as ``'mindrecord'`` . The transform table shows how to do
        type casting.

        .. list-table:: Implicit Type Casting when Saving as `mindrecord`
           :widths: 25 25 50
           :header-rows: 1

           * - Type in `dataset`
             - Type in `mindrecord`
             - Details
           * - bool
             - int32
             - transform to int32
           * - int8
             - int32
             -
           * - uint8
             - int32
             -
           * - int16
             - int32
             -
           * - uint16
             - int32
             -
           * - int32
             - int32
             -
           * - uint32
             - int64
             -
           * - int64
             - int64
             -
           * - uint64
             - int64
             - Maybe reverse
           * - float16
             - float32
             -
           * - float32
             - float32
             -
           * - float64
             - float64
             -
           * - string
             - string
             - Multi-dimensional string not supported
           * - bytes
             - bytes
             - Multi-dimensional bytes not supported

        Note:
            1. To save the samples in order, set dataset's `shuffle` to ``False`` and `num_files` to ``1``.
            2. Before calling the function, do not use batch operation, repeat operation or data augmentation operations
               with random attribute in map operation.
            3. When array dimension is variable, one-dimensional arrays or
               multidimensional arrays with variable dimension 0 are supported.
            4. MindRecord does not support multidimensional string or multidimensional bytes.

        Args:
            file_name (str): Path to dataset file.
            num_files (int, optional): Number of dataset files. Default: ``1`` .
            file_type (str, optional): Dataset format. Default: ``'mindrecord'`` .

        Examples:
            >>> import mindspore.dataset as ds
            >>> import numpy as np
            >>>
            >>> def generator_1d():
            ...     for i in range(10):
            ...         yield (np.array([i]),)
            >>>
            >>> # apply dataset operations
            >>> d1 = ds.GeneratorDataset(generator_1d, ["data"], shuffle=False)
            >>> d1.save('/path/to/save_file')
        """
        if _get_enc_key() is not None and num_files > 1:
            raise RuntimeError("When encode mode is enabled, " +
                               "the automatic sharding function is unavailable.")

        ir_tree, api_tree = self.create_ir_tree()

        runtime_context = cde.PythonRuntimeContext()
        runtime_context.Init()
        consumer = cde.PythonSaveToDisk(file_name, num_files, file_type)
        consumer.Init(ir_tree)
        runtime_context.AssignConsumer(consumer)

        consumer.Save()

        if _get_enc_key() is not None:
            encrypt(file_name, _get_enc_key(), _get_enc_mode())
            encrypt(file_name + ".db", _get_enc_key(), _get_enc_mode())

        _set_dataset_permissions(file_name, num_files)
        del api_tree

    @check_tuple_iterator
    def create_tuple_iterator(self, columns=None, num_epochs=-1, output_numpy=False, do_copy=False):
        """
        Create an iterator over the dataset that yields samples of type list, whose elements are
        the data for each column.

        Args:
            columns (list[str], optional): Specify the output columns and the order.
                Default: ``None``, keep all the output columns and their original order.
            num_epochs (int, optional): The number of epochs to iterate over the entire dataset.
                Default: ``-1`` , the dataset can be iterated indefinitely.
            output_numpy (bool, optional): Whether to keep the output data as NumPy ndarray, or
                convert non-string data to :class:`mindspore.Tensor`. Default: ``False`` .
            do_copy (bool, optional): Whether to copy the data when converting output to Tensor,
                or reuse the buffer for better performance, only works when `output_numpy` is ``False`` .
                Default: ``False`` .

        Returns:
            Iterator, a dataset iterator that yields samples of type list.

        Examples:
            >>> import mindspore.dataset as ds
            >>>
            >>> dataset = ds.GeneratorDataset([i for i in range(10)], "data")
            >>> num_epochs = 3
            >>> iterator = dataset.create_tuple_iterator(num_epochs=num_epochs)
            >>> for epoch in range(num_epochs):
            ...     for item in iterator:
            ...         # output is of type tuple
            ...         print(type(item))
            ...         break
            ...     break
            <class 'list'>
        """
        if output_numpy is None:
            output_numpy = False

        if Dataset._noop_mode():
            return DummyIterator(self, 'tuple', output_numpy)
        return TupleIterator(self, columns, num_epochs, output_numpy, do_copy)

    @check_dict_iterator
    def create_dict_iterator(self, num_epochs=-1, output_numpy=False, do_copy=False):
        """
        Create an iterator over the dataset that yields samples of type dict,
        while the key is the column name and the value is the data.

        Args:
            num_epochs (int, optional): The number of epochs to iterate over the entire dataset.
                Default: ``-1`` , the dataset can be iterated indefinitely.
            output_numpy (bool, optional): Whether to keep the output data as NumPy ndarray, or
                convert non-string data to :class:`mindspore.Tensor`. Default: ``False`` .
            do_copy (bool, optional): Whether to copy the data when converting output to Tensor,
                or reuse the buffer for better performance, only works when `output_numpy` is ``False`` .
                Default: ``False`` .

        Returns:
            Iterator, a dataset iterator that yields samples of type dict.

        Examples:
            >>> import mindspore.dataset as ds
            >>>
            >>> dataset = ds.GeneratorDataset([i for i in range(10)], "data")
            >>> num_epochs = 3
            >>> iterator = dataset.create_dict_iterator(num_epochs=num_epochs)
            >>> for epoch in range(num_epochs):
            ...     for item in iterator:
            ...         # output is of type dict
            ...         print(type(item))
            ...         break
            ...     break
            <class 'dict'>
        """
        if output_numpy is None:
            output_numpy = False

        if Dataset._noop_mode():
            return DummyIterator(self, 'dict', output_numpy)
        return DictIterator(self, num_epochs, output_numpy, do_copy)

    def __iter__(self):
        """Create an iterator over the dataset."""
        return self.create_tuple_iterator(num_epochs=1)

    @property
    def input_indexs(self):
        """
        Get the column index, which represents the corresponding relationship between the data column order
        and the network when using the sink mode.

        Returns:
            int, tuple of the input index information.

        Examples:
            >>> import mindspore.dataset as ds
            >>> dataset = ds.GeneratorDataset([i for i in range(10)], "column1")
            >>> # set input_indexs
            >>> dataset.input_indexs = 10
            >>> print(dataset.input_indexs)
            10
        """
        if self._input_indexs != ():
            return self._input_indexs

        # find input_indexes of children
        children_input_index = [child.input_indexs for child in self.children]

        # in case of more than one child, return the first input_indexes
        for cix in children_input_index:
            if cix != ():
                return cix

        # if all children's input_indexes are () or the node is a leaf
        return self._input_indexs

    @input_indexs.setter
    def input_indexs(self, value):
        self._input_indexs = value

    def copy_batch_size(self, value):
        self._batch_size = value

    def _init_tree_getters(self, getter_mode=True):
        """
        Get pipeline information.

        Args:
            getter_mode (bool, optional): Whether to build IR tree in pull mode. Default: ``True``.
        """
        ir_tree, api_tree = self.create_ir_tree(getter_mode)

        runtime_context = cde.PythonRuntimeContext()
        runtime_context.Init()
        getter = cde.TreeGetters()
        getter.Init(ir_tree)
        runtime_context.AssignConsumer(getter)
        return getter, runtime_context, api_tree

    def __init_size_getter(self):
        """
        Get pipeline information.
        """
        ir_tree, api_tree = self.create_ir_tree()

        runtime_context = cde.PythonRuntimeContext()
        runtime_context.Init()
        getter = cde.DatasetSizeGetters()
        getter.Init(ir_tree)
        runtime_context.AssignConsumer(getter)
        return getter, runtime_context, api_tree

    def get_col_names(self):
        """
        Return the names of the columns in dataset.

        Returns:
            list, list of column names in the dataset.

        Examples:
            >>> import mindspore.dataset as ds
            >>> dataset = ds.GeneratorDataset([i for i in range(10)], "column1")
            >>> col_names = dataset.get_col_names()
            >>> print(col_names)
            ['column1']

        """
        if self._col_names is None:
            runtime_getter = self._init_tree_getters()
            self._col_names = runtime_getter[0].GetColumnNames()

        return self._col_names

    @check_output_shape
    @_cleanup_the_iterators_if_created
    def output_shapes(self, estimate=False):
        """
        Get the shapes of output data.

        Args:
            estimate (bool, optional): If `estimate` is ``False`` , will return the shapes of first data row.
                Otherwise, will iterate the whole dataset and return the estimated shapes of data row,
                where dynamic shape is marked as None (used in dynamic data shapes scenario).
                Default: ``False`` .

        Returns:
            list, list of shapes of each column.

        Examples:
            >>> import mindspore.dataset as ds
            >>> import numpy as np
            >>>
            >>> def generator1():
            ...     for i in range(1, 100):
            ...         yield np.ones((16, 83, 83)), np.array([i])
            >>>
            >>> dataset = ds.GeneratorDataset(generator1, ["data1", "data2"])
            >>> output_shapes = dataset.output_shapes()
            >>> print(output_shapes)
            [[16, 83, 83], [1]]
        """
        # cache single shape
        if not estimate and self.saved_output_shapes is not None:
            return self.saved_output_shapes
        # cache estimate shape
        if estimate and self.estimated_output_shapes is not None:
            return self.estimated_output_shapes

        # We have a hang problem when two-level pipeline with multiprocessing, we need to extend the life cycle
        # of runtime_context. We found this hang problem only occur on output_types and output_shapes.
        runtime_getter = self._init_tree_getters()
        self.runtime_context = runtime_getter[1]
        api_tree = runtime_getter[2]
        output_shapes = runtime_getter[0].GetOutputShapes(estimate)
        del api_tree
        # Need to terminate the runtime context to avoid the occasional hang problem for
        # Python (with multiprocessing enabled) in sink mode.
        self.runtime_context.Terminate()
        del self.runtime_context

        if estimate:
            self.estimated_output_shapes = output_shapes
        else:
            self.saved_output_shapes = output_shapes
        return output_shapes

    @_cleanup_the_iterators_if_created
    def output_types(self):
        """
        Get the types of output data.

        Returns:
            list, list of data types.

        Examples:
            >>> import mindspore.dataset as ds
            >>> import numpy as np
            >>>
            >>> def generator1():
            ...     for i in range(1, 100):
            ...         yield np.ones((16, 83, 83)).astype(np.float32), np.array([i]).astype(np.int32)
            >>>
            >>> dataset = ds.GeneratorDataset(generator1, ["data1", "data2"])
            >>> output_types = dataset.output_types()
            >>> print(output_types)
            [dtype('float32'), dtype('int32')]
        """
        if self.saved_output_types is None:
            runtime_getter = self._init_tree_getters()
            # We have a hang problem when two-level pipeline with multiprocessing, we need to extend the life cycle
            # of runtime_context. We found this hang problem only occur on output_types and output_shapes.
            self.runtime_context = runtime_getter[1]
            api_tree = runtime_getter[2]
            self.saved_output_types = runtime_getter[0].GetOutputTypes()
            del api_tree
            # Need to terminate the runtime context to avoid the occasional hang problem for
            # Python (with multiprocessing enabled) in sink mode.
            self.runtime_context.Terminate()
            del self.runtime_context
        return self.saved_output_types

    @_cleanup_the_iterators_if_created
    def get_dataset_size(self):
        """
        Return the number of batches in an epoch.

        Returns:
            int, number of batches.

        Examples:
            >>> import mindspore.dataset as ds
            >>> import numpy as np
            >>>
            >>> # A generator return 66 samples
            >>> def generator1():
            ...     for i in range(66):
            ...         yield np.ones((16, 83, 83)), np.array([i])
            >>>
            >>> dataset = ds.GeneratorDataset(generator1, ["data1", "data2"])
            >>> dataset_size = dataset.get_dataset_size()
            >>> print(dataset_size)
            66
        """
        if self.dataset_size is None:
            runtime_getter = self.__init_size_getter()
            self.dataset_size = runtime_getter[0].GetDatasetSize(False)
            if self.dataset_size == 0:
                logger.warning("Got 0 sample from dataset pipeline, check if drop all data or load dataset fail.")

        return self.dataset_size

    def num_classes(self):
        """
        Get the number of classes in a dataset.

        Returns:
            int, number of classes.

        Examples:
            >>> import mindspore.dataset as ds
            >>> # Read image files
            >>> image_folder_dataset_dir = "/path/to/image_folder_dataset_directory"
            >>> dataset = ds.ImageFolderDataset(dataset_dir=image_folder_dataset_dir)
            >>> # Check how many classes exist in image folder
            >>> num_classes = dataset.num_classes()
        """
        if self._num_classes is None:
            runtime_getter = self._init_tree_getters()
            self._num_classes = runtime_getter[0].GetNumClasses()

        if self._num_classes == -1:
            return None
        return self._num_classes

    def get_sync_notifiers(self):
        if self.children:
            return self.children[0].get_sync_notifiers()
        return {}

    def disable_sync(self):
        if self.children:
            return self.children[0].disable_sync()
        return {}

    def is_sync(self):
        if self.children:
            return self.children[0].is_sync()
        return False

    @check_sync_update
    def sync_update(self, condition_name, num_batch=None, data=None):
        """
        Release a blocking condition and trigger callback with given data.

        Args:
            condition_name (str): The condition name that is used to toggle sending next row.
            num_batch (Union[int, None], optional): The number of batches (rows) that are released.
                When `num_batch` is ``None``, it will default to the number specified by the
                `sync_wait` operation. Default: ``None``.
            data (Any, optional): The data passed to the callback, user defined. Default: ``None``.

        Examples:
            >>> import numpy as np
            >>> import mindspore.dataset as ds
            >>>
            >>> def gen():
            ...     for i in range(100):
            ...         yield (np.array(i),)
            >>>
            >>> class Augment:
            ...     def __init__(self, loss):
            ...         self.loss = loss
            ...
            ...     def preprocess(self, input_):
            ...         return input_
            ...
            ...     def update(self, data):
            ...         self.loss = data["loss"]
            >>>
            >>> batch_size = 10
            >>> dataset = ds.GeneratorDataset(gen, column_names=["input"])
            >>> aug = Augment(0)
            >>> dataset = dataset.sync_wait(condition_name='', num_batch=1)
            >>> dataset = dataset.map(input_columns=["input"], operations=[aug.preprocess])
            >>> dataset = dataset.batch(batch_size)
            >>>
            >>> count = 0
            >>> for data in dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
            ...     count += 1
            ...     data = {"loss": count}
            ...     dataset.sync_update(condition_name="", data=data)
        """
        if (not isinstance(num_batch, int) and num_batch is not None) or \
                (isinstance(num_batch, int) and num_batch <= 0):
            # throwing exception, disable all sync_wait in pipeline
            self.disable_sync()
            raise RuntimeError(f"Sync_update batch size can only be positive integer, got: {num_batch}.")
        notifiers_dict = self.get_sync_notifiers()
        if not isinstance(condition_name, str):
            raise TypeError(f"Argument condition_name with value {condition_name} is not of type str, " +
                            f"but got {type(condition_name)}.")
        if condition_name not in notifiers_dict:
            # throwing exception, disable all sync_wait in pipeline
            self.disable_sync()
            raise RuntimeError("Condition name not found.")
        if num_batch is not None:
            num_batch *= self.get_batch_size()
        notifiers_dict[condition_name](num_batch, data)

    def get_batch_size(self):
        """
        Return the size of batch.

        Returns:
            int, the batch size of data.

        Examples:
            >>> import mindspore.dataset as ds
            >>> dataset = ds.GeneratorDataset([i for i in range(10)], "column1")
            >>> dataset = dataset.batch(2)
            >>> batch_size = dataset.get_batch_size()
            >>> print(batch_size)
            2
        """
        if self._batch_size is None:
            runtime_getter = self._init_tree_getters()
            self._batch_size = runtime_getter[0].GetBatchSize()
        if self._batch_size is None:
            self._batch_size = 1
        return self._batch_size

    def get_repeat_count(self):
        """
        Get the replication times in RepeatDataset. Default: ``1`` .

        Returns:
            int, the count of repeat.

        Examples:
            >>> import mindspore.dataset as ds
            >>> dataset = ds.GeneratorDataset([i for i in range(10)], "column1")
            >>> dataset = dataset.repeat(5)
            >>> repeat_count = dataset.get_repeat_count()
            >>> print(repeat_count)
            5
        """
        if self._repeat_count is None:
            runtime_getter = self._init_tree_getters()
            self._repeat_count = runtime_getter[0].GetRepeatCount()
        if self._repeat_count is None:
            self._repeat_count = 1
        return self._repeat_count

    def get_class_indexing(self):
        """
        Get the mapping dictionary from category names to category indexes.

        This dictionary can be used to look up which category name corresponds to a particular category index.

        Returns:
            Dict[str, int], the mappings from category names to category indexes.

        Examples:
            >>> import mindspore.dataset as ds
            >>> # Read image files
            >>> image_folder_dataset_dir = "/path/to/image_folder_dataset_directory"
            >>> dataset = ds.ImageFolderDataset(dataset_dir=image_folder_dataset_dir)
            >>> # Check how many classes exist in image folder
            >>> class_indexing = dataset.get_class_indexing()
        """
        if self.children:
            return self.children[0].get_class_indexing()
        return {}

    def reset(self):
        """
        Reset the dataset for next epoch.

        Examples:
            >>> import mindspore.dataset as ds
            >>> mind_dataset_dir = ["/path/to/mind_dataset_file"]
            >>> dataset = ds.MindDataset(dataset_files=mind_dataset_dir)
            >>> for _ in range(5):
            ...     num_iter = 0
            ...     for data in dataset.create_tuple_iterator(num_epochs=1, output_numpy=True):
            ...         num_iter += 1
            ...     dataset.reset()
        """

    def is_shuffled(self):
        """Returns True if the dataset or its children is shuffled."""
        for input_dataset in self.children:
            if input_dataset.is_shuffled():
                return True

        return False

    def is_sharded(self):
        """Returns True if the dataset or its children is sharded."""
        for input_dataset in self.children:
            if input_dataset.is_sharded():
                return True

        return False

    def parse(self, children=None):
        raise NotImplementedError("Dataset has to implement parse method.")

    def __len__(self):
        """
        Get the length of dataset.

        Returns:
            int, the length of dataset.
        """
        return self.get_dataset_size()


    def pre_parse(self, getter_mode):
        if getter_mode:
            if hasattr(self, "python_multiprocessing"):
                self.python_multiprocessing = False
            if hasattr(self, "num_parallel_workers"):
                self.num_parallel_workers = 1

    def post_parse(self, ir_node):
        if self.cache:
            ir_node = ir_node.set_cache_client(self.cache.cache_client)
        if self.num_parallel_workers:
            ir_node = ir_node.set_num_workers(self.num_parallel_workers)

        return ir_node

    def set_init_step(self, init_step):
        self._global_step = init_step

    def get_init_step(self):
        if self._global_step is not None:
            return self._global_step
        if len(self.children) == 1:
            return self.children[0].get_init_step()
        # When there are multiple children, we cannot tell from which child to get the initial step,
        # so we initialize from the beginning
        return 0

    @staticmethod
    def _send(tensor_list, dst_list, group):
        """Sending data in two steps"""
        # get meta by calculate list[Tensor]
        meta_info = []
        meta_info.append(len(tensor_list))                 # num of tensor_list
        for t in tensor_list:
            meta_info.append(t.ndim)                       # ndim of tensor_list[i]
            for shape in t.shape:
                meta_info.append(shape)                    # tensor_list[i].shape[0], ...
            if t.dtype not in mstype_to_int:
                raise RuntimeError(f"Tensor of dtype: {t.dtype} is not supported to send.")
            meta_info.append(mstype_to_int[t.dtype])       # tensor_list[i].dtype

        if len(meta_info) >= MAX_METADATA_LENGTH:
            raise RuntimeError("The metadata for sending data is too large.")

        meta_info = meta_info + [0] * (MAX_METADATA_LENGTH - len(meta_info))
        meta = Tensor(meta_info, dtype=ms.int64)

        for dst_rank in dst_list:
            ## first: send the meta to dst
            ms.mint.distributed.send(meta, dst_rank, group)

            ## second: send the tensor to dst
            for t in tensor_list:
                ms.mint.distributed.send(t, dst_rank, group)

    @check_send
    def send(self, tensor=None, dst=0, group=None):
        """
        The dataset communication interface sends data to the target `Dataset`,
        which can be received through :class:`mindspore.dataset.Dataset.recv`.

        The send operation only send data once.

        Note:
            This is an experimental API that is subject to change or deletion.

        Args:
            tensor (Union[Tensor, list[Tensor]], optional): List of the Tensor(s) to send. Default: ``None`` ,
                retrieve data from the current dataset and send it.
            dst (Union[int, list[int]], optional): List of the dst rank id(s) to send. It cannot be the
                current Rank ID or contain the current Rank ID. Default: ``0`` ,
                which indicates send data to dst rank 0.
            group (str, optional): The communication group to work on. The group is created
                by :func:`mindspore.mint.distributed.init_process_group` or
                :func:`mindspore.mint.distributed.new_group` . Default: ``None``,
                which indicates ``GlobalComm.WORLD_COMM_GROUP`` .

        Examples:
            >>> import mindspore as ms
            >>> from mindspore.mint.distributed import init_process_group
            >>> from mindspore.mint.distributed import get_rank
            >>> from mindspore import Tensor
            >>> import mindspore.dataset as ds
            >>> import numpy as np
            >>>
            >>> # Launch 8 processes by msrun --worker_num=8 --local_worker_num=8 script.py
            >>> init_process_group()
            >>> this_rank = get_rank()
            >>>
            >>> # Create a dataset with 3 columns
            >>> input_columns = ["column1", "column2", "column3"]
            >>> dataset = ds.GeneratorDataset([(1, 2, 3), (3, 4, 5), (5, 6, 7)], column_names=input_columns)
            >>>
            >>> # Send a data from the current dataset to the dst rank: 0
            >>> if this_rank == 2:
            >>>     dataset.send()
            >>> if this_rank == 0:
            >>>     data = dataset.recv(2)
            >>>
            >>> # Send the data "send_tensor" to the dst rank: 7
            >>> if this_rank == 0:
            >>>     send_tensor = Tensor(np.zeros([2, 2, 3]), ms.float32)
            >>>     dataset.send(send_tensor, 7)
            >>> if this_rank == 7:
            >>>     recv_tensor = dataset.recv(0)
            >>>
            >>> # Send the list of data to dst rank [0, 2, 4, 6]
            >>> if this_rank in [1, 3, 5, 7]:
            >>>     send_data = Tensor(np.zeros([2, 2, 3]), ms.float32)
            >>>     send_label = Tensor(np.zeros([3,]), ms.bool)
            >>>     dataset.send([send_data, send_label], [0, 2, 4, 6])
            >>> if this_rank in [0, 2, 4, 6]:
            >>>     recv_tensors = dataset.recv([1, 3, 5, 7])
        """

        # cast the dst to list[int]
        dst_list = dst
        if isinstance(dst, int):
            dst_list = [dst]

        data_to_send = None
        if tensor is not None:
            # the input tensor is not None
            tensor_list = tensor
            if isinstance(tensor, Tensor):
                tensor_list = [tensor]
            data_to_send = tensor_list
        else:
            # get data from dataset_iter
            if self._dataset_iter is None:
                self._dataset_iter = self.create_tuple_iterator()

            while True:
                try:
                    data_to_send = next(self._dataset_iter)
                except StopIteration:
                    continue
                break  # already got data

            # check the type of the data from dataset
            for index, item in enumerate(data_to_send):
                if not isinstance(item, Tensor):
                    raise RuntimeError(f"The data column at index: {index} is not Tensor which is not " \
                                        "supported to send. You can remove it using the dataset.project operation.")
        # check the dtype of the Tensor
        for t in data_to_send:
            if t.dtype not in mstype_to_int:
                raise RuntimeError(f"Tensor of dtype: {t.dtype} is not supported to send.")

        return Dataset._send(data_to_send, dst_list, group)

    @staticmethod
    def _recv(src_rank, group):
        """Receiving data in two steps"""
        ## first: get meta
        meta = Tensor(np.zeros([MAX_METADATA_LENGTH,]).astype(np.int64))
        out = ms.mint.distributed.recv(meta, src_rank, group)
        if out != 0:
            raise RuntimeError("Receive meta failed by mint.recv(...)")

        meta_index = 0
        tensor_list = []
        tensor_list_size = int(meta[0])
        meta_index += 1
        for _ in range(tensor_list_size):                 # num of tensor_list
            ndim_tensor = int(meta[meta_index])           # ndim of tensor_list[i]
            meta_index += 1
            shape = []
            for _ in range(ndim_tensor):
                shape.append(int(meta[meta_index]))       # tensor_list[i].shape[0]
                meta_index += 1

            dtype = int_to_mstype[int(meta[meta_index])]  # dtype
            meta_index += 1
            tensor_list.append(Tensor(np.zeros(shape), dtype=dtype))

        ## second: get data
        for item in tensor_list:
            out = ms.mint.distributed.recv(item, src_rank, group)
            if out != 0:
                raise RuntimeError("Receive data failed by mint.recv(...)")

        flatten_tensor_list = flatten_single_lists(tensor_list)
        if len(flatten_tensor_list) == 1:
            return flatten_tensor_list[0]

        return flatten_tensor_list

    @check_recv
    def recv(self, src=0, group=None):
        """
        The dataset communication interface receives data sent by the source `Dataset`
        using :class:`mindspore.dataset.Dataset.send` .

        Each call to the recv operation only receives data once.

        Note:
            This is an experimental API that is subject to change or deletion.

        Args:
            src (Union[int, list[int]], optional): List of the src rank id(s) to receive. If the Rank ID of the
                current process is specified, data will be obtained directly from itself. Default: ``0`` ,
                which indicates receive data from src rank 0.
            group (str, optional): The communication group to work on. The group is created
                by :func:`mindspore.mint.distributed.init_process_group` or
                :func:`mindspore.mint.distributed.new_group` . Default: ``None``,
                which indicates ``GlobalComm.WORLD_COMM_GROUP`` .

        Returns:
            Union[Tensor, list[Tensor]], List of the Tensor(s) received.

        Examples:
            >>> from mindspore.mint.distributed import init_process_group
            >>> from mindspore.mint.distributed import get_rank
            >>> import mindspore.dataset as ds
            >>>
            >>> # Launch 8 processes by msrun --worker_num=8 --local_worker_num=8 script.py
            >>> init_process_group()
            >>> this_rank = get_rank()
            >>>
            >>> # Create a dataset with 3 columns
            >>> input_columns = ["column1", "column2", "column3"]
            >>> dataset = ds.GeneratorDataset([(1, 2, 3), (3, 4, 5), (5, 6, 7)], column_names=input_columns)
            >>>
            >>> # Send data from rank: 0 to rank: [1, 2, 3, 4, 5, 6, 7]
            >>> if this_rank == 0:
            >>>     dataset.send(dst=[1, 2, 3, 4, 5, 6, 7])
            >>> if this_rank in [1, 2, 3, 4, 5, 6, 7]:
            >>>     data1 = dataset.recv()
            >>>
            >>> # Receive a data from the current dataset
            >>> data2 = dataset.recv(src=get_rank())
            >>>
            >>> # Send data from rank: [1, 2, 3, 4, 5, 6, 7] to rank: 0
            >>> if this_rank in [1, 2, 3, 4, 5, 6, 7]:
            >>>     dataset.send()
            >>> if this_rank == 0:
            >>>     data3 = dataset.recv(src=[1, 2, 3, 4, 5, 6, 7])
        """

        # cast the src to list[int]
        src_list = src
        if isinstance(src, int):
            src_list = [src]

        current_rank = ms.mint.distributed.get_rank()

        # get data from src
        all_tensor_list = []
        for src_rank in src_list:
            # get data from current dataset
            if src_rank == current_rank:
                if self._dataset_iter is None:
                    self._dataset_iter = self.create_tuple_iterator()

                while True:
                    try:
                        data_to_return = next(self._dataset_iter)
                    except StopIteration:
                        continue
                    break  # already got data

                # convert the dict to list
                all_tensor_list.append(data_to_return)
            else:
                # get data from remote dataset
                all_tensor_list.append(Dataset._recv(src_rank, group))

        flatten_all_tensor_list = flatten_single_lists(all_tensor_list)
        if len(flatten_all_tensor_list) == 1:
            return flatten_all_tensor_list[0]

        return flatten_all_tensor_list


class VisionBaseDataset(Dataset):
    """
    Abstract class to represent a vision source dataset which produces content to the data pipeline.
    """

    def __init__(self, children=None, num_parallel_workers=None, cache=None):
        super().__init__(children=children, num_parallel_workers=num_parallel_workers, cache=cache)

    def parse(self, children=None):
        raise NotImplementedError("Dataset has to implement parse method.")


class TextBaseDataset(Dataset):
    """
    Abstract class to represent a text source dataset which produces content to the data pipeline.
    """

    def __init__(self, children=None, num_parallel_workers=None, cache=None):
        super().__init__(children=children, num_parallel_workers=num_parallel_workers, cache=cache)

    def parse(self, children=None):
        raise NotImplementedError("Dataset has to implement parse method.")

    def build_vocab(self, columns, freq_range, top_k, special_tokens, special_first):
        """
        Function to create a Vocab from source dataset.
        Desired source dataset is a text type dataset.

        Build a vocab from a dataset. This would collect all the unique words in a dataset and return a vocab
        which contains top_k most frequent words (if top_k is specified).

        Note:
            mindspore.dataset.Dataset.build_vocab is deprecated from version 2.0
            and will be removed in a future version. Use mindspore.dataset.text.Vocab.from_dataset instead.

        Args:
            columns(Union[str, list[str]]): Column names to get words from.
            freq_range(tuple[int]): A tuple of integers (min_frequency, max_frequency). Words within the frequency
                range will be stored.
                Naturally 0 <= min_frequency <= max_frequency <= total_words. min_frequency/max_frequency
                can be set to default, which corresponds to 0/total_words separately.
            top_k(int): Number of words to be built into vocab. top_k most frequent words are
                taken. The top_k is taken after freq_range. If not enough top_k, all words will be taken
            special_tokens(list[str]): A list of strings, each one is a special token.
            special_first(bool): Whether special_tokens will be prepended/appended to vocab, If special_tokens
                is specified and special_first is set to default, special_tokens will be prepended.

        Returns:
            Vocab, vocab built from the dataset.
        """
        warnings.warn("mindspore.dataset.Dataset.build_vocab is deprecated from version 2.0 "
                      "and will be removed in a future version. "
                      "Use mindspore.dataset.text.Vocab.from_dataset instead.", DeprecationWarning)

    def build_sentencepiece_vocab(self, columns, vocab_size, character_coverage, model_type, params):
        """
        Function to create a SentencePieceVocab from source dataset.
        Desired source dataset is a text type dataset.

        Note:
            mindspore.dataset.Dataset.build_sentencepiece_vocab is deprecated from version 2.0
            and will be removed in a future version. Use mindspore.dataset.text.SentencePieceVocab.from_dataset instead.

        Args:
            columns(list[str]): Column names to get words from.
            vocab_size(int): Vocabulary size.
            character_coverage(float): Percentage of characters covered by the model, must be between
                0.98 and 1.0 Good defaults are: 0.9995 for languages with rich character sets like
                Japanese or Chinese character sets, and 1.0 for other languages with small character sets
                like English or Latin.
            model_type(SentencePieceModel): Model type. Choose from unigram (default), bpe, char, or word.
                The input sentence must be pre-tokenized when using word type.
            params(dict): Any extra optional parameters of sentencepiece library according to your raw data

        Returns:
            SentencePieceVocab, vocab built from the dataset.
        """
        warnings.warn("mindspore.dataset.Dataset.build_sentencepiece_vocab is deprecated from version 2.0 "
                      "and will be removed in a future version. "
                      "Use mindspore.dataset.text.SentencePieceVocab.from_dataset instead.", DeprecationWarning)

    def _build_vocab(self, columns, freq_range, top_k, special_tokens, special_first):
        """
        Function to create a Vocab from source dataset.
        Desired source dataset is a text type dataset.

        Build a vocab from a dataset. This would collect all the unique words in a dataset and return a vocab
        which contains top_k most frequent words (if top_k is specified).

        Args:
            columns(Union[str, list[str]]): Column names to get words from.
            freq_range(tuple[int]): A tuple of integers (min_frequency, max_frequency). Words within the frequency
                range will be stored.
                Naturally 0 <= min_frequency <= max_frequency <= total_words. min_frequency/max_frequency
                can be set to default, which corresponds to 0/total_words separately.
            top_k(int): Number of words to be built into vocab. top_k most frequent words are
                taken. The top_k is taken after freq_range. If not enough top_k, all words will be taken
            special_tokens(list[str]): A list of strings, each one is a special token.
            special_first(bool): Whether special_tokens will be prepended/appended to vocab, If special_tokens
                is specified and special_first is set to default, special_tokens will be prepended.

        Returns:
            Vocab, vocab built from the dataset.
        """
        vocab = cde.Vocab()
        columns = replace_none(columns, [])
        if not isinstance(columns, list):
            columns = [columns]

        freq_range = replace_none(freq_range, (0, 9223372036854775807))
        if freq_range[0] is None:
            freq_range = (0, freq_range[1])
        if freq_range[1] is None:
            freq_range = (freq_range[0], 9223372036854775807)
        special_tokens = replace_none(special_tokens, [])
        top_k = replace_none(top_k, 9223372036854775807)

        ir_tree, api_tree = self.create_ir_tree()

        # vocab node
        vocab_node = cde.BuildVocabNode(ir_tree, vocab, columns, freq_range, top_k, special_tokens, special_first)

        runtime_context = cde.PythonRuntimeContext()
        runtime_context.Init()

        # build vocab
        consumer = cde.PythonBuildVocabConsumer()
        consumer.Init(vocab_node)
        runtime_context.AssignConsumer(consumer)

        consumer.Start()
        del api_tree

        return vocab

    def _build_sentencepiece_vocab(self, columns, vocab_size, character_coverage, model_type, params):
        """
        Function to create a SentencePieceVocab from source dataset.
        Desired source dataset is a text type dataset.

        Args:
            columns(list[str]): Column names to get words from.
            vocab_size(int): Vocabulary size.
            character_coverage(float): Percentage of characters covered by the model, must be between
                0.98 and 1.0 Good defaults are: 0.9995 for languages with rich character sets like
                Japanese or Chinese character sets, and 1.0 for other languages with small character sets
                like English or Latin.
            model_type(SentencePieceModel): Model type. Choose from unigram (default), bpe, char, or word.
                The input sentence must be pre-tokenized when using word type.
            params(dict): Any extra optional parameters of sentencepiece library according to your raw data

        Returns:
            SentencePieceVocab, vocab built from the dataset.
        """
        if not isinstance(model_type, SentencePieceModel):
            raise TypeError(f"Argument model_type with value {model_type} is not of type SentencePieceModel, " +
                            f"but got {type(model_type)}.")
        model_type = DE_C_INTER_SENTENCEPIECE_MODE[model_type]
        vocab = cde.SentencePieceVocab()

        ir_tree, api_tree = self.create_ir_tree()

        # vocab node
        vocab_node = cde.BuildSentenceVocabNode(ir_tree, vocab, columns, vocab_size, character_coverage, model_type,
                                                params)

        runtime_context = cde.PythonRuntimeContext()
        runtime_context.Init()

        # build vocab
        consumer = cde.PythonBuildVocabConsumer()
        consumer.Init(vocab_node)
        runtime_context.AssignConsumer(consumer)

        consumer.Start()
        del api_tree

        return vocab


class AudioBaseDataset(Dataset):
    """
    Abstract class to represent a audio source dataset which produces content to the data pipeline.
    """

    def __init__(self, children=None, num_parallel_workers=None, cache=None):
        super().__init__(children=children, num_parallel_workers=num_parallel_workers, cache=cache)

    def parse(self, children=None):
        raise NotImplementedError("Dataset has to implement parse method.")


class UnionBaseDataset(VisionBaseDataset, TextBaseDataset, AudioBaseDataset):
    """
    Abstract class to represent a union source dataset which produces content to the data pipeline.
    """

    def __init__(self, children=None, num_parallel_workers=None, cache=None):
        super().__init__(children=children, num_parallel_workers=num_parallel_workers, cache=cache)

    def parse(self, children=None):
        raise NotImplementedError("Dataset has to implement parse method.")


class SourceDataset(Dataset):
    """
    Abstract class to represent a source dataset which produces content to the data pipeline.
    """

    def __init__(self, num_parallel_workers=None, num_samples=None, shuffle=True, num_shards=None, shard_id=None,
                 cache=None):
        super().__init__(num_parallel_workers=num_parallel_workers, cache=cache)
        self.num_samples = replace_none(num_samples, 0)
        self.num_shards = replace_none(num_shards, 1)
        self.shard_id = replace_none(shard_id, 0)

        if shuffle is not None and not isinstance(shuffle, (bool, Shuffle)):
            raise TypeError("shuffle must be of boolean or enum of 'Shuffle' values like 'Shuffle.ADAPTIVE' or "
                            "'Shuffle.GLOBAL' or 'Shuffle.PARTIAL' or 'Shuffle.FILES' or 'Shuffle.INFILE'.")

        self.shuffle_flag = 5  # Adaptive shuffle
        if not isinstance(shuffle, Shuffle):
            if shuffle is None or shuffle:
                self.shuffle_flag = 2  # Global shuffle
            else:
                self.shuffle_flag = 0  # No shuffle
        else:
            if shuffle == Shuffle.GLOBAL:
                self.shuffle_flag = 2  # Global shuffle
            elif shuffle == Shuffle.FILES:
                self.shuffle_flag = 1  # Files shuffle
            elif shuffle == Shuffle.INFILE:
                self.shuffle_flag = 3  # Infile shuffle
            elif shuffle == Shuffle.ADAPTIVE:
                self.shuffle_flag = 5
            elif shuffle == Shuffle.PARTIAL:
                self.shuffle_flag = 4

    def parse(self, children=None):
        raise NotImplementedError("Dataset has to implement parse method.")

    @staticmethod
    def _find_files(patterns):
        """
        Utility function to search for files with the given glob patterns.

        Args:
            patterns (Union[str, list[str]]): String or list of patterns to be searched.

        Returns:
            list, list of files.
        """

        if not isinstance(patterns, list):
            patterns = [patterns]

        file_list = []
        unmatched_patterns = []
        for pattern in patterns:
            matches = [match for match in glob.glob(pattern, recursive=True) if os.path.isfile(match)]

            if matches:
                file_list.extend(matches)
            else:
                unmatched_patterns.append(pattern)

        if unmatched_patterns:
            raise ValueError(f"The following patterns did not match any files: {unmatched_patterns}.")

        if file_list:  # not empty
            return file_list
        raise ValueError("The list of path names matching the patterns is empty.")

    def is_shuffled(self):
        return self.shuffle_flag > 0

    def is_sharded(self):
        if self.num_shards is not None:
            return self.num_shards > 1
        return False


class MappableDataset(SourceDataset):
    """
    Abstract class to represent a source dataset which supports use of samplers.
    """

    def parse(self, children=None):
        raise NotImplementedError("Dataset has to implement parse method.")

    def __init__(self, num_parallel_workers=None, sampler=None, num_samples=None, shuffle=None, num_shards=None,
                 shard_id=None, cache=None):
        if sampler is None:
            if shuffle is None or shuffle is True:
                shuffle = Shuffle.GLOBAL
            elif shuffle is False:
                shuffle = Shuffle.FALSE
        super().__init__(num_parallel_workers=num_parallel_workers, num_samples=num_samples, shuffle=shuffle,
                         num_shards=num_shards, shard_id=shard_id, cache=cache)
        self.sampler = samplers.select_sampler(num_samples, sampler, shuffle, num_shards, shard_id)

    def add_sampler(self, new_sampler):
        """
        Add a child sampler for the current dataset.

        Note:
            - If the sampler is added and it has a shuffle option, its value must be ``Shuffle.GLOBAL`` .
              Additionally, the original sampler's shuffle value cannot be ``Shuffle.PARTIAL`` .

        Args:
            new_sampler (Sampler): The child sampler to be added.

        Examples:
            >>> import mindspore.dataset as ds
            >>> dataset = ds.GeneratorDataset([i for i in range(10)], "column1")
            >>>
            >>> new_sampler = ds.DistributedSampler(10, 2)
            >>> dataset.add_sampler(new_sampler)
        """
        # Note: By adding a sampler, the sampled IDs will flow to the new_sampler
        # after first passing through the current samplers attached to this dataset.
        self.dataset_size = None

        if self.sampler is not None and self.sampler.get_shuffle_mode() == Shuffle.PARTIAL:
            raise RuntimeError("When multiple samplers are used, ensure that the shuffle of the current sampler "
                               "must not be Shuffle.PARTIAL.")

        if new_sampler.get_shuffle_mode() != Shuffle.GLOBAL and new_sampler.get_shuffle_mode() != Shuffle.FALSE:
            raise RuntimeError("When multiple samplers are used, ensure that the shuffle of the input sampler " +
                               f"must be Shuffle.FALSE or Shuffle.GLOBAL, but got: {new_sampler.get_shuffle_mode()}.")

        new_sampler.add_child(self.sampler)
        self.sampler = new_sampler

    def use_sampler(self, new_sampler):
        """
        Replace the last child sampler of the current dataset, leaving the parent sampler unchanged.

        Args:
            new_sampler (Sampler): The new sampler to replace with.

        Examples:
            >>> import mindspore.dataset as ds
            >>> dataset = ds.GeneratorDataset([i for i in range(10)], "column1")
            >>>
            >>> # use a DistributedSampler instead
            >>> new_sampler = ds.DistributedSampler(10, 2)
            >>> dataset.use_sampler(new_sampler)
        """
        if new_sampler is None:
            raise TypeError("Input sampler can not be None.")
        if not isinstance(new_sampler, (samplers.BuiltinSampler, samplers.Sampler)):
            raise TypeError("Input sampler is not an instance of a sampler.")
        self.dataset_size = None

        self.sampler = self.sampler.child_sampler
        self.add_sampler(new_sampler)

    def is_shuffled(self):
        return self.sampler.is_shuffled()

    def is_sharded(self):
        return self.sampler.is_sharded()

    @check_split
    def split(self, sizes, randomize=True):
        """
        Split the dataset into smaller, non-overlapping datasets.

        Args:
            sizes (Union[list[int], list[float]]): If a list of integers [s1, s2, …, sn] is
                provided, the dataset will be split into n datasets of size s1, size s2, …, size sn
                respectively. If the sum of all sizes does not equal the original dataset size, an
                error will occur.
                If a list of floats [f1, f2, …, fn] is provided, all floats must be between 0 and 1
                and must sum to 1, otherwise an error will occur. The dataset will be split into n
                Datasets of size round(f1*K), round(f2*K), …, round(fn*K) where K is the size of the
                original dataset.
                If after rounding:

                - Any size equals 0, an error will occur.
                - The sum of split sizes < K, the difference will be added to the first split.
                - The sum of split sizes > K, the difference will be removed from the first large
                  enough split such that it will have at least 1 row after removing the difference.

            randomize (bool, optional): Determines whether or not to split the data randomly. Default: ``True``.
                If ``True``, the data will be randomly split. Otherwise, each split will be created with
                consecutive rows from the dataset.

        Note:
            1. There is an optimized split function, which will be called automatically when the dataset
               that calls this function is a MappableDataset.
            2. Dataset should not be sharded if split is going to be called. Instead, create a
               :class:`mindspore.dataset.DistributedSampler` and specify a split to shard after splitting.
               If the dataset is sharded after a split, it is strongly recommended setting the same
               seed in each instance of execution, otherwise each shard may not be part of the same
               split (see Examples).
            3. It is strongly recommended to not shuffle the dataset, but set `randomize` to ``True`` instead.
               Shuffling the dataset may not be deterministic, which means the data in each split
               will be different in each epoch. Furthermore, if sharding occurs after split, each
               shard may not be part of the same split.

        Returns:
            Tuple[Dataset], a tuple of new datasets split from the original one.

        Raises:
            RuntimeError: If get_dataset_size returns None or is not supported for this dataset.
            RuntimeError: If `sizes` is list of integers and sum of all elements in sizes does not
                equal the dataset size.
            RuntimeError: If `sizes` is list of float and there is a split with size 0 after calculations.
            RuntimeError: If the dataset is sharded prior to calling split.
            ValueError: If `sizes` is list of float and not all floats are between 0 and 1, or if the
                floats don't sum to 1.

        Examples:
            >>> import mindspore.dataset as ds
            >>> # Since many datasets have shuffle on by default, set shuffle to False if split will be called!
            >>> image_folder_dataset_dir = "/path/to/image_folder_dataset_directory"
            >>> dataset = ds.ImageFolderDataset(image_folder_dataset_dir, shuffle=False)
            >>>
            >>> # Set the seed, and tell split to use this seed when randomizing.
            >>> # This is needed because sharding will be done later
            >>> ds.config.set_seed(58)
            >>> train_dataset, test_dataset = dataset.split([0.9, 0.1])
            >>>
            >>> # To shard the train dataset, use a DistributedSampler
            >>> train_sampler = ds.DistributedSampler(10, 2)
            >>> train_dataset.use_sampler(train_sampler)
        """
        if self.is_shuffled():
            logger.warning("Dataset is shuffled before split.")

        if self.is_sharded():
            raise RuntimeError("Dataset should not be sharded before split.")

        absolute_sizes = self._get_absolute_split_sizes(sizes)
        splits = []
        current_split_start_index = 0
        for size in absolute_sizes:
            ds = copy.deepcopy(self)
            ds.dataset_size = None
            if randomize:
                # want to shuffle the same way every epoch before split, we are assuming
                # that the user will call set_seed
                random_sampler = samplers.RandomSampler()
                random_sampler.reshuffle_each_epoch = False
                ds.add_sampler(random_sampler)

            subset_sampler = samplers.SequentialSampler(current_split_start_index, size)
            ds.add_sampler(subset_sampler)

            # add sequential sampler, so that if user calls use_sampler, we will
            # get rid of the sequential sampler instead of something we need
            ds.add_sampler(samplers.SequentialSampler())

            splits.append(ds)

            current_split_start_index += size

        return tuple(splits)


class BucketBatchByLengthDataset(UnionBaseDataset):
    """
    The result of applying BucketBatchByLength operation to the input dataset.
    """

    def __init__(self, input_dataset, column_names, bucket_boundaries, bucket_batch_sizes, element_length_function,
                 pad_info, pad_to_bucket_boundary, drop_remainder):
        super().__init__(children=input_dataset)

        self.column_names = to_list(column_names)
        self.bucket_boundaries = replace_none(bucket_boundaries, [])
        self.bucket_batch_sizes = replace_none(bucket_batch_sizes, [])
        self.element_length_function = element_length_function
        self.pad_info = replace_none(pad_info, {})
        self.pad_to_bucket_boundary = replace_none(pad_to_bucket_boundary, False)
        self.drop_remainder = replace_none(drop_remainder, False)

    def parse(self, children=None):
        return cde.BucketBatchByLengthNode(children[0], self.column_names, self.bucket_boundaries,
                                           self.bucket_batch_sizes, self.element_length_function, self.pad_info,
                                           self.pad_to_bucket_boundary, self.drop_remainder)


class BatchDataset(UnionBaseDataset):
    """
    The result of applying Batch operation to the input dataset.

    Args:
        input_dataset (Dataset): Input Dataset to be batched.
        batch_size (Union[int, function]): The number of rows each batch is created with. An
            int or callable which takes exactly 1 parameter, BatchInfo.
        drop_remainder (bool, optional): Determines whether or not to drop the last
            possibly incomplete batch. Default: ``False``. If True, and if there are less
            than batch_size rows available to make the last batch, then those rows will
            be dropped and not propagated to the child node.
        num_parallel_workers (int, optional): Number of workers to process the dataset in parallel. Default: ``None``.
        per_batch_map (callable, optional): Per batch map callable. A callable which takes
            (list[Tensor], list[Tensor], ..., BatchInfo) as input parameters. Each list[Tensor] represents a batch of
            Tensors on a given column. The number of lists should match with number of entries in input_columns. The
            last parameter of the callable must always be a BatchInfo object.
        input_columns (Union[str, list[str]], optional): List of names of the input columns. The size of the list must
            match with signature of per_batch_map callable.
        output_columns (Union[str, list[str]], optional): List of names assigned to the columns outputted by
            the last operation. This parameter is mandatory if len(input_columns) !=
            len(output_columns). The size of this list must match the number of output
            columns of the last operation. Default: ``None``, output columns will have the same
            name as the input columns, i.e., the columns will be replaced.
        max_rowsize(Union[int, list[int]], optional): Maximum size of row in MB that is used for shared memory
            allocation to copy data between processes, the total occupied shared memory will increase as
            ``num_parallel_workers`` and :func:`mindspore.dataset.config.set_prefetch_size` increase. If set to -1,
            shared memory will be dynamically allocated with the actual size of data. This is only used if
            ``python_multiprocessing`` is set to True. If it is an int value, it represents
            ``input_columns`` and ``output_columns`` use this value as the unit to create shared memory.
            If it is a list, the first element represents the ``input_columns`` use this value as the unit to
            create shared memory, and the second element represents ``output_columns`` use this value as the unit
            to create shared memory. Default: ``None`` , allocate shared memory dynamically.

    """

    def __init__(self, input_dataset, batch_size, drop_remainder=False, num_parallel_workers=None, per_batch_map=None,
                 input_columns=None, output_columns=None, python_multiprocessing=False, max_rowsize=None):
        super().__init__(children=input_dataset, num_parallel_workers=num_parallel_workers)

        if BatchDataset._is_ancestor_of_repeat(input_dataset):
            logger.warning("Repeat is located before batch, data from two epochs can be batched together.")

        BatchDataset._update_batch_size_for_syncwait(input_dataset, batch_size)

        # if batch_size is callable, set batch_size to 1 and batch_size_func to that callable function
        self.batch_size = batch_size if not callable(batch_size) else 1
        self.batch_size_func = None if not callable(batch_size) else batch_size

        self.drop_remainder = replace_none(drop_remainder, False)

        self.per_batch_map = per_batch_map

        self.input_columns = to_list(input_columns)
        self.output_columns = to_list(output_columns)

        self.python_multiprocessing = python_multiprocessing
        self.process_pool = None
        if max_rowsize is None:
            self.max_rowsize = [-1, -1]
        elif isinstance(max_rowsize, int):
            self.max_rowsize = [max_rowsize * self.batch_size] * 2 if max_rowsize != -1 else [max_rowsize, max_rowsize]
        else:
            self.max_rowsize = [max_rowsize[0] * self.batch_size, max_rowsize[1] * self.batch_size]

    def parse(self, children=None):
        return cde.BatchNode(children[0], self.batch_size, self.drop_remainder, False, self.input_columns,
                             self.output_columns, self.batch_size_func, self.per_batch_map, {},
                             self.process_pool)

    @staticmethod
    def _is_ancestor_of_repeat(dataset):
        """
        Utility function to find the case where repeat is used before batch.

        Args:
             dataset (Dataset): Dataset to be checked.

        Returns:
            bool, whether repeat is used before batch.
        """
        if isinstance(dataset, RepeatDataset):
            return True
        flag = False
        for input_dataset in dataset.children:
            flag = flag | BatchDataset._is_ancestor_of_repeat(input_dataset)
        return flag

    @staticmethod
    def _update_batch_size_for_syncwait(dataset, batch_size):
        """
        Utility function to notify batch size to sync_wait.

        Args:
             dataset (Dataset): Dataset to be checked.
             batch_size (int): batch size to notify.
        """
        if isinstance(dataset, SyncWaitDataset):
            dataset.update_sync_batch_size(batch_size)
        for input_dataset in dataset.children:
            BatchDataset._update_batch_size_for_syncwait(input_dataset, batch_size)

    def __deepcopy__(self, memodict):
        return self.__safe_deepcopy__(memodict, exclude=("per_batch_map", "batch_size_func", "__transfer_dataset__"))

    # Iterator bootstrap will be called on iterator construction.
    # A deep copy of Dataset object is created prior of iterator_bootstrap.
    # This method will create per iterator process pool and bind pyfunc execution to the pool.
    def iterator_bootstrap(self):
        """
        Per iterator bootstrap callback.
        """
        if self.python_multiprocessing and platform.system().lower() == 'windows':
            logger.warning("Python multiprocessing is not supported on Windows platform.")
        if self.python_multiprocessing and get_debug_mode():
            logger.warning("Python multiprocessing is not supported in debug mode."
                           " Ignoring Python multiprocessing for batch operation.")
            self.python_multiprocessing = False
        if self.python_multiprocessing and platform.system().lower() != 'windows':
            if self.per_batch_map is None:
                logger.warning("per_batch_map is None so python_multiprocessing is ignored for batch.")
                return

            # If user didn't specify num_parallel_workers, set it to default
            if self.num_parallel_workers is None:
                self.num_parallel_workers = get_num_parallel_workers()

            self.process_pool = _PythonMultiprocessing(get_multiprocessing_start_method(), self.num_parallel_workers,
                                                       str(self), [self.per_batch_map], self.max_rowsize)
        else:
            if self.per_batch_map is not None:
                self.per_batch_map = FuncWrapper(self.per_batch_map)


class BatchInfo(cde.CBatchInfo):
    """
    This class helps to get dataset information dynamically when the input of `batch_size` or `per_batch_map`
    in `batch` operation is a callable object.
    """

    def get_batch_num(self):
        """
        Return the batch number being processed in current epoch, starting from 0.

        Examples:
            >>> # Create a dataset where its batch size is dynamic
            >>> # Define a callable batch size function and let batch size increase 1 each time.
            >>> import mindspore.dataset as ds
            >>> from mindspore.dataset import BatchInfo
            >>>
            >>> dataset = ds.GeneratorDataset([i for i in range(3)], "column1", shuffle=False)
            >>> def add_one(BatchInfo):
            ...     return BatchInfo.get_batch_num() + 1
            >>> dataset = dataset.batch(batch_size=add_one)
            >>> print(list(dataset))
            [[Tensor(shape=[1], dtype=Int64, value= [0])], [Tensor(shape=[2], dtype=Int64, value= [1, 2])]]
        """
        return

    def get_epoch_num(self):
        """
        Return the epoch number, starting from 0.

        Examples:
            >>> # Create a dataset where its batch size is dynamic
            >>> # Define a callable batch size function and let batch size increase 1 each epoch.
            >>> import mindspore.dataset as ds
            >>> from mindspore.dataset import BatchInfo
            >>>
            >>> dataset = ds.GeneratorDataset([i for i in range(4)], "column1", shuffle=False)
            >>> def add_one_by_epoch(BatchInfo):
            ...     return BatchInfo.get_epoch_num() + 1
            >>> dataset = dataset.batch(batch_size=add_one_by_epoch)
            >>>
            >>> result = []
            >>> epoch = 2
            >>> iterator = dataset.create_tuple_iterator(num_epochs=epoch)
            >>> for i in range(epoch):
            ...    result.extend(list(iterator))
            >>> # result:
            >>> # [[Tensor(shape=[1], dtype=Int64, value= [0])], [Tensor(shape=[1], dtype=Int64, value= [1])],
            >>> #  [Tensor(shape=[1], dtype=Int64, value= [2])], [Tensor(shape=[1], dtype=Int64, value= [3])],
            >>> #  [Tensor(shape=[2], dtype=Int64, value= [0, 1])], [Tensor(shape=[2], dtype=Int64, value= [2, 3])]]
        """
        return


class BlockReleasePair:
    """
    The blocking condition class used by SyncWaitDataset.

    Args:
        init_release_rows (int): Number of lines to allow through the pipeline.
        callback (function): The callback function that will be called when release is called. Default: ``None``.
    """

    def __init__(self, init_release_rows, callback=None):
        if isinstance(init_release_rows, int) and init_release_rows <= 0:
            raise ValueError("release_rows need to be greater than 0.")
        self.row_count = -init_release_rows
        self.cv = threading.Condition()
        self.callback = callback
        self.default_rows = init_release_rows
        self.disable = False

    def __deepcopy__(self, memodict):
        return self

    def reset(self):
        with self.cv:
            self.row_count = -self.default_rows
            self.cv.notify_all()

    def update_batched_size(self, batch_size):
        # sanity check
        if isinstance(batch_size, int) and batch_size <= 0:
            raise ValueError("batch_size need to be greater than 0.")

        # should only use before the pipeline creates
        self.row_count *= batch_size
        self.default_rows *= batch_size

    def block_func(self):
        """
        Function for handing blocking condition.

        Returns:
            bool, True.
        """
        with self.cv:
            # if disable is true, the always evaluate to true
            not_time_out = self.cv.wait_for(lambda: (self.row_count < 0 or self.disable),
                                            timeout=get_callback_timeout())
            # time_out will be False if time out occurs
            if not not_time_out:
                logger.warning("Timeout happened in sync_wait, maybe dataset.sync_update(condition=...) "
                               "is not added after dataset.create_dict_iterator(...), now disabling lock.")
                self.disable = True
            self.row_count += 1
        return True

    def release_func(self, pass_rows=None, data=None):
        with self.cv:
            if pass_rows is None:
                pass_rows = self.default_rows
            self.row_count -= pass_rows
            if self.callback is not None:
                self.callback(data)
            self.cv.notify_all()

    def disable_lock(self):
        with self.cv:
            self.disable = True
            self.cv.notify_all()


class PaddedBatchDataset(UnionBaseDataset):
    """
    The result of applying Batch operation to the input dataset.

    Args:
        input_dataset (Dataset): Input Dataset to be batched.
        batch_size (Union[int, function]): The number of rows each batch is created with. An
            int or callable which takes exactly 1 parameter, BatchInfo.
        drop_remainder (bool, optional): Determines whether or not to drop the last
            possibly incomplete batch. Default: ``False``. If True, and if there are less
            than batch_size rows available to make the last batch, then those rows will
            be dropped and not propagated to the child node.
        num_parallel_workers (int, optional): Number of workers to process the dataset in parallel. Default: ``None``.
        pad_info (dict, optional): Whether to perform padding on selected columns. pad_info={"col1":([224,224],0)}
            will pad column with name "col1" to a tensor of size [224,224] and fill the missing with 0.
    """

    def __init__(self, input_dataset, batch_size, drop_remainder=False, num_parallel_workers=None, pad_info=None):
        super().__init__(children=input_dataset, num_parallel_workers=num_parallel_workers)

        if PaddedBatchDataset._is_ancestor_of_repeat(input_dataset):
            logger.warning("Repeat is located before padded_batch, data from two epochs can be batched together.")

        PaddedBatchDataset._update_batch_size_for_syncwait(input_dataset, batch_size)

        # if batch_size is callable, set batch_size to 1 and batch_size_func to that callable function
        self.batch_size = batch_size if not callable(batch_size) else 1
        self.batch_size_func = None if not callable(batch_size) else batch_size

        self.drop_remainder = replace_none(drop_remainder, False)

        self.pad = bool(pad_info is not None)
        self.pad_info = replace_none(pad_info, {})

    def parse(self, children=None):
        return cde.BatchNode(children[0], self.batch_size, self.drop_remainder, self.pad, [],
                             [], self.batch_size_func, None, self.pad_info, None)

    @staticmethod
    def _is_ancestor_of_repeat(dataset):
        """
        Utility function to find the case where repeat is used before batch.

        Args:
             dataset (Dataset): Dataset to be checked.

        Returns:
            bool, whether repeat is used before batch.
        """
        if isinstance(dataset, RepeatDataset):
            return True
        flag = False
        for input_dataset in dataset.children:
            flag = flag | PaddedBatchDataset._is_ancestor_of_repeat(input_dataset)
        return flag

    @staticmethod
    def _update_batch_size_for_syncwait(dataset, batch_size):
        """
        Utility function to notify batch size to sync_wait.

        Args:
             dataset (Dataset): Dataset to be checked.
             batch_size (int): batch size to notify.
        """
        if isinstance(dataset, SyncWaitDataset):
            dataset.update_sync_batch_size(batch_size)
        for input_dataset in dataset.children:
            PaddedBatchDataset._update_batch_size_for_syncwait(input_dataset, batch_size)

    def __deepcopy__(self, memodict):
        return self.__safe_deepcopy__(memodict, exclude=("batch_size_func", "__transfer_dataset__"))


class SyncWaitDataset(UnionBaseDataset):
    """
    The result of adding a blocking condition to the input Dataset.

    Args:
        input_dataset (Dataset): Input dataset to apply flow control.
        num_batch (int): Number of batches without blocking at the start of each epoch.
        condition_name (str): Condition name that is used to toggle sending next row.
        callback (function): Callback function that will be invoked when sync_update is called. Default: ``None``.

    Raises:
        RuntimeError: If condition name already exists.
    """

    def __init__(self, input_dataset, condition_name, num_batch, callback=None):
        super().__init__(children=input_dataset)

        # set to the default value, waiting for the batch to update it
        self._condition_name = condition_name
        if isinstance(num_batch, int) and num_batch <= 0:
            raise ValueError("num_batch need to be greater than 0.")

        self._pair = BlockReleasePair(num_batch, callback)
        if self._condition_name in self.children[0].get_sync_notifiers():
            raise RuntimeError("Condition name is already in use.")
        logger.info("Please remember to add dataset.sync_update(condition=%s), otherwise hanging will result. "
                    "If dataset.sync_update(condition=%s) has already been added, you can ignore the info.",
                    condition_name, condition_name)

    def parse(self, children=None):
        return cde.SyncWaitNode(children[0], self._condition_name, self._pair.block_func)

    def get_sync_notifiers(self):
        return {**self.children[0].get_sync_notifiers(), self._condition_name: self._pair.release_func}

    def is_sync(self):
        return True

    def update_sync_batch_size(self, batch_size):
        if isinstance(batch_size, int) and batch_size <= 0:
            raise ValueError("num_batch need to be greater than 0.")
        self._pair.update_batched_size(batch_size)

    def disable_sync(self):
        logger.info("Disabling Sync")
        self._pair.disable_lock()

    @staticmethod
    def _is_ancestor_of_batch(dataset):
        """
        Utility function to find the case where sync_wait is used before batch.

        Args:
             dataset (Dataset): Dataset to be checked.

        Returns:
            bool, whether sync_wait is used before batch.
        """
        if isinstance(dataset, (BatchDataset, PaddedBatchDataset)):
            return True
        flag = False
        for input_dataset in dataset.children:
            flag = flag | SyncWaitDataset._is_ancestor_of_batch(input_dataset)
        return flag

    def iterator_bootstrap(self):
        self._pair.reset()


class ShuffleDataset(UnionBaseDataset):
    """
    The result of applying shuffle operation to the input Dataset.

    Args:
        input_dataset (Dataset): Input Dataset to be shuffled.
        buffer_size (int): Size of the buffer.

    Raises:
        RuntimeError: If exist sync operations before shuffle.
    """

    def __init__(self, input_dataset, buffer_size):
        super().__init__(children=input_dataset)
        self.buffer_size = buffer_size
        self.reshuffle_each_epoch = True

        if self.is_sync():
            raise RuntimeError("No shuffle after sync operators.")

    def parse(self, children=None):
        return cde.ShuffleNode(children[0], self.buffer_size, self.reshuffle_each_epoch)

    def is_shuffled(self):
        return True


# Pyfunc collection for multiprocess pyfunc
# This global variable will only be used within subprocesses
_OP_NAME = {}
_OP_PROCESS = {}


def _main_process_already_exit():
    """
    Judge whether main process already exit.
    """
    ppid = os.getppid()

    if (platform.system().lower() != 'windows' and
            not _PythonMultiprocessing.is_process_alive(ppid)):
        return True
    return False


def _worker_loop(quit_signal, operations, worker_id, op_type, key, video_backend=None):
    """
    Multiprocess worker process loop.
    The worker process(Python Layer) gets data from / sends data to map / batch thread(C++ layer) by message queue
    and shared memory. This logic no longer uses the Python multi-process pool, in_queue, and out_queue for
    data transferring.
    """
    # Release the lock which had been holded in map_op.cc::Launch()/batch_op.cc::Launch()
    cde.unlock_shm_id_and_msg_id_mutex()

    # Initialize C++ side signal handlers
    cde.register_worker_handlers()

    if video_backend is not None:
        set_video_backend(video_backend)

    def _ignore_sigint():
        """
        We need to ignore sigint signal here so subprocesses can exit normally and clear.
        """
        signal.signal(signal.SIGINT, signal.SIG_IGN)

    # If the default random seed has not been changed, there is no need to fix the randomness.
    # Otherwise, set the random seed for each child process to "base_seed + worker_id" to ensure
    # that the random results of each process are different.
    if get_seed() != 5489:
        set_seed(get_seed() + worker_id)

    msg_queue = cde.MessageQueue(key)
    msg_queue.set_release_flag(False)
    shm_queue = cde.SharedMemoryQueue(key)
    shm_queue.set_release_flag(False)

    pid = str(os.getpid())
    ppid = str(os.getppid())

    # Scenario: when the main process is killed, worker processe needs to release shm & msg.
    # The shm id and msg id should be released by SIGTERM in worker handler
    cde.register_shm_id_and_msg_id(pid + "_" + ppid + "_" + str(op_type), shm_queue.get_shm_id(),
                                   msg_queue.msg_queue_id)

    num_receive = 0
    num_send = 0
    while not _main_process_already_exit():
        _ignore_sigint()

        # quit by close_worker
        if quit_signal.is_set():
            return

        # >> receive procedure >>
        ## 1. get message queue which contains shared memory info from map C++ thread in main process
        try:
            cde.register_shm_id_and_msg_id(pid + "_" + ppid + "_" + str(op_type), shm_queue.get_shm_id(),
                                           msg_queue.msg_queue_id)
            msg_queue.msg_rcv(cde.MASTER_SEND_DATA_MSG)
            cde.register_shm_id_and_msg_id(pid + "_" + ppid + "_" + str(op_type), shm_queue.get_shm_id(),
                                           msg_queue.msg_queue_id)
        except RuntimeError as err:
            cde.register_shm_id_and_msg_id(pid + "_" + ppid + "_" + str(op_type), shm_queue.get_shm_id(),
                                           msg_queue.msg_queue_id)
            # the msg_queue had been released by main process, ignore it in worker process
            if "errno: 2" in str(err):
                # Because the worker process does not release msg and shm, continue
                continue
            raise err

        ## when the message queue had been released, break the loop
        if msg_queue.message_queue_state() == cde.MessageState.RELEASED:
            logger.info("The message queue had been released, worker loop end.")
            break

        num_receive += 1

        logger.info(f"Python process {op_type} worker({worker_id}) receives {num_receive} samples from map thread.")

        # convert the data from shm to python data
        if op_type == cde.MAP_OP:
            ## 2. construct shared memory to TensorRow which contains one / more columns
            tensor_row = shm_queue.to_tensor_row(msg_queue.shm_id, msg_queue.shm_size)

            ## 3. convert TensorRow to Python tuple which elements are a column
            tuple_column = cde.convert_tensor_row_to_py_tuple(tensor_row)

            py_func_input = tuple_column
        elif op_type == cde.BATCH_OP:
            ## 2. construct shard memory to TensorTable which contains one / more TensorRow & CBatchInfo
            tensor_table, batch_info, _ = shm_queue.to_tensor_table(msg_queue.shm_id, msg_queue.shm_size)

            ## 3. convert TensorTable to Python tuple tuple
            # The tuple indicate the multi columns
            # The list indicate the multi rows
            tuple_list_column = cde.convert_tensor_table_to_py_tuple_list(tensor_table)

            py_func_input = (*tuple_list_column, batch_info)
        else:
            raise RuntimeError(f"The op_type: {op_type} is invalid.")

        # execute the pyfunc
        try:
            py_func_output = py_func_input

            # execute the remaining operations
            for idx, _ in enumerate(operations):
                if isinstance(py_func_output, tuple):
                    py_func_output = operations[idx](*py_func_output)
                else:
                    py_func_output = operations[idx](py_func_output)

            # << send procedure <<
            # the result is None
            if py_func_output is None:
                raise RuntimeError(f"Got None from Python Function which is defined by {op_type}")

            # convert the output to tuple
            if not isinstance(py_func_output, tuple):
                py_func_output = (py_func_output,)

            if op_type == cde.MAP_OP:
                # check if the map return Generator type
                for item in py_func_output:
                    if isinstance(item, GeneratorType):
                        raise RuntimeError("Cannot pickle <class 'generator'> object, please verify pyfunc "
                                           "return with numpy array")

                ## 1. convert Python tuple to TensorRow
                output_tensor_row = cde.convert_py_tuple_to_tensor_row(py_func_output)

                ## 2. convert TensorRow to shared memory
                shm_queue.from_tensor_row(output_tensor_row)
            elif op_type == cde.BATCH_OP:
                ## 1. convert Python tuple tuple to TensorTable
                output_tensor_table, concat_batch = cde.convert_py_tuple_list_to_tensor_table(py_func_output)

                ## 2. convert TensorTable to shared memory
                shm_queue.from_tensor_table(output_tensor_table, batch_info, concat_batch)
            else:
                raise RuntimeError(f"The op_type: {op_type} is invalid.")

            ## 3. send message queue which contains shared memory to map C++ thread in main process
            cde.register_shm_id_and_msg_id(pid + "_" + ppid + "_" + str(op_type), shm_queue.get_shm_id(),
                                           msg_queue.msg_queue_id)
            msg_queue.msg_snd(cde.WORKER_SEND_DATA_MSG, shm_queue.get_shm_id(), shm_queue.get_shm_size())
            cde.register_shm_id_and_msg_id(pid + "_" + ppid + "_" + str(op_type), shm_queue.get_shm_id(),
                                           msg_queue.msg_queue_id)

            num_send += 1
            logger.info(f"Python process {op_type} worker({worker_id}) sends {num_send} samples to map thread.")
        except Exception:
            try:
                if op_type == cde.MAP_OP:
                    pyfunc_err = ExceptionHandler(where="in map worker and execute Python function")
                elif op_type == cde.BATCH_OP:
                    pyfunc_err = ExceptionHandler(where="in batch(per_batch_map) worker and execute Python function")
                else:
                    pyfunc_err = f"The op_type: {op_type} is invalid."
                pyfunc_err.reraise()
            except Exception as err:
                _, _, exc_tb = sys.exc_info()
                fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]

                if op_type == cde.MAP_OP:
                    logger.info(f"Got exception {str(err)} from Map Worker({worker_id})")
                elif op_type == cde.BATCH_OP:
                    logger.info(f"Got exception {str(err)} from Batch Worker({worker_id})")
                else:
                    logger.info(f"The op_type: {op_type} is invalid.")

                # err_code, lineno, filename, err_desc
                msg_queue.serialize_status(cde.StatusCode.MD_PY_FUNC_EXCEPTION, exc_tb.tb_lineno, fname, str(err))

                cde.register_shm_id_and_msg_id(pid + "_" + ppid + "_" + str(op_type), shm_queue.get_shm_id(),
                                               msg_queue.msg_queue_id)
                msg_queue.msg_snd(cde.WORKER_SEND_DATA_MSG, shm_queue.get_shm_id(), shm_queue.get_shm_size())
                cde.register_shm_id_and_msg_id(pid + "_" + ppid + "_" + str(op_type), shm_queue.get_shm_id(),
                                               msg_queue.msg_queue_id)

                # worker error
                if get_error_samples_mode() == ErrorSamplesMode.RETURN:
                    break

                # continue the loop, when the get_error_samples_mode() is REPLACE or SKIP
                continue

    # release the eager executor which is used by current process
    transforms.transforms.clean_unused_executors()

    while not _main_process_already_exit():
        logger.info("The worker process is waiting for the main process to exit.")
        time.sleep(0.1)

        # quit by close_worker
        if quit_signal.is_set():
            return


    # the main process is not exist yet which maybe killed -9
    msg_queue.set_release_flag(True)
    msg_queue.release()
    shm_queue.set_release_flag(True)
    shm_queue.release()


class WorkerTarget:
    """Mulitprocess mode for dataset map or batch"""
    def __init__(self, quit_signal, operations, worker_id, op_type, ftok_key):
        self.quit_signal = quit_signal
        self.operations = operations
        self.worker_id = worker_id
        self.op_type = op_type
        self.ftok_key = ftok_key
        start_method = multiprocessing.get_start_method()
        logger.info(f"Multiprocessing start method: {start_method}")
        self.video_backend = get_video_backend() if start_method == 'spawn' else None

    def __call__(self):
        return _worker_loop(self.quit_signal, self.operations, self.worker_id, self.op_type, self.ftok_key,
                            self.video_backend)


def worker_is_alive(worker):
    """Check the subprocess worker status"""
    try:
        return worker.is_alive()
    except ValueError:
        return False


def close_worker(worker, eof):
    """Close the subprocess worker"""
    try:
        if worker_is_alive(worker):
            # release the eager executor which is used by current process
            transforms.transforms.clean_unused_executors()

            # wait timeout
            wait_timeout = 2
            start_time = time.time()
            process_dir = os.path.join('/proc', str(worker.pid))
            while worker_is_alive(worker) and os.path.exists(process_dir):
                # let the worker exit
                logger.info(f"Set eof flag for worker with PID: {worker.pid}.")
                eof.set()

                logger.info(f"Waiting for worker {worker.pid} to close ...")
                time.sleep(0.5)

                # maybe the worker is hung by msg_queue.MsgRcv, so break the loop and terminate it in next step
                if time.time() - start_time > wait_timeout:
                    break

            # del the handle which hold by master
            worker.terminate()
            worker.join()
            worker.close()
    except ValueError:
        # Process has been closed already
        return
    return


class _PythonMultiprocessing(cde.PythonMultiprocessingRuntime):
    """
    A wrapper to multiprocessing.pool that performs cleanup and ensure proper termination of forked processes.
    """

    class _ExceptHookHandler:
        """
        Internal class ExceptionHandler
        """

        def __init__(self):
            self.origin_hook = sys.excepthook
            sys.excepthook = self.__handler_exception

        @staticmethod
        def mp_pool_exit_preprocess():
            if check_iterator_cleanup() is False:
                # Set the iterator_cleanup flag to True before exiting, and wait 3s for all apply_async
                # applied to the multiprocessing task to prevent multiprocessing from hang when exiting
                _set_iterator_cleanup()
                time.sleep(3)

        def __handler_exception(self, ex_type, value, tb):
            self.origin_hook(ex_type, value, tb)
            self.mp_pool_exit_preprocess()

    def __init__(self, start_method, num_parallel_workers, op_name, operations, max_rowsize=(-1, -1)):
        super().__init__()
        self.start_method = start_method  # python multiprocssing start method: fork / spawn
        self.num_parallel_workers = num_parallel_workers
        self.op_name = op_name
        self.operations = operations
        self.max_rowsize = max_rowsize

        self.workers = None
        self.pids = None
        self.op_id = -1

        self.queues_map = {}
        self.next_queue = 0

        self.cleaning_process = None
        self.ppid = None
        self.hook = None
        self.warning_ctl = None
        # cache thread (get_ident()) to worker_id mapping in Python layer
        self.python_threads_to_workers = {}
        self.eof_workers = []
        self.eof_clean_process = None
        self.running = False

    def __del__(self):
        try:
            self.terminate()
        except TypeError:
            pass

    @staticmethod
    def _terminate_processes(processes):
        """Terminate subprocesses"""

        for p in processes:
            try:
                if p.exitcode is None:
                    p.terminate()
            except Exception:  # pylint: disable=broad-except
                # process has been closed already
                pass
        for p in processes:
            if p._closed is False:  # pylint: disable=W0212
                # We don't use w.join because join can only used in main process or join will raise an error.
                p._popen.wait()  # pylint: disable=W0212

    @staticmethod
    def is_process_alive(pid):
        """
        Check if the process is alive or not.
        Note:  We hit a deadlock when we use psutil or w.exitcode to check whether a process is alive.
        Instead, we use os.kill(ppid, 0).

        Args:
            pid: pid of the process to be checked

        Returns:
            True if the process is alive
        """

        try:
            os.kill(pid, 0)
        except OSError:
            return False
        return True

    # When main process exit, subprocesses will be terminate
    @staticmethod
    def _clean_process(ppid, workers, quit_signal):
        """
            This is the execute function of clean process, if we found main process exited, we will clean subprocesses.

        Args:
            ppid: The process id of main process.
            workers: The list of subprocesses.
            quit_signal: The flag of quit.
        """
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        # Initialize C++ side signal handlers
        cde.register_worker_handlers()
        while _PythonMultiprocessing.is_process_alive(ppid):
            if quit_signal.is_set():
                return

            # independent dataset mode, the subprocess of GeneratorDataset / map / batch should exit when
            # independent dataset process have exit
            if os.getppid() != ppid:
                break

            time.sleep(0.1)

        logger.info(f"Clean process detects that the main process {ppid} has exited, begin to terminate the " +
                    f"worker process(es): {[worker.pid for worker in workers]}")
        _PythonMultiprocessing._terminate_processes(workers)
        del workers
        os.kill(os.getpid(), signal.SIGTERM)

    def launch(self, op_id, op_type, ftok_keys):
        """
        Launch Python multiprocessing pool.

        Args:
            op_id (int): ID for operation to have Python multiprocessing pool launched
            op_type (str): Indicate MapOp / BatchOp
            ftok_keys (list[int]): the ftok key of list for msg queue and shm queue

        Returns:
            Python multiprocessing pool is launched.
        """
        self.python_threads_to_workers = {}

        if not isinstance(op_id, int):
            raise RuntimeError("The op_id is not int.")
        self.op_id = op_id

        valid_op_type = [cde.MAP_OP, cde.BATCH_OP]
        if op_type not in valid_op_type:
            raise RuntimeError(f"The op_type: {op_type} is not in {valid_op_type}.")
        self.op_type = op_type

        if not isinstance(ftok_keys, list):
            raise RuntimeError("The ftok_keys is not a list.")
        if not all(isinstance(x, int) for x in ftok_keys):
            raise RuntimeError("The item in ftok_keys is not all int.")
        if len(ftok_keys) != self.num_parallel_workers:
            raise RuntimeError("The len of ftok_keys is not equal to num_parallel_workers.")
        self.ftok_keys = ftok_keys

        logger.info("Launching new Python multiprocessing pool for Op: " + self.op_type + "(" + str(self.op_id) + \
                    "), ftok_keys: " + str(self.ftok_keys))
        if self.is_mp_enabled():
            message = "Launching a new Python multiprocessing pool while a pool already exists!" + \
                      " The existing pool will be terminated first."
            logger.warning(message)
            self.terminate()
            self.reset()
        self.ppid = os.getpid()
        self.create_pool()

    def create_pool(self):
        """

        Returns:

        """
        if self.workers is not None:
            raise Exception("Pool was already created, close it first.")

        self.workers = []
        self.warning_ctl = multiprocessing.Value('i', 0)

        multiprocessing.set_start_method(self.start_method, True)

        # Construct python worker processes
        for worker_id in range(self.num_parallel_workers):
            eof = multiprocessing.Event()
            worker = multiprocessing.Process(target=WorkerTarget(eof, self.operations, worker_id, self.op_type,
                                                                 self.ftok_keys[worker_id]),
                                             name="MapWorker" + str(worker_id), daemon=True)
            self.eof_workers.append(eof)
            self.workers.append(worker)
            worker.start()

        multiprocessing.set_start_method("fork", True)

        logger.info(f"Launch worker process(es): {self.get_pids()}")

        self.hook = _PythonMultiprocessing._ExceptHookHandler()

        # Launch a clean process and register worker processes to be monitored by the watch dog.
        self._launch_monitor()
        self.running = True

        # Register a termination function using weakref to avoid the object from unable to properly destruct.
        atexit.register(lambda cleanup: cleanup()() if cleanup() is not None else None,
                        weakref.WeakMethod(self.terminate))

        # Ensure that all workers are in the running state
        start = time.time()
        wait_time = 120  # 120s
        while True:
            if self.is_running():
                logger.info("All workers has been running state.")
                break
            time.sleep(0.5)
            if time.time() - start > wait_time:
                logger.error("All worker processes have not reached the running state within " + str(wait_time) +
                             " seconds, data processing errors may occur.")
                break

    def terminate(self):
        if self.running:
            # abort the monitor first and then close all the workers
            self._abort_monitor()
            self.close_all_workers()
            if hasattr(self, "warning_ctl"):
                del self.warning_ctl
            self.running = False

    def get_pids(self):
        """
        Get list of worker's PIDs

        Returns:
            list of strings
        """
        if not self.is_mp_enabled():
            return []
        if not self.pids:
            self.pids = []
            if self.workers:
                for w in self.workers:
                    try:
                        self.pids.append(w.pid)
                    except ValueError:
                        continue
        return self.pids

    def add_new_workers(self, num_new_workers, op_type, ftok_keys):
        """Used by AutoTune"""
        logger.info(
            "Increasing num_parallel_workers of Python Multiprocessing pool for Op:" + str(self.op_id) +
            ", old num_workers=" + str(self.num_parallel_workers) + " new num_workers=" + str(
                self.num_parallel_workers +
                num_new_workers) + ".")
        self.terminate()
        self.num_parallel_workers += num_new_workers

        if self.num_parallel_workers != len(ftok_keys):
            raise RuntimeError("Add new workers failed, the num_workers is not equal size of ftok_keys.")

        self.launch(self.op_id, op_type, ftok_keys)

    def remove_workers(self, num_removed_workers, op_type, ftok_keys):
        """Used by AutoTune"""
        logger.info(
            "Decreasing num_parallel_workers of Python Multiprocessing pool for Op:" + str(self.op_id) +
            ", old num_workers=" + str(self.num_parallel_workers) + " new num_workers=" + str(
                self.num_parallel_workers -
                num_removed_workers) + ".")
        self.terminate()
        self.num_parallel_workers -= num_removed_workers

        if self.num_parallel_workers != len(ftok_keys):
            raise RuntimeError("Remove workers failed, the num_workers is not equal size of ftok_keys.")

        self.launch(self.op_id, op_type, ftok_keys)

    def is_mp_enabled(self):
        return self.workers is not None

    def _launch_monitor(self):
        """
        Launch a clean process and register subprocess to be monitored by the watch dog.
        The clean process will clean up subprocesses when main process exited.
        The watch dog will clean up subprocesses and main process when any subprocess exited.
        """
        if platform.system().lower() != 'windows':
            self.eof_clean_process = multiprocessing.Event()
            self.cleaning_process = multiprocessing.Process(target=self._clean_process,
                                                            name="MapCleanProcess",
                                                            args=(self.ppid, self.workers, self.eof_clean_process),
                                                            daemon=True)
            self.cleaning_process.start()
            logger.info(f"Launch clean process {self.cleaning_process.pid} to monitor worker " +
                        f"process(es): {self.get_pids()}")

            if get_enable_watchdog():
                worker_ids = [os.getpid()]
                worker_ids.extend([worker.pid for worker in self.workers])
                worker_ids.append(self.cleaning_process.pid)
                cde.register_worker_pids(id(self), worker_ids)

    def _abort_monitor(self):
        """Deregister workers monitored by the watch dog and join clean process."""
        if get_enable_watchdog():
            cde.deregister_worker_pids(id(self))
        if hasattr(self, 'eof_clean_process') and self.eof_clean_process is not None:
            logger.info("Set eof flag for cleaning_process.")
            self.eof_clean_process.set()
        if hasattr(self, 'cleaning_process') and self.cleaning_process is not None:
            # let the quit event notify the cleaning process to exit
            self.cleaning_process.join(timeout=5)
            if self.cleaning_process.is_alive():
                # if the cleaning process did not exit, it may hang, try to terminate it
                _PythonMultiprocessing._terminate_processes([self.cleaning_process])
            del self.cleaning_process

    def is_running(self):
        if hasattr(self, 'workers') and self.workers is not None:
            return all(worker_is_alive(w) for w in self.workers)
        return False

    def close_all_workers(self):
        """Close all the subprocess workers"""
        if hasattr(self, 'workers') and self.workers is not None:
            for index, _ in enumerate(self.workers):
                close_worker(self.workers[index], self.eof_workers[index])

            check_interval = get_multiprocessing_timeout_interval()
            for w in self.workers:
                try:
                    subprocess_file_descriptor = w.sentinel
                    st = time.time()
                    while _PythonMultiprocessing.is_process_alive(w.pid):
                        time.sleep(0.01)  # sleep 10ms, waiting for the subprocess exit
                        if time.time() - st > check_interval:
                            logger.warning(f"Waiting for the subprocess worker [{w.pid}] to exit.")
                            st += check_interval
                except ValueError as e:
                    if "process object is closed" in str(e):
                        continue
                    raise e
                try:
                    if worker_is_alive(w):
                        os.close(subprocess_file_descriptor)
                except OSError as e:
                    # Maybe the file descriptor had been released, so ignore the 'Bad file descriptor'
                    if "Bad file descriptor" not in str(e):
                        raise e

            # use clear to release the handle which is better than self.workers = None
            self.workers.clear()
            self.workers = None
            self.eof_workers.clear()
            self.eof_workers = []

            # as it can cause the main process to not exit when PyFunc executes very slowly so release
            # the shm & msg here
            cde.release_shm_and_msg_by_worker_pids(self.pids)
            self.pids = None


class MapDataset(UnionBaseDataset):
    """
    The result of applying the Map operation to the input Dataset.

    Args:
        input_dataset (Dataset): Input Dataset to be mapped.
        operations (Union[list[TensorOperation], list[functions]]): A function mapping a nested structure of tensors
            to another nested structure of tensor. Default: ``None``.
        input_columns (Union[str, list[str]]): List of names of the input columns.
            Default: ``None``, the operations will be applied on the first columns in the dataset.
            The size of the list should match the number of inputs of the first operation.
        output_columns (Union[str, list[str]], optional): List of names of the output columns.
            The size of the list should match the number of outputs of the last operation.
            Default: ``None``, output columns will be the input columns, i.e., the columns will
            be replaced.
        num_parallel_workers (int, optional): Number of workers to process the dataset
            in parallel. Default: ``None``.
        python_multiprocessing (bool, optional): Parallelize Python operations with multiple worker process. This
            option could be beneficial if the Python operation is computational heavy. Default: ``False``.
        cache (DatasetCache, optional): Use tensor caching service to speed up dataset processing.
            Default: ``None``, which means no cache is used.
        callbacks (DSCallback, list[DSCallback], optional): List of Dataset callbacks to be called. Default: ``None``.
        max_rowsize(Union[int, list[int]], optional): Maximum size of row in MB that is used for shared memory
            allocation to copy data between processes, the total occupied shared memory will increase as
            ``num_parallel_workers`` and :func:`mindspore.dataset.config.set_prefetch_size` increase. If set to -1,
            shared memory will be dynamically allocated with the actual size of data. This is only used if
            ``python_multiprocessing`` is set to True. If it is an int value, it represents ``input_columns`` and
            ``output_columns`` use this value as the unit to create shared memory. If it is a list, the first element
            represents the ``input_columns`` use this value as the unit to create shared memory, and the second element
            represents ``output_columns`` use this value as the unit to create shared memory. Default: ``None`` ,
            allocate shared memory dynamically.
        offload (bool, optional): Flag to indicate whether offload is used. Default: ``None``.
    """

    def __init__(self, input_dataset, operations=None, input_columns=None, output_columns=None,
                 num_parallel_workers=None, python_multiprocessing=False, cache=None, callbacks=None, max_rowsize=None,
                 offload=None):
        super().__init__(children=input_dataset, num_parallel_workers=num_parallel_workers, cache=cache)
        self.operations = to_list(operations)
        for op in self.operations:
            # user define vision.HWC2CHW without parentheses is error
            if type(op) == type:  # pylint: disable=unidiomatic-typecheck
                raise ValueError("Parameter operations's element of method map should be a dataset processing " +
                                 f"operation instance, but got: {op}. It may be missing parentheses for " +
                                 "instantiation.")
            if not callable(op):
                raise ValueError("Parameter operations's element of method map should be a python function or " +
                                 f"class method which should be callable, but got: {op}. It doesn't need parentheses " +
                                 "for python function or class method.")

        self.input_columns = to_list(input_columns)
        self.output_columns = to_list(output_columns)

        #  If output_columns were not provided then use input_columns
        self.output_columns = self.input_columns if not self.output_columns else self.output_columns

        self.python_multiprocessing = python_multiprocessing
        self.process_pool = None

        self.callbacks = to_list(callbacks)
        if max_rowsize is None:
            self.max_rowsize = [-1, -1]
        elif isinstance(max_rowsize, int):
            self.max_rowsize = [max_rowsize] * 2
        else:
            self.max_rowsize = max_rowsize
        self.offload = offload

    def parse(self, children=None):
        operations = self.__decompose_callable_operations()

        count_new_transforms, count_non_data_vision_transforms = self.__count_transforms(operations)
        count_py_ops = self.__count_py_ops(operations)
        count_pyfunc = self.__count_pyfuncs(operations)

        # Whether to execute ops in the thread mode
        # op_type                      python_multiprocessing  run_in_thread
        # c_op(s)                      false                   yes
        # c_op(s)                      true                    yes
        # py_op(s) / PyFunc            false                   yes
        # py_op(s) / PyFunc            true                    no
        # c_op(s) + py_op(s) / PyFunc  false                   yes
        # c_op(s) + py_op(s) / PyFunc  true                    no
        run_in_thread = not self.python_multiprocessing or (count_pyfunc == 0 and count_py_ops == 0) or get_debug_mode()

        if self.python_multiprocessing and platform.system().lower() == 'windows':
            run_in_thread = True

        if count_new_transforms + count_pyfunc == len(operations):
            prev_op = None
            for op in operations:
                # skip user added DebugHook to avoid changing to Py-implementation.
                if self.__is_debug_hook_op(op):
                    if prev_op:
                        # manually set previous_op_name
                        prev_op_name = self.__parse_op_name(prev_op)
                        op.set_previous_op_name(prev_op_name)
                    continue
                if op.implementation is None:
                    if prev_op and prev_op.implementation == Implementation.PY:
                        op.implementation = Implementation.PY
                    else:
                        op.implementation = Implementation.C
                prev_op = op
            operations = self.__insert_debug_wrapper(operations)
            if run_in_thread:
                operations = transforms.Compose.reduce(operations)
        elif count_pyfunc + count_non_data_vision_transforms == len(operations):
            operations = self.__insert_debug_wrapper(operations)
            if run_in_thread:
                operations = transforms.Compose.reduce(operations)
        else:
            raise RuntimeError("Mixing old legacy c/py_transforms and new unified transforms is not allowed.")

        if run_in_thread:
            self.operations = self.__process_final_operations(operations)
        else:
            self.operations = operations
        self.prepare_multiprocessing()

        callbacks = [cb.create_runtime_obj() for cb in self.callbacks]

        ## thread mode
        if run_in_thread:
            return cde.MapNode(children[0], self.operations, self.input_columns, self.output_columns,
                               callbacks, OffloadToManualOffloadMode.get(self.offload), self.process_pool)

        # Bind self.operations with self.process_pool
        class _BindProcessPoolWithOperations:
            def __init__(self, pool, operations):
                self.pool = pool
                self.operations = operations

            def __call__(self):
                pass

        self.bound = _BindProcessPoolWithOperations(self.process_pool, self.operations)

        ## process mode
        # in multi process mode, we just transfer the self.bound which is not really used in c layer
        # because when the pipeline is running, map thread transfer data through c++ shm & msg to Python Worker Process
        return cde.MapNode(children[0], [self.bound], self.input_columns, self.output_columns,
                           callbacks, OffloadToManualOffloadMode.get(self.offload), self.process_pool)

    def __deepcopy__(self, memodict):
        return self.__safe_deepcopy__(memodict, exclude=("operations", "callbacks", "__transfer_dataset__"))

    @staticmethod
    def __parse_op_name(op):
        """
        Utility method to get operation name.
        """
        op_name = ""
        if isinstance(op, transforms.py_transforms_util.FuncWrapper):
            try:
                op_name = op.transform.__name__
            except AttributeError:
                op_name = op.transform.__class__.__name__
        else:
            op_name = op.__class__.__name__
        return op_name

    @staticmethod
    def __construct_debug_hook(previous_op_name=None, is_first_op=False):
        """
        Wrap debug hook into FuncWrapper.
        """
        inserted_functions = []
        debug_hook_list = _get_debug_hook_list()
        if debug_hook_list:
            for fn in debug_hook_list:
                # making deep copy to allow each debug hook instance hold unique variables
                new_fn = copy.deepcopy(fn)
                new_fn.set_previous_op_name(previous_op_name)
                new_fn.set_is_first(is_first_op)
                inserted_func = transforms.py_transforms_util.FuncWrapper(new_fn)
                inserted_func.implementation = Implementation.PY
                inserted_functions.append(inserted_func)
        return inserted_functions

    @staticmethod
    def __is_debug_hook_op(op):
        """
        Check if the op is user added DebugHook and skip it to avoid changing transforms implementation.
        """
        if isinstance(op, DebugHook):
            if not get_debug_mode():
                raise ValueError("It is not allowed to inject DebugHook object in non-debug mode.")
            return True
        return False

    @staticmethod
    def __count_pyfuncs(operations):
        """
        Count the number of pyfuncs operations which is defined by user
        """
        return sum(1 if isinstance(op, FuncWrapper) else 0 for op in operations)

    @staticmethod
    def __count_py_ops(operations):
        """
        Count the number of python operations which is built-in
        """
        count = 0
        for op in operations:
            if hasattr(op, "implementation") and op.implementation != Implementation.C \
                and op.implementation is not None:
                count += 1
        return count

    @staticmethod
    def __count_transforms(operations):
        """
        Count the various flavors of transforms operations
        """
        # Count the number of old legacy data and vision c_layer_transforms and py_layer_transforms
        # Count the number of new unified data and vision transforms
        count_new_transforms = sum(1 if hasattr(op, "implementation") and not isinstance(op, FuncWrapper)
                                   else 0 for op in operations)
        # Count the number of non-data transforms and non-vision transforms
        count_non_data_vision_transforms = sum(
            1 if "text.transforms" in str(op) or "audio.transforms" in str(op) else 0 for op in operations)
        return count_new_transforms, count_non_data_vision_transforms

    @staticmethod
    def __operation_valid_for_multiprocessing(op):
        if callable(op):
            return True
        return False

    @staticmethod
    def __process_final_operations(operations):
        """
        Build final list of operations
        """
        operations_fin = []
        for op in operations:
            if hasattr(op, "implementation"):
                if op.implementation == Implementation.C and not isinstance(op, (FuncWrapper, ToNumpy)):
                    operations_fin.append(op.parse())
                elif op.implementation == Implementation.PY:
                    operations_fin.append(op)
                elif isinstance(op, (FuncWrapper, ToNumpy)):
                    operations_fin.append(op)
                else:
                    raise RuntimeError("Wrong implementation")
            else:
                if op and getattr(op, 'parse', None):
                    operations_fin.append(op.parse())
                else:
                    operations_fin.append(op)
        return operations_fin

    # Iterator bootstrap will be called on iterator construction.
    # A deep copy of Dataset object is created prior of iterator_bootstrap.
    # This method will create per iterator process pool and bind pyfunc execution to the pool.
    def prepare_multiprocessing(self):
        """
        Per iterator bootstrap callback.
        """
        if self.python_multiprocessing and platform.system().lower() == 'windows':
            logger.warning("Python multiprocessing is not supported on Windows platform.")
            return
        if self.python_multiprocessing and get_debug_mode():
            logger.warning("Python multiprocessing is not supported in debug mode."
                           " Ignoring Python multiprocessing for map operation.")
            return
        if self.python_multiprocessing:
            callable_list = []

            # If user didn't specify num_parallel_workers, set it to default
            if self.num_parallel_workers is None:
                self.num_parallel_workers = get_num_parallel_workers()

            # Pass #1, look for Python callables and build list
            for op in self.operations:
                if MapDataset.__operation_valid_for_multiprocessing(op):
                    callable_list.append(op)

            if callable_list:
                self.process_pool = _PythonMultiprocessing(get_multiprocessing_start_method(),
                                                           self.num_parallel_workers, str(self),
                                                           callable_list, self.max_rowsize)

    def __insert_debug_wrapper(self, operations):
        """
        Insert DebuggerWrapper before and after each op if debug mode is on.
        """
        if not get_debug_mode():
            return operations
        first_op_name = self.__parse_op_name(operations[0])
        inserted_operations = self.__construct_debug_hook(first_op_name, is_first_op=True)
        for op in operations:
            inserted_operations.append(op)
            op_name = self.__parse_op_name(op)
            inserted_operations.extend(self.__construct_debug_hook(op_name))
        return inserted_operations

    def __decompose_callable_operations(self):
        """
        Decompose operations and build list of old legacy ops which are callable
        """
        decomposed_operations = transforms.Compose.decompose(self.operations)
        operations = []
        for op in decomposed_operations:
            if callable(op) and not hasattr(op, "implementation"):
                op = transforms.py_transforms_util.FuncWrapper(op)
            operations.append(op)
        return operations


class FilterDataset(UnionBaseDataset):
    """
    The result of applying filter predicate to the input Dataset.

    Args:
        input_dataset (Dataset): Input Dataset to be mapped.
        predicate (callable): Python callable which returns a boolean value. If False then filter the element.
        input_columns (Union[str, list[str]], optional): List of names of the input columns.
            Default: ``None``, the predicate will be applied to all columns in the dataset.
        num_parallel_workers (int, optional): Number of workers to process the dataset
            in parallel. Default: ``None``.
    """

    def __init__(self, input_dataset, predicate, input_columns=None, num_parallel_workers=None):
        super().__init__(children=input_dataset, num_parallel_workers=num_parallel_workers)
        self.predicate = lambda *args: bool(predicate(*args))
        self.input_columns = to_list(input_columns)

    def parse(self, children=None):
        return cde.FilterNode(children[0], self.predicate, self.input_columns)


class RepeatDataset(UnionBaseDataset):
    """
    The result of applying Repeat operation to the input Dataset.

    Args:
        input_dataset (Dataset): Input Dataset to be repeated.
        count (int): Number of times the dataset will be repeated. Default: -1, repeat indefinitely.
    """

    def __init__(self, input_dataset, count):
        super().__init__(children=input_dataset)
        self.count = replace_none(count, -1)

    def parse(self, children=None):
        return cde.RepeatNode(children[0], self.count)


class SkipDataset(UnionBaseDataset):
    """
    The result of applying Skip operation to the input Dataset.

    Args:
        input_dataset (Dataset): Input dataset to have elements skipped.
        count (int): Number of elements to be skipped in the dataset.
    """

    def __init__(self, input_dataset, count):
        super().__init__(input_dataset)
        self.count = count

    def parse(self, children=None):
        return cde.SkipNode(children[0], self.count)


class TakeDataset(UnionBaseDataset):
    """
    The result of applying Take operation to the input Dataset.

    Args:
        input_dataset (Dataset): Input Dataset to have elements taken from.
        count (int): Number of elements to be taken from the dataset.
    """

    def __init__(self, input_dataset, count):
        super().__init__(children=input_dataset)
        self.count = count

    def parse(self, children=None):
        return cde.TakeNode(children[0], self.count)


class ZipDataset(UnionBaseDataset):
    """
    The result of applying Zip operation to the input Dataset.

    Args:
        datasets (tuple): A tuple of datasets to be zipped together.

    Raises:
        TypeError: If dataset is not an instance of Dataset.
    """

    def __init__(self, datasets):
        super().__init__(children=datasets)

    def parse(self, children=None):
        return cde.ZipNode(children)

    def is_sync(self):
        return any(c.is_sync() for c in self.children)


class ConcatDataset(UnionBaseDataset):
    """
    The result of applying Concat operation to the input Dataset.

    Args:
        datasets (list): A list of datasets to be concatenated together.

    Raises:
        TypeError: If dataset is not an instance of Dataset.
        ValueError: If there is no samples in the one of the datasets.
    """

    def __init__(self, datasets):
        super().__init__(children=datasets)
        for dataset in datasets:
            if not isinstance(dataset, Dataset):
                raise TypeError(f"Invalid dataset, expected Dataset object, but got {type(dataset)}!")
        self.datasets = datasets
        self._sampler = samplers.SequentialSampler(num_samples=None)

        self.children_sizes_ = [c.get_dataset_size() for c in self.children]
        child_index = 0
        for item in self.children_sizes_:
            if item == 0:
                raise ValueError(f"There are no samples in the dataset number {child_index}. " +
                                 "Please make sure there are valid samples in the dataset.")
            child_index += 1

        self._children_sizes = self.children_sizes_.copy()

        # _children_flag_and_nums: A list of pair<int ,int>.The first element of pair is flag that characterizes
        # whether the dataset is mappable. The second element of pair is length of the dataset
        self._children_flag_and_nums = []

        # _children_start_end_index_: A list of pair<int ,int>.The elements of pair are used to characterize
        # the valid position of the dataset corresponding to the subscript when sampling
        self._children_start_end_index_ = []
        for index, child in enumerate(self.children):
            tem_list = [-1, -1]
            self._children_start_end_index_.append(tem_list)
            dataset_len = self.children_sizes_[index]

            from mindspore.dataset.engine.datasets_user_defined import GeneratorDataset  \
            # pylint: disable=import-outside-toplevel
            if isinstance(child, GeneratorDataset) and not hasattr(child.source, "__getitem__"):
                dataset_len = 0
                self.children_sizes_[index] = 0

            if isinstance(child, MappableDataset):
                self._children_flag_and_nums.append((0, dataset_len))
            else:
                self._children_flag_and_nums.append((1, dataset_len))

    def parse(self, children=None):
        return cde.ConcatNode(children, self._sampler, self._children_flag_and_nums, self._children_start_end_index_,
                              self._children_sizes)

    def use_sampler(self, sampler):
        """
        Set the distributedSampler to concat dataset

        Args:
            sampler (Sampler): The sampler to use for the current dataset.
                Currently supported: DistributedSampler.

        Raises:
            TypeError: If the sampler is not an instance of DistributedSampler
            ValueError: If the parameter shuffle of sampler is True
            ValueError: If the parameter NumSamples of sampler is not None.
            ValueError: If num_shards <=0.
        """
        if not isinstance(sampler, (samplers.DistributedSampler, samplers.RandomSampler)):
            raise TypeError(f"The parameter {sampler} of concat must be DistributedSampler or RandomSampler!")

        if isinstance(sampler, samplers.RandomSampler):
            if sampler.replacement:
                raise ValueError("The parameter replacement of RandomSampler must be False!")

            if sampler.get_num_samples() is not None:
                raise ValueError("The parameter num_samples of RandomSampler is not support to be set!")

            self._sampler = sampler
            self._children_sizes = [c.get_dataset_size() for c in self.children]

            # Recursive access to other child concat nodes
            def set_child(node):
                for c in node.children:
                    if isinstance(c, ConcatDataset):
                        c.use_sampler(sampler)
                    set_child(c)

            set_child(self)

            return

        if sampler.is_shuffled():
            raise ValueError("The parameter shuffle of DistributedSampler must be False!")

        if sampler.num_shards <= 0:
            raise ValueError("The parameter num_shards of DistributedSampler must be positive int!")

        if sampler.get_num_samples() is not None:
            raise ValueError("The parameter num_samples of DistributedSampler is not support to be set!")

        self.dataset_size = None

        self._sampler = sampler
        cumulative_samples_nums = 0
        for index, child in enumerate(self.children):
            if hasattr(child, 'sampler') and child.sampler.get_num_samples() is not None:
                raise ValueError(f"The parameter NumSamples of {child} is not support to be set!")

            if isinstance(child, (BatchDataset, PaddedBatchDataset)):
                raise TypeError(f"The parameter {child} of concat must not be BatchDataset or PaddedBatchDataset!")

            # if child is mappable and the length is greater than 0
            if not self._children_flag_and_nums[index][0] and self._children_flag_and_nums[index][1]:

                tem_value = cumulative_samples_nums + self._children_flag_and_nums[index][1]

                if not self._children_flag_and_nums[index][1] >= sampler.num_shards:
                    if tem_value < sampler.num_shards:
                        self._children_start_end_index_[index][0] = cumulative_samples_nums
                        self._children_start_end_index_[index][1] = tem_value
                    else:
                        self._children_start_end_index_[index][0] = cumulative_samples_nums
                        self._children_start_end_index_[index][1] = tem_value % sampler.num_shards

                tem_sampler = copy.deepcopy(sampler)
                tem_sampler.set_offset(cumulative_samples_nums)
                child.use_sampler(tem_sampler)

            cumulative_samples_nums += self.children_sizes_[index]
            cumulative_samples_nums %= sampler.num_shards


class RenameDataset(UnionBaseDataset):
    """
    The result of applying Rename operation to the input Dataset.

    Args:
        input_dataset (Dataset): Input Dataset to be Renamed.
        input_columns (Union[str, list[str]]): List of names of the input columns.
        output_columns (Union[str, list[str]]): List of names of the output columns.
    """

    def __init__(self, input_dataset, input_columns, output_columns):
        super().__init__(children=input_dataset)
        self.input_column_names = to_list(input_columns)
        self.output_column_names = to_list(output_columns)

    def parse(self, children=None):
        return cde.RenameNode(children[0], self.input_column_names, self.output_column_names)


def to_list(items):
    if items is None:
        return []
    if isinstance(items, tuple):
        return list(items)
    if not isinstance(items, list):
        return [items]
    return items


class ProjectDataset(UnionBaseDataset):
    """
    The result of applying Project operation to the input Dataset.

    Args:
        input_dataset (Dataset): Input Dataset to be Projected.
        columns (Union[str, list[str]]): List of names of the columns to project.
    """

    def __init__(self, input_dataset, columns):
        super().__init__(children=input_dataset)
        self.columns = to_list(columns)

    def parse(self, children=None):
        return cde.ProjectNode(children[0], self.columns)


class _ToDevice:
    """
    Internal class to handle sending data to device.
    """

    def __init__(self, dataset, num_epochs):
        if get_debug_mode():
            logger.error("MindData debugger cannot be used in dataset sink mode. Please manually turn off "
                         "sink mode and try debugger again.")
        ir_tree, _ = dataset.create_ir_tree()

        self._runtime_context = cde.PythonRuntimeContext()
        self._runtime_context.Init()
        self._to_device = cde.ToDevice(num_epochs)
        if dataset.get_init_step() != 0:
            init_step = dataset.get_init_step()
            dataset_size = dataset.get_dataset_size()
            self._to_device.Init(ir_tree, init_step, dataset_size)
        else:
            self._to_device.Init(ir_tree, 0, -1)
        self._runtime_context.AssignConsumer(self._to_device)

        ITERATORS_LIST.append(weakref.ref(self))
        _unset_iterator_cleanup()

    def send(self):
        self._to_device.Send()

    def stop_send(self):
        """
        send stop send signal to pipeline, it is used when end of sequence is sent at the epoch end.
        """
        self._to_device.StopSend()

    def continue_send(self):
        """
        send continue send signal to pipeline, it is used when end of sequence is sent at the epoch end.
        """
        self._to_device.ContinueSend()

    def get_data_info(self):
        """
        Get type and shape of current batch.
        """
        return self._to_device.GetDataInfo()

    def get_mbuf_queue_size(self):
        """
        Get element numbers inside mbuf.
        """
        return self._to_device.GetMbufQueueSize()

    def get_send_info(self):
        """
        In sink mode, it returns the send information of dataset at this moment.
        Send information includes number of send batches, time summary of fetching data on host
        and time summary of sending data.
        """
        return self._to_device.GetSendInfo()

    def release(self):
        """
        Manually terminate Device Queue instead of relying on out of scope destruction.
        """
        if hasattr(self, '_runtime_context') and self._runtime_context:
            if hasattr(self, '_to_device') and self._to_device:
                self._runtime_context.Terminate()
                del self._to_device
            del self._runtime_context

    def __deepcopy__(self, memodict):
        return self

    def get_offload_model(self, col_names):
        """
        Get offload model containing removed offload ops from pipeline.
        """
        offload_model = GetOffloadModel(self._to_device, col_names)
        return offload_model

    def _reset(self, step, dataset_size):
        self._to_device.Reset(step, dataset_size)


class TransferDataset(Dataset):
    """
    The result of applying TDT operation to the input Dataset.

    Args:
        input_dataset (Dataset): Input Dataset to be transferred.
        send_epoch_end (bool, optional): Whether to send end of sequence to device or not. Default: ``True``.
        create_data_info_queue (bool, optional): Whether to create queue which stores
            types and shapes of data or not. Default: ``False``.

    Raises:
        TypeError: If device_type is empty.
        ValueError: If device_type is not 'Ascend', 'GPU' or 'CPU'.
        RuntimeError: If dataset is unknown.
    """

    def __init__(self, input_dataset, send_epoch_end=True, create_data_info_queue=False, queue_name=""):
        super().__init__(children=input_dataset)
        if queue_name == "":
            self.queue_name = str(uuid.uuid1())
            logger.info(f"queue_name is newly generated. value is {self.queue_name}")
        else:
            self.queue_name = queue_name
            logger.info(f"queue_name is read from compile cache. value is {self.queue_name}")
        self.device_type = context.get_context("device_target") if context else "CPU"
        self.device_id = context.get_context("device_id") if context else 0

        self._send_epoch_end = replace_none(send_epoch_end, True)
        self._create_data_info_queue = create_data_info_queue
        self._to_device = None
        self.column_name = input_dataset.get_col_names()

    def parse(self, children=None):
        total_batch = 0
        if hasattr(self.children[0], "__total_batch__"):
            total_batch = self.children[0].__total_batch__
            check_total_batch(total_batch)
        return cde.DataQueueNode(children[0], self.queue_name, self.device_type, self.device_id, self._send_epoch_end,
                                 total_batch, self._create_data_info_queue)

    def create_dict_iterator(self, num_epochs=-1, output_numpy=False):
        raise RuntimeError("TransferDataset is not iterable.")

    def create_tuple_iterator(self, columns=None, num_epochs=-1, output_numpy=False, do_copy=False):
        raise RuntimeError("TransferDataset is not iterable.")

    def __iter__(self):
        raise RuntimeError("TransferDataset is not iterable.")

    def output_shapes(self):
        raise RuntimeError("TransferDataset does not support obtaining output_shapes.")

    def output_types(self):
        raise RuntimeError("TransferDataset does not support obtaining output_types.")

    @check_to_device_send
    def send(self, num_epochs=-1):  # pylint: disable=huawei-arguments-differ
        """
        Send to device
        """
        if Dataset._noop_mode():
            return
        if self._to_device is not None:
            del self._to_device
        self._to_device = _ToDevice(self, num_epochs)
        self._to_device.send()

    def stop_send(self):
        if self._to_device is not None:
            self._to_device.stop_send()

    def continue_send(self):
        if self._to_device is not None:
            self._to_device.continue_send()

    def get_data_info(self):
        """
        Get type and shape of current batch
        """
        if self._to_device is not None:
            return self._to_device.get_data_info()
        raise RuntimeError("Calling get_data_info with bad state.")

    def get_mbuf_queue_size(self):
        """
        Get element numbers inside mbuf.
        """
        if self._to_device is not None:
            return self._to_device.get_mbuf_queue_size()
        raise RuntimeError("Device queue is not init, call get_mbuf_queue_size failed.")

    def get_send_info(self):
        """
        In sink mode, it returns the send information of dataset at this moment.
        Send information includes number of send batches, time summary of fetching data on host
        and time summary of sending data.
        """
        if self._to_device is not None:
            return self._to_device.get_send_info()
        raise RuntimeError("Calling get_send_info with bad state, data queue is not initialized.")

    def get_offload_model(self):
        if self._to_device is not None:
            return self._to_device.get_offload_model(self.column_name)

        raise RuntimeError("get_offload_model, _to_device is None")

    def release(self):
        """
        Manually terminate Device Queue instead of relying on out of scope destruction.
        """
        if self._to_device is not None:
            self._to_device.release()

    def _reset(self, step, dataset_size):
        if self._to_device is not None:
            logger.info("Reset the dataset pipeline to step: " + str(step) + ", epoch: " + str(step // dataset_size))
            self._to_device._reset(step, dataset_size)  # pylint: disable=protected-access


class Schema:
    """
    Class to represent a schema of a dataset.

    Args:
        schema_file (str, optional): Path of the schema file. Default: ``None``.

    Raises:
        RuntimeError: If schema file failed to load.

    Examples:
        >>> import mindspore.dataset as ds
        >>> from mindspore import dtype as mstype
        >>>
        >>> # Create schema; specify column name, mindspore.dtype and shape of the column
        >>> schema = ds.Schema()
        >>> schema.add_column(name='col1', de_type=mstype.int64, shape=[2])
    """

    @check_schema
    def __init__(self, schema_file=None):
        self.schema_file = replace_none(schema_file, "")
        self.cpp_schema = cde.SchemaObj(self.schema_file)

    @check_add_column
    def add_column(self, name, de_type, shape=None):
        """
        Add new column to the schema.

        Args:
            name (str): The new name of the column.
            de_type (str): Data type of the column.
            shape (list[int], optional): Shape of the column.
                Default: ``None``, [-1] which is an unknown shape of rank 1.

        Raises:
            ValueError: If column type is unknown.

        Examples:
            >>> import mindspore.dataset as ds
            >>> from mindspore import dtype as mstype
            >>>
            >>> schema = ds.Schema()
            >>> schema.add_column('col_1d', de_type=mstype.int64, shape=[2])
        """
        if isinstance(de_type, typing.Type):
            de_type = mstype_to_detype(de_type)
            col_type = str(de_type)
        else:
            col_type = str(cde.DataType(de_type))
        if shape is None:
            self.cpp_schema.add_column(name, col_type)
        else:
            self.cpp_schema.add_column(name, col_type, shape)

    def parse_columns(self, columns):
        """
        Parse the columns and add them to the schema.

        Args:
            columns (Union[dict, list[dict], tuple[dict]]): Dataset attribute information, decoded from schema file.

                - list[dict], `name` and `type` must be in keys, `shape` optional.

                - dict, columns.keys() as name, columns.values() is dict, and `type` inside, `shape` optional.

        Raises:
            RuntimeError: If failed to parse columns.
            RuntimeError: If column's name field is missing.
            RuntimeError: If column's type field is missing.

        Examples:
            >>> from mindspore.dataset import Schema
            >>> schema = Schema()
            >>> columns1 = [{'name': 'image', 'type': 'int8', 'shape': [3, 3]},
            ...             {'name': 'label', 'type': 'int8', 'shape': [1]}]
            >>> schema.parse_columns(columns1)
            >>> columns2 = {'image': {'shape': [3, 3], 'type': 'int8'}, 'label': {'shape': [1], 'type': 'int8'}}
            >>> schema.parse_columns(columns2)
        """
        self.cpp_schema.parse_columns(json.dumps(columns, indent=2))

    def to_json(self):
        """
        Get a JSON string of the schema.

        Returns:
            str, JSON string of the schema.

        Examples:
            >>> from mindspore.dataset import Schema
            >>> from mindspore import dtype as mstype
            >>>
            >>> schema = Schema()
            >>> schema.add_column('col_1d', de_type=mstype.int64, shape=[2])
            >>> json = schema.to_json()
        """
        return self.cpp_schema.to_json()

    def from_json(self, json_obj):
        """
        Get schema file from JSON object.

        Args:
            json_obj (dict): Object of JSON parsed.

        Raises:
            RuntimeError: If there is an unknown item in the object.
            RuntimeError: If dataset type is missing in the object.
            RuntimeError: If columns are missing in the object.

        Examples:
            >>> import json
            >>> from mindspore.dataset import Schema
            >>>
            >>> with open("/path/to/schema_file", "r") as file:
            ...     json_obj = json.load(file)
            ...     schema = Schema()
            ...     schema.from_json(json_obj)
        """
        self.cpp_schema.from_string(json.dumps(json_obj, indent=2))

    def __str__(self):
        return self.to_json()

    @staticmethod
    def get_num_rows(schema):
        schema_obj = schema
        if not isinstance(schema_obj, Schema):
            schema_obj = Schema(schema_obj)
        return schema_obj.cpp_schema.get_num_rows()


class DeserializedDataset(Dataset):
    def __init__(self, input_obj):
        super().__init__()
        self.input_obj = input_obj

    def parse(self, children=None):
        if isinstance(self.input_obj, dict):
            json_str = json.dumps(self.input_obj)
            return cde.Dataset.from_json_string(json_str)
        return cde.Dataset.from_json_file(self.input_obj)
