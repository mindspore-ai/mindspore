# Copyright 2022-2023 Huawei Technologies Co., Ltd
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
datasets_audio.py, datasets_standard_format.py and dataets_user_defined.py files.
    datasets_vision.py: contains vision dataset loading classes.
    datasets_text.py: contains text dataset loading classes.
    datasets_audio.py: contains audio dataset loading classes.
    datasets_standard_format.py: contains standard format loading classes which
                                 any other kinds of datasets can be converted to.
    dataets_user_defined.py: contains basic classes that help users to define
                             flexible ways to load dataset.
"""
import atexit
import glob
import json
import os
import signal
import stat

import gc
import time
import uuid
import multiprocessing
from enum import Enum
from importlib import import_module
import sys
import threading

import copy
import weakref
import platform
import psutil

import mindspore._c_dataengine as cde
from mindspore._c_expression import typing

from mindspore import log as logger
from mindspore.parallel._ps_context import _is_role_pserver, _is_role_sched, _get_ps_context,\
                                           _enable_distributed_mindrt
from mindspore.dataset.engine.offload import GetOffloadModel

import mindspore.dataset.transforms.c_transforms as c_transforms
import mindspore.dataset.transforms.py_transforms as py_transforms
import mindspore.dataset.transforms as transforms
from mindspore.dataset.text.utils import SentencePieceModel, DE_C_INTER_SENTENCEPIECE_MODE
from mindspore.parallel._utils import _get_device_num
from mindspore.dataset.engine.debug import DebugWrapper

from . import samplers
from .iterators import DictIterator, TupleIterator, DummyIterator, check_iterator_cleanup, _set_iterator_cleanup, \
    ITERATORS_LIST, _unset_iterator_cleanup
from .queue import _SharedQueue, _Queue
from .validators import check_batch, check_shuffle, check_map, check_filter, check_repeat, check_skip, check_zip, \
    check_rename, check_device_send, check_take, check_output_shape, check_project, \
    check_sync_wait, check_zip_dataset, check_add_column, check_concat, check_split, check_bucket_batch_by_length, \
    check_save, check_tuple_iterator, check_dict_iterator, check_schema, check_to_device_send, check_padded_batch
from ..core.config import get_callback_timeout, _init_device_info, get_enable_shared_mem, get_num_parallel_workers, \
    get_enable_watchdog, get_seed, set_seed, get_debug_mode, get_multiprocessing_timeout_interval
from ..core.datatypes import mstype_to_detype
from ..core.validator_helpers import replace_none
from ..core.py_util_helpers import ExceptionHandler
from ..transforms.py_transforms_util import FuncWrapper, Implementation
from ..vision.transforms import ToNumpy

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


def _reset_training_dataset(step, epoch):
    """
    Reset the training dataset to the given step and epoch number.

    Args:
        step (int): Global step number.
        epoch (int): Global epoch number
    """
    dataset = _get_training_dataset()
    if dataset is not None:
        dataset._reset(step, epoch)  # pylint: disable=protected-access
    else:
        raise RuntimeError("Training dataset is not set.")


class Shuffle(str, Enum):
    """Specify the shuffle mode.

    - Shuffle.GLOBAL: Shuffle both the files and samples.
    - Shuffle.FILES: Shuffle files only.
    - Shuffle.INFILE: Shuffle data within each file.
    """
    GLOBAL: str = "global"
    FILES: str = "files"
    INFILE: str = "infile"


ShuffleToShuffleMode = {Shuffle.FILES: cde.ShuffleMode.FILES,
                        Shuffle.GLOBAL: cde.ShuffleMode.GLOBAL,
                        Shuffle.INFILE: cde.ShuffleMode.INFILE}


def shuffle_to_shuffle_mode(shuffle):
    """
    Shuffle Enum to Shuffle Mode

    Args:
        shuffle (Shuffle): shuffle flag to shuffle mode in C layer

    Returns:
        ShuffleMode, shuffle mode
    """
    shuffle_mode = cde.ShuffleMode.GLOBAL  # Global shuffle
    if not isinstance(shuffle, Shuffle):
        if shuffle is None or shuffle:
            shuffle_mode = cde.ShuffleMode.GLOBAL  # Global shuffle
        else:
            shuffle_mode = cde.ShuffleMode.FALSE  # No shuffle
    else:
        shuffle_mode = ShuffleToShuffleMode[shuffle]
    return shuffle_mode


def shuffle_to_bool(shuffle):
    """
    Shuffle Enum to bool

    Args:
        shuffle (Shuffle): shuffle flag to bool

    Returns:
        bool, True / False
    """
    if shuffle is not None and not isinstance(shuffle, (bool, Shuffle)):
        raise TypeError("shuffle must be of boolean or enum of 'Shuffle' values like 'Shuffle.GLOBAL' or "
                        "'Shuffle.FILES' or 'Shuffle.INFILE'.")

    shuffle_bool = True
    if not isinstance(shuffle, Shuffle):
        if shuffle is None:
            shuffle_bool = None
        elif shuffle:
            shuffle_bool = True
        else:
            shuffle_bool = False
    else:
        shuffle_bool = True
    return shuffle_bool


@check_zip
def zip(datasets):
    """
    Zip the datasets in the input tuple of datasets.

    Args:
        datasets (tuple[Dataset]): A tuple of datasets to be zipped together.
            The number of datasets must be more than 1.

    Returns:
        Dataset, dataset zipped.

    Raises:
        ValueError: If the number of datasets is 1.
        TypeError: If datasets is not a tuple.

    Examples:
            >>> # Create a dataset which is the combination of dataset_1 and dataset_2
            >>> dataset = ds.zip((dataset_1, dataset_2))
    """
    if len(datasets) <= 1:
        raise ValueError(
            "Can't zip empty or just one dataset!")
    for dataset in datasets:
        if not isinstance(dataset, Dataset):
            raise TypeError("Invalid dataset, expected Dataset object, but got %s!" % type(dataset))
    return ZipDataset(datasets)


def _get_operator_process():
    """
    Inner implemented method, mainly for passing sub-process id in C layer

    Returns:
         dict, mapping dict of operation id and corresponding process id.
    """
    global _OP_PROCESS
    process_info = _OP_PROCESS
    op_process = dict()
    keys = process_info.keys()
    fetched_all = True
    for key in keys:
        try:
            op_process[key] = list(process_info[key][1])
            item_full = (len(process_info[key][1]) == process_info[key][0])
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
        paths = ["{}{}".format(file_name, str(x).rjust(num_digits, '0')) for x in range(num_files)]

    for item in paths:
        if os.path.exists(item):
            os.chmod(item, stat.S_IRUSR | stat.S_IWUSR)
            index_file = item + ".db"
            if os.path.exists(index_file):
                os.chmod(index_file, stat.S_IRUSR | stat.S_IWUSR)


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
            Default: None.
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

    @staticmethod
    def _get_operator_id(dataset):
        """
        Internal method to iterate the tree and obtain op_id of each operation.

        Returns:
            Dataset, the root dataset of the tree.
        """
        op_name = dict()
        generator_process = dict()
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

                    from mindspore.dataset.engine.datasets_user_defined import GeneratorDataset
                    if isinstance(d, GeneratorDataset) and d.sample_fn and d.sample_fn.pids:
                        generator_process[operator_id] = [d.num_parallel_workers, set(d.sample_fn.pids)]

            operator_id = operator_id + 1
            return process_name(temp, operator_id)

        process_name([dataset], op_id)
        if generator_process:
            global _OP_PROCESS
            _OP_PROCESS.update(generator_process)
        return op_name

    def create_ir_tree(self):
        """
        Internal method to build an IR tree.

        Returns:
            DatasetNode, the root node of the IR tree.
            Dataset, the root dataset of the IR tree.
        """
        parent = self.parent
        self.parent = []
        dataset = copy.deepcopy(self)
        global _OP_NAME
        _OP_NAME = Dataset._get_operator_id(dataset)
        ir_tree = dataset.parse_tree()
        self.parent = parent
        _init_device_info()
        return ir_tree, dataset

    def parse_tree(self):
        """
        Internal method to parse the API tree into an IR tree.

        Returns:
            DatasetNode, the root node of the IR tree.
        """
        if len(self.parent) > 1:
            raise ValueError("The data pipeline is not a tree (i.e., one node has 2 consumers)")
        ir_children = [d.parse_tree() for d in self.children]
        # Bootstrap can only be performed on a copy of the original dataset node.
        # Bootstrap on original dataset node will make all iterators share the same process pool
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
            filename (str): filename of JSON file to be saved as. Default: ''.

        Returns:
            str, JSON string of the pipeline.

        Examples:
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
                provided, parameter M the len(column_names) must be 1, and the size of the first
                dimension of that column will be taken as the length. Default: None.
            pad_info (dict, optional): The information about how to batch each column. The key
                corresponds to the column name, and the value must be a tuple of 2 elements.
                The first element corresponds to the shape to pad to, and the second
                element corresponds to the value to pad with. If a column is not
                specified, then that column will be padded to the longest in the current
                batch, and 0 will be used as the padding value. Any None dimensions will
                be padded to the longest in the current batch, unless if
                pad_to_bucket_boundary is True. If no padding is wanted, set pad_info
                to None. Default: None.
            pad_to_bucket_boundary (bool, optional): If True, will pad each None
                dimension in pad_info to the bucket_boundary minus 1. If there are any
                elements that fall into the last bucket, an error will occur.
                Default: False.
            drop_remainder (bool, optional): If True, will drop the last batch for each
                bucket if it is not a full batch. Default: False.

        Returns:
            Dataset, dataset bucketed and batched by length.

        Examples:
            >>> # Create a dataset where certain counts rows are combined into a batch
            >>> # and drops the last incomplete batch if there is one.
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
        Combine batch_size number of consecutive rows into batch which apply per_batch_map to the samples first.

        For any column, all the elements within that column must have the same shape.

        Refer to the following figure for the execution process:

        .. image:: batch_en.png

        Note:
            The order of using repeat and batch reflects the number of batches and per_batch_map.
            It is recommended that the repeat operation applied after the batch operation finished.

        Args:
            batch_size (Union[int, Callable]): The number of rows each batch is created with. An
                int or callable object which takes exactly 1 parameter, BatchInfo.
            drop_remainder (bool, optional): Determines whether or not to drop the last block
                whose data row number is less than batch size. Default: False. If True, and if there are less
                than batch_size rows available to make the last batch, then those rows will
                be dropped and not propagated to the child node.
            num_parallel_workers (int, optional): Number of workers(threads) to process the dataset in parallel.
                Default: None.
            **kwargs:

                - per_batch_map (Callable[[List[numpy.ndarray], ..., List[numpy.ndarray], BatchInfo], \
                  (List[numpy.ndarray], ..., List[numpy.ndarray])], optional): Per batch map callable. Default: None.
                  A callable which takes (List[numpy.ndarray], ..., List[numpy.ndarray], BatchInfo) as input parameters.
                  Each list[numpy.ndarray] represents a batch of numpy.ndarray on a given column. The number of lists
                  should match with the number of entries in input_columns. The last parameter of the callable should
                  always be a BatchInfo object. Per_batch_map should return
                  (list[numpy.ndarray], list[numpy.ndarray], ...). The length of each list in output should be the same
                  as the input. output_columns is required if the number of output lists is different from input.

                - input_columns (Union[str, list[str]], optional): List of names of the input columns. The size of
                  the list should match with signature of per_batch_map callable. Default: None.

                - output_columns (Union[str, list[str]], optional): List of names assigned to the columns
                  outputted by the last operation. This parameter is mandatory if len(input_columns) !=
                  len(output_columns). The size of this list must match the number of output
                  columns of the last operation. Default: None, output columns will have the same
                  name as the input columns, i.e., the columns will be replaced.

                - python_multiprocessing (bool, optional): Parallelize Python function per_batch_map with
                  multi-processing. This option could be beneficial if the function is computational heavy.
                  Default: False.

                - max_rowsize(int, optional): Maximum size of row in MB that is used for shared memory allocation to
                  copy data between processes. This is only used if python_multiprocessing is set to True. Default: 16.

        Returns:
            BatchDataset, dataset batched.

        Examples:
            >>> # 1) Create a dataset where every 100 rows are combined into a batch
            >>> # and drops the last incomplete batch if there is one.
            >>> dataset = dataset.batch(100, True)
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
        Combine batch_size number of consecutive rows into batch which apply pad_info to the samples first.

        Refer to the following figure for the execution process:

        .. image:: padded_batch_en.png

        Note:
            The order of using repeat and padded_batch reflects the number of batches.
            It is recommended that the repeat operation applied after the padded_batch operation finished.

        Args:
            batch_size (Union[int, Callable]): The number of rows each batch is created with. An
                int or callable object which takes exactly 1 parameter, BatchInfo.
            drop_remainder (bool, optional): Determines whether or not to drop the last block
                whose data row number is less than batch size. Default: False. If True, and if there are less
                than batch_size rows available to make the last batch, then those rows will
                be dropped and not propagated to the child node.
            num_parallel_workers (int, optional): Number of workers(threads) to process the dataset in parallel.
                Default: None.
            pad_info (dict, optional): The information about how to batch each column. The key
                corresponds to the column name, and the value must be a tuple of 2 elements.
                The first element corresponds to the shape to pad to, and the second
                element corresponds to the value to pad with. If a column is not
                specified, then that column will be padded to the longest in the current
                batch, and 0 will be used as the padding value. Any None dimensions will
                be padded to the longest in the current batch, unless if
                pad_to_bucket_boundary is True. If no padding is wanted, set pad_info
                to None. Default: None.

        Returns:
            PaddedBatchDataset, dataset batched.

        Examples:
            >>> # 1) Pad every sample to the largest sample's shape and batch the samples
            >>> dataset = dataset.padded_batch(100, True, pad_info={})
            >>>
            >>> # 2) Create a dataset where every 100 rows are combined into a batch
            >>> # and drops the last incomplete batch if there is one.
            >>> dataset = dataset.padded_batch(100, True)
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
            num_batch (int): the number of batches without blocking at the start of each epoch. Default: 1.
            callback (function): The callback function that will be invoked when sync_update is called. Default: None.

        Returns:
            SyncWaitDataset, dataset added a blocking condition.

        Raises:
            RuntimeError: If condition name already exists.

        Examples:
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
            Dataset, dataset shuffled.

        Raises:
            RuntimeError: If exist sync operations before shuffle.

        Examples:
            >>> # dataset is an instance object of Dataset
            >>> # Optionally set the seed for the first epoch
            >>> ds.config.set_seed(58)
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
            Dataset, dataset applied by the function.

        Examples:
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

        Raises:
            TypeError: If `func` is not a function.
            TypeError: If `func` doesn't return a Dataset.
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
    def map(self, operations, input_columns=None, output_columns=None, column_order=None,
            num_parallel_workers=None, **kwargs):
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
                input columns expected by the first operation. Default: None, the first
                operation will be passed however many columns that are required, starting from
                the first column.
            output_columns (Union[str, list[str]], optional): List of names assigned to the columns outputted by
                the last operation. This parameter is mandatory if len(input_columns) !=
                len(output_columns). The size of this list must match the number of output
                columns of the last operation. Default: None, output columns will have the same
                name as the input columns, i.e., the columns will be replaced.
            num_parallel_workers (int, optional): Number of threads used to process the dataset in
                parallel. Default: None, the value from the configuration will be used.
            **kwargs:

                - python_multiprocessing (bool, optional): Parallelize Python operations with multiple worker processes.
                  This option could be beneficial if the Python operation is computational heavy. Default: False.

                - max_rowsize (int, optional): Maximum size of row in MB that is used for shared memory allocation to
                  copy data between processes.  This is only used if python_multiprocessing is set to True. Default: 16.

                - cache (DatasetCache, optional): Use tensor caching service to speed up dataset processing.
                  Default: None, which means no cache is used.

                - callbacks (DSCallback, list[DSCallback], optional): List of Dataset callbacks to be called.
                  Default: None.

                - offload (bool, optional): Flag to indicate whether offload is used. Default: None.

        Note:
            - Input `operations` accepts TensorOperations defined in mindspore.dataset part, plus user-defined
              Python functions (PyFuncs).
            - Do not add network computing operators from mindspore.nn and mindspore.ops or others into this
              `operations` .

        Returns:
            Dataset, dataset after mapping operation.

        Examples:
            >>> # dataset is an instance of Dataset which has 2 columns, "image" and "label".
            >>> # image is of type bytes type which can be decoded to RGB
            >>> # label is of type int32
            >>>
            >>> # Define two operations, where each operation accepts 1 input column and outputs 1 column.
            >>> decode_op = c_vision.Decode(rgb=True)
            >>> random_jitter_op = c_vision.RandomColorAdjust(brightness=(0.8, 0.8), contrast=(1, 1),
            ...                                               saturation=(1, 1), hue=(0, 0))
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
        if hasattr(self, 'operator_mixed') and getattr(self, 'operator_mixed') is True:
            num_parallel_workers = 1
            logger.warning(
                "Input 'operations' of 'map' includes network computing operators like in mindspore.nn, mindspore.ops, "
                "mindspore.numpy module and etc, which do not support multi-thread compiling, recommend to replace it "
                "with python implemented operator like numpy etc. Here decrease 'num_parallel_workers' into 1.")

        return MapDataset(self, operations, input_columns, output_columns, num_parallel_workers, **kwargs)

    @check_filter
    def filter(self, predicate, input_columns=None, num_parallel_workers=None):
        """
        Filter dataset by prediction.

        Args:
            predicate (callable): Python callable which returns a boolean value. If False then filter the element.
            input_columns (Union[str, list[str]], optional): List of names of the input columns. If not provided
                or provided with None, the predicate will be applied on all columns in the dataset. Default: None.
            num_parallel_workers (int, optional): Number of workers to process the dataset
                in parallel. Default: None.

        Returns:
            Dataset, dataset filtered.

        Examples:
            >>> # generator data(0 ~ 63)
            >>> # filter the data that greater than or equal to 11
            >>> dataset = dataset.filter(predicate=lambda data: data < 11, input_columns = ["data"])
        """
        return FilterDataset(self, predicate, input_columns, num_parallel_workers)

    @check_repeat
    def repeat(self, count=None):
        """
        Repeat this dataset `count` times. Repeat infinitely if the count is None or -1.

        Note:
            The order of using repeat and batch reflects the number of batches. It is recommended that
            the repeat operation is used after the batch operation.

        Args:
            count (int): Number of times the dataset is going to be repeated. Default: None.

        Returns:
            Dataset, dataset repeated.

        Examples:
            >>> # dataset is an instance object of Dataset
            >>>
            >>> # Create a dataset where the dataset is repeated for 50 epochs
            >>> dataset = dataset.repeat(50)
            >>>
            >>> # Create a dataset where each epoch is shuffled individually
            >>> dataset = dataset.shuffle(10)
            >>> dataset = dataset.repeat(50)
            >>>
            >>> # Create a dataset where the dataset is first repeated for
            >>> # 50 epochs before shuffling. The shuffle operation will treat
            >>> # the entire 50 epochs as one big dataset.
            >>> dataset = dataset.repeat(50)
            >>> dataset = dataset.shuffle(10)
        """
        return RepeatDataset(self, count)

    @check_skip
    def skip(self, count):
        """
        Skip the first N elements of this dataset.

        Args:
            count (int): Number of elements in the dataset to be skipped.

        Returns:
            Dataset, dataset that containing rows like origin rows subtract skipped rows.

        Examples:
            >>> # dataset is an instance object of Dataset
            >>> # Create a dataset which skips first 3 elements from data
            >>> dataset = dataset.skip(3)
        """
        return SkipDataset(self, count)

    @check_take
    def take(self, count=-1):
        """
        Takes at most given numbers of elements from the dataset.

        Note:
            1. If count is greater than the number of elements in the dataset or equal to -1,
               all the elements in dataset will be taken.
            2. The order of using take and batch matters. If take is before batch operation,
               then take the given number of rows; otherwise take the given number of batches.

        Args:
            count (int, optional): Number of elements to be taken from the dataset. Default: -1.

        Returns:
            Dataset, dataset taken.

        Examples:
            >>> # dataset is an instance object of Dataset
            >>> # Create a dataset where the dataset includes 50 elements.
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
                raise RuntimeError("Sum of split sizes {} is not equal to dataset size {}."
                                   .format(sizes_sum, dataset_size))
            return sizes

        absolute_sizes = []
        for item in sizes:
            absolute_size = int(round(item * dataset_size))
            if absolute_size == 0:
                raise RuntimeError("Split percentage {} is too small.".format(item))
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
            raise RuntimeError("Sum of calculated split sizes {} is not equal to dataset size {}."
                               .format(absolute_sizes_sum, dataset_size))

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
                Datasets of size round(f1*K), round(f2*K), …, round(fn*K) where K is the size of the
                original dataset.
                If after rounding:

                - Any size equals 0, an error will occur.
                - The sum of split sizes < K, the difference of K - sigma(round(fi * k)) will be added to the first
                  split.
                - The sum of split sizes > K, the difference of sigma(round(fi * K)) - K will be removed from the first
                  large enough split such that it will have at least 1 row after removing the difference.

            randomize (bool, optional): Determines whether or not to split the data randomly. Default: True.
                If True, the data will be randomly split. Otherwise, each split will be created with
                consecutive rows from the dataset.

        Note:
            1. Dataset cannot be sharded if split is going to be called.
            2. It is strongly recommended to not shuffle the dataset, but use randomize=True instead.
               Shuffling the dataset may not be deterministic, which means the data in each split
               will be different in each epoch.

        Returns:
            tuple(Dataset), a tuple of datasets that have been split.

        Raises:
            RuntimeError: If get_dataset_size returns None or is not supported for this dataset.
            RuntimeError: If `sizes` is list of integers and sum of all elements in sizes does not
                equal the dataset size.
            RuntimeError: If `sizes` is list of float and there is a split with size 0 after calculations.
            RuntimeError: If the dataset is sharded prior to calling split.
            ValueError: If `sizes` is list of float and not all floats are between 0 and 1, or if the
                floats don't sum to 1.

        Examples:
            >>> # TextFileDataset is not a mappable dataset, so this non-optimized split will be called.
            >>> # Since many datasets have shuffle on by default, set shuffle to False if split will be called!
            >>> dataset = ds.TextFileDataset(text_file_dataset_dir, shuffle=False)
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
        name.

        Args:
            datasets (Union[Dataset, tuple[Dataset]]): A tuple of datasets or a single class Dataset
                to be zipped together with this dataset.

        Returns:
            Dataset, dataset zipped.

        Raises:
            TypeError: The parameter is not dataset object or tuple of dataset objects.

        Examples:
            >>> # Create a dataset which is the combination of dataset and dataset_1
            >>> dataset = dataset.zip(dataset_1)
        """
        if isinstance(datasets, tuple):
            datasets = (self, *datasets)
        elif isinstance(datasets, Dataset):
            datasets = (self, datasets)
        else:
            raise TypeError("Invalid datasets, expected Dataset object or tuple of Dataset, but got %s!" % datasets)
        return ZipDataset(datasets)

    @check_concat
    def concat(self, datasets):
        """
        Concatenate the dataset objects in the input list.
        Performing "+" operation on dataset objects can achieve the same effect.

        Note:
            The column name, and rank and type of the column data must be the same in the input datasets.

        Args:
            datasets (Union[list, Dataset]): A list of datasets or a single class Dataset
                to be concatenated together with this dataset.

        Returns:
            Dataset, dataset concatenated.

        Examples:
            >>> # Create a dataset by concatenating dataset_1 and dataset_2 with "+" operator
            >>> dataset = dataset_1 + dataset_2
            >>> # Create a dataset by concatenating dataset_1 and dataset_2 with concat operation
            >>> dataset = dataset_1.concat(dataset_2)
        """
        if isinstance(datasets, Dataset):
            datasets = [self] + [datasets]
        elif isinstance(datasets, list):
            datasets = [self] + datasets
        else:
            raise TypeError("Invalid datasets, expected Dataset object or list of Dataset, but got %s!" % datasets)
        return ConcatDataset(datasets)

    @check_rename
    def rename(self, input_columns, output_columns):
        """
        Rename the columns in input datasets.

        Args:
            input_columns (Union[str, list[str]]): List of names of the input columns.
            output_columns (Union[str, list[str]]): List of names of the output columns.

        Returns:
            Dataset, dataset renamed.

        Examples:
            >>> # dataset is an instance object of Dataset
            >>> input_columns = ["input_col1", "input_col2", "input_col3"]
            >>> output_columns = ["output_col1", "output_col2", "output_col3"]
            >>>
            >>> # Create a dataset where input_col1 is renamed to output_col1, and
            >>> # input_col2 is renamed to output_col2, and input_col3 is renamed
            >>> # to output_col3.
            >>> dataset = dataset.rename(input_columns=input_columns, output_columns=output_columns)
        """

        return RenameDataset(self, input_columns, output_columns)

    @check_project
    def project(self, columns):
        """
        The specified columns will be selected from the dataset and passed into
        the pipeline with the order specified. The other columns are discarded.

        Args:
            columns(Union[str, list[str]]): List of names of the columns to project.

        Returns:
            Dataset, dataset projected.

        Examples:
            >>> # dataset is an instance object of Dataset
            >>> columns_to_project = ["column3", "column1", "column2"]
            >>>
            >>> # Create a dataset that consists of column3, column1, column2
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
            Dataset, dataset applied by the function.

        Examples:
            >>> # dataset is an instance object of Dataset
            >>>
            >>> # Declare an apply_func function which returns a Dataset object
            >>> def apply_func(data):
            ...     data = data.batch(2)
            ...     return data
            >>>
            >>> # Use apply to call apply_func
            >>> dataset = dataset.apply(apply_func)

        Raises:
            TypeError: If apply_func is not a function.
            TypeError: If apply_func doesn't return a Dataset.
        """

        if not hasattr(apply_func, '__call__'):
            raise TypeError("apply_func must be a function.")

        dataset = apply_func(self)
        if not isinstance(dataset, Dataset):
            raise TypeError("apply_func must return a dataset.")
        return dataset

    @check_device_send
    def device_que(self, send_epoch_end=True, create_data_info_queue=False):
        """
        Return a transferred Dataset that transfers data through a device.

        Args:
            send_epoch_end (bool, optional): Whether to send end of sequence to device or not. Default: True.
            create_data_info_queue (bool, optional): Whether to create queue which stores
                types and shapes of data or not. Default: False.

        Note:
            If device is Ascend, features of data will be transferred one by one. The limitation
            of data transmission per time is 256M.

        Returns:
            Dataset, dataset for transferring.

        Examples:
            >>> import time
            >>>
            >>> data = ds.TFRecordDataset('/path/to/TF_FILES', '/path/to/TF_SCHEMA_FILE', shuffle=ds.Shuffle.FILES)
            >>>
            >>> data = data.device_que()
            >>> data.send()
            >>> time.sleep(0.1)
            >>> data.stop_send()
        """
        return TransferDataset(self, send_epoch_end, create_data_info_queue)

    @check_save
    def save(self, file_name, num_files=1, file_type='mindrecord'):
        """
        Save the dynamic data processed by the dataset pipeline in common dataset format.
        Supported dataset formats: `mindrecord` only. And you can use `MindDataset` API to read the saved file(s).

        Implicit type casting exists when saving data as `mindrecord` . The transform table shows how to do
        type casting.

        .. list-table:: Implicit Type Casting when Saving as `mindrecord`
           :widths: 25 25 50
           :header-rows: 1

           * - Type in `dataset`
             - Type in `mindrecord`
             - Details
           * - bool
             - None
             - Not supported
           * - int8
             - int32
             -
           * - uint8
             - bytes(1D uint8)
             - Drop dimension
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
             - None
             - Not supported
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

        Note:
            1. To save the samples in order, set dataset's shuffle to False and num_files to 1.
            2. Before calling the function, do not use batch operation, repeat operation or data augmentation operations
               with random attribute in map operation.
            3. When array dimension is variable, one-dimensional arrays or
               multi-dimensional arrays with variable dimension 0 are supported.
            4. Mindrecord does not support uint64, multi-dimensional uint8(drop dimension) nor
               multi-dimensional string.

        Args:
            file_name (str): Path to dataset file.
            num_files (int, optional): Number of dataset files. Default: 1.
            file_type (str, optional): Dataset format. Default: 'mindrecord'.

        Examples:
            >>> import numpy as np
            >>>
            >>> def generator_1d():
            ...     for i in range(10):
            ...         yield (np.array([i]),)
            >>>
            >>>
            >>> # apply dataset operations
            >>> d1 = ds.GeneratorDataset(generator_1d, ["data"], shuffle=False)
            >>> d1.save('/path/to/save_file')
        """
        ir_tree, api_tree = self.create_ir_tree()

        runtime_context = cde.PythonRuntimeContext()
        runtime_context.Init()
        consumer = cde.PythonSaveToDisk(file_name, num_files, file_type)
        consumer.Init(ir_tree)
        runtime_context.AssignConsumer(consumer)

        consumer.Save()
        _set_dataset_permissions(file_name, num_files)
        del api_tree

    @check_tuple_iterator
    def create_tuple_iterator(self, columns=None, num_epochs=-1, output_numpy=False, do_copy=True):
        """
        Create an iterator over the dataset. The datatype retrieved back will be a list of `numpy.ndarray` .

        To specify which columns to list and the order needed, use columns_list. If columns_list
        is not provided, the order of the columns will remain unchanged.

        Args:
            columns (list[str], optional): List of columns to be used to specify the order of columns.
                Default: None, means all columns.
            num_epochs (int, optional): Maximum number of epochs that iterator can be iterated.
                Default: -1, iterator can be iterated infinite number of epochs.
            output_numpy (bool, optional): Whether or not to output NumPy datatype.
                If output_numpy=False, iterator will output MSTensor. Default: False.
            do_copy (bool, optional): When output data type is mindspore.Tensor,
                use this param to select the conversion method, only take False for better performance. Default: True.

        Returns:
            Iterator, tuple iterator over the dataset.

        Examples:
            >>> # dataset is an instance object of Dataset
            >>> iterator = dataset.create_tuple_iterator()
            >>> for item in iterator:
            ...     # item is a list
            ...     print(type(item))
            ...     break
            <class 'list'>
        """
        if output_numpy is None:
            output_numpy = False

        if Dataset._noop_mode():
            return DummyIterator(self, 'tuple', output_numpy)
        return TupleIterator(self, columns, num_epochs, output_numpy, do_copy)

    @check_dict_iterator
    def create_dict_iterator(self, num_epochs=-1, output_numpy=False, do_copy=True):
        """
        Create an iterator over the dataset. The data retrieved will be a dictionary datatype.

        Args:
            num_epochs (int, optional): Maximum number of epochs that iterator can be iterated.
                Default: -1, iterator can be iterated infinite number of epochs.
            output_numpy (bool, optional): Whether or not to output NumPy datatype,
                if output_numpy=False, iterator will output MSTensor. Default: False.
            do_copy (bool, optional): When output data type is mindspore.Tensor,
                use this param to select the conversion method, only take False for better performance. Default: True.

        Returns:
            Iterator, dictionary iterator over the dataset.

        Examples:
            >>> # dataset is an instance object of Dataset
            >>> iterator = dataset.create_dict_iterator()
            >>> for item in iterator:
            ...     # item is a dict
            ...     print(type(item))
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
            >>> # dataset is an instance object of Dataset
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

    def _init_tree_getters(self):
        """
        Get pipeline information.
        """
        ir_tree, api_tree = self.create_ir_tree()

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
            >>> # dataset is an instance object of Dataset
            >>> col_names = dataset.get_col_names()
        """
        if self._col_names is None:
            runtime_getter = self._init_tree_getters()
            self._col_names = runtime_getter[0].GetColumnNames()

        return self._col_names

    @check_output_shape
    def output_shapes(self, estimate=False):
        """
        Get the shapes of output data.

        Args:
            estimate (bool): If `estimate` is False, will return the shapes of first data row.
                Otherwise, will iterate the whole dataset and return the estimated shapes of data row,
                where dynamic shape is marked as None (used in dynamic data shapes scenario). Default: False.

        Returns:
            list, list of shapes of each column.

        Examples:
            >>> import numpy as np
            >>>
            >>> def generator1():
            ...     for i in range(1, 100):
            ...         yield np.ones((16, i, 83)), np.array(i)
            >>>
            >>> dataset = ds.GeneratorDataset(generator1, ["data1", "data2"])
            >>> output_shapes = dataset.output_shapes()
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

    def output_types(self):
        """
        Get the types of output data.

        Returns:
            list, list of data types.

        Examples:
            >>> # dataset is an instance object of Dataset
            >>> output_types = dataset.output_types()
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

    def get_dataset_size(self):
        """
        Return the number of batches in an epoch.

        Returns:
            int, number of batches.

        Examples:
            >>> # dataset is an instance object of Dataset
            >>> dataset_size = dataset.get_dataset_size()
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
            >>> # dataset is an instance object of Dataset
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

    def sync_update(self, condition_name, num_batch=None, data=None):
        """
        Release a blocking condition and trigger callback with given data.

        Args:
            condition_name (str): The condition name that is used to toggle sending next row.
            num_batch (Union[int, None]): The number of batches (rows) that are released.
                When num_batch is None, it will default to the number specified by the
                sync_wait operation. Default: None.
            data (Any): The data passed to the callback, user defined. Default: None.

        Examples:
            >>> import numpy as np
            >>>
            >>>
            >>> def gen():
            ...     for i in range(100):
            ...         yield (np.array(i),)
            >>>
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
            raise RuntimeError("Sync_update batch size can only be positive integer, got : {}.".format(num_batch))
        notifiers_dict = self.get_sync_notifiers()
        if not isinstance(condition_name, str):
            raise TypeError("Argument condition_name with value {} is not of type str, but got {}."
                            .format(condition_name, type(condition_name)))
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
            int, the number of data in a batch.

        Examples:
            >>> # dataset is an instance object of Dataset
            >>> batch_size = dataset.get_batch_size()
        """
        if self._batch_size is None:
            runtime_getter = self._init_tree_getters()
            self._batch_size = runtime_getter[0].GetBatchSize()
        if self._batch_size is None:
            self._batch_size = 1
        return self._batch_size

    def get_repeat_count(self):
        """
        Get the replication times in RepeatDataset. Default: 1.

        Returns:
            int, the count of repeat.

        Examples:
            >>> # dataset is an instance object of Dataset
            >>> repeat_count = dataset.get_repeat_count()
        """
        if self._repeat_count is None:
            runtime_getter = self._init_tree_getters()
            self._repeat_count = runtime_getter[0].GetRepeatCount()
        if self._repeat_count is None:
            self._repeat_count = 1
        return self._repeat_count

    def get_class_indexing(self):
        """
        Return the class index.

        Returns:
            dict, a str-to-int mapping from label name to index.
            dict, a str-to-list<int> mapping from label name to index for Coco ONLY. The second number
            in the list is used to indicate the super category.

        Examples:
            >>> # dataset is an instance object of Dataset
            >>> class_indexing = dataset.get_class_indexing()
        """
        if self.children:
            return self.children[0].get_class_indexing()
        return {}

    def reset(self):
        """
        Reset the dataset for next epoch.

        Examples:
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

    @staticmethod
    def _update_data_shard(num_shards, shard_id):
        """
        Update the shard number and shard id if necessary.
        This is normally used in distributed training mode like Parameter Server training.
        """
        # If this is in distributed execution mode,
        # the shard number and shard id might need to be updated according to the process's rank or role.
        worker_num = _get_ps_context("worker_num")
        server_num = _get_ps_context("server_num")
        if _is_role_pserver() and _enable_distributed_mindrt() and (worker_num != server_num):
            num_shards = worker_num
            shard_id = 0
        return num_shards, shard_id

    def post_parse(self, ir_node):
        if self.cache:
            ir_node = ir_node.set_cache_client(self.cache.cache_client)
        if self.num_parallel_workers:
            ir_node = ir_node.set_num_workers(self.num_parallel_workers)

        return ir_node


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

        Examples:
            >>> import numpy as np
            >>>
            >>> def gen_corpus():
            ...     # key: word, value: number of occurrences, reason for using letters is so their order is apparent
            ...     corpus = {"Z": 4, "Y": 4, "X": 4, "W": 3, "U": 3, "V": 2, "T": 1}
            ...     for k, v in corpus.items():
            ...         yield (np.array([k] * v, dtype='S'),)
            >>> column_names = ["column1"]
            >>> dataset = ds.GeneratorDataset(gen_corpus, column_names)
            >>> dataset = dataset.build_vocab(columns=["column1"],
            ...                               freq_range=(1, 10), top_k=5,
            ...                               special_tokens=["<pad>", "<unk>"],
            ...                               special_first=True)

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

    def build_sentencepiece_vocab(self, columns, vocab_size, character_coverage, model_type, params):
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
                The input sentence must be pretokenized when using word type.
            params(dict): Any extra optional parameters of sentencepiece library according to your raw data

        Returns:
            SentencePieceVocab, vocab built from the dataset.

        Examples:
            >>> from mindspore.dataset.text import SentencePieceModel
            >>>
            >>> # You can construct any text dataset as source, take TextFileDataset as example.
            >>> dataset = ds.TextFileDataset("/path/to/sentence/piece/vocab/file", shuffle=False)
            >>> dataset = dataset.build_sentencepiece_vocab(["text"], 5000, 0.9995, SentencePieceModel.UNIGRAM, {})
        """
        if not isinstance(model_type, SentencePieceModel):
            raise TypeError("Argument model_type with value {0} is not of type SentencePieceModel, but got {1}." \
                            .format(model_type, type(model_type)))
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
            raise TypeError("shuffle must be of boolean or enum of 'Shuffle' values like 'Shuffle.GLOBAL' or "
                            "'Shuffle.FILES' or 'Shuffle.INFILE'.")

        self.shuffle_flag = 2  # Global shuffle
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
            raise ValueError("The following patterns did not match any files: {}.".format(unmatched_patterns))

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
        num_shards, shard_id = self._update_data_shard(num_shards, shard_id)
        super().__init__(num_parallel_workers=num_parallel_workers, num_samples=num_samples, shuffle=shuffle,
                         num_shards=num_shards, shard_id=shard_id, cache=cache)
        self.shuffle_flag = replace_none(shuffle, True)
        self.sampler = samplers.select_sampler(num_samples, sampler, shuffle, num_shards, shard_id)

    def add_sampler(self, new_sampler):
        """
        Add a child sampler for the current dataset.

        Args:
            new_sampler (Sampler): The child sampler to be added.

        Examples:
            >>> new_sampler = ds.DistributedSampler(10, 2)
            >>> dataset.add_sampler(new_sampler)  # dataset is an instance of Dataset
        """
        # Note: By adding a sampler, the sampled IDs will flow to the new_sampler
        # after first passing through the current samplers attached to this dataset.
        self.dataset_size = None
        new_sampler.add_child(self.sampler)
        self.sampler = new_sampler

    def use_sampler(self, new_sampler):
        """
        Replace the last child sampler of the current dataset, remaining the parent sampler unchanged.

        Args:
            new_sampler (Sampler): The new sampler to replace with.

        Examples:
            >>> # dataset is an instance object of Dataset
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

            randomize (bool, optional): Determines whether or not to split the data randomly. Default: True.
                If True, the data will be randomly split. Otherwise, each split will be created with
                consecutive rows from the dataset.

        Note:
            1. There is an optimized split function, which will be called automatically when the dataset
               that calls this function is a MappableDataset.
            2. Dataset should not be sharded if split is going to be called. Instead, create a
               DistributedSampler and specify a split to shard after splitting. If the dataset is
               sharded after a split, it is strongly recommended setting the same seed in each instance
               of execution, otherwise each shard may not be part of the same split (see Examples).
            3. It is strongly recommended to not shuffle the dataset, but use randomize=True instead.
               Shuffling the dataset may not be deterministic, which means the data in each split
               will be different in each epoch. Furthermore, if sharding occurs after split, each
               shard may not be part of the same split.

        Returns:
            tuple(Dataset), a tuple of datasets that have been split.

        Raises:
            RuntimeError: If get_dataset_size returns None or is not supported for this dataset.
            RuntimeError: If `sizes` is list of integers and sum of all elements in sizes does not
                equal the dataset size.
            RuntimeError: If `sizes` is list of float and there is a split with size 0 after calculations.
            RuntimeError: If the dataset is sharded prior to calling split.
            ValueError: If `sizes` is list of float and not all floats are between 0 and 1, or if the
                floats don't sum to 1.

        Examples:
            >>> # Since many datasets have shuffle on by default, set shuffle to False if split will be called!
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


def _check_shm_usage(num_worker, queue_size, max_rowsize, num_queues=1):
    """
    Check sufficient shared memory is available for shared memory queues
    when training in parallel mode.
    """
    threshold_ratio = 0.8
    if platform.system().lower() not in {"windows", "darwin"}:
        device_num = _get_device_num()
        # In the cluster, _get_device_num indicates the number of the entire cluster. The maximum number of cards
        # on the ascend server is 8.
        if device_num > 1 and context.get_context("device_target") == "Ascend":
            device_num = min(device_num, 8)
        shm_estimate_usage = device_num * num_worker * num_queues * \
                             (queue_size + 2) * max_rowsize * 1024 * 1024
        try:
            shm_available = psutil.disk_usage('/dev/shm').free
            if shm_estimate_usage >= threshold_ratio * shm_available:
                raise RuntimeError(
                    "Insufficient shared memory available. Required: {}, Available: {}. "
                    "The required memory can't exceed 80% of the available shared memory, "
                    "it's recommended to reduce memory usage by following methods:\n"
                    "1. reduce value of parameter max_rowsize or num_parallel_workers.\n"
                    "2. reduce prefetch size by set_prefetch_size().\n"
                    "3. disable shared memory by set_enable_shared_mem().".format(shm_estimate_usage, shm_available))
        except FileNotFoundError:
            raise RuntimeError("Expected /dev/shm to exist.")


class BatchDataset(UnionBaseDataset):
    """
    The result of applying Batch operation to the input dataset.

    Args:
        input_dataset (Dataset): Input Dataset to be batched.
        batch_size (Union[int, function]): The number of rows each batch is created with. An
            int or callable which takes exactly 1 parameter, BatchInfo.
        drop_remainder (bool, optional): Determines whether or not to drop the last
            possibly incomplete batch. Default: False. If True, and if there are less
            than batch_size rows available to make the last batch, then those rows will
            be dropped and not propagated to the child node.
        num_parallel_workers (int, optional): Number of workers to process the dataset in parallel. Default: None.
        per_batch_map (callable, optional): Per batch map callable. A callable which takes
            (list[Tensor], list[Tensor], ..., BatchInfo) as input parameters. Each list[Tensor] represents a batch of
            Tensors on a given column. The number of lists should match with number of entries in input_columns. The
            last parameter of the callable must always be a BatchInfo object.
        input_columns (Union[str, list[str]], optional): List of names of the input columns. The size of the list must
            match with signature of per_batch_map callable.
        output_columns (Union[str, list[str]], optional): List of names assigned to the columns outputted by
            the last operation. This parameter is mandatory if len(input_columns) !=
            len(output_columns). The size of this list must match the number of output
            columns of the last operation. Default: None, output columns will have the same
            name as the input columns, i.e., the columns will be replaced.
        max_rowsize(int, optional): Maximum size of row in MB that is used for shared memory allocation to copy
            data between processes.  This is only used if python_multiprocessing is set to True. Default: 16.

    """

    def __init__(self, input_dataset, batch_size, drop_remainder=False, num_parallel_workers=None, per_batch_map=None,
                 input_columns=None, output_columns=None, python_multiprocessing=False, max_rowsize=16):
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
        self.max_rowsize = max_rowsize

    def __del__(self):
        if hasattr(self, "process_pool") and self.process_pool is not None:
            self.process_pool.terminate()
            del self.process_pool

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

            self.process_pool = _PythonMultiprocessing(str(self), self.num_parallel_workers, [self.per_batch_map],
                                                       self.max_rowsize * self.batch_size)
            # Wrap per_batch_map into _PythonCallable
            self.per_batch_map = _PythonCallable(self.per_batch_map, 0, self.process_pool)
        else:
            if self.per_batch_map is not None:
                self.per_batch_map = FuncWrapper(self.per_batch_map)


class BatchInfo(cde.CBatchInfo):
    """
    Only the batch size function and per_batch_map of the batch operation can dynamically adjust parameters
    based on the number of batches and epochs during training.
    """

    def get_batch_num(self):
        """
        Return the batch number of the current batch.
        """
        return

    def get_epoch_num(self):
        """
        Return the epoch number of the current batch.
        """
        return


class BlockReleasePair:
    """
    The blocking condition class used by SyncWaitDataset.

    Args:
        init_release_rows (int): Number of lines to allow through the pipeline.
        callback (function): The callback function that will be called when release is called. Default: None.
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
            possibly incomplete batch. Default: False. If True, and if there are less
            than batch_size rows available to make the last batch, then those rows will
            be dropped and not propagated to the child node.
        num_parallel_workers (int, optional): Number of workers to process the dataset in parallel. Default: None.
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
        self.pad_info = replace_none(pad_info, dict())

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
        callback (function): Callback function that will be invoked when sync_update is called. Default: None.

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
        return {**self.children[0].get_sync_notifiers(), **{self._condition_name: self._pair.release_func}}

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
    The result of applying Shuffle operation to the input Dataset.

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
_OP_NAME = dict()
_OP_PROCESS = dict()


# PythonCallable wrapper for multiprocess pyfunc
class _PythonCallable:
    """
    Internal Python function wrapper for multiprocessing pyfunc.
    """

    def __init__(self, py_callable, idx, pool=None):
        # Original Python callable from user.
        self.py_callable = py_callable
        # Process pool created for current iterator.
        self.pool = pool
        # Python callable index
        self.idx = idx
        self.check_interval = get_multiprocessing_timeout_interval()

    def __call__(self, *args):
        result = None
        start_time = time.time()
        count = 1
        get_data_from_worker_process = False
        while get_data_from_worker_process is False:
            cost_time = time.time() - start_time
            if cost_time > (self.check_interval * count):
                logger.warning("It has been waiting for " + str(cost_time) + "s because the multi "
                               "workers of map operation cost long time to process next data. "
                               "Worker process list are: " + str(self.pool.get_pids()) + ", you can use "
                               "\"py-spy dump -p {PID} -l -s \""
                               "to dump the worker process stack. You can also set the timeout interval by "
                               "ds.config.set_multiprocessing_interval to adjust the output frequency of this "
                               "log.")
                count += 1
            if self.pool.is_running() and check_iterator_cleanup() is False:
                try:
                    result = self.pool.execute(self.idx, *args)
                except multiprocessing.TimeoutError:
                    continue
                get_data_from_worker_process = True
            else:
                # worker process is stopped
                logger.warning("The worker process of map operation is stopped. "
                               "So return None to main thread and break the main thread.")
                return None
        # got value from worker process
        if not isinstance(result, tuple) and get_data_from_worker_process is True:
            result = (result,)
        return result

    def to_json(self):
        return self.py_callable.to_json()


class Pipe:
    """
    Class to handle communication between the master process and the worker processes.
    """

    def __init__(self, warning_ctl, shared_memory=False, max_rowsize=16):
        self.shared_memory = shared_memory
        self.eof = multiprocessing.Event()
        if self.shared_memory:
            self.in_queue = _SharedQueue(1, warning_ctl, max_rowsize=max_rowsize)
            self.res_queue = _SharedQueue(1, warning_ctl, max_rowsize=max_rowsize)
        else:
            self.in_queue = _Queue(1)
            self.res_queue = _Queue(1)
        self.in_queue._joincancelled = True  # pylint: disable=W0212
        self.res_queue._joincancelled = True  # pylint: disable=W0212

    def master_send(self, func_index, data):
        self.in_queue.put_nowait((func_index, *data))

    def master_receive(self):
        return self.res_queue.get_until(timeout=1, exit_signal=self.eof)

    def master_close(self):
        self.eof.set()
        self.send_finish_signal()
        self.res_queue.cancel_join_thread()
        self.in_queue.cancel_join_thread()

    def send_finish_signal(self):
        self.worker_send(None)

    def worker_send(self, data):
        self.res_queue.put_until(data, timeout=1, exit_signal=self.eof)

    def worker_receive(self):
        result = self.in_queue.get_until(timeout=1, exit_signal=self.eof)
        if result is None:
            return result
        if len(result) == 1:
            raise RuntimeError(f"Corrupted data. Worker received {len(result)} elements, it should be more than 1.")
        func_index, *data = result
        return func_index, tuple(data)

    def worker_close(self):
        self.res_queue.cancel_join_thread()
        self.in_queue.cancel_join_thread()


def _main_process_already_exit():
    """
    Judge whether main process already exit.
    """
    ppid = os.getppid()

    if (platform.system().lower() != 'windows' and
            not _PythonMultiprocessing.is_process_alive(ppid)):
        return True
    return False


def _worker_loop(operations, pipe, seed=get_seed()):
    """
    Multiprocess worker process loop.
    """

    def _ignore_sigint():
        """
        We need to ignore sigint signal here so subprocesses can exit normally and clear.
        """
        signal.signal(signal.SIGINT, signal.SIG_IGN)

    # We set the seed here so the main process will have the same seed, while the child process
    # will have different seed depending on what is being passed
    set_seed(seed)
    while not _main_process_already_exit():
        _ignore_sigint()

        result = pipe.worker_receive()
        if result is None:
            pipe.worker_close()
            return
        (idx, input_tensors) = result
        try:
            output_tensors = operations[idx](*input_tensors)

            pipe.worker_send(output_tensors)
        except Exception:
            pipe.worker_send(ExceptionHandler(where="in map(or batch) worker and execute Python function"))
            # Do not return


def worker_target(operations, seed=get_seed()):
    return lambda pipe: _worker_loop(operations, pipe, seed)


class _MPWorker(multiprocessing.Process):
    """
    Worker process for multiprocessing.
    """

    def __init__(self, operations, warning_ctl, max_rowsize=16, seed=get_seed()):
        shared_memory = get_enable_shared_mem()
        self.pipe = Pipe(warning_ctl, shared_memory=shared_memory, max_rowsize=max_rowsize)
        super().__init__(target=worker_target(operations, seed), args=(self.pipe,), daemon=True)

    def execute(self, idx, *args):
        self.pipe.master_send(idx, args)
        res = self.pipe.master_receive()
        if isinstance(res, ExceptionHandler):
            res.reraise()
        return res

    def close(self):
        try:
            if self.is_alive():
                logger.info(f"Closing worker with PID: {self.pid}")
                self.pipe.master_close()
                super().terminate()
                super().join()
                super().close()

        except ValueError:
            # Process has been closed already
            return
        return

    def is_alive(self):
        try:
            return super().is_alive()
        except ValueError:
            return False


class _PythonMultiprocessing(cde.PythonMultiprocessingRuntime):
    """
    A wrapper to multiprocessing.pool that performs cleanup and ensure proper termination of forked processes.
    """

    class _ExceptHookHandler:
        """
        Internal class ExceptionHandler
        """

        def __init__(self):
            sys.excepthook = self.__handler_exception

        @staticmethod
        def mp_pool_exit_preprocess():
            if check_iterator_cleanup() is False:
                # Set the iterator_cleanup flag to True before exiting, and wait 3s for all apply_async
                # applied to the multiprocessing task to prevent multiprocessing from hang when exiting
                _set_iterator_cleanup()
                time.sleep(3)

        def __handler_exception(self, ex_type, value, tb):
            logger.critical("Uncaught exception: ", exc_info=(ex_type, value, tb))
            self.mp_pool_exit_preprocess()

    def __init__(self, op_name, num_parallel_workers, operations, max_row_size=16):
        super(_PythonMultiprocessing, self).__init__()
        self.op_name = op_name
        self.num_parallel_workers = num_parallel_workers
        self.operations = operations
        self.max_row_size = max_row_size

        self.workers = None
        self.pids = None
        self.op_id = -1

        self.queues_map = {}
        self.next_queue = 0

        self.eot = None
        self.watch_dog = None
        self.ppid = os.getpid()
        self.hook = None
        self.warning_ctl = None
        # cache thread (get_ident()) to worker_id mapping in Python layer
        self.python_threads_to_workers = {}

    def __del__(self):
        try:
            self.terminate()
        except TypeError:
            pass

    # This wait function is for cleaning zombie subprocesses
    @staticmethod
    def wait_pid():
        """
        This function is used by the main process to release subprocess resources.
        """
        try:
            while True:
                child_pid, _ = os.waitpid(-1, os.WNOHANG)
                if child_pid == 0:
                    break
        except OSError:
            # waitpid may be failed for some reasons so we ignore this error
            pass

    # Dataset need watch_dog thread to monitoring fork multi-processing,
    # and thread can't be a member function otherwise python won't collect and release resources.
    @staticmethod
    def _watch_dog(eot, workers):
        """
        This thread is for monitoring subprocesses forked by GeneratorDataset/map/batch
        """
        if not isinstance(workers, list):
            raise TypeError("[Internal Error] The 2nd parameter of watch dog thread should be list of process, "
                            "but got {}.".format(type(workers)))

        while not eot.is_set():
            # Monitoring and count how many subprocesses already exit
            clear_subprocess_timeout = _PythonMultiprocessing._monitor_subprocess_exit(workers)
            # If find subprocess exit, we will wait for 30s and do some waitpid operations
            if clear_subprocess_timeout > 0:
                start = time.time()
                while time.time() - start < clear_subprocess_timeout:
                    # We need to distinguishing get_dataset_size or train finished normally and hang scenario.
                    # If get_dataset_size or train finished normally, _stop_subprocess can be execute and
                    # self.need_abort can be set to True. If main process is hang in get(), self.need_abort
                    # will never set to True, then we wait for 30s and kill main process
                    if eot.is_set():
                        return
                    # Sometimes subprocess may be zombie, so in 30s we can wait and do some useful tasks(waitpid).
                    _PythonMultiprocessing.wait_pid()
                # multiprocessing.queue may hang in .get() forever when put() process was killed.
                # We have to exit main process otherwise main process will hang.
                _PythonMultiprocessing._terminate_processes(workers)
                logger.critical("The subprocess of dataset may exit unexpected or be killed, "
                                "main process will exit. If this is not an artificial operation, you can use "
                                "ds.config.set_enable_watchdog(False) to block this error.")
                os.kill(os.getpid(), signal.SIGTERM)

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

    # Monitor the exit number of subprocesses
    @staticmethod
    def _monitor_subprocess_exit(workers):
        """
        To monitor whether process is exit.

        Args:
            workers (list of multiprocessing.Process): multiprocessing.Process.

        Returns:
            int, the timeout(in seconds) when process exit.
        """
        for w in workers:
            try:
                exit_code = w.exitcode
                if exit_code is not None:
                    # For kill -9, we can exit quickly
                    if exit_code == -9:
                        return 1
                    # For kill -15, we still exit after 30s
                    if exit_code == -15:
                        return 30
                # In some cases the subprocess has been killed but the exitcode is still None.
                # So we use os.kill(pid, 0) to check if it is alive.
                subprocess_alive = _PythonMultiprocessing.is_process_alive(w.pid)
                if not subprocess_alive:
                    # Like kill -15, we wait 30s before exit
                    return 30
            except ValueError:
                # process has been closed already
                return 0
        return 0

    @staticmethod
    def is_process_alive(pid):
        """
        Check if the process is alive or not.
        Note:  We hit a deadlock when we use psutil or w.exitcode to check whether a process is alive.
        Instead we use os.kill(ppid, 0).

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
    def _clean_process(ppid, workers):
        """
            This is the execute function of clean process, if we found main process exited, we will clean subprocesses.

        Args:
            ppid: The process id of main process.
            workers: The list of subprocesses.

        """
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        while _PythonMultiprocessing.is_process_alive(ppid):
            time.sleep(0.1)

        _PythonMultiprocessing._terminate_processes(workers)
        os.kill(os.getpid(), signal.SIGTERM)

    def launch(self, op_id=-1):
        """
        Launch Python multiprocessing pool.

        Args:
            pop_id: ID for operation to have Python multiprocessing pool launched

        Returns:
            Python multiprocssing pool is launched.
        """
        self.python_threads_to_workers = {}
        self.op_id = op_id
        logger.info("Launching new Python Multiprocessing pool for Op:" + str(self.op_id))
        if self.is_mp_enabled():
            message = "Launching a new Python multiprocessing pool while a pool already exists!" + \
                " The existing pool will be terminated first."
            logger.warning(message)
            self.terminate()
            self.reset()
        self.create_pool()

    def create_pool(self):
        """

        Returns:

        """
        if get_enable_shared_mem():
            self.check_shared_memory()

        if self.workers is not None:
            raise Exception("Pool was already created, close it first.")

        # Let gc collect unreferenced memory to avoid child processes in the pool to do it
        gc.collect()

        # Construct python worker processes
        self.workers = []
        self.warning_ctl = multiprocessing.Value('i', 0)
        for i in range(self.num_parallel_workers):
            worker = _MPWorker(self.operations, self.warning_ctl, self.max_row_size, i + get_seed())
            worker.start()
            self.workers.append(worker)

        logger.info("Op: " + str(self.op_id) + " Python multiprocessing pool workers' PIDs: " + str(self.get_pids()))

        self.hook = _PythonMultiprocessing._ExceptHookHandler()

        # The op (Map, Batch, etc) multiprocessing will launch a watch dog thread for monitoring sub processes
        self._launch_watch_dog()

        atexit.register(self.terminate)

    def terminate(self):
        self.close_all_workers()
        self.abort_watchdog()

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

    def add_new_workers(self, num_new_workers):
        logger.info(
            "Increasing num_parallel_workers of Python Multiprocessing pool for Op:" + str(self.op_id) +
            ", old num_workers=" + str(self.num_parallel_workers) + " new num_workers=" + str(
                self.num_parallel_workers +
                num_new_workers) + ".")
        self.terminate()
        self.num_parallel_workers += num_new_workers
        self.launch(self.op_id)

    def remove_workers(self, num_removed_workers):
        logger.info(
            "Decreasing num_parallel_workers of Python Multiprocessing pool for Op:" + str(self.op_id) +
            ", old num_workers=" + str(self.num_parallel_workers) + " new num_workers=" + str(
                self.num_parallel_workers -
                num_removed_workers) + ".")
        self.terminate()
        self.num_parallel_workers -= num_removed_workers
        self.launch(self.op_id)

    def is_mp_enabled(self):
        return self.workers is not None

    def check_shared_memory(self):
        """
        Check if there is enough shared memory in the system.
        """
        _check_shm_usage(self.num_parallel_workers, 1, self.max_row_size, 2)

    def execute(self, idx, *args):
        """
        Execute
        """
        t_id = threading.get_ident()
        # get the worker_id from Python layer cache first, get from Cpp layer if not found.
        worker_id = self.python_threads_to_workers.setdefault(t_id, self.get_thread_to_worker())
        if worker_id >= len(self.workers):
            raise RuntimeError("[Internal] worker_id value is greater than number of available workers!")

        # todo check_iterator_cleanup
        if self.is_running() and check_iterator_cleanup() is False:
            return self.workers[worker_id].execute(idx, *args)

        return None

    def _launch_watch_dog(self):
        """
        We will launch a watchdog thread and a clean process to cleaning subprocess when there is process was killed.
        The watchdog thread will cleanup subprocesses and main process when one of the subprocesses was killed.
        The cleaning subprocess will cleanup subprocesses when main process was killed.
        """
        if platform.system().lower() != 'windows':
            self.cleaning_process = multiprocessing.Process(target=self._clean_process,
                                                            args=(self.ppid, self.workers),
                                                            name="OrphanCleaner",
                                                            daemon=True)
            self.cleaning_process.start()

            if get_enable_watchdog():
                self.eot = threading.Event()
                self.watch_dog = threading.Thread(target=self._watch_dog,
                                                  args=(self.eot, self.workers + [self.cleaning_process]),
                                                  name="WatchDog",
                                                  daemon=True)
                self.watch_dog.start()

    def _abort_watchdog(self):
        if not self.eot.is_set():
            self.eot.set()

    def abort_watchdog(self):
        if hasattr(self, 'watch_dog') and self.watch_dog is not None and hasattr(self, 'eot') and self.eot is not None:
            self._abort_watchdog()
        if hasattr(self, 'cleaning_process') and self.cleaning_process is not None:
            _PythonMultiprocessing._terminate_processes([self.cleaning_process])

    def is_running(self):
        if hasattr(self, 'workers') and self.workers is not None:
            return all([w.is_alive() for w in self.workers])
        return False

    def close_all_workers(self):
        if hasattr(self, 'workers') and self.workers is not None:
            for w in self.workers:
                w.close()
            self.workers = None
            self.pids = None


class MapDataset(UnionBaseDataset):
    """
    The result of applying the Map operation to the input Dataset.

    Args:
        input_dataset (Dataset): Input Dataset to be mapped.
        operations (Union[list[TensorOperation], list[functions]]): A function mapping a nested structure of tensors
            to another nested structure of tensor. Default: None.
        input_columns (Union[str, list[str]]): List of names of the input columns.
            Default: None, the operations will be applied on the first columns in the dataset.
            The size of the list should match the number of inputs of the first operation.
        output_columns (Union[str, list[str]], optional): List of names of the output columns.
            The size of the list should match the number of outputs of the last operation.
            Default: None, output columns will be the input columns, i.e., the columns will
            be replaced.
        num_parallel_workers (int, optional): Number of workers to process the dataset
            in parallel. Default: None.
        python_multiprocessing (bool, optional): Parallelize Python operations with multiple worker process. This
            option could be beneficial if the Python operation is computational heavy. Default: False.
        cache (DatasetCache, optional): Use tensor caching service to speed up dataset processing.
            Default: None, which means no cache is used.
        callbacks (DSCallback, list[DSCallback], optional): List of Dataset callbacks to be called. Default: None.
        max_rowsize(int, optional): Maximum size of row in MB that is used for shared memory allocation to copy
            data between processes. This is only used if python_multiprocessing is set to True. Default: 16.
        offload (bool, optional): Flag to indicate whether offload is used. Default: None.
    """

    def __init__(self, input_dataset, operations=None, input_columns=None, output_columns=None,
                 num_parallel_workers=None, python_multiprocessing=False, cache=None, callbacks=None, max_rowsize=16,
                 offload=None):
        super().__init__(children=input_dataset, num_parallel_workers=num_parallel_workers, cache=cache)
        self.operations = to_list(operations)
        for op in self.operations:
            # user define c_vision.HWC2CHW without parentheses is error
            if type(op) == type:  # pylint: disable=unidiomatic-typecheck
                raise ValueError("Parameter operations's element of method map should be a dataset processing "
                                 "operation instance, but got: {}. It may be missing parentheses for "
                                 "instantiation.".format(op))
            if not isinstance(op, (c_transforms.TensorOperation, py_transforms.PyTensorOperation)) \
                    and not callable(op):
                raise ValueError("Parameter operations's element of method map should be a python function or "
                                 "class method which should be callable, but got: {}. It doesn't need parentheses "
                                 "for python function or class method.".format(op))

        self.input_columns = to_list(input_columns)
        self.output_columns = to_list(output_columns)

        #  If output_columns were not provided then use input_columns
        self.output_columns = self.input_columns if not self.output_columns else self.output_columns

        self.python_multiprocessing = python_multiprocessing
        self.process_pool = None

        self.callbacks = to_list(callbacks)
        self.max_rowsize = max_rowsize
        self.offload = offload

    def parse(self, children=None):
        operations = self.__decompose_callable_operations()

        count_old_transforms, count_new_transforms, count_non_data_vision_transforms = \
            self.__count_transforms(operations)
        count_pyfunc = self.__count_pyfuncs(operations)
        if count_new_transforms + count_pyfunc == len(operations):
            prev_op = None
            for op in operations:
                if op.implementation is None:
                    if prev_op and prev_op.implementation == Implementation.PY:
                        op.implementation = Implementation.PY
                    else:
                        op.implementation = Implementation.C
                prev_op = op
            operations = self.__insert_debug_wrapper(operations)
            operations = transforms.transforms.Compose.reduce(operations)
        elif count_old_transforms + count_pyfunc + count_non_data_vision_transforms == len(operations):
            operations = self.__insert_debug_wrapper(operations)
            operations = transforms.py_transforms.Compose.reduce(operations)
        else:
            raise RuntimeError("Mixing old legacy c/py_transforms and new unified transforms is not allowed.")

        self.operations = self.__process_final_operations(operations)
        self.prepare_multiprocessing()

        callbacks = [cb.create_runtime_obj() for cb in self.callbacks]
        return cde.MapNode(children[0], self.operations, self.input_columns, self.output_columns,
                           callbacks, self.max_rowsize, OffloadToManualOffloadMode.get(self.offload), self.process_pool)

    def __deepcopy__(self, memodict):
        return self.__safe_deepcopy__(memodict, exclude=("operations", "callbacks", "__transfer_dataset__"))

    def __del__(self):
        if hasattr(self, "process_pool") and self.process_pool is not None:
            self.process_pool.terminate()
            del self.process_pool

    @staticmethod
    def __insert_debug_wrapper(operations):
        """
        Insert DebuggerWrapper before and after each op if debug mode is on.
        """
        if not get_debug_mode():
            return operations
        inserted_func = transforms.py_transforms_util.FuncWrapper(DebugWrapper())
        inserted_func.implementation = Implementation.PY
        inserted_operations = [inserted_func]
        for op in operations:
            if isinstance(op, transforms.py_transforms_util.FuncWrapper):
                try:
                    op_name = op.transform.__name__
                except Exception:
                    op_name = op.transform.__class__.__name__
            else:
                op_name = op.__class__.__name__
            inserted_func = transforms.py_transforms_util.FuncWrapper(DebugWrapper(op_name))
            inserted_func.implementation = Implementation.PY
            inserted_operations.extend([op, inserted_func])
        return inserted_operations

    @staticmethod
    def __count_pyfuncs(operations):
        """
        Count the number of pyfuncs operations
        """
        return sum([1 if isinstance(op, FuncWrapper) else 0 for op in operations])

    @staticmethod
    def __count_transforms(operations):
        """
        Count the various flavors of transforms operations
        """
        # Count the number of old legacy data and vision c_transforms and py_transforms
        count_old_transforms = sum(
            [1 if "c_transforms" in str(op)
             or isinstance(op, (c_transforms.TensorOperation, py_transforms.PyTensorOperation))
             or ("py_transforms" in str(op) and not isinstance(op, FuncWrapper))
             else 0 for op in operations])
        # Count the number of new unified data and vision transforms
        count_new_transforms = sum([1 if hasattr(op, "implementation") and not isinstance(op, FuncWrapper)
                                    else 0 for op in operations])
        # Count the number of non-data transforms and non-vision transforms
        count_non_data_vision_transforms = sum(
            [1 if "text.transforms" in str(op) or "audio.transforms" in str(op) else 0 for op in operations])
        return count_old_transforms, count_new_transforms, count_non_data_vision_transforms

    @staticmethod
    def __operation_valid_for_multiprocessing(op):
        if callable(op) and str(op).find("c_transform") < 0:
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
            iter_specific_operations = []
            callable_list = []

            # If user didn't specify num_parallel_workers, set it to default
            if self.num_parallel_workers is None:
                self.num_parallel_workers = get_num_parallel_workers()

            # Pass #1, look for Python callables and build list
            for op in self.operations:
                # our c transforms is now callable and should not be run in Python multithreading
                if MapDataset.__operation_valid_for_multiprocessing(op):
                    callable_list.append(op)

            if callable_list:
                self.process_pool = _PythonMultiprocessing(str(self), self.num_parallel_workers, callable_list,
                                                           self.max_rowsize)
                # Pass #2
                idx = 0
                for op in self.operations:
                    # our c transforms is now callable and should not be run in Python multithreading
                    if MapDataset.__operation_valid_for_multiprocessing(op):
                        # Wrap Python callable into _PythonCallable
                        iter_specific_operations.append(_PythonCallable(op, idx, self.process_pool))
                        idx += 1
                    else:
                        # CPP ops remain the same
                        iter_specific_operations.append(op)
                self.operations = iter_specific_operations

    def __decompose_callable_operations(self):
        """
        Decompose operations and build list of old legacy ops which are callable
        """
        decomposed_operations = transforms.transforms.Compose.decompose(self.operations)
        operations = []
        for op in decomposed_operations:
            if callable(op) and not hasattr(op, "implementation") and str(op).find(
                    "c_transform") < 0 and not isinstance(op, c_transforms.TensorOperation) and \
                    not isinstance(op, py_transforms.PyTensorOperation):
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
            Default: None, the predicate will be applied to all columns in the dataset.
        num_parallel_workers (int, optional): Number of workers to process the dataset
            in parallel. Default: None.
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
        return any([c.is_sync() for c in self.children])


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
                raise TypeError("Invalid dataset, expected Dataset object, but got %s!" % type(dataset))
        self.datasets = datasets
        self._sampler = samplers.SequentialSampler(num_samples=None)

        self.children_sizes_ = [c.get_dataset_size() for c in self.children]
        child_index = 0
        for item in self.children_sizes_:
            if item == 0:
                raise ValueError("There are no samples in the dataset number %d. Please make sure there are "
                                 "valid samples in the dataset." % child_index)
            child_index += 1

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

            from mindspore.dataset.engine.datasets_user_defined import GeneratorDataset
            if isinstance(child, GeneratorDataset) and not hasattr(child.source, "__getitem__"):
                dataset_len = 0
                self.children_sizes_[index] = 0

            if isinstance(child, MappableDataset):
                self._children_flag_and_nums.append((0, dataset_len))
            else:
                self._children_flag_and_nums.append((1, dataset_len))

    def parse(self, children=None):
        return cde.ConcatNode(children, self._sampler, self._children_flag_and_nums, self._children_start_end_index_)

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
        if not isinstance(sampler, samplers.DistributedSampler):
            raise TypeError("The parameter %s of concat must be DistributedSampler!" % sampler)

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
                raise ValueError("The parameter NumSamples of %s is not support to be set!" % child)

            if isinstance(child, (BatchDataset, PaddedBatchDataset)):
                raise TypeError("The parameter %s of concat must not be BatchDataset or PaddedBatchDataset!" % child)

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
        ir_tree, self.api_tree = dataset.create_ir_tree()

        self._runtime_context = cde.PythonRuntimeContext()
        self._runtime_context.Init()
        self._to_device = cde.ToDevice(num_epochs)
        self._to_device.Init(ir_tree)
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

    def _reset(self, step, epoch):
        self._to_device.Reset(step, epoch)


class TransferDataset(Dataset):
    """
    The result of applying TDT operation to the input Dataset.

    Args:
        input_dataset (Dataset): Input Dataset to be transferred.
        send_epoch_end (bool, optional): Whether to send end of sequence to device or not. Default: True.
        create_data_info_queue (bool, optional): Whether to create queue which stores
            types and shapes of data or not. Default: False.

    Raises:
        TypeError: If device_type is empty.
        ValueError: If device_type is not 'Ascend', 'GPU' or 'CPU'.
        RuntimeError: If dataset is unknown.
    """

    def __init__(self, input_dataset, send_epoch_end=True, create_data_info_queue=False):
        super().__init__(children=input_dataset)
        self.queue_name = str(uuid.uuid1())
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
        return cde.DataQueueNode(children[0], self.queue_name, self.device_type, self.device_id, self._send_epoch_end,
                                 total_batch, self._create_data_info_queue)

    def create_dict_iterator(self, num_epochs=-1, output_numpy=False):
        raise RuntimeError("TransferDataset is not iterable.")

    def create_tuple_iterator(self, columns=None, num_epochs=-1, output_numpy=False, do_copy=True):
        raise RuntimeError("TransferDataset is not iterable.")

    def __iter__(self):
        raise RuntimeError("TransferDataset is not iterable.")

    def output_shapes(self):
        raise RuntimeError("TransferDataset does not support obtaining output_shapes.")

    def output_types(self):
        raise RuntimeError("TransferDataset does not support obtaining output_types.")

    @check_to_device_send
    def send(self, num_epochs=-1):
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

    def _reset(self, step, epoch):
        if self._to_device is not None:
            logger.info("Reset the dataset pipeline to step: " + str(step) + ", epoch: " + str(epoch))
            self._to_device._reset(step, epoch)  # pylint: disable=protected-access


class Schema:
    """
    Class to represent a schema of a dataset.

    Args:
        schema_file(str): Path of the schema file. Default: None.

    Returns:
        Schema object, schema info about dataset.

    Raises:
        RuntimeError: If schema file failed to load.

    Examples:
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
                Default: None, [-1] which is an unknown shape of rank 1.

        Raises:
            ValueError: If column type is unknown.

        Examples:
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
        Parse the columns and add it to self.

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
            >>>
            >>> schema1 = ds.Schema()
            >>> schema2 = schema1.to_json()
        """
        return self.cpp_schema.to_json()

    def from_json(self, json_obj):
        """
        Get schema file from JSON object.

        Args:
            json_obj(dictionary): Object of JSON parsed.

        Raises:
            RuntimeError: if there is unknown item in the object.
            RuntimeError: if dataset type is missing in the object.
            RuntimeError: if columns are missing in the object.

        Examples:
            >>> import json
            >>>
            >>> from mindspore.dataset import Schema
            >>>
            >>> with open("/path/to/schema_file") as file:
            ...     json_obj = json.load(file)
            ...     schema = ds.Schema()
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
