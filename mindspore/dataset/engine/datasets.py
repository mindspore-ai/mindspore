# Copyright 2019-2021 Huawei Technologies Co., Ltd
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
This dataset module supports various formats of datasets, including ImageNet, TFData,
MNIST, Cifar10/100, Manifest, MindRecord, and more. This module loads data with
high performance and parses data precisely. Some of the operations that are
provided to users to preprocess data include shuffle, batch, repeat, map, and zip.
"""
import atexit
import glob
import json
import math
import os
import signal
import time
import uuid
import multiprocessing
from multiprocessing.pool import RUN
import queue
from enum import Enum
from functools import partial
from importlib import import_module
import sys
import threading

import copy
import weakref
import numpy as np

import mindspore._c_dataengine as cde
from mindspore._c_expression import typing

from mindspore import log as logger
from mindspore.parallel._ps_context import _is_role_pserver, _is_role_sched

import mindspore.dataset.transforms.py_transforms as py_transforms

from . import samplers
from .iterators import DictIterator, TupleIterator, DummyIterator, check_iterator_cleanup, _set_iterator_cleanup, \
    ITERATORS_LIST, _unset_iterator_cleanup
from .validators import check_batch, check_shuffle, check_map, check_filter, check_repeat, check_skip, check_zip, \
    check_rename, check_numpyslicesdataset, check_device_send, \
    check_take, check_project, check_imagefolderdataset, check_mnist_cifar_dataset, check_manifestdataset, \
    check_tfrecorddataset, check_vocdataset, check_cocodataset, check_celebadataset, check_minddataset, \
    check_generatordataset, check_sync_wait, check_zip_dataset, check_add_column, check_textfiledataset, check_concat, \
    check_random_dataset, check_split, check_bucket_batch_by_length, check_cluedataset, check_save, check_csvdataset, \
    check_paddeddataset, check_tuple_iterator, check_dict_iterator, check_schema, check_to_device_send
from ..core.config import get_callback_timeout, _init_device_info
from ..core.datatypes import mstype_to_detype, mstypelist_to_detypelist
from ..core.validator_helpers import replace_none

try:
    context = import_module("mindspore.context")
except ModuleNotFoundError:
    context = None


class Shuffle(str, Enum):
    GLOBAL: str = "global"
    FILES: str = "file"


@check_zip
def zip(datasets):
    """
    Zip the datasets in the input tuple of datasets.

    Args:
        datasets (tuple of class Dataset): A tuple of datasets to be zipped together.
            The number of datasets must be more than 1.

    Returns:
        ZipDataset, dataset zipped.

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
         dict, mapping dict of operator id and corresponding process id.
    """
    global _OP_PROCESS
    process_info = _OP_PROCESS
    op_process = dict()
    keys = process_info.keys()
    fetched_all = True
    for key in keys:
        op_process[key] = list(process_info[key][1])
        item_full = (len(process_info[key][1]) == process_info[key][0])
        fetched_all = fetched_all and item_full
    return op_process, fetched_all


class Dataset:
    """
    Abstract class to represent a dataset in DataEngine's data pipeline.

    This class is the base class of SourceDataset and Dataset, and represents
    a node in the data flow graph.

    Args:
        num_parallel_workers (int, optional): Number of workers to process the dataset in parallel
            (default=None).
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

        # todo check the following:
        self._device_iter = 0
        self._input_indexs = ()
        self.saved_output_types = None
        self.saved_output_shapes = None
        self._col_names = None
        self.dataset_size = None
        self._batch_size = None
        self._num_classes = None
        self._repeat_count = None
        self._class_indexing = None
        self._sync = False

    def create_ir_tree(self):
        """
        Internal method to create an IR tree.

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

    def close_pool(self):
        """
        Close multiprocessing pool in dataset.
        """
        if hasattr(self, 'process_pool') and self.process_pool is not None:
            self.process_pool.close()
        for child in self.children:
            child.close_pool()

    @staticmethod
    def _get_operator_id(dataset):
        """
        Internal method to iterate the tree and obtain op_id of each operator.

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
                    if isinstance(d, GeneratorDataset) and d.sample_fn:
                        if d.sample_fn.pid:
                            generator_process[operator_id] = [d.num_parallel_workers, set(d.sample_fn.pid)]

            operator_id = operator_id + 1
            return process_name(temp, operator_id)

        process_name([dataset], op_id)
        if generator_process:
            global _OP_PROCESS
            _OP_PROCESS.update(generator_process)
        return op_name

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

    def iterator_bootstrap(self):
        pass

    @staticmethod
    def _noop_mode():
        if _is_role_sched() or _is_role_pserver():
            return True
        return False

    def __add__(self, datasets):
        return self.concat(datasets)

    def to_json(self, filename=""):
        """
        Serialize a pipeline into JSON string and dump into file if filename is provided.

        Args:
            filename (str): filename of json file to be saved as

        Returns:
            str, JSON string of the pipeline.
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
        bucketed based on its length and bucket_boundaries. When a bucket reaches its
        corresponding size specified in bucket_batch_sizes, the entire bucket will be
        padded according to batch_info, and then batched. Each batch will be full,
        except for maybe the last batch for each bucket.

        Args:
            column_names (list[str]): Columns passed to element_length_function.
            bucket_boundaries (list[int]): A list consisting of the upper boundaries
                of the buckets. Must be strictly increasing. If there are n boundaries,
                n+1 buckets are created: One bucket for [0, bucket_boundaries[0]), one
                bucket for [bucket_boundaries[i], bucket_boundaries[i+1]) for each
                0<i<n, and one bucket for [bucket_boundaries[n-1], inf).
            bucket_batch_sizes (list[int]): A list consisting of the batch sizes for
                each bucket. Must contain len(bucket_boundaries)+1 elements.
            element_length_function (Callable, optional): A function that takes in
                len(column_names) arguments and returns an int. If no value is
                provided, then len(column_names) must be 1, and the size of the first
                dimension of that column will be taken as the length (default=None).
            pad_info (dict, optional): Represents how to batch each column. The key
                corresponds to the column name, and the value must be a tuple of 2 elements.
                The first element corresponds to the shape to pad to, and the second
                element corresponds to the value to pad with. If a column is not
                specified, then that column will be padded to the longest in the current
                batch, and 0 will be used as the padding value. Any None dimensions will
                be padded to the longest in the current batch, unless if
                pad_to_bucket_boundary is True. If no padding is wanted, set pad_info
                to None (default=None).
            pad_to_bucket_boundary (bool, optional): If True, will pad each None
                dimension in pad_info to the bucket_boundary minus 1. If there are any
                elements that fall into the last bucket, an error will occur
                (default=False).
            drop_remainder (bool, optional): If True, will drop the last batch for each
                bucket if it is not a full batch (default=False).

        Returns:
            BucketBatchByLengthDataset, dataset bucketed and batched by length.

        Examples:
            >>> # Create a dataset where every 100 rows is combined into a batch
            >>> # and drops the last incomplete batch if there is one.
            >>> import numpy as np
            >>> def generate_2_columns(n):
            ...     for i in range(n):
            ...         yield (np.array([i]), np.array([j for j in range(i + 1)]))
            >>> column_names = ["col1", "col2"]
            >>> dataset = ds.GeneratorDataset(generate_2_columns(202), column_names)
            >>> bucket_boundaries = [5, 10]
            >>> bucket_batch_sizes = [5, 1, 1]
            >>> element_length_function = (lambda col1, col2: max(len(col1), len(col2)))
            >>> # Will pad col1 to shape [2, bucket_boundaries[i]] where i is the
            >>> # index of the bucket that is currently being batched.
            >>> # Will pad col2 to a shape where each dimension is the longest in all
            >>> # the elements currently being batched.
            >>> pad_info = {"col1": ([2, None], -1)}
            >>> pad_to_bucket_boundary = True
            >>> dataset = dataset.bucket_batch_by_length(column_names, bucket_boundaries,
            ...                                          bucket_batch_sizes,
            ...                                          element_length_function, pad_info,
            ...                                          pad_to_bucket_boundary)
        """
        return BucketBatchByLengthDataset(self, column_names, bucket_boundaries, bucket_batch_sizes,
                                          element_length_function, pad_info, pad_to_bucket_boundary, drop_remainder)

    @check_batch
    def batch(self, batch_size, drop_remainder=False, num_parallel_workers=None, per_batch_map=None,
              input_columns=None, output_columns=None, column_order=None, pad_info=None, python_multiprocessing=False):
        """
        Combine batch_size number of consecutive rows into batches.

        For any child node, a batch is treated as a single row.
        For any column, all the elements within that column must have the same shape.
        If a per_batch_map callable is provided, it will be applied to the batches of tensors.

        Note:
            The order of using repeat and batch reflects the number of batches and per_batch_map.
            It is recommended that the repeat operation be used after the batch operation.

        Args:
            batch_size (int or function): The number of rows each batch is created with. An
                int or callable which takes exactly 1 parameter, BatchInfo.
            drop_remainder (bool, optional): Determines whether or not to drop the last
                possibly incomplete batch (default=False). If True, and if there are less
                than batch_size rows available to make the last batch, then those rows will
                be dropped and not propagated to the child node.
            num_parallel_workers (int, optional): Number of workers to process the dataset in parallel (default=None).
            per_batch_map (callable, optional): Per batch map callable. A callable which takes
                (list[Tensor], list[Tensor], ..., BatchInfo) as input parameters. Each list[Tensor] represents a batch
                of Tensors on a given column. The number of lists should match with number of entries in input_columns.
                The last parameter of the callable should always be a BatchInfo object. Per_batch_map should return
                (list[Tensor], list[Tensor], ...). The length of each list in output should be same as the input.
                output_columns is required if the number of output lists is different from input.
            input_columns (Union[str, list[str]], optional): List of names of the input columns. The size of the list
                should match with signature of per_batch_map callable.
            output_columns (Union[str, list[str]], optional): List of names assigned to the columns
                outputted by the last operation. This parameter is mandatory if len(input_columns) !=
                len(output_columns). The size of this list must match the number of output
                columns of the last operation. (default=None, output columns will have the same
                name as the input columns, i.e., the columns will be replaced).
            column_order (Union[str, list[str]], optional): List of all the desired columns to propagate to
                the child node. This list must be a subset of all the columns in the dataset after
                all operations are applied. The order of the columns in each row propagated to the
                child node follow the order they appear in this list. The parameter is mandatory
                if the len(input_columns) != len(output_columns). (default=None, all columns
                will be propagated to the child node, the order of the columns will remain the
                same).
            pad_info (dict, optional): Whether to perform padding on selected columns. pad_info={"col1":([224,224],0)}
                would pad column with name "col1" to a tensor of size [224,224] and fill the missing with 0.
            python_multiprocessing (bool, optional): Parallelize Python function per_batch_map with multiple worker
             processes. This option could be beneficial if the function is computational heavy (default=False).

        Returns:
            BatchDataset, dataset batched.

        Examples:
            >>> # Create a dataset where every 100 rows is combined into a batch
            >>> # and drops the last incomplete batch if there is one.
            >>> dataset = dataset.batch(100, True)
            >>> # resize image according to its batch number, if it's 5-th batch, resize to (5^2, 5^2) = (25, 25)
            >>> def np_resize(col, batchInfo):
            ...     output = col.copy()
            ...     s = (batchInfo.get_batch_num() + 1) ** 2
            ...     index = 0
            ...     for c in col:
            ...         img = Image.fromarray(c.astype('uint8')).convert('RGB')
            ...         img = img.resize((s, s), Image.ANTIALIAS)
            ...         output[index] = np.array(img)
            ...         index += 1
            ...     return (output,)
            >>> dataset = dataset.batch(batch_size=8, input_columns=["image"], per_batch_map=np_resize)
        """
        return BatchDataset(self, batch_size, drop_remainder, num_parallel_workers, per_batch_map, input_columns,
                            output_columns, column_order, pad_info, python_multiprocessing)

    @check_sync_wait
    def sync_wait(self, condition_name, num_batch=1, callback=None):
        """
        Add a blocking condition to the input Dataset.

        Args:
            condition_name (str): The condition name that is used to toggle sending next row.
            num_batch (int): the number of batches without blocking at the start of each epoch.
            callback (function): The callback function that will be invoked when sync_update is called.

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
        Randomly shuffles the rows of this dataset using the following algorithm:

        1. Make a shuffle buffer that contains the first buffer_size rows.
        2. Randomly select an element from the shuffle buffer to be the next row
           propagated to the child node.
        3. Get the next row (if any) from the parent node and put it in the shuffle buffer.
        4. Repeat steps 2 and 3 until there are no more rows left in the shuffle buffer.

        A seed can be provided to be used on the first epoch. In every subsequent
        epoch, the seed is changed to a new one, randomly generated value.

        Args:
            buffer_size (int): The size of the buffer (must be larger than 1) for
                shuffling. Setting buffer_size equal to the number of rows in the entire
                dataset will result in a global shuffle.

        Returns:
            ShuffleDataset, dataset shuffled.

        Raises:
            RuntimeError: If exist sync operators before shuffle.

        Examples:
            >>> # dataset is an instance of Dataset object.
            >>> # Optionally set the seed for the first epoch
            >>> ds.config.set_seed(58)
            >>> # Create a shuffled dataset using a shuffle buffer of size 4
            >>> dataset = dataset.shuffle(4)
        """
        return ShuffleDataset(self, buffer_size)

    def flat_map(self, func):
        """
        Map `func` to each row in dataset and flatten the result.

        The specified `func` is a function that must take one 'Ndarray' as input
        and return a 'Dataset'.

        Args:
            func (function): A function that must take one 'Ndarray' as an argument and
                return a 'Dataset'.

        Returns:
            Dataset, dataset applied by the function.

        Examples:
            >>> # use NumpySlicesDataset as an example
            >>> dataset = ds.NumpySlicesDataset([[0, 1], [2, 3]])
            >>>
            >>> def flat_map_func(array):
            ...     # create a NumpySlicesDataset with the array
            ...     dataset = ds.NumpySlicesDataset(array)
            ...     # repeat the dataset twice
            ...     dataset = dataset.repeat(2)
            ...     return dataset
            >>>
            >>> dataset = dataset.flat_map(flat_map_func)
            >>> # [[0, 1], [0, 1], [2, 3], [2, 3]]

        Raises:
            TypeError: If `func` is not a function.
            TypeError: If `func` doesn't return a Dataset.
        """
        dataset = None
        if not hasattr(func, '__call__'):
            logger.error("func must be a function.")
            raise TypeError("func must be a function.")

        for row_data in self.create_tuple_iterator(output_numpy=True):
            if dataset is None:
                dataset = func(row_data)
            else:
                dataset += func(row_data)

        if not isinstance(dataset, Dataset):
            logger.error("flat_map must return a Dataset object.")
            raise TypeError("flat_map must return a Dataset object.")
        return dataset

    @check_map
    def map(self, operations, input_columns=None, output_columns=None, column_order=None,
            num_parallel_workers=None, python_multiprocessing=False, cache=None, callbacks=None):
        """
        Apply each operation in operations to this dataset.

        The order of operations is determined by the position of each operation in the operations parameter.
        operations[0] will be applied first, then operations[1], then operations[2], etc.

        Each operation will be passed one or more columns from the dataset as input, and zero or
        more columns will be outputted. The first operation will be passed the columns specified
        in input_columns as input. If there is more than one operator in operations, the outputted
        columns of the previous operation are used as the input columns for the next operation.
        The columns outputted by the very last operation will be assigned names specified by
        output_columns.

        Only the columns specified in column_order will be propagated to the child node. These
        columns will be in the same order as specified in column_order.

        Args:
            operations (Union[list[TensorOp], list[functions]]): List of operations to be
                applied on the dataset. Operations are applied in the order they appear in this list.
            input_columns (Union[str, list[str]], optional): List of the names of the columns that will be passed to
                the first operation as input. The size of this list must match the number of
                input columns expected by the first operator. (default=None, the first
                operation will be passed however many columns that is required, starting from
                the first column).
            output_columns (Union[str, list[str]], optional): List of names assigned to the columns outputted by
                the last operation. This parameter is mandatory if len(input_columns) !=
                len(output_columns). The size of this list must match the number of output
                columns of the last operation. (default=None, output columns will have the same
                name as the input columns, i.e., the columns will be replaced).
            column_order (list[str], optional): List of all the desired columns to propagate to the
                child node. This list must be a subset of all the columns in the dataset after
                all operations are applied. The order of the columns in each row propagated to the
                child node follow the order they appear in this list. The parameter is mandatory
                if the len(input_columns) != len(output_columns). (default=None, all columns
                will be propagated to the child node, the order of the columns will remain the
                same).
            num_parallel_workers (int, optional): Number of threads used to process the dataset in
                parallel (default=None, the value from the configuration will be used).
            python_multiprocessing (bool, optional): Parallelize Python operations with multiple worker processes. This
                option could be beneficial if the Python operation is computational heavy (default=False).
            cache (DatasetCache, optional): Use tensor caching service to speed up dataset processing.
                (default=None, which means no cache is used).
            callbacks: (DSCallback, list[DSCallback], optional): List of Dataset callbacks to be called (Default=None).


        Returns:
            MapDataset, dataset after mapping operation.

        Examples:
            >>> # dataset is an instance of Dataset which has 2 columns, "image" and "label".
            >>>
            >>> # Define two operations, where each operation accepts 1 input column and outputs 1 column.
            >>> decode_op = c_vision.Decode(rgb=True)
            >>> random_jitter_op = c_vision.RandomColorAdjust(brightness=(0.8, 0.8), contrast=(1, 1),
            ...                                               saturation=(1, 1), hue=(0, 0))
            >>>
            >>> # 1) Simple map example.
            >>>
            >>> # Apply decode_op on column "image". This column will be replaced by the outputted
            >>> # column of decode_op. Since column_order is not provided, both columns "image"
            >>> # and "label" will be propagated to the child node in their original order.
            >>> dataset = dataset.map(operations=[decode_op], input_columns=["image"])
            >>>
            >>> # Decode and rename column "image" to "decoded_image".
            >>> dataset = dataset.map(operations=[decode_op], input_columns=["image"], output_columns=["decoded_image"])
            >>>
            >>> # Specify the order of the output columns.
            >>> dataset = dataset.map(operations=[decode_op], input_columns=["image"],
            ...                       output_columns=None, column_order=["label", "image"])
            >>>
            >>> # Rename column "image" to "decoded_image" and also specify the order of the output columns.
            >>> dataset = dataset.map(operations=[decode_op], input_columns=["image"],
            ...                       output_columns=["decoded_image"], column_order=["label", "decoded_image"])
            >>>
            >>> # Rename column "image" to "decoded_image" and keep only this column.
            >>> dataset = dataset.map(operations=[decode_op], input_columns=["image"],
            ...                       output_columns=["decoded_image"], column_order=["decoded_image"])
            >>>
            >>> # A simple example for mapping pyfunc. Renaming columns and specifying column order
            >>> # work in the same way as the previous examples.
            >>> dataset = ds.NumpySlicesDataset(data=[[0, 1, 2]], column_names=["data"])
            >>> dataset = dataset.map(operations=[(lambda x: x + 1)], input_columns=["data"])
            >>>
            >>> # 2) Map example with more than one operation.
            >>>
            >>> # Create a dataset where the images are decoded, then randomly color jittered.
            >>> # decode_op takes column "image" as input and outputs one column. The column
            >>> # outputted by decode_op is passed as input to random_jitter_op.
            >>> # random_jitter_op will output one column. Column "image" will be replaced by
            >>> # the column outputted by random_jitter_op (the very last operation). All other
            >>> # columns are unchanged. Since column_order is not specified, the order of the
            >>> # columns will remain the same.
            >>> dataset = dataset.map(operations=[decode_op, random_jitter_op], input_columns=["image"])
            >>>
            >>> # Rename the column outputted by random_jitter_op to "image_mapped".
            >>> # Specifying column order works in the same way as examples in 1).
            >>> dataset = dataset.map(operations=[decode_op, random_jitter_op], input_columns=["image"],
            ...                       output_columns=["image_mapped"])
            >>>
            >>> # Map with multiple operations using pyfunc. Renaming columns and specifying column order
            >>> # work in the same way as examples in 1).
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
            >>>
            >>> # Note: Since the number of input columns is not the same as the number of
            >>> # output columns, the output_columns and column_order parameters must be
            >>> # specified. Otherwise, this map call will also result in an error.
            >>>
            >>> dataset = ds.NumpySlicesDataset(data=([[0, 1, 2]], [[3, 4, 5]]), column_names=["x", "y"])
            >>>
            >>> # Propagate all columns to the child node in this order:
            >>> dataset = dataset.map(operations, input_columns=["x", "y"],
            ...                       output_columns=["mod2", "mod3", "mod5", "mod7"],
            ...                       column_order=["mod2", "mod3", "mod5", "mod7"])
            >>>
            >>> # Propagate some columns to the child node in this order:
            >>> dataset = dataset.map(operations, input_columns=["x", "y"],
            ...                       output_columns=["mod2", "mod3", "mod5", "mod7"],
            ...                       column_order=["mod7", "mod3", "col2"])
        """

        return MapDataset(self, operations, input_columns, output_columns, column_order, num_parallel_workers,
                          python_multiprocessing, cache, callbacks)

    @check_filter
    def filter(self, predicate, input_columns=None, num_parallel_workers=None):
        """
        Filter dataset by predicate.

        Note:
             If input_columns not provided or empty, all columns will be used.

        Args:
            predicate (callable): Python callable which returns a boolean value. If False then filter the element.
            input_columns (Union[str, list[str]], optional): List of names of the input columns, when
                default=None, the predicate will be applied on all columns in the dataset.
            num_parallel_workers (int, optional): Number of workers to process the dataset
                in parallel (default=None).

        Returns:
            FilterDataset, dataset filtered.

        Examples:
            >>> # generator data(0 ~ 63)
            >>> # filter the data that greater than or equal to 11
            >>> dataset = dataset.filter(predicate=lambda data: data < 11, input_columns = ["data"])
        """
        return FilterDataset(self, predicate, input_columns, num_parallel_workers)

    @check_repeat
    def repeat(self, count=None):
        """
        Repeat this dataset count times. Repeat indefinitely if the count is None or -1.

        Note:
            The order of using repeat and batch reflects the number of batches. It is recommended that
            the repeat operation be used after the batch operation.
            If dataset_sink_mode is False, the repeat operation is invalid.
            If dataset_sink_mode is True, repeat count must be equal to the epoch of training. Otherwise,
            errors could occur since the amount of data is not the amount training requires.

        Args:
            count (int): Number of times the dataset is repeated (default=None).

        Returns:
            RepeatDataset, dataset repeated.

        Examples:
            >>> # dataset is an instance of Dataset object.
            >>>
            >>> # Create a dataset where the dataset is repeated for 50 epochs
            >>> dataset = dataset.repeat(50)
            >>>
            >>> # Create a dataset where each epoch is shuffled individually
            >>> dataset = dataset.shuffle(10)
            >>> dataset = dataset.repeat(50)
            >>>
            >>> # Create a dataset where the dataset is first repeated for
            >>> # 50 epochs before shuffling. The shuffle operator will treat
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
            SkipDataset, dataset skipped.

        Examples:
            >>> # dataset is an instance of Dataset object.
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
               then take given number of rows; otherwise take given number of batches.

        Args:
            count (int, optional): Number of elements to be taken from the dataset (default=-1).

        Returns:
            TakeDataset, dataset taken.

        Examples:
            >>> # dataset is an instance of Dataset object.
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

        This is a general purpose split function which can be called from any operator in the pipeline.
        There is another, optimized split function, which will be called automatically if ds.split is
        called where ds is a MappableDataset.

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

            randomize (bool, optional): Determines whether or not to split the data randomly (default=True).
                If True, the data will be randomly split. Otherwise, each split will be created with
                consecutive rows from the dataset.

        Note:
            1. Dataset cannot be sharded if split is going to be called.
            2. It is strongly recommended to not shuffle the dataset, but use randomize=True instead.
               Shuffling the dataset may not be deterministic, which means the data in each split
               will be different in each epoch.

        Raises:
            RuntimeError: If get_dataset_size returns None or is not supported for this dataset.
            RuntimeError: If sizes is list of integers and sum of all elements in sizes does not
                equal the dataset size.
            RuntimeError: If sizes is list of float and there is a split with size 0 after calculations.
            RuntimeError: If the dataset is sharded prior to calling split.
            ValueError: If sizes is list of float and not all floats are between 0 and 1, or if the
                floats don’t sum to 1.

        Returns:
            tuple(Dataset), a tuple of datasets that have been split.

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
        Zip the datasets in the input tuple of datasets. Columns in the input datasets must not have the same name.

        Args:
            datasets (Union[tuple, class Dataset]): A tuple of datasets or a single class Dataset
                to be zipped together with this dataset.

        Returns:
            ZipDataset, dataset zipped.

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
        Concatenate the datasets in the input list of datasets. The "+" operator is also supported to concatenate.

        Note:
            The column name, and rank and type of the column data must be the same in the input datasets.

        Args:
            datasets (Union[list, class Dataset]): A list of datasets or a single class Dataset
                to be concatenated together with this dataset.

        Returns:
            ConcatDataset, dataset concatenated.

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
            RenameDataset, dataset renamed.

        Examples:
            >>> # dataset is an instance of Dataset object.
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
        Project certain columns in input dataset.

        The specified columns will be selected from the dataset and passed down
        the pipeline in the order specified. The other columns are discarded.

        Args:
            columns(Union[str, list[str]]): List of names of the columns to project.

        Returns:
            ProjectDataset, dataset projected.

        Examples:
            >>> # dataset is an instance of Dataset object
            >>> columns_to_project = ["column3", "column1", "column2"]
            >>>
            >>> # Create a dataset that consists of column3, column1, column2
            >>> # in that order, regardless of the original order of columns.
            >>> dataset = dataset.project(columns=columns_to_project)
        """

        return ProjectDataset(self, columns)

    def build_vocab(self, columns, freq_range, top_k, special_tokens, special_first):
        """
        Function to create a Vocab from source dataset

        Build a vocab from a dataset. This would collect all the unique words in a dataset and return a vocab
        which contains top_k most frequent words (if top_k is specified)

        Args:

            columns(Union[str, list[str]]): Column names to get words from.
            freq_range(tuple[int]): A tuple of integers (min_frequency, max_frequency). Words within the frequency
                range would be kept. 0 <= min_frequency <= max_frequency <= total_words. min_frequency/max_frequency
                an be set to default, which corresponds to 0/total_words separately
            top_k(int): Number of words to be built into vocab. top_k most frequent words are
                taken. The top_k is taken after freq_range. If not enough top_k, all words will be taken
            special_tokens(list[str]): A list of strings, each one is a special token
            special_first(bool): Whether special_tokens will be prepended/appended to vocab, If special_tokens
                is specified and special_first is set to default, special_tokens will be prepended

        Returns:
            Vocab, vocab built from the dataset.

        Example:
            >>> def gen_corpus():
            ...     # key: word, value: number of occurrences, reason for using letters is so their order is apparent
            ...     corpus = {"Z": 4, "Y": 4, "X": 4, "W": 3, "U": 3, "V": 2, "T": 1}
            ...     for k, v in corpus.items():
            ...         yield (np.array([k] * v, dtype='S'),)
            >>> column_names = ["column1", "column2", "column3"]
            >>> dataset = ds.GeneratorDataset(gen_corpus, column_names)
            >>> dataset = dataset.build_vocab(columns=["column3", "column1", "column2"],
            ...                               freq_range=(1, 10), top_k=5,
            ...                               special_tokens=["<pad>", "<unk>"],
            ...                               special_first=True,vocab='vocab')

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
        Function to create a SentencePieceVocab from source dataset

        Build a SentencePieceVocab from a dataset.

        Args:

            columns(list[str]): Column names to get words from.
            vocab_size(int): Vocabulary size.
            character_coverage(int): Percentage of characters covered by the model, must be between
                        0.98 and 1.0 Good defaults are: 0.9995 for languages with rich character sets like
                        Japanese or Chinese character sets, and 1.0 for other languages with small character sets.
            model_type(SentencePieceModel): Model type. Choose from unigram (default), bpe, char, or word.
                                        The input sentence must be pretokenized when using word type.
            params(dict): contains more optional parameters of sentencepiece library

        Returns:
            SentencePieceVocab, vocab built from the dataset.

        Example:
            >>> from mindspore.dataset.text import SentencePieceModel
            >>> def gen_corpus():
            ...     # key: word, value: number of occurrences, reason for using letters is so their order is apparent
            ...     corpus = {"Z": 4, "Y": 4, "X": 4, "W": 3, "U": 3, "V": 2, "T": 1}
            ...     for k, v in corpus.items():
            ...         yield (np.array([k] * v, dtype='S'),)
            >>> column_names = ["column1","column2","column3"]
            >>> dataset = ds.GeneratorDataset(gen_corpus, column_names)
            >>> dataset = dataset.build_sentencepiece_vocab(columns=["column3", "column1", "column2"],
            ...                                             vocab_size=5000,
            ...                                             character_coverage=0.9995,
            ...                                             model_type=SentencePieceModel.Unigram,
            ...                                             params={},vocab='vocab')
        """
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

    def apply(self, apply_func):
        """
        Apply a function in this dataset.

        Args:
            apply_func (function): A function that must take one 'Dataset' as an argument and
                                   return a preprogressing 'Dataset'.

        Returns:
            Dataset, dataset applied by the function.

        Examples:
            >>> # dataset is an instance of Dataset object
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
    def device_que(self, prefetch_size=None, send_epoch_end=True, create_data_info_queue=False):
        """
        Return a transferred Dataset that transfers data through a device.

        Args:
            prefetch_size (int, optional): Prefetch number of records ahead of the
                user's request (default=None).
            send_epoch_end (bool, optional): Whether to send end of sequence to device or not (default=True).
            create_data_info_queue (bool, optional): Whether to create queue which stores
                types and shapes of data or not(default=False).

        Note:
            If device is Ascend, features of data will be transferred one by one. The limitation
            of data transmission per time is 256M.

        Returns:
            TransferDataset, dataset for transferring.
        """
        return self.to_device(send_epoch_end=send_epoch_end, create_data_info_queue=create_data_info_queue)

    @check_device_send
    def to_device(self, send_epoch_end=True, create_data_info_queue=False):
        """
        Transfer data through CPU, GPU or Ascend devices.

        Args:
            send_epoch_end (bool, optional): Whether to send end of sequence to device or not (default=True).
            create_data_info_queue (bool, optional): Whether to create queue which stores
                types and shapes of data or not(default=False).

        Note:
            If device is Ascend, features of data will be transferred one by one. The limitation
            of data transmission per time is 256M.

        Returns:
            TransferDataset, dataset for transferring.

        Raises:
            RuntimeError: If distribution file path is given but failed to read.
        """
        return TransferDataset(self, send_epoch_end, create_data_info_queue)

    @check_save
    def save(self, file_name, num_files=1, file_type='mindrecord'):
        """
        Save the dynamic data processed by the dataset pipeline in common dataset format.
        Supported dataset formats: 'mindrecord' only

        Implicit type casting exists when saving data as 'mindrecord'. The table below shows how to do type casting.

        .. list-table:: Implicit Type Casting when Saving as 'mindrecord'
           :widths: 25 25 50
           :header-rows: 1

           * - Type in 'dataset'
             - Type in 'mindrecord'
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
            2. Before calling the function, do not use batch operator, repeat operator or data augmentation operators
               with random attribute in map operator.
            3. Can not save number type tensor whose shape is dynamic.
            4. Mindrecord does not support DE_UINT64, multi-dimensional DE_UINT8(drop dimension) nor
               multi-dimensional DE_STRING.

        Args:
            file_name (str): Path to dataset file.
            num_files (int, optional): Number of dataset files (default=1).
            file_type (str, optional): Dataset format (default='mindrecord').

        """
        ir_tree, api_tree = self.create_ir_tree()

        runtime_context = cde.PythonRuntimeContext()
        runtime_context.Init()
        consumer = cde.PythonSaveToDisk(file_name, num_files, file_type)
        consumer.Init(ir_tree)
        runtime_context.AssignConsumer(consumer)

        consumer.Save()
        del api_tree

    @check_tuple_iterator
    def create_tuple_iterator(self, columns=None, num_epochs=-1, output_numpy=False, do_copy=True):
        """
        Create an iterator over the dataset. The data retrieved will be a list of ndarrays of data.

        To specify which columns to list and the order needed, use columns_list. If columns_list
        is not provided, the order of the columns will not be changed.

        Args:
            columns (list[str], optional): List of columns to be used to specify the order of columns
                (default=None, means all columns).
            num_epochs (int, optional): Maximum number of epochs that iterator can be iterated.
                (default=-1, iterator can be iterated infinite number of epochs)
            output_numpy (bool, optional): Whether or not to output NumPy datatype.
                If output_numpy=False, iterator will output MSTensor (default=False).
            do_copy (bool, optional): when output data type is mindspore.Tensor,
                use this param to select the conversion method, only take False for better performance (default=True).

        Returns:
            TupleIterator, tuple iterator over the dataset.

        Examples:
            >>> # dataset is an instance of Dataset object
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
            return DummyIterator(self, 'tuple')
        return TupleIterator(self, columns, num_epochs, output_numpy, do_copy)

    @check_dict_iterator
    def create_dict_iterator(self, num_epochs=-1, output_numpy=False):
        """
        Create an iterator over the dataset. The data retrieved will be a dictionary.

        The order of the columns in the dictionary may not be the same as the original order.

        Args:
            num_epochs (int, optional): Maximum number of epochs that iterator can be iterated
                (default=-1, iterator can be iterated infinite number of epochs).
            output_numpy (bool, optional): Whether or not to output NumPy datatype,
                if output_numpy=False, iterator will output MSTensor (default=False).

        Returns:
            DictIterator, dictionary iterator over the dataset.

        Examples:
            >>> # dataset is an instance of Dataset object
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
            return DummyIterator(self, 'dict')
        return DictIterator(self, num_epochs, output_numpy)

    def __iter__(self):
        """Create an iterator over the dataset."""
        return self.create_tuple_iterator(num_epochs=1)

    @property
    def input_indexs(self):
        """
        Get Input Index Information

        Returns:
            tuple, tuple of the input index information.

        Examples:
            >>> # dataset is an instance of Dataset object
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

    def _init_size_getter(self):
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
        Get names of the columns in the dataset

        Returns:
            list, list of column names in the dataset.
        """
        if self._col_names is None:
            runtime_getter = self._init_tree_getters()
            self._col_names = runtime_getter[0].GetColumnNames()
            self.close_pool()
        return self._col_names

    def output_shapes(self):
        """
        Get the shapes of output data.

        Returns:
            list, list of shapes of each column.
        """
        if self.saved_output_shapes is None:
            runtime_getter = self._init_tree_getters()
            self.saved_output_shapes = runtime_getter[0].GetOutputShapes()
            self.saved_output_types = runtime_getter[0].GetOutputTypes()
            self.close_pool()
        return self.saved_output_shapes

    def output_types(self):
        """
        Get the types of output data.

        Returns:
            list, list of data types.
        """
        if self.saved_output_types is None:
            runtime_getter = self._init_tree_getters()
            self.saved_output_shapes = runtime_getter[0].GetOutputShapes()
            self.saved_output_types = runtime_getter[0].GetOutputTypes()
            self.close_pool()
        return self.saved_output_types

    def get_dataset_size(self):
        """
        Get the number of batches in an epoch.

        Returns:
            int, number of batches.
        """
        if self.dataset_size is None:
            runtime_getter = self._init_size_getter()
            self.dataset_size = runtime_getter[0].GetDatasetSize(False)
            self.close_pool()
        return self.dataset_size

    def num_classes(self):
        """
        Get the number of classes in a dataset.

        Returns:
            int, number of classes.
        """
        if self._num_classes is None:
            runtime_getter = self._init_tree_getters()
            self._num_classes = runtime_getter[0].GetNumClasses()
            self.close_pool()
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
                sync_wait operator (default=None).
            data (Any): The data passed to the callback, user defined (default=None).
        """
        if (not isinstance(num_batch, int) and num_batch is not None) or \
                (isinstance(num_batch, int) and num_batch <= 0):
            # throwing exception, disable all sync_wait in pipeline
            self.disable_sync()
            raise RuntimeError("Sync_update batch size can only be positive, got : {}.".format(num_batch))
        notifiers_dict = self.get_sync_notifiers()
        if condition_name not in notifiers_dict:
            # throwing exception, disable all sync_wait in pipeline
            self.disable_sync()
            raise RuntimeError("Condition name not found.")
        if num_batch is not None:
            num_batch *= self.get_batch_size()
        notifiers_dict[condition_name](num_batch, data)

    def get_batch_size(self):
        """
        Get the size of a batch.

        Returns:
            int, the number of data in a batch.
        """
        if self._batch_size is None:
            runtime_getter = self._init_tree_getters()
            self._batch_size = runtime_getter[0].GetBatchSize()
        if self._batch_size is None:
            self._batch_size = 1
        return self._batch_size

    def get_repeat_count(self):
        """
        Get the replication times in RepeatDataset else 1.

        Returns:
            int, the count of repeat.
        """
        if self._repeat_count is None:
            runtime_getter = self._init_tree_getters()
            self._repeat_count = runtime_getter[0].GetRepeatCount()
        if self._repeat_count is None:
            self._repeat_count = 1
        return self._repeat_count

    def get_class_indexing(self):
        """
        Get the class index.

        Returns:
            dict, a str-to-int mapping from label name to index.
            dict, a str-to-list<int> mapping from label name to index for Coco ONLY. The second number
            in the list is used to indicate the super category
        """
        if self.children:
            return self.children[0].get_class_indexing()
        return {}

    def reset(self):
        """Reset the dataset for next epoch."""

    def is_shuffled(self):
        for input_dataset in self.children:
            if input_dataset.is_shuffled():
                return True

        return False

    def is_sharded(self):
        for input_dataset in self.children:
            if input_dataset.is_sharded():
                return True

        return False

    def parse(self, children=None):
        raise NotImplementedError("Dataset has to implement parse method.")

    def post_parse(self, ir_node):
        if self.cache:
            ir_node = ir_node.set_cache_client(self.cache.cache_client)
        if self.num_parallel_workers:
            ir_node = ir_node.set_num_workers(self.num_parallel_workers)

        return ir_node


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
            raise TypeError(
                "shuffle must be of boolean or enum of 'Shuffle' values like 'Shuffle.GLOBAL' or 'Shuffle.FILES'.")

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
        super().__init__(num_parallel_workers=num_parallel_workers, num_samples=num_samples, shuffle=shuffle,
                         num_shards=num_shards, shard_id=shard_id, cache=cache)
        self.shuffle_flag = replace_none(shuffle, True)
        self.sampler = samplers.select_sampler(num_samples, sampler, shuffle, num_shards, shard_id)

    def add_sampler(self, new_sampler):
        # note: By adding a sampler, the sampled IDs will flow to new_sampler
        # after first passing through the current samplers attached to this dataset.
        self.dataset_size = None
        new_sampler.add_child(self.sampler)
        self.sampler = new_sampler

    def use_sampler(self, new_sampler):
        """
        Will make the current dataset use the new_sampler provided.

        Args:
            new_sampler (Sampler): The sampler to use for the current dataset.

        Examples:
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

            randomize (bool, optional): Determines whether or not to split the data randomly (default=True).
                If True, the data will be randomly split. Otherwise, each split will be created with
                consecutive rows from the dataset.

        Note:
            1. There is an optimized split function, which will be called automatically when the dataset
               that calls this function is a MappableDataset.
            2. Dataset should not be sharded if split is going to be called. Instead, create a
               DistributedSampler and specify a split to shard after splitting. If dataset is
               sharded after a split, it is strongly recommended to set the same seed in each instance
               of execution, otherwise each shard may not be part of the same split (see Examples).
            3. It is strongly recommended to not shuffle the dataset, but use randomize=True instead.
               Shuffling the dataset may not be deterministic, which means the data in each split
               will be different in each epoch. Furthermore, if sharding occurs after split, each
               shard may not be part of the same split.

        Raises:
            RuntimeError: If get_dataset_size returns None or is not supported for this dataset.
            RuntimeError: If sizes is list of integers and sum of all elements in sizes does not
                equal the dataset size.
            RuntimeError: If sizes is list of float and there is a split with size 0 after calculations.
            RuntimeError: If the dataset is sharded prior to calling split.
            ValueError: If sizes is list of float and not all floats are between 0 and 1, or if the
                floats don’t sum to 1.

        Returns:
            tuple(Dataset), a tuple of datasets that have been split.

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


class BucketBatchByLengthDataset(Dataset):
    """
    The result of applying BucketBatchByLength operator to the input dataset.
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


class BatchDataset(Dataset):
    """
    The result of applying Batch operator to the input dataset.

    Args:
        input_dataset (Dataset): Input Dataset to be batched.
        batch_size (Union[int, function]): The number of rows each batch is created with. An
            int or callable which takes exactly 1 parameter, BatchInfo.
        drop_remainder (bool, optional): Determines whether or not to drop the last
            possibly incomplete batch (default=False). If True, and if there are less
            than batch_size rows available to make the last batch, then those rows will
            be dropped and not propagated to the child node.
        num_parallel_workers (int, optional): Number of workers to process the dataset in parallel (default=None).
        per_batch_map (callable, optional): Per batch map callable. A callable which takes
            (list[Tensor], list[Tensor], ..., BatchInfo) as input parameters. Each list[Tensor] represents a batch of
            Tensors on a given column. The number of lists should match with number of entries in input_columns. The
            last parameter of the callable must always be a BatchInfo object.
        input_columns (Union[str, list[str]], optional): List of names of the input columns. The size of the list must
            match with signature of per_batch_map callable.
        output_columns (Union[str, list[str]], optional): List of names assigned to the columns outputted by
            the last operation. This parameter is mandatory if len(input_columns) !=
            len(output_columns). The size of this list must match the number of output
            columns of the last operation. (default=None, output columns will have the same
            name as the input columns, i.e., the columns will be replaced).
        column_order (Union[str, list[str]], optional): List of all the desired columns to propagate to the
            child node. This list must be a subset of all the columns in the dataset after
            all operations are applied. The order of the columns in each row propagated to the
            child node follow the order they appear in this list. The parameter is mandatory
            if the len(input_columns) != len(output_columns). (default=None, all columns
            will be propagated to the child node, the order of the columns will remain the
            same).
        pad_info (dict, optional): Whether to perform padding on selected columns. pad_info={"col1":([224,224],0)}
            will pad column with name "col1" to a tensor of size [224,224] and fill the missing with 0.

    """

    def __init__(self, input_dataset, batch_size, drop_remainder=False, num_parallel_workers=None, per_batch_map=None,
                 input_columns=None, output_columns=None, column_order=None, pad_info=None,
                 python_multiprocessing=False):
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
        self.column_order = to_list(column_order)

        self.pad = bool(pad_info is not None)
        self.pad_info = replace_none(pad_info, dict())

        self.python_multiprocessing = python_multiprocessing
        self.process_pool = None
        self.hook = None

    def parse(self, children=None):
        return cde.BatchNode(children[0], self.batch_size, self.drop_remainder, self.pad, self.input_columns,
                             self.output_columns, self.column_order, self.batch_size_func, self.per_batch_map,
                             self.pad_info)

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
        if self.python_multiprocessing:
            # Construct pool with the callable list
            # The callable list and _pyfunc_worker_init are used to pass lambda function in to subprocesses
            self.process_pool = multiprocessing.Pool(processes=self.num_parallel_workers,
                                                     initializer=_pyfunc_worker_init, initargs=([self.per_batch_map],))

            idx = 0
            global _OP_NAME, _OP_PROCESS, _LOCK
            op_id = _OP_NAME[str(self)]
            process_id = {op_id: [self.num_parallel_workers, set()]}
            # obtain process id from multiprocessing.pool
            for pool in self.process_pool._pool:  # pylint: disable=W0212
                process_id[op_id][1].add(pool.pid)
            with _LOCK:
                _OP_PROCESS.update(process_id)

            # Wrap per_batch_map into _PythonCallable
            self.per_batch_map = _PythonCallable(self.per_batch_map, idx, self.process_pool)
            self.hook = _ExceptHookHandler()
            atexit.register(_mp_pool_exit_preprocess)
            # If python version greater than 3.8, we need to close ThreadPool in atexit for unclean pool teardown.
            if sys.version_info >= (3, 8):
                atexit.register(self.process_pool.close)

    def __del__(self):
        if hasattr(self, 'process_pool') and self.process_pool is not None:
            self.process_pool.close()


class BatchInfo(cde.CBatchInfo):
    """
    The information object associates with the current batch of tensors.
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
        callback (function): The callback function that will be called when release is called (default=None).
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


class SyncWaitDataset(Dataset):
    """
    The result of adding a blocking condition to the input Dataset.

    Args:
        input_dataset (Dataset): Input dataset to apply flow control.
        num_batch (int): Number of batches without blocking at the start of each epoch.
        condition_name (str): Condition name that is used to toggle sending next row.
        callback (function): Callback function that will be invoked when sync_update is called (default=None).

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
        if isinstance(dataset, BatchDataset):
            return True
        flag = False
        for input_dataset in dataset.children:
            flag = flag | SyncWaitDataset._is_ancestor_of_batch(input_dataset)
        return flag

    def iterator_bootstrap(self):
        self._pair.reset()


class ShuffleDataset(Dataset):
    """
    The result of applying Shuffle operator to the input Dataset.

    Args:
        input_dataset (Dataset): Input Dataset to be shuffled.
        buffer_size (int): Size of the buffer.

    Raises:
        RuntimeError: If exist sync operators before shuffle.
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
_GLOBAL_PYFUNC_LIST = []
_OP_NAME = dict()
_OP_PROCESS = dict()
_LOCK = multiprocessing.Lock()


# Pyfunc worker init function
# Python multiprocessing library forbid sending lambda function through pipe.
# This init function allow us to add all Python function to a global collection and then fork afterwards.
def _pyfunc_worker_init(pyfunc_list):
    global _GLOBAL_PYFUNC_LIST
    _GLOBAL_PYFUNC_LIST = pyfunc_list


# Pyfunc worker execution function
# All exceptions will be raised to main processes
def _pyfunc_worker_exec(index, *args):
    """
    Internal function for call certain pyfunc in python process.
    """
    # Some threads in multiprocess.pool can't process sigint signal,
    # and will occur hang problem, so ctrl+c will pass to parent process.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    return _GLOBAL_PYFUNC_LIST[index](*args)


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
        # Python callable index for subprocess _GLOBAL_PYFUNC_LIST
        self.idx = idx

    def __call__(self, *args):
        # note here: the RUN state of python3.7 and python3.8 is different:
        # python3.7: RUN = 0
        # python3.8: RUN = "RUN"
        # so we use self.pool._state == RUN instead and we can't use _state == 0 any more.
        if self.pool is not None and self.pool._state == RUN and check_iterator_cleanup() is False:  # pylint: disable=W0212
            # This call will send the tensors along with Python callable index to the process pool.
            # Block, yield GIL. Current thread will reacquire GIL once result is returned.
            result = self.pool.apply_async(_pyfunc_worker_exec, [self.idx, *args])

            # todo this check might be wrong
            while check_iterator_cleanup() is False:
                try:
                    return result.get(30)
                except multiprocessing.TimeoutError:
                    continue
                except KeyboardInterrupt:
                    _set_iterator_cleanup()
                    self.pool.close()
                    self.pool.join()
                    raise Exception("Multiprocess MapOp worker receives KeyboardInterrupt.")
            return (None,)
        # Invoke original Python callable in master process in case the pool is gone.
        return self.py_callable(*args)


def _mp_pool_exit_preprocess():
    if check_iterator_cleanup() is False:
        logger.info("Execution preprocessing process before map exit.")
        # Set the iterator_cleanup flag to True before exiting, and wait 3s for all apply_async
        # applied to the multiprocessing task to prevent multiprocessing from hang when exiting
        _set_iterator_cleanup()
        time.sleep(3)


class _ExceptHookHandler:
    def __init__(self):
        sys.excepthook = self.__handler_exception

    def __handler_exception(self, ex_type, value, tb):
        logger.error("Uncaught exception: ", exc_info=(ex_type, value, tb))
        _mp_pool_exit_preprocess()


class MapDataset(Dataset):
    """
    The result of applying the Map operator to the input Dataset.

    Args:
        input_dataset (Dataset): Input Dataset to be mapped.
        operations (TensorOp): A function mapping a nested structure of tensors
            to another nested structure of tensor (default=None).
        input_columns (Union[str, list[str]]): List of names of the input columns
            (default=None, the operations will be applied on the first columns in the dataset).
            The size of the list should match the number of inputs of the first operator.
        output_columns (Union[str, list[str]], optional): List of names of the output columns.
            The size of the list should match the number of outputs of the last operator
            (default=None, output columns will be the input columns, i.e., the columns will
            be replaced).
        column_order (list[str], optional): List of all the desired columns of the dataset (default=None).
            The argument is mandatory if len(input_columns) != len(output_columns).
        num_parallel_workers (int, optional): Number of workers to process the dataset
            in parallel (default=None).
        python_multiprocessing (bool, optional): Parallelize Python operations with multiple worker process. This
            option could be beneficial if the Python operation is computational heavy (default=False).
        cache (DatasetCache, optional): Use tensor caching service to speed up dataset processing.
            (default=None, which means no cache is used).
        callbacks: (DSCallback, list[DSCallback], optional): List of Dataset callbacks to be called (Default=None)

        Raises:
            ValueError: If len(input_columns) != len(output_columns) and column_order is not specified.
    """

    def __init__(self, input_dataset, operations=None, input_columns=None, output_columns=None, column_order=None,
                 num_parallel_workers=None, python_multiprocessing=False, cache=None, callbacks=None):
        super().__init__(children=input_dataset, num_parallel_workers=num_parallel_workers, cache=cache)
        self.operations = to_list(operations)
        self.operations = py_transforms.Compose.reduce(self.operations)
        self.input_columns = to_list(input_columns)
        self.output_columns = to_list(output_columns)
        self.column_order = replace_none(column_order, [])

        #  If output_columns were not provided then use input_columns
        self.output_columns = self.input_columns if not self.output_columns else self.output_columns

        # todo(crc): move to @check_map
        if self.input_columns and self.output_columns \
                and len(self.input_columns) != len(self.output_columns) \
                and not self.column_order:
            raise ValueError("When length of input_columns and output_columns are not equal,"
                             " column_order must be specified.")

        self.python_multiprocessing = python_multiprocessing
        self.process_pool = None
        self.hook = None

        self.callbacks = to_list(callbacks)

    def parse(self, children=None):
        operations = []
        for op in self.operations:
            if op and getattr(op, 'parse', None):
                operations.append(op.parse())
            else:
                operations.append(op)

        callbacks = [cb.create_runtime_obj() for cb in self.callbacks]
        return cde.MapNode(children[0], operations, self.input_columns, self.output_columns, self.column_order,
                           callbacks)

    def __deepcopy__(self, memodict):
        return self.__safe_deepcopy__(memodict, exclude=("operations", "callbacks", "__transfer_dataset__"))

    # Iterator bootstrap will be called on iterator construction.
    # A deep copy of Dataset object is created prior of iterator_bootstrap.
    # This method will create per iterator process pool and bind pyfunc execution to the pool.
    def iterator_bootstrap(self):
        """
        Per iterator bootstrap callback.
        """

        if self.python_multiprocessing:
            iter_specific_operations = []
            callable_list = []

            # Pass #1, look for Python callables and build list
            for op in self.operations:
                # our c transforms is now callable and should not be run in python multithreading
                if callable(op) and str(op).find("c_transform") < 0:
                    callable_list.append(op)

            if callable_list:
                # Construct pool with the callable list
                # The callable list and _pyfunc_worker_init are used to pass lambda function in to subprocesses
                self.process_pool = multiprocessing.Pool(processes=self.num_parallel_workers,
                                                         initializer=_pyfunc_worker_init, initargs=(callable_list,))

                # Pass #2
                idx = 0
                global _OP_NAME, _OP_PROCESS, _LOCK
                op_id = _OP_NAME[str(self)]
                # obtain process id from multiprocessing.pool
                process_id = {op_id: [self.num_parallel_workers, set()]}
                for pool in self.process_pool._pool:  # pylint: disable=W0212
                    process_id[op_id][1].add(pool.pid)
                with _LOCK:
                    _OP_PROCESS.update(process_id)
                for op in self.operations:
                    # our c transforms is now callable and should not be run in python multithreading
                    if callable(op) and str(op).find("c_transform") < 0:
                        # Wrap Python callable into _PythonCallable
                        iter_specific_operations.append(_PythonCallable(op, idx, self.process_pool))
                        idx += 1
                    else:
                        # CPP ops remain the same
                        iter_specific_operations.append(op)
                self.operations = iter_specific_operations
                self.hook = _ExceptHookHandler()
                atexit.register(_mp_pool_exit_preprocess)
                # If python version greater than 3.8, we need to close ThreadPool in atexit for unclean pool teardown.
                if sys.version_info >= (3, 8):
                    atexit.register(self.process_pool.close)

    def __del__(self):
        if hasattr(self, 'process_pool') and self.process_pool is not None:
            self.process_pool.close()
            self.process_pool.join()


class FilterDataset(Dataset):
    """
    The result of applying filter predicate to the input Dataset.

    Args:
        input_dataset (Dataset): Input Dataset to be mapped.
        predicate (callable): Python callable which returns a boolean value. If False then filter the element.
        input_columns (Union[str, list[str]], optional): List of names of the input columns
        (default=None, the predicate will be applied to all columns in the dataset).
        num_parallel_workers (int, optional): Number of workers to process the dataset
            in parallel (default=None).
    """

    def __init__(self, input_dataset, predicate, input_columns=None, num_parallel_workers=None):
        super().__init__(children=input_dataset, num_parallel_workers=num_parallel_workers)
        self.predicate = lambda *args: bool(predicate(*args))
        self.input_columns = to_list(input_columns)

    def parse(self, children=None):
        return cde.FilterNode(children[0], self.predicate, self.input_columns)


class RepeatDataset(Dataset):
    """
    The result of applying Repeat operator to the input Dataset.

    Args:
        input_dataset (Dataset): Input Dataset to be repeated.
        count (int): Number of times the dataset will be repeated (default=-1, repeat indefinitely).
    """

    def __init__(self, input_dataset, count):
        super().__init__(children=input_dataset)
        self.count = replace_none(count, -1)

    def parse(self, children=None):
        return cde.RepeatNode(children[0], self.count)


class SkipDataset(Dataset):
    """
    The result of applying Skip operator to the input Dataset.

    Args:
        input_dataset (Dataset): Input dataset to have elements skipped.
        count (int): Number of elements to be skipped in the dataset.
    """

    def __init__(self, input_dataset, count):
        super().__init__(input_dataset)
        self.count = count

    def parse(self, children=None):
        return cde.SkipNode(children[0], self.count)


class TakeDataset(Dataset):
    """
    The result of applying Take operator to the input Dataset.

    Args:
        input_dataset (Dataset): Input Dataset to have elements taken from.
        count (int): Number of elements to be taken from the dataset.
    """

    def __init__(self, input_dataset, count):
        super().__init__(children=input_dataset)
        self.count = count

    def parse(self, children=None):
        return cde.TakeNode(children[0], self.count)


class ZipDataset(Dataset):
    """
    The result of applying Zip operator to the input Dataset.

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


class ConcatDataset(Dataset):
    """
    The result of applying concat dataset operator to the input Dataset.

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
        # whether the data set is mappable. The second element of pair is length of the dataset
        self._children_flag_and_nums = []

        # _children_start_end_index_: A list of pair<int ,int>.The elements of pair are used to characterize
        # the valid position of the dataset corresponding to the subscript when sampling
        self._children_start_end_index_ = []
        for index, child in enumerate(self.children):
            tem_list = [-1, -1]
            self._children_start_end_index_.append(tem_list)
            dataset_len = self.children_sizes_[index]
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

            if isinstance(child, BatchDataset):
                raise TypeError("The parameter %s of concat must not be BatchDataset!" % child)

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


class RenameDataset(Dataset):
    """
    The result of applying Rename operator to the input Dataset.

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
    if not isinstance(items, list):
        return [items]
    return items


class ProjectDataset(Dataset):
    """
    The result of applying Project operator to the input Dataset.

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
        ir_tree, self.api_tree = dataset.create_ir_tree()

        self._runtime_context = cde.PythonRuntimeContext()
        self._runtime_context.Init()
        self._to_device = cde.ToDevice(num_epochs)
        self._to_device.Init(ir_tree)
        self._runtime_context.AssignConsumer(self._to_device)

        # todo remove next when ContextManager is done
        ITERATORS_LIST.append(weakref.ref(self))
        _unset_iterator_cleanup()

    def send(self):
        self._to_device.Send()

    def stop_send(self):
        self._to_device.StopSend()

    def continue_send(self):
        self._to_device.ContinueSend()

    def get_data_info(self):
        return self._to_device.GetDataInfo()

    def release(self):
        """
        Manually terminate Device Queue instead of relying on out of scope destruction.
        """
        logger.info("Terminating Device Queue. This will also terminate C++ pipeline.")
        if hasattr(self, '_runtime_context') and self._runtime_context:
            if hasattr(self, '_to_device') and self._to_device:
                self._runtime_context.Terminate()
                del self._to_device
            del self._runtime_context

    def __deepcopy__(self, memodict):
        return self


class TransferDataset(Dataset):
    """
    The result of applying TDT operator to the input Dataset.

    Args:
        input_dataset (Dataset): Input Dataset to be transferred.
        send_epoch_end (bool, optional): Whether to send end of sequence to device or not (default=True).
        create_data_info_queue (bool, optional): Whether to create queue which stores
            types and shapes of data or not(default=False).

    Raises:
        TypeError: If device_type is empty.
        ValueError: If device_type is not 'Ascend', 'GPU' or 'CPU'.
        RuntimeError: If dataset is unknown.
    """

    def __init__(self, input_dataset, send_epoch_end=True, create_data_info_queue=False):
        super().__init__(children=input_dataset)
        self.queue_name = str(uuid.uuid1())
        self.device_type = context.get_context("device_target") if context else "CPU"

        self._send_epoch_end = replace_none(send_epoch_end, True)
        self._create_data_info_queue = create_data_info_queue
        self._to_device = None

    def parse(self, children=None):
        total_batch = 0
        device_id = context.get_context("device_id")
        if hasattr(self.children[0], "__total_batch__"):
            total_batch = self.children[0].__total_batch__
        return cde.TransferNode(children[0], self.queue_name, self.device_type, device_id, self._send_epoch_end,
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
        if self._to_device is not None:
            return self._to_device.get_data_info()
        raise RuntimeError("Calling get_data_info with bad state.")

    def release(self):
        """
        Manually terminate Device Queue instead of relying on out of scope destruction.
        """
        if self._to_device is not None:
            self._to_device.release()


class RangeDataset(MappableDataset):
    """
    A source dataset that reads and parses datasets stored on disk in a range.

    Args:
        start (int): Starting index.
        stop (int): Ending index.
        step (int): Step size in the range specified by start and stop.
    """

    def __init__(self, start, stop, step):
        super().__init__()
        self.start = start
        self.stop = stop
        self.step = step

    def parse(self, children=None):
        raise NotImplementedError("Dataset has to implement parse method.")

    def is_shuffled(self):
        return False

    def is_sharded(self):
        return False

    def get_dataset_size(self):
        if self.dataset_size is None:
            self.dataset_size = math.ceil((self.stop - self.start) / self.step)
        return self.dataset_size


class ImageFolderDataset(MappableDataset):
    """
    A source dataset that reads images from a tree of directories.

    All images within one folder have the same label.
    The generated dataset has two columns ['image', 'label'].
    The shape of the image column is [image_size] if decode flag is False, or [H,W,C]
    otherwise.
    The type of the image tensor is uint8. The label is a scalar int32 tensor.
    This dataset can take in a sampler. 'sampler' and 'shuffle' are mutually exclusive. The table
    below shows what input arguments are allowed and their expected behavior.

    .. list-table:: Expected Order Behavior of Using 'sampler' and 'shuffle'
       :widths: 25 25 50
       :header-rows: 1

       * - Parameter 'sampler'
         - Parameter 'shuffle'
         - Expected Order Behavior
       * - None
         - None
         - random order
       * - None
         - True
         - random order
       * - None
         - False
         - sequential order
       * - Sampler object
         - None
         - order defined by sampler
       * - Sampler object
         - True
         - not allowed
       * - Sampler object
         - False
         - not allowed

    Args:
        dataset_dir (str): Path to the root directory that contains the dataset.
        num_samples (int, optional): The number of images to be included in the dataset
            (default=None, all images).
        num_parallel_workers (int, optional): Number of workers to read the data
            (default=None, set in the config).
        shuffle (bool, optional): Whether or not to perform shuffle on the dataset
            (default=None, expected order behavior shown in the table).
        sampler (Sampler, optional): Object used to choose samples from the
            dataset (default=None, expected order behavior shown in the table).
        extensions (list[str], optional): List of file extensions to be
            included in the dataset (default=None).
        class_indexing (dict, optional): A str-to-int mapping from folder name to index
            (default=None, the folder names will be sorted
            alphabetically and each class will be given a
            unique index starting from 0).
        decode (bool, optional): Decode the images after reading (default=False).
        num_shards (int, optional): Number of shards that the dataset will be divided
            into (default=None). When this argument is specified, 'num_samples' reflects
            the max sample number of per shard.
        shard_id (int, optional): The shard ID within num_shards (default=None). This
            argument can only be specified when num_shards is also specified.
        cache (DatasetCache, optional): Use tensor caching service to speed up dataset processing.
            (default=None, which means no cache is used).

    Raises:
        RuntimeError: If sampler and shuffle are specified at the same time.
        RuntimeError: If sampler and sharding are specified at the same time.
        RuntimeError: If num_shards is specified but shard_id is None.
        RuntimeError: If shard_id is specified but num_shards is None.
        RuntimeError: If class_indexing is not a dictionary.
        ValueError: If shard_id is invalid (< 0 or >= num_shards).

    Examples:
        >>> image_folder_dataset_dir = "/path/to/image_folder_dataset_directory"
        >>>
        >>> # 1) Read all samples (image files) in image_folder_dataset_dir with 8 threads
        >>> dataset = ds.ImageFolderDataset(dataset_dir=image_folder_dataset_dir,
        ...                                 num_parallel_workers=8)
        >>>
        >>> # 2) Read all samples (image files) from folder cat and folder dog with label 0 and 1
        >>> dataset = ds.ImageFolderDataset(dataset_dir=image_folder_dataset_dir,
        ...                                 class_indexing={"cat":0, "dog":1})
        >>>
        >>> # 3) Read all samples (image files) in image_folder_dataset_dir with extensions .JPEG and .png (case sensitive)
        >>> dataset = ds.ImageFolderDataset(dataset_dir=image_folder_dataset_dir,
        ...                                 extensions=[".JPEG", ".png"])
    """

    @check_imagefolderdataset
    def __init__(self, dataset_dir, num_samples=None, num_parallel_workers=None, shuffle=None, sampler=None,
                 extensions=None, class_indexing=None, decode=False, num_shards=None, shard_id=None, cache=None):
        super().__init__(num_parallel_workers=num_parallel_workers, sampler=sampler, num_samples=num_samples,
                         shuffle=shuffle, num_shards=num_shards, shard_id=shard_id, cache=cache)

        self.dataset_dir = dataset_dir
        self.extensions = replace_none(extensions, [])
        self.class_indexing = replace_none(class_indexing, {})
        self.decode = replace_none(decode, False)

    def parse(self, children=None):
        return cde.ImageFolderNode(self.dataset_dir, self.decode, self.sampler, self.extensions, self.class_indexing)


class MnistDataset(MappableDataset):
    """
    A source dataset for reading and parsing the MNIST dataset.

    The generated dataset has two columns ['image', 'label'].
    The type of the image tensor is uint8. The label is a scalar uint32 tensor.
    This dataset can take in a sampler. `sampler` and `shuffle` are mutually exclusive. The table
    below shows what input arguments are allowed and their expected behavior.

    .. list-table:: Expected Order Behavior of Using 'sampler' and 'shuffle'
       :widths: 25 25 50
       :header-rows: 1

       * - Parameter `sampler`
         - Parameter `shuffle`
         - Expected Order Behavior
       * - None
         - None
         - random order
       * - None
         - True
         - random order
       * - None
         - False
         - sequential order
       * - Sampler object
         - None
         - order defined by sampler
       * - Sampler object
         - True
         - not allowed
       * - Sampler object
         - False
         - not allowed

    Citation of Mnist dataset.

    .. code-block::

        @article{lecun2010mnist,
        title        = {MNIST handwritten digit database},
        author       = {LeCun, Yann and Cortes, Corinna and Burges, CJ},
        journal      = {ATT Labs [Online]},
        volume       = {2},
        year         = {2010},
        howpublished = {http://yann.lecun.com/exdb/mnist},
        description  = {The MNIST database of handwritten digits has a training set of 60,000 examples,
                        and a test set of 10,000 examples. It is a subset of a larger set available from
                        NIST. The digits have been size-normalized and centered in a fixed-size image.}
        }

    Args:
        dataset_dir (str): Path to the root directory that contains the dataset.
        usage (str, optional): Usage of this dataset, can be "train", "test" or "all" . "train" will read from 60,000
            train samples, "test" will read from 10,000 test samples, "all" will read from all 70,000 samples.
            (default=None, will read all samples)
        num_samples (int, optional): The number of images to be included in the dataset
            (default=None, will read all images).
        num_parallel_workers (int, optional): Number of workers to read the data
            (default=None, will use value set in the config).
        shuffle (bool, optional): Whether or not to perform shuffle on the dataset
            (default=None, expected order behavior shown in the table).
        sampler (Sampler, optional): Object used to choose samples from the
            dataset (default=None, expected order behavior shown in the table).
        num_shards (int, optional): Number of shards that the dataset will be divided into (default=None).
            When this argument is specified, `num_samples` reflects the max sample number of per shard.
        shard_id (int, optional): The shard ID within `num_shards` (default=None). This
            argument can only be specified when `num_shards` is also specified.
        cache (DatasetCache, optional): Use tensor caching service to speed up dataset processing.
            (default=None, which means no cache is used).

    Raises:
        RuntimeError: If sampler and shuffle are specified at the same time.
        RuntimeError: If sampler and sharding are specified at the same time.
        RuntimeError: If num_shards is specified but shard_id is None.
        RuntimeError: If shard_id is specified but num_shards is None.
        ValueError: If shard_id is invalid (< 0 or >= num_shards).

    Examples:
        >>> mnist_dataset_dir = "/path/to/mnist_dataset_directory"
        >>>
        >>> # Read 3 samples from MNIST dataset
        >>> dataset = ds.MnistDataset(dataset_dir=mnist_dataset_dir, num_samples=3)
        >>>
        >>> # Note: In mnist_dataset dataset, each dictionary has keys "image" and "label"
    """

    @check_mnist_cifar_dataset
    def __init__(self, dataset_dir, usage=None, num_samples=None, num_parallel_workers=None, shuffle=None, sampler=None,
                 num_shards=None, shard_id=None, cache=None):
        super().__init__(num_parallel_workers=num_parallel_workers, sampler=sampler, num_samples=num_samples,
                         shuffle=shuffle, num_shards=num_shards, shard_id=shard_id, cache=cache)

        self.dataset_dir = dataset_dir
        self.usage = replace_none(usage, "all")

    def parse(self, children=None):
        return cde.MnistNode(self.dataset_dir, self.usage, self.sampler)


class MindDataset(MappableDataset):
    """
    A source dataset for reading and parsing MindRecord dataset.

    Args:
        dataset_file (Union[str, list[str]]): If dataset_file is a str, it represents for
            a file name of one component of a mindrecord source, other files with identical source
            in the same path will be found and loaded automatically. If dataset_file is a list,
            it represents for a list of dataset files to be read directly.
        columns_list (list[str], optional): List of columns to be read (default=None).
        num_parallel_workers (int, optional): The number of readers (default=None).
        shuffle (bool, optional): Whether or not to perform shuffle on the dataset
            (default=None, performs shuffle).
        num_shards (int, optional): Number of shards that the dataset will be divided into (default=None).
            When this argument is specified, 'num_samples' reflects the max sample number of per shard.
        shard_id (int, optional): The shard ID within num_shards (default=None). This
            argument can only be specified when num_shards is also specified.
        sampler (Sampler, optional): Object used to choose samples from the
            dataset (default=None, sampler is exclusive
            with shuffle and block_reader). Support list: SubsetRandomSampler,
            PkSampler, RandomSampler, SequentialSampler, DistributedSampler.
        padded_sample (dict, optional): Samples will be appended to dataset, where
            keys are the same as column_list.
        num_padded (int, optional): Number of padding samples. Dataset size
            plus num_padded should be divisible by num_shards.
        num_samples (int, optional): The number of samples to be included in the dataset
            (default=None, all samples).

    Raises:
        ValueError: If num_shards is specified but shard_id is None.
        ValueError: If shard_id is specified but num_shards is None.

    Examples:
        >>> mind_dataset_dir = ["/path/to/mind_dataset_file"] # contains 1 or multiple MindRecord files
        >>> dataset = ds.MindDataset(dataset_file=mind_dataset_dir)
    """

    def parse(self, children=None):
        return cde.MindDataNode(self.dataset_file, self.columns_list, self.sampler, self.new_padded_sample,
                                self.num_padded)

    @check_minddataset
    def __init__(self, dataset_file, columns_list=None, num_parallel_workers=None, shuffle=None, num_shards=None,
                 shard_id=None, sampler=None, padded_sample=None, num_padded=None, num_samples=None):
        super().__init__(num_parallel_workers=num_parallel_workers, sampler=sampler, num_samples=num_samples,
                         shuffle=shuffle, num_shards=num_shards, shard_id=shard_id)
        if isinstance(dataset_file, list):
            self.load_dataset = False
        else:
            self.load_dataset = True
        self.dataset_file = dataset_file
        self.columns_list = replace_none(columns_list, [])
        self.shuffle_option = shuffle

        if shuffle is False:
            logger.warning("WARN: global shuffle is not used.")

        if sampler is not None:
            if isinstance(sampler, (
                    samplers.SubsetRandomSampler, samplers.SubsetSampler, samplers.PKSampler,
                    samplers.DistributedSampler,
                    samplers.RandomSampler, samplers.SequentialSampler)) is False:
                raise ValueError("The sampler is not supported yet.")

        self.padded_sample = padded_sample
        self.num_padded = replace_none(num_padded, 0)

        self.new_padded_sample = {}
        if padded_sample:
            for k, v in padded_sample.items():
                if isinstance(v, np.ndarray):
                    self.new_padded_sample[k] = v.tobytes()
                else:
                    self.new_padded_sample[k] = v


def _iter_fn(dataset, num_samples):
    """
    Generator function wrapper for iterable dataset.
    """
    if num_samples is not None and num_samples != 0:
        ds_iter = iter(dataset)
        for _ in range(num_samples):
            try:
                val = next(ds_iter)
            except StopIteration:
                return
            # convert output tensors to ndarrays
            yield tuple([np.array(x, copy=False) for x in val])
    else:
        for val in dataset:
            # convert output tensors to ndarrays
            yield tuple([np.array(x, copy=False) for x in val])


def _generator_fn(generator, num_samples):
    """
    Generator function wrapper for generator function dataset.
    """
    if num_samples is not None and num_samples != 0:
        gen_iter = generator()
        for _ in range(num_samples):
            try:
                val = next(gen_iter)
            except StopIteration:
                return
            yield val
    else:
        gen_iter = generator()
        for val in gen_iter:
            yield val


def _cpp_sampler_fn(sample_ids, dataset):
    """
    Generator function wrapper for mappable dataset with cpp sampler.
    """
    if not isinstance(sample_ids, np.ndarray):
        raise RuntimeError("Sample IDs are not in a numpy array.")
    if sample_ids.size == 0:
        raise RuntimeError("Sampler passed an empty sample IDs list.")

    for i in sample_ids:
        val = dataset[i]
        # convert output tensors to ndarrays
        yield tuple([np.array(x, copy=False) for x in val])


def _cpp_sampler_fn_mp(sample_ids, sample_fn):
    """
    Multiprocessing generator function wrapper for mappable dataset with cpp sampler.
    """
    if not isinstance(sample_ids, np.ndarray):
        raise RuntimeError("Sample IDs are not in a numpy array.")
    if sample_ids.size == 0:
        raise RuntimeError("Sampler passed an empty sample IDs list.")

    return sample_fn.process(sample_ids)


def _fill_worker_indices(workers, indices, idx):
    """
    Worker index queue filler, fill worker index queue in round robin order.
    """
    num_worker = len(workers)
    while idx < len(indices):
        try:
            workers[idx % num_worker].put(indices[idx])
            idx += 1
        except queue.Full:
            break
    return idx


class SamplerFn:
    """
    Multiprocessing or multithread generator function wrapper master process.
    """

    def __init__(self, dataset, num_worker, multi_process):
        self.workers = []
        self.num_worker = num_worker
        self.multi_process = multi_process
        self.need_join = False
        self.ppid = os.getpid()
        self.pid = []
        # Event for end of epoch
        if multi_process is True:
            self.eof = multiprocessing.Event()
        else:
            self.eof = threading.Event()
        # Create workers
        for _ in range(num_worker):
            if multi_process is True:
                worker = _GeneratorWorkerMp(dataset, self.eof)
                worker.daemon = True
                # When multi processes fork a subprocess, the lock of the main process is copied to the subprocess,
                # which may cause deadlock. Therefore, the subprocess startup is performed in che initialization phase.
                # In this phase, the main process is not locked.
                worker.start()
                self.pid.append(worker.pid)
                self.need_join = True
            else:
                worker = _GeneratorWorkerMt(dataset, self.eof)
                worker.daemon = True
            self.workers.append(worker)

    def process(self, indices):
        """
        The main process, start the child process or child thread, and fill the index queue.
        Get the result and return.
        """
        for w in self.workers:
            # Check whether the queue of the subprocess is empty.
            if not w.queue_empty():
                raise Exception("The queue of the subprocess is not empty.")
            # Start all workers
            if not w.is_alive():
                w.start()

        # Fill initial index queues
        idx_cursor = 0
        idx_cursor = _fill_worker_indices(self.workers, indices, idx_cursor)

        # Fetch results
        for i in range(len(indices)):
            if self.eof.is_set():
                self._stop_subprocess()
                return
            # Fetch result and put index
            try:
                result = self.workers[i % self.num_worker].get()
            except queue.Empty:
                self._stop_subprocess()
                raise Exception("Generator worker process timeout.")
            except KeyboardInterrupt:
                self._stop_subprocess()
                raise Exception("Generator worker receives KeyboardInterrupt.")
            if self.eof.is_set():
                self._stop_subprocess()
                return
            if idx_cursor < len(indices):
                idx_cursor = _fill_worker_indices(self.workers, indices, idx_cursor)
            yield tuple([np.array(x, copy=False) for x in result])

    def _stop_subprocess(self):
        # Only the main process can call join
        if self.need_join is True and self.ppid == os.getpid():
            self.eof.set()
            self.need_join = False
            for w in self.workers:
                w.join()

    def __del__(self):
        self._stop_subprocess()


def _subprocess_handle(eof, signum, frame):
    eof.set()


def _generator_worker_loop(dataset, idx_queue, result_queue, eof, is_multiprocessing):
    """
    Multithread or multiprocess generator worker process loop.
    """
    if is_multiprocessing:
        signal.signal(signal.SIGTERM, partial(_subprocess_handle, eof))
    while True:
        # Fetch index, block
        try:
            idx = idx_queue.get(timeout=1)
        except KeyboardInterrupt:
            if is_multiprocessing:
                eof.set()
                idx_queue.cancel_join_thread()
                result_queue.cancel_join_thread()
            raise Exception("Generator worker receives KeyboardInterrupt.")
        except queue.Empty:
            if eof.is_set():
                if is_multiprocessing:
                    idx_queue.cancel_join_thread()
                    result_queue.cancel_join_thread()
                return
            # If end-of-file (eof) is not set, continue to get data from idx_queue
            continue
        if idx is None:
            # When the queue is out of scope from master process, a None item can be fetched from the queue.
            # Upon receiving None, worker process should check if eof is set.
            assert eof.is_set(), ""
            return
        if eof.is_set():
            if is_multiprocessing:
                idx_queue.cancel_join_thread()
                result_queue.cancel_join_thread()
            return
        # Fetch data, any exception from __getitem__ will terminate worker and timeout master process
        result = dataset[idx]
        # Send data, block
        while True:
            try:
                result_queue.put(result, timeout=5)
            except KeyboardInterrupt:
                if is_multiprocessing:
                    eof.set()
                    idx_queue.cancel_join_thread()
                    result_queue.cancel_join_thread()
                raise Exception("Generator worker receives KeyboardInterrupt.")
            except queue.Full:
                if eof.is_set():
                    if is_multiprocessing:
                        idx_queue.cancel_join_thread()
                        result_queue.cancel_join_thread()
                    return
                # If eof is not set, continue to put data to result_queue
                continue
            break
        del result, idx


class _GeneratorWorkerMt(threading.Thread):
    """
    Worker process for multi-thread Generator.
    """

    def __init__(self, dataset, eof):
        self.idx_queue = queue.Queue(16)
        self.res_queue = queue.Queue(16)
        super().__init__(target=_generator_worker_loop, args=(dataset, self.idx_queue, self.res_queue, eof, False))

    def put(self, item):
        """
        Put function for worker index queue. Never block. Raise queue.Full on failure.
        """
        self.idx_queue.put_nowait(item)

    def get(self):
        """
        Get function for worker result queue. Block with timeout.
        """
        return self.res_queue.get(timeout=30)

    def queue_empty(self):
        if not self.idx_queue.empty():
            logger.warning("idx_queue is not empty")
            return False
        if not self.res_queue.empty():
            logger.warning("res_queue is not empty")
            return False
        return True


class _GeneratorWorkerMp(multiprocessing.Process):
    """
    Worker process for multiprocess Generator.
    """

    def __init__(self, dataset, eof):
        self.idx_queue = multiprocessing.Queue(16)
        self.res_queue = multiprocessing.Queue(16)
        super().__init__(target=_generator_worker_loop, args=(dataset, self.idx_queue, self.res_queue, eof, True))

    def put(self, item):
        """
        Put function for worker index queue. Never block. Raise queue.Full on failure.
        """
        self.idx_queue.put_nowait(item)

    def get(self):
        """
        Get function for worker result queue. Block with timeout.
        """
        # Relax 10s to 30s, since it sometimes will cause "Generator worker process timeout"
        # when we run too many iterators with infinite epoch(num_epoch=-1)
        return self.res_queue.get(timeout=30)

    def queue_empty(self):
        if not self.idx_queue.empty():
            logger.warning("idx_queue is not empty.")
            return False
        if not self.res_queue.empty():
            logger.warning("res_queue is not empty.")
            return False
        return True


class GeneratorDataset(MappableDataset):
    """
    A source dataset that generates data from Python by invoking Python data source each epoch.

    This dataset can take in a sampler. 'sampler' and 'shuffle' are mutually exclusive. The table
    below shows what input arguments are allowed and their expected behavior.

    .. list-table:: Expected Order Behavior of Using 'sampler' and 'shuffle'
       :widths: 25 25 50
       :header-rows: 1

       * - Parameter 'sampler'
         - Parameter 'shuffle'
         - Expected Order Behavior
       * - None
         - None
         - random order
       * - None
         - True
         - random order
       * - None
         - False
         - sequential order
       * - Sampler object
         - None
         - order defined by sampler
       * - Sampler object
         - True
         - not allowed
       * - Sampler object
         - False
         - not allowed

    Args:
        source (Union[Callable, Iterable, Random Accessible]):
            A generator callable object, an iterable Python object or a random accessible Python object.
            Callable source is required to return a tuple of NumPy arrays as a row of the dataset on source().next().
            Iterable source is required to return a tuple of NumPy arrays as a row of the dataset on
            iter(source).next().
            Random accessible source is required to return a tuple of NumPy arrays as a row of the dataset on
            source[idx].
        column_names (Union[str, list[str]], optional): List of column names of the dataset (default=None). Users are
            required to provide either column_names or schema.
        column_types (list[mindspore.dtype], optional): List of column data types of the dataset (default=None).
            If provided, sanity check will be performed on generator output.
        schema (Union[Schema, str], optional): Path to the JSON schema file or schema object (default=None). Users are
            required to provide either column_names or schema. If both are provided, schema will be used.
        num_samples (int, optional): The number of samples to be included in the dataset
            (default=None, all images).
        num_parallel_workers (int, optional): Number of subprocesses used to fetch the dataset in parallel (default=1).
        shuffle (bool, optional): Whether or not to perform shuffle on the dataset. Random accessible input is required.
            (default=None, expected order behavior shown in the table).
        sampler (Union[Sampler, Iterable], optional): Object used to choose samples from the dataset. Random accessible
            input is required (default=None, expected order behavior shown in the table).
        num_shards (int, optional): Number of shards that the dataset will be divided into (default=None).
            Random accessible input is required. When this argument is specified, 'num_samples' reflects the max sample
            number of per shard.
        shard_id (int, optional): The shard ID within num_shards (default=None). This argument must be specified only
            when num_shards is also specified. Random accessible input is required.
        python_multiprocessing (bool, optional): Parallelize Python operations with multiple worker process. This
            option could be beneficial if the Python operation is computational heavy (default=True).

    Examples:
        >>> import numpy as np
        >>>
        >>> # 1) Multidimensional generator function as callable input.
        >>> def generator_multidimensional():
        ...     for i in range(64):
        ...         yield (np.array([[i, i + 1], [i + 2, i + 3]]),)
        >>>
        >>> dataset = ds.GeneratorDataset(source=generator_multidimensional, column_names=["multi_dimensional_data"])
        >>>
        >>> # 2) Multi-column generator function as callable input.
        >>> def generator_multi_column():
        ...     for i in range(64):
        ...         yield np.array([i]), np.array([[i, i + 1], [i + 2, i + 3]])
        >>>
        >>> dataset = ds.GeneratorDataset(source=generator_multi_column, column_names=["col1", "col2"])
        >>>
        >>> # 3) Iterable dataset as iterable input.
        >>> class MyIterable:
        ...     def __init__(self):
        ...         self._index = 0
        ...         self._data = np.random.sample((5, 2))
        ...         self._label = np.random.sample((5, 1))
        ...
        ...     def __next__(self):
        ...         if self._index >= len(self._data):
        ...             raise StopIteration
        ...         else:
        ...             item = (self._data[self._index], self._label[self._index])
        ...             self._index += 1
        ...             return item
        ...
        ...     def __iter__(self):
        ...         self._index = 0
        ...         return self
        ...
        ...     def __len__(self):
        ...         return len(self._data)
        >>>
        >>> dataset = ds.GeneratorDataset(source=MyIterable(), column_names=["data", "label"])
        >>>
        >>> # 4) Random accessible dataset as random accessible input.
        >>> class MyAccessible:
        ...     def __init__(self):
        ...         self._data = np.random.sample((5, 2))
        ...         self._label = np.random.sample((5, 1))
        ...
        ...     def __getitem__(self, index):
        ...         return self._data[index], self._label[index]
        ...
        ...     def __len__(self):
        ...         return len(self._data)
        >>>
        >>> dataset = ds.GeneratorDataset(source=MyAccessible(), column_names=["data", "label"])
        >>>
        >>> # list, dict, tuple of Python is also random accessible
        >>> dataset = ds.GeneratorDataset(source=[(np.array(0),), (np.array(1),), (np.array(2),)], column_names=["col"])
    """

    @check_generatordataset
    def __init__(self, source, column_names=None, column_types=None, schema=None, num_samples=None,
                 num_parallel_workers=1, shuffle=None, sampler=None, num_shards=None, shard_id=None,
                 python_multiprocessing=True):
        super().__init__(num_parallel_workers=num_parallel_workers, sampler=sampler, num_samples=num_samples,
                         shuffle=shuffle, num_shards=num_shards, shard_id=shard_id)
        self.source = source
        self.prepared_source = None  # source to be sent to C++

        self.python_multiprocessing = python_multiprocessing

        self.column_names = to_list(column_names)

        if column_types is not None:
            self.column_types = mstypelist_to_detypelist(column_types)
        else:
            self.column_types = []

        self.schema = schema
        if schema is not None:
            self.schema = schema
            if not isinstance(schema, Schema):
                self.schema = Schema(schema)
        # Move get dataset_size by len from parse to here, because self.source will
        # lose attribution of '__len__' after deepcopy.
        self.source_len = -1  # unknown
        if hasattr(self.source, "__len__"):
            self.source_len = len(self.source)

    def __deepcopy__(self, memodict):
        if id(self) in memodict:
            return memodict[id(self)]
        new_op = self.__safe_deepcopy__(memodict, exclude=("source", "__transfer_dataset__"))

        sample_fn = None
        if new_op.sampler is not None and hasattr(self.source, "__getitem__"):
            if new_op.num_parallel_workers > 1:
                sample_fn = SamplerFn(self.source, new_op.num_parallel_workers, self.python_multiprocessing)
                new_op.prepared_source = (lambda sample_ids: _cpp_sampler_fn_mp(sample_ids, sample_fn))
            else:
                new_op.prepared_source = (lambda sample_ids: _cpp_sampler_fn(sample_ids, self.source))
            new_op.sample_fn = sample_fn
        else:
            try:
                new_op.sampler = None
                new_op.sample_fn = sample_fn
                new_op.source_len = min(new_op.source_len,
                                        new_op.num_samples) if new_op.num_samples != 0 else new_op.source_len
                iter(self.source)
            except TypeError:
                # Use generator function if input callable
                new_op.prepared_source = (lambda: _generator_fn(self.source, new_op.num_samples))
            else:
                # Use iterator function if input is iterable
                # Random accessible input is also iterable
                new_op.prepared_source = (lambda: _iter_fn(self.source, new_op.num_samples))

        return new_op

    def is_shuffled(self):
        return self.sampler.is_shuffled()

    def is_sharded(self):
        return self.sampler.is_sharded()

    def parse(self, children=None):
        if self.schema is None:
            return cde.GeneratorNode(self.prepared_source, self.column_names, self.column_types, self.source_len,
                                     self.sampler)
        schema = self.schema
        if isinstance(schema, Schema):
            schema = self.schema.cpp_schema
        return cde.GeneratorNode(self.prepared_source, schema, self.source_len, self.sampler)


class TFRecordDataset(SourceDataset):
    """
    A source dataset for reading and parsing datasets stored on disk in TFData format.

    Args:
        dataset_files (Union[str, list[str]]): String or list of files to be read or glob strings to search for a
            pattern of files. The list will be sorted in a lexicographical order.
        schema (Union[str, Schema], optional): Path to the JSON schema file or schema object (default=None).
            If the schema is not provided, the meta data from the TFData file is considered the schema.
        columns_list (list[str], optional): List of columns to be read (default=None, read all columns)
        num_samples (int, optional): Number of samples (rows) to read (default=None).
            If num_samples is None and numRows(parsed from schema) does not exist, read the full dataset;
            If num_samples is None and numRows(parsed from schema) is greater than 0, read numRows rows;
            If both num_samples and numRows(parsed from schema) are greater than 0, read num_samples rows.
        num_parallel_workers (int, optional): Number of workers to read the data
            (default=None, number set in the config).
        shuffle (Union[bool, Shuffle level], optional): Perform reshuffling of the data every epoch
            (default=Shuffle.GLOBAL).
            If shuffle is False, no shuffling will be performed;
            If shuffle is True, the behavior is the same as setting shuffle to be Shuffle.GLOBAL
            Otherwise, there are two levels of shuffling:

            - Shuffle.GLOBAL: Shuffle both the files and samples.

            - Shuffle.FILES: Shuffle files only.

        num_shards (int, optional): Number of shards that the dataset will be divided
            into (default=None). When this argument is specified, 'num_samples' reflects
            the max sample number of per shard.
        shard_id (int, optional): The shard ID within num_shards (default=None). This
            argument can only be specified when num_shards is also specified.
        shard_equal_rows (bool, optional): Get equal rows for all shards(default=False). If shard_equal_rows
            is false, number of rows of each shard may be not equal. This
            argument should only be specified when num_shards is also specified.
        cache (DatasetCache, optional): Use tensor caching service to speed up dataset processing.
            (default=None, which means no cache is used).

    Examples:
        >>> import mindspore.common.dtype as mstype
        >>>
        >>> tfrecord_dataset_dir = ["/path/to/tfrecord_dataset_file"] # contains 1 or multiple TFRecord files
        >>> tfrecord_schema_file = "/path/to/tfrecord_schema_file"
        >>>
        >>> # 1) Get all rows from tfrecord_dataset_dir with no explicit schema.
        >>> # The meta-data in the first row will be used as a schema.
        >>> dataset = ds.TFRecordDataset(dataset_files=tfrecord_dataset_dir)
        >>>
        >>> # 2) Get all rows from tfrecord_dataset_dir with user-defined schema.
        >>> schema = ds.Schema()
        >>> schema.add_column(name='col_1d', de_type=mstype.int64, shape=[2])
        >>> dataset = ds.TFRecordDataset(dataset_files=tfrecord_dataset_dir, schema=schema)
        >>>
        >>> # 3) Get all rows from tfrecord_dataset_dir with schema file.
        >>> dataset = ds.TFRecordDataset(dataset_files=tfrecord_dataset_dir, schema=tfrecord_schema_file)
    """

    @check_tfrecorddataset
    def __init__(self, dataset_files, schema=None, columns_list=None, num_samples=None, num_parallel_workers=None,
                 shuffle=Shuffle.GLOBAL, num_shards=None, shard_id=None, shard_equal_rows=False, cache=None):
        super().__init__(num_parallel_workers=num_parallel_workers, num_samples=num_samples, shuffle=shuffle,
                         num_shards=num_shards, shard_id=shard_id, cache=cache)
        # todo push down to c++
        self.dataset_files = self._find_files(dataset_files)
        self.dataset_files.sort()

        self.schema = schema
        self.columns_list = replace_none(columns_list, [])
        self.shard_equal_rows = replace_none(shard_equal_rows, False)

        if self.schema is not None and (self.num_samples is None or self.num_samples == 0):
            self.num_samples = Schema.get_num_rows(self.schema)

    def parse(self, children=None):
        schema = self.schema.cpp_schema if isinstance(self.schema, Schema) else self.schema
        return cde.TFRecordNode(self.dataset_files, schema, self.columns_list, self.num_samples, self.shuffle_flag,
                                self.num_shards, self.shard_id, self.shard_equal_rows)


class ManifestDataset(MappableDataset):
    """
    A source dataset for reading images from a Manifest file.

    The generated dataset has two columns ['image', 'label'].
    The shape of the image column is [image_size] if decode flag is False, or [H,W,C]
    otherwise.
    The type of the image tensor is uint8. The label is a scalar uint64 tensor.
    This dataset can take in a sampler. `sampler` and `shuffle` are mutually exclusive. The table
    below shows what input arguments are allowed and their expected behavior.

    .. list-table:: Expected Order Behavior of Using `sampler` and `shuffle`
       :widths: 25 25 50
       :header-rows: 1

       * - Parameter `sampler`
         - Parameter `shuffle`
         - Expected Order Behavior
       * - None
         - None
         - random order
       * - None
         - True
         - random order
       * - None
         - False
         - sequential order
       * - Sampler object
         - None
         - order defined by sampler
       * - Sampler object
         - True
         - not allowed
       * - Sampler object
         - False
         - not allowed

    Args:
        dataset_file (str): File to be read.
        usage (str, optional): Acceptable usages include "train", "eval" and "inference" (default="train").
        num_samples (int, optional): The number of images to be included in the dataset.
            (default=None, will include all images).
        num_parallel_workers (int, optional): Number of workers to read the data
            (default=None, will use value set in the config).
        shuffle (bool, optional): Whether to perform shuffle on the dataset (default=None, expected
            order behavior shown in the table).
        sampler (Sampler, optional): Object used to choose samples from the
            dataset (default=None, expected order behavior shown in the table).
        class_indexing (dict, optional): A str-to-int mapping from label name to index
            (default=None, the folder names will be sorted alphabetically and each
            class will be given a unique index starting from 0).
        decode (bool, optional): decode the images after reading (default=False).
        num_shards (int, optional): Number of shards that the dataset will be divided
            into (default=None). When this argument is specified, `num_samples` reflects
            the max number of samples per shard.
        shard_id (int, optional): The shard ID within `num_shards` (default=None). This
            argument can only be specified when `num_shards` is also specified.
        cache (DatasetCache, optional): Use tensor caching service to speed up dataset processing.
            (default=None, which means no cache is used).

    Raises:
        RuntimeError: If sampler and shuffle are specified at the same time.
        RuntimeError: If sampler and sharding are specified at the same time.
        RuntimeError: If num_shards is specified but shard_id is None.
        RuntimeError: If shard_id is specified but num_shards is None.
        RuntimeError: If class_indexing is not a dictionary.
        ValueError: If shard_id is invalid (< 0 or >= num_shards).

    Examples:
        >>> manifest_dataset_dir = "/path/to/manifest_dataset_file"
        >>>
        >>> # 1) Read all samples specified in manifest_dataset_dir dataset with 8 threads for training
        >>> dataset = ds.ManifestDataset(dataset_file=manifest_dataset_dir, usage="train", num_parallel_workers=8)
        >>>
        >>> # 2) Read samples (specified in manifest_file.manifest) for shard 0 in a 2-way distributed training setup
        >>> dataset = ds.ManifestDataset(dataset_file=manifest_dataset_dir, num_shards=2, shard_id=0)
    """

    @check_manifestdataset
    def __init__(self, dataset_file, usage="train", num_samples=None, num_parallel_workers=None, shuffle=None,
                 sampler=None, class_indexing=None, decode=False, num_shards=None, shard_id=None, cache=None):
        super().__init__(num_parallel_workers=num_parallel_workers, sampler=sampler, num_samples=num_samples,
                         shuffle=shuffle, num_shards=num_shards, shard_id=shard_id, cache=cache)

        self.dataset_file = dataset_file
        self.decode = replace_none(decode, False)
        self.usage = replace_none(usage, "train")
        self.class_indexing = replace_none(class_indexing, {})

    def parse(self, children=None):
        return cde.ManifestNode(self.dataset_file, self.usage, self.sampler, self.class_indexing, self.decode)

    def get_class_indexing(self):
        """
        Get the class index.

        Returns:
            dict, a str-to-int mapping from label name to index.
        """
        if self.class_indexing is None or not self.class_indexing:
            if self._class_indexing is None:
                runtime_getter = self._init_tree_getters()
                self._class_indexing = runtime_getter[0].GetClassIndexing()
            self.class_indexing = {}
            for pair in self._class_indexing:
                self.class_indexing[pair[0]] = pair[1][0]
        return self.class_indexing


class Cifar10Dataset(MappableDataset):
    """
    A source dataset for reading and parsing Cifar10 dataset.

    The generated dataset has two columns ['image', 'label'].
    The type of the image tensor is uint8. The label is a scalar uint32 tensor.
    This dataset can take in a sampler. 'sampler' and 'shuffle' are mutually exclusive. The table
    below shows what input arguments are allowed and their expected behavior.

    .. list-table:: Expected Order Behavior of Using 'sampler' and 'shuffle'
       :widths: 25 25 50
       :header-rows: 1

       * - Parameter 'sampler'
         - Parameter 'shuffle'
         - Expected Order Behavior
       * - None
         - None
         - random order
       * - None
         - True
         - random order
       * - None
         - False
         - sequential order
       * - Sampler object
         - None
         - order defined by sampler
       * - Sampler object
         - True
         - not allowed
       * - Sampler object
         - False
         - not allowed

    Citation of Cifar10 dataset.

    .. code-block::

        @techreport{Krizhevsky09,
        author       = {Alex Krizhevsky},
        title        = {Learning multiple layers of features from tiny images},
        institution  = {},
        year         = {2009},
        howpublished = {http://www.cs.toronto.edu/~kriz/cifar.html},
        description  = {The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes,
                        with 6000 images per class. There are 50000 training images and 10000 test images.}
        }

    Args:
        dataset_dir (str): Path to the root directory that contains the dataset.
        usage (str, optional): Usage of this dataset, can be "train", "test" or "all" . "train" will read from 50,000
            train samples, "test" will read from 10,000 test samples, "all" will read from all 60,000 samples.
            (default=None, all samples)
        num_samples (int, optional): The number of images to be included in the dataset.
            (default=None, all images).
        num_parallel_workers (int, optional): Number of workers to read the data
            (default=None, number set in the config).
        shuffle (bool, optional): Whether to perform shuffle on the dataset (default=None, expected
            order behavior shown in the table).
        sampler (Sampler, optional): Object used to choose samples from the
            dataset (default=None, expected order behavior shown in the table).
        num_shards (int, optional): Number of shards that the dataset will be divided
            into (default=None). When this argument is specified, 'num_samples' reflects
            the max sample number of per shard.
        shard_id (int, optional): The shard ID within num_shards (default=None). This
            argument can only be specified when num_shards is also specified.
        cache (DatasetCache, optional): Use tensor caching service to speed up dataset processing.
            (default=None, which means no cache is used).

    Raises:
        RuntimeError: If sampler and shuffle are specified at the same time.
        RuntimeError: If sampler and sharding are specified at the same time.
        RuntimeError: If num_shards is specified but shard_id is None.
        RuntimeError: If shard_id is specified but num_shards is None.
        ValueError: If shard_id is invalid (< 0 or >= num_shards).

    Examples:
        >>> cifar10_dataset_dir = "/path/to/cifar10_dataset_directory"
        >>>
        >>> # 1) Get all samples from CIFAR10 dataset in sequence
        >>> dataset = ds.Cifar10Dataset(dataset_dir=cifar10_dataset_dir, shuffle=False)
        >>>
        >>> # 2) Randomly select 350 samples from CIFAR10 dataset
        >>> dataset = ds.Cifar10Dataset(dataset_dir=cifar10_dataset_dir, num_samples=350, shuffle=True)
        >>>
        >>> # 3) Get samples from CIFAR10 dataset for shard 0 in a 2-way distributed training
        >>> dataset = ds.Cifar10Dataset(dataset_dir=cifar10_dataset_dir, num_shards=2, shard_id=0)
        >>>
        >>> # In CIFAR10 dataset, each dictionary has keys "image" and "label"
    """

    @check_mnist_cifar_dataset
    def __init__(self, dataset_dir, usage=None, num_samples=None, num_parallel_workers=None, shuffle=None, sampler=None,
                 num_shards=None, shard_id=None, cache=None):
        super().__init__(num_parallel_workers=num_parallel_workers, sampler=sampler, num_samples=num_samples,
                         shuffle=shuffle, num_shards=num_shards, shard_id=shard_id, cache=cache)

        self.dataset_dir = dataset_dir
        self.usage = replace_none(usage, "all")

    def parse(self, children=None):
        return cde.Cifar10Node(self.dataset_dir, self.usage, self.sampler)


class Cifar100Dataset(MappableDataset):
    """
    A source dataset for reading and parsing Cifar100 dataset.

    The generated dataset has three columns ['image', 'coarse_label', 'fine_label'].
    The type of the image tensor is uint8. The coarse and fine labels are each a scalar uint32 tensor.
    This dataset can take in a sampler. 'sampler' and 'shuffle' are mutually exclusive. The table
    below shows what input arguments are allowed and their expected behavior.

    .. list-table:: Expected Order Behavior of Using 'sampler' and 'shuffle'
       :widths: 25 25 50
       :header-rows: 1

       * - Parameter 'sampler'
         - Parameter 'shuffle'
         - Expected Order Behavior
       * - None
         - None
         - random order
       * - None
         - True
         - random order
       * - None
         - False
         - sequential order
       * - Sampler object
         - None
         - order defined by sampler
       * - Sampler object
         - True
         - not allowed
       * - Sampler object
         - False
         - not allowed

    Citation of Cifar100 dataset.

    .. code-block::

        @techreport{Krizhevsky09,
        author       = {Alex Krizhevsky},
        title        = {Learning multiple layers of features from tiny images},
        institution  = {},
        year         = {2009},
        howpublished = {http://www.cs.toronto.edu/~kriz/cifar.html},
        description  = {This dataset is just like the CIFAR-10, except it has 100 classes containing 600 images
                        each. There are 500 training images and 100 testing images per class. The 100 classes in
                        the CIFAR-100 are grouped into 20 superclasses. Each image comes with a "fine" label (the
                        class to which it belongs) and a "coarse" label (the superclass to which it belongs).}
        }

    Args:
        dataset_dir (str): Path to the root directory that contains the dataset.
        usage (str, optional): Usage of this dataset, can be "train", "test" or "all" . "train" will read from 50,000
            train samples, "test" will read from 10,000 test samples, "all" will read from all 60,000 samples.
            (default=None, all samples)
        num_samples (int, optional): The number of images to be included in the dataset.
            (default=None, all images).
        num_parallel_workers (int, optional): Number of workers to read the data
            (default=None, number set in the config).
        shuffle (bool, optional): Whether to perform shuffle on the dataset (default=None, expected
            order behavior shown in the table).
        sampler (Sampler, optional): Object used to choose samples from the
            dataset (default=None, expected order behavior shown in the table).
        num_shards (int, optional): Number of shards that the dataset will be divided
            into (default=None). When this argument is specified, 'num_samples' reflects
            the max sample number of per shard.
        shard_id (int, optional): The shard ID within num_shards (default=None). This
            argument can only be specified when num_shards is also specified.
        cache (DatasetCache, optional): Use tensor caching service to speed up dataset processing.
            (default=None, which means no cache is used).

    Raises:
        RuntimeError: If sampler and shuffle are specified at the same time.
        RuntimeError: If sampler and sharding are specified at the same time.
        RuntimeError: If num_shards is specified but shard_id is None.
        RuntimeError: If shard_id is specified but num_shards is None.
        ValueError: If shard_id is invalid (< 0 or >= num_shards).

    Examples:
        >>> cifar100_dataset_dir = "/path/to/cifar100_dataset_directory"
        >>>
        >>> # 1) Get all samples from CIFAR100 dataset in sequence
        >>> dataset = ds.Cifar100Dataset(dataset_dir=cifar100_dataset_dir, shuffle=False)
        >>>
        >>> # 2) Randomly select 350 samples from CIFAR100 dataset
        >>> dataset = ds.Cifar100Dataset(dataset_dir=cifar100_dataset_dir, num_samples=350, shuffle=True)
        >>>
        >>> # In CIFAR100 dataset, each dictionary has 3 keys: "image", "fine_label" and "coarse_label"
    """

    @check_mnist_cifar_dataset
    def __init__(self, dataset_dir, usage=None, num_samples=None, num_parallel_workers=None, shuffle=None, sampler=None,
                 num_shards=None, shard_id=None, cache=None):
        super().__init__(num_parallel_workers=num_parallel_workers, sampler=sampler, num_samples=num_samples,
                         shuffle=shuffle, num_shards=num_shards, shard_id=shard_id, cache=cache)

        self.dataset_dir = dataset_dir
        self.usage = replace_none(usage, "all")

    def parse(self, children=None):
        return cde.Cifar100Node(self.dataset_dir, self.usage, self.sampler)


class RandomDataset(SourceDataset):
    """
    A source dataset that generates random data.

    Args:
        total_rows (int): Number of rows for the dataset to generate (default=None, number of rows is random)
        schema (Union[str, Schema], optional): Path to the JSON schema file or schema object (default=None).
            If the schema is not provided, the random dataset generates a random schema.
        columns_list (list[str], optional): List of columns to be read (default=None, read all columns)
        num_samples (int): number of samples to draw from the total. (default=None, which means all rows)
        num_parallel_workers (int, optional): Number of workers to read the data
            (default=None, number set in the config).
        cache (DatasetCache, optional): Use tensor caching service to speed up dataset processing.
            (default=None, which means no cache is used).
        shuffle (bool, optional): Whether or not to perform shuffle on the dataset
            (default=None, expected order behavior shown in the table).
        num_shards (int, optional): Number of shards that the dataset will be divided
            into (default=None). When this argument is specified, 'num_samples' reflects
            the max sample number of per shard.
        shard_id (int, optional): The shard ID within num_shards (default=None). This
            argument can only be specified when num_shards is also specified.
    """

    @check_random_dataset
    def __init__(self, total_rows=None, schema=None, columns_list=None, num_samples=None, num_parallel_workers=None,
                 cache=None, shuffle=None, num_shards=None, shard_id=None):
        super().__init__(num_parallel_workers=num_parallel_workers, num_samples=num_samples, shuffle=shuffle,
                         num_shards=num_shards, shard_id=shard_id, cache=cache)
        self.total_rows = total_rows
        if schema is not None:
            self.total_rows = replace_none(total_rows, Schema.get_num_rows(schema))
        self.schema = schema
        self.columns_list = replace_none(columns_list, [])

    def parse(self, children=None):
        schema = self.schema.cpp_schema if isinstance(self.schema, Schema) else self.schema
        return cde.RandomNode(self.total_rows, schema, self.columns_list)


class Schema:
    """
    Class to represent a schema of a dataset.

    Args:
        schema_file(str): Path of schema file (default=None).

    Returns:
        Schema object, schema info about dataset.

    Raises:
        RuntimeError: If schema file failed to load.

    Example:
        >>> import mindspore.common.dtype as mstype
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
            name (str): Name of the column.
            de_type (str): Data type of the column.
            shape (list[int], optional): Shape of the column
                (default=None, [-1] which is an unknown shape of rank 1).

        Raises:
            ValueError: If column type is unknown.
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

                - list[dict], 'name' and 'type' must be in keys, 'shape' optional.

                - dict, columns.keys() as name, columns.values() is dict, and 'type' inside, 'shape' optional.

        Raises:
            RuntimeError: If failed to parse columns.
            RuntimeError: If column's name field is missing.
            RuntimeError: If column's type field is missing.

        Example:
            >>> schema = Schema()
            >>> columns1 = [{'name': 'image', 'type': 'int8', 'shape': [3, 3]},
            >>>             {'name': 'label', 'type': 'int8', 'shape': [1]}]
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


class VOCDataset(MappableDataset):
    """
    A source dataset for reading and parsing VOC dataset.

    The generated dataset has multiple columns :

    - task='Detection', column: [['image', dtype=uint8], ['bbox', dtype=float32], ['label', dtype=uint32],
      ['difficult', dtype=uint32], ['truncate', dtype=uint32]].
    - task='Segmentation', column: [['image', dtype=uint8], ['target',dtype=uint8]].

    This dataset can take in a sampler. 'sampler' and 'shuffle' are mutually exclusive. The table
    below shows what input arguments are allowed and their expected behavior.

    .. list-table:: Expected Order Behavior of Using 'sampler' and 'shuffle'
       :widths: 25 25 50
       :header-rows: 1

       * - Parameter 'sampler'
         - Parameter 'shuffle'
         - Expected Order Behavior
       * - None
         - None
         - random order
       * - None
         - True
         - random order
       * - None
         - False
         - sequential order
       * - Sampler object
         - None
         - order defined by sampler
       * - Sampler object
         - True
         - not allowed
       * - Sampler object
         - False
         - not allowed

    Citation of VOC dataset.

    .. code-block::

        @article{Everingham10,
        author       = {Everingham, M. and Van~Gool, L. and Williams, C. K. I. and Winn, J. and Zisserman, A.},
        title        = {The Pascal Visual Object Classes (VOC) Challenge},
        journal      = {International Journal of Computer Vision},
        volume       = {88},
        year         = {2010},
        number       = {2},
        month        = {jun},
        pages        = {303--338},
        biburl       = {http://host.robots.ox.ac.uk/pascal/VOC/pubs/everingham10.html#bibtex},
        howpublished = {http://host.robots.ox.ac.uk/pascal/VOC/voc{year}/index.html},
        description  = {The PASCAL Visual Object Classes (VOC) challenge is a benchmark in visual
                        object category recognition and detection, providing the vision and machine
                        learning communities with a standard dataset of images and annotation, and
                        standard evaluation procedures.}
        }

    Args:
        dataset_dir (str): Path to the root directory that contains the dataset.
        task (str): Set the task type of reading voc data, now only support "Segmentation" or "Detection"
            (default="Segmentation").
        usage (str): The type of data list text file to be read (default="train").
        class_indexing (dict, optional): A str-to-int mapping from label name to index, only valid in
            "Detection" task (default=None, the folder names will be sorted alphabetically and each
            class will be given a unique index starting from 0).
        num_samples (int, optional): The number of images to be included in the dataset
            (default=None, all images).
        num_parallel_workers (int, optional): Number of workers to read the data
            (default=None, number set in the config).
        shuffle (bool, optional): Whether to perform shuffle on the dataset (default=None, expected
            order behavior shown in the table).
        decode (bool, optional): Decode the images after reading (default=False).
        sampler (Sampler, optional): Object used to choose samples from the dataset
            (default=None, expected order behavior shown in the table).
        num_shards (int, optional): Number of shards that the dataset will be divided
            into (default=None). When this argument is specified, 'num_samples' reflects
            the max sample number of per shard.
        shard_id (int, optional): The shard ID within num_shards (default=None). This
            argument can only be specified when num_shards is also specified.
        cache (DatasetCache, optional): Use tensor caching service to speed up dataset processing.
            (default=None, which means no cache is used).

    Raises:
        RuntimeError: If xml of Annotations is an invalid format.
        RuntimeError: If xml of Annotations loss attribution of "object".
        RuntimeError: If xml of Annotations loss attribution of "bndbox".
        RuntimeError: If sampler and shuffle are specified at the same time.
        RuntimeError: If sampler and sharding are specified at the same time.
        RuntimeError: If num_shards is specified but shard_id is None.
        RuntimeError: If shard_id is specified but num_shards is None.
        ValueError: If task is not equal 'Segmentation' or 'Detection'.
        ValueError: If task equal 'Segmentation' but class_indexing is not None.
        ValueError: If txt related to mode is not exist.
        ValueError: If shard_id is invalid (< 0 or >= num_shards).

    Examples:
        >>> voc_dataset_dir = "/path/to/voc_dataset_directory"
        >>>
        >>> # 1) Read VOC data for segmentatation training
        >>> dataset = ds.VOCDataset(dataset_dir=voc_dataset_dir, task="Segmentation", usage="train")
        >>>
        >>> # 2) Read VOC data for detection training
        >>> dataset = ds.VOCDataset(dataset_dir=voc_dataset_dir, task="Detection", usage="train")
        >>>
        >>> # 3) Read all VOC dataset samples in voc_dataset_dir with 8 threads in random order
        >>> dataset = ds.VOCDataset(dataset_dir=voc_dataset_dir, task="Detection", usage="train",
        ...                         num_parallel_workers=8)
        >>>
        >>> # 4) Read then decode all VOC dataset samples in voc_dataset_dir in sequence
        >>> dataset = ds.VOCDataset(dataset_dir=voc_dataset_dir, task="Detection", usage="train",
        ...                         decode=True, shuffle=False)
        >>>
        >>> # In VOC dataset, if task='Segmentation', each dictionary has keys "image" and "target"
        >>> # In VOC dataset, if task='Detection', each dictionary has keys "image" and "annotation"
    """

    @check_vocdataset
    def __init__(self, dataset_dir, task="Segmentation", usage="train", class_indexing=None, num_samples=None,
                 num_parallel_workers=None, shuffle=None, decode=False, sampler=None, num_shards=None, shard_id=None,
                 cache=None):
        super().__init__(num_parallel_workers=num_parallel_workers, sampler=sampler, num_samples=num_samples,
                         shuffle=shuffle, num_shards=num_shards, shard_id=shard_id, cache=cache)
        self.dataset_dir = dataset_dir
        self.task = replace_none(task, "Segmentation")
        self.usage = replace_none(usage, "train")
        self.class_indexing = replace_none(class_indexing, {})
        self.decode = replace_none(decode, False)

    def parse(self, children=None):
        return cde.VOCNode(self.dataset_dir, self.task, self.usage, self.class_indexing, self.decode, self.sampler)

    def get_class_indexing(self):
        """
        Get the class index.

        Returns:
            dict, a str-to-int mapping from label name to index.
        """
        if self.task != "Detection":
            raise NotImplementedError("Only 'Detection' support get_class_indexing.")
        if self.class_indexing is None or not self.class_indexing:
            if self._class_indexing is None:
                runtime_getter = self._init_tree_getters()
                self._class_indexing = runtime_getter[0].GetClassIndexing()
            self.class_indexing = {}
            for pair in self._class_indexing:
                self.class_indexing[pair[0]] = pair[1][0]
        return self.class_indexing


class CocoDataset(MappableDataset):
    """
    A source dataset for reading and parsing COCO dataset.

    `CocoDataset` supports four kinds of tasks, which are Object Detection, Keypoint Detection, Stuff Segmentation and
    Panoptic Segmentation of 2017 Train/Val/Test dataset.

    The generated dataset has multi-columns :

    - task='Detection', column: [['image', dtype=uint8], ['bbox', dtype=float32], ['category_id', dtype=uint32],
      ['iscrowd', dtype=uint32]].
    - task='Stuff', column: [['image', dtype=uint8], ['segmentation',dtype=float32], ['iscrowd',dtype=uint32]].
    - task='Keypoint', column: [['image', dtype=uint8], ['keypoints', dtype=float32],
      ['num_keypoints', dtype=uint32]].
    - task='Panoptic', column: [['image', dtype=uint8], ['bbox', dtype=float32], ['category_id', dtype=uint32],
      ['iscrowd', dtype=uint32], ['area', dtype=uint32]].

    This dataset can take in a sampler. 'sampler' and 'shuffle' are mutually exclusive. CocoDataset doesn't support
    PKSampler. The table below shows what input arguments are allowed and their expected behavior.

    .. list-table:: Expected Order Behavior of Using 'sampler' and 'shuffle'
       :widths: 25 25 50
       :header-rows: 1

       * - Parameter 'sampler'
         - Parameter 'shuffle'
         - Expected Order Behavior
       * - None
         - None
         - random order
       * - None
         - True
         - random order
       * - None
         - False
         - sequential order
       * - Sampler object
         - None
         - order defined by sampler
       * - Sampler object
         - True
         - not allowed
       * - Sampler object
         - False
         - not allowed

    Citation of Coco dataset.

    .. code-block::

        @article{DBLP:journals/corr/LinMBHPRDZ14,
        author        = {Tsung{-}Yi Lin and Michael Maire and Serge J. Belongie and
                         Lubomir D. Bourdev and  Ross B. Girshick and James Hays and
                         Pietro Perona and Deva Ramanan and Piotr Doll{\'{a}}r and C. Lawrence Zitnick},
        title         = {Microsoft {COCO:} Common Objects in Context},
        journal       = {CoRR},
        volume        = {abs/1405.0312},
        year          = {2014},
        url           = {http://arxiv.org/abs/1405.0312},
        archivePrefix = {arXiv},
        eprint        = {1405.0312},
        timestamp     = {Mon, 13 Aug 2018 16:48:13 +0200},
        biburl        = {https://dblp.org/rec/journals/corr/LinMBHPRDZ14.bib},
        bibsource     = {dblp computer science bibliography, https://dblp.org},
        description   = {COCO is a large-scale object detection, segmentation, and captioning dataset.
                         It contains 91 common object categories with 82 of them having more than 5,000
                         labeled instances. In contrast to the popular ImageNet dataset, COCO has fewer
                         categories but more instances per category.}
        }

    Args:
        dataset_dir (str): Path to the root directory that contains the dataset.
        annotation_file (str): Path to the annotation JSON.
        task (str): Set the task type for reading COCO data.  Supported task types:
            'Detection', 'Stuff', 'Panoptic' and 'Keypoint' (default='Detection').
        num_samples (int, optional): The number of images to be included in the dataset
            (default=None, all images).
        num_parallel_workers (int, optional): Number of workers to read the data
            (default=None, number set in the configuration file).
        shuffle (bool, optional): Whether to perform shuffle on the dataset (default=None, expected
            order behavior shown in the table).
        decode (bool, optional): Decode the images after reading (default=False).
        sampler (Sampler, optional): Object used to choose samples from the dataset
            (default=None, expected order behavior shown in the table).
        num_shards (int, optional): Number of shards that the dataset will be divided
            into (default=None). When this argument is specified, 'num_samples' reflects
            the max sample number of per shard.
        shard_id (int, optional): The shard ID within num_shards (default=None). This
            argument can only be specified when num_shards is also specified.
        cache (DatasetCache, optional): Use tensor caching service to speed up dataset processing.
            (default=None, which means no cache is used).

    Raises:
        RuntimeError: If sampler and shuffle are specified at the same time.
        RuntimeError: If sampler and sharding are specified at the same time.
        RuntimeError: If num_shards is specified but shard_id is None.
        RuntimeError: If shard_id is specified but num_shards is None.
        RuntimeError: If parse JSON file failed.
        ValueError: If task is not in ['Detection', 'Stuff', 'Panoptic', 'Keypoint'].
        ValueError: If annotation_file is not exist.
        ValueError: If dataset_dir is not exist.
        ValueError: If shard_id is invalid (< 0 or >= num_shards).

    Examples:
        >>> coco_dataset_dir = "/path/to/coco_dataset_directory/images"
        >>> coco_annotation_file = "/path/to/coco_dataset_directory/annotation_file"
        >>>
        >>> # 1) Read COCO data for Detection task
        >>> dataset = ds.CocoDataset(dataset_dir=coco_dataset_dir,
        ...                          annotation_file=coco_annotation_file,
        ...                          task='Detection')
        >>>
        >>> # 2) Read COCO data for Stuff task
        >>> dataset = ds.CocoDataset(dataset_dir=coco_dataset_dir,
        ...                          annotation_file=coco_annotation_file,
        ...                          task='Stuff')
        >>>
        >>> # 3) Read COCO data for Panoptic task
        >>> dataset = ds.CocoDataset(dataset_dir=coco_dataset_dir,
        ...                          annotation_file=coco_annotation_file,
        ...                          task='Panoptic')
        >>>
        >>> # 4) Read COCO data for Keypoint task
        >>> dataset = ds.CocoDataset(dataset_dir=coco_dataset_dir,
        ...                          annotation_file=coco_annotation_file,
        ...                          task='Keypoint')
        >>>
        >>> # In COCO dataset, each dictionary has keys "image" and "annotation"
    """

    @check_cocodataset
    def __init__(self, dataset_dir, annotation_file, task="Detection", num_samples=None, num_parallel_workers=None,
                 shuffle=None, decode=False, sampler=None, num_shards=None, shard_id=None, cache=None):
        super().__init__(num_parallel_workers=num_parallel_workers, sampler=sampler, num_samples=num_samples,
                         shuffle=shuffle, num_shards=num_shards, shard_id=shard_id, cache=cache)
        self.dataset_dir = dataset_dir
        self.annotation_file = annotation_file
        self.task = replace_none(task, "Detection")
        self.decode = replace_none(decode, False)

    def parse(self, children=None):
        return cde.CocoNode(self.dataset_dir, self.annotation_file, self.task, self.decode, self.sampler)

    def get_class_indexing(self):
        """
        Get the class index.

        Returns:
            dict, a str-to-list<int> mapping from label name to index
        """
        if self.task not in {"Detection", "Panoptic"}:
            raise NotImplementedError("Only 'Detection' and 'Panoptic' support get_class_indexing.")
        if self._class_indexing is None:
            runtime_getter = self._init_tree_getters()
            self._class_indexing = dict(runtime_getter[0].GetClassIndexing())
        return self._class_indexing


class CelebADataset(MappableDataset):
    """
    A source dataset for reading and parsing CelebA dataset. Only support to read `list_attr_celeba.txt` currently,
    which is the attribute annotations of the dataset.

    Note:
        The generated dataset has two columns ['image', 'attr'].
        The image tensor is of the uint8 type. The attribute tensor is of the uint32 type and one hot encoded.

    Citation of CelebA dataset.

    .. code-block::

        @article{DBLP:journals/corr/LiuLWT14,
        author    = {Ziwei Liu and Ping Luo and Xiaogang Wang and Xiaoou Tang},
        title     = {Deep Learning Face Attributes in the Wild},
        journal   = {CoRR},
        volume    = {abs/1411.7766},
        year      = {2014},
        url       = {http://arxiv.org/abs/1411.7766},
        archivePrefix = {arXiv},
        eprint    = {1411.7766},
        timestamp = {Tue, 10 Dec 2019 15:37:26 +0100},
        biburl    = {https://dblp.org/rec/journals/corr/LiuLWT14.bib},
        bibsource = {dblp computer science bibliography, https://dblp.org},
        howpublished = {http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html},
        description  = {CelebFaces Attributes Dataset (CelebA) is a large-scale face attributes dataset
                        with more than 200K celebrity images, each with 40 attribute annotations.
                        The images in this dataset cover large pose variations and background clutter.
                        CelebA has large diversities, large quantities, and rich annotations, including
                        * 10,177 number of identities,
                        * 202,599 number of face images, and
                        * 5 landmark locations, 40 binary attributes annotations per image.
                        The dataset can be employed as the training and test sets for the following computer
                        vision tasks: face attribute recognition, face detection, landmark (or facial part)
                        localization, and face editing & synthesis.}
        }

    Args:
        dataset_dir (str): Path to the root directory that contains the dataset.
        num_parallel_workers (int, optional): Number of workers to read the data (default=None, will use value set in
            the config).
        shuffle (bool, optional): Whether to perform shuffle on the dataset (default=None).
        usage (str): one of 'all', 'train', 'valid' or 'test' (default='all', will read all samples).
        sampler (Sampler, optional): Object used to choose samples from the dataset (default=None).
        decode (bool, optional): decode the images after reading (default=False).
        extensions (list[str], optional): List of file extensions to be included in the dataset (default=None).
        num_samples (int, optional): The number of images to be included in the dataset
            (default=None, will include all images).
        num_shards (int, optional): Number of shards that the dataset will be divided
            into (default=None). When this argument is specified, `num_samples` reflects
            the max sample number of per shard.
        shard_id (int, optional): The shard ID within `num_shards` (default=None). This
            argument can only be specified when `num_shards` is also specified.
        cache (DatasetCache, optional): Use tensor caching service to speed up dataset processing.
            (default=None, which means no cache is used).

    Examples:
        >>> celeba_dataset_dir = "/path/to/celeba_dataset_directory"
        >>> dataset = ds.CelebADataset(dataset_dir=celeba_dataset_dir, usage='train')
    """

    @check_celebadataset
    def __init__(self, dataset_dir, num_parallel_workers=None, shuffle=None, usage='all', sampler=None, decode=False,
                 extensions=None, num_samples=None, num_shards=None, shard_id=None, cache=None):
        super().__init__(num_parallel_workers=num_parallel_workers, sampler=sampler, num_samples=num_samples,
                         shuffle=shuffle, num_shards=num_shards, shard_id=shard_id, cache=cache)
        self.dataset_dir = dataset_dir
        self.decode = replace_none(decode, False)
        self.extensions = replace_none(extensions, [])
        self.usage = replace_none(usage, "all")

    def parse(self, children=None):
        if self.usage != "all":
            dir = os.path.realpath(self.dataset_dir)
            partition_file = os.path.join(dir, "list_eval_partition.txt")
            if os.path.exists(partition_file) is False:
                raise RuntimeError("Partition file can not be found when usage is not 'all'.")
        return cde.CelebANode(self.dataset_dir, self.usage, self.sampler, self.decode, self.extensions)


class CLUEDataset(SourceDataset):
    """
    A source dataset that reads and parses CLUE datasets.
    CLUE, the Chinese Language Understanding Evaluation Benchmark, is a collection of datasets, baselines,
    pre-trained models, corpus and leaderboard. Supported CLUE classification tasks: 'AFQMC', 'TNEWS', 'IFLYTEK',
    'CMNLI', 'WSC' and 'CSL'.

    Citation of CLUE dataset.

    .. code-block::

        @article{CLUEbenchmark,
        title   = {CLUE: A Chinese Language Understanding Evaluation Benchmark},
        author  = {Liang Xu, Xuanwei Zhang, Lu Li, Hai Hu, Chenjie Cao, Weitang Liu, Junyi Li, Yudong Li,
                   Kai Sun, Yechen Xu, Yiming Cui, Cong Yu, Qianqian Dong, Yin Tian, Dian Yu, Bo Shi, Jun Zeng,
                   Rongzhao Wang, Weijian Xie, Yanting Li, Yina Patterson, Zuoyu Tian, Yiwen Zhang, He Zhou,
                   Shaoweihua Liu, Qipeng Zhao, Cong Yue, Xinrui Zhang, Zhengliang Yang, Zhenzhong Lan},
        journal = {arXiv preprint arXiv:2004.05986},
        year    = {2020},
        howpublished = {https://github.com/CLUEbenchmark/CLUE},
        description  = {CLUE, a Chinese Language Understanding Evaluation benchmark. It contains eight different
                        tasks, including single-sentence classification, sentence pair classification, and machine
                        reading comprehension.}
        }

    Args:
        dataset_files (Union[str, list[str]]): String or list of files to be read or glob strings to search for
            a pattern of files. The list will be sorted in a lexicographical order.
        task (str, optional): The kind of task, one of 'AFQMC', 'TNEWS', 'IFLYTEK', 'CMNLI', 'WSC' and 'CSL'.
            (default=AFQMC).
        usage (str, optional): Need train, test or eval data (default="train").
        num_samples (int, optional): Number of samples (rows) to read (default=None, reads the full dataset).
        num_parallel_workers (int, optional): Number of workers to read the data
            (default=None, number set in the config).
        shuffle (Union[bool, Shuffle level], optional): Perform reshuffling of the data every epoch
            (default=Shuffle.GLOBAL).
            If shuffle is False, no shuffling will be performed;
            If shuffle is True, the behavior is the same as setting shuffle to be Shuffle.GLOBAL
            Otherwise, there are two levels of shuffling:

            - Shuffle.GLOBAL: Shuffle both the files and samples.

            - Shuffle.FILES: Shuffle files only.

        num_shards (int, optional): Number of shards that the dataset will be divided into (default=None).
            When this argument is specified, 'num_samples' reflects the max sample number of per shard.
        shard_id (int, optional): The shard ID within num_shards (default=None). This
            argument can only be specified when num_shards is also specified.
        cache (DatasetCache, optional): Use tensor caching service to speed up dataset processing.
            (default=None, which means no cache is used).

    Examples:
        >>> clue_dataset_dir = ["/path/to/clue_dataset_file"] # contains 1 or multiple clue files
        >>> dataset = ds.CLUEDataset(dataset_files=clue_dataset_dir, task='AFQMC', usage='train')
    """

    @check_cluedataset
    def __init__(self, dataset_files, task='AFQMC', usage='train', num_samples=None, num_parallel_workers=None,
                 shuffle=Shuffle.GLOBAL, num_shards=None, shard_id=None, cache=None):
        super().__init__(num_parallel_workers=num_parallel_workers, num_samples=num_samples, shuffle=shuffle,
                         num_shards=num_shards, shard_id=shard_id, cache=cache)
        self.dataset_files = self._find_files(dataset_files)

        self.task_dict = {
            'AFQMC': {
                'train': {
                    'sentence1': 'sentence1', 'sentence2': 'sentence2', 'label': 'label'
                },
                'test': {
                    'id': 'id', 'sentence1': 'sentence1', 'sentence2': 'sentence2'
                },
                'eval': {
                    'sentence1': 'sentence1', 'sentence2': 'sentence2', 'label': 'label'
                }
            },
            'CMNLI': {
                'train': {
                    'sentence1': 'sentence1', 'sentence2': 'sentence2', 'label': 'label'
                },
                'test': {
                    'id': 'id', 'sentence1': 'sentence1', 'sentence2': 'sentence2'
                },
                'eval': {
                    'sentence1': 'sentence1', 'sentence2': 'sentence2', 'label': 'label'
                }
            },
            'CSL': {
                'train': {
                    'id': 'id', 'abst': 'abst', 'keyword': 'keyword', 'label': 'label'
                },
                'test': {
                    'id': 'id', 'abst': 'abst', 'keyword': 'keyword'
                },
                'eval': {
                    'id': 'id', 'abst': 'abst', 'keyword': 'keyword', 'label': 'label'
                }
            },
            'IFLYTEK': {
                'train': {
                    'label': 'label', 'label_des': 'label_des', 'sentence': 'sentence'
                },
                'test': {
                    'id': 'id', 'sentence': 'sentence',
                },
                'eval': {
                    'label': 'label', 'label_des': 'label_des', 'sentence': 'sentence'
                }
            },
            'TNEWS': {
                'train': {
                    'label': 'label', 'label_desc': 'label_desc', 'sentence': 'sentence', 'keywords': 'keywords'
                },
                'test': {
                    'id': 'id', 'sentence': 'sentence', 'keywords': 'keywords'
                },
                'eval': {
                    'label': 'label', 'label_desc': 'label_desc', 'sentence': 'sentence', 'keywords': 'keywords'
                }
            },
            'WSC': {
                'train': {
                    'span1_index': 'target/span1_index', 'span2_index': 'target/span2_index',
                    'span1_text': 'target/span1_text', 'span2_text': 'target/span2_text', 'idx': 'idx',
                    'label': 'label', 'text': 'text'
                },
                'test': {
                    'span1_index': 'target/span1_index', 'span2_index': 'target/span2_index',
                    'span1_text': 'target/span1_text', 'span2_text': 'target/span2_text', 'idx': 'idx', 'text': 'text'
                },
                'eval': {
                    'span1_index': 'target/span1_index', 'span2_index': 'target/span2_index',
                    'span1_text': 'target/span1_text', 'span2_text': 'target/span2_text', 'idx': 'idx',
                    'label': 'label', 'text': 'text'
                }
            }
        }
        self.usage = replace_none(usage, 'train')
        self.cols_to_keyword = self.task_dict[task][self.usage]
        self.task = replace_none(task, 'AFQMC')

    def parse(self, children=None):
        return cde.CLUENode(self.dataset_files, self.task, self.usage, self.num_samples, self.shuffle_flag,
                            self.num_shards, self.shard_id)


class CSVDataset(SourceDataset):
    """
    A source dataset that reads and parses comma-separated values (CSV) datasets.

    Args:
        dataset_files (Union[str, list[str]]): String or list of files to be read or glob strings to search
            for a pattern of files. The list will be sorted in a lexicographical order.
        field_delim (str, optional): A string that indicates the char delimiter to separate fields (default=',').
        column_defaults (list, optional): List of default values for the CSV field (default=None). Each item
            in the list is either a valid type (float, int, or string). If this is not provided, treats all
            columns as string type.
        column_names (list[str], optional): List of column names of the dataset (default=None). If this
            is not provided, infers the column_names from the first row of CSV file.
        num_samples (int, optional): Number of samples (rows) to read (default=None, reads the full dataset).
        num_parallel_workers (int, optional): Number of workers to read the data
            (default=None, number set in the config).
        shuffle (Union[bool, Shuffle level], optional): Perform reshuffling of the data every epoch
            (default=Shuffle.GLOBAL).
            If shuffle is False, no shuffling will be performed;
            If shuffle is True, the behavior is the same as setting shuffle to be Shuffle.GLOBAL
            Otherwise, there are two levels of shuffling:

            - Shuffle.GLOBAL: Shuffle both the files and samples.

            - Shuffle.FILES: Shuffle files only.

        num_shards (int, optional): Number of shards that the dataset will be divided into (default=None).
            When this argument is specified, 'num_samples' reflects the max sample number of per shard.
        shard_id (int, optional): The shard ID within num_shards (default=None). This
            argument can only be specified when num_shards is also specified.
        cache (DatasetCache, optional): Use tensor caching service to speed up dataset processing.
            (default=None, which means no cache is used).


    Examples:
        >>> csv_dataset_dir = ["/path/to/csv_dataset_file"] # contains 1 or multiple csv files
        >>> dataset = ds.CSVDataset(dataset_files=csv_dataset_dir, column_names=['col1', 'col2', 'col3', 'col4'])
    """

    @check_csvdataset
    def __init__(self, dataset_files, field_delim=',', column_defaults=None, column_names=None, num_samples=None,
                 num_parallel_workers=None, shuffle=Shuffle.GLOBAL, num_shards=None, shard_id=None, cache=None):
        super().__init__(num_parallel_workers=num_parallel_workers, num_samples=num_samples, shuffle=shuffle,
                         num_shards=num_shards, shard_id=shard_id, cache=cache)
        self.dataset_files = self._find_files(dataset_files)
        self.dataset_files.sort()
        self.field_delim = replace_none(field_delim, ',')
        self.column_defaults = replace_none(column_defaults, [])
        self.column_names = replace_none(column_names, [])

    def parse(self, children=None):
        return cde.CSVNode(self.dataset_files, self.field_delim, self.column_defaults, self.column_names,
                           self.num_samples, self.shuffle_flag, self.num_shards, self.shard_id)


class TextFileDataset(SourceDataset):
    """
    A source dataset that reads and parses datasets stored on disk in text format.
    The generated dataset has one column ['text'].

    Args:
        dataset_files (Union[str, list[str]]): String or list of files to be read or glob strings to search for a
            pattern of files. The list will be sorted in a lexicographical order.
        num_samples (int, optional): Number of samples (rows) to read (default=None, reads the full dataset).
        num_parallel_workers (int, optional): Number of workers to read the data
            (default=None, number set in the config).
        shuffle (Union[bool, Shuffle level], optional): Perform reshuffling of the data every epoch
            (default=Shuffle.GLOBAL).
            If shuffle is False, no shuffling will be performed;
            If shuffle is True, the behavior is the same as setting shuffle to be Shuffle.GLOBAL
            Otherwise, there are two levels of shuffling:

            - Shuffle.GLOBAL: Shuffle both the files and samples.

            - Shuffle.FILES: Shuffle files only.

        num_shards (int, optional): Number of shards that the dataset will be divided into (default=None).
            When this argument is specified, 'num_samples' reflects the max sample number of per shard.
        shard_id (int, optional): The shard ID within num_shards (default=None). This
            argument can only be specified when num_shards is also specified.
        cache (DatasetCache, optional): Use tensor caching service to speed up dataset processing.
            (default=None, which means no cache is used).

    Examples:
        >>> text_file_dataset_dir = ["/path/to/text_file_dataset_file"] # contains 1 or multiple text files
        >>> dataset = ds.TextFileDataset(dataset_files=text_file_dataset_dir)
    """

    @check_textfiledataset
    def __init__(self, dataset_files, num_samples=None, num_parallel_workers=None, shuffle=Shuffle.GLOBAL,
                 num_shards=None, shard_id=None, cache=None):
        super().__init__(num_parallel_workers=num_parallel_workers, num_samples=num_samples, shuffle=shuffle,
                         num_shards=num_shards, shard_id=shard_id, cache=cache)
        self.dataset_files = self._find_files(dataset_files)
        self.dataset_files.sort()

    def parse(self, children=None):
        return cde.TextFileNode(self.dataset_files, self.num_samples, self.shuffle_flag, self.num_shards, self.shard_id)


class _NumpySlicesDataset:
    """
    Mainly for dealing with several kinds of formats of Python data, and return one row each time.
    """

    def __init__(self, data, column_list=None):
        self.column_list = None
        # Convert dict data into tuple
        if isinstance(data, dict):
            data = self.process_dict(data)

        if isinstance(data, tuple):
            self.data = ()
            data_len = len(data)
            for i in range(data_len):
                self.data = self.data + (np.array(data[i]),)
        else:
            self.data = (np.array(data),)

        # check whether the data length in each column is equal
        data_len = [len(data_item) for data_item in self.data]
        if data_len[1:] != data_len[:-1]:
            raise ValueError("Data length in each column is not equal.")

        # Init column_name
        if column_list is not None:
            self.column_list = column_list
        elif self.column_list is None:
            self.column_list = []
            column_num = len(self.data)
            for i in range(column_num):
                self.column_list.append("column_" + str(i))

    def __getitem__(self, index):
        data_row = [d[index, ...] for d in self.data]
        data_res = tuple(data_row)
        return data_res

    def __len__(self):
        return len(self.data[0])

    def process_dict(self, input_data):
        """
        Convert the dict like data into tuple format, when input is a tuple of dicts then compose it into a dict first.
        """
        # Convert pandas like dict(has "values" column) into General dict
        data_keys = list(input_data.keys())
        data_col = input_data[data_keys[0]]
        if hasattr(data_col, "values"):
            new_dict = {}
            for key in data_keys:
                item1 = input_data.pop(key)
                new_dict[key] = item1.values
            input_data = new_dict

        # Convert the data in dict into tuple
        data = ()
        keys = list(input_data.keys())
        self.column_list = keys
        for key in keys:
            value = input_data[key]
            data = data + (list(value),)

        return data


class NumpySlicesDataset(GeneratorDataset):
    """
    Creates a dataset with given data slices, mainly for loading Python data into dataset.

    This dataset can take in a sampler. 'sampler' and 'shuffle' are mutually exclusive. The table
    below shows what input arguments are allowed and their expected behavior.

    .. list-table:: Expected Order Behavior of Using 'sampler' and 'shuffle'
       :widths: 25 25 50
       :header-rows: 1

       * - Parameter 'sampler'
         - Parameter 'shuffle'
         - Expected Order Behavior
       * - None
         - None
         - random order
       * - None
         - True
         - random order
       * - None
         - False
         - sequential order
       * - Sampler object
         - None
         - order defined by sampler
       * - Sampler object
         - True
         - not allowed
       * - Sampler object
         - False
         - not allowed

    Args:
        data (Union[list, tuple, dict]) Input of given data. Supported data types include: list, tuple, dict and other
            NumPy formats. Input data will be sliced along the first dimension and generate additional rows, if input is
            list, there will be one column in each row, otherwise there tends to be multi columns. Large data is not
            recommended to be loaded in this way as data is loading into memory.
        column_names (list[str], optional): List of column names of the dataset (default=None). If column_names is not
            provided, when data is dict, column_names will be its keys, otherwise it will be like column_0, column_1 ...
        num_samples (int, optional): The number of samples to be included in the dataset (default=None, all images).
        num_parallel_workers (int, optional): Number of subprocesses used to fetch the dataset in parallel (default=1).
        shuffle (bool, optional): Whether or not to perform shuffle on the dataset. Random accessible input is required.
            (default=None, expected order behavior shown in the table).
        sampler (Union[Sampler, Iterable], optional): Object used to choose samples from the dataset. Random accessible
            input is required (default=None, expected order behavior shown in the table).
        num_shards (int, optional): Number of shards that the dataset will be divided into (default=None).
            Random accessible input is required. When this argument is specified, 'num_samples' reflects the max
            sample number of per shard.
        shard_id (int, optional): The shard ID within num_shards (default=None). This argument must be specified only
            when num_shards is also specified. Random accessible input is required.

    Examples:
        >>> # 1) Input data can be a list
        >>> data = [1, 2, 3]
        >>> dataset = ds.NumpySlicesDataset(data=data, column_names=["column_1"])
        >>>
        >>> # 2) Input data can be a dictionary, and column_names will be its keys
        >>> data = {"a": [1, 2], "b": [3, 4]}
        >>> dataset = ds.NumpySlicesDataset(data=data)
        >>>
        >>> # 3) Input data can be a tuple of lists (or NumPy arrays), each tuple element refers to data in each column
        >>> data = ([1, 2], [3, 4], [5, 6])
        >>> dataset = ds.NumpySlicesDataset(data=data, column_names=["column_1", "column_2", "column_3"])
        >>>
        >>> # 4) Load data from CSV file
        >>> import pandas as pd
        >>> df = pd.read_csv(filepath_or_buffer=csv_dataset_dir[0])
        >>> dataset = ds.NumpySlicesDataset(data=dict(df), shuffle=False)
    """

    @check_numpyslicesdataset
    def __init__(self, data, column_names=None, num_samples=None, num_parallel_workers=1, shuffle=None, sampler=None,
                 num_shards=None, shard_id=None):
        dataset = _NumpySlicesDataset(data, column_names)
        super().__init__(dataset, column_names=dataset.column_list, num_samples=num_samples,
                         num_parallel_workers=num_parallel_workers, shuffle=shuffle, sampler=sampler,
                         num_shards=num_shards, shard_id=shard_id)


class _PaddedDataset:
    """
    Mainly for combining false samples provided by users into a dataset.

    Args:
        padded_samples (list(dict)): Data provided by user to be added to the initial Dataset.
    """

    def __init__(self, padded_samples):
        self.column_names = list(padded_samples[0].keys())
        self.padded_samples = padded_samples

    def __getitem__(self, item):
        return (self.padded_samples[item][key] for key in self.column_names)

    def __len__(self):
        return len(self.padded_samples)


class PaddedDataset(GeneratorDataset):
    """
    Creates a dataset with filler data provided by user. Mainly used to add to the original data set
    and assign it to the corresponding shard.

    Args:
        padded_samples (list(dict)): Samples provided by user.

    Raises:
        TypeError: If padded_samples is not an instance of list.
        TypeError: If the element of padded_samples is not an instance of dict.
        ValueError: If the padded_samples is empty.

    Examples:
        >>> import numpy as np
        >>> data = [{'image': np.zeros(1, np.uint8)}, {'image': np.zeros(2, np.uint8)}]
        >>> dataset = ds.PaddedDataset(padded_samples=data)
    """

    @check_paddeddataset
    def __init__(self, padded_samples):
        dataset = _PaddedDataset(padded_samples)
        super().__init__(dataset, column_names=dataset.column_names, num_shards=None, shard_id=None, shuffle=False)
        self._dataset_size = len(dataset.padded_samples)
        self.padded_samples = padded_samples
