# Copyright 2019 Huawei Technologies Co., Ltd
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
datasets.py supports various formats of datasets, including ImageNet, TFData,
MNIST, Cifar10/100, Manifest, MindRecord, etc. This module could load data in
high performance and parse data precisely. It also provides the following
operations for users to preprocess data: shuffle, batch, repeat, map, and zip.
"""
import glob
import json
import math
import os
import random
import uuid
from enum import Enum
from importlib import import_module

import numpy as np
from mindspore._c_dataengine import DataType, TFReaderOp, ImageFolderOp, CifarOp, MnistOp, ManifestOp, \
    MindRecordOp, TextFileOp, CBatchInfo
from mindspore._c_expression import typing

from mindspore import log as logger
from . import samplers
from .iterators import DictIterator, TupleIterator
from .validators import check, check_batch, check_shuffle, check_map, check_filter, check_repeat, check_skip, check_zip, check_rename, \
    check_take, check_project, check_imagefolderdatasetv2, check_mnist_cifar_dataset, check_manifestdataset, \
    check_tfrecorddataset, check_vocdataset, check_celebadataset, check_minddataset, check_generatordataset, \
    check_zip_dataset, check_add_column, check_textfiledataset
from ..core.datatypes import mstype_to_detype, mstypelist_to_detypelist

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
    Zips the datasets in the input tuple of datasets.

    Args:
        datasets (tuple of class Dataset): A tuple of datasets to be zipped together.
            The number of datasets should be more than 1.

    Returns:
        DatasetOp, ZipDataset.

    Raises:
        ValueError: If the number of datasets is 1.
        TypeError: If datasets is not a tuple.

    Examples:
            >>> import mindspore.dataset as ds
            >>>
            >>> dataset_dir1 = "path/to/imagefolder_directory1"
            >>> dataset_dir2 = "path/to/imagefolder_directory2"
            >>> ds1 = ds.ImageFolderDatasetV2(dataset_dir1, num_parallel_workers=8)
            >>> ds2 = ds.ImageFolderDatasetV2(dataset_dir2, num_parallel_workers=8)
            >>>
            >>> # creates a dataset which is the combination of ds1 and ds2
            >>> data = ds.zip((ds1, ds2))
    """
    if len(datasets) <= 1:
        raise ValueError(
            "Can't zip empty or just one dataset!")
    return ZipDataset(datasets)


def get_num_rows(num_rows, num_shards):
    """
    Get the number rows of the dataset according to the shards.

    Args:
        num_rows (int): The number rows of the dataset should be more than 0.
            The number rows of the dataset should be more than 0.
        num_shards (int or None): Number of shards that the dataset should be divided into.
            The number of shards should be None or more than 1.

    Returns:
        Int, number of rows.

    Raises:
        ValueError: If num_rows is invalid (< 0).
        ValueError: If num_shards is invalid (<= 0).
    """
    if num_rows < 0:
        raise ValueError("num_rows is invalid (< 0)")

    if num_shards is not None:
        if num_shards <= 0:
            raise ValueError("num_shards is invalid (<= 0)")
        if num_rows % num_shards == 0:
            num_rows = num_rows // num_shards
        else:
            num_rows = num_rows // num_shards + 1
    return num_rows


class Dataset:
    """
    Abstract class to represent a dataset in DataEngine's data pipeline.

    This class is the base class of SourceDataset and DatasetOp, and represents
    a node in the data flow graph.

    Args:
        num_parallel_workers (int, optional): Number of workers to process the Dataset in parallel
            (default=None).
    """

    def __init__(self, num_parallel_workers=None):
        self.input = []
        self.output = []
        self.num_parallel_workers = num_parallel_workers
        self._device_iter = 0
        self._input_indexs = ()
        self._output_types = None
        self._output_shapes = None
        self._dataset_size = None
        self._batch_size = None
        self._num_classes = None
        self._repeat_count = None

    def get_args(self):
        """
        Returns attributes (member variables) related to the current class.

        Must include all arguments passed to the __init__() of the current class, excluding 'input_dataset'.

        Args:

        Returns:
            Python dictionary.
        """
        args = dict()
        args["num_parallel_workers"] = self.num_parallel_workers
        return args

    @check_batch
    def batch(self, batch_size, drop_remainder=False, num_parallel_workers=None, per_batch_map=None,
              input_columns=None):
        """
        Combines batch_size number of consecutive rows into batches.

        For any child node, a batch is treated as a single row.
        For any column, all the elements within that column must have the same shape.
        If a per_batch_map callable is provided, it will be applied to the batches of tensors.

        Note:
            The order of using repeat and batch reflects the number of batches. Recommend that
            repeat operation should be used after batch operation.

        Args:
            batch_size (int or function): The number of rows each batch is created with. An
                int or callable which takes exactly 1 parameter, BatchInfo.
            drop_remainder (bool, optional): Determines whether or not to drop the last
                possibly incomplete batch (default=False). If True, and if there are less
                than batch_size rows available to make the last batch, then those rows will
                be dropped and not propogated to the child node.
            num_parallel_workers (int, optional): Number of workers to process the Dataset in parallel (default=None).
            per_batch_map (callable, optional): Per batch map callable. A callable which takes
                (list[Tensor], list[Tensor], ..., BatchInfo) as input parameters. Each list[Tensor] represent a batch of
                Tensors on a given column. The number of lists should match with number of entries in input_columns. The
                last parameter of the callable should always be a BatchInfo object.
            input_columns (list of string, optional): List of names of the input columns. The size of the list should
                match with signature of per_batch_map callable.

        Returns:
            BatchDataset, dataset batched.

        Examples:
            >>> import mindspore.dataset as ds
            >>> # data is an instance of Dataset object.
            >>> # creates a dataset where every 100 rows is combined into a batch
            >>> # and drops the last incomplete batch if there is one.
            >>> data = data.batch(100, True)
        """
        return BatchDataset(self, batch_size, drop_remainder, num_parallel_workers, per_batch_map, input_columns)

    @check_shuffle
    def shuffle(self, buffer_size):
        """
        Randomly shuffles the rows of this dataset using the following algorithm:

        1. Make a shuffle buffer that contains the first buffer_size rows.
        2. Randomly select an element from the shuffle buffer to be the next row
           propogated to the child node.
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

        Examples:
            >>> import mindspore.dataset as ds
            >>> # data is an instance of Dataset object
            >>> # optionally set the seed for the first epoch
            >>> ds.config.set_seed(58)
            >>>
            >>> # creates a shuffled dataset using a shuffle buffer of size 4
            >>> data = data.shuffle(4)
        """
        return ShuffleDataset(self, buffer_size)

    @check_map
    def map(self, input_columns=None, operations=None, output_columns=None, columns_order=None,
            num_parallel_workers=None):
        """
        Applies each operation in operations to this dataset.

        The order of operations is determined by the position of each operation in operations.
        operations[0] will be applied first, then operations[1], then operations[2], etc.

        Each operation will be passed one or more columns from the dataset as input, and zero or
        more columns will be outputted. The first operation will be passed the columns specified
        in input_columns as input. If there is more than one operator in operations, the outputted
        columns of the previous operation are used as the input columns for the next operation.
        The columns outputted by the very last operation will be assigned names specified by
        output_columns.

        Only the columns specified in columns_order will be propagated to the child node. These
        columns will be in the same order as specified in columns_order.

        Args:
            input_columns (list[str]): List of the names of the columns that will be passed to
                the first operation as input. The size of this list must match the number of
                input columns expected by the first operator. (default=None, the first
                operation will be passed however many columns that is required, starting from
                the first column).
            operations (list[TensorOp] or Python list[functions]): List of operations to be
                applied on the dataset. Operations are applied in the order they appear in this list.
            output_columns (list[str], optional): List of names assigned to the columns outputted by
                the last operation. This parameter is mandatory if len(input_columns) !=
                len(output_columns). The size of this list must match the number of output
                columns of the last operation. (default=None, output columns will have the same
                name as the input columns, i.e., the columns will be replaced).
            columns_order (list[str], optional): list of all the desired columns to propagate to the
                child node. This list must be a subset of all the columns in the dataset after
                all operations are applied. The order of the columns in each row propagated to the
                child node follow the order they appear in this list. The parameter is mandatory
                if the len(input_columns) != len(output_columns). (default=None, all columns
                will be propagated to the child node, the order of the columns will remain the
                same).
            num_parallel_workers (int, optional): Number of threads used to process the dataset in
                parallel (default=None, the value from the config will be used).

        Returns:
            MapDataset, dataset after mapping operation.

        Examples:
            >>> import mindspore.dataset as ds
            >>> import mindspore.dataset.transforms.vision.c_transforms as c_transforms
            >>>
            >>> # data is an instance of Dataset which has 2 columns, "image" and "label".
            >>> # ds_pyfunc is an instance of Dataset which has 3 columns, "col0", "col1", and "col2". Each column is
            >>> # a 2d array of integers.
            >>>
            >>> # This config is a global setting, meaning that all future operations which
            >>> # uses this config value will use 2 worker threads, unless if specified
            >>> # otherwise in their constructor. set_num_parallel_workers can be called
            >>> # again later if a different number of worker threads are needed.
            >>> ds.config.set_num_parallel_workers(2)
            >>>
            >>> # Two operations, which takes 1 column for input and outputs 1 column.
            >>> decode_op = c_transforms.Decode(rgb_format=True)
            >>> random_jitter_op = c_transforms.RandomColorAdjust((0.8, 0.8), (1, 1), (1, 1), (0, 0))
            >>>
            >>> # 1) Simple map example
            >>>
            >>> operations = [decode_op]
            >>> input_columns = ["image"]
            >>>
            >>> # Applies decode_op on column "image". This column will be replaced by the outputed
            >>> # column of decode_op. Since columns_order is not provided, both columns "image"
            >>> # and "label" will be propagated to the child node in their original order.
            >>> ds_decoded = data.map(input_columns, operations)
            >>>
            >>> # Rename column "image" to "decoded_image"
            >>> output_columns = ["decoded_image"]
            >>> ds_decoded = data.map(input_columns, operations, output_columns)
            >>>
            >>> # Specify the order of the columns.
            >>> columns_order ["label", "image"]
            >>> ds_decoded = data.map(input_columns, operations, None, columns_order)
            >>>
            >>> # Rename column "image" to "decoded_image" and also specify the order of the columns.
            >>> columns_order ["label", "decoded_image"]
            >>> output_columns = ["decoded_image"]
            >>> ds_decoded = data.map(input_columns, operations, output_columns, columns_order)
            >>>
            >>> # Rename column "image" to "decoded_image" and keep only this column.
            >>> columns_order ["decoded_image"]
            >>> output_columns = ["decoded_image"]
            >>> ds_decoded = data.map(input_columns, operations, output_columns, columns_order)
            >>>
            >>> # Simple example using pyfunc. Renaming columns and specifying column order
            >>> # work in the same way as the previous examples.
            >>> input_columns = ["col0"]
            >>> operations = [(lambda x: x + 1)]
            >>> ds_mapped = ds_pyfunc.map(input_columns, operations)
            >>>
            >>> # 2) Map example with more than one operation
            >>>
            >>> # If this list of operations is used with map, decode_op will be applied
            >>> # first, then random_jitter_op will be applied.
            >>> operations = [decode_op, random_jitter_op]
            >>>
            >>> input_columns = ["image"]
            >>>
            >>> # Creates a dataset where the images are decoded, then randomly color jittered.
            >>> # decode_op takes column "image" as input and outputs one column. The column
            >>> # outputted by decode_op is passed as input to random_jitter_op.
            >>> # random_jitter_op will output one column. Column "image" will be replaced by
            >>> # the column outputted by random_jitter_op (the very last operation). All other
            >>> # columns are unchanged. Since columns_order is not specified, the order of the
            >>> # columns will remain the same.
            >>> ds_mapped = data.map(input_columns, operations)
            >>>
            >>> # Creates a dataset that is identical to ds_mapped, except the column "image"
            >>> # that is outputted by random_jitter_op is renamed to "image_transformed".
            >>> # Specifying column order works in the same way as examples in 1).
            >>> output_columns = ["image_transformed"]
            >>> ds_mapped_and_renamed = data.map(input_columns, operation, output_columns)
            >>>
            >>> # Multiple operations using pyfunc. Renaming columns and specifying column order
            >>> # work in the same way as examples in 1).
            >>> input_columns = ["col0"]
            >>> operations = [(lambda x: x + x), (lambda x: x - 1)]
            >>> output_columns = ["col0_mapped"]
            >>> ds_mapped = ds_pyfunc.map(input_columns, operations, output_columns)
            >>>
            >>> # 3) Example where number of input columns is not equal to number of output columns
            >>>
            >>> # operations[0] is a lambda that takes 2 columns as input and outputs 3 columns.
            >>> # operations[1] is a lambda that takes 3 columns as input and outputs 1 column.
            >>> # operations[1] is a lambda that takes 1 column as input and outputs 4 columns.
            >>> #
            >>> # Note: the number of output columns of operation[i] must equal the number of
            >>> # input columns of operation[i+1]. Otherwise, this map call will also result
            >>> # in an error.
            >>> operations = [(lambda x y: (x, x + y, x + y + 1)),
            >>>               (lambda x y z: x * y * z),
            >>>               (lambda x: (x % 2, x % 3, x % 5, x % 7))]
            >>>
            >>> # Note: because the number of input columns is not the same as the number of
            >>> # output columns, the output_columns and columns_order parameter must be
            >>> # specified. Otherwise, this map call will also result in an error.
            >>> input_columns = ["col2", "col0"]
            >>> output_columns = ["mod2", "mod3", "mod5", "mod7"]
            >>>
            >>> # Propagate all columns to the child node in this order:
            >>> columns_order = ["col0", "col2", "mod2", "mod3", "mod5", "mod7", "col1"]
            >>> ds_mapped = ds_pyfunc.map(input_columns, operations, output_columns, columns_order)
            >>>
            >>> # Propagate some columns to the child node in this order:
            >>> columns_order = ["mod7", "mod3", "col1"]
            >>> ds_mapped = ds_pyfunc.map(input_columns, operations, output_columns, columns_order)
        """
        return MapDataset(self, input_columns, operations, output_columns, columns_order, num_parallel_workers)

    @check_filter
    def filter(self, predicate, input_columns=None, num_parallel_workers=1):
        """
        Filter dataset by predicate.

        Note:
             If input_columns not provided or empty, all columns will be used.

        Args:
            predicate: python callable which returns a boolean value.
            input_columns: (list[str]): List of names of the input columns, when
            default=None, the predicate will be applied on all columns in the dataset.
            num_parallel_workers (int, optional): Number of workers to process the Dataset
            in parallel (default=None).

        Returns:
            FilterDataset, dataset filter.

        Examples:
            >>> import mindspore.dataset as ds
            >>> # generator data(0 ~ 63)
            >>> # filter the data that greater than or equal to 11
            >>> dataset_f = dataset.filter(predicate=lambda data: data < 11, input_columns = ["data"])
        """
        return FilterDataset(self, predicate, input_columns, num_parallel_workers)

    @check_repeat
    def repeat(self, count=None):
        """
        Repeats this dataset count times. Repeat indefinitely if the count is None or -1.

        Note:
            The order of using repeat and batch reflects the number of batches. Recommend that
            repeat operation should be used after batch operation.
            If dataset_sink_mode is False, here repeat operation is invalid.
            If dataset_sink_mode is True, repeat count should be euqal to the epoch of training. Otherwise,
            errors could occur since the amount of data is not the amount training requires.

        Args:
            count (int): Number of times the dataset should be repeated (default=None).

        Returns:
            RepeatDataset, dataset repeated.

        Examples:
            >>> import mindspore.dataset as ds
            >>> # data is an instance of Dataset object.
            >>> # creates a dataset where the dataset is repeated for 50 epochs
            >>> repeated = data.repeat(50)
            >>>
            >>> # creates a dataset where each epoch is shuffled individually
            >>> shuffled_and_repeated = data.shuffle(10)
            >>> shuffled_and_repeated = shuffled_and_repeated.repeat(50)
            >>>
            >>> # creates a dataset where the dataset is first repeated for
            >>> # 50 epochs before shuffling. the shuffle operator will treat
            >>> # the entire 50 epochs as one big dataset.
            >>> repeat_and_shuffle = data.repeat(50)
            >>> repeat_and_shuffle = repeat_and_shuffle.shuffle(10)
        """
        if count == 1:
            return self
        return RepeatDataset(self, count)

    @check_skip
    def skip(self, count):
        """
        Skip the first N elements of this dataset.

        Args:
            count (int): Number of elements the dataset should be skipped.

        Returns:
            SkipDataset, dataset skipped.

        Examples:
            >>> import mindspore.dataset as ds
            >>> # data is an instance of Dataset object.
            >>> # creates a dataset which skips first 3 elements from data
            >>> data = data.skip(3)
        """
        return SkipDataset(self, count)

    @check_take
    def take(self, count=-1):
        """
        Takes at most given numbers of elements from the dataset.

        Note:
            1. If count is greater than the number of element in dataset or equal to -1,
            all the element in dataset will be taken.
            2. The order of using take and batch effects. If take before batch operation,
            then taken given number of rows, otherwise take given number of batches.

        Args:
            count (int, optional): Number of elements to be taken from the dataset (default=-1).

        Returns:
            TakeDataset, dataset taken.

        Examples:
            >>> import mindspore.dataset as ds
            >>> # data is an instance of Dataset object.
            >>> # creates a dataset where the dataset including 50 elements.
            >>> data = data.take(50)
        """
        if count == -1:
            return self
        return TakeDataset(self, count)

    @check_zip_dataset
    def zip(self, datasets):
        """
        Zips the datasets in the input tuple of datasets. Columns in the input datasets must not have the same name.

        Args:
            datasets (tuple or class Dataset): A tuple of datasets or a single class Dataset
                to be zipped together with this dataset.

        Returns:
            ZipDataset, dataset zipped.

        Examples:
            >>> import mindspore.dataset as ds
            >>> # ds1 and ds2 are instances of Dataset object
            >>> # creates a dataset which is the combination of ds1 and ds2
            >>> data = ds1.zip(ds2)
        """
        if isinstance(datasets, tuple):
            datasets = (self, *datasets)
        elif isinstance(datasets, Dataset):
            datasets = (self, datasets)
        else:
            raise TypeError("The zip function %s type error!" % (datasets))
        return ZipDataset(datasets)

    @check_rename
    def rename(self, input_columns, output_columns):
        """
        Renames the columns in input datasets.

        Args:
            input_columns (list[str]): list of names of the input columns.
            output_columns (list[str]): list of names of the output columns.

        Returns:
            RenameDataset, dataset renamed.

        Examples:
            >>> import mindspore.dataset as ds
            >>> # data is an instance of Dataset object.
            >>> input_columns = ["input_col1", "input_col2", "input_col3"]
            >>> output_columns = ["output_col1", "output_col2", "output_col3"]
            >>>
            >>> # creates a dataset where input_col1 is renamed to output_col1, and
            >>> # input_col2 is renamed to output_col2, and input_col3 is renamed
            >>> # to output_col3.
            >>> data = data.rename(input_columns=input_columns, output_columns=output_columns)
        """

        return RenameDataset(self, input_columns, output_columns)

    @check_project
    def project(self, columns):
        """
        Projects certain columns in input datasets.

        The specified columns will be selected from the dataset and passed down
        the pipeline in the order specified. The other columns are discarded.

        Args:
            columns(list[str]): list of names of the columns to project.

        Returns:
            ProjectDataset, dataset projected.

        Examples:
            >>> import mindspore.dataset as ds
            >>> # data is an instance of Dataset object
            >>> columns_to_project = ["column3", "column1", "column2"]
            >>>
            >>> # creates a dataset that consist of column3, column1, column2
            >>> # in that order, regardless of the original order of columns.
            >>> data = data.project(columns=columns_to_project)
        """

        return ProjectDataset(self, columns)

    def apply(self, apply_func):
        """
        Apply a function in this dataset.

        The specified apply_func is a function that must take one 'Dataset' as an argument
        and return a preprogressing 'Dataset'.

        Args:
            apply_func (function): A function that must take one 'Dataset' as an argument and
                                   return a preprogressing 'Dataset'.

        Returns:
            Dataset, applied by the function.

        Examples:
            >>> import mindspore.dataset as ds
            >>> # data is an instance of Dataset object
            >>> # declare an apply_func function which returns a Dataset object
            >>> def apply_func(ds):
            >>>     ds = ds.batch(2)
            >>>     return ds
            >>> # use apply to call apply_func
            >>> data = data.apply(apply_func)

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

    def device_que(self, prefetch_size=None):
        """
        Returns a transferredDataset that transfer data through device.

        Args:
            prefetch_size (int, optional): prefetch number of records ahead of the
                user's request (default=None).

        Note:
            If device is Ascend, features of data will be transferred one by one. The limitation
            of data transmission per time is 256M.

        Return:
            TransferDataset, dataset for transferring.
        """
        return self.to_device()

    def to_device(self, num_batch=None):
        """
        Transfers data through CPU, GPU or Ascend devices.

        Args:
            num_batch (int, optional): limit the number of batch to be sent to device (default=None).

        Note:
            If device is Ascend, features of data will be transferred one by one. The limitation
            of data transmission per time is 256M.

        Returns:
            TransferDataset, dataset for transferring.

        Raises:
            TypeError: If device_type is empty.
            ValueError: If device_type is not 'Ascend', 'GPU' or 'CPU'.
            ValueError: If num_batch is None or 0 or larger than int_max.
            RuntimeError: If dataset is unknown.
            RuntimeError: If distribution file path is given but failed to read.
        """
        if num_batch is None:
            num_batch = self.get_dataset_size()
            repeat_count = self.get_repeat_count()
            num_batch = num_batch * repeat_count

        queue_name = str(uuid.uuid1())

        if context:
            device_type = context.get_context("device_target")
        else:
            device_type = "CPU"

        if device_type == "":
            raise TypeError("Please set device_type in context")

        if device_type not in ('Ascend', 'GPU', 'CPU'):
            raise ValueError("only support CPU, Ascend, GPU")

        if num_batch is None or num_batch == 0:
            raise ValueError("num_batch is None or 0.")

        def get_distribution(output_dataset):
            dev_id = 0
            if isinstance(output_dataset, (StorageDataset, MindDataset)):
                return output_dataset.distribution, dev_id
            if isinstance(output_dataset, (Cifar10Dataset, Cifar100Dataset, GeneratorDataset, ImageFolderDatasetV2,
                                           ManifestDataset, MnistDataset, VOCDataset, CelebADataset)):
                sampler = output_dataset.sampler
                if isinstance(sampler, samplers.DistributedSampler):
                    dev_id = sampler.shard_id
                return "", dev_id
            if isinstance(output_dataset, TFRecordDataset):
                if output_dataset.shard_id is not None:
                    dev_id = output_dataset.shard_id
                return "", dev_id

            if not output_dataset.input:
                raise RuntimeError("Unknown output_dataset: {}".format(type(output_dataset)))
            input_dataset = output_dataset.input[0]
            return get_distribution(input_dataset)

        distribution_path, device_id = get_distribution(self)
        if distribution_path == "":
            return TransferDataset(self, queue_name, device_id, device_type, num_batch)
        try:
            with open(distribution_path, 'r') as distribution_f:
                dist = json.load(distribution_f)
                device_id = dist["deviceId"]
        except json.decoder.JSONDecodeError:
            raise RuntimeError("Json decode error when load distribution file")
        except Exception:
            raise RuntimeError("Distribution file failed to read")

        return TransferDataset(self, queue_name, device_id, device_type, num_batch)

    def create_tuple_iterator(self, columns=None):
        """
        Create an Iterator over the dataset. The data retrieved will be a list of ndarray of data.

        To specify which columns to list and the order needed, use columns_list. If columns_list
        is not provided, the order of the columns will not be changed.

        Args:
            columns (list[str], optional): List of columns to be used to specify the order of columns
                (defaults=None, means all columns).

        Returns:
            Iterator, list of ndarray.

        Examples:
            >>> import mindspore.dataset as ds
            >>> # data is an instance of Dataset object
            >>> # creates an iterator. The columns in the data obtained by the
            >>> # iterator will not be changed.
            >>> iterator = data.create_tuple_iterator()
            >>> for item in iterator:
            >>>     # convert the returned tuple to a list and print
            >>>     print(list(item))
        """
        return TupleIterator(self, columns)

    def create_dict_iterator(self):
        """
        Create an Iterator over the dataset.

        The data retrieved will be a dictionary. The order
        of the columns in the dictionary may not be the same as the original order.

        Returns:
            Iterator, dictionary of column_name-ndarray pair.

        Examples:
            >>> import mindspore.dataset as ds
            >>> # data is an instance of Dataset object
            >>> # creates an iterator. The columns in the data obtained by the
            >>> # iterator might be changed.
            >>> iterator = data.create_dict_iterator()
            >>> for item in iterator:
            >>>     # print the data in column1
            >>>     print(item["column1"])

        """
        return DictIterator(self)

    def __iter__(self):
        """Create an Iterator over the dataset."""
        return self.create_tuple_iterator()

    @staticmethod
    def read_dir(dir_path, schema, columns_list=None, num_parallel_workers=None,
                 deterministic_output=True, prefetch_size=None, shuffle=False, seed=None, distribution=""):
        """
        Append the path of all files in the dir_path to StorageDataset.

        Args:
            dir_path (str): Path to the directory that contains the dataset.
            schema (str): Path to the json schema file.
            columns_list (list[str], optional): List of columns to be read (default=None).
                If not provided, read all columns.
            num_parallel_workers (int, optional): Number of workers to process the Dataset in parallel
                (default=None).
            deterministic_output (bool, optional): Whether the result of this dataset can be reproduced
                or not (default=True). If True, performance might be affected.
            prefetch_size (int, optional): Prefetch number of records ahead of the
                user's request (default=None).
            shuffle (bool, optional): Shuffle the list of files in the directory (default=False).
            seed (int, optional): Create a random generator with a fixed seed. If set to None,
                create a random seed (default=None).
            distribution (str, optional): The path of distribution config file (default="").

        Returns:
            StorageDataset.

        Raises:
            ValueError: If dataset folder does not exist.
            ValueError: If dataset folder permission denied.
        """
        logger.warning("WARN_DEPRECATED: The usage of read_dir is deprecated, please use TFRecordDataset with GLOB.")

        list_files = []

        if not os.path.isdir(dir_path):
            raise ValueError("The dataset folder does not exist!")
        if not os.access(dir_path, os.R_OK):
            raise ValueError("The dataset folder permission denied!")

        for root, _, files in os.walk(dir_path):
            for file in files:
                list_files.append(os.path.join(root, file))

        list_files.sort()

        if shuffle:
            rand = random.Random(seed)
            rand.shuffle(list_files)

        return StorageDataset(list_files, schema, distribution, columns_list, num_parallel_workers,
                              deterministic_output, prefetch_size)

    @property
    def input_indexs(self):
        return self._input_indexs

    @input_indexs.setter
    def input_indexs(self, value):
        self._input_indexs = value

    def _get_pipeline_info(self):
        device_iter = TupleIterator(self)
        self._output_shapes = device_iter.get_output_shapes()
        self._output_types = device_iter.get_output_types()
        if self._dataset_size is None:
            self._dataset_size = device_iter.get_dataset_size()
        self._batch_size = device_iter.get_batch_size()
        self._num_classes = device_iter.num_classes()
        self._repeat_count = device_iter.get_repeat_count()
        device_iter.release()

    def output_shapes(self):
        """
        Get the shapes of output data.

        Return:
            List, list of shape of each column.
        """
        if self._output_shapes is None:
            self._get_pipeline_info()
        return self._output_shapes

    def output_types(self):
        """
        Get the types of output data.

        Return:
            List of data type.
        """
        if self._output_types is None:
            self._get_pipeline_info()
        return self._output_types

    def get_dataset_size(self):
        """
        Get the number of batches in an epoch.

        Return:
            Number, number of batches.
        """
        if self.input:
            return self.input[0].get_dataset_size()
        return None

    def num_classes(self):
        """
        Get the number of classes in a dataset.

        Return:
            Number, number of classes.
        """
        if self.input:
            return self.input[0].num_classes()
        return None

    def get_batch_size(self):
        """
        Get the size of a batch.

        Return:
            Number, the number of data in a batch.
        """
        if self.input:
            return self.input[0].get_batch_size()
        return 1

    def get_repeat_count(self):
        """
        Get the replication times in RepeatDataset else 1

        Return:
            Number, the count of repeat.
        """
        if self.input:
            return self.input[0].get_repeat_count()
        return 1

    def get_class_indexing(self):
        """
        Get the class index.

        Return:
            Dict, A str-to-int mapping from label name to index.
        """
        if self.input:
            return self.input[0].get_class_indexing()
        raise NotImplementedError("Dataset {} has not supported api get_class_indexing yet.".format(type(self)))

    def reset(self):
        """Reset the dataset for next epoch"""


class SourceDataset(Dataset):
    """
    Abstract class to represent a source dataset  which produces content to the data pipeline.
    """

    # No need for __init__ since it is the same as the super's init

    @staticmethod
    def _find_files(patterns):
        """
        Utility function to search for files with the given glob patterns.

        Args:
            patterns (str or list[str]): string or list of patterns to be searched.

        Returns:
            List, files.
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
            raise ValueError("The following patterns did not match any files: ", unmatched_patterns)

        if file_list:  # not empty
            return file_list
        raise ValueError("The list of path names matching the patterns is empty.")


class DatasetOp(Dataset):
    """
    Abstract class to represent a operations on dataset.
    """

    # No need for __init__ since it is the same as the super's init


class BatchDataset(DatasetOp):
    """
    The result of applying Batch operator to the input dataset.

    Args:
        input_dataset (Dataset): Input Dataset to be batched.
        batch_size (int): The size of the batch.
        drop_remainder (bool, optional): Whether drop the remainder batch of data (drop_remainder=False).
            If True, the last incomplete batch will be dropped.
    """

    def __init__(self, input_dataset, batch_size, drop_remainder=False, num_parallel_workers=None,
                 per_batch_map=None, input_columns=None):
        super().__init__(num_parallel_workers)

        if BatchDataset._is_ancestor_of_repeat(input_dataset):
            logger.warning("Repeat is located before batch, data from two epochs can be batched together.")

        self.batch_size = batch_size
        self.drop_remainder = drop_remainder
        self.per_batch_map = per_batch_map
        self.input_columns = input_columns
        self.input.append(input_dataset)
        input_dataset.output.append(self)
        self._input_indexs = input_dataset.input_indexs

    def get_args(self):
        args = super().get_args()
        args["batch_size"] = self.batch_size
        args["drop_remainder"] = self.drop_remainder
        args["per_batch_map"] = self.per_batch_map
        args["input_columns"] = self.input_columns
        return args

    def get_dataset_size(self):
        """
        Get the number of batches in an epoch.

        Return:
            Number, number of batches.
        """
        child_size = self.input[0].get_dataset_size()
        if child_size is not None:
            if self.drop_remainder:
                return math.floor(child_size / self.batch_size)
            return math.ceil(child_size / self.batch_size)
        return None

    def get_batch_size(self):
        """
        Get the size of a batch.

        Return:
            Number, the number of data in a batch.
        """
        return self.batch_size

    @staticmethod
    def _is_ancestor_of_repeat(dataset):
        """
        Utility function to find the case where repeat is used before batch.

        Args:
             dataset (Dataset): dataset to be checked
        Return:
            True or False
        """
        if isinstance(dataset, RepeatDataset):
            return True
        flag = False
        for input_dataset in dataset.input:
            flag = flag | BatchDataset._is_ancestor_of_repeat(input_dataset)
        return flag


class BatchInfo(CBatchInfo):
    """
    The information object associates with the current batch of tensors.
    """

    def get_batch_num(self):
        """
        Return the batch number of the current batch.

        Return:
            Number, number of the current batch.
        """
        return

    def get_epoch_num(self):
        """
        Return the epoch number of the current batch.

        Return:
            Number, number of the current epoch.
        """
        return


class ShuffleDataset(DatasetOp):
    """
    The result of applying Shuffle operator to the input Dataset.

    Args:
        input_dataset (Dataset): Input Dataset to be shuffled.
        buffer_size (int): The size of the buffer.
    """

    def __init__(self, input_dataset, buffer_size):
        super().__init__()
        self.buffer_size = buffer_size
        self.input.append(input_dataset)
        input_dataset.output.append(self)
        self._input_indexs = input_dataset.input_indexs

    def get_args(self):
        args = super().get_args()
        args["buffer_size"] = self.buffer_size
        return args


class MapDataset(DatasetOp):
    """
    The result of applying Map operator to the input Dataset.

    Args:
        input_dataset (Dataset): Input Dataset to be mapped.
        input_columns (list[str]): List of names of the input columns
            (default=None, the operations will be applied on the first columns in the dataset).
            The size of the list should match the number of inputs of the first operator.
        operations (TensorOp): A function mapping a nested structure of tensors
            to another nested structure of tensor (default=None).
        output_columns (list[str], optional): list of names of the output columns.
            The size of the list should match the number of outputs of the last operator
            (default=None, output columns will be the input columns, i.e., the columns will
            be replaced).
        columns_order (list[str], optional): list of all the desired columns of the dataset (default=None).
            The argument is mandatory if len(input_columns) != len(output_columns).
        num_parallel_workers (int, optional): Number of workers to process the Dataset
            in parallel (default=None).

        Raises:
            ValueError: If len(input_columns) != len(output_columns) and columns_order is not specified.
    """

    def __init__(self, input_dataset, input_columns=None, operations=None, output_columns=None, columns_order=None,
                 num_parallel_workers=None):
        super().__init__(num_parallel_workers)
        self.input.append(input_dataset)
        if input_columns is not None and not isinstance(input_columns, list):
            input_columns = [input_columns]
        self.input_columns = input_columns
        if operations is not None and not isinstance(operations, list):
            operations = [operations]
        self.operations = operations
        if output_columns is not None and not isinstance(output_columns, list):
            output_columns = [output_columns]
        self.output_columns = output_columns
        self.columns_order = columns_order

        if self.input_columns and self.output_columns \
                and len(self.input_columns) != len(self.output_columns) \
                and self.columns_order is None:
            raise ValueError("When (len(input_columns) != len(output_columns)), columns_order must be specified.")

        input_dataset.output.append(self)
        self._input_indexs = input_dataset.input_indexs

    def get_args(self):
        args = super().get_args()
        args["input_columns"] = self.input_columns
        args["operations"] = self.operations
        args["output_columns"] = self.output_columns
        return args

    def get_dataset_size(self):
        """
        Get the number of batches in an epoch.

        Return:
            Number, number of batches.
        """
        return self.input[0].get_dataset_size()


class FilterDataset(DatasetOp):
    """
    The result of applying filter predicate to the input Dataset.

    Args:
        input_dataset: Input Dataset to be mapped.
        predicate: python callable which returns a boolean value.
        input_columns: (list[str]): List of names of the input columns, when
        default=None, the predicate will be applied all columns in the dataset.
        num_parallel_workers (int, optional): Number of workers to process the Dataset
            in parallel (default=None).
    """

    def __init__(self, input_dataset, predicate, input_columns=None, num_parallel_workers=None):
        super().__init__(num_parallel_workers)
        self.predicate = lambda *args: bool(predicate(*args))
        self.input.append(input_dataset)
        input_dataset.output.append(self)
        if input_columns is not None and not isinstance(input_columns, list):
            input_columns = [input_columns]
        self.input_columns = input_columns

    def get_args(self):
        args = super().get_args()
        args["predicate"] = self.predicate
        args["input_columns"] = self.input_columns
        return args

    def get_dataset_size(self):
        """
        Get the number of batches in an epoch.
        the size cannot be determined before we run the pipeline
        Return:
            0
        """
        return 0


class RepeatDataset(DatasetOp):
    """
    The result of applying Repeat operator to the input Dataset.

    Args:
        input_dataset (Dataset): Input Dataset to be repeated.
        count (int): Number of times the dataset should be repeated.
    """

    def __init__(self, input_dataset, count):
        super().__init__()
        if count is None:
            self.count = -1
        else:
            self.count = count
        self.input.append(input_dataset)
        input_dataset.output.append(self)
        self._input_indexs = input_dataset.input_indexs

    def get_args(self):
        args = super().get_args()
        args["count"] = self.count
        return args

    def get_dataset_size(self):
        """
        Get the number of batches in an epoch.

        Return:
            Number, number of batches.
        """
        child_size = self.input[0].get_dataset_size()
        if child_size is not None:
            return child_size
        return None

    def get_repeat_count(self):
        """
        Get the replication times in RepeatDataset.

        Return:
            Number, the count of repeat.
        """
        return self.count


class SkipDataset(DatasetOp):
    """
    The result of applying Skip operator to the input Dataset.

    Args:
        datasets (tuple): A tuple of datasets to be skipped.
        count (int): Number of rows the dataset should be skipped.
    """

    def __init__(self, input_dataset, count):
        super().__init__()
        self.count = count
        self.input.append(input_dataset)
        input_dataset.output.append(self)
        self._input_indexs = input_dataset.input_indexs

    def get_args(self):
        args = super().get_args()
        args["count"] = self.count
        return args

    def get_dataset_size(self):
        """
        Get the number of batches in an epoch.

        Return:
            Number, number of batches.
        """
        child_size = self.input[0].get_dataset_size()
        output_size = 0
        if self.count >= 0 and self.count < child_size:
            output_size = child_size - self.count
        return output_size


class TakeDataset(DatasetOp):
    """
    The result of applying Take operator to the input Dataset.

    Args:
        input_dataset (Dataset): Input Dataset to be taken element from.
        count (int): Number of elements to be taken from the dataset.
    """

    def __init__(self, input_dataset, count):
        super().__init__()
        self.count = count
        self.input.append(input_dataset)
        input_dataset.output.append(self)
        self._input_indexs = input_dataset.input_indexs

    def get_args(self):
        args = super().get_args()
        args["count"] = self.count
        return args

    def get_dataset_size(self):
        """
        Get the number of batches in an epoch.

        Return:
            Number, number of batches.
        """
        child_size = self.input[0].get_dataset_size()
        if child_size < self.count:
            return child_size
        return self.count


class ZipDataset(DatasetOp):
    """
    The result of applying Zip operator to the input Dataset.

    Args:
        datasets (tuple): A tuple of datasets to be zipped together.

    Raises:
        TypeError: If dataset is not an instance of Dataset.
    """

    def __init__(self, datasets):
        super().__init__()
        for dataset in datasets:
            if not isinstance(dataset, Dataset):
                raise TypeError("The parameter %s of zip has type error!" % (dataset))
        self.datasets = datasets
        for data in datasets:
            self.input.append(data)
            data.output.append(self)

    def get_dataset_size(self):
        """
        Get the number of batches in an epoch.

        Return:
            Number, number of batches.
        """
        children_sizes = [c.get_dataset_size() for c in self.input]
        if all(c is not None for c in children_sizes):
            return min(children_sizes)
        return None

    def num_classes(self):
        """
        Get the number of classes in a dataset.

        Return:
            Number, number of classes.
        """
        return None

    def get_args(self):
        args = super().get_args()
        return args


class RenameDataset(DatasetOp):
    """
    The result of applying Rename operator to the input Dataset.

    Args:
        input_dataset (Dataset): Input Dataset to be Renamed.
        input_column_names (list[str]): list of names of the input columns.
        output_column_names (list[str]): list of names of the output columns.
    """

    def __init__(self, input_dataset, input_columns, output_columns):
        super().__init__()
        if not isinstance(input_columns, list):
            input_columns = [input_columns]
        if not isinstance(output_columns, list):
            output_columns = [output_columns]
        self.input_column_names = input_columns
        self.output_column_names = output_columns
        self.input.append(input_dataset)
        input_dataset.output.append(self)
        self._input_indexs = input_dataset.input_indexs

    def get_args(self):
        args = super().get_args()
        args["input_columns"] = self.input_column_names
        args["output_columns"] = self.output_column_names
        return args


class ProjectDataset(DatasetOp):
    """
    The result of applying Project operator to the input Dataset.

    Args:
        input_dataset (Dataset): Input Dataset to be Project.
        columns (list[str]): List of names of the columns to project.
        prefetch_size (int, optional): Prefetch number of records ahead of the
            user's request (default=None).
    """

    def __init__(self, input_dataset, columns, prefetch_size=None):
        super().__init__()
        if not isinstance(columns, list):
            columns = [columns]
        self.columns = columns
        self.input.append(input_dataset)
        self.prefetch_size = prefetch_size

        input_dataset.output.append(self)
        self._input_indexs = input_dataset.input_indexs

    def get_args(self):
        args = super().get_args()
        args["columns"] = self.columns
        args["prefetch_size"] = self.prefetch_size
        return args


class TransferDataset(DatasetOp):
    """
    The result of applying TDT operator to the input Dataset.

    Args:
        input_dataset (Dataset): Input Dataset to be transferred.
        queue_name (str): Name of device queue.
        device_id (int): Id of device.
        device_type (str): Type of device, including "CPU", "GPU", and "Ascend".
        num_batch (int): limit the number of batch to be sent to device (default=None).
    """

    def __init__(self, input_dataset, queue_name, device_id, device_type, num_batch=None):
        super().__init__()
        self.input.append(input_dataset)
        input_dataset.output.append(self)
        self.queue_name = queue_name
        self._input_indexs = input_dataset.input_indexs
        self._device_type = device_type
        self._device_id = device_id
        self.__num_batch = num_batch
        self.iterator = None

    def get_args(self):
        args = super().get_args()
        args["queue_name"] = self.queue_name
        args["device_type"] = self._device_type
        args["device_id"] = self._device_id
        args["num_batch"] = self.__num_batch
        return args

    def create_dict_iterator(self):
        raise RuntimeError("TransferDataset is not iterable")

    def create_tuple_iterator(self, columns=None):
        raise RuntimeError("TransferDataset is not iterable")

    def __iter__(self):
        raise RuntimeError("TransferDataset is not iterable")

    def output_shapes(self):
        raise RuntimeError("TransferDataset does not support output_shapes")

    def output_types(self):
        raise RuntimeError("TransferDataset does not support output_types")

    def send(self):
        # need to keep iterator alive so the executionTree is not destroyed
        self.iterator = TupleIterator(self)


class StorageDataset(SourceDataset):
    """
    A source dataset that reads and parses datasets stored on disk in various formats, including TFData format.

    Args:
        dataset_files (list[str]): List of files to be read.
        schema (str): Path to the json schema file.
        distribution (str, optional): Path of distribution config file (default="").
        columns_list (list[str], optional): List of columns to be read (default=None, read all columns).
        num_parallel_workers (int, optional): Number of parallel working threads (default=None).
        deterministic_output (bool, optional): Whether the result of this dataset can be reproduced
                or not (default=True). If True, performance might be affected.
        prefetch_size (int, optional): Prefetch number of records ahead of the user's request (default=None).

    Raises:
        RuntimeError: If schema file failed to read.
        RuntimeError: If distribution file path is given but failed to read.
    """

    @check
    def __init__(self, dataset_files, schema, distribution="", columns_list=None, num_parallel_workers=None,
                 deterministic_output=None, prefetch_size=None):
        super().__init__(num_parallel_workers)
        logger.warning("WARN_DEPRECATED: The usage of StorageDataset is deprecated, please use TFRecordDataset.")
        self.dataset_files = dataset_files
        try:
            with open(schema, 'r') as load_f:
                json.load(load_f)
        except json.decoder.JSONDecodeError:
            raise RuntimeError("Json decode error when load schema file")
        except Exception:
            raise RuntimeError("Schema file failed to load")

        if distribution != "":
            try:
                with open(distribution, 'r') as load_d:
                    json.load(load_d)
            except json.decoder.JSONDecodeError:
                raise RuntimeError("Json decode error when load distribution file")
            except Exception:
                raise RuntimeError("Distribution file failed to load")
        if self.dataset_files is None:
            schema = None
            distribution = None
        self.schema = schema
        self.distribution = distribution
        self.columns_list = columns_list
        self.deterministic_output = deterministic_output
        self.prefetch_size = prefetch_size

    def get_args(self):
        args = super().get_args()
        args["dataset_files"] = self.dataset_files
        args["schema"] = self.schema
        args["distribution"] = self.distribution
        args["columns_list"] = self.columns_list
        args["deterministic_output"] = self.deterministic_output
        args["prefetch_size"] = self.prefetch_size
        return args

    def get_dataset_size(self):
        """
        Get the number of batches in an epoch.

        Return:
            Number, number of batches.
        """
        if self._dataset_size is None:
            self._get_pipeline_info()
        return self._dataset_size

    # manually set dataset_size as a temporary solution.
    def set_dataset_size(self, value):
        logger.warning("WARN_DEPRECATED: This method is deprecated. Please use get_dataset_size directly.")
        if value >= 0:
            self._dataset_size = value
        else:
            raise ValueError('set dataset_size with negative value {}'.format(value))

    def num_classes(self):
        """
        Get the number of classes in dataset.

        Return:
            Number, number of classes.

        Raises:
            ValueError: If dataset type is invalid.
            ValueError: If dataset is not Imagenet dataset or manifest dataset.
            RuntimeError: If schema file is given but failed to load.
        """
        cur_dataset = self
        while cur_dataset.input:
            cur_dataset = cur_dataset.input[0]
        if not hasattr(cur_dataset, "schema"):
            raise ValueError("Dataset type is invalid")
        # Only IMAGENET/MANIFEST support numclass
        try:
            with open(cur_dataset.schema, 'r') as load_f:
                load_dict = json.load(load_f)
        except json.decoder.JSONDecodeError:
            raise RuntimeError("Json decode error when load schema file")
        except Exception:
            raise RuntimeError("Schema file failed to load")
        if load_dict["datasetType"] != "IMAGENET" and load_dict["datasetType"] != "MANIFEST":
            raise ValueError("%s dataset does not support num_classes!" % (load_dict["datasetType"]))

        if self._num_classes is None:
            self._get_pipeline_info()
        return self._num_classes


class RangeDataset(SourceDataset):
    """
    A source dataset that reads and parses datasets stored on disk in a range.

    Args:
        start (int): starting index.
        stop (int): ending index.
        step (int): step size in a range.
    """

    def __init__(self, start, stop, step):
        super().__init__()
        self.start = start
        self.stop = stop
        self.step = step

    def get_args(self):
        args = super().get_args()
        args["start"] = self.start
        args["stop"] = self.stop
        args["step"] = self.step
        return args


def _select_sampler(num_samples, input_sampler, shuffle, num_shards, shard_id):
    """
    Create sampler based on user input.

    Args:
        num_samples (int): Number of samples
        input_sampler (Iterable / Sampler): Sampler from user
        shuffle (bool): Shuffle
        num_shards (int): Number of shard for sharding
        shard_id (int): Shard ID
    """
    if shuffle is None:
        if input_sampler is not None:
            # If shuffle is not specified, user provided sampler, use user's sampler
            return input_sampler
        if num_shards is not None:
            # If shuffle is not specified, sharding enabled, use distributed random sampler
            shuffle = True
            return samplers.DistributedSampler(num_shards, shard_id, shuffle=shuffle)
        # If shuffle is not specified, sharding disabled, use random sampler
        if num_samples is not None:
            return samplers.RandomSampler(replacement=True, num_samples=num_samples)
        return samplers.RandomSampler()
    if shuffle is True:
        if num_shards is not None:
            # If shuffle enabled, sharding enabled, use distributed random sampler
            return samplers.DistributedSampler(num_shards, shard_id, shuffle=shuffle)
        # If shuffle enabled, sharding disabled, use random sampler
        if num_samples is not None:
            return samplers.RandomSampler(replacement=True, num_samples=num_samples)
        return samplers.RandomSampler()
    if num_shards is not None:
        # If shuffle disabled, sharding enabled, use distributed sequential sampler
        return samplers.DistributedSampler(num_shards, shard_id, shuffle=shuffle)
    # If shuffle disabled, sharding disabled, use sequential sampler
    return samplers.SequentialSampler()


class ImageFolderDatasetV2(SourceDataset):
    """
    A source dataset that reads images from a tree of directories.

    All images within one folder have the same label.
    The generated dataset has two columns ['image', 'label'].
    The shape of the image column is [image_size] if decode flag is False, or [H,W,C]
    otherwise.
    The type of the image tensor is uint8. The label is just a scalar uint64
    tensor.
    This dataset can take in a sampler. sampler and shuffle are mutually exclusive. Table
    below shows what input args are allowed and their expected behavior.

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
        decode (bool, optional): decode the images after reading (default=False).
        num_shards (int, optional): Number of shards that the dataset should be divided
            into (default=None).
        shard_id (int, optional): The shard ID within num_shards (default=None). This
            argument should be specified only when num_shards is also specified.

    Raises:
        RuntimeError: If sampler and shuffle are specified at the same time.
        RuntimeError: If sampler and sharding are specified at the same time.
        RuntimeError: If num_shards is specified but shard_id is None.
        RuntimeError: If shard_id is specified but num_shards is None.
        RuntimeError: If class_indexing is not a dictionary.
        ValueError: If shard_id is invalid (< 0 or >= num_shards).

    Examples:
        >>> import mindspore.dataset as ds
        >>> # path to imagefolder directory. This directory needs to contain sub-directories which contain the images
        >>> dataset_dir = "/path/to/imagefolder_directory"
        >>> # 1) read all samples (image files) in dataset_dir with 8 threads
        >>> imagefolder_dataset = ds.ImageFolderDatasetV2(dataset_dir, num_parallel_workers=8)
        >>> # 2) read all samples (image files) from folder cat and folder dog with label 0 and 1
        >>> imagefolder_dataset = ds.ImageFolderDatasetV2(dataset_dir,class_indexing={"cat":0,"dog":1})
        >>> # 3) read all samples (image files) in dataset_dir with extensions .JPEG and .png (case sensitive)
        >>> imagefolder_dataset = ds.ImageFolderDatasetV2(dataset_dir, extensions={".JPEG",".png"})
    """

    @check_imagefolderdatasetv2
    def __init__(self, dataset_dir, num_samples=None, num_parallel_workers=None,
                 shuffle=None, sampler=None, extensions=None, class_indexing=None,
                 decode=False, num_shards=None, shard_id=None):
        super().__init__(num_parallel_workers)

        self.dataset_dir = dataset_dir
        self.sampler = _select_sampler(num_samples, sampler, shuffle, num_shards, shard_id)
        self.num_samples = num_samples
        self.shuffle_level = shuffle
        self.extensions = extensions
        self.class_indexing = class_indexing
        self.decode = decode
        self.num_shards = num_shards
        self.shard_id = shard_id

    def get_args(self):
        args = super().get_args()
        args["dataset_dir"] = self.dataset_dir
        args["num_samples"] = self.num_samples
        args["sampler"] = self.sampler
        args["shuffle"] = self.shuffle_level
        args["extensions"] = self.extensions
        args["class_indexing"] = self.class_indexing
        args["decode"] = self.decode
        args["num_shards"] = self.num_shards
        args["shard_id"] = self.shard_id
        return args

    def get_dataset_size(self):
        """
        Get the number of batches in an epoch.

        Return:
            Number, number of batches.
        """
        if self.num_samples is None:
            num_samples = 0
        else:
            num_samples = self.num_samples
        num_rows = ImageFolderOp.get_num_rows_and_classes(self.dataset_dir, num_samples)[0]

        return get_num_rows(num_rows, self.num_shards)

    def num_classes(self):
        """
        Get the number of classes in dataset.

        Return:
            Number, number of classes.
        """
        if self.num_samples is None:
            num_samples = 0
        else:
            num_samples = self.num_samples
        return ImageFolderOp.get_num_rows_and_classes(self.dataset_dir, num_samples)[1]


class MnistDataset(SourceDataset):
    """
    A source dataset for reading and parsing the Mnist dataset.

    The generated dataset has two columns ['image', 'label'].
    The type of the image tensor is uint8. The label is just a scalar uint32 tensor.
    This dataset can take in a sampler. sampler and shuffle are mutually exclusive. Table
    below shows what input args are allowed and their expected behavior.

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
            (default=value, set in the config).
        shuffle (bool, optional): Whether or not to perform shuffle on the dataset
            (default=None, expected order behavior shown in the table).
        sampler (Sampler, optional): Object used to choose samples from the
            dataset (default=None, expected order behavior shown in the table).
        num_shards (int, optional): Number of shards that the dataset should be divided
            into (default=None).
        shard_id (int, optional): The shard ID within num_shards (default=None). This
            argument should be specified only when num_shards is also specified.

    Raises:
        RuntimeError: If sampler and shuffle are specified at the same time.
        RuntimeError: If sampler and sharding are specified at the same time.
        RuntimeError: If num_shards is specified but shard_id is None.
        RuntimeError: If shard_id is specified but num_shards is None.
        ValueError: If shard_id is invalid (< 0 or >= num_shards).

    Examples:
        >>> import mindspore.dataset as ds
        >>> dataset_dir = "/path/to/mnist_folder"
        >>> # 1) read 3 samples from mnist_dataset
        >>> mnist_dataset = ds.MnistDataset(dataset_dir=dataset_dir, num_samples=3)
        >>> # in mnist_dataset dataset, each dictionary has keys "image" and "label"
    """

    @check_mnist_cifar_dataset
    def __init__(self, dataset_dir, num_samples=None, num_parallel_workers=None,
                 shuffle=None, sampler=None, num_shards=None, shard_id=None):
        super().__init__(num_parallel_workers)

        self.dataset_dir = dataset_dir
        self.sampler = _select_sampler(num_samples, sampler, shuffle, num_shards, shard_id)
        self.num_samples = num_samples
        self.shuffle_level = shuffle
        self.num_shards = num_shards
        self.shard_id = shard_id

    def get_args(self):
        args = super().get_args()
        args["dataset_dir"] = self.dataset_dir
        args["num_samples"] = self.num_samples
        args["shuffle"] = self.shuffle_level
        args["sampler"] = self.sampler
        args["num_shards"] = self.num_shards
        args["shard_id"] = self.shard_id
        return args

    def get_dataset_size(self):
        """
        Get the number of batches in an epoch.

        Return:
            Number, number of batches.
        """
        if self.num_samples is None:
            num_samples = 0
        else:
            num_samples = self.num_samples

        num_rows = MnistOp.get_num_rows(self.dataset_dir, num_samples)

        return get_num_rows(num_rows, self.num_shards)


class MindDataset(SourceDataset):
    """
    A source dataset that reads from shard files and database.

    Args:
        dataset_file (str): one of file names in dataset.
        columns_list (list[str], optional): List of columns to be read (default=None).
        num_parallel_workers (int, optional): The number of readers (default=None).
        shuffle (bool, optional): Whether or not to perform shuffle on the dataset
            (default=None, performs shuffle).
        num_shards (int, optional): Number of shards that the dataset should be divided into (default=None).
        shard_id (int, optional): The shard ID within num_shards (default=None). This
            argument should be specified only when num_shards is also specified.
        block_reader (bool, optional): Whether read data by block mode (default=False).
        sampler (Sampler, optional): Object used to choose samples from the
            dataset (default=None, sampler is exclusive
            with shuffle and block_reader). Support list: SubsetRandomSampler,
            PkSampler

    Raises:
        ValueError: If num_shards is specified but shard_id is None.
        ValueError: If shard_id is specified but num_shards is None.
        ValueError: If block reader is true but partition is specified.
    """

    @check_minddataset
    def __init__(self, dataset_file, columns_list=None, num_parallel_workers=None,
                 shuffle=None, num_shards=None, shard_id=None,
                 block_reader=False, sampler=None):
        super().__init__(num_parallel_workers)
        self.dataset_file = dataset_file
        self.columns_list = columns_list
        self.global_shuffle = shuffle
        self.distribution = ""
        self.sampler = sampler

        if num_shards is None or shard_id is None:
            self.partitions = None
        else:
            self.partitions = [num_shards, shard_id]

        if block_reader is True and self.partitions is not None:
            raise ValueError("block reader not allowed true when use partitions")

        if block_reader is True and shuffle is True:
            raise ValueError("block reader not allowed true when use shuffle")

        if block_reader is True:
            logger.warning("WARN: global shuffle is not used.")

        if sampler is not None:
            if isinstance(sampler, samplers.SubsetRandomSampler) is False and \
            isinstance(sampler, samplers.PKSampler) is False:
                raise ValueError("the sampler is not supported yet.")

        # sampler exclusive
        if block_reader is True and sampler is not None:
            raise ValueError("block reader not allowed true when use sampler")

        if shuffle is True and sampler is not None:
            raise ValueError("shuffle not allowed true when use sampler")

        if block_reader is False and sampler is None:
            self.global_shuffle = not bool(shuffle is False)

        self.num_shards = num_shards
        self.shard_id = shard_id
        self.block_reader = block_reader

    def get_args(self):
        args = super().get_args()
        args["dataset_file"] = self.dataset_file
        args["columns_list"] = self.columns_list
        args["global_shuffle"] = self.global_shuffle
        args["partitions"] = self.partitions
        args["block_reader"] = self.block_reader
        args["num_shards"] = self.num_shards
        args["shard_id"] = self.shard_id
        args["sampler"] = self.sampler
        return args

    def get_dataset_size(self):
        """
        Get the number of batches in an epoch.

        Return:
            Number, number of batches.
        """

        num_rows = MindRecordOp.get_num_rows(self.dataset_file, self.sampler)
        if self.partitions is not None and self.partitions[0] > 0:
            if num_rows % self.partitions[0] == 0:
                num_rows = num_rows // self.partitions[0]
            else:
                num_rows = num_rows // self.partitions[0] + 1
        return num_rows


def _iter_fn(dataset, num_samples):
    """
    Generator function wrapper for iterable dataset
    """
    if num_samples is not None:
        ds_iter = iter(dataset)
        for _ in range(num_samples):
            try:
                val = next(ds_iter)
            except StopIteration:
                return
            # convert output tensors to ndarrays
            yield tuple([np.array(x) for x in val])
    else:
        for val in dataset:
            # convert output tensors to ndarrays
            yield tuple([np.array(x) for x in val])


def _generator_fn(generator, num_samples):
    """
    Generator function wrapper for generator function dataset
    """
    if num_samples is not None:
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


def _py_sampler_fn(sampler, num_samples, dataset):
    """
    Generator function wrapper for mappable dataset with python sampler
    """
    if num_samples is not None:
        sampler_iter = iter(sampler)
        for _ in range(num_samples):
            try:
                idx = next(sampler_iter)
            except StopIteration:
                return
            val = dataset[idx]
            # convert output tensors to ndarrays
            yield tuple([np.array(x) for x in val])
    else:
        for i in sampler:
            val = dataset[i]
            # convert output tensors to ndarrays
            yield tuple([np.array(x) for x in val])


def _cpp_sampler_fn(sampler, dataset):
    """
    Generator function wrapper for mappable dataset with cpp sampler
    """
    indices = sampler.get_indices()
    for i in indices:
        val = dataset[i]
        # convert output tensors to ndarrays
        yield tuple([np.array(x) for x in val])


class GeneratorDataset(SourceDataset):
    """
    A source dataset that generate data from python by invoking python data source each epoch.

    This dataset can take in a sampler. sampler and shuffle are mutually exclusive. Table
    below shows what input args are allowed and their expected behavior.

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
        source (Callable/Iterable/Random Accessible):
            A generator callable object, an iterable python object or a random accessible python object.
            Callable source is required to return a tuple of numpy array as a row of the dataset on source().next().
            Iterable source is required to return a tuple of numpy array as a row of the dataset on iter(source).next().
            Random accessible source is required to return a tuple of numpy array as a row of the dataset on
            source[idx].
        column_names (list[str]): List of column names of the dataset.
        column_types (list[mindspore.dtype], optional): List of column data types of the dataset (default=None).
            If provided, sanity check will be performed on generator output.
        schema (Schema/String, optional): Path to the json schema file or schema object (default=None).
            If the schema is not provided, the meta data from column_names and column_types is considered the schema.
        num_samples (int, optional): The number of samples to be included in the dataset
            (default=None, all images).
        shuffle (bool, optional): Whether or not to perform shuffle on the dataset. Random accessible input is required.
            (default=None, expected order behavior shown in the table).
        sampler (Sampler/Iterable, optional): Object used to choose samples from the dataset. Random accessible input is
        required.
            (default=None, expected order behavior shown in the table).
        num_shards (int, optional): Number of shards that the dataset should be divided into (default=None).
            This argument should be specified only when 'num_samples' is "None". Random accessible input is required.
        shard_id (int, optional): The shard ID within num_shards (default=None). This argument should be specified only
            when num_shards is also specified. Random accessible input is required.

    Examples:
        >>> import mindspore.dataengine as de
        >>> # 1) Multidimensional generator function as callable input
        >>> def generator_md():
        >>>     for i in range(64):
        >>>         yield (np.array([[i, i + 1], [i + 2, i + 3]]),)
        >>> # create multi_dimension_generator_dataset with GeneratorMD and column name "multi_dimensional_data"
        >>> multi_dimension_generator_dataset = de.GeneratorDataset(generator_md, ["multi_dimensional_data"])
        >>> # 2) Multi-column generator function as callable input
        >>> def generator_mc(maxid = 64):
        >>>     for i in range(maxid):
        >>>         yield (np.array([i]), np.array([[i, i + 1], [i + 2, i + 3]]))
        >>> # create multi_column_generator_dataset with GeneratorMC and column names "col1" and "col2"
        >>> multi_column_generator_dataset = de.GeneratorDataset(generator_mc, ["col1", "col2"])
        >>> # 3) Iterable dataset as iterable input
        >>> class MyIterable():
        >>>     def __iter__(self):
        >>>         return # User implementation
        >>> # create iterable_generator_dataset with MyIterable object
        >>> iterable_generator_dataset = de.GeneratorDataset(MyIterable(), ["col1"])
        >>> # 4) Random accessible dataset as Random accessible input
        >>> class MyRA():
        >>>     def __getitem__(self, index):
        >>>         return # User implementation
        >>> # create ra_generator_dataset with MyRA object
        >>> ra_generator_dataset = de.GeneratorDataset(MyRA(), ["col1"])
        >>> # List/Dict/Tuple is also random accessible
        >>> list_generator = de.GeneratorDataset([(np.array(0),), (np.array(1)), (np.array(2))], ["col1"])
        >>> # 5) Built-in Sampler
        >>> my_generator = de.GeneratorDataset(my_ds, ["img", "label"], sampler=samplers.RandomSampler())
        >>>
    """

    @check_generatordataset
    def __init__(self, source, column_names, column_types=None, schema=None, num_samples=None, num_parallel_workers=1,
                 shuffle=None, sampler=None, num_shards=None, shard_id=None):
        super().__init__(num_parallel_workers)
        self.sampler = _select_sampler(num_samples, sampler, shuffle, num_shards, shard_id)
        if self.sampler is not None and hasattr(source, "__getitem__"):
            if isinstance(self.sampler, (samplers.SequentialSampler, samplers.DistributedSampler,
                                         samplers.RandomSampler, samplers.SubsetRandomSampler,
                                         samplers.WeightedRandomSampler, samplers.Sampler)):
                if num_samples is None:
                    num_samples = len(source)
                sampler_instance = self.sampler.create()
                sampler_instance.set_num_rows(len(source))
                sampler_instance.set_num_samples(num_samples)
                sampler_instance.initialize()
                self.source = (lambda: _cpp_sampler_fn(sampler_instance, source))
            else:
                self.source = (lambda: _py_sampler_fn(self.sampler, num_samples, source))
        else:
            try:
                iter(source)
            except TypeError:
                # Use generator function if input callable
                self.source = (lambda: _generator_fn(source, num_samples))
            else:
                # Use iterator function if input is iterable
                # Random accessible input is also iterable
                self.source = (lambda: _iter_fn(source, num_samples))

        self.column_names = column_names

        if column_types is not None:
            self.column_types = mstypelist_to_detypelist(column_types)
        else:
            self.column_types = column_types

    def get_args(self):
        args = super().get_args()
        args["source"] = self.source
        args["column_names"] = self.column_names
        args["column_types"] = self.column_types
        return args

    def get_dataset_size(self):
        """
        Get the number of batches in an epoch.

        Return:
            Number, number of batches.
        """
        return self._dataset_size

    # manually set dataset_size as a temporary solution.
    def set_dataset_size(self, value):
        if value >= 0:
            self._dataset_size = value
        else:
            raise ValueError('set dataset_size with negative value {}'.format(value))


class TFRecordDataset(SourceDataset):
    """
    A source dataset that reads and parses datasets stored on disk in TFData format.

    Args:
        dataset_files (str or list[str]): String or list of files to be read or glob strings to search for a pattern of
            files. The list will be sorted in a lexicographical order.
        schema (str or Schema, optional): Path to the json schema file or schema object (default=None).
            If the schema is not provided, the meta data from the TFData file is considered the schema.
        columns_list (list[str], optional): List of columns to be read (default=None, read all columns)
        num_samples (int, optional): number of samples(rows) to read (default=None, reads the full dataset).
        num_parallel_workers (int, optional): number of workers to read the data
            (default=None, number set in the config).
        shuffle (bool, Shuffle level, optional): perform reshuffling of the data every epoch (default=Shuffle.GLOBAL).
            If shuffle is False, no shuffling will be performed;
            If shuffle is True, the behavior is the same as setting shuffle to be Shuffle.GLOBAL
            Otherwise, there are two levels of shuffling:

            - Shuffle.GLOBAL: Shuffle both the files and samples.

            - Shuffle.FILES: Shuffle files only.

        num_shards (int, optional): Number of shards that the dataset should be divided
            into (default=None).
        shard_id (int, optional): The shard ID within num_shards (default=None). This
            argument should be specified only when num_shards is also specified.
        shard_equal_rows (bool): Get equal rows for all shards(default=False). If shard_equal_rows is false, number
            of rows of each shard may be not equal.
    Examples:
        >>> import mindspore.dataset as ds
        >>> import mindspore.common.dtype as mstype
        >>> dataset_files = ["/path/to/1", "/path/to/2"] # contains 1 or multiple tf data files
        >>> # 1) get all rows from dataset_files with no explicit schema:
        >>> # The meta-data in the first row will be used as a schema.
        >>> tfdataset = ds.TFRecordDataset(dataset_files=dataset_files)
        >>> # 2) get all rows from dataset_files with user-defined schema:
        >>> schema = ds.Schema()
        >>> schema.add_column('col_1d', de_type=mindspore.int64, shape=[2])
        >>> tfdataset = ds.TFRecordDataset(dataset_files=dataset_files, schema=schema)
        >>> # 3) get all rows from dataset_files with schema file "./schema.json":
        >>> tfdataset = ds.TFRecordDataset(dataset_files=dataset_files, schema="./schema.json")
    """
    @check_tfrecorddataset
    def __init__(self, dataset_files, schema=None, columns_list=None, num_samples=None, num_parallel_workers=None,
                 shuffle=Shuffle.GLOBAL, num_shards=None, shard_id=None, shard_equal_rows=False):
        super().__init__(num_parallel_workers)
        self.dataset_files = self._find_files(dataset_files)
        self.dataset_files.sort()
        self.num_shards = num_shards
        self.shard_id = shard_id
        schema_obj = None
        if (schema is not None) and (not isinstance(schema, Schema)):
            schema_obj = Schema(schema)  # read the schema file and convert to schema object to validate it
        self.schema = schema
        self.columns_list = columns_list
        self.num_samples = num_samples
        if schema_obj is not None and num_samples is None:
            self.num_samples = schema_obj.num_rows

        if not isinstance(shuffle, (bool, Shuffle)):
            raise TypeError("shuffle should be of boolean or enum 'Shuffle'.")
        if not isinstance(shuffle, Shuffle):
            if shuffle:
                self.shuffle_level = Shuffle.GLOBAL
                self.shuffle_files = True
            else:
                self.shuffle_level = None
                self.shuffle_files = False
        else:
            self.shuffle_level = shuffle
            self.shuffle_files = True
        self.shard_equal_rows = shard_equal_rows

    def get_args(self):
        args = super().get_args()
        args["dataset_files"] = self.dataset_files
        if self.schema is not None:
            if isinstance(self.schema, Schema):
                self.schema.datasetType = 'TF'
                if self.num_samples is not None:
                    self.schema.num_rows = self.num_samples
                args["schema_json_string"] = self.schema.to_json()
            else:
                args["schema_file_path"] = self.schema
        args["schema"] = self.schema
        args["columns_list"] = self.columns_list
        args["num_samples"] = self.num_samples
        if self.shuffle_files is not None:
            args["shuffle_files"] = self.shuffle_files
        args["shuffle"] = self.shuffle_level
        args["num_shards"] = self.num_shards
        args["shard_id"] = self.shard_id
        args["shard_equal_rows"] = self.shard_equal_rows
        return args

    def get_dataset_size(self, estimate=False):
        """
        Get the number of batches in an epoch.

        Args:
            estimate (bool, optional): Fast estimation of the dataset size instead of a full scan.

        Return:
            Number, number of batches.
        """
        if self._dataset_size is None:
            num_rows = TFReaderOp.get_num_rows(self.dataset_files, 8, estimate)
            num_rows = get_num_rows(num_rows, self.num_shards)
            if self.num_samples is None:
                return num_rows
            return min(self.num_samples, num_rows)
        return self._dataset_size

    # manually set dataset_size as a tempoary solution.
    def set_dataset_size(self, value):
        logger.warning("WARN_DEPRECATED: This method is deprecated. Please use get_dataset_size directly.")
        if value >= 0:
            self._dataset_size = value
        else:
            raise ValueError('set dataset_size with negative value {}'.format(value))


class ManifestDataset(SourceDataset):
    """
    A source dataset that reads images from a manifest file.

    The generated dataset has two columns ['image', 'label'].
    The shape of the image column is [image_size] if decode flag is False, or [H,W,C]
    otherwise.
    The type of the image tensor is uint8. The label is just a scalar uint64
    tensor.
    This dataset can take in a sampler. sampler and shuffle are mutually exclusive. Table
    below shows what input args are allowed and their expected behavior.

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
        dataset_file (str): File to be read.
        usage (str, optional): Need train, eval or inference data (default="train").
        num_samples (int, optional): The number of images to be included in the dataset.
            (default=None, all images).
        num_parallel_workers (int, optional): Number of workers to read the data
            (default=None, number set in the config).
        shuffle (bool, optional): Whether to perform shuffle on the dataset (default=None, expected
            order behavior shown in the table).
        sampler (Sampler, optional): Object used to choose samples from the
            dataset (default=None, expected order behavior shown in the table).
        class_indexing (dict, optional): A str-to-int mapping from label name to index
            (default=None, the folder names will be sorted alphabetically and each
            class will be given a unique index starting from 0).
        decode (bool, optional): decode the images after reading (defaults=False).
        num_shards (int, optional): Number of shards that the dataset should be divided
            into (default=None).
        shard_id (int, optional): The shard ID within num_shards (default=None). This
            argument should be specified only when num_shards is also specified.

    Raises:
        RuntimeError: If sampler and shuffle are specified at the same time.
        RuntimeError: If sampler and sharding are specified at the same time.
        RuntimeError: If num_shards is specified but shard_id is None.
        RuntimeError: If shard_id is specified but num_shards is None.
        RuntimeError: If class_indexing is not a dictionary.
        ValueError: If shard_id is invalid (< 0 or >= num_shards).

    Examples:
        >>> import mindspore.dataset as ds
        >>> dataset_file = "/path/to/manifest_file.manifest"
        >>> # 1) read all samples specified in manifest_file dataset with 8 threads for training:
        >>> manifest_dataset = ds.ManifestDataset(dataset_file, usage="train", num_parallel_workers=8)
        >>> # 2) reads samples (specified in manifest_file.manifest) for shard 0 in a 2-way distributed training setup:
        >>> manifest_dataset = ds.ManifestDataset(dataset_file, num_shards=2, shard_id=0)

    """

    @check_manifestdataset
    def __init__(self, dataset_file, usage="train", num_samples=None, num_parallel_workers=None,
                 shuffle=None, sampler=None, class_indexing=None, decode=False, num_shards=None, shard_id=None):
        super().__init__(num_parallel_workers)

        self.dataset_file = dataset_file
        self.sampler = _select_sampler(num_samples, sampler, shuffle, num_shards, shard_id)

        if class_indexing is not None and not isinstance(class_indexing, dict):
            raise RuntimeError("class_indexing should be a dictionary.")

        self.num_samples = num_samples
        self.class_indexing = class_indexing
        self.decode = decode
        self.usage = usage
        self.shuffle_level = shuffle
        self.num_shards = num_shards
        self.shard_id = shard_id

    def get_args(self):
        args = super().get_args()
        args["dataset_file"] = self.dataset_file
        args["usage"] = self.usage
        args["num_samples"] = self.num_samples
        args["shuffle"] = self.shuffle_level
        args["sampler"] = self.sampler
        args["class_indexing"] = self.class_indexing
        args["decode"] = self.decode
        args["num_shards"] = self.num_shards
        args["shard_id"] = self.shard_id
        return args

    def get_dataset_size(self):
        """
        Get the number of batches in an epoch.

        Return:
            Number, number of batches.
        """
        if self.num_samples is None:
            num_samples = 0
        else:
            num_samples = self.num_samples

        if self.class_indexing is None:
            class_indexing = dict()
        else:
            class_indexing = self.class_indexing

        num_rows = ManifestOp.get_num_rows_and_classes(self.dataset_file, num_samples, class_indexing, self.usage)[0]

        return get_num_rows(num_rows, self.num_shards)

    def num_classes(self):
        """
        Get the number of classes in a dataset.

        Return:
            Number, number of classes.
        """
        if self.num_samples is None:
            num_samples = 0
        else:
            num_samples = self.num_samples

        if self.class_indexing is None:
            class_indexing = dict()
        else:
            class_indexing = self.class_indexing

        return ManifestOp.get_num_rows_and_classes(self.dataset_file, num_samples, class_indexing, self.usage)[1]

    def get_class_indexing(self):
        """
        Get the class index

        Return:
            Dict, A str-to-int mapping from label name to index.
        """
        if self.num_samples is None:
            num_samples = 0
        else:
            num_samples = self.num_samples

        if self.class_indexing is None:
            class_indexing = dict()
        else:
            class_indexing = self.class_indexing

        return ManifestOp.get_class_indexing(self.dataset_file, num_samples, class_indexing, self.usage)


class Cifar10Dataset(SourceDataset):
    """
    A source dataset that reads cifar10 data.

    The generated dataset has two columns ['image', 'label'].
    The type of the image tensor is uint8. The label is just a scalar uint32
    tensor.
    This dataset can take in a sampler. sampler and shuffle are mutually exclusive. Table
    below shows what input args are allowed and their expected behavior.

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
        num_samples (int, optional): The number of images to be included in the dataset.
            (default=None, all images).
        num_parallel_workers (int, optional): Number of workers to read the data
            (default=None, number set in the config).
        shuffle (bool, optional): Whether to perform shuffle on the dataset (default=None, expected
            order behavior shown in the table).
        sampler (Sampler, optional): Object used to choose samples from the
            dataset (default=None, expected order behavior shown in the table).
        num_shards (int, optional): Number of shards that the dataset should be divided
            into (default=None).
        shard_id (int, optional): The shard ID within num_shards (default=None). This
            argument should be specified only when num_shards is also specified.

    Raises:
        RuntimeError: If sampler and shuffle are specified at the same time.
        RuntimeError: If sampler and sharding are specified at the same time.
        RuntimeError: If num_shards is specified but shard_id is None.
        RuntimeError: If shard_id is specified but num_shards is None.
        ValueError: If shard_id is invalid (< 0 or >= num_shards).

    Examples:
        >>> import mindspore.dataset as ds
        >>> dataset_dir = "/path/to/cifar10_dataset_directory"
        >>> # 1) get all samples from CIFAR10 dataset in sequence:
        >>> dataset = ds.Cifar10Dataset(dataset_dir=dataset_dir,shuffle=False)
        >>> # 2) randomly select 350 samples from CIFAR10 dataset:
        >>> dataset = ds.Cifar10Dataset(dataset_dir=dataset_dir,num_samples=350, shuffle=True)
        >>> # 3) get samples from CIFAR10 dataset for shard 0 in a 2 way distributed training:
        >>> dataset = ds.Cifar10Dataset(dataset_dir=dataset_dir,num_shards=2,shard_id=0)
        >>> # in CIFAR10 dataset, each dictionary has keys "image" and "label"
    """

    @check_mnist_cifar_dataset
    def __init__(self, dataset_dir, num_samples=None, num_parallel_workers=None,
                 shuffle=None, sampler=None, num_shards=None, shard_id=None):
        super().__init__(num_parallel_workers)

        self.dataset_dir = dataset_dir
        self.sampler = _select_sampler(num_samples, sampler, shuffle, num_shards, shard_id)
        self.num_samples = num_samples
        self.num_shards = num_shards
        self.shard_id = shard_id
        self.shuffle_level = shuffle

    def get_args(self):
        args = super().get_args()
        args["dataset_dir"] = self.dataset_dir
        args["num_samples"] = self.num_samples
        args["sampler"] = self.sampler
        args["num_shards"] = self.num_shards
        args["shard_id"] = self.shard_id
        args["shuffle"] = self.shuffle_level
        return args

    def get_dataset_size(self):
        """
        Get the number of batches in an epoch.

        Return:
            Number, number of batches.
        """
        if self.num_samples is None:
            num_samples = 0
        else:
            num_samples = self.num_samples

        num_rows = CifarOp.get_num_rows(self.dataset_dir, num_samples, True)

        return get_num_rows(num_rows, self.num_shards)


class Cifar100Dataset(SourceDataset):
    """
    A source dataset that reads cifar100 data.

    The generated dataset has three columns ['image', 'coarse_label', 'fine_label'].
    The type of the image tensor is uint8. The coarse and fine are just a scalar uint32
    tensor.
    This dataset can take in a sampler. sampler and shuffle are mutually exclusive. Table
    below shows what input args are allowed and their expected behavior.

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
        num_samples (int, optional): The number of images to be included in the dataset.
            (default=None, all images).
        num_parallel_workers (int, optional): Number of workers to read the data
            (default=None, number set in the config).
        shuffle (bool, optional): Whether to perform shuffle on the dataset (default=None, expected
            order behavior shown in the table).
        sampler (Sampler, optional): Object used to choose samples from the
            dataset (default=None, expected order behavior shown in the table).
        num_shards (int, optional): Number of shards that the dataset should be divided
            into (default=None).
        shard_id (int, optional): The shard ID within num_shards (default=None). This
            argument should be specified only when num_shards is also specified.

    Raises:
        RuntimeError: If sampler and shuffle are specified at the same time.
        RuntimeError: If sampler and sharding are specified at the same time.
        RuntimeError: If num_shards is specified but shard_id is None.
        RuntimeError: If shard_id is specified but num_shards is None.
        ValueError: If shard_id is invalid (< 0 or >= num_shards).

    Examples:
        >>> import mindspore.dataset as ds
        >>> dataset_dir = "/path/to/cifar100_dataset_directory"
        >>> # 1) get all samples from CIFAR100 dataset in sequence:
        >>> cifar100_dataset = ds.Cifar100Dataset(dataset_dir=dataset_dir,shuffle=False)
        >>> # 2) randomly select 350 samples from CIFAR100 dataset:
        >>> cifar100_dataset = ds.Cifar100Dataset(dataset_dir=dataset_dir,num_samples=350, shuffle=True)
        >>> # in CIFAR100 dataset, each dictionary has 3 keys: "image", "fine_label" and "coarse_label"
    """

    @check_mnist_cifar_dataset
    def __init__(self, dataset_dir, num_samples=None, num_parallel_workers=None,
                 shuffle=None, sampler=None, num_shards=None, shard_id=None):
        super().__init__(num_parallel_workers)

        self.dataset_dir = dataset_dir
        self.sampler = _select_sampler(num_samples, sampler, shuffle, num_shards, shard_id)
        self.num_samples = num_samples
        self.num_shards = num_shards
        self.shard_id = shard_id
        self.shuffle_level = shuffle

    def get_args(self):
        args = super().get_args()
        args["dataset_dir"] = self.dataset_dir
        args["num_samples"] = self.num_samples
        args["sampler"] = self.sampler
        args["num_shards"] = self.num_shards
        args["shard_id"] = self.shard_id
        args["shuffle"] = self.shuffle_level
        return args

    def get_dataset_size(self):
        """
        Get the number of batches in an epoch.

        Return:
            Number, number of batches.
        """
        if self.num_samples is None:
            num_samples = 0
        else:
            num_samples = self.num_samples

        num_rows = CifarOp.get_num_rows(self.dataset_dir, num_samples, False)

        return get_num_rows(num_rows, self.num_shards)


class Schema:
    """
    Class to represent a schema of dataset.

    Args:
        schema_file(str): Path of schema file (default=None).

    Return:
        Schema object, schema info about dataset.

    Raises:
        RuntimeError: If schema file failed to load.

    Example:
        >>> import mindspore.dataset as ds
        >>> import mindspore.common.dtype as mstype
        >>> # create schema, specify column name, mindspore.dtype and shape of the column
        >>> schema = ds.Schema()
        >>> schema.add_column('col1', de_type=mindspore.int64, shape=[2])
    """

    def __init__(self, schema_file=None):
        if schema_file is None:
            self.columns = []
            self.dataset_type = ''
            self.num_rows = 0
        else:
            if not os.path.isfile(schema_file) or not os.access(schema_file, os.R_OK):
                raise ValueError("The file %s does not exist or permission denied!" % schema_file)
            try:
                with open(schema_file, 'r') as load_f:
                    json_obj = json.load(load_f)
            except json.decoder.JSONDecodeError:
                raise RuntimeError("Schema file failed to load.")
            except UnicodeDecodeError:
                raise RuntimeError("Schema file failed to decode.")
            except Exception:
                raise RuntimeError("Schema file failed to open.")
            self.from_json(json_obj)

    @check_add_column
    def add_column(self, name, de_type, shape=None):
        """
        Add new column to the schema.

        Args:
            name (str): name of the column.
            de_type (str): data type of the column.
            shape (list[int], optional): shape of the column
                (default=None, [-1] which is an unknown shape of rank 1).

        Raises:
            ValueError: If column type is unknown.
        """
        new_column = dict()
        new_column["name"] = name
        if isinstance(de_type, typing.Type):
            de_type = mstype_to_detype(de_type)
            new_column["type"] = str(de_type)
        else:
            new_column["type"] = str(DataType(de_type))

        if shape is not None:
            new_column["shape"] = shape
            new_column["rank"] = len(shape)
        else:
            new_column["rank"] = 1
        self.columns.append(new_column)

    def to_json(self):
        """
        Get a JSON string of the schema.

        Returns:
            Str, JSON string of the schema.
        """
        json_file = dict()
        json_file["columns"] = self.columns
        if self.dataset_type:
            json_file["datasetType"] = self.dataset_type
        if self.num_rows:
            json_file["numRows"] = self.num_rows
        return json.dumps(json_file, indent=2)

    def parse_columns(self, columns):
        """
        Parse the columns and add it to self.

        Args:
            columns (dict or list[dict]): dataset attribution information, decoded from schema file.

                - list[dict], 'name' and 'type' must be in keys, 'shape' optional.

                - dict, columns.keys() as name, columns.values() is dict, and 'type' inside, 'shape' optional.

        Raises:
            RuntimeError: If failed to parse columns.
            RuntimeError: If unknown items in columns.
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
        self.columns = []
        if isinstance(columns, list):
            for column in columns:
                try:
                    name = column.pop("name")
                except KeyError:
                    raise RuntimeError("Column's name is missing")
                try:
                    de_type = column.pop("type")
                except KeyError:
                    raise RuntimeError("Column' type is missing")
                shape = column.pop("shape", None)
                column.pop("t_impl", None)
                column.pop("rank", None)
                if column:
                    raise RuntimeError("Unknown field {}".format(",".join(column.keys())))
                self.add_column(name, de_type, shape)
        elif isinstance(columns, dict):
            for key, value in columns.items():
                name = key
                try:
                    de_type = value.pop("type")
                except KeyError:
                    raise RuntimeError("Column' type is missing")
                shape = value.pop("shape", None)
                value.pop("t_impl", None)
                value.pop("rank", None)
                if value:
                    raise RuntimeError("Unknown field {}".format(",".join(value.keys())))
                self.add_column(name, de_type, shape)
        else:
            raise RuntimeError("columns must be dict or list, columns contain name, type, shape(optional).")

    def from_json(self, json_obj):
        """
        Get schema file from json file.

        Args:
            json_obj(dictionary): object of json parsed.

        Raises:
            RuntimeError: if there is unknown item in the object.
            RuntimeError: if dataset type is missing in the object.
            RuntimeError: if columns are missing in the object.
        """
        if not isinstance(json_obj, dict) or json_obj is None:
            raise ValueError("Expected non-empty dict.")
        for k, v in json_obj.items():
            if k == "datasetType":
                self.dataset_type = v
            elif k == "numRows":
                self.num_rows = v
            elif k == "columns":
                self.parse_columns(v)
            else:
                raise RuntimeError("Unknown field %s" % k)

        if self.dataset_type is None:
            raise RuntimeError("DatasetType field is missing.")
        if self.columns is None:
            raise RuntimeError("Columns are missing.")

    def __str__(self):
        return self.to_json()


class VOCDataset(SourceDataset):
    """
    A source dataset for reading and parsing VOC dataset.

    The generated dataset has two columns ['image', 'target'].
    The shape of both column is [image_size] if decode flag is False, or [H, W, C]
    otherwise.
    The type of both tensor is uint8.
    This dataset can take in a sampler. sampler and shuffle are mutually exclusive. Table
    below shows what input args are allowed and their expected behavior.

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
            (default=None, number set in the config).
        shuffle (bool, optional): Whether to perform shuffle on the dataset (default=None, expected
            order behavior shown in the table).
        decode (bool, optional): Decode the images after reading (default=False).
        sampler (Sampler, optional): Object used to choose samples from the dataset
            (default=None, expected order behavior shown in the table).
        distribution (str, optional): Path to the json distribution file to configure
            dataset sharding (default=None). This argument should be specified
            only when no 'sampler' is used.

    Raises:
        RuntimeError: If distribution and sampler are specified at the same time.
        RuntimeError: If distribution is failed to read.
        RuntimeError: If shuffle and sampler are specified at the same time.

    Examples:
        >>> import mindspore.dataset as ds
        >>> dataset_dir = "/path/to/voc_dataset_directory"
        >>> # 1) read all VOC dataset samples in dataset_dir with 8 threads in random order:
        >>> voc_dataset = ds.VOCDataset(dataset_dir, num_parallel_workers=8)
        >>> # 2) read then decode all VOC dataset samples in dataset_dir in sequence:
        >>> voc_dataset = ds.VOCDataset(dataset_dir, decode=True, shuffle=False)
        >>> # in VOC dataset, each dictionary has keys "image" and "target"
    """

    @check_vocdataset
    def __init__(self, dataset_dir, num_samples=None, num_parallel_workers=None,
                 shuffle=None, decode=False, sampler=None, distribution=None):
        super().__init__(num_parallel_workers)
        self.dataset_dir = dataset_dir
        self.sampler = sampler
        if distribution is not None:
            if sampler is not None:
                raise RuntimeError("Cannot specify distribution and sampler at the same time.")
            try:
                with open(distribution, 'r') as load_d:
                    json.load(load_d)
            except json.decoder.JSONDecodeError:
                raise RuntimeError("Json decode error when load distribution file")
            except Exception:
                raise RuntimeError("Distribution file has failed to load.")
        elif shuffle is not None:
            if sampler is not None:
                raise RuntimeError("Cannot specify shuffle and sampler at the same time.")
        self.num_samples = num_samples
        self.decode = decode
        self.distribution = distribution
        self.shuffle_level = shuffle

    def get_args(self):
        args = super().get_args()
        args["dataset_dir"] = self.dataset_dir
        args["num_samples"] = self.num_samples
        args["sampler"] = self.sampler
        args["decode"] = self.decode
        args["shuffle"] = self.shuffle_level
        args["distribution"] = self.distribution
        return args

    def get_dataset_size(self):
        """
        Get the number of batches in an epoch.

        Return:
            Number, number of batches.
        """
        return self.num_samples


class CelebADataset(SourceDataset):
    """
    A source dataset for reading and parsing CelebA dataset.Only support list_attr_celeba.txt currently

    Note:
        The generated dataset has two columns ['image', 'attr'].
        The type of the image tensor is uint8. The attr tensor is uint32 and one hot type.

    Args:
        dataset_dir (str): Path to the root directory that contains the dataset.
        num_parallel_workers (int, optional): Number of workers to read the data (default=value set in the config).
        shuffle (bool, optional): Whether to perform shuffle on the dataset (default=None).
        dataset_type (string): one of 'all', 'train', 'valid' or 'test'.
        sampler (Sampler, optional): Object used to choose samples from the dataset (default=None).
        decode (bool, optional): decode the images after reading (default=False).
        extensions (list[str], optional): List of file extensions to be
            included in the dataset (default=None).
        num_samples (int, optional): The number of images to be included in the dataset.
            (default=None, all images).
        num_shards (int, optional): Number of shards that the dataset should be divided
            into (default=None).
        shard_id (int, optional): The shard ID within num_shards (default=None). This
            argument should be specified only when num_shards is also specified.
    """

    @check_celebadataset
    def __init__(self, dataset_dir, num_parallel_workers=None, shuffle=None, dataset_type='all',
                 sampler=None, decode=False, extensions=None, num_samples=None, num_shards=None, shard_id=None):
        super().__init__(num_parallel_workers)
        self.dataset_dir = dataset_dir
        self.sampler = _select_sampler(num_samples, sampler, shuffle, num_shards, shard_id)
        self.num_parallel_workers = num_parallel_workers
        self.decode = decode
        self.extensions = extensions
        self.num_samples = num_samples
        self.dataset_type = dataset_type
        self.num_shards = num_shards
        self.shard_id = shard_id
        self.shuffle_level = shuffle

    def get_args(self):
        args = super().get_args()
        args["dataset_dir"] = self.dataset_dir
        args["sampler"] = self.sampler
        args["shuffle"] = self.shuffle_level
        args["decode"] = self.decode
        args["extensions"] = self.extensions
        args["num_samples"] = self.num_samples
        args["dataset_type"] = self.dataset_type
        args["num_shards"] = self.num_shards
        args["shard_id"] = self.shard_id
        return args

class TextFileDataset(SourceDataset):
    """
    A source dataset that reads and parses datasets stored on disk in text format.
    The generated dataset has one columns ['text'].

    Args:
        dataset_files (str or list[str]): String or list of files to be read or glob strings to search for a pattern of
            files. The list will be sorted in a lexicographical order.
        num_samples (int, optional): number of samples(rows) to read (default=None, reads the full dataset).
        num_parallel_workers (int, optional): number of workers to read the data
            (default=None, number set in the config).
        shuffle (bool, Shuffle level, optional): perform reshuffling of the data every epoch (default=Shuffle.GLOBAL).
            If shuffle is False, no shuffling will be performed;
            If shuffle is True, the behavior is the same as setting shuffle to be Shuffle.GLOBAL
            Otherwise, there are two levels of shuffling:

            - Shuffle.GLOBAL: Shuffle both the files and samples.

            - Shuffle.FILES: Shuffle files only.

        num_shards (int, optional): Number of shards that the dataset should be divided into (default=None).
        shard_id (int, optional): The shard ID within num_shards (default=None). This
            argument should be specified only when num_shards is also specified.
    Examples:
        >>> import mindspore.dataset as ds
        >>> dataset_files = ["/path/to/1", "/path/to/2"] # contains 1 or multiple text files
        >>> dataset = ds.TextFileDataset(dataset_files=dataset_files)
    """

    @check_textfiledataset
    def __init__(self, dataset_files, num_samples=None, num_parallel_workers=None,
                 shuffle=Shuffle.GLOBAL, num_shards=None, shard_id=None):
        super().__init__(num_parallel_workers)
        self.dataset_files = self._find_files(dataset_files)
        self.dataset_files.sort()
        self.num_samples = num_samples

        if not isinstance(shuffle, (bool, Shuffle)):
            raise TypeError("shuffle should be of boolean or enum 'Shuffle'.")
        if not isinstance(shuffle, Shuffle):
            if shuffle:
                self.shuffle_level = Shuffle.GLOBAL
                self.shuffle_files = True
            else:
                self.shuffle_level = None
                self.shuffle_files = False
        else:
            self.shuffle_level = shuffle
            self.shuffle_files = True

        self.num_shards = num_shards
        self.shard_id = shard_id

    def get_args(self):
        args = super().get_args()
        args["dataset_files"] = self.dataset_files
        args["num_samples"] = self.num_samples
        if self.shuffle_files is not None:
            args["shuffle_files"] = self.shuffle_files
        args["shuffle"] = self.shuffle_level
        args["num_shards"] = self.num_shards
        args["shard_id"] = self.shard_id
        return args

    def get_dataset_size(self):
        """
        Get the number of batches in an epoch.

        Return:
            Number, number of batches.
        """
        if self._dataset_size is None:
            num_rows = TextFileOp.get_num_rows(self.dataset_files)
            num_rows = get_num_rows(num_rows, self.num_shards)
            if self.num_samples is None:
                return num_rows
            return min(self.num_samples, num_rows)
        return self._dataset_size
