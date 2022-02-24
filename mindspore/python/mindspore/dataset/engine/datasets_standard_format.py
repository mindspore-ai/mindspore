# Copyright 2019-2022 Huawei Technologies Co., Ltd
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
This file contains standard format dataset loading classes.
You can convert a dataset to a standard format using the following steps:
    1. Use mindspore.mindrecord.FileWriter / tf.io.TFRecordWriter api to
       convert dataset to MindRecord / TFRecord.
    2. Use MindDataset / TFRecordDataset to load MindRecord / TFRecrod files.
After declaring the dataset object, you can further apply dataset operations
(e.g. filter, skip, concat, map, batch) on it.
"""
import platform

import numpy as np

import mindspore._c_dataengine as cde

from mindspore import log as logger
from .datasets import UnionBaseDataset, SourceDataset, MappableDataset, Shuffle, Schema, \
    shuffle_to_shuffle_mode, shuffle_to_bool
from .validators import check_minddataset, check_tfrecorddataset, check_csvdataset

from ..core.validator_helpers import replace_none
from . import samplers


class CSVDataset(SourceDataset, UnionBaseDataset):
    """
    A source dataset that reads and parses comma-separated values
    `(CSV) <http://en.volupedia.org/wiki/Comma-separated_values>`_ files as dataset.

    The columns of generated dataset depend on the source CSV files.

    Args:
        dataset_files (Union[str, list[str]]): String or list of files to be read or glob strings to search
            for a pattern of files. The list will be sorted in a lexicographical order.
        field_delim (str, optional): A string that indicates the char delimiter to separate fields (default=',').
        column_defaults (list, optional): List of default values for the CSV field (default=None). Each item
            in the list is either a valid type (float, int, or string). If this is not provided, treats all
            columns as string type.
        column_names (list[str], optional): List of column names of the dataset (default=None). If this
            is not provided, infers the column_names from the first row of CSV file.
        num_samples (int, optional): The number of samples to be included in the dataset
            (default=None, will include all images).
        num_parallel_workers (int, optional): Number of workers to read the data
            (default=None, number set in the config).
        shuffle (Union[bool, Shuffle level], optional): Perform reshuffling of the data every epoch
            (default=Shuffle.GLOBAL).
            If shuffle is False, no shuffling will be performed.
            If shuffle is True, performs global shuffle.
            There are three levels of shuffling, desired shuffle enum defined by mindspore.dataset.Shuffle.

            - Shuffle.GLOBAL: Shuffle both the files and samples, same as setting shuffle to True.

            - Shuffle.FILES: Shuffle files only.

        num_shards (int, optional): Number of shards that the dataset will be divided into (default=None).
            When this argument is specified, `num_samples` reflects the maximum sample number of per shard.
        shard_id (int, optional): The shard ID within `num_shards` (default=None). This
            argument can only be specified when `num_shards` is also specified.
        cache (DatasetCache, optional): Use tensor caching service to speed up dataset processing.
            (default=None, which means no cache is used).

    Raises:
        RuntimeError: If dataset_files are not valid or do not exist.
        ValueError: If field_delim is invalid.
        ValueError: If `num_parallel_workers` exceeds the max thread numbers.
        RuntimeError: If `num_shards` is specified but `shard_id` is None.
        RuntimeError: If `shard_id` is specified but `num_shards` is None.
        ValueError: If `shard_id` is invalid (< 0 or >= `num_shards`).

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


class MindDataset(MappableDataset, UnionBaseDataset):
    """
    A source dataset that reads and parses MindRecord dataset.

    The columns of generated dataset depend on the source MindRecord files.

    Args:
        dataset_files (Union[str, list[str]]): If dataset_file is a str, it represents for
            a file name of one component of a mindrecord source, other files with identical source
            in the same path will be found and loaded automatically. If dataset_file is a list,
            it represents for a list of dataset files to be read directly.
        columns_list (list[str], optional): List of columns to be read (default=None).
        num_parallel_workers (int, optional): The number of readers (default=None).
        shuffle (Union[bool, Shuffle level], optional): Perform reshuffling of the data every epoch
            (default=None, performs global shuffle).
            If shuffle is False, no shuffling will be performed.
            If shuffle is True, performs global shuffle.
            There are three levels of shuffling, desired shuffle enum defined by mindspore.dataset.Shuffle.

            - Shuffle.GLOBAL: Global shuffle of all rows of data in dataset, same as setting shuffle to True.

            - Shuffle.FILES: Shuffle the file sequence but keep the order of data within each file.

            - Shuffle.INFILE: Keep the file sequence the same but shuffle the data within each file.

        num_shards (int, optional): Number of shards that the dataset will be divided into (default=None).
            When this argument is specified, 'num_samples' reflects the maximum sample number of per shard.
        shard_id (int, optional): The shard ID within `num_shards` (default=None). This
            argument can only be specified when `num_shards` is also specified.
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
        cache (DatasetCache, optional): Use tensor caching service to speed up dataset processing.
            (default=None, which means no cache is used).

    Raises:
        ValueError: If dataset_files are not valid or do not exist.
        ValueError: If `num_parallel_workers` exceeds the max thread numbers.
        RuntimeError: If `num_shards` is specified but `shard_id` is None.
        RuntimeError: If `shard_id` is specified but `num_shards` is None.
        ValueError: If `shard_id` is invalid (< 0 or >= `num_shards`).

    Note:
        - This dataset can take in a `sampler`. `sampler` and `shuffle` are mutually exclusive.
          The table below shows what input arguments are allowed and their expected behavior.

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

    Examples:
        >>> mind_dataset_dir = ["/path/to/mind_dataset_file"] # contains 1 or multiple MindRecord files
        >>> dataset = ds.MindDataset(dataset_files=mind_dataset_dir)
    """

    def parse(self, children=None):
        return cde.MindDataNode(self.dataset_files, self.columns_list, self.sampler, self.new_padded_sample,
                                self.num_padded, shuffle_to_shuffle_mode(self.shuffle_option))

    @check_minddataset
    def __init__(self, dataset_files, columns_list=None, num_parallel_workers=None, shuffle=None, num_shards=None,
                 shard_id=None, sampler=None, padded_sample=None, num_padded=None, num_samples=None, cache=None):
        super().__init__(num_parallel_workers=num_parallel_workers, sampler=sampler, num_samples=num_samples,
                         shuffle=shuffle_to_bool(shuffle), num_shards=num_shards, shard_id=shard_id, cache=cache)
        if shuffle is not None and not isinstance(shuffle, (bool, Shuffle)):
            raise TypeError("shuffle must be of boolean or enum of 'Shuffle' values like 'Shuffle.GLOBAL' or "
                            "'Shuffle.FILES' or 'Shuffle.INFILE'.")
        if num_samples and shuffle in (Shuffle.FILES, Shuffle.INFILE):
            raise ValueError("'Shuffle.FILES' or 'Shuffle.INFILE' and 'num_samples' "
                             "cannot be specified at the same time.")
        self.shuffle_option = shuffle
        if isinstance(dataset_files, list):
            self.load_dataset = False
        else:
            self.load_dataset = True

        self.dataset_files = ""
        if platform.system().lower() == "windows":
            if isinstance(dataset_files, list):
                self.dataset_files = [item.replace("\\", "/") for item in dataset_files]
            else:
                self.dataset_files = dataset_files.replace("\\", "/")
        else:
            self.dataset_files = dataset_files
        self.columns_list = replace_none(columns_list, [])

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


class TFRecordDataset(SourceDataset, UnionBaseDataset):
    """
    A source dataset that reads and parses datasets stored on disk in TFData format.

    The columns of generated dataset depend on the source TFRecord files.

    Args:
        dataset_files (Union[str, list[str]]): String or list of files to be read or glob strings to search for a
            pattern of files. The list will be sorted in a lexicographical order.
        schema (Union[str, Schema], optional): Path to the JSON schema file or schema object (default=None).
            If the schema is not provided, the meta data from the TFData file is considered the schema.
        columns_list (list[str], optional): List of columns to be read (default=None, read all columns).
        num_samples (int, optional): The number of samples (rows) to be included in the dataset (default=None).
            If num_samples is None and numRows(parsed from schema) does not exist, read the full dataset;
            If num_samples is None and numRows(parsed from schema) is greater than 0, read numRows rows;
            If both num_samples and numRows(parsed from schema) are greater than 0, read num_samples rows.
        num_parallel_workers (int, optional): Number of workers to read the data
            (default=None, number set in the config).
        shuffle (Union[bool, Shuffle level], optional): Perform reshuffling of the data every epoch
            (default=Shuffle.GLOBAL).
            If shuffle is False, no shuffling will be performed.
            If shuffle is True, performs global shuffle.
            There are three levels of shuffling, desired shuffle enum defined by mindspore.dataset.Shuffle.

            - Shuffle.GLOBAL: Shuffle both the files and samples, same as setting shuffle to True.

            - Shuffle.FILES: Shuffle files only.

        num_shards (int, optional): Number of shards that the dataset will be divided
            into (default=None). When this argument is specified, `num_samples` reflects
            the maximum sample number of per shard.
        shard_id (int, optional): The shard ID within `num_shards` (default=None). This
            argument can only be specified when `num_shards` is also specified.
        shard_equal_rows (bool, optional): Get equal rows for all shards(default=False). If shard_equal_rows
            is false, number of rows of each shard may be not equal, and may lead to a failure in distributed training.
            When the number of samples of per TFRecord file are not equal, it is suggested to set to true.
            This argument should only be specified when `num_shards` is also specified.
        cache (DatasetCache, optional): Use tensor caching service to speed up dataset processing.
            (default=None, which means no cache is used).

    Raises:
        ValueError: If dataset_files are not valid or do not exist.
        ValueError: If `num_parallel_workers` exceeds the max thread numbers.
        RuntimeError: If `num_shards` is specified but `shard_id` is None.
        RuntimeError: If `shard_id` is specified but `num_shards` is None.
        ValueError: If `shard_id` is invalid (< 0 or >= `num_shards`).

    Examples:
        >>> from mindspore import dtype as mstype
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
