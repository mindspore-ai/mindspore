# Copyright 2019-2023 Huawei Technologies Co., Ltd
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
import os
import platform
import weakref

import numpy as np

import mindspore._c_dataengine as cde
from mindspore import log as logger

from .datasets import UnionBaseDataset, SourceDataset, MappableDataset, Schema
from .samplers import Shuffle, shuffle_to_shuffle_mode
from .datasets_user_defined import GeneratorDataset
from .obs.obs_mindrecord_dataset import MindRecordFromOBS
from .validators import check_csvdataset, check_minddataset, check_tfrecorddataset, check_obsminddataset
from ..core.validator_helpers import type_check
from ...mindrecord.config import _get_enc_key, _get_dec_mode, decrypt, cleanup_decrypt_files


from ..core.validator_helpers import replace_none
from . import samplers


class CSVDataset(SourceDataset, UnionBaseDataset):
    """
    A source dataset that reads and parses comma-separated values
    `(CSV) <https://en.wikipedia.org/wiki/Comma-separated_values>`_ files as dataset.

    The columns of generated dataset depend on the source CSV files.

    Args:
        dataset_files (Union[str, list[str]]): String or list of files to be read or glob strings to search
            for a pattern of files. The list will be sorted in a lexicographical order.
        field_delim (str, optional): A string that indicates the char delimiter to separate fields.
            Default: ``','``.
        column_defaults (list, optional): List of default values for the CSV field. Default: ``None``. Each item
            in the list is either a valid type (float, int, or string). If this is not provided, treats all
            columns as string type.
        column_names (list[str], optional): List of column names of the dataset. Default: ``None``. If this
            is not provided, infers the column_names from the first row of CSV file.
        num_samples (int, optional): The number of samples to be included in the dataset.
            Default: ``None``, will include all samples.
        num_parallel_workers (int, optional): Number of worker threads to read the data.
            Default: ``None``, will use global default workers(8), it can be set
            by :func:`mindspore.dataset.config.set_num_parallel_workers` .
        shuffle (Union[bool, Shuffle], optional): Perform reshuffling of the data every epoch.
            Default: ``Shuffle.GLOBAL`` . Bool type and Shuffle enum are both supported to pass in.
            If `shuffle` is ``False`` , no shuffling will be performed.
            If `shuffle` is ``True`` , performs global shuffle.
            There are three levels of shuffling, desired shuffle enum defined by :class:`mindspore.dataset.Shuffle` .

            - ``Shuffle.GLOBAL`` : Shuffle both the files and samples, same as setting shuffle to True.

            - ``Shuffle.FILES`` : Shuffle files only.

        num_shards (int, optional): Number of shards that the dataset will be divided into. Default: ``None`` .
            When this argument is specified, `num_samples` reflects the maximum sample number of per shard.
            Used in `data parallel training <https://www.mindspore.cn/tutorials/en/master/
            parallel/data_parallel.html#loading-datasets>`_ .
        shard_id (int, optional): The shard ID within `num_shards` . Default: ``None``. This
            argument can only be specified when `num_shards` is also specified.
        cache (DatasetCache, optional): Use tensor caching service to speed up dataset processing. More details:
            `Single-Node Data Cache <https://www.mindspore.cn/tutorials/en/master/dataset/cache.html>`_ .
            Default: ``None``, which means no cache is used.

    Raises:
        RuntimeError: If `dataset_files` are not valid or do not exist.
        ValueError: If `field_delim` is invalid.
        ValueError: If `num_parallel_workers` exceeds the max thread numbers.
        RuntimeError: If `num_shards` is specified but `shard_id` is None.
        RuntimeError: If `shard_id` is specified but `num_shards` is None.
        ValueError: If `shard_id` is not in range of [0, `num_shards` ).

    Examples:
        >>> import mindspore.dataset as ds
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
        columns_list (list[str], optional): List of columns to be read. Default: ``None`` , read all columns.
        num_parallel_workers (int, optional): Number of worker threads to read the data.
            Default: ``None`` , will use global default workers(8), it can be set
            by :func:`mindspore.dataset.config.set_num_parallel_workers` .
        shuffle (Union[bool, Shuffle], optional): Perform reshuffling of the data every epoch, bool type and ``Shuffle``
            enum are both supported to pass in. Default: ``None``, performs ``mindspore.dataset.Shuffle.ADAPTIVE`` .
            If `shuffle` is set to ``False`` , no shuffling will be performed.
            If `shuffle` is set to ``True`` , `shuffle` is set to ``mindspore.dataset.Shuffle.ADAPTIVE`` .
            There are several levels of shuffling, desired shuffle enum defined by :class:`mindspore.dataset.Shuffle` .

            - ``Shuffle.ADAPTIVE`` : When the number of dataset samples is less than or equal to 100 million,
              ``Shuffle.GLOBAL`` is used. When the number of dataset samples is greater than 100
              million, ``Shuffle.PARTIAL`` is used. The shuffle is performed once
              every 1 million samples.

            - ``Shuffle.GLOBAL`` : Global shuffle of all rows of data in dataset. The memory usage is large.

            - ``Shuffle.PARTIAL`` : Partial shuffle of data in dataset for every 1 million samples.
              The memory usage is less than ``Shuffle.GLOBAL`` .

            - ``Shuffle.FILES`` : Shuffle the file sequence but keep the order of data within each file.

            - ``Shuffle.INFILE`` : Keep the file sequence the same but shuffle the data within each file.

        num_shards (int, optional): Number of shards that the dataset will be divided into. Default: ``None`` .
            When this argument is specified, `num_samples` reflects the maximum sample number of per shard.
            Used in `data parallel training <https://www.mindspore.cn/tutorials/en/master/
            parallel/data_parallel.html#loading-datasets>`_ .
        shard_id (int, optional): The shard ID within `num_shards` . Default: ``None`` . This
            argument can only be specified when `num_shards` is also specified.
        sampler (Sampler, optional): Object used to choose samples from the
            dataset. Default: ``None`` , sampler is exclusive
            with shuffle and block_reader. Support list: :class:`mindspore.dataset.SubsetRandomSampler`,
            :class:`mindspore.dataset.PKSampler`, :class:`mindspore.dataset.RandomSampler`,
            :class:`mindspore.dataset.SequentialSampler`, :class:`mindspore.dataset.DistributedSampler`.
        padded_sample (dict, optional): Samples will be appended to dataset, where
            keys are the same as columns_list. Default: ``None``.
        num_padded (int, optional): Number of padding samples. Dataset size
            plus num_padded should be divisible by num_shards. Default: ``None``.
        num_samples (int, optional): The number of samples to be included in the dataset.
            Default: ``None`` , all samples.
        cache (DatasetCache, optional): Use tensor caching service to speed up dataset processing. More details:
            `Single-Node Data Cache <https://www.mindspore.cn/tutorials/en/master/dataset/cache.html>`_ .
            Default: ``None`` , which means no cache is used.

    Raises:
        ValueError: If dataset_files are not valid or do not exist.
        ValueError: If `num_parallel_workers` exceeds the max thread numbers.
        RuntimeError: If `num_shards` is specified but `shard_id` is None.
        RuntimeError: If `shard_id` is specified but `num_shards` is None.
        ValueError: If `shard_id` is not in range of [0, `num_shards` ).
        TypeError: If `shuffle` is not of type None, bool or Shuffle.

    Note:
        - When sharding MindRecord (by configuring `num_shards` and `shard_id`), there are two strategies to implement
          the data sharding logic. This API uses the strategy 1 by default, which can be switched to strategy 2 by
          setting the environment variable `MS_DEV_MINDRECORD_SHARD_BY_BLOCK=True` . This environment variable only
          applies to the `DistributedSampler` sampler.

        .. list-table:: Data sharding strategy 1
            :widths: 50 50 50 50
            :header-rows: 1

            * - rank 0
              - rank 1
              - rank 2
              - rank 3
            * - 0
              - 1
              - 2
              - 3
            * - 4
              - 5
              - 6
              - 7
            * - 8
              - 9
              - 10
              - 11

        .. list-table:: Data sharding strategy 2
            :widths: 50 50 50 50
            :header-rows: 1

            * - rank 0
              - rank 1
              - rank 2
              - rank 3
            * - 0
              - 3
              - 6
              - 9
            * - 1
              - 4
              - 7
              - 10
            * - 2
              - 5
              - 8
              - 11

    Note:
        - The parameters `num_samples` , `shuffle` , `num_shards` , `shard_id` can be used to control the sampler
          used in the dataset, and their effects when combined with parameter `sampler` are as follows.

    .. include:: mindspore.dataset.sampler.txt

    Examples:
        >>> import mindspore.dataset as ds
        >>> mindrecord_files = ["/path/to/mind_dataset_file"] # contains 1 or multiple MindRecord files
        >>> dataset = ds.MindDataset(dataset_files=mindrecord_files)
    """

    def parse(self, children=None):
        child_sampler = self.sampler.get_child()
        if (child_sampler is not None and not isinstance(child_sampler, samplers.DistributedSampler)
                and self.num_padded > 0):
            raise RuntimeError("When the padded sample logic is enabled and use sampler chain,"
                               "the first sampler which is specified by parameter "
                               "sampler or (num_shards, shard_id) is not distributed sampling.")
        return cde.MindDataNode(self.dataset_files, self.columns_list, self.sampler, self.new_padded_sample,
                                self.num_padded, shuffle_to_shuffle_mode(self.shuffle_option))

    @check_minddataset
    def __init__(self, dataset_files, columns_list=None, num_parallel_workers=None, shuffle=None, num_shards=None,
                 shard_id=None, sampler=None, padded_sample=None, num_padded=None, num_samples=None, cache=None):
        if sampler is None:
            if shuffle is None or shuffle is True:
                shuffle = Shuffle.ADAPTIVE
            elif shuffle is False:
                shuffle = Shuffle.FALSE
        super().__init__(num_parallel_workers=num_parallel_workers, sampler=sampler, num_samples=num_samples,
                         shuffle=shuffle, num_shards=num_shards, shard_id=shard_id, cache=cache)
        if num_samples and shuffle in (Shuffle.FILES, Shuffle.INFILE):
            raise ValueError("'Shuffle.FILES' or 'Shuffle.INFILE' and 'num_samples' "
                             "cannot be specified at the same time.")
        self.shuffle_option = shuffle
        self._decrypted_files = []
        self.load_dataset = not isinstance(dataset_files, list)
        self.dataset_files = self._normalize_dataset_files(dataset_files)
        self.dataset_files = self._decrypt_dataset_files(self.dataset_files)
        self._decrypt_file_cleaner = weakref.finalize(
            self, cleanup_decrypt_files, tuple(self._decrypted_files))
        self.columns_list = replace_none(columns_list, [])
        self._validate_sampler(sampler)
        self.padded_sample = padded_sample
        self.num_padded = replace_none(num_padded, 0)
        self.new_padded_sample = self._convert_padded_sample(padded_sample)

    def __deepcopy__(self, memodict):
        if id(self) in memodict:
            return memodict[id(self)]
        return self.__safe_deepcopy__(memodict, exclude="mindrecord_op")

    def __getitem__(self, index):
        type_check(index, (int,), "index")
        if index < 0:
            raise ValueError("index cannot be negative, but got {0}.".format(index))
        if not hasattr(self, "mindrecord_op"):
            minddata_node = cde.MindDataNode(
                self.dataset_files, self.columns_list, self.sampler, self.new_padded_sample,
                self.num_padded, shuffle_to_shuffle_mode(self.shuffle_option))
            self.mindrecord_op = minddata_node.Build()
        return [t.as_array() for t in self.mindrecord_op[index]]

    def __del__(self):
        cleaner = getattr(self, "_decrypt_file_cleaner", None)
        if cleaner is not None and cleaner.alive:
            cleaner()

    def _normalize_dataset_files(self, dataset_files):
        """Normalize dataset file paths for the current platform."""
        if platform.system().lower() != "windows":
            return dataset_files
        if isinstance(dataset_files, list):
            return [item.replace("\\", "/") for item in dataset_files]
        return dataset_files.replace("\\", "/")

    def _decrypt_dataset_files(self, dataset_files):
        """Decrypt the dataset files and the paired index files when encryption is enabled."""
        if isinstance(dataset_files, list):
            return [self._decrypt_dataset_item(item) for item in dataset_files]
        self._warn_single_file_decrypt(dataset_files)
        return self._decrypt_dataset_item(dataset_files)

    def _warn_single_file_decrypt(self, dataset_file):
        """Warn about single-file decrypt limitations when encryption is enabled."""
        if _get_enc_key() is None:
            return
        logger.warning("When a single mindrecord file which is generated by " +
                       "`mindspore.mindrecord.FileWriter` with `shard_num` > 1 is used as the input, " +
                       "enabling decryption check may fail. Please use file list as the input.")

    def _decrypt_dataset_item(self, dataset_file):
        """Decrypt one dataset file and its paired index file."""
        decrypt_filename = self._decrypt_and_track(dataset_file)
        decrypt_index_file_name = decrypt_filename + ".db"
        self._decrypt_index_file(dataset_file + ".db", decrypt_index_file_name)
        return decrypt_filename

    def _decrypt_and_track(self, target_file):
        """Decrypt a file and record the temporary output if a new file is created."""
        decrypt_filename = decrypt(target_file, _get_enc_key(), _get_dec_mode())
        if decrypt_filename != target_file:
            self._decrypted_files.append(decrypt_filename)
        return decrypt_filename

    def _decrypt_index_file(self, index_file_name, decrypt_index_file_name):
        """Decrypt the paired index file and keep its name aligned with the data file."""
        decrypt_filename = decrypt(index_file_name, _get_enc_key(), _get_dec_mode())
        if decrypt_filename == index_file_name:
            return decrypt_filename
        if decrypt_filename != decrypt_index_file_name:
            os.replace(decrypt_filename, decrypt_index_file_name)
            decrypt_filename = decrypt_index_file_name
        self._decrypted_files.append(decrypt_filename)
        return decrypt_filename

    def _validate_sampler(self, sampler):
        """Validate the supported sampler types for MindDataset."""
        if sampler is None:
            return
        valid_samplers = (
            samplers.SubsetRandomSampler, samplers.SubsetSampler, samplers.PKSampler,
            samplers.DistributedSampler, samplers.RandomSampler, samplers.SequentialSampler)
        if not isinstance(sampler, valid_samplers):
            raise ValueError("The sampler is not supported yet.")

    def _convert_padded_sample(self, padded_sample):
        """Convert padded samples into the backend-friendly representation."""
        new_padded_sample = {}
        if not padded_sample:
            return new_padded_sample
        for key, value in padded_sample.items():
            new_padded_sample[key] = value.tobytes() if isinstance(value, np.ndarray) else value
        return new_padded_sample


class TFRecordDataset(SourceDataset, UnionBaseDataset):
    """
    A source dataset that reads and parses datasets stored on disk in TFData format.

    The columns of generated dataset depend on the source TFRecord files.

    Note:
        'TFRecordDataset' is not supported on Windows platform yet.

    Args:
        dataset_files (Union[str, list[str]]): String or list of files to be read or glob strings to search for a
            pattern of files. The list will be sorted in lexicographical order.
        schema (Union[str, Schema], optional): Data format policy, which specifies the data types and shapes of the data
            column to be read. Both JSON file path and objects constructed by :class:`mindspore.dataset.Schema` are
            acceptable. Default: ``None`` .
        columns_list (list[str], optional): List of columns to be read. Default: ``None`` , read all columns.
        num_samples (int, optional): The number of samples (rows) to be included in the dataset. Default: ``None`` .
            When `num_shards` and `shard_id` are specified, it will be interpreted as number of rows per shard.
            Processing priority for `num_samples` is as the following:

            - If specify `num_samples` with value > 0, read `num_samples` samples.

            - If no `num_samples` and specify numRows(parsed from `schema`) with value > 0, read numRows samples.

            - If no `num_samples` and no `schema`, read the full dataset.

        num_parallel_workers (int, optional): Number of worker threads to read the data.
            Default: ``None`` , will use global default workers(8), it can be set
            by :func:`mindspore.dataset.config.set_num_parallel_workers` .
        shuffle (Union[bool, Shuffle], optional): Perform reshuffling of the data every epoch.
            Default: ``Shuffle.GLOBAL`` . Bool type and Shuffle enum are both supported to pass in.
            If `shuffle` is ``False``, no shuffling will be performed.
            If `shuffle` is ``True``, perform global shuffle.
            There are three levels of shuffling, desired shuffle enum defined by :class:`mindspore.dataset.Shuffle` .

            - ``Shuffle.GLOBAL`` : Shuffle both the files and samples, same as setting `shuffle` to ``True``.

            - ``Shuffle.FILES`` : Shuffle files only.

        num_shards (int, optional): Number of shards that the dataset will be divided
            into. Default: ``None`` . When this argument is specified, `num_samples` reflects
            the maximum sample number per shard.
            Used in `data parallel training <https://www.mindspore.cn/tutorials/en/master/
            parallel/data_parallel.html#loading-datasets>`_ .
        shard_id (int, optional): The shard ID within `num_shards` . Default: ``None`` . This
            argument can only be specified when `num_shards` is also specified.
        shard_equal_rows (bool, optional): Get equal rows for all shards. Default: ``False``. If `shard_equal_rows`
            is False, the number of rows of each shard may not be equal, and may lead to a failure in distributed
            training. When the number of samples per TFRecord file are not equal, it is suggested to set it to ``True``.
            This argument should only be specified when `num_shards` is also specified.
            When `compression_type` is not ``None``, and `num_samples` or numRows (parsed from `schema` ) is provided,
            `shard_equal_rows` will be implied as ``True``.
        cache (DatasetCache, optional): Use tensor caching service to speed up dataset processing. More details:
            `Single-Node Data Cache <https://www.mindspore.cn/tutorials/en/master/dataset/cache.html>`_ .
            Default: ``None`` , which means no cache is used.
        compression_type (str, optional): The type of compression used for all files, must be either ``''``,
            ``'GZIP'``, or ``'ZLIB'``. Default: ``None`` , as in empty string. It is highly recommended to
            provide `num_samples` or numRows (parsed from `schema`) when `compression_type` is ``"GZIP"`` or
            ``"ZLIB"`` to avoid performance degradation caused by multiple decompressions of the same file
            to obtain the file size.

    Raises:
        ValueError: If dataset_files are not valid or do not exist.
        ValueError: If `num_parallel_workers` exceeds the max thread numbers.
        RuntimeError: If `num_shards` is specified but `shard_id` is None.
        RuntimeError: If `shard_id` is specified but `num_shards` is None.
        ValueError: If `shard_id` is not in range of [0, `num_shards` ).
        ValueError: If `compression_type` is not ``''``, ``'GZIP'`` or ``'ZLIB'`` .
        ValueError: If `compression_type` is provided, but the number of dataset files < `num_shards` .
        ValueError: If `num_samples` < 0.

    Examples:
        >>> import mindspore.dataset as ds
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
        >>> # 3) Get all rows from tfrecord_dataset_dir with the schema file.
        >>> dataset = ds.TFRecordDataset(dataset_files=tfrecord_dataset_dir, schema=tfrecord_schema_file)
    """

    @check_tfrecorddataset
    def __init__(self, dataset_files, schema=None, columns_list=None, num_samples=None, num_parallel_workers=None,
                 shuffle=Shuffle.GLOBAL, num_shards=None, shard_id=None, shard_equal_rows=False,
                 cache=None, compression_type=None):
        super().__init__(num_parallel_workers=num_parallel_workers, num_samples=num_samples, shuffle=shuffle,
                         num_shards=num_shards, shard_id=shard_id, cache=cache)
        if platform.system().lower() == "windows":
            raise NotImplementedError("TFRecordDataset is not supported for windows.")
        self.dataset_files = self._find_files(dataset_files)
        self.dataset_files.sort()

        self.schema = schema
        self.columns_list = replace_none(columns_list, [])
        self.shard_equal_rows = replace_none(shard_equal_rows, False)
        self.compression_type = replace_none(compression_type, "")

        # Only take numRows from schema when num_samples is not provided
        if self.schema is not None and (self.num_samples is None or self.num_samples == 0):
            self.num_samples = Schema.get_num_rows(self.schema)

        if self.compression_type in ['ZLIB', 'GZIP'] and (self.num_samples is None or self.num_samples == 0):
            logger.warning("Since compression_type is set, but neither num_samples nor numRows (from schema file) " +
                           "is provided, performance might be degraded.")

    def parse(self, children=None):
        schema = self.schema.cpp_schema if isinstance(self.schema, Schema) else self.schema
        return cde.TFRecordNode(self.dataset_files, schema, self.columns_list, self.num_samples, self.shuffle_flag,
                                self.num_shards, self.shard_id, self.shard_equal_rows, self.compression_type)


class OBSMindDataset(GeneratorDataset):
    """
    A source dataset that reads and parses MindRecord dataset which stored in cloud storage
    such as OBS, Minio or AWS S3.

    The columns of generated dataset depend on the source MindRecord files.

    Note:
        - This interface accesses the `/cache` directory for node synchronization and requires the user to ensure
          access to the `/cache` directory.

    Args:
        dataset_files (list[str]): List of files in cloud storage to be read and file path is in
            the format of s3://bucketName/objectKey.
        server (str): Endpoint for accessing cloud storage.
            If it's OBS Service of Huawei Cloud, the endpoint is
            like ``<obs.cn-north-4.myhuaweicloud.com>`` (Region cn-north-4).
            If it's Minio which starts locally, the endpoint is like ``<https://127.0.0.1:9000>``.
        ak (str): The access key ID used to access the OBS data.
        sk (str): The secret access key used to access the OBS data.
        sync_obs_path (str): Remote dir path used for synchronization, users need to
            create it on cloud storage in advance. Path is in the format of s3://bucketName/objectKey.
        columns_list (list[str], optional): List of columns to be read. Default: ``None`` , read all columns.
        shuffle (Union[bool, Shuffle], optional): Perform reshuffling of the data every epoch.
            Default: ``Shuffle.GLOBAL``. Bool type and Shuffle enum are both supported to pass in.
            If `shuffle` is ``False`` , no shuffling will be performed.
            If `shuffle` is ``True`` , performs global shuffle.
            There are three levels of shuffling, desired shuffle enum defined by :class:`mindspore.dataset.Shuffle` .

            - ``Shuffle.GLOBAL`` : Global shuffle of all rows of data in dataset, same as setting shuffle to True.

            - ``Shuffle.FILES`` : Shuffle the file sequence but keep the order of data within each file.

            - ``Shuffle.INFILE`` : Keep the file sequence the same but shuffle the data within each file.

        num_shards (int, optional): Number of shards that the dataset will be divided
            into. Default: ``None`` .
            Used in `data parallel training <https://www.mindspore.cn/tutorials/en/master/
            parallel/data_parallel.html#loading-datasets>`_ .
        shard_id (int, optional): The shard ID within num_shards. Default: ``None`` . This
            argument can only be specified when `num_shards` is also specified.
        shard_equal_rows (bool, optional): Get equal rows for all shards. Default: ``True``. If shard_equal_rows
            is false, number of rows of each shard may be not equal, and may lead to a failure in distributed training.
            When the number of samples of per MindRecord file are not equal, it is suggested to set to ``True``.
            This argument should only be specified when `num_shards` is also specified.

    Raises:
        RuntimeError: If `sync_obs_path` does not exist.
        ValueError: If `columns_list` is invalid.
        RuntimeError: If `num_shards` is specified but `shard_id` is None.
        RuntimeError: If `shard_id` is specified but `num_shards` is None.
        ValueError: If `shard_id` is not in range of [0, `num_shards` ).

    Note:
        - It's necessary to create a synchronization directory on cloud storage in
          advance which be defined by parameter: `sync_obs_path` .
        - If training is offline(no cloud), it's recommended to set the
          environment variable `BATCH_JOB_ID` .
        - In distributed training, if there are multiple nodes(servers), all 8
          devices must be used in each node(server). If there is only one
          node(server), there is no such restriction.

    Examples:
        >>> import mindspore.dataset as ds
        >>> # OBS
        >>> bucket = "iris"  # your obs bucket name
        >>> # the bucket directory structure is similar to the following:
        >>> #  - imagenet21k
        >>> #        | - mr_imagenet21k_01
        >>> #        | - mr_imagenet21k_02
        >>> #  - sync_node
        >>> dataset_obs_dir = ["s3://" + bucket + "/imagenet21k/mr_imagenet21k_01",
        ...                    "s3://" + bucket + "/imagenet21k/mr_imagenet21k_02"]
        >>> sync_obs_dir = "s3://" + bucket + "/sync_node"
        >>> num_shards = 8
        >>> shard_id = 0
        >>> dataset = ds.OBSMindDataset(dataset_obs_dir, "obs.cn-north-4.myhuaweicloud.com",
        ...                             "AK of OBS", "SK of OBS",
        ...                             sync_obs_dir, shuffle=True, num_shards=num_shards, shard_id=shard_id)
    """

    @check_obsminddataset
    def __init__(self, dataset_files, server, ak, sk, sync_obs_path,
                 columns_list=None,
                 shuffle=Shuffle.GLOBAL,
                 num_shards=None,
                 shard_id=None,
                 shard_equal_rows=True):

        from .obs.config_loader import config
        obs_config = config.copy()
        obs_config.AK = ak
        obs_config.SK = sk
        obs_config.SERVER = server
        obs_config.SYNC_OBS_PATH = sync_obs_path

        if shuffle is not None and not isinstance(shuffle, (bool, Shuffle)):
            raise TypeError("shuffle must be of boolean or enum of 'Shuffle' values like 'Shuffle.GLOBAL' or "
                            "'Shuffle.FILES'.")

        self.num_shards = replace_none(num_shards, 1)
        self.shard_id = replace_none(shard_id, 0)
        self.shuffle = replace_none(shuffle, True)

        dataset = MindRecordFromOBS(dataset_files, columns_list, shuffle, self.num_shards, self.shard_id,
                                    shard_equal_rows, obs_config=obs_config)
        if not columns_list:
            columns_list = dataset.get_col_names()
        else:
            full_columns_list = dataset.get_col_names()
            if not set(columns_list).issubset(full_columns_list):
                raise ValueError("columns_list: {} can not found in MindRecord fields: {}".format(columns_list,
                                                                                                  full_columns_list))
        super().__init__(source=dataset, column_names=columns_list, num_shards=None, shard_id=None, shuffle=False)


    def add_sampler(self, new_sampler):
        raise NotImplementedError("add_sampler is not supported for OBSMindDataset.")


    def use_sampler(self, new_sampler):
        raise NotImplementedError("use_sampler is not supported for OBSMindDataset.")
