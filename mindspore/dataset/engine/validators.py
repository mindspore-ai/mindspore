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
# See the License foNtest_resr the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""
Built-in validators.
"""
import inspect as ins
import os
import re
from functools import wraps

import numpy as np
from mindspore._c_expression import typing
from ..core.validator_helpers import parse_user_args, type_check, type_check_list, check_value, \
    INT32_MAX, check_valid_detype, check_dir, check_file, check_sampler_shuffle_shard_options, \
    validate_dataset_param_value, check_padding_options, check_gnn_list_or_ndarray, check_num_parallel_workers, \
    check_columns, check_pos_int32, check_valid_str

from . import datasets
from . import samplers
from . import cache_client


def check_imagefolderdataset(method):
    """A wrapper that wraps a parameter checker around the original Dataset(ImageFolderDataset)."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        _, param_dict = parse_user_args(method, *args, **kwargs)

        nreq_param_int = ['num_samples', 'num_parallel_workers', 'num_shards', 'shard_id']
        nreq_param_bool = ['shuffle', 'decode']
        nreq_param_list = ['extensions']
        nreq_param_dict = ['class_indexing']

        dataset_dir = param_dict.get('dataset_dir')
        check_dir(dataset_dir)

        validate_dataset_param_value(nreq_param_int, param_dict, int)
        validate_dataset_param_value(nreq_param_bool, param_dict, bool)
        validate_dataset_param_value(nreq_param_list, param_dict, list)
        validate_dataset_param_value(nreq_param_dict, param_dict, dict)
        check_sampler_shuffle_shard_options(param_dict)

        cache = param_dict.get('cache')
        check_cache_option(cache)

        return method(self, *args, **kwargs)

    return new_method


def check_mnist_cifar_dataset(method):
    """A wrapper that wraps a parameter checker around the original Dataset(ManifestDataset, Cifar10/100Dataset)."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        _, param_dict = parse_user_args(method, *args, **kwargs)

        nreq_param_int = ['num_samples', 'num_parallel_workers', 'num_shards', 'shard_id']
        nreq_param_bool = ['shuffle']

        dataset_dir = param_dict.get('dataset_dir')
        check_dir(dataset_dir)

        usage = param_dict.get('usage')
        if usage is not None:
            check_valid_str(usage, ["train", "test", "all"], "usage")

        validate_dataset_param_value(nreq_param_int, param_dict, int)
        validate_dataset_param_value(nreq_param_bool, param_dict, bool)

        check_sampler_shuffle_shard_options(param_dict)

        cache = param_dict.get('cache')
        check_cache_option(cache)

        return method(self, *args, **kwargs)

    return new_method


def check_manifestdataset(method):
    """A wrapper that wraps a parameter checker around the original Dataset(ManifestDataset)."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        _, param_dict = parse_user_args(method, *args, **kwargs)

        nreq_param_int = ['num_samples', 'num_parallel_workers', 'num_shards', 'shard_id']
        nreq_param_bool = ['shuffle', 'decode']
        nreq_param_str = ['usage']
        nreq_param_dict = ['class_indexing']

        dataset_file = param_dict.get('dataset_file')
        check_file(dataset_file)

        validate_dataset_param_value(nreq_param_int, param_dict, int)
        validate_dataset_param_value(nreq_param_bool, param_dict, bool)
        validate_dataset_param_value(nreq_param_str, param_dict, str)
        validate_dataset_param_value(nreq_param_dict, param_dict, dict)

        check_sampler_shuffle_shard_options(param_dict)

        cache = param_dict.get('cache')
        check_cache_option(cache)

        return method(self, *args, **kwargs)

    return new_method


def check_tfrecorddataset(method):
    """A wrapper that wraps a parameter checker around the original Dataset(TFRecordDataset)."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        _, param_dict = parse_user_args(method, *args, **kwargs)

        nreq_param_int = ['num_samples', 'num_parallel_workers', 'num_shards', 'shard_id']
        nreq_param_list = ['columns_list']
        nreq_param_bool = ['shard_equal_rows']

        dataset_files = param_dict.get('dataset_files')
        if not isinstance(dataset_files, (str, list)):
            raise TypeError("dataset_files should be type str or a list of strings.")

        validate_dataset_param_value(nreq_param_int, param_dict, int)
        validate_dataset_param_value(nreq_param_list, param_dict, list)
        validate_dataset_param_value(nreq_param_bool, param_dict, bool)

        check_sampler_shuffle_shard_options(param_dict)

        cache = param_dict.get('cache')
        check_cache_option(cache)

        return method(self, *args, **kwargs)

    return new_method


def check_vocdataset(method):
    """A wrapper that wraps a parameter checker around the original Dataset(VOCDataset)."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        _, param_dict = parse_user_args(method, *args, **kwargs)

        nreq_param_int = ['num_samples', 'num_parallel_workers', 'num_shards', 'shard_id']
        nreq_param_bool = ['shuffle', 'decode']
        nreq_param_dict = ['class_indexing']

        dataset_dir = param_dict.get('dataset_dir')
        check_dir(dataset_dir)

        task = param_dict.get('task')
        type_check(task, (str,), "task")

        usage = param_dict.get('usage')
        type_check(usage, (str,), "usage")

        if task == "Segmentation":
            imagesets_file = os.path.join(dataset_dir, "ImageSets", "Segmentation", usage + ".txt")
            if param_dict.get('class_indexing') is not None:
                raise ValueError("class_indexing is not supported in Segmentation task.")
        elif task == "Detection":
            imagesets_file = os.path.join(dataset_dir, "ImageSets", "Main", usage + ".txt")
        else:
            raise ValueError("Invalid task : " + task + ".")

        check_file(imagesets_file)

        validate_dataset_param_value(nreq_param_int, param_dict, int)
        validate_dataset_param_value(nreq_param_bool, param_dict, bool)
        validate_dataset_param_value(nreq_param_dict, param_dict, dict)
        check_sampler_shuffle_shard_options(param_dict)

        cache = param_dict.get('cache')
        check_cache_option(cache)

        return method(self, *args, **kwargs)

    return new_method


def check_cocodataset(method):
    """A wrapper that wraps a parameter checker around the original Dataset(CocoDataset)."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        _, param_dict = parse_user_args(method, *args, **kwargs)

        nreq_param_int = ['num_samples', 'num_parallel_workers', 'num_shards', 'shard_id']
        nreq_param_bool = ['shuffle', 'decode']

        dataset_dir = param_dict.get('dataset_dir')
        check_dir(dataset_dir)

        annotation_file = param_dict.get('annotation_file')
        check_file(annotation_file)

        task = param_dict.get('task')
        type_check(task, (str,), "task")

        if task not in {'Detection', 'Stuff', 'Panoptic', 'Keypoint'}:
            raise ValueError("Invalid task type: " + task + ".")

        validate_dataset_param_value(nreq_param_int, param_dict, int)

        validate_dataset_param_value(nreq_param_bool, param_dict, bool)

        sampler = param_dict.get('sampler')
        if sampler is not None and isinstance(sampler, samplers.PKSampler):
            raise ValueError("CocoDataset doesn't support PKSampler.")
        check_sampler_shuffle_shard_options(param_dict)

        cache = param_dict.get('cache')
        check_cache_option(cache)

        return method(self, *args, **kwargs)

    return new_method


def check_celebadataset(method):
    """A wrapper that wraps a parameter checker around the original Dataset(CelebADataset)."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        _, param_dict = parse_user_args(method, *args, **kwargs)

        nreq_param_int = ['num_samples', 'num_parallel_workers', 'num_shards', 'shard_id']
        nreq_param_bool = ['shuffle', 'decode']
        nreq_param_list = ['extensions']
        nreq_param_str = ['dataset_type']

        dataset_dir = param_dict.get('dataset_dir')

        check_dir(dataset_dir)

        validate_dataset_param_value(nreq_param_int, param_dict, int)
        validate_dataset_param_value(nreq_param_bool, param_dict, bool)
        validate_dataset_param_value(nreq_param_list, param_dict, list)
        validate_dataset_param_value(nreq_param_str, param_dict, str)

        usage = param_dict.get('usage')
        if usage is not None and usage not in ('all', 'train', 'valid', 'test'):
            raise ValueError("usage should be 'all', 'train', 'valid' or 'test'.")

        check_sampler_shuffle_shard_options(param_dict)

        sampler = param_dict.get('sampler')
        if sampler is not None and isinstance(sampler, samplers.PKSampler):
            raise ValueError("CelebADataset doesn't support PKSampler.")

        cache = param_dict.get('cache')
        check_cache_option(cache)

        return method(self, *args, **kwargs)

    return new_method


def check_save(method):
    """A wrapper that wraps a parameter checker around the saved operator."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        _, param_dict = parse_user_args(method, *args, **kwargs)

        nreq_param_int = ['num_files']
        nreq_param_str = ['file_name', 'file_type']
        validate_dataset_param_value(nreq_param_int, param_dict, int)
        if (param_dict.get('num_files') <= 0 or param_dict.get('num_files') > 1000):
            raise ValueError("num_files should between {} and {}.".format(1, 1000))
        validate_dataset_param_value(nreq_param_str, param_dict, str)
        if param_dict.get('file_type') != 'mindrecord':
            raise ValueError("{} dataset format is not supported.".format(param_dict.get('file_type')))
        return method(self, *args, **kwargs)

    return new_method


def check_tuple_iterator(method):
    """A wrapper that wraps a parameter checker around the original create_tuple_iterator and create_dict_iterator."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        [columns, num_epochs, _, _], param_dict = parse_user_args(method, *args, **kwargs)
        nreq_param_bool = ['output_numpy']
        validate_dataset_param_value(nreq_param_bool, param_dict, bool)
        if num_epochs is not None:
            type_check(num_epochs, (int,), "num_epochs")
            check_value(num_epochs, [-1, INT32_MAX], "num_epochs")

        if columns is not None:
            check_columns(columns, "column_names")

        return method(self, *args, **kwargs)

    return new_method


def check_dict_iterator(method):
    """A wrapper that wraps a parameter checker around the original create_tuple_iterator and create_dict_iterator."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        [num_epochs, _], param_dict = parse_user_args(method, *args, **kwargs)
        nreq_param_bool = ['output_numpy']
        validate_dataset_param_value(nreq_param_bool, param_dict, bool)
        if num_epochs is not None:
            type_check(num_epochs, (int,), "num_epochs")
            check_value(num_epochs, [-1, INT32_MAX], "num_epochs")

        return method(self, *args, **kwargs)

    return new_method


def check_minddataset(method):
    """A wrapper that wraps a parameter checker around the original Dataset(MindDataset)."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        _, param_dict = parse_user_args(method, *args, **kwargs)

        nreq_param_int = ['num_samples', 'num_parallel_workers', 'seed', 'num_shards', 'shard_id', 'num_padded']
        nreq_param_list = ['columns_list']
        nreq_param_dict = ['padded_sample']

        dataset_file = param_dict.get('dataset_file')
        if isinstance(dataset_file, list):
            if len(dataset_file) > 4096:
                raise ValueError("length of dataset_file should less than or equal to {}.".format(4096))
            for f in dataset_file:
                check_file(f)
        else:
            check_file(dataset_file)

        validate_dataset_param_value(nreq_param_int, param_dict, int)
        validate_dataset_param_value(nreq_param_list, param_dict, list)
        validate_dataset_param_value(nreq_param_dict, param_dict, dict)

        check_sampler_shuffle_shard_options(param_dict)

        check_padding_options(param_dict)
        return method(self, *args, **kwargs)

    return new_method


def check_generatordataset(method):
    """A wrapper that wraps a parameter checker around the original Dataset(GeneratorDataset)."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        _, param_dict = parse_user_args(method, *args, **kwargs)

        source = param_dict.get('source')

        if not callable(source):
            try:
                iter(source)
            except TypeError:
                raise TypeError("source should be callable, iterable or random accessible.")

        column_names = param_dict.get('column_names')
        if column_names is not None:
            check_columns(column_names, "column_names")
        schema = param_dict.get('schema')
        if column_names is None and schema is None:
            raise ValueError("Neither columns_names nor schema are provided.")

        if schema is not None:
            if not isinstance(schema, datasets.Schema) and not isinstance(schema, str):
                raise ValueError("schema should be a path to schema file or a schema object.")

        # check optional argument
        nreq_param_int = ["num_samples", "num_parallel_workers", "num_shards", "shard_id"]
        validate_dataset_param_value(nreq_param_int, param_dict, int)
        nreq_param_list = ["column_types"]
        validate_dataset_param_value(nreq_param_list, param_dict, list)
        nreq_param_bool = ["shuffle"]
        validate_dataset_param_value(nreq_param_bool, param_dict, bool)

        num_shards = param_dict.get("num_shards")
        shard_id = param_dict.get("shard_id")
        if (num_shards is None) != (shard_id is None):
            # These two parameters appear together.
            raise ValueError("num_shards and shard_id need to be passed in together.")
        if num_shards is not None:
            check_pos_int32(num_shards, "num_shards")
            if shard_id >= num_shards:
                raise ValueError("shard_id should be less than num_shards.")

        sampler = param_dict.get("sampler")
        if sampler is not None:
            if isinstance(sampler, samplers.PKSampler):
                raise ValueError("GeneratorDataset doesn't support PKSampler.")
            if not isinstance(sampler, samplers.BuiltinSampler):
                try:
                    iter(sampler)
                except TypeError:
                    raise TypeError("sampler should be either iterable or from mindspore.dataset.samplers.")

        if sampler is not None and not hasattr(source, "__getitem__"):
            raise ValueError("sampler is not supported if source does not have attribute '__getitem__'.")
        if num_shards is not None and not hasattr(source, "__getitem__"):
            raise ValueError("num_shards is not supported if source does not have attribute '__getitem__'.")

        return method(self, *args, **kwargs)

    return new_method


def check_random_dataset(method):
    """A wrapper that wraps a parameter checker around the original Dataset(RandomDataset)."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        _, param_dict = parse_user_args(method, *args, **kwargs)

        nreq_param_int = ['num_samples', 'num_parallel_workers', 'num_shards', 'shard_id', 'total_rows']
        nreq_param_bool = ['shuffle']
        nreq_param_list = ['columns_list']

        validate_dataset_param_value(nreq_param_int, param_dict, int)
        validate_dataset_param_value(nreq_param_bool, param_dict, bool)
        validate_dataset_param_value(nreq_param_list, param_dict, list)

        check_sampler_shuffle_shard_options(param_dict)

        cache = param_dict.get('cache')
        check_cache_option(cache)

        return method(self, *args, **kwargs)

    return new_method


def check_pad_info(key, val):
    """check the key and value pair of pad_info in batch"""
    type_check(key, (str,), "key in pad_info")

    if val is not None:
        assert len(val) == 2, "value of pad_info should be a tuple of size 2."
        type_check(val, (tuple,), "value in pad_info")

        if val[0] is not None:
            type_check(val[0], (list,), "pad_shape")

            for dim in val[0]:
                if dim is not None:
                    type_check(dim, (int,), "dim in pad_shape")
                    assert dim > 0, "pad shape should be positive integers"
        if val[1] is not None:
            type_check(val[1], (int, float, str, bytes), "pad_value")


def check_bucket_batch_by_length(method):
    """check the input arguments of bucket_batch_by_length."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        [column_names, bucket_boundaries, bucket_batch_sizes, element_length_function, pad_info,
         pad_to_bucket_boundary, drop_remainder], _ = parse_user_args(method, *args, **kwargs)

        nreq_param_list = ['column_names', 'bucket_boundaries', 'bucket_batch_sizes']

        type_check_list([column_names, bucket_boundaries, bucket_batch_sizes], (list,), nreq_param_list)

        nbool_param_list = ['pad_to_bucket_boundary', 'drop_remainder']
        type_check_list([pad_to_bucket_boundary, drop_remainder], (bool,), nbool_param_list)

        # check column_names: must be list of string.
        check_columns(column_names, "column_names")

        if element_length_function is None and len(column_names) != 1:
            raise ValueError("If element_length_function is not specified, exactly one column name should be passed.")

        # check bucket_boundaries: must be list of int, positive and strictly increasing
        if not bucket_boundaries:
            raise ValueError("bucket_boundaries cannot be empty.")

        all_int = all(isinstance(item, int) for item in bucket_boundaries)
        if not all_int:
            raise TypeError("bucket_boundaries should be a list of int.")

        all_non_negative = all(item > 0 for item in bucket_boundaries)
        if not all_non_negative:
            raise ValueError("bucket_boundaries must only contain positive numbers.")

        for i in range(len(bucket_boundaries) - 1):
            if not bucket_boundaries[i + 1] > bucket_boundaries[i]:
                raise ValueError("bucket_boundaries should be strictly increasing.")

        # check bucket_batch_sizes: must be list of int and positive
        if len(bucket_batch_sizes) != len(bucket_boundaries) + 1:
            raise ValueError("bucket_batch_sizes must contain one element more than bucket_boundaries.")

        all_int = all(isinstance(item, int) for item in bucket_batch_sizes)
        if not all_int:
            raise TypeError("bucket_batch_sizes should be a list of int.")

        all_non_negative = all(item > 0 for item in bucket_batch_sizes)
        if not all_non_negative:
            raise ValueError("bucket_batch_sizes should be a list of positive numbers.")

        if pad_info is not None:
            type_check(pad_info, (dict,), "pad_info")

            for k, v in pad_info.items():
                check_pad_info(k, v)

        return method(self, *args, **kwargs)

    return new_method


def check_batch(method):
    """check the input arguments of batch."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        [batch_size, drop_remainder, num_parallel_workers, per_batch_map, input_columns, output_columns,
         column_order, pad_info, python_multiprocessing], param_dict = parse_user_args(method, *args, **kwargs)

        if not (isinstance(batch_size, int) or (callable(batch_size))):
            raise TypeError("batch_size should either be an int or a callable.")

        if callable(batch_size):
            sig = ins.signature(batch_size)
            if len(sig.parameters) != 1:
                raise ValueError("callable batch_size should take one parameter (BatchInfo).")
        else:
            check_pos_int32(int(batch_size), "batch_size")

        if num_parallel_workers is not None:
            check_num_parallel_workers(num_parallel_workers)
        type_check(drop_remainder, (bool,), "drop_remainder")

        if (pad_info is not None) and (per_batch_map is not None):
            raise ValueError("pad_info and per_batch_map can't both be set.")

        if pad_info is not None:
            type_check(param_dict["pad_info"], (dict,), "pad_info")
            for k, v in param_dict.get('pad_info').items():
                check_pad_info(k, v)

        if (per_batch_map is None) != (input_columns is None):
            # These two parameters appear together.
            raise ValueError("per_batch_map and input_columns need to be passed in together.")

        if input_columns is not None:
            check_columns(input_columns, "input_columns")
            if len(input_columns) != (len(ins.signature(per_batch_map).parameters) - 1):
                raise ValueError("The signature of per_batch_map should match with input columns.")

        if output_columns is not None:
            check_columns(output_columns, "output_columns")

        if column_order is not None:
            check_columns(column_order, "column_order")

        if python_multiprocessing is not None:
            type_check(python_multiprocessing, (bool,), "python_multiprocessing")

        return method(self, *args, **kwargs)

    return new_method


def check_sync_wait(method):
    """check the input arguments of sync_wait."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        [condition_name, num_batch, _], _ = parse_user_args(method, *args, **kwargs)

        type_check(condition_name, (str,), "condition_name")
        type_check(num_batch, (int,), "num_batch")

        return method(self, *args, **kwargs)

    return new_method


def check_shuffle(method):
    """check the input arguments of shuffle."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        [buffer_size], _ = parse_user_args(method, *args, **kwargs)

        type_check(buffer_size, (int,), "buffer_size")

        check_value(buffer_size, [2, INT32_MAX], "buffer_size")

        return method(self, *args, **kwargs)

    return new_method


def check_map(method):
    """check the input arguments of map."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        from mindspore.dataset.callback import DSCallback
        [_, input_columns, output_columns, column_order, num_parallel_workers, python_multiprocessing, cache,
         callbacks], _ = \
            parse_user_args(method, *args, **kwargs)

        nreq_param_columns = ['input_columns', 'output_columns', 'column_order']

        if column_order is not None:
            type_check(column_order, (list,), "column_order")
        if num_parallel_workers is not None:
            check_num_parallel_workers(num_parallel_workers)
        type_check(python_multiprocessing, (bool,), "python_multiprocessing")
        check_cache_option(cache)

        if callbacks is not None:
            if isinstance(callbacks, (list, tuple)):
                type_check_list(callbacks, (DSCallback,), "callbacks")
            else:
                type_check(callbacks, (DSCallback,), "callbacks")

        for param_name, param in zip(nreq_param_columns, [input_columns, output_columns, column_order]):
            if param is not None:
                check_columns(param, param_name)
        if callbacks is not None:
            type_check(callbacks, (list, DSCallback), "callbacks")

        return method(self, *args, **kwargs)

    return new_method


def check_filter(method):
    """"check the input arguments of filter."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        [predicate, input_columns, num_parallel_workers], _ = parse_user_args(method, *args, **kwargs)
        if not callable(predicate):
            raise TypeError("Predicate should be a Python function or a callable Python object.")

        if num_parallel_workers is not None:
            check_num_parallel_workers(num_parallel_workers)

        if input_columns is not None:
            check_columns(input_columns, "input_columns")

        return method(self, *args, **kwargs)

    return new_method


def check_repeat(method):
    """check the input arguments of repeat."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        [count], _ = parse_user_args(method, *args, **kwargs)

        type_check(count, (int, type(None)), "repeat")
        if isinstance(count, int):
            if (count <= 0 and count != -1) or count > INT32_MAX:
                raise ValueError("count should be either -1 or positive integer.")
        return method(self, *args, **kwargs)

    return new_method


def check_skip(method):
    """check the input arguments of skip."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        [count], _ = parse_user_args(method, *args, **kwargs)

        type_check(count, (int,), "count")
        check_value(count, (-1, INT32_MAX), "count")

        return method(self, *args, **kwargs)

    return new_method


def check_take(method):
    """check the input arguments of take."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        [count], _ = parse_user_args(method, *args, **kwargs)
        type_check(count, (int,), "count")
        if (count <= 0 and count != -1) or count > INT32_MAX:
            raise ValueError("count should be either -1 or positive integer.")

        return method(self, *args, **kwargs)

    return new_method


def check_positive_int32(method):
    """check whether the input argument is positive and int, only works for functions with one input."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        [count], param_dict = parse_user_args(method, *args, **kwargs)
        para_name = None
        for key in list(param_dict.keys()):
            if key not in ['self', 'cls']:
                para_name = key
        # Need to get default value of param
        if count is not None:
            check_pos_int32(count, para_name)

        return method(self, *args, **kwargs)

    return new_method


def check_device_send(method):
    """check the input argument for to_device and device_que."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        param, param_dict = parse_user_args(method, *args, **kwargs)
        para_list = list(param_dict.keys())
        if "prefetch_size" in para_list:
            if param[0] is not None:
                check_pos_int32(param[0], "prefetch_size")
            type_check(param[1], (bool,), "send_epoch_end")
        else:
            type_check(param[0], (bool,), "send_epoch_end")

        return method(self, *args, **kwargs)

    return new_method


def check_zip(method):
    """check the input arguments of zip."""

    @wraps(method)
    def new_method(*args, **kwargs):
        [ds], _ = parse_user_args(method, *args, **kwargs)
        type_check(ds, (tuple,), "datasets")

        return method(*args, **kwargs)

    return new_method


def check_zip_dataset(method):
    """check the input arguments of zip method in `Dataset`."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        [ds], _ = parse_user_args(method, *args, **kwargs)
        type_check(ds, (tuple, datasets.Dataset), "datasets")

        return method(self, *args, **kwargs)

    return new_method


def check_concat(method):
    """check the input arguments of concat method in `Dataset`."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        [ds], _ = parse_user_args(method, *args, **kwargs)
        type_check(ds, (list, datasets.Dataset), "datasets")
        if isinstance(ds, list):
            type_check_list(ds, (datasets.Dataset,), "dataset")
        return method(self, *args, **kwargs)

    return new_method


def check_rename(method):
    """check the input arguments of rename."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        values, _ = parse_user_args(method, *args, **kwargs)

        req_param_columns = ['input_columns', 'output_columns']
        for param_name, param in zip(req_param_columns, values):
            check_columns(param, param_name)

        input_size, output_size = 1, 1
        input_columns, output_columns = values
        if isinstance(input_columns, list):
            input_size = len(input_columns)
        if isinstance(output_columns, list):
            output_size = len(output_columns)
        if input_size != output_size:
            raise ValueError("Number of column in input_columns and output_columns is not equal.")

        return method(self, *args, **kwargs)

    return new_method


def check_project(method):
    """check the input arguments of project."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        [columns], _ = parse_user_args(method, *args, **kwargs)
        check_columns(columns, 'columns')

        return method(self, *args, **kwargs)

    return new_method


def check_schema(method):
    """check the input arguments of Schema.__init__."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        [schema_file], _ = parse_user_args(method, *args, **kwargs)

        if schema_file is not None:
            type_check(schema_file, (str,), "schema_file")
            check_file(schema_file)

        return method(self, *args, **kwargs)

    return new_method


def check_add_column(method):
    """check the input arguments of add_column."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        [name, de_type, shape], _ = parse_user_args(method, *args, **kwargs)

        type_check(name, (str,), "name")

        if not name:
            raise TypeError("Expected non-empty string for column name.")

        if de_type is not None:
            if not isinstance(de_type, typing.Type) and not check_valid_detype(de_type):
                raise TypeError("Unknown column type: {}.".format(de_type))
        else:
            raise TypeError("Expected non-empty string for de_type.")

        if shape is not None:
            type_check(shape, (list,), "shape")
            type_check_list(shape, (int,), "shape")

        return method(self, *args, **kwargs)

    return new_method


def check_cluedataset(method):
    """A wrapper that wraps a parameter checker around the original Dataset(CLUEDataset)."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        _, param_dict = parse_user_args(method, *args, **kwargs)

        nreq_param_int = ['num_samples', 'num_parallel_workers', 'num_shards', 'shard_id']

        dataset_files = param_dict.get('dataset_files')
        type_check(dataset_files, (str, list), "dataset files")

        # check task
        task_param = param_dict.get('task')
        if task_param not in ['AFQMC', 'TNEWS', 'IFLYTEK', 'CMNLI', 'WSC', 'CSL']:
            raise ValueError("task should be 'AFQMC', 'TNEWS', 'IFLYTEK', 'CMNLI', 'WSC' or 'CSL'.")

        # check usage
        usage_param = param_dict.get('usage')
        if usage_param not in ['train', 'test', 'eval']:
            raise ValueError("usage should be 'train', 'test' or 'eval'.")

        validate_dataset_param_value(nreq_param_int, param_dict, int)
        check_sampler_shuffle_shard_options(param_dict)

        cache = param_dict.get('cache')
        check_cache_option(cache)

        return method(self, *args, **kwargs)

    return new_method


def check_csvdataset(method):
    """A wrapper that wraps a parameter checker around the original Dataset(CSVDataset)."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        _, param_dict = parse_user_args(method, *args, **kwargs)

        nreq_param_int = ['num_samples', 'num_parallel_workers', 'num_shards', 'shard_id']

        # check dataset_files; required argument
        dataset_files = param_dict.get('dataset_files')
        type_check(dataset_files, (str, list), "dataset files")

        # check field_delim
        field_delim = param_dict.get('field_delim')
        if field_delim is not None:
            type_check(field_delim, (str,), 'field delim')
            if field_delim in ['"', '\r', '\n'] or len(field_delim) > 1:
                raise ValueError("field_delim is invalid.")

        # check column_defaults
        column_defaults = param_dict.get('column_defaults')
        if column_defaults is not None:
            if not isinstance(column_defaults, list):
                raise TypeError("column_defaults should be type of list.")
            for item in column_defaults:
                if not isinstance(item, (str, int, float)):
                    raise TypeError("column type in column_defaults is invalid.")

        # check column_names: must be list of string.
        column_names = param_dict.get("column_names")
        if column_names is not None:
            all_string = all(isinstance(item, str) for item in column_names)
            if not all_string:
                raise TypeError("column_names should be a list of str.")

        validate_dataset_param_value(nreq_param_int, param_dict, int)
        check_sampler_shuffle_shard_options(param_dict)

        cache = param_dict.get('cache')
        check_cache_option(cache)

        return method(self, *args, **kwargs)

    return new_method


def check_textfiledataset(method):
    """A wrapper that wraps a parameter checker around the original Dataset(TextFileDataset)."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        _, param_dict = parse_user_args(method, *args, **kwargs)

        nreq_param_int = ['num_samples', 'num_parallel_workers', 'num_shards', 'shard_id']

        dataset_files = param_dict.get('dataset_files')
        type_check(dataset_files, (str, list), "dataset files")
        validate_dataset_param_value(nreq_param_int, param_dict, int)
        check_sampler_shuffle_shard_options(param_dict)

        cache = param_dict.get('cache')
        check_cache_option(cache)

        return method(self, *args, **kwargs)

    return new_method


def check_split(method):
    """check the input arguments of split."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        [sizes, randomize], _ = parse_user_args(method, *args, **kwargs)

        type_check(sizes, (list,), "sizes")
        type_check(randomize, (bool,), "randomize")

        # check sizes: must be list of float or list of int
        if not sizes:
            raise ValueError("sizes cannot be empty.")

        all_int = all(isinstance(item, int) for item in sizes)
        all_float = all(isinstance(item, float) for item in sizes)

        if not (all_int or all_float):
            raise ValueError("sizes should be list of int or list of float.")

        if all_int:
            all_positive = all(item > 0 for item in sizes)
            if not all_positive:
                raise ValueError("sizes is a list of int, but there should be no negative or zero numbers.")

        if all_float:
            all_valid_percentages = all(0 < item <= 1 for item in sizes)
            if not all_valid_percentages:
                raise ValueError("sizes is a list of float, but there should be no numbers outside the range (0, 1].")

            epsilon = 0.00001
            if not abs(sum(sizes) - 1) < epsilon:
                raise ValueError("sizes is a list of float, but the percentages do not sum up to 1.")

        return method(self, *args, **kwargs)

    return new_method


def check_hostname(hostname):
    if not hostname or len(hostname) > 255:
        return False
    if hostname[-1] == ".":
        hostname = hostname[:-1]  # strip exactly one dot from the right, if present
    allowed = re.compile("(?!-)[A-Z\\d-]{1,63}(?<!-)$", re.IGNORECASE)
    return all(allowed.match(x) for x in hostname.split("."))


def check_gnn_graphdata(method):
    """check the input arguments of graphdata."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        [dataset_file, num_parallel_workers, working_mode, hostname,
         port, num_client, auto_shutdown], _ = parse_user_args(method, *args, **kwargs)
        check_file(dataset_file)
        if num_parallel_workers is not None:
            check_num_parallel_workers(num_parallel_workers)
        type_check(hostname, (str,), "hostname")
        if check_hostname(hostname) is False:
            raise ValueError("The hostname is illegal")
        type_check(working_mode, (str,), "working_mode")
        if working_mode not in {'local', 'client', 'server'}:
            raise ValueError("Invalid working mode, please enter 'local', 'client' or 'server'.")
        type_check(port, (int,), "port")
        check_value(port, (1024, 65535), "port")
        type_check(num_client, (int,), "num_client")
        check_value(num_client, (1, 255), "num_client")
        type_check(auto_shutdown, (bool,), "auto_shutdown")
        return method(self, *args, **kwargs)

    return new_method


def check_gnn_get_all_nodes(method):
    """A wrapper that wraps a parameter checker around the GNN `get_all_nodes` function."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        [node_type], _ = parse_user_args(method, *args, **kwargs)
        type_check(node_type, (int,), "node_type")

        return method(self, *args, **kwargs)

    return new_method


def check_gnn_get_all_edges(method):
    """A wrapper that wraps a parameter checker around the GNN `get_all_edges` function."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        [edge_type], _ = parse_user_args(method, *args, **kwargs)
        type_check(edge_type, (int,), "edge_type")

        return method(self, *args, **kwargs)

    return new_method


def check_gnn_get_nodes_from_edges(method):
    """A wrapper that wraps a parameter checker around the GNN `get_nodes_from_edges` function."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        [edge_list], _ = parse_user_args(method, *args, **kwargs)
        check_gnn_list_or_ndarray(edge_list, "edge_list")

        return method(self, *args, **kwargs)

    return new_method


def check_gnn_get_all_neighbors(method):
    """A wrapper that wraps a parameter checker around the GNN `get_all_neighbors` function."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        [node_list, neighbour_type], _ = parse_user_args(method, *args, **kwargs)

        check_gnn_list_or_ndarray(node_list, 'node_list')
        type_check(neighbour_type, (int,), "neighbour_type")

        return method(self, *args, **kwargs)

    return new_method


def check_gnn_get_sampled_neighbors(method):
    """A wrapper that wraps a parameter checker around the GNN `get_sampled_neighbors` function."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        [node_list, neighbor_nums, neighbor_types, _], _ = parse_user_args(method, *args, **kwargs)

        check_gnn_list_or_ndarray(node_list, 'node_list')

        check_gnn_list_or_ndarray(neighbor_nums, 'neighbor_nums')
        if not neighbor_nums or len(neighbor_nums) > 6:
            raise ValueError("Wrong number of input members for {0}, should be between 1 and 6, got {1}.".format(
                'neighbor_nums', len(neighbor_nums)))

        check_gnn_list_or_ndarray(neighbor_types, 'neighbor_types')
        if not neighbor_types or len(neighbor_types) > 6:
            raise ValueError("Wrong number of input members for {0}, should be between 1 and 6, got {1}.".format(
                'neighbor_types', len(neighbor_types)))

        if len(neighbor_nums) != len(neighbor_types):
            raise ValueError(
                "The number of members of neighbor_nums and neighbor_types is inconsistent.")

        return method(self, *args, **kwargs)

    return new_method


def check_gnn_get_neg_sampled_neighbors(method):
    """A wrapper that wraps a parameter checker around the GNN `get_neg_sampled_neighbors` function."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        [node_list, neg_neighbor_num, neg_neighbor_type], _ = parse_user_args(method, *args, **kwargs)

        check_gnn_list_or_ndarray(node_list, 'node_list')
        type_check(neg_neighbor_num, (int,), "neg_neighbor_num")
        type_check(neg_neighbor_type, (int,), "neg_neighbor_type")

        return method(self, *args, **kwargs)

    return new_method


def check_gnn_random_walk(method):
    """A wrapper that wraps a parameter checker around the GNN `random_walk` function."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        [target_nodes, meta_path, step_home_param, step_away_param, default_node], _ = parse_user_args(method, *args,
                                                                                                       **kwargs)
        check_gnn_list_or_ndarray(target_nodes, 'target_nodes')
        check_gnn_list_or_ndarray(meta_path, 'meta_path')
        type_check(step_home_param, (float,), "step_home_param")
        type_check(step_away_param, (float,), "step_away_param")
        type_check(default_node, (int,), "default_node")
        check_value(default_node, (-1, INT32_MAX), "default_node")

        return method(self, *args, **kwargs)

    return new_method


def check_aligned_list(param, param_name, member_type):
    """Check whether the structure of each member of the list is the same."""

    type_check(param, (list,), "param")
    if not param:
        raise TypeError(
            "Parameter {0} or its members are empty".format(param_name))
    member_have_list = None
    list_len = None
    for member in param:
        if isinstance(member, list):
            check_aligned_list(member, param_name, member_type)

            if member_have_list not in (None, True):
                raise TypeError("The type of each member of the parameter {0} is inconsistent.".format(
                    param_name))
            if list_len is not None and len(member) != list_len:
                raise TypeError("The size of each member of parameter {0} is inconsistent.".format(
                    param_name))
            member_have_list = True
            list_len = len(member)
        else:
            type_check(member, (member_type,), param_name)
            if member_have_list not in (None, False):
                raise TypeError("The type of each member of the parameter {0} is inconsistent.".format(
                    param_name))
            member_have_list = False


def check_gnn_get_node_feature(method):
    """A wrapper that wraps a parameter checker around the GNN `get_node_feature` function."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        [node_list, feature_types], _ = parse_user_args(method, *args, **kwargs)

        type_check(node_list, (list, np.ndarray), "node_list")
        if isinstance(node_list, list):
            check_aligned_list(node_list, 'node_list', int)
        elif isinstance(node_list, np.ndarray):
            if not node_list.dtype == np.int32:
                raise TypeError("Each member in {0} should be of type int32. Got {1}.".format(
                    node_list, node_list.dtype))

        check_gnn_list_or_ndarray(feature_types, 'feature_types')

        return method(self, *args, **kwargs)

    return new_method


def check_gnn_get_edge_feature(method):
    """A wrapper that wraps a parameter checker around the GNN `get_edge_feature` function."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        [edge_list, feature_types], _ = parse_user_args(method, *args, **kwargs)

        type_check(edge_list, (list, np.ndarray), "edge_list")
        if isinstance(edge_list, list):
            check_aligned_list(edge_list, 'edge_list', int)
        elif isinstance(edge_list, np.ndarray):
            if not edge_list.dtype == np.int32:
                raise TypeError("Each member in {0} should be of type int32. Got {1}.".format(
                    edge_list, edge_list.dtype))

        check_gnn_list_or_ndarray(feature_types, 'feature_types')

        return method(self, *args, **kwargs)

    return new_method


def check_numpyslicesdataset(method):
    """A wrapper that wraps a parameter checker around the original Dataset(NumpySlicesDataset)."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        _, param_dict = parse_user_args(method, *args, **kwargs)

        data = param_dict.get("data")
        column_names = param_dict.get("column_names")
        if not data:
            raise ValueError("Argument data cannot be empty")
        type_check(data, (list, tuple, dict, np.ndarray), "data")
        if isinstance(data, tuple):
            type_check(data[0], (list, np.ndarray), "data[0]")

        # check column_names
        if column_names is not None:
            check_columns(column_names, "column_names")

            # check num of input column in column_names
            column_num = 1 if isinstance(column_names, str) else len(column_names)
            if isinstance(data, dict):
                data_column = len(list(data.keys()))
                if column_num != data_column:
                    raise ValueError("Num of input column names is {0}, but required is {1}."
                                     .format(column_num, data_column))

            elif isinstance(data, tuple):
                if column_num != len(data):
                    raise ValueError("Num of input column names is {0}, but required is {1}."
                                     .format(column_num, len(data)))
            else:
                if column_num != 1:
                    raise ValueError("Num of input column names is {0}, but required is {1} as data is list."
                                     .format(column_num, 1))

        return method(self, *args, **kwargs)

    return new_method


def check_paddeddataset(method):
    """A wrapper that wraps a parameter checker around the original Dataset(PaddedDataset)."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        _, param_dict = parse_user_args(method, *args, **kwargs)

        padded_samples = param_dict.get("padded_samples")
        if not padded_samples:
            raise ValueError("padded_samples cannot be empty.")
        type_check(padded_samples, (list,), "padded_samples")
        type_check(padded_samples[0], (dict,), "padded_element")
        return method(self, *args, **kwargs)

    return new_method


def check_cache_option(cache):
    """Sanity check for cache parameter"""
    if cache is not None:
        type_check(cache, (cache_client.DatasetCache,), "cache")


def check_to_device_send(method):
    """Check the input arguments of send function for TransferDataset."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        [num_epochs], _ = parse_user_args(method, *args, **kwargs)

        if num_epochs is not None:
            type_check(num_epochs, (int,), "num_epochs")
            check_value(num_epochs, [-1, INT32_MAX], "num_epochs")

        return method(self, *args, **kwargs)

    return new_method
