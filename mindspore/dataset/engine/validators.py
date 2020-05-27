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

"""Built-in validators.
"""
import inspect as ins
import os
from functools import wraps
from multiprocessing import cpu_count
import numpy as np
from mindspore._c_expression import typing
from . import samplers
from . import datasets

INT32_MAX = 2147483647
valid_detype = [
    "bool", "int8", "int16", "int32", "int64", "uint8", "uint16",
    "uint32", "uint64", "float16", "float32", "float64", "string"
]


def check(method):
    """Check the function parameters and return the function ."""
    func_name = method.__name__
    # Required parameter
    req_param_int = []
    req_param_bool = []
    # Non-required parameter
    nreq_param_int = []
    nreq_param_bool = []

    if func_name in 'repeat':
        nreq_param_int = ['count', 'prefetch_size']

    if func_name in 'take':
        req_param_int = ['count']
        nreq_param_int = ['prefetch_size']

    elif func_name in 'shuffle':
        req_param_int = ['buffer_size']
        nreq_param_bool = ['reshuffle_each_iteration']
        nreq_param_int = ['prefetch_size', 'seed']

    elif func_name in 'batch':
        req_param_int = ['batch_size']
        nreq_param_int = ['num_parallel_workers', 'prefetch_size']
        nreq_param_bool = ['drop_remainder']

    elif func_name in ('zip', 'filter', 'cache', 'rename', 'project'):
        nreq_param_int = ['prefetch_size']

    elif func_name in ('map', '__init__'):
        nreq_param_int = ['num_parallel_workers', 'prefetch_size', 'seed']
        nreq_param_bool = ['block_reader']

    @wraps(method)
    def wrapper(*args, **kwargs):

        def _make_key():
            sig = ins.signature(method)
            params = sig.parameters
            keys = list(params.keys())
            param_dic = dict()
            for name, value in enumerate(args):
                param_dic[keys[name]] = value
            param_dic.update(zip(params.keys(), args))
            param_dic.update(kwargs)

            for name, value in params.items():
                if name not in param_dic:
                    param_dic[name] = value.default
            return param_dic

        # check type
        def _check_param_type(arg, param_name, param_type=None):
            if param_type is not None and not isinstance(arg, param_type):
                raise ValueError(
                    "The %s function %s type error!" % (func_name, param_name))

        # check range
        def _check_param_range(arg, param_name):
            if isinstance(arg, int) and param_name == "seed" and (
                    arg < 0 or arg > 2147483647):
                raise ValueError(
                    "The %s function %s exceeds the boundary!" % (
                        func_name, param_name))
            if isinstance(arg, int) and param_name == "count" and ((arg <= 0 and arg != -1) or arg > 2147483647):
                raise ValueError(
                    "The %s function %s exceeds the boundary!" % (
                        func_name, param_name))
            if isinstance(arg, int) and param_name == "prefetch_size" and (
                    arg <= 0 or arg > 1024):
                raise ValueError(
                    "The %s function %s exceeds the boundary!" % (
                        func_name, param_name))
            if isinstance(arg, int) and param_name == "num_parallel_workers" and (
                    arg < 1 or arg > cpu_count()):
                raise ValueError(
                    "The %s function %s exceeds the boundary(%s)!" % (
                        func_name, param_name, cpu_count()))
            if isinstance(arg, int) and param_name != "seed" \
                    and param_name != "count" and param_name != "prefetch_size" \
                    and param_name != "num_parallel_workers" and (arg < 1 or arg > 2147483647):
                raise ValueError(
                    "The %s function %s exceeds the boundary!" % (
                        func_name, param_name))

        key = _make_key()
        # check integer
        for karg in req_param_int:
            _check_param_type(key[karg], karg, int)
            _check_param_range(key[karg], karg)
        for karg in nreq_param_int:
            if karg in key:
                if key[karg] is not None:
                    _check_param_type(key[karg], karg, int)
                    _check_param_range(key[karg], karg)
        # check bool
        for karg in req_param_bool:
            _check_param_type(key[karg], karg, bool)
        for karg in nreq_param_bool:
            if karg in key:
                if key[karg] is not None:
                    _check_param_type(key[karg], karg, bool)

        if func_name in '__init__':
            if 'columns_list' in key.keys():
                columns_list = key['columns_list']
                if columns_list is not None:
                    _check_param_type(columns_list, 'columns_list', list)

            if 'columns' in key.keys():
                columns = key['columns']
                if columns is not None:
                    _check_param_type(columns, 'columns', list)

            if 'partitions' in key.keys():
                partitions = key['partitions']
                if partitions is not None:
                    _check_param_type(partitions, 'partitions', list)

            if 'schema' in key.keys():
                schema = key['schema']
                if schema is not None:
                    check_filename(schema)
                    if not os.path.isfile(schema) or not os.access(schema, os.R_OK):
                        raise ValueError(
                            "The file %s does not exist or permission denied!" % schema)

            if 'dataset_dir' in key.keys():
                dataset_dir = key['dataset_dir']
                if dataset_dir is not None:
                    if not os.path.isdir(dataset_dir) or not os.access(dataset_dir, os.R_OK):
                        raise ValueError(
                            "The folder %s does not exist or permission denied!" % dataset_dir)

            if 'dataset_files' in key.keys():
                dataset_files = key['dataset_files']
                if not dataset_files:
                    raise ValueError(
                        "The dataset file does not exists!")
                if dataset_files is not None:
                    _check_param_type(dataset_files, 'dataset_files', list)
                    for file in dataset_files:
                        if not os.path.isfile(file) or not os.access(file, os.R_OK):
                            raise ValueError(
                                "The file %s does not exist or permission denied!" % file)

            if 'dataset_file' in key.keys():
                dataset_file = key['dataset_file']
                if not dataset_file:
                    raise ValueError(
                        "The dataset file does not exists!")
                check_filename(dataset_file)
                if dataset_file is not None:
                    if not os.path.isfile(dataset_file) or not os.access(dataset_file, os.R_OK):
                        raise ValueError(
                            "The file %s does not exist or permission denied!" % dataset_file)

        return method(*args, **kwargs)

    return wrapper


def check_valid_detype(type_):
    if type_ not in valid_detype:
        raise ValueError("Unknown column type")
    return True


def check_filename(path):
    """
    check the filename in the path

    Args:
        path (str): the path

    Returns:
        Exception: when error
    """
    if not isinstance(path, str):
        raise ValueError("path: {} is not string".format(path))
    filename = os.path.basename(path)

    # '#', ':', '|', ' ', '}', '"', '+', '!', ']', '[', '\\', '`',
    # '&', '.', '/', '@', "'", '^', ',', '_', '<', ';', '~', '>',
    # '*', '(', '%', ')', '-', '=', '{', '?', '$'
    forbidden_symbols = set(r'\/:*?"<>|`&\';')

    if set(filename) & forbidden_symbols:
        raise ValueError(r"filename should not contains \/:*?\"<>|`&;\'")

    if filename.startswith(' ') or filename.endswith(' '):
        raise ValueError("filename should not start/end with space")

    return True


def make_param_dict(method, args, kwargs):
    """Return a dictionary of the method's args and kwargs."""
    sig = ins.signature(method)
    params = sig.parameters
    keys = list(params.keys())
    param_dict = dict()
    try:
        for name, value in enumerate(args):
            param_dict[keys[name]] = value
    except IndexError:
        raise TypeError("{0}() expected {1} arguments, but {2} were given".format(
            method.__name__, len(keys) - 1, len(args) - 1))

    param_dict.update(zip(params.keys(), args))
    param_dict.update(kwargs)

    for name, value in params.items():
        if name not in param_dict:
            param_dict[name] = value.default
    return param_dict


def check_type(param, param_name, valid_type):
    if (not isinstance(param, valid_type)) or (valid_type == int and isinstance(param, bool)):
        raise TypeError("Wrong input type for {0}, should be {1}, got {2}".format(param_name, valid_type, type(param)))


def check_param_type(param_list, param_dict, param_type):
    for param_name in param_list:
        if param_dict.get(param_name) is not None:
            if param_name == 'num_parallel_workers':
                check_num_parallel_workers(param_dict.get(param_name))
            if param_name == 'num_samples':
                check_num_samples(param_dict.get(param_name))
            else:
                check_type(param_dict.get(param_name), param_name, param_type)


def check_positive_int32(param, param_name):
    check_interval_closed(param, param_name, [1, INT32_MAX])


def check_interval_closed(param, param_name, valid_range):
    if param < valid_range[0] or param > valid_range[1]:
        raise ValueError("The value of {0} exceeds the closed interval range {1}.".format(param_name, valid_range))


def check_num_parallel_workers(value):
    check_type(value, 'num_parallel_workers', int)
    if value < 1 or value > cpu_count():
        raise ValueError("num_parallel_workers exceeds the boundary between 1 and {}!".format(cpu_count()))


def check_num_samples(value):
    check_type(value, 'num_samples', int)
    if value <= 0:
        raise ValueError("num_samples must be greater than 0!")


def check_dataset_dir(dataset_dir):
    if not os.path.isdir(dataset_dir) or not os.access(dataset_dir, os.R_OK):
        raise ValueError("The folder {} does not exist or permission denied!".format(dataset_dir))


def check_dataset_file(dataset_file):
    check_filename(dataset_file)
    if not os.path.isfile(dataset_file) or not os.access(dataset_file, os.R_OK):
        raise ValueError("The file {} does not exist or permission denied!".format(dataset_file))


def check_sampler_shuffle_shard_options(param_dict):
    """check for valid shuffle, sampler, num_shards, and shard_id inputs."""
    shuffle, sampler = param_dict.get('shuffle'), param_dict.get('sampler')
    num_shards, shard_id = param_dict.get('num_shards'), param_dict.get('shard_id')

    if sampler is not None and not isinstance(sampler, (samplers.BuiltinSampler, samplers.Sampler)):
        raise ValueError("sampler is not a valid Sampler type.")

    if sampler is not None:
        if shuffle is not None:
            raise RuntimeError("sampler and shuffle cannot be specified at the same time.")

        if num_shards is not None:
            raise RuntimeError("sampler and sharding cannot be specified at the same time.")

    if num_shards is not None:
        if shard_id is None:
            raise RuntimeError("num_shards is specified and currently requires shard_id as well.")
        if shard_id < 0 or shard_id >= num_shards:
            raise ValueError("shard_id is invalid, shard_id={}".format(shard_id))

    if num_shards is None and shard_id is not None:
        raise RuntimeError("shard_id is specified but num_shards is not.")


def check_padding_options(param_dict):
    """ check for valid padded_sample and num_padded of padded samples"""
    columns_list = param_dict.get('columns_list')
    block_reader = param_dict.get('block_reader')
    padded_sample, num_padded = param_dict.get('padded_sample'), param_dict.get('num_padded')
    if padded_sample is not None:
        if num_padded is None:
            raise RuntimeError("padded_sample is specified and requires num_padded as well.")
        if num_padded < 0:
            raise ValueError("num_padded is invalid, num_padded={}.".format(num_padded))
        if columns_list is None:
            raise RuntimeError("padded_sample is specified and requires columns_list as well.")
        for column in columns_list:
            if column not in padded_sample:
                raise ValueError("padded_sample cannot match columns_list.")
        if block_reader:
            raise RuntimeError("block_reader and padded_sample cannot be specified at the same time.")

    if padded_sample is None and num_padded is not None:
        raise RuntimeError("num_padded is specified but padded_sample is not.")

def check_imagefolderdatasetv2(method):
    """A wrapper that wrap a parameter checker to the original Dataset(ImageFolderDatasetV2)."""

    @wraps(method)
    def new_method(*args, **kwargs):
        param_dict = make_param_dict(method, args, kwargs)

        nreq_param_int = ['num_samples', 'num_parallel_workers', 'num_shards', 'shard_id']
        nreq_param_bool = ['shuffle', 'decode']
        nreq_param_list = ['extensions']
        nreq_param_dict = ['class_indexing']

        # check dataset_dir; required argument
        dataset_dir = param_dict.get('dataset_dir')
        if dataset_dir is None:
            raise ValueError("dataset_dir is not provided.")
        check_dataset_dir(dataset_dir)

        check_param_type(nreq_param_int, param_dict, int)

        check_param_type(nreq_param_bool, param_dict, bool)

        check_param_type(nreq_param_list, param_dict, list)

        check_param_type(nreq_param_dict, param_dict, dict)

        check_sampler_shuffle_shard_options(param_dict)

        return method(*args, **kwargs)

    return new_method


def check_mnist_cifar_dataset(method):
    """A wrapper that wrap a parameter checker to the original Dataset(ManifestDataset, Cifar10/100Dataset)."""

    @wraps(method)
    def new_method(*args, **kwargs):
        param_dict = make_param_dict(method, args, kwargs)

        nreq_param_int = ['num_samples', 'num_parallel_workers', 'num_shards', 'shard_id']
        nreq_param_bool = ['shuffle']

        # check dataset_dir; required argument
        dataset_dir = param_dict.get('dataset_dir')
        if dataset_dir is None:
            raise ValueError("dataset_dir is not provided.")
        check_dataset_dir(dataset_dir)

        check_param_type(nreq_param_int, param_dict, int)

        check_param_type(nreq_param_bool, param_dict, bool)

        check_sampler_shuffle_shard_options(param_dict)

        return method(*args, **kwargs)

    return new_method


def check_manifestdataset(method):
    """A wrapper that wrap a parameter checker to the original Dataset(ManifestDataset)."""

    @wraps(method)
    def new_method(*args, **kwargs):
        param_dict = make_param_dict(method, args, kwargs)

        nreq_param_int = ['num_samples', 'num_parallel_workers', 'num_shards', 'shard_id']
        nreq_param_bool = ['shuffle', 'decode']
        nreq_param_str = ['usage']
        nreq_param_dict = ['class_indexing']

        # check dataset_file; required argument
        dataset_file = param_dict.get('dataset_file')
        if dataset_file is None:
            raise ValueError("dataset_file is not provided.")
        check_dataset_file(dataset_file)

        check_param_type(nreq_param_int, param_dict, int)

        check_param_type(nreq_param_bool, param_dict, bool)

        check_param_type(nreq_param_str, param_dict, str)

        check_param_type(nreq_param_dict, param_dict, dict)

        check_sampler_shuffle_shard_options(param_dict)

        return method(*args, **kwargs)

    return new_method


def check_tfrecorddataset(method):
    """A wrapper that wrap a parameter checker to the original Dataset(TFRecordDataset)."""

    @wraps(method)
    def new_method(*args, **kwargs):
        param_dict = make_param_dict(method, args, kwargs)

        nreq_param_int = ['num_samples', 'num_parallel_workers', 'num_shards', 'shard_id']
        nreq_param_list = ['columns_list']
        nreq_param_bool = ['shard_equal_rows']

        # check dataset_files; required argument
        dataset_files = param_dict.get('dataset_files')
        if dataset_files is None:
            raise ValueError("dataset_files is not provided.")
        if not isinstance(dataset_files, (str, list)):
            raise TypeError("dataset_files should be of type str or a list of strings.")

        check_param_type(nreq_param_int, param_dict, int)

        check_param_type(nreq_param_list, param_dict, list)

        check_param_type(nreq_param_bool, param_dict, bool)

        check_sampler_shuffle_shard_options(param_dict)

        return method(*args, **kwargs)

    return new_method


def check_vocdataset(method):
    """A wrapper that wrap a parameter checker to the original Dataset(VOCDataset)."""

    @wraps(method)
    def new_method(*args, **kwargs):
        param_dict = make_param_dict(method, args, kwargs)

        nreq_param_int = ['num_samples', 'num_parallel_workers', 'num_shards', 'shard_id']
        nreq_param_bool = ['shuffle', 'decode']
        nreq_param_dict = ['class_indexing']

        # check dataset_dir; required argument
        dataset_dir = param_dict.get('dataset_dir')
        if dataset_dir is None:
            raise ValueError("dataset_dir is not provided.")
        check_dataset_dir(dataset_dir)
        # check task; required argument
        task = param_dict.get('task')
        if task is None:
            raise ValueError("task is not provided.")
        if not isinstance(task, str):
            raise ValueError("task is not str type.")
        # check mode; required argument
        mode = param_dict.get('mode')
        if mode is None:
            raise ValueError("mode is not provided.")
        if not isinstance(mode, str):
            raise ValueError("mode is not str type.")

        imagesets_file = ""
        if task == "Segmentation":
            imagesets_file = os.path.join(dataset_dir, "ImageSets", "Segmentation", mode + ".txt")
            if param_dict.get('class_indexing') is not None:
                raise ValueError("class_indexing is invalid in Segmentation task")
        elif task == "Detection":
            imagesets_file = os.path.join(dataset_dir, "ImageSets", "Main", mode + ".txt")
        else:
            raise ValueError("Invalid task : " + task)

        check_dataset_file(imagesets_file)

        check_param_type(nreq_param_int, param_dict, int)

        check_param_type(nreq_param_bool, param_dict, bool)

        check_param_type(nreq_param_dict, param_dict, dict)

        check_sampler_shuffle_shard_options(param_dict)

        return method(*args, **kwargs)

    return new_method


def check_celebadataset(method):
    """A wrapper that wrap a parameter checker to the original Dataset(CelebADataset)."""

    @wraps(method)
    def new_method(*args, **kwargs):
        param_dict = make_param_dict(method, args, kwargs)

        nreq_param_int = ['num_samples', 'num_parallel_workers', 'num_shards', 'shard_id']
        nreq_param_bool = ['shuffle', 'decode']
        nreq_param_list = ['extensions']
        nreq_param_str = ['dataset_type']

        # check dataset_dir; required argument
        dataset_dir = param_dict.get('dataset_dir')
        if dataset_dir is None:
            raise ValueError("dataset_dir is not provided.")
        check_dataset_dir(dataset_dir)

        check_param_type(nreq_param_int, param_dict, int)

        check_param_type(nreq_param_bool, param_dict, bool)

        check_param_type(nreq_param_list, param_dict, list)

        check_param_type(nreq_param_str, param_dict, str)

        dataset_type = param_dict.get('dataset_type')
        if dataset_type is not None and dataset_type not in ('all', 'train', 'valid', 'test'):
            raise ValueError("dataset_type should be one of 'all', 'train', 'valid' or 'test'.")

        check_sampler_shuffle_shard_options(param_dict)

        sampler = param_dict.get('sampler')
        if sampler is not None and isinstance(sampler, samplers.PKSampler):
            raise ValueError("CelebADataset does not support PKSampler.")

        return method(*args, **kwargs)

    return new_method


def check_minddataset(method):
    """A wrapper that wrap a parameter checker to the original Dataset(MindDataset)."""

    @wraps(method)
    def new_method(*args, **kwargs):
        param_dict = make_param_dict(method, args, kwargs)

        nreq_param_int = ['num_samples', 'num_parallel_workers', 'seed', 'num_shards', 'shard_id', 'num_padded']
        nreq_param_list = ['columns_list']
        nreq_param_bool = ['block_reader']
        nreq_param_dict = ['padded_sample']

        # check dataset_file; required argument
        dataset_file = param_dict.get('dataset_file')
        if dataset_file is None:
            raise ValueError("dataset_file is not provided.")
        if isinstance(dataset_file, list):
            for f in dataset_file:
                check_dataset_file(f)
        else:
            check_dataset_file(dataset_file)

        check_param_type(nreq_param_int, param_dict, int)

        check_param_type(nreq_param_list, param_dict, list)

        check_param_type(nreq_param_bool, param_dict, bool)

        check_param_type(nreq_param_dict, param_dict, dict)

        check_sampler_shuffle_shard_options(param_dict)

        check_padding_options(param_dict)
        return method(*args, **kwargs)

    return new_method


def check_generatordataset(method):
    """A wrapper that wrap a parameter checker to the original Dataset(GeneratorDataset)."""

    @wraps(method)
    def new_method(*args, **kwargs):
        param_dict = make_param_dict(method, args, kwargs)

        # check generator_function; required argument
        source = param_dict.get('source')
        if source is None:
            raise ValueError("source is not provided.")
        if not callable(source):
            try:
                iter(source)
            except TypeError:
                raise TypeError("source should be callable, iterable or random accessible")

        # check column_names or schema; required argument
        column_names = param_dict.get('column_names')
        schema = param_dict.get('schema')
        if column_names is None and schema is None:
            raise ValueError("Neither columns_names not schema are provided.")

        if schema is not None:
            if not isinstance(schema, datasets.Schema) and not isinstance(schema, str):
                raise ValueError("schema should be a path to schema file or a schema object.")

        # check optional argument
        nreq_param_int = ["num_samples", "num_parallel_workers", "num_shards", "shard_id"]
        check_param_type(nreq_param_int, param_dict, int)
        nreq_param_list = ["column_types"]
        check_param_type(nreq_param_list, param_dict, list)
        nreq_param_bool = ["shuffle"]
        check_param_type(nreq_param_bool, param_dict, bool)

        num_shards = param_dict.get("num_shards")
        shard_id = param_dict.get("shard_id")
        if (num_shards is None) != (shard_id is None):
            # These two parameters appear together.
            raise ValueError("num_shards and shard_id need to be passed in together")
        if num_shards is not None:
            if shard_id >= num_shards:
                raise ValueError("shard_id should be less than num_shards")

        sampler = param_dict.get("sampler")
        if sampler is not None:
            if isinstance(sampler, samplers.PKSampler):
                raise ValueError("PKSampler is not supported by GeneratorDataset")
            if not isinstance(sampler, (samplers.SequentialSampler, samplers.DistributedSampler,
                                        samplers.RandomSampler, samplers.SubsetRandomSampler,
                                        samplers.WeightedRandomSampler, samplers.Sampler)):
                try:
                    iter(sampler)
                except TypeError:
                    raise TypeError("sampler should be either iterable or from mindspore.dataset.samplers")

        if sampler is not None and not hasattr(source, "__getitem__"):
            raise ValueError("sampler is not supported if source does not have attribute '__getitem__'")
        if num_shards is not None and not hasattr(source, "__getitem__"):
            raise ValueError("num_shards is not supported if source does not have attribute '__getitem__'")

        return method(*args, **kwargs)

    return new_method


def check_batch_size(batch_size):
    if not (isinstance(batch_size, int) or (callable(batch_size))):
        raise ValueError("batch_size should either be an int or a callable.")
    if callable(batch_size):
        sig = ins.signature(batch_size)
        if len(sig.parameters) != 1:
            raise ValueError("batch_size callable should take one parameter (BatchInfo).")


def check_count(count):
    check_type(count, 'count', int)
    if (count <= 0 and count != -1) or count > INT32_MAX:
        raise ValueError("count should be either -1 or positive integer.")


def check_columns(columns, name):
    if isinstance(columns, list):
        for column in columns:
            if not isinstance(column, str):
                raise TypeError("Each column in {0} should be of type str. Got {1}.".format(name, type(column)))
    elif not isinstance(columns, str):
        raise TypeError("{} should be either a list of strings or a single string.".format(name))


def check_pad_info(key, val):
    """check the key and value pair of pad_info in batch"""
    check_type(key, "key in pad_info", str)
    if val is not None:
        assert len(val) == 2, "value of pad_info should be a tuple of size 2"
        check_type(val, "value in pad_info", tuple)
        if val[0] is not None:
            check_type(val[0], "pad_shape", list)
            for dim in val[0]:
                if dim is not None:
                    check_type(dim, "dim in pad_shape", int)
                    assert dim > 0, "pad shape should be positive integers"
        if val[1] is not None:
            check_type(val[1], "pad_value", (int, float))


def check_batch(method):
    """check the input arguments of batch."""

    @wraps(method)
    def new_method(*args, **kwargs):
        param_dict = make_param_dict(method, args, kwargs)

        nreq_param_int = ['num_parallel_workers']
        nreq_param_bool = ['drop_remainder']
        nreq_param_columns = ['input_columns']

        # check batch_size; required argument
        batch_size = param_dict.get("batch_size")
        if batch_size is None:
            raise ValueError("batch_size is not provided.")
        check_batch_size(batch_size)

        check_param_type(nreq_param_int, param_dict, int)

        check_param_type(nreq_param_bool, param_dict, bool)

        if (param_dict.get('pad_info') is not None) and (param_dict.get('per_batch_map') is not None):
            raise ValueError("pad_info and per_batch_map can't both be set")

        if param_dict.get('pad_info') is not None:
            check_type(param_dict["pad_info"], "pad_info", dict)
            for k, v in param_dict.get('pad_info').items():
                check_pad_info(k, v)

        for param_name in nreq_param_columns:
            param = param_dict.get(param_name)
            if param is not None:
                check_columns(param, param_name)

        per_batch_map, input_columns = param_dict.get('per_batch_map'), param_dict.get('input_columns')
        if (per_batch_map is None) != (input_columns is None):
            # These two parameters appear together.
            raise ValueError("per_batch_map and input_columns need to be passed in together.")

        if input_columns is not None:
            if not input_columns:  # Check whether input_columns is empty.
                raise ValueError("input_columns can not be empty")
            if len(input_columns) != (len(ins.signature(per_batch_map).parameters) - 1):
                raise ValueError("the signature of per_batch_map should match with input columns")

        return method(*args, **kwargs)

    return new_method

def check_sync_wait(method):
    """check the input arguments of sync_wait."""
    @wraps(method)
    def new_method(*args, **kwargs):
        param_dict = make_param_dict(method, args, kwargs)

        nreq_param_str = ['condition_name']
        nreq_param_int = ['step_size']

        check_param_type(nreq_param_int, param_dict, int)

        check_param_type(nreq_param_str, param_dict, str)

        return method(*args, **kwargs)

    return new_method

def check_shuffle(method):
    """check the input arguments of shuffle."""

    @wraps(method)
    def new_method(*args, **kwargs):
        param_dict = make_param_dict(method, args, kwargs)

        # check buffer_size; required argument
        buffer_size = param_dict.get("buffer_size")
        if buffer_size is None:
            raise ValueError("buffer_size is not provided.")
        check_type(buffer_size, 'buffer_size', int)
        check_interval_closed(buffer_size, 'buffer_size', [2, INT32_MAX])

        return method(*args, **kwargs)

    return new_method


def check_map(method):
    """check the input arguments of map."""

    @wraps(method)
    def new_method(*args, **kwargs):
        param_dict = make_param_dict(method, args, kwargs)

        nreq_param_list = ['columns_order']
        nreq_param_int = ['num_parallel_workers']
        nreq_param_columns = ['input_columns', 'output_columns']
        nreq_param_bool = ['python_multiprocessing']

        check_param_type(nreq_param_list, param_dict, list)
        check_param_type(nreq_param_int, param_dict, int)
        check_param_type(nreq_param_bool, param_dict, bool)
        for param_name in nreq_param_columns:
            param = param_dict.get(param_name)
            if param is not None:
                check_columns(param, param_name)

        return method(*args, **kwargs)

    return new_method


def check_filter(method):
    """"check the input arguments of filter."""

    @wraps(method)
    def new_method(*args, **kwargs):
        param_dict = make_param_dict(method, args, kwargs)
        predicate = param_dict.get("predicate")
        if not callable(predicate):
            raise ValueError("Predicate should be a python function or a callable python object.")

        nreq_param_int = ['num_parallel_workers']
        check_param_type(nreq_param_int, param_dict, int)
        param_name = "input_columns"
        param = param_dict.get(param_name)
        if param is not None:
            check_columns(param, param_name)
        return method(*args, **kwargs)

    return new_method


def check_repeat(method):
    """check the input arguments of repeat."""

    @wraps(method)
    def new_method(*args, **kwargs):
        param_dict = make_param_dict(method, args, kwargs)

        count = param_dict.get('count')
        if count is not None:
            check_count(count)

        return method(*args, **kwargs)

    return new_method


def check_skip(method):
    """check the input arguments of skip."""

    @wraps(method)
    def new_method(*args, **kwargs):
        param_dict = make_param_dict(method, args, kwargs)

        count = param_dict.get('count')
        check_type(count, 'count', int)
        if count < 0:
            raise ValueError("Skip count must be positive integer or 0.")

        return method(*args, **kwargs)

    return new_method


def check_take(method):
    """check the input arguments of take."""

    @wraps(method)
    def new_method(*args, **kwargs):
        param_dict = make_param_dict(method, args, kwargs)

        count = param_dict.get('count')
        check_count(count)

        return method(*args, **kwargs)

    return new_method


def check_zip(method):
    """check the input arguments of zip."""

    @wraps(method)
    def new_method(*args, **kwargs):
        param_dict = make_param_dict(method, args, kwargs)

        # check datasets; required argument
        ds = param_dict.get("datasets")
        if ds is None:
            raise ValueError("datasets is not provided.")
        check_type(ds, 'datasets', tuple)

        return method(*args, **kwargs)

    return new_method


def check_zip_dataset(method):
    """check the input arguments of zip method in `Dataset`."""

    @wraps(method)
    def new_method(*args, **kwargs):
        param_dict = make_param_dict(method, args, kwargs)

        # check datasets; required argument
        ds = param_dict.get("datasets")
        if ds is None:
            raise ValueError("datasets is not provided.")

        if not isinstance(ds, (tuple, datasets.Dataset)):
            raise ValueError("datasets is not tuple or of type Dataset.")

        return method(*args, **kwargs)

    return new_method


def check_concat(method):
    """check the input arguments of concat method in `Dataset`."""

    @wraps(method)
    def new_method(*args, **kwargs):
        param_dict = make_param_dict(method, args, kwargs)

        # check datasets; required argument
        ds = param_dict.get("datasets")
        if ds is None:
            raise ValueError("datasets is not provided.")

        if not isinstance(ds, (list, datasets.Dataset)):
            raise ValueError("datasets is not list or of type Dataset.")

        return method(*args, **kwargs)

    return new_method


def check_rename(method):
    """check the input arguments of rename."""

    @wraps(method)
    def new_method(*args, **kwargs):
        param_dict = make_param_dict(method, args, kwargs)

        req_param_columns = ['input_columns', 'output_columns']
        # check req_param_list; required arguments
        for param_name in req_param_columns:
            param = param_dict.get(param_name)
            if param is None:
                raise ValueError("{} is not provided.".format(param_name))
            check_columns(param, param_name)

        input_size, output_size = 1, 1
        if isinstance(param_dict.get(req_param_columns[0]), list):
            input_size = len(param_dict.get(req_param_columns[0]))
        if isinstance(param_dict.get(req_param_columns[1]), list):
            output_size = len(param_dict.get(req_param_columns[1]))
        if input_size != output_size:
            raise ValueError("Number of column in input_columns and output_columns is not equal.")

        return method(*args, **kwargs)

    return new_method


def check_project(method):
    """check the input arguments of project."""

    @wraps(method)
    def new_method(*args, **kwargs):
        param_dict = make_param_dict(method, args, kwargs)

        # check columns; required argument
        columns = param_dict.get("columns")
        if columns is None:
            raise ValueError("columns is not provided.")
        check_columns(columns, 'columns')

        return method(*args, **kwargs)

    return new_method


def check_shape(shape, name):
    if isinstance(shape, list):
        for element in shape:
            if not isinstance(element, int):
                raise TypeError(
                    "Each element in {0} should be of type int. Got {1}.".format(name, type(element)))
    else:
        raise TypeError("Expected int list.")


def check_add_column(method):
    """check the input arguments of add_column."""

    @wraps(method)
    def new_method(*args, **kwargs):
        param_dict = make_param_dict(method, args, kwargs)

        # check name; required argument
        name = param_dict.get("name")
        if not isinstance(name, str) or not name:
            raise TypeError("Expected non-empty string.")

        # check type; required argument
        de_type = param_dict.get("de_type")
        if de_type is not None:
            if not isinstance(de_type, typing.Type) and not check_valid_detype(de_type):
                raise ValueError("Unknown column type.")
        else:
            raise TypeError("Expected non-empty string.")

        # check shape
        shape = param_dict.get("shape")
        if shape is not None:
            check_shape(shape, "shape")

        return method(*args, **kwargs)

    return new_method


def check_textfiledataset(method):
    """A wrapper that wrap a parameter checker to the original Dataset(TextFileDataset)."""

    @wraps(method)
    def new_method(*args, **kwargs):
        param_dict = make_param_dict(method, args, kwargs)

        nreq_param_int = ['num_samples', 'num_parallel_workers', 'num_shards', 'shard_id']

        # check dataset_files; required argument
        dataset_files = param_dict.get('dataset_files')
        if dataset_files is None:
            raise ValueError("dataset_files is not provided.")
        if not isinstance(dataset_files, (str, list)):
            raise TypeError("dataset_files should be of type str or a list of strings.")

        check_param_type(nreq_param_int, param_dict, int)

        check_sampler_shuffle_shard_options(param_dict)

        return method(*args, **kwargs)

    return new_method


def check_split(method):
    """check the input arguments of split."""

    @wraps(method)
    def new_method(*args, **kwargs):
        param_dict = make_param_dict(method, args, kwargs)

        nreq_param_list = ['sizes']
        nreq_param_bool = ['randomize']
        check_param_type(nreq_param_list, param_dict, list)
        check_param_type(nreq_param_bool, param_dict, bool)

        # check sizes: must be list of float or list of int
        sizes = param_dict.get('sizes')

        if not sizes:
            raise ValueError("sizes cannot be empty.")
        all_int = all(isinstance(item, int) for item in sizes)
        all_float = all(isinstance(item, float) for item in sizes)

        if not (all_int or all_float):
            raise ValueError("sizes should be list of int or list of float.")

        if all_int:
            all_positive = all(item > 0 for item in sizes)
            if not all_positive:
                raise ValueError("sizes is a list of int, but there should be no negative numbers.")

        if all_float:
            all_valid_percentages = all(0 < item <= 1 for item in sizes)
            if not all_valid_percentages:
                raise ValueError("sizes is a list of float, but there should be no numbers outside the range [0, 1].")

            epsilon = 0.00001
            if not abs(sum(sizes) - 1) < epsilon:
                raise ValueError("sizes is a list of float, but the percentages do not sum up to 1.")

        return method(*args, **kwargs)

    return new_method


def check_gnn_graphdata(method):
    """check the input arguments of graphdata."""

    @wraps(method)
    def new_method(*args, **kwargs):
        param_dict = make_param_dict(method, args, kwargs)

        # check dataset_file; required argument
        dataset_file = param_dict.get('dataset_file')
        if dataset_file is None:
            raise ValueError("dataset_file is not provided.")
        check_dataset_file(dataset_file)

        nreq_param_int = ['num_parallel_workers']

        check_param_type(nreq_param_int, param_dict, int)

        return method(*args, **kwargs)

    return new_method


def check_gnn_list_or_ndarray(param, param_name):
    """Check if the input parameter is list or numpy.ndarray."""

    if isinstance(param, list):
        for m in param:
            if not isinstance(m, int):
                raise TypeError(
                    "Each membor in {0} should be of type int. Got {1}.".format(param_name, type(m)))
    elif isinstance(param, np.ndarray):
        if not param.dtype == np.int32:
            raise TypeError("Each membor in {0} should be of type int32. Got {1}.".format(
                param_name, param.dtype))
    else:
        raise TypeError("Wrong input type for {0}, should be list or numpy.ndarray, got {1}".format(
            param_name, type(param)))


def check_gnn_get_all_nodes(method):
    """A wrapper that wrap a parameter checker to the GNN `get_all_nodes` function."""

    @wraps(method)
    def new_method(*args, **kwargs):
        param_dict = make_param_dict(method, args, kwargs)

        # check node_type; required argument
        check_type(param_dict.get("node_type"), 'node_type', int)

        return method(*args, **kwargs)

    return new_method


def check_gnn_get_all_neighbors(method):
    """A wrapper that wrap a parameter checker to the GNN `get_all_neighbors` function."""

    @wraps(method)
    def new_method(*args, **kwargs):
        param_dict = make_param_dict(method, args, kwargs)

        # check node_list; required argument
        check_gnn_list_or_ndarray(param_dict.get("node_list"), 'node_list')

        # check neighbor_type; required argument
        check_type(param_dict.get("neighbor_type"), 'neighbor_type', int)

        return method(*args, **kwargs)

    return new_method


def check_aligned_list(param, param_name, membor_type):
    """Check whether the structure of each member of the list is the same."""

    if not isinstance(param, list):
        raise TypeError("Parameter {0} is not a list".format(param_name))
    if not param:
        raise TypeError(
            "Parameter {0} or its members are empty".format(param_name))
    membor_have_list = None
    list_len = None
    for membor in param:
        if isinstance(membor, list):
            check_aligned_list(membor, param_name, membor_type)
            if membor_have_list not in (None, True):
                raise TypeError("The type of each member of the parameter {0} is inconsistent".format(
                    param_name))
            if list_len is not None and len(membor) != list_len:
                raise TypeError("The size of each member of parameter {0} is inconsistent".format(
                    param_name))
            membor_have_list = True
            list_len = len(membor)
        else:
            if not isinstance(membor, membor_type):
                raise TypeError("Each membor in {0} should be of type int. Got {1}.".format(
                    param_name, type(membor)))
            if membor_have_list not in (None, False):
                raise TypeError("The type of each member of the parameter {0} is inconsistent".format(
                    param_name))
            membor_have_list = False


def check_gnn_get_node_feature(method):
    """A wrapper that wrap a parameter checker to the GNN `get_node_feature` function."""

    @wraps(method)
    def new_method(*args, **kwargs):
        param_dict = make_param_dict(method, args, kwargs)

        # check node_list; required argument
        node_list = param_dict.get("node_list")
        if isinstance(node_list, list):
            check_aligned_list(node_list, 'node_list', int)
        elif isinstance(node_list, np.ndarray):
            if not node_list.dtype == np.int32:
                raise TypeError("Each membor in {0} should be of type int32. Got {1}.".format(
                    node_list, node_list.dtype))
        else:
            raise TypeError("Wrong input type for {0}, should be list or numpy.ndarray, got {1}".format(
                'node_list', type(node_list)))

        # check feature_types; required argument
        check_gnn_list_or_ndarray(param_dict.get(
            "feature_types"), 'feature_types')

        return method(*args, **kwargs)

    return new_method
