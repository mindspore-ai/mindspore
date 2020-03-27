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
from . import samplers
from . import datasets

INT32_MAX = 2147483647


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
                    arg <= 0 or arg > cpu_count()):
                raise ValueError(
                    "The %s function %s exceeds the boundary(%s)!" % (
                        func_name, param_name, cpu_count()))
            if isinstance(arg, int) and param_name != "seed" \
                    and param_name != "count" and param_name != "prefetch_size" \
                    and param_name != "num_parallel_workers" and (arg <= 0 or arg > 2147483647):
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
    for name, value in enumerate(args):
        param_dict[keys[name]] = value
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
            else:
                check_type(param_dict.get(param_name), param_name, param_type)


def check_positive_int32(param, param_name):
    check_interval_closed(param, param_name, [1, INT32_MAX])


def check_interval_closed(param, param_name, valid_range):
    if param < valid_range[0] or param > valid_range[1]:
        raise ValueError("The value of {0} exceeds the closed interval range {1}.".format(param_name, valid_range))


def check_num_parallel_workers(value):
    check_type(value, 'num_parallel_workers', int)
    if value <= 0 or value > cpu_count():
        raise ValueError("num_parallel_workers exceeds the boundary between 0 and {}!".format(cpu_count()))


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

    if sampler is not None and not isinstance(sampler, (
            samplers.DistributedSampler, samplers.PKSampler, samplers.RandomSampler, samplers.SequentialSampler,
            samplers.SubsetRandomSampler, samplers.WeightedRandomSampler)):
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

        # check dataset_files; required argument
        dataset_files = param_dict.get('dataset_files')
        if dataset_files is None:
            raise ValueError("dataset_files is not provided.")
        if not isinstance(dataset_files, (str, list)):
            raise TypeError("dataset_files should be of type str or a list of strings.")

        check_param_type(nreq_param_int, param_dict, int)

        check_param_type(nreq_param_list, param_dict, list)

        return method(*args, **kwargs)

    return new_method


def check_vocdataset(method):
    """A wrapper that wrap a parameter checker to the original Dataset(VOCDataset)."""
    @wraps(method)
    def new_method(*args, **kwargs):
        param_dict = make_param_dict(method, args, kwargs)

        nreq_param_int = ['num_samples', 'num_parallel_workers']
        nreq_param_bool = ['shuffle', 'decode']
        nreq_param_str = ['distribution']

        # check dataset_dir; required argument
        dataset_dir = param_dict.get('dataset_dir')
        if dataset_dir is None:
            raise ValueError("dataset_dir is not provided.")
        check_dataset_dir(dataset_dir)

        check_param_type(nreq_param_int, param_dict, int)

        check_param_type(nreq_param_bool, param_dict, bool)

        check_param_type(nreq_param_str, param_dict, str)

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

        nreq_param_int = ['num_samples', 'num_parallel_workers', 'seed', 'num_shards', 'shard_id']
        nreq_param_list = ['columns_list']
        nreq_param_bool = ['block_reader']

        # check dataset_file; required argument
        dataset_file = param_dict.get('dataset_file')
        if dataset_file is None:
            raise ValueError("dataset_file is not provided.")
        check_dataset_file(dataset_file)

        check_param_type(nreq_param_int, param_dict, int)

        check_param_type(nreq_param_list, param_dict, list)

        check_param_type(nreq_param_bool, param_dict, bool)

        num_shards, shard_id = param_dict.get('num_shards'), param_dict.get('shard_id')
        if (num_shards is not None and shard_id is None) or (num_shards is None and shard_id is not None):
            raise ValueError("num_shards and shard_id need to be set or not set at the same time")

        return method(*args, **kwargs)

    return new_method


def check_generatordataset(method):
    """A wrapper that wrap a parameter checker to the original Dataset(GeneratorDataset)."""
    @wraps(method)
    def new_method(*args, **kwargs):
        param_dict = make_param_dict(method, args, kwargs)

        nreq_param_int = ['prefetch_size']
        nreq_param_list = ['column_names', 'column_types']

        # check generator_function; required argument
        generator_function = param_dict.get('generator_function')
        if generator_function is None:
            raise ValueError("generator_function is not provided.")

        # check column_names; required argument
        column_names = param_dict.get('column_names')
        if column_names is None:
            raise ValueError("column_names is not provided.")

        check_param_type(nreq_param_int, param_dict, int)

        check_param_type(nreq_param_list, param_dict, list)

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
        raise ValueError("repeat count should be either -1 or positive integer.")


def check_columns(columns, name):
    if isinstance(columns, list):
        for column in columns:
            if not isinstance(column, str):
                raise TypeError("Each column in {0} should be of type str. Got {1}.".format(name, type(column)))
    elif not isinstance(columns, str):
        raise TypeError("{} should be either a list of strings or a single string.".format(name))


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

        check_param_type(nreq_param_list, param_dict, list)
        check_param_type(nreq_param_int, param_dict, int)
        for param_name in nreq_param_columns:
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
