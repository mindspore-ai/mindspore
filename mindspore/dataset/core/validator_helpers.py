# Copyright 2020-2021 Huawei Technologies Co., Ltd
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
General Validators.
"""
import inspect
from multiprocessing import cpu_count
import os
import numpy as np

import mindspore._c_dataengine as cde

# POS_INT_MIN is used to limit values from starting from 0
POS_INT_MIN = 1
UINT8_MAX = 255
UINT8_MIN = 0
UINT32_MAX = 4294967295
UINT32_MIN = 0
UINT64_MAX = 18446744073709551615
UINT64_MIN = 0
INT32_MAX = 2147483647
INT32_MIN = -2147483648
INT64_MAX = 9223372036854775807
INT64_MIN = -9223372036854775808
FLOAT_MAX_INTEGER = 16777216
FLOAT_MIN_INTEGER = -16777216
DOUBLE_MAX_INTEGER = 9007199254740992
DOUBLE_MIN_INTEGER = -9007199254740992

valid_detype = [
    "bool", "int8", "int16", "int32", "int64", "uint8", "uint16",
    "uint32", "uint64", "float16", "float32", "float64", "string"
]


def is_iterable(obj):
    """
    Helper function to check if object is iterable.

    Args:
        obj (any): object to check if iterable

    Returns:
        bool, true if object iteratable
    """
    try:
        iter(obj)
    except TypeError:
        return False
    return True


def pad_arg_name(arg_name):
    if arg_name != "":
        arg_name = arg_name + " "
    return arg_name


def check_value(value, valid_range, arg_name=""):
    arg_name = pad_arg_name(arg_name)
    if value < valid_range[0] or value > valid_range[1]:
        raise ValueError(
            "Input {0}is not within the required interval of [{1}, {2}].".format(arg_name, valid_range[0],
                                                                                 valid_range[1]))


def check_value_cutoff(value, valid_range, arg_name=""):
    arg_name = pad_arg_name(arg_name)
    if value < valid_range[0] or value >= valid_range[1]:
        raise ValueError(
            "Input {0}is not within the required interval of [{1}, {2}).".format(arg_name, valid_range[0],
                                                                                 valid_range[1]))


def check_value_ratio(value, valid_range, arg_name=""):
    arg_name = pad_arg_name(arg_name)
    if value <= valid_range[0] or value > valid_range[1]:
        raise ValueError(
            "Input {0}is not within the required interval of ({1}, {2}].".format(arg_name, valid_range[0],
                                                                                 valid_range[1]))


def check_value_normalize_std(value, valid_range, arg_name=""):
    arg_name = pad_arg_name(arg_name)
    if value <= valid_range[0] or value > valid_range[1]:
        raise ValueError(
            "Input {0}is not within the required interval of ({1}, {2}].".format(arg_name, valid_range[0],
                                                                                 valid_range[1]))


def check_range(values, valid_range, arg_name=""):
    arg_name = pad_arg_name(arg_name)
    if not valid_range[0] <= values[0] <= values[1] <= valid_range[1]:
        raise ValueError(
            "Input {0}is not within the required interval of [{1}, {2}].".format(arg_name, valid_range[0],
                                                                                 valid_range[1]))


def check_positive(value, arg_name=""):
    arg_name = pad_arg_name(arg_name)
    if value <= 0:
        raise ValueError("Input {0}must be greater than 0.".format(arg_name))


def check_2tuple(value, arg_name=""):
    if not (isinstance(value, tuple) and len(value) == 2):
        raise ValueError("Value {0}needs to be a 2-tuple.".format(arg_name))


def check_uint8(value, arg_name=""):
    type_check(value, (int,), arg_name)
    check_value(value, [UINT8_MIN, UINT8_MAX])


def check_uint32(value, arg_name=""):
    type_check(value, (int,), arg_name)
    check_value(value, [UINT32_MIN, UINT32_MAX])


def check_pos_int32(value, arg_name=""):
    type_check(value, (int,), arg_name)
    check_value(value, [POS_INT_MIN, INT32_MAX], arg_name)


def check_uint64(value, arg_name=""):
    type_check(value, (int,), arg_name)
    check_value(value, [UINT64_MIN, UINT64_MAX])


def check_pos_int64(value, arg_name=""):
    type_check(value, (int,), arg_name)
    check_value(value, [UINT64_MIN, INT64_MAX])


def check_float32(value, arg_name=""):
    check_value(value, [FLOAT_MIN_INTEGER, FLOAT_MAX_INTEGER], arg_name)


def check_float64(value, arg_name=""):
    check_value(value, [DOUBLE_MIN_INTEGER, DOUBLE_MAX_INTEGER], arg_name)


def check_pos_float32(value, arg_name=""):
    check_value(value, [UINT32_MIN, FLOAT_MAX_INTEGER], arg_name)


def check_pos_float64(value, arg_name=""):
    check_value(value, [UINT64_MIN, DOUBLE_MAX_INTEGER], arg_name)


def check_valid_detype(type_):
    if type_ not in valid_detype:
        raise TypeError("Unknown column type.")
    return True


def check_valid_str(value, valid_strings, arg_name=""):
    type_check(value, (str,), arg_name)
    if value not in valid_strings:
        raise ValueError("Input {0} is not within the valid set of {1}.".format(arg_name, str(valid_strings)))


def check_columns(columns, name):
    """
    Validate strings in column_names.

    Args:
        columns (list): list of column_names.
        name (str): name of columns.

    Returns:
        Exception: when the value is not correct, otherwise nothing.
    """
    type_check(columns, (list, str), name)
    if isinstance(columns, str):
        if not columns:
            raise ValueError("{0} should not be an empty str.".format(name))
    elif isinstance(columns, list):
        if not columns:
            raise ValueError("{0} should not be empty.".format(name))
        for i, column_name in enumerate(columns):
            if not column_name:
                raise ValueError("{0}[{1}] should not be empty.".format(name, i))

        col_names = ["{0}[{1}]".format(name, i) for i in range(len(columns))]
        type_check_list(columns, (str,), col_names)
        if len(set(columns)) != len(columns):
            raise ValueError("Every column name should not be same with others in column_names.")


def parse_user_args(method, *args, **kwargs):
    """
    Parse user arguments in a function.

    Args:
        method (method): a callable function.
        args: user passed args.
        kwargs: user passed kwargs.

    Returns:
        user_filled_args (list): values of what the user passed in for the arguments.
        ba.arguments (Ordered Dict): ordered dict of parameter and argument for what the user has passed.
    """
    sig = inspect.signature(method)
    if 'self' in sig.parameters or 'cls' in sig.parameters:
        ba = sig.bind(method, *args, **kwargs)
        ba.apply_defaults()
        params = list(sig.parameters.keys())[1:]
    else:
        ba = sig.bind(*args, **kwargs)
        ba.apply_defaults()
        params = list(sig.parameters.keys())

    user_filled_args = [ba.arguments.get(arg_value) for arg_value in params]
    return user_filled_args, ba.arguments


def type_check_list(args, types, arg_names):
    """
    Check the type of each parameter in the list.

    Args:
        args (Union[list, tuple]): a list or tuple of any variable.
        types (tuple): tuple of all valid types for arg.
        arg_names (Union[list, tuple of str]): the names of args.

    Returns:
        Exception: when the type is not correct, otherwise nothing.
    """
    type_check(args, (list, tuple,), arg_names)
    if len(args) != len(arg_names) and not isinstance(arg_names, str):
        raise ValueError("List of arguments is not the same length as argument_names.")
    if isinstance(arg_names, str):
        arg_names = ["{0}[{1}]".format(arg_names, i) for i in range(len(args))]
    for arg, arg_name in zip(args, arg_names):
        type_check(arg, types, arg_name)


def type_check(arg, types, arg_name):
    """
    Check the type of the parameter.

    Args:
        arg (Any) : any variable.
        types (tuple): tuple of all valid types for arg.
        arg_name (str): the name of arg.

    Returns:
        Exception: when the type is not correct, otherwise nothing.
    """
    # handle special case of booleans being a subclass of ints
    print_value = '\"\"' if repr(arg) == repr('') else arg

    if int in types and bool not in types:
        if isinstance(arg, bool):
            raise TypeError("Argument {0} with value {1} is not of type {2}.".format(arg_name, print_value, types))
    if not isinstance(arg, types):
        raise TypeError("Argument {0} with value {1} is not of type {2}.".format(arg_name, print_value, types))


def check_filename(path):
    """
    check the filename in the path.

    Args:
        path (str): the path.

    Returns:
        Exception: when error.
    """
    if not isinstance(path, str):
        raise TypeError("path: {} is not string".format(path))
    filename = os.path.basename(path)

    # '#', ':', '|', ' ', '}', '"', '+', '!', ']', '[', '\\', '`',
    # '&', '.', '/', '@', "'", '^', ',', '_', '<', ';', '~', '>',
    # '*', '(', '%', ')', '-', '=', '{', '?', '$'
    forbidden_symbols = set(r'\/:*?"<>|`&\';')

    if set(filename) & forbidden_symbols:
        raise ValueError(r"filename should not contain \/:*?\"<>|`&;\'")

    if filename.startswith(' ') or filename.endswith(' '):
        raise ValueError("filename should not start/end with space.")

    return True


def check_dir(dataset_dir):
    if not os.path.isdir(dataset_dir) or not os.access(dataset_dir, os.R_OK):
        raise ValueError("The folder {} does not exist or permission denied!".format(dataset_dir))


def check_file(dataset_file):
    check_filename(dataset_file)
    if not os.path.isfile(dataset_file) or not os.access(dataset_file, os.R_OK):
        raise ValueError("The file {} does not exist or permission denied!".format(dataset_file))


def check_sampler_shuffle_shard_options(param_dict):
    """
    Check for valid shuffle, sampler, num_shards, and shard_id inputs.
    Args:
        param_dict (dict): param_dict.

    Returns:
        Exception: ValueError or RuntimeError if error.
    """
    shuffle, sampler = param_dict.get('shuffle'), param_dict.get('sampler')
    num_shards, shard_id = param_dict.get('num_shards'), param_dict.get('shard_id')
    num_samples = param_dict.get('num_samples')

    if sampler is not None:
        if shuffle is not None:
            raise RuntimeError("sampler and shuffle cannot be specified at the same time.")
        if num_shards is not None or shard_id is not None:
            raise RuntimeError("sampler and sharding cannot be specified at the same time.")
        if num_samples is not None:
            raise RuntimeError("sampler and num_samples cannot be specified at the same time.")

    if num_shards is not None:
        check_pos_int32(num_shards)
        if shard_id is None:
            raise RuntimeError("num_shards is specified and currently requires shard_id as well.")
        check_value(shard_id, [0, num_shards - 1], "shard_id")

    if num_shards is None and shard_id is not None:
        raise RuntimeError("shard_id is specified but num_shards is not.")


def check_padding_options(param_dict):
    """
    Check for valid padded_sample and num_padded of padded samples.

    Args:
        param_dict (dict): param_dict.

    Returns:
        Exception: ValueError or RuntimeError if error.
    """

    columns_list = param_dict.get('columns_list')
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
    if padded_sample is None and num_padded is not None:
        raise RuntimeError("num_padded is specified but padded_sample is not.")


def check_num_parallel_workers(value):
    type_check(value, (int,), "num_parallel_workers")
    if value < 1 or value > cpu_count():
        raise ValueError("num_parallel_workers exceeds the boundary between 1 and {}!".format(cpu_count()))


def check_num_samples(value):
    type_check(value, (int,), "num_samples")
    if value < 0 or value > INT64_MAX:
        raise ValueError(
            "num_samples exceeds the boundary between {} and {}(INT64_MAX)!".format(0, INT64_MAX))


def validate_dataset_param_value(param_list, param_dict, param_type):
    for param_name in param_list:
        if param_dict.get(param_name) is not None:
            if param_name == 'num_parallel_workers':
                check_num_parallel_workers(param_dict.get(param_name))
            if param_name == 'num_samples':
                check_num_samples(param_dict.get(param_name))
            else:
                type_check(param_dict.get(param_name), (param_type,), param_name)


def check_gnn_list_or_ndarray(param, param_name):
    """
    Check if the input parameter is list or numpy.ndarray.

    Args:
        param (Union[list, nd.ndarray]): param.
        param_name (str): param_name.

    Returns:
        Exception: TypeError if error.
    """

    type_check(param, (list, np.ndarray), param_name)
    if isinstance(param, list):
        param_names = ["param_{0}".format(i) for i in range(len(param))]
        type_check_list(param, (int,), param_names)

    elif isinstance(param, np.ndarray):
        if not param.dtype == np.int32:
            raise TypeError("Each member in {0} should be of type int32. Got {1}.".format(
                param_name, param.dtype))


def check_tensor_op(param, param_name):
    """check whether param is a tensor op or a callable Python function"""
    if not isinstance(param, cde.TensorOp) and not callable(param) and not getattr(param, 'parse', None):
        raise TypeError("{0} is neither a c_transform op (TensorOperation) nor a callable pyfunc.".format(param_name))


def check_c_tensor_op(param, param_name):
    """check whether param is a tensor op or a callable Python function but not a py_transform"""
    if callable(param) and str(param).find("py_transform") >= 0:
        raise TypeError("{0} is a py_transform op which is not allow to use.".format(param_name))
    if not isinstance(param, cde.TensorOp) and not callable(param) and not getattr(param, 'parse', None):
        raise TypeError("{0} is neither a c_transform op (TensorOperation) nor a callable pyfunc.".format(param_name))


def replace_none(value, default):
    return value if value is not None else default
