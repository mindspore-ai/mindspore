# Copyright 2020-2022 Huawei Technologies Co., Ltd
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
from __future__ import absolute_import

import inspect
from multiprocessing import cpu_count
import os
import numpy as np

import mindspore._c_dataengine as cde
from mindspore import log as logger

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
    """
    Appends a space to the arg_name (if not empty)

    :param arg_name: the input string
    :return: the padded string
    """
    if arg_name != "":
        arg_name = arg_name + " "
    return arg_name


def check_value(value, valid_range, arg_name="", left_open_interval=False, right_open_interval=False):
    """
    Validates a value is within a desired range with left and right interval open or close.

    :param value: the value to be validated.
    :param valid_range: the desired range.
    :param arg_name: name of the variable to be validated.
    :param left_open_interval: True for left interval open and False for close.
    :param right_open_interval: True for right interval open and False for close.
    :return: Exception: when the validation fails, nothing otherwise.
    """
    arg_name = pad_arg_name(arg_name)
    if not left_open_interval and not right_open_interval:
        if value < valid_range[0] or value > valid_range[1]:
            raise ValueError(
                "Input {0}is not within the required interval of [{1}, {2}].".format(arg_name, valid_range[0],
                                                                                     valid_range[1]))
    elif left_open_interval and not right_open_interval:
        if value <= valid_range[0] or value > valid_range[1]:
            raise ValueError(
                "Input {0}is not within the required interval of ({1}, {2}].".format(arg_name, valid_range[0],
                                                                                     valid_range[1]))
    elif not left_open_interval and right_open_interval:
        if value < valid_range[0] or value >= valid_range[1]:
            raise ValueError(
                "Input {0}is not within the required interval of [{1}, {2}).".format(arg_name, valid_range[0],
                                                                                     valid_range[1]))
    else:
        if value <= valid_range[0] or value >= valid_range[1]:
            raise ValueError(
                "Input {0}is not within the required interval of ({1}, {2}).".format(arg_name, valid_range[0],
                                                                                     valid_range[1]))


def check_value_cutoff(value, valid_range, arg_name=""):
    """
    Validates a value is within a desired range [inclusive, exclusive).

    :param value: the value to be validated
    :param valid_range: the desired range
    :param arg_name: arg_name: arg_name: name of the variable to be validated
    :return: Exception: when the validation fails, nothing otherwise.
    """
    check_value(value, valid_range, arg_name, False, True)


def check_value_ratio(value, valid_range, arg_name=""):
    """
    Validates a value is within a desired range (exclusive, inclusive].

    :param value: the value to be validated
    :param valid_range: the desired range
    :param arg_name: arg_name: name of the variable to be validated
    :return: Exception: when the validation fails, nothing otherwise.
    """
    check_value(value, valid_range, arg_name, True, False)


def check_value_normalize_std(value, valid_range, arg_name=""):
    """
    Validates a value is within a desired range (exclusive, inclusive].

    :param value: the value to be validated
    :param valid_range: the desired range
    :param arg_name: arg_name: name of the variable to be validated
    :return: Exception: when the validation fails, nothing otherwise.
    """
    check_value(value, valid_range, arg_name, True, False)


def check_range(values, valid_range, arg_name=""):
    """
    Validates the boundaries a range are within a desired range [inclusive, inclusive].

    :param values: the two values to be validated
    :param valid_range: the desired range
    :param arg_name: arg_name: name of the variable to be validated
    :return: Exception: when the validation fails, nothing otherwise.
    """
    arg_name = pad_arg_name(arg_name)
    if not valid_range[0] <= values[0] <= values[1] <= valid_range[1]:
        raise ValueError(
            "Input {0}is not within the required interval of [{1}, {2}].".format(arg_name, valid_range[0],
                                                                                 valid_range[1]))


def check_positive(value, arg_name=""):
    """
    Validates the value of a variable is positive.

    :param value: the value of the variable
    :param arg_name: name of the variable to be validated
    :return: Exception: when the validation fails, nothing otherwise.
    """
    arg_name = pad_arg_name(arg_name)
    if value <= 0:
        raise ValueError("Input {0}must be greater than 0.".format(arg_name))


def check_int32_not_zero(value, arg_name=""):
    arg_name = pad_arg_name(arg_name)
    type_check(value, (int,), arg_name)
    if value < INT32_MIN or value > INT32_MAX or value == 0:
        raise ValueError(
            "Input {0}is not within the required interval of [-2147483648, 0) and (0, 2147483647].".format(arg_name))


def check_odd(value, arg_name=""):
    arg_name = pad_arg_name(arg_name)
    if value % 2 != 1:
        raise ValueError(
            "Input {0}is not an odd value.".format(arg_name))


def check_2tuple(value, arg_name=""):
    """
    Validates a variable is a tuple with two entries.

    :param value: the value of the variable
    :param arg_name: name of the variable to be validated
    :return: Exception: when the validation fails, nothing otherwise.
    """
    if not (isinstance(value, tuple) and len(value) == 2):
        raise ValueError("Value {0} needs to be a 2-tuple.".format(arg_name))


def check_int32(value, arg_name=""):
    """
    Validates the value of a variable is within the range of int32.

    :param value: the value of the variable
    :param arg_name: name of the variable to be validated
    :return: Exception: when the validation fails, nothing otherwise.
    """
    type_check(value, (int,), arg_name)
    check_value(value, [INT32_MIN, INT32_MAX], arg_name)


def check_uint8(value, arg_name=""):
    """
    Validates the value of a variable is within the range of uint8.

    :param value: the value of the variable
    :param arg_name: name of the variable to be validated
    :return: Exception: when the validation fails, nothing otherwise.
    """
    type_check(value, (int,), arg_name)
    check_value(value, [UINT8_MIN, UINT8_MAX], arg_name)


def check_uint32(value, arg_name=""):
    """
    Validates the value of a variable is within the range of uint32.

    :param value: the value of the variable
    :param arg_name: name of the variable to be validated
    :return: Exception: when the validation fails, nothing otherwise.
    """
    type_check(value, (int,), arg_name)
    check_value(value, [UINT32_MIN, UINT32_MAX], arg_name)


def check_pos_uint32(value, arg_name=""):
    """
    Validates the value of a variable is within the range of positive uint32.

    :param value: the value of the variable
    :param arg_name: name of the variable to be validated
    :return: Exception: when the validation fails, nothing otherwise.
    """
    type_check(value, (int,), arg_name)
    check_value(value, [POS_INT_MIN, UINT32_MAX], arg_name)


def check_pos_int32(value, arg_name=""):
    """
    Validates the value of a variable is within the range of int32.

    :param value: the value of the variable
    :param arg_name: name of the variable to be validated
    :return: Exception: when the validation fails, nothing otherwise.
    """
    type_check(value, (int,), arg_name)
    check_value(value, [POS_INT_MIN, INT32_MAX], arg_name)


def check_uint64(value, arg_name=""):
    """
    Validates the value of a variable is within the range of uint64.

    :param value: the value of the variable
    :param arg_name: name of the variable to be validated
    :return: Exception: when the validation fails, nothing otherwise.
    """
    type_check(value, (int,), arg_name)
    check_value(value, [UINT64_MIN, UINT64_MAX], arg_name)


def check_pos_int64(value, arg_name=""):
    """
    Validates the value of a variable is within the range of int64.

    :param value: the value of the variable
    :param arg_name: name of the variable to be validated
    :return: Exception: when the validation fails, nothing otherwise.
    """
    type_check(value, (int,), arg_name)
    check_value(value, [POS_INT_MIN, INT64_MAX], arg_name)


def check_non_negative_int32(value, arg_name=""):
    """
    Validates the value of a variable is within the range of non negative int32.

    :param value: the value of the variable.
    :param arg_name: name of the variable to be validated.
    :return: Exception: when the validation fails, nothing otherwise.
    """
    type_check(value, (int,), arg_name)
    check_value(value, [0, INT32_MAX], arg_name)


def check_float32(value, arg_name=""):
    """
    Validates the value of a variable is within the range of float32.

    :param value: the value of the variable
    :param arg_name: name of the variable to be validated
    :return: Exception: when the validation fails, nothing otherwise.
    """
    type_check(value, (float, int), arg_name)
    check_value(value, [FLOAT_MIN_INTEGER, FLOAT_MAX_INTEGER], arg_name)


def check_float64(value, arg_name=""):
    """
    Validates the value of a variable is within the range of float64.

    :param value: the value of the variable
    :param arg_name: name of the variable to be validated
    :return: Exception: when the validation fails, nothing otherwise.
    """
    type_check(value, (float, int), arg_name)
    check_value(value, [DOUBLE_MIN_INTEGER, DOUBLE_MAX_INTEGER], arg_name)


def check_pos_float32(value, arg_name=""):
    """
    Validates the value of a variable is within the range of positive float32.

    :param value: the value of the variable
    :param arg_name: name of the variable to be validated
    :return: Exception: when the validation fails, nothing otherwise.
    """
    type_check(value, (float, int), arg_name)
    check_value(value, [UINT32_MIN, FLOAT_MAX_INTEGER], arg_name, True)


def check_pos_float64(value, arg_name=""):
    """
    Validates the value of a variable is within the range of positive float64.

    :param value: the value of the variable
    :param arg_name: name of the variable to be validated
    :return: Exception: when the validation fails, nothing otherwise.
    """
    type_check(value, (float, int), arg_name)
    check_value(value, [UINT64_MIN, DOUBLE_MAX_INTEGER], arg_name, True)


def check_non_negative_float32(value, arg_name=""):
    """
    Validates the value of a variable is within the range of non negative float32.

    :param value: the value of the variable
    :param arg_name: name of the variable to be validated
    :return: Exception: when the validation fails, nothing otherwise.
    """
    type_check(value, (float, int), arg_name)
    check_value(value, [UINT32_MIN, FLOAT_MAX_INTEGER], arg_name)


def check_non_negative_float64(value, arg_name=""):
    """
    Validates the value of a variable is within the range of non negative float64.

    :param value: the value of the variable
    :param arg_name: name of the variable to be validated
    :return: Exception: when the validation fails, nothing otherwise.
    """
    type_check(value, (float, int), arg_name)
    check_value(value, [UINT32_MIN, DOUBLE_MAX_INTEGER], arg_name)


def check_float32_not_zero(value, arg_name=""):
    arg_name = pad_arg_name(arg_name)
    type_check(value, (float, int), arg_name)
    if value < FLOAT_MIN_INTEGER or value > FLOAT_MAX_INTEGER or value == 0:
        raise ValueError(
            "Input {0}is not within the required interval of [-16777216, 0) and (0, 16777216].".format(arg_name))


def check_valid_detype(type_):
    """
    Validates if a type is a DE Type.

    :param type_: the type_ to be validated
    :return: Exception: when the type is not a DE type, True otherwise.
    """
    if type_ not in valid_detype:
        raise TypeError("Unknown column type.")
    return True


def check_valid_str(value, valid_strings, arg_name=""):
    """
    Validates the content stored in a string.

    :param value: the value to be validated
    :param valid_strings: a list/set of valid strings
    :param arg_name: name of the variable to be validated
    :return: Exception: when the type is not a DE type, nothing otherwise.
    """
    type_check(value, (str,), arg_name)
    if value not in valid_strings:
        raise ValueError("Input {0} is not within the valid set of {1}.".format(arg_name, str(valid_strings)))


def check_valid_list_tuple(value, valid_list_tuple, data_type, arg_name=""):
    """
    Validate value in valid_list_tuple.

    Args:
        value (Union[list, tuple]): the value to be validated.
        valid_strings (Union[list, tuple]): name of columns.
        type (tuple): tuple of all valid types for value.
        arg_name (str): the names of value.
    Returns:
        Exception: when the value is not correct, otherwise nothing.
    """
    valid_length = len(valid_list_tuple[0])
    type_check(value, (list, tuple), arg_name)
    type_check_list(value, data_type, arg_name)
    if len(value) != valid_length:
        raise ValueError("Input {0} is a list or tuple of length {1}.".format(arg_name, valid_length))
    if value not in valid_list_tuple:
        raise ValueError(
            "Input {0}{1} is not within the valid set of {2}.".format(arg_name, value, valid_list_tuple))


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
        Exception: when the validation fails, otherwise nothing.
    """

    if int in types and bool not in types:
        if isinstance(arg, bool):
            # handle special case of booleans being a subclass of ints
            print_value = '\"\"' if repr(arg) == repr('') else arg
            raise TypeError("Argument {0} with value {1} is not of type {2}, but got {3}.".format(arg_name, print_value,
                                                                                                  types, type(arg)))
    if not isinstance(arg, types):
        print_value = '\"\"' if repr(arg) == repr('') else arg
        raise TypeError("Argument {0} with value {1} is not of type {2}, but got {3}.".format(arg_name, print_value,
                                                                                              list(types), type(arg)))


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
    filename = os.path.basename(os.path.realpath(path))
    forbidden_symbols = set(r'\/:*?"<>|`&\';')

    if set(filename) & forbidden_symbols:
        raise ValueError(r"filename should not contain \/:*?\"<>|`&;\'")

    if filename.startswith(' ') or filename.endswith(' '):
        raise ValueError("filename should not start/end with space.")


def check_dir(dataset_dir):
    """
    Validates if the argument is a directory.

    :param dataset_dir: string containing directory path
    :return: Exception: when the validation fails, nothing otherwise.
    """
    type_check(dataset_dir, (str,), "dataset_dir")
    if not os.path.isdir(dataset_dir) or not os.access(dataset_dir, os.R_OK):
        raise ValueError("The folder {} does not exist or is not a directory or permission denied!".format(dataset_dir))


def check_list_same_size(list1, list2, list1_name="", list2_name=""):
    """
    Validates the two lists as the same size.

    :param list1: the first list to be validated
    :param list2: the secend list to be validated
    :param list1_name: name of the list1
    :param list2_name: name of the list2
    :return: Exception: when the two list no same size, nothing otherwise.
    """
    if len(list1) != len(list2):
        raise ValueError("The size of {0} should be the same as that of {1}.".format(list1_name, list2_name))


def check_file(dataset_file):
    """
    Validates if the argument is a valid file name.

    :param dataset_file: string containing file path
    :return: Exception: when the validation fails, nothing otherwise.
    """
    check_filename(dataset_file)
    dataset_file = os.path.realpath(dataset_file)
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
        check_pos_int32(num_shards, "num_shards")
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
    """
    Validates the value for num_parallel_workers.
.
    :param value: an integer corresponding to the number of parallel workers
    :return: Exception: when the validation fails, nothing otherwise.
    """
    type_check(value, (int,), "num_parallel_workers")
    if value < 1 or value > cpu_count():
        raise ValueError("num_parallel_workers exceeds the boundary between 1 and {}!".format(cpu_count()))


def check_num_samples(value):
    """
    Validates number of samples are valid.
.
    :param value: an integer corresponding to the number of samples.
    :return: Exception: when the validation fails, nothing otherwise.
    """
    type_check(value, (int,), "num_samples")
    if value < 0 or value > INT64_MAX:
        raise ValueError(
            "num_samples exceeds the boundary between {} and {}(INT64_MAX)!".format(0, INT64_MAX))


def validate_dataset_param_value(param_list, param_dict, param_type):
    """

    :param param_list: a list of parameter names.
    :param param_dict: a dcitionary containing parameter names and their values.
    :param param_type: a tuple containing type of parameters.
    :return: Exception: when the validation fails, nothing otherwise.
    """
    for param_name in param_list:
        if param_dict.get(param_name) is not None:
            if param_name == 'num_parallel_workers':
                check_num_parallel_workers(param_dict.get(param_name))
            if param_name == 'num_samples':
                check_num_samples(param_dict.get(param_name))
            else:
                type_check(param_dict.get(param_name), (param_type,), param_name)


def check_gnn_list_of_pair_or_ndarray(param, param_name):
    """
    Check if the input parameter is a list of tuple or numpy.ndarray.

    Args:
        param (Union[list[tuple], nd.ndarray]): param.
        param_name (str): param_name.

    Returns:
        Exception: TypeError if error.
    """
    type_check(param, (list, np.ndarray), param_name)
    if isinstance(param, list):
        param_names = ["node_list[{0}]".format(i) for i in range(len(param))]
        type_check_list(param, (tuple,), param_names)
        for idx, pair in enumerate(param):
            if not len(pair) == 2:
                raise ValueError("Each member in {0} must be a pair which means length == 2. Got length {1}".format(
                    param_names[idx], len(pair)))
            column_names = ["node_list[{0}], number #{1} element".format(idx, i + 1) for i in range(len(pair))]
            type_check_list(pair, (int,), column_names)
    elif isinstance(param, np.ndarray):
        if param.ndim != 2:
            raise ValueError("Input ndarray must be in dimension 2. Got {0}".format(param.ndim))
        if param.shape[1] != 2:
            raise ValueError("Each member in {0} must be a pair which means length == 2. Got length {1}".format(
                param_name, param.shape[1]))
        if not param.dtype == np.int32:
            raise TypeError("Each member in {0} should be of type int32. Got {1}.".format(
                param_name, param.dtype))


def check_gnn_list_or_ndarray(param, param_name, data_type=int):
    """
    Check if the input parameter is list or numpy.ndarray.

    Args:
        param (Union[list, nd.ndarray]): param.
        param_name (str): param_name.
        data_type(object): data type.

    Returns:
        Exception: TypeError if error.
    """

    type_check(param, (list, np.ndarray), param_name)
    if isinstance(param, list):
        param_names = ["param_{0}".format(i) for i in range(len(param))]
        type_check_list(param, (data_type,), param_names)

    elif isinstance(param, np.ndarray):
        if data_type == int:
            data_type = np.int32
        elif data_type == str:
            data_type = np.str_

        if param.dtype != data_type:
            raise TypeError("Each member in {0} should be of type {1}. Got {2}.".format(
                param_name, data_type, param.dtype))


def check_tensor_op(param, param_name):
    """check whether param is a tensor op or a callable Python function"""
    if not isinstance(param, cde.TensorOp) and not callable(param) and not getattr(param, 'parse', None):
        raise TypeError("{0} is neither a transforms op (TensorOperation) nor a callable pyfunc.".format(param_name))


def check_c_tensor_op(param, param_name):
    """check whether param is a tensor op or a callable Python function but not a py_transform"""
    if callable(param) and str(param).find("py_transform") >= 0:
        raise TypeError("{0} is a py_transform op which is not allowed to use.".format(param_name))
    if not isinstance(param, cde.TensorOp) and not callable(param) and not getattr(param, 'parse', None):
        raise TypeError("{0} is neither a c_transform op (TensorOperation) nor a callable pyfunc.".format(param_name))


def replace_none(value, default):
    """ replaces None with a default value."""
    return value if value is not None else default


def check_dataset_num_shards_shard_id(num_shards, shard_id):
    if (num_shards is None) != (shard_id is None):
        # These two parameters appear together.
        raise ValueError("num_shards and shard_id need to be passed in together.")
    if num_shards is not None:
        check_pos_int32(num_shards, "num_shards")
        if shard_id >= num_shards:
            raise ValueError("shard_id should be less than num_shards.")


def deprecator_factory(version, old_module, new_module, substitute_name=None, substitute_module=None):
    """Decorator factory function for deprecated operation to log deprecation warning message.

    Args:
        version (str): Version that the operation is deprecated.
        old_module (str): Old module for deprecated operation.
        new_module (str): New module for deprecated operation.
        substitute_name (str, optional): The substitute name for deprecated operation.
        substitute_module (str, optional): The substitute module for deprecated operation.
    """

    def decorator(op):
        def wrapper(*args, **kwargs):
            # Get operation class name for operation class which applies decorator to __init__()
            name = str(op).split()[1].split(".")[0]
            # Build message
            message = f"'{name}' from " + f"{old_module}" + f" is deprecated from version " f"{version}" + \
                      " and will be removed in a future version."
            message += f" Use '{substitute_name}'" if substitute_name else f" Use '{name}'"
            message += f" from {substitute_module} instead." if substitute_module \
                else f" from " f"{new_module}" + " instead."

            # Log warning message
            logger.warning(message)

            ret = op(*args, **kwargs)
            return ret

        return wrapper

    return decorator


def check_dict(data, key_type, value_type, param_name):
    """ check key and value type in dict."""
    if data is not None:
        if not isinstance(data, dict):
            raise TypeError("{0} should be dict type, but got: {1}".format(param_name, type(data)))

        for key, value in data.items():
            if not isinstance(key, key_type):
                raise TypeError("key '{0}' in parameter {1} should be {2} type, but got: {3}"
                                .format(key, param_name, key_type, type(key)))
            if not isinstance(value, value_type):
                raise TypeError("value of '{0}' in parameter {1} should be {2} type, but got: {3}"
                                .format(key, param_name, value_type, type(value)))


def check_feature_shape(data, shape, param_name):
    if isinstance(data, dict):
        for key, value in data.items():
            if len(value.shape) != 2 or value.shape[0] != shape:
                raise ValueError("Shape of '{0}' in '{1}' should be of 2 dimension, and shape of first dimension "
                                 "should be: {2}, but got: {3}.".format(key, param_name, shape, value.shape))
