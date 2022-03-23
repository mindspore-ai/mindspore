# Copyright 2021 Huawei Technologies Co., Ltd
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
General Validator Helper Functions.
"""
import os
import inspect

UINT32_MAX = 4294967295
UINT32_MIN = 0
UINT64_MAX = 18446744073709551615
UINT64_MIN = 0


def pad_arg_name(arg_name):
    """Add a space for arg_name."""
    if arg_name != "":
        arg_name = arg_name + " "
    return arg_name


def check_value(arg, valid_range, arg_name=""):
    """Check the value of arg is in a valid range."""
    arg_name = pad_arg_name(arg_name)
    if arg < valid_range[0] or arg > valid_range[1]:
        raise ValueError(
            "Input {0}is not within the required interval of ({1} to {2}).".format(arg_name,
                                                                                   valid_range[0], valid_range[1]))


def check_uint32(arg, arg_name=""):
    """Check arg type is uint32."""
    type_check(arg, (int,), arg_name)
    check_value(arg, [UINT32_MIN, UINT32_MAX])


def check_uint64(arg, arg_name=""):
    """Check arg type is uint64."""
    type_check(arg, (int,), arg_name)
    check_value(arg, [UINT64_MIN, UINT64_MAX])


def check_iteration(arg, arg_name=""):
    """Check arg is in a valid range."""
    type_check(arg, (int,), arg_name)
    check_value(arg, [-1, UINT64_MAX])


def check_dir(dataset_dir):
    """Check the dataset_dir is a valid dir."""
    if not os.path.isdir(dataset_dir) or not os.access(dataset_dir, os.R_OK):
        raise ValueError("The folder {} does not exist or permission denied!".format(dataset_dir))


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


def replace_minus_one(value):
    """ replace -1 with a default value """
    return value if value != -1 else UINT32_MAX


def check_param_id(info_param, info_name):
    """
    Check the type of info_param.

    Args:
        info_param (Union[list[int], str]): Info parameters of check_node_list that is either list of ints or *.
        info_name (str): Info name of check_node_list.

    Raises:
        ValueError: When the type of info_param is not correct, otherwise nothing.
    """
    if isinstance(info_param, str):
        if info_param not in ["*"]:
            raise ValueError("Node parameter {} only accepts '*' as string.".format(info_name))
    else:
        for param in info_param:
            check_uint32(param, info_name)
