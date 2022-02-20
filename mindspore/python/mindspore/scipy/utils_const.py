# Copyright 2022 Huawei Technologies Co., Ltd
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
# ============================================================================
"""internal graph-compatible utility functions"""
from collections.abc import Iterable
from ..ops.primitive import constexpr
from .._c_expression import typing
from ..common.dtype import tensor_type


@constexpr
def _callable_const(x):
    """Returns true if x is a function in graph mode."""
    return isinstance(x, typing.Function)


@constexpr
def _type_convert(new_type, obj):
    """
    Convert type of `obj` to `force`.
    """
    return new_type(obj)


@constexpr
def _raise_value_error(*info):
    """
    Raise ValueError in both graph/pynative mode

    Args:
        info(tuple): info contains any object that can be recognized by graph mode.
            All info's objects will be concatenated into a string to display.
    """
    info_str = ""
    for obj in info:
        info_str = info_str + f"{obj}"
    raise ValueError(info_str)


@constexpr
def _raise_type_error(*info):
    """
    Raise TypeError in both graph/pynative mode

    Args:
        info(tuple): info contains any object that can be recognized by graph mode.
            All info's objects will be concatenated into a string to display.
    """
    info_str = ""
    for obj in info:
        info_str = info_str + f"{obj}"
    raise TypeError(info_str)


@constexpr
def _type_check(arg_name, arg_value, valid_types, prim_name=None):
    """
    Checks whether a value is instance of some types.
    The same as mindspore._checkparam.Validator.check_value_type.
    This copy is to make it work in graph mode.
    """
    valid_types = valid_types if isinstance(valid_types, Iterable) else (valid_types,)

    def raise_error_msg():
        """func for raising error message when check failed"""
        type_names = [t.__name__ if hasattr(t, '__name__') else str(t) for t in valid_types]
        num_types = len(valid_types)
        msg_prefix = f"For '{prim_name}', the" if prim_name else "The"
        raise TypeError(f'{msg_prefix} type of `{arg_name}` should be {"one of " if num_types > 1 else ""}'
                        f'{type_names if num_types > 1 else type_names[0]}, '
                        f'but got \'{arg_value}\' with type {type(arg_value).__name__}.')

    # Notice: bool is subclass of int, so `check_value_type('x', True, [int])` will check fail, and
    #         `check_value_type('x', True, [bool, int])` will check pass
    if isinstance(arg_value, bool) and bool not in tuple(valid_types):
        raise_error_msg()
    if not isinstance(arg_value, tuple(valid_types)):
        raise_error_msg()

    return arg_value


class StringDict:
    """Registry class uses str to choose function."""

    def __init__(self):
        self.data = {}

    def register(self, obj_str, obj):
        """Register the str."""
        if not isinstance(obj_str, str):
            raise TypeError("key for Registry class must be string.")

        self.data[obj_str] = obj

    def get(self, obj_str):
        """Get the value by str."""
        if not isinstance(obj_str, str):
            raise TypeError("key for Registry class must be string.")

        obj = self.data.get(obj_str)
        return obj


def _tuple(x):
    x = x if isinstance(x, Iterable) else (x,)
    return tuple(x)


_op_dict = StringDict()
_op_dict.register("in", lambda x, y: x in _tuple(y))
_op_dict.register("is", lambda x, y: x is y)
_op_dict.register("isinstance", lambda x, y: isinstance(x, _tuple(y)))
_op_dict.register("istensor", lambda _, y: isinstance(y[0], tensor_type))


def _attr(arg_name, arg_value, valid_value, prim_name):
    attr, arg = arg_name
    num_values = len(valid_value) if isinstance(valid_value, Iterable) else 1
    return f"For '{prim_name}', the {attr} of '{arg}' should be {'one of ' if num_values > 1 else ''}" + \
           f"{valid_value if num_values > 1 else valid_value}, " + \
           f"but got {arg_value}."


def _type(arg_name, arg_value, valid_value, prim_name):
    valid_value = valid_value if isinstance(valid_value, Iterable) else (valid_value,)
    type_names = [t.__name__ if hasattr(t, '__name__') else str(t) for t in valid_value]
    num_values = len(valid_value)
    return f"For '{prim_name}', the type of '{arg_name}' should be {'one of ' if num_values > 1 else ''}" + \
           f"{type_names if num_values > 1 else type_names[0]}, " + \
           f"but got '{arg_value}' with type {type(arg_value).__name__}."


def _square(arg_name, arg_value, valid_value, prim_name):
    return f"For '{prim_name}', the matrix '{arg_name}' should be a square matrix like (N, N), " + \
           f"but got ({arg_value}, {valid_value})."


def _match(arg_name, arg_value, valid_value, prim_name):
    attr, arg1, arg2 = arg_name
    return f"For '{prim_name}', the {attr} of '{arg1}' and '{arg2}' should be the same, but got " + \
           f"the {attr} of '{arg1}' is {arg_value} and the {attr} of '{arg2}' is {valid_value}."


def _tensor(arg_name, arg_value, valid_value, prim_name):
    return _type(arg_name, arg_value, valid_value[1], prim_name)


_fmt_dict = StringDict()
_fmt_dict.register("attr", _attr)
_fmt_dict.register("square", _square)
_fmt_dict.register("type", _type)
_fmt_dict.register("match", _match)
_fmt_dict.register("tensor", _tensor)


@constexpr
def _super_check(op, arg_value, valid_value, prim_name, arg_name, fmt, msg, val_err):
    """Checks whether an input is valid."""
    op_fn = _op_dict.get(op)
    if not op_fn(arg_value, valid_value):
        if not msg:
            fmt_fn = _fmt_dict.get(fmt)
            msg = fmt_fn(arg_name, arg_value, valid_value, prim_name)

        if val_err:
            _raise_value_error(*_tuple(msg))
        else:
            _raise_type_error(*_tuple(msg))

    return arg_value
