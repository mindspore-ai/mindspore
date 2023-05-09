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
from __future__ import absolute_import
from types import FunctionType
from collections.abc import Iterable
from mindspore.ops import functional as F
from .. import context
from ..ops.primitive import constexpr
from ..common import Tensor, CSRTensor
from ..common import dtype as mstype


@constexpr
def _callable_const(x):
    """Returns true if x is a function in graph mode."""
    return isinstance(x, mstype.FunctionType)


@constexpr
def is_pynative():
    """Returns true if the current mode is PYNATIVE mode."""
    return context.get_context("mode") == context.PYNATIVE_MODE


def is_within_graph(x):
    """
    Returns true if x is None. It's aim to check whether the call is within MindSpore graph.
    Because in graph mode, x should be None in constexpr when x is a variable of MindSpore.
    Note that always return true if the call is in pynative mode.
    """
    return is_pynative() or not F.isconstant(x) or x is None


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
        info: info contains any object that can be recognized by graph mode.
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
        info: info contains any object that can be recognized by graph mode.
            All info's objects will be concatenated into a string to display.
    """
    info_str = ""
    for obj in info:
        info_str = info_str + f"{obj}"
    raise TypeError(info_str)


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
    if not isinstance(x, (tuple, list)):
        return (x,)

    tuple_x = ()
    for _x in x:
        tuple_x = tuple_x + _tuple(_x)

    return tuple_x


def mstype_to_pytype(type_):
    """
    Convert MindSpore type to Python type.

    Args:
        type_: A MindSpore type object.

    Returns:
        Type of Python type.
    """
    return {
        mstype.TensorType: Tensor,
        mstype.CSRTensorType: CSRTensor,
        mstype.FunctionType: FunctionType,
    }.get(type_)


_op_dict = StringDict()
_op_dict.register("in", lambda x: x[0] in _tuple(x[1]))
_op_dict.register("is", lambda x: x[0] is x[1])
# Here we use type() to check if two objects have the same class instead of isinstance(),
# since the later will return true in case of subclasses, e.g., 'bool' is a subclass of 'int'.
_op_dict.register("isinstance", lambda x: type(x[0]) in _tuple(x[1]))  # pylint: disable=C0123
_op_dict.register("solve", lambda x: x[0][1] == x[1][0])
_op_dict.register("==", lambda x: x[0] == x[1])


def _attr(args, names):
    func_name, arg_name, attr_name = _tuple(names)
    arg_value, valid_value = args
    num_values = len(valid_value) if isinstance(valid_value, Iterable) else 1
    return f"For '{func_name}', the {attr_name} of '{arg_name}' should be {'one of ' if num_values > 1 else ''}" + \
           f"{valid_value if num_values > 1 else valid_value}, " + \
           f"but got {arg_value}."


def _type(args, names):
    arg_value, valid_value = args
    func_name, arg_name = names
    valid_value = valid_value if isinstance(valid_value, Iterable) else (valid_value,)
    type_names = [t.__name__ if hasattr(t, '__name__') else str(t) for t in valid_value]
    num_values = len(valid_value)
    return f"For '{func_name}', the type of '{arg_name}' should be {'one of ' if num_values > 1 else ''}" + \
           f"{type_names if num_values > 1 else type_names[0]}, " + \
           f"but got '{arg_value}' with type {type(arg_value).__name__}."


def _square(args, names):
    func_name, arg_name, *_ = names
    return f"For '{func_name}', the matrix '{arg_name}' should be a square matrix like (N, N), " + \
           f"but got {args}."


def _match(args, names):
    arg1_value, arg2_value = args
    func_name, arg1_name, arg2_name, attr_name = _tuple(names)
    return f"For '{func_name}', the {attr_name} of '{arg1_name}' and '{arg2_name}' should be the same, but got " + \
           f"the {attr_name} of '{arg1_name}' is {arg1_value} and the {attr_name} of '{arg2_name}' is {arg2_value}."


def _mstype(_, names):
    arg, ms_type, func_name, arg_name = names
    py_type = [mstype_to_pytype(t) for t in _tuple(ms_type)]
    return _type((arg, py_type), (func_name, arg_name))


def _not_support(args, names):
    _, valid_value = args
    func_name, arg_name, *_ = names
    return f"For '{func_name}', currently only case {arg_name}={valid_value} of '{func_name}' is implemented."


def _solve(args, names):
    a_shape, b_shape = args
    func_name, a_name, b_name, sparse = names
    return f"For '{func_name}', the last two dimensions of '{a_name}' and '{b_name}' should be matched, " + \
           f"but got shape of {a_shape} and {b_shape}. " + \
           f"Please make sure that the shape of '{a_name}' and '{b_name}' be like (N, N) X (N, " \
           f"{'1' if sparse else 'M'}) or (N, N) X (N)."


_fmt_dict = StringDict()
_fmt_dict.register("attr", _attr)
_fmt_dict.register("square", _square)
_fmt_dict.register("type", _type)
_fmt_dict.register("match", _match)
_fmt_dict.register("mstype", _mstype)
_fmt_dict.register("todo", _not_support)
_fmt_dict.register("solve", _solve)


@constexpr
def _super_check(args, names, op, fmt, msg, val_err):
    """
    A flexible function is used to check whether type or value of variables is valid,
    which supports in both graph/pynative mode.

    Args:
        args(any): 'args' is used as one of argument for operation function and format function.
        names(any): 'names' is used as one of argument for format function.
        op(str): 'op' is a string to specify an operation. This operation will be obtained
            an actual function from a StringDict object, with 'args' as argument.
        fmt(str): 'fmt' is a string to specify a format. This format will be obtained
            an actual function from a StringDict object, with 'args' and 'names' as arguments.
        msg(str, tuple): 'msg' is used the case where format function is not necessary. When 'msg' is
            not None, we will throw the 'msg' as the error message.
        val_err(bool): Determine the type of TypeError/ValueError. When 'val_err' is True, raises
            ValueError, otherwise TypeError.

    Note:
        This function does not contain any parameter checks.
    """
    op_fn = _op_dict.get(op)
    if not op_fn(args):
        if not msg:
            fmt_fn = _fmt_dict.get(fmt)
            msg = fmt_fn(args, names)

        if val_err:
            _raise_value_error(*_tuple(msg))
        else:
            _raise_type_error(*_tuple(msg))

    return args


@constexpr
def pack(*args):
    return args
