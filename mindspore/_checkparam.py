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
# ============================================================================
"""Check parameters."""

import re
import inspect
import math
from enum import Enum
from functools import reduce, wraps
from itertools import repeat
from collections.abc import Iterable
import numpy as np
from mindspore import log as logger
from mindspore.common import dtype as mstype


class Rel(Enum):
    """Numerical relationship between variables, logical relationship enumeration definition of range."""
    # scalar compare
    EQ = 1  # ==
    NE = 2  # !=
    LT = 3  # <
    LE = 4  # <=
    GT = 5  # >
    GE = 6  # >=
    # scalar range check
    INC_NEITHER = 7  # (), include neither
    INC_LEFT = 8  # [), include left
    INC_RIGHT = 9  # (], include right
    INC_BOTH = 10  # [], include both
    # collection in, not in
    IN = 11
    NOT_IN = 12

    @staticmethod
    def get_strs(rel):
        """Get value from rel_strs."""
        return rel_strs.get(rel, "")

    @staticmethod
    def get_fns(rel):
        """Get value from rel_fns."""
        return rel_fns.get(rel, lambda *args: False)


rel_fns = {
    # scalar compare
    Rel.EQ: lambda x, y: x == y,
    Rel.NE: lambda x, y: x != y,
    Rel.LT: lambda x, y: x < y,
    Rel.LE: lambda x, y: x <= y,
    Rel.GT: lambda x, y: x > y,
    Rel.GE: lambda x, y: x >= y,
    # scalar range check
    Rel.INC_NEITHER: lambda x, lower, upper: (lower < x < upper),
    Rel.INC_LEFT: lambda x, lower, upper: (lower <= x < upper),
    Rel.INC_RIGHT: lambda x, lower, upper: (lower < x <= upper),
    Rel.INC_BOTH: lambda x, lower, upper: (lower <= x <= upper),
    # collection in, not in
    Rel.IN: lambda x, y: x in y,
    Rel.NOT_IN: lambda x, y: x not in y,
}

rel_strs = {
    # scalar compare
    Rel.EQ: "== {}",
    Rel.NE: "!= {}",
    Rel.LT: "< {}",
    Rel.LE: "<= {}",
    Rel.GT: "> {}",
    Rel.GE: ">= {}",
    # scalar range check
    Rel.INC_NEITHER: "({}, {})",
    Rel.INC_LEFT: "[{}, {})",
    Rel.INC_RIGHT: "({}, {}]",
    Rel.INC_BOTH: "[{}, {}]",
    # collection in, not in
    Rel.IN: "in {}",
    Rel.NOT_IN: "not in {}",
}


def _check_3d_int_or_tuple(arg_name, arg_value, prim_name, allow_five=False, ret_five=False,
                           greater_zero=True, third_one=False, three_input=False):
    """
    Checks whether an argument is a positive int or tuple with 3 or 5(when allow_five is True) positive int elements.
    """

    def _raise_message(third_one_flag=False, three_input_flag=False):
        if third_one_flag:
            raise ValueError(f"For '{prim_name}' the depth of attr '{arg_name}' should be 1, but got {ret_value[-3]}")
        if three_input_flag:
            raise ValueError(f"For '{prim_name}' attr '{arg_name}' should be an positive int number or a tuple of "
                             f"three positive int numbers, but got {arg_value}")
        raise ValueError(f"For '{prim_name}' attr '{arg_name}' should be an positive int number or a tuple of three "
                         f"{'or five ' if allow_five else ''}positive int numbers, but got {arg_value}")

    def _get_return_value():
        if isinstance(arg_value, int):
            ret = (1, 1, arg_value, arg_value, arg_value) if ret_five else (arg_value, arg_value, arg_value)
        elif len(arg_value) == 3:
            ret = (1, 1, arg_value[0], arg_value[1], arg_value[2]) if ret_five else arg_value
        elif len(arg_value) == 5:
            if not allow_five:
                _raise_message()
            ret = arg_value if ret_five else (arg_value[1], arg_value[2], arg_value[3])
        else:
            _raise_message()
        return ret

    Validator.check_value_type(arg_name, arg_value, (int, tuple), prim_name)
    if three_input and isinstance(arg_value, tuple):
        if len(arg_value) != 3:
            _raise_message(three_input_flag=three_input)
    ret_value = _get_return_value()
    for item in ret_value:
        if isinstance(item, int) and not isinstance(item, bool):
            if greater_zero and item > 0:
                continue
            if not greater_zero and item >= 0:
                continue
            _raise_message()

    if third_one:
        if ret_value[-3] != 1:
            _raise_message(third_one_flag=third_one)

    return tuple(ret_value)


def check_number(arg_value, value, rel, arg_type=int, arg_name=None, prim_name=None):
    """
    Check argument integer.

    Example:
    - number = check_int(number, 0, Rel.GE, "number", None) # number >= 0
    """
    rel_fn = Rel.get_fns(rel)
    type_mismatch = not isinstance(arg_value, arg_type) or isinstance(arg_value, bool)
    type_except = TypeError if type_mismatch else ValueError

    prim_name = f'in `{prim_name}`' if prim_name else ''
    arg_name = f'`{arg_name}`' if arg_name else ''
    if math.isinf(arg_value) or math.isnan(arg_value) or np.isinf(arg_value) or np.isnan(arg_value):
        raise ValueError(f'{arg_name} {prim_name} must be legal value, but got `{arg_value}`.')
    if type_mismatch or not rel_fn(arg_value, value):
        rel_str = Rel.get_strs(rel).format(value)
        raise type_except(f'{arg_name} {prim_name} should be an {arg_type.__name__} and must {rel_str}, '
                          f'but got `{arg_value}` with type `{type(arg_value).__name__}`.')

    return arg_value


def check_is_number(arg_value, arg_type, arg_name=None, prim_name=None):
    """
    Checks input value is float type or not.

    Usage:
    - number = check_is_number(number, int)
    - number = check_is_number(number, int, "bias")
    - number = check_is_number(number, int, "bias", "bias_class")
    """
    prim_name = f'in \'{prim_name}\'' if prim_name else ''
    arg_name = f'\'{arg_name}\'' if arg_name else 'Input value'
    if isinstance(arg_value, arg_type) and not isinstance(arg_value, bool):
        if math.isinf(arg_value) or math.isnan(arg_value) or np.isinf(arg_value) or np.isnan(arg_value):
            raise ValueError(f'{arg_name} {prim_name} must be legal float, but got `{arg_value}`.')
        return arg_value
    raise TypeError(f'{arg_name} {prim_name} must be {arg_type.__name__}, but got `{type(arg_value).__name__}`')


def check_number_range(arg_value, lower_limit, upper_limit, rel, value_type, arg_name=None, prim_name=None):
    """
    Method for checking whether an int value is in some range.

    Usage:
    - number = check_number_range(number, 0.0, 1.0, Rel.INC_NEITHER, "number", float) # number in [0.0, 1.0]
    - number = check_number_range(number, 0, 1, Rel.INC_NEITHER, "number", int) # number in [0, 1]
    """
    rel_fn = Rel.get_fns(rel)
    prim_name = f'in `{prim_name}`' if prim_name else ''
    arg_name = f'`{arg_name}`' if arg_name else ''
    type_mismatch = not isinstance(arg_value, (np.ndarray, np.generic, value_type)) or isinstance(arg_value, bool)
    if type_mismatch:
        raise TypeError("{} {} must be `{}`,  but got `{}`.".format(
            arg_name, prim_name, value_type.__name__, type(arg_value).__name__))
    if not rel_fn(arg_value, lower_limit, upper_limit):
        rel_str = Rel.get_strs(rel).format(lower_limit, upper_limit)
        raise ValueError("{} {} should be in range of {}, but got {:.3e} with type `{}`.".format(
            arg_name, prim_name, rel_str, arg_value, type(arg_value).__name__))
    return arg_value


class Validator:
    """validator for checking input parameters"""

    @staticmethod
    def check(arg_name, arg_value, value_name, value, rel=Rel.EQ, prim_name=None, excp_cls=ValueError):
        """
        Method for judging relation between two int values or list/tuple made up of ints.
        This method is not suitable for judging relation between floats, since it does not consider float error.
        """
        rel_fn = Rel.get_fns(rel)
        if not rel_fn(arg_value, value):
            rel_str = Rel.get_strs(rel).format(f'{value_name}: {value}')
            msg_prefix = f'For \'{prim_name}\' the' if prim_name else "The"
            raise excp_cls(f'{msg_prefix} `{arg_name}` should be {rel_str}, but got {arg_value}.')
        return arg_value

    @staticmethod
    def check_int(arg_value, value, rel, arg_name=None, prim_name=None):
        """
        Checks input integer value `arg_value` compare to `value`.

        Usage:
        - number = check_int(number, 0, Rel.GE, "number", None) # number >= 0
        """
        return check_number(arg_value, value, rel, int, arg_name, prim_name)

    @staticmethod
    def check_is_int(arg_value, arg_name=None, prim_name=None):
        """
        Checks input value is float type or not.

        Usage:
        - number = check_is_int(number, int)
        - number = check_is_int(number, int, "bias")
        - number = check_is_int(number, int, "bias", "bias_class")
        """
        return check_is_number(arg_value, int, arg_name, prim_name)

    @staticmethod
    def check_equal_int(arg_value, value, arg_name=None, prim_name=None):
        """
        Checks input integer value `arg_value` compare to `value`.

        Usage:
        - number = check_int(number, 0, Rel.GE, "number", None) # number >= 0
        """
        return check_number(arg_value, value, Rel.EQ, int, arg_name, prim_name)

    @staticmethod
    def check_positive_int(arg_value, arg_name=None, prim_name=None):
        """
        Check argument is positive integer, which mean arg_value > 0.

        Usage:
        - number = check_positive_int(number)
        - number = check_positive_int(number, "bias")
        """
        return check_number(arg_value, 0, Rel.GT, int, arg_name, prim_name)

    @staticmethod
    def check_negative_int(arg_value, arg_name=None, prim_name=None):
        """
        Check argument is negative integer, which mean arg_value < 0.

        Usage:
        - number = check_negative_int(number)
        - number = check_negative_int(number, "bias")
        """
        return check_number(arg_value, 0, Rel.LT, int, arg_name, prim_name)

    @staticmethod
    def check_non_positive_int(arg_value, arg_name=None, prim_name=None):
        """
        Check argument is non-negative integer, which mean arg_value <= 0.

        Usage:
        - number = check_non_positive_int(number)
        - number = check_non_positive_int(number, "bias")
        """
        return check_number(arg_value, 0, Rel.LE, int, arg_name, prim_name)

    @staticmethod
    def check_non_negative_int(arg_value, arg_name=None, prim_name=None):
        """
        Check argument is non-negative integer, which mean arg_value >= 0.

        Usage:
        - number = check_non_negative_int(number)
        - number = check_non_negative_int(number, "bias")
        """
        return check_number(arg_value, 0, Rel.GE, int, arg_name, prim_name)

    @staticmethod
    def check_float(arg_value, value, rel, arg_name=None, prim_name=None):
        """
        Checks input float value `arg_value` compare to `value`.

        Usage:
        - number = check_float(number, 0.0, Rel.GE, "number", None) # number >= 0
        """
        return check_number(arg_value, value, rel, float, arg_name, prim_name)

    @staticmethod
    def check_is_float(arg_value, arg_name=None, prim_name=None):
        """
        Checks input value is float type or not.

        Usage:
        - number = check_is_float(number, int)
        - number = check_is_float(number, int, "bias")
        - number = check_is_float(number, int, "bias", "bias_class")
        """
        return check_is_number(arg_value, float, arg_name, prim_name)

    @staticmethod
    def check_positive_float(arg_value, arg_name=None, prim_name=None):
        """
        Check argument is positive float, which mean arg_value > 0.

        Usage:
        - number = check_positive_float(number)
        - number = check_positive_float(number, "bias")
        - number = check_positive_float(number, "bias", "bias_class")
        """
        return check_number(arg_value, 0, Rel.GT, float, arg_name, prim_name)

    @staticmethod
    def check_negative_float(arg_value, arg_name=None, prim_name=None):
        """
        Check argument is negative float, which mean arg_value < 0.

        Usage:
        - number = check_negative_float(number)
        - number = check_negative_float(number, "bias")
        """
        return check_number(arg_value, 0, Rel.LT, float, arg_name, prim_name)

    @staticmethod
    def check_non_positive_float(arg_value, arg_name=None, prim_name=None):
        """
        Check argument is non-negative float, which mean arg_value <= 0.

        Usage:
        - number = check_non_positive_float(number)
        - number = check_non_positive_float(number, "bias")
        """
        return check_number(arg_value, 0, Rel.LE, float, arg_name, prim_name)

    @staticmethod
    def check_non_negative_float(arg_value, arg_name=None, prim_name=None):
        """
        Check argument is non-negative float, which mean arg_value >= 0.

        Usage:
        - number = check_non_negative_float(number)
        - number = check_non_negative_float(number, "bias")
        """
        return check_number(arg_value, 0, Rel.GE, float, arg_name, prim_name)

    @staticmethod
    def check_number(arg_name, arg_value, value, rel, prim_name):
        """Number value judgment."""
        rel_fn = Rel.get_fns(rel)
        if not rel_fn(arg_value, value):
            rel_str = Rel.get_strs(rel).format(value)
            raise ValueError(f'For \'{prim_name}\' the `{arg_name}` must {rel_str}, but got {arg_value}.')
        return arg_value

    @staticmethod
    def check_isinstance(arg_name, arg_value, classes):
        """Check arg isinstance of classes"""
        if not isinstance(arg_value, classes):
            raise ValueError(f'The `{arg_name}` should be isinstance of {classes}, but got {arg_value}.')
        return arg_value

    @staticmethod
    def check_bool(arg_value, arg_name=None):
        """
        Check argument is instance of bool.

        Usage:
        - has_bias = check_bool(has_bias)
        - has_bias = check_bool(has_bias, "has_bias")
        """
        if not isinstance(arg_value, bool):
            arg_name = arg_name if arg_name else "Parameter"
            raise TypeError(f'`{arg_name}` should be isinstance of bool, but got `{arg_value}`.')
        return arg_value

    @staticmethod
    def check_int_range(arg_value, lower_limit, upper_limit, rel, arg_name=None, prim_name=None):
        """
        Method for checking whether input value is in int range.

        Usage:
        - number = check_int_range(number, 0, 1, Rel.INC_NEITHER) # number in [0, 1]
        - number = check_int_range(number, 0, 1, Rel.INC_NEITHER, "number") # number in [0, 1]
        """
        return check_number_range(arg_value, lower_limit, upper_limit, rel, int, arg_name, prim_name)

    @staticmethod
    def check_float_range(arg_value, lower_limit, upper_limit, rel, arg_name=None, prim_name=None):
        """
        Method for checking whether input value is in float range.

        Usage:
        - number = check_float_range(number, 0.0, 1.0, Rel.INC_NEITHER) # number in [0.0, 1.0]
        - number = check_float_range(number, 0.0, 1.0, Rel.INC_NEITHER, "number") # number in [0.0, 1.0]
        """
        return check_number_range(arg_value, lower_limit, upper_limit, rel, float, arg_name, prim_name)

    @staticmethod
    def check_string(arg_value, valid_values, arg_name=None, prim_name=None):
        """
        Check whether string is in some value list.

        Usage:
        - method = check_string(method, ["string1", "string2", "string3"], "method")
        """
        if isinstance(arg_value, str) and arg_value in valid_values:
            return arg_value
        arg_name = arg_name if arg_name else "Parameter"
        msg_prefix = f'For \'{prim_name}\' the' if prim_name else "The"
        raise ValueError(f'{msg_prefix} `{arg_name}` should be str and must be in `{valid_values}`,'
                         f' but got `{arg_value}`.')

    @staticmethod
    def check_str_by_regular(target, reg=None, flag=re.ASCII, prim_name=None):
        if reg is None:
            # Named string regular expression
            reg = r"^\w+[0-9a-zA-Z\_\.]*$"
        if re.match(reg, target, flag) is None:
            prim_name = f'in `{prim_name}`' if prim_name else ""
            raise ValueError("'{}' {} is illegal, it should be match regular'{}' by flags'{}'".format(
                target, prim_name, reg, flag))
        return True

    @staticmethod
    def check_file_name_by_regular(target, reg=None, flag=re.ASCII, prim_name=None):
        """Check whether file name is legitimate."""
        if reg is None:
            reg = r"^[0-9a-zA-Z\_\-\.\:\/\\]+$"
        if re.match(reg, target, flag) is None:
            prim_name = f'in `{prim_name}`' if prim_name else ""
            raise ValueError("'{}' {} is illegal, it should be match regular'{}' by flags'{}'".format(
                target, prim_name, reg, flag))
        return True

    @staticmethod
    def check_pad_value_by_mode(pad_mode, padding, prim_name):
        """Validates value of padding according to pad_mode"""
        if pad_mode != 'pad' and padding != 0:
            raise ValueError(f"For '{prim_name}', padding must be zero when pad_mode is '{pad_mode}'.")
        return padding

    @staticmethod
    def check_subclass(arg_name, type_, template_types, prim_name, addition_error_info=None):
        """Checks whether some type is subclass of another type"""
        if not isinstance(template_types, Iterable):
            template_types = (template_types,)
        hit = False
        for template_type in template_types:
            if isinstance(template_type, mstype.Type):
                if mstype.issubclass_(type_, template_type):
                    hit = True
                    break
            elif type_ is template_type:
                hit = True
                break
        if not hit:
            if addition_error_info is None:
                addition_error_info = ''
            type_str = (type(type_).__name__ if isinstance(type_, (tuple, list)) else "") + str(type_)
            raise TypeError(f'For \'{prim_name}\', the type of `{arg_name}` should be subclass'
                            f' of {", ".join((str(x) for x in template_types))}, but got {type_str}.'
                            f' {addition_error_info}')

    @staticmethod
    def check_const_input(arg_name, arg_value, prim_name):
        """Checks valid value."""
        if arg_value is None:
            raise ValueError(f'For \'{prim_name}\', the `{arg_name}` must be a const input, but got {arg_value}.')
        return arg_value

    @staticmethod
    def check_types_same_and_valid(args, valid_values, prim_name):
        """Checks whether the types of inputs are the same and valid."""

        def _check_type_valid(arg):
            arg_key, arg_val = arg
            elem_type = arg_val
            Validator.check_subclass(arg_key, elem_type, valid_values, prim_name)
            return (arg_key, elem_type)

        def _check_types_same(arg1, arg2):
            arg1_name, arg1_type = arg1
            arg2_name, arg2_type = arg2
            if arg1_type != arg2_type:
                raise TypeError(f'For \'{prim_name}\', type of `{arg2_name}` should be same as `{arg1_name}`,'
                                f' but `{arg1_name}` with type {arg1_type} and `{arg2_name}` with type {arg2_type}.')
            return arg1

        elem_types = map(_check_type_valid, args.items())
        reduce(_check_types_same, elem_types)

    @staticmethod
    def check_tensors_dtypes_same_and_valid(args, valid_dtypes, prim_name):
        """Checks whether the element types of input tensors are the same and valid."""
        valid_dtypes = valid_dtypes if isinstance(valid_dtypes, Iterable) else [valid_dtypes]
        tensor_types = [mstype.tensor_type(t) for t in valid_dtypes]
        Validator.check_types_same_and_valid(args, tensor_types, prim_name)

    @staticmethod
    def check_tensor_dtype_valid(arg_name, arg_type, valid_dtypes, prim_name):
        """Checks whether the element types of input tensors are valid."""
        valid_dtypes = valid_dtypes if isinstance(valid_dtypes, Iterable) else [valid_dtypes]
        tensor_types = [mstype.tensor_type(t) for t in valid_dtypes]
        Validator.check_subclass(arg_name, arg_type, tensor_types, prim_name)

    @staticmethod
    def check_scalar_or_tensor_types_same(args, valid_values, prim_name, allow_mix=False):
        """
        Checks whether the types of inputs are the same. If the input args are tensors, checks their element types.
        If `allow_mix` is True, Tensor(float32) and float32 are type compatible, otherwise an exception will be raised.
        """

        def _check_argument_type(arg):
            arg_key, arg_val = arg
            if isinstance(arg_val, type(mstype.tensor)):
                arg_val = arg_val.element_type()
            if not arg_val in valid_values:
                raise TypeError(f'For \'{prim_name}\', the `{arg_key}` should be in {valid_values},'
                                f' but `{arg_key}` is {arg_val}.')
            return arg

        def _check_types_same(arg1, arg2):
            arg1_name, arg1_type = arg1
            arg2_name, arg2_type = arg2
            except_flag = False
            if isinstance(arg1_type, type(mstype.tensor)) and isinstance(arg2_type, type(mstype.tensor)):
                arg1_type = arg1_type.element_type()
                arg2_type = arg2_type.element_type()
            elif not (isinstance(arg1_type, type(mstype.tensor)) or isinstance(arg2_type, type(mstype.tensor))):
                pass
            elif allow_mix:
                arg1_type = arg1_type.element_type() if isinstance(arg1_type, type(mstype.tensor)) else arg1_type
                arg2_type = arg2_type.element_type() if isinstance(arg2_type, type(mstype.tensor)) else arg2_type
            else:
                except_flag = True

            if except_flag or arg1_type != arg2_type:
                raise TypeError(f'For \'{prim_name}\' type of `{arg2_name}` should be same as `{arg1_name}`,'
                                f' but `{arg1_name}` is {arg1_type} and `{arg2_name}` is {arg2_type}.')
            return arg1

        reduce(_check_types_same, map(_check_argument_type, args.items()))

    @staticmethod
    def check_value_type(arg_name, arg_value, valid_types, prim_name=None):
        """Checks whether a value is instance of some types."""
        valid_types = valid_types if isinstance(valid_types, Iterable) else (valid_types,)

        def raise_error_msg():
            """func for raising error message when check failed"""
            type_names = [t.__name__ if hasattr(t, '__name__') else str(t) for t in valid_types]
            num_types = len(valid_types)
            msg_prefix = f"For '{prim_name}', the" if prim_name else "The"
            raise TypeError(f'{msg_prefix} type of `{arg_name}` should be {"one of " if num_types > 1 else ""}'
                            f'{type_names if num_types > 1 else type_names[0]}, '
                            f'but got {arg_value} with type {type(arg_value).__name__}.')

        # Notice: bool is subclass of int, so `check_value_type('x', True, [int])` will check fail, and
        #         `check_value_type('x', True, [bool, int])` will check pass
        if isinstance(arg_value, bool) and bool not in tuple(valid_types):
            raise_error_msg()
        if not isinstance(arg_value, tuple(valid_types)):
            raise_error_msg()
        return arg_value

    @staticmethod
    def check_type_name(arg_name, arg_type, valid_types, prim_name):
        """Checks whether a type in some specified types"""
        valid_types = valid_types if isinstance(valid_types, Iterable) else (valid_types,)

        def raise_error_msg():
            """func for raising error message when check failed"""
            type_names = [t.__name__ if hasattr(t, '__name__') else t for t in valid_types]
            num_types = len(valid_types)
            msg_prefix = f"For '{prim_name}', the" if prim_name else "The"
            raise TypeError(f"{msg_prefix} '{arg_name}' should be {'one of ' if num_types > 1 else ''}"
                            f"{type_names if num_types > 1 else type_names[0]}, "
                            f"but got {arg_type.__name__ if hasattr(arg_type, '__name__') else repr(arg_type)}.")

        if isinstance(arg_type, type(mstype.tensor)):
            arg_type = arg_type.element_type()
        if arg_type not in valid_types:
            raise_error_msg()
        return arg_type

    @staticmethod
    def check_reduce_shape(ori_shape, shape, axis, prim_name):
        """Checks whether shape is ori_shape reduced on axis"""
        axis = axis if isinstance(axis, Iterable) else (axis,)
        exp_shape = [ori_shape[i] for i in range(len(ori_shape)) if i not in axis]
        if list(shape) != exp_shape:
            raise ValueError(f'For {prim_name}, {ori_shape} reduce on {axis} should be '
                             f'{tuple(exp_shape)}, but got {shape}.')

    @staticmethod
    def check_astype_dtype(dtype):
        """Check whether dtype is a valid input, and convert to mstype"""
        all_types = mstype.__dtype__ + ["int", "float", "bool"]
        if isinstance(dtype, str):
            if dtype.lower() not in all_types:
                raise TypeError(f"`{dtype}` not understood.")
            dtype = mstype.pytype_to_dtype(np.dtype(dtype.lower()))
        elif isinstance(dtype, type):
            dtype = mstype.pytype_to_dtype(dtype)
        elif not dtype in mstype.number_type + (mstype.bool_,):
            raise TypeError(f"`{dtype}` not understood.")
        return dtype

    @staticmethod
    def check_transpose_axis(axes, ndim):
        """Check the axis argument for tensor.transpose"""
        if not axes or (len(axes) == 1 and axes[0] is None):
            return tuple(range(ndim-1, -1, -1))

        if len(axes) == 1:
            perm = axes[0]
            # if only one argument provided, it must be tuple or list
            if isinstance(perm, list):
                perm = tuple(perm)
            else:
                if not isinstance(perm, tuple):
                    raise TypeError(f"The `axes` should be a tuple/list, or series of int, but got {type(axes[0])}")
            return perm

        # if multiple arguments provided, it must be `ndim` number of ints
        if len(axes) != ndim:
            raise ValueError("The number of axes must equal to the dimension of tensor.")
        return axes

    @staticmethod
    def check_reshape_shp(shp):
        """Check the shape argument for tensor.reshape"""

        if len(shp) == 1:
            new_shape = shp[0]
            # if only one argument provided, it must be int, tuple or list
            if isinstance(new_shape, int):
                return shp
            if isinstance(new_shape, list):
                new_shape = tuple(new_shape)
            else:
                if not isinstance(new_shape, tuple):
                    raise TypeError(
                        f"The `shape` should be an int, or tuple/list, or series of int, but got {type(shp[0])}")
            return new_shape

        return shp

    @staticmethod
    def check_flatten_order(order):
        """Check flatten function input order"""
        if not isinstance(order, str):
            raise TypeError(f"The order variable should be a string, but got {type(order)}")
        if order not in ('C', 'F'):
            raise ValueError(f"only `C` and `F` are supported as order, but got {order}")
        return order

    @staticmethod
    def check_swapaxes_axis(axes, ndim):
        """Check all the axes argument for tensor.swapaxes"""
        if isinstance(axes, int):
            check_axis_in_range(axes, ndim)
            return axes % ndim
        if isinstance(axes, (tuple, list)):
            for axis in axes:
                if not isinstance(axis, int):
                    raise TypeError(f"axis argument should be integer, but got {type(axis)}.")
                check_axis_in_range(axis, ndim)
            axes = tuple(map(lambda x: x % ndim, axes))
            return axes
        raise TypeError(f"axes should be integer, list or tuple for check, but got {type(axes)}.")

    @staticmethod
    def prepare_shape_for_squeeze(shape, axes):
        """
        Creates the squeezed new shape based on the tensor and given axes.

        Args:
            shape (tuple): the shape of the tensor
            axes Union[int, tuple(int), list(int)]: the axes with dimensions need to
                be squeezed.

        Returns:
            new_shape(tuple): the shape with dimensions squeezed.
        """
        new_shape = []
        ndim = len(shape)

        # Convert to set
        if isinstance(axes, int):
            if axes >= ndim or axes < -ndim:
                raise ValueError(f"axis {axes} is out of bounds for tensor of dimension {ndim}")
            axes = {axes}

        elif isinstance(axes, (list, tuple)):
            for axis in axes:
                if axis >= ndim or axis < -ndim:
                    raise ValueError(f"axis {axis} is out of bounds for tensor of dimension {ndim}")
            axes = set(axes)

        else:
            raise TypeError(f"only int, tuple and list are allowed for axes, but got {type(axes)}")

        for idx, s in enumerate(shape):
            if s != 1 or (idx not in axes) and (idx - ndim not in axes):
                new_shape.append(s)
            # if an axis is selected with shape entry greater than one, an error is raised.
            if s != 1 and ((idx in axes) or (idx - ndim in axes)):
                raise ValueError(f"axis {axes} has shape entry {s} > 1, cannot be squeezed.")
        return tuple(new_shape)


def check_input_format(input_param):
    """Judge input format."""
    if input_param == "NCHW":
        return input_param
    raise ValueError("The data format must be NCHW.")


def _expand_tuple(n_dimensions):
    """To expand a int number to tuple."""

    def convert(m):
        if not isinstance(m, tuple):
            if isinstance(m, int) and not isinstance(m, bool):
                return tuple(repeat(m, n_dimensions))
            raise TypeError("Input type must be int or tuple[int].")

        if not len(m) is n_dimensions:
            raise TypeError("Input tuple dimension is incorrect.")

        for i in m:
            if not isinstance(i, int) or isinstance(i, bool):
                raise TypeError("Incorrect type inside of a tuple, must be int!")
        return m

    return convert


def check_axis_in_range(axis, ndim):
    """Checks axes are with the bounds of ndim"""
    if -ndim <= axis < ndim:
        return True
    raise ValueError(f'axis {axis} is out of bounds for tensor of dimension {ndim}')


def _check_data_type_valid(data, valid_type):
    """Check data type valid."""
    if valid_type is None:
        return data is None
    if isinstance(data, valid_type):
        if hasattr(data, 'size') and data.size == 0:
            msg = "Please provide non-empty data."
            logger.error(msg)
            raise ValueError(msg)
        return True
    return False


def check_input_data(*data, data_class):
    """Input data check."""
    for item in data:
        if isinstance(item, (list, tuple)):
            for v in item:
                check_input_data(v, data_class=data_class)
        elif isinstance(item, dict):
            for v in item.values():
                check_input_data(v, data_class=data_class)
        else:
            if isinstance(data_class, (tuple, list)):
                ret = True in tuple(_check_data_type_valid(item, data_type) for data_type in data_class)
            else:
                ret = _check_data_type_valid(item, data_class)
            if not ret:
                data_class_str = tuple(i.__name__ if hasattr(i, '__name__') else i for i in data_class) \
                                 if isinstance(data_class, (tuple, list)) else \
                                 (data_class if data_class is None else data_class.__name__)
                raise ValueError(f'Please provide as model inputs either a single or '
                                 f'a tuple or a list or a dict of {data_class_str}, '
                                 f'but got part data type is {item if item is None else type(item).__name__}.')


def check_output_data(data):
    """Output data check."""
    if data is None:
        raise RuntimeError('Executor return data ' + str(data) + ', please check your net or input data.')


once = _expand_tuple(1)
twice = _expand_tuple(2)
triple = _expand_tuple(3)


def args_type_check(*type_args, **type_kwargs):
    """Check whether input data type is correct."""

    def type_check(func):
        sig = inspect.signature(func)
        bound_types = sig.bind_partial(*type_args, **type_kwargs).arguments

        @wraps(func)
        def wrapper(*args, **kwargs):
            nonlocal bound_types
            bound_values = sig.bind(*args, **kwargs)
            argument_dict = bound_values.arguments
            if "kwargs" in bound_types:
                bound_types = bound_types["kwargs"]
            if "kwargs" in argument_dict:
                argument_dict = argument_dict["kwargs"]
            for name, value in argument_dict.items():
                if name in bound_types:
                    if value is not None and not isinstance(value, bound_types[name]):
                        raise TypeError('Argument {} must be {}'.format(name, bound_types[name]))
            return func(*args, **kwargs)

        return wrapper

    return type_check
