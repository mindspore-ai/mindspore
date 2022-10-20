# Copyright 2019-2022 Huawei Technologies Co., Ltd
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
"""Validators for data processing operations.
"""
from functools import wraps
import inspect
import numpy as np

from mindspore._c_expression import typing
from ..core.validator_helpers import parse_user_args, type_check, check_pos_int64, check_value, check_positive, \
    check_tensor_op, type_check_list, deprecator_factory

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


def check_fill_value(method):
    """Wrapper method to check the parameters of fill_value."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        [fill_value], _ = parse_user_args(method, *args, **kwargs)
        type_check(fill_value, (str, float, bool, int, bytes), "fill_value")

        return method(self, *args, **kwargs)

    return new_method


def check_one_hot_op(method):
    """Wrapper method to check the parameters of one_hot_op."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        [num_classes, smoothing_rate], _ = parse_user_args(method, *args, **kwargs)
        type_check(smoothing_rate, (int, float), "smoothing_rate")
        type_check(num_classes, (int,), "num_classes")
        check_positive(num_classes)

        if smoothing_rate is not None:
            check_value(smoothing_rate, [0., 1.], "smoothing_rate")

        return method(self, *args, **kwargs)

    return new_method


def check_num_classes(method):
    """Wrapper method to check the parameters of number of classes."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        [num_classes], _ = parse_user_args(method, *args, **kwargs)

        type_check(num_classes, (int,), "num_classes")
        check_positive(num_classes)

        return method(self, *args, **kwargs)

    return new_method


def check_ms_type(method):
    """Wrapper method to check the parameters of data type."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        [data_type], _ = parse_user_args(method, *args, **kwargs)

        type_check(data_type, (typing.Type,), "data_type")

        return method(self, *args, **kwargs)

    return new_method


def check_slice_option(method):
    """Wrapper method to check the parameters of SliceOption."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        [slice_option], _ = parse_user_args(method, *args, **kwargs)
        from .transforms import _SliceOption
        if slice_option is not None:
            type_check(slice_option, (int, list, slice, bool, type(Ellipsis), _SliceOption), "slice_option")

            if isinstance(slice_option, list):
                type_check_list(slice_option, (int,), "slice_option")

        return method(self, *args, **kwargs)

    return new_method


def check_slice_op(method):
    """Wrapper method to check the parameters of slice."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        [slice_op], _ = parse_user_args(method, *args, **kwargs)

        for s in slice_op:
            from .transforms import _SliceOption
            if s is not None:
                type_check(s, (int, list, slice, bool, type(Ellipsis), _SliceOption), "slice")
                if isinstance(s, list) and s:
                    if isinstance(s[0], int):
                        type_check_list(s, (int,), "slice")

        return method(self, *args, **kwargs)

    return new_method


def check_mask_op(method):
    """Wrapper method to check the parameters of mask."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        [operator, constant, dtype], _ = parse_user_args(method, *args, **kwargs)

        from .c_transforms import Relational
        type_check(operator, (Relational,), "operator")
        type_check(constant, (str, float, bool, int, bytes), "constant")
        type_check(dtype, (typing.Type,), "dtype")

        return method(self, *args, **kwargs)

    return new_method


def check_mask_op_new(method):
    """Wrapper method to check the parameters of mask."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        [operator, constant, dtype], _ = parse_user_args(method, *args, **kwargs)

        from .transforms import Relational
        type_check(operator, (Relational,), "operator")
        type_check(constant, (str, float, bool, int, bytes), "constant")
        type_check(dtype, (typing.Type,), "dtype")

        return method(self, *args, **kwargs)

    return new_method


def check_pad_end(method):
    """Wrapper method to check the parameters of PadEnd."""

    @wraps(method)
    def new_method(self, *args, **kwargs):

        [pad_shape, pad_value], _ = parse_user_args(method, *args, **kwargs)

        if pad_value is not None:
            type_check(pad_value, (str, float, bool, int, bytes), "pad_value")
        type_check(pad_shape, (list,), "pad_shape")

        for dim in pad_shape:
            if dim is not None:
                if isinstance(dim, int):
                    check_pos_int64(dim)
                else:
                    raise TypeError("a value in the list is not an integer.")

        return method(self, *args, **kwargs)

    return new_method


def check_concat_type(method):
    """Wrapper method to check the parameters of concatenation op."""

    @wraps(method)
    def new_method(self, *args, **kwargs):

        [axis, prepend, append], _ = parse_user_args(method, *args, **kwargs)

        if axis is not None:
            type_check(axis, (int,), "axis")
            if axis not in (0, -1):
                raise ValueError("only 1D concatenation supported.")

        if prepend is not None:
            type_check(prepend, (np.ndarray,), "prepend")
            if len(prepend.shape) != 1:
                raise ValueError("can only prepend 1D arrays.")

        if append is not None:
            type_check(append, (np.ndarray,), "append")
            if len(append.shape) != 1:
                raise ValueError("can only append 1D arrays.")

        return method(self, *args, **kwargs)

    return new_method


def check_random_transform_ops(method):
    """Wrapper method to check the parameters of RandomChoice, RandomApply and Compose."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        arg_list, _ = parse_user_args(method, *args, **kwargs)
        type_check(arg_list[0], (list,), "transforms list")
        if not arg_list[0]:
            raise ValueError("transforms list can not be empty.")
        for ind, op in enumerate(arg_list[0]):
            check_tensor_op(op, "transforms[{0}]".format(ind))
            check_transform_op_type(ind, op)
        if len(arg_list) == 2:  # random apply takes an additional arg
            type_check(arg_list[1], (float, int), "prob")
            check_value(arg_list[1], (0, 1), "prob")
        return method(self, *args, **kwargs)

    return new_method


def check_transform_op_type(ind, op):
    """Check the operation."""
    # c_vision.HWC2CHW error
    # py_vision.HWC2CHW error
    if type(op) == type:  # pylint: disable=unidiomatic-typecheck
        raise ValueError("op_list[{}] should be a dataset processing operation instance, "
                         "but got: {}. It may be missing parentheses for instantiation.".format(ind, op))


def check_compose_list(method):
    """Wrapper method to check the transform list of Python Compose."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        [transforms], _ = parse_user_args(method, *args, **kwargs)

        type_check(transforms, (list,), transforms)
        if not transforms:
            raise ValueError("transforms list is empty.")
        for i, transform in enumerate(transforms):
            if not callable(transform):
                raise ValueError("transforms[{}] is not callable.".format(i))
            check_transform_op_type(i, transform)
        return method(self, *args, **kwargs)

    return new_method


def check_compose_call(method):
    """Wrapper method to check the transform list of Compose."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        sig = inspect.signature(method)
        ba = sig.bind_partial(method, *args, **kwargs)
        img = ba.arguments.get("args")
        if img is None:
            raise TypeError(
                "Compose was called without an image. Fix invocation (avoid it being invoked as Compose([...])()).")
        return method(self, *args, **kwargs)

    return new_method


def check_random_apply(method):
    """Wrapper method to check the parameters of random apply."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        [transforms, prob], _ = parse_user_args(method, *args, **kwargs)
        type_check(transforms, (list,), "transforms")

        for i, transform in enumerate(transforms):
            if str(transform).find("c_transform") >= 0:
                raise ValueError(
                    "transforms[{}] is not a py transforms. Should not use a c transform in py transform" \
                        .format(i))
            check_transform_op_type(i, transform)

        if prob is not None:
            type_check(prob, (float, int,), "prob")
            check_value(prob, [0., 1.], "prob")

        return method(self, *args, **kwargs)

    return new_method


def check_transforms_list(method):
    """Wrapper method to check the parameters of transform list."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        [transforms], _ = parse_user_args(method, *args, **kwargs)

        type_check(transforms, (list,), "transforms")
        for i, transform in enumerate(transforms):
            if str(transform).find("c_transform") >= 0:
                raise ValueError(
                    "transforms[{}] is not a py transforms. Should not use a c transform in py transform" \
                        .format(i))
            check_transform_op_type(i, transform)
        return method(self, *args, **kwargs)

    return new_method


def check_plugin(method):
    """Wrapper method to check the parameters of plugin."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        [lib_path, func_name, user_args], _ = parse_user_args(method, *args, **kwargs)

        type_check(lib_path, (str,), "lib_path")
        type_check(func_name, (str,), "func_name")
        if user_args is not None:
            type_check(user_args, (str,), "user_args")

        return method(self, *args, **kwargs)

    return new_method


def invalidate_callable(method):
    """Wrapper method to invalidate cached callable_op_ used in eager mode. \
    This decorator must be added to any method which modifies the state of transform."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        self.callable_op_ = None
        return method(self, *args, **kwargs)

    return new_method


def check_type_cast(method):
    """Wrapper method to check the parameters of TypeCast."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        [data_type], _ = parse_user_args(method, *args, **kwargs)

        # Check if data_type is mindspore.dtype
        if isinstance(data_type, (typing.Type,)):
            return method(self, *args, **kwargs)

        # Special case: Check if data_type is None (which is invalid)
        if data_type is None:
            # Use type_check to raise error with descriptive error message
            type_check(data_type, (typing.Type, np.dtype,), "data_type")

        try:
            # Check if data_type can be converted to numpy type
            _ = np.dtype(data_type)
        except (TypeError, ValueError):
            # Use type_check to raise error with descriptive error message
            type_check(data_type, (typing.Type, np.dtype,), "data_type")

        return method(self, *args, **kwargs)

    return new_method


def deprecated_c_transforms(substitute_name=None, substitute_module=None):
    """Decorator for version 1.8 deprecation warning for legacy mindspore.dataset.transforms.c_transforms operation.

    Args:
        substitute_name (str, optional): The substitute name for deprecated operation.
        substitute_module (str, optional): The substitute module for deprecated operation.
    """
    return deprecator_factory("1.8", "mindspore.dataset.transforms.c_transforms", "mindspore.dataset.transforms",
                              substitute_name, substitute_module)


def deprecated_py_transforms(substitute_name=None, substitute_module=None):
    """Decorator for version 1.8 deprecation warning for legacy mindspore.dataset.transforms.py_transforms operation.

    Args:
        substitute_name (str, optional): The substitute name for deprecated operation.
        substitute_module (str, optional): The substitute module for deprecated operation.
    """
    return deprecator_factory("1.8", "mindspore.dataset.transforms.py_transforms", "mindspore.dataset.transforms",
                              substitute_name, substitute_module)
