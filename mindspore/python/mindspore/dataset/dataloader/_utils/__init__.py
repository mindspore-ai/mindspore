# Copyright 2025 Huawei Technologies Co., Ltd
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
"""Init for DataLoader utilities."""

import inspect
from functools import wraps

WORKER_TIME_OUT = 5


def check_args(method):
    """Validate the arguments of a class method."""
    sig = inspect.signature(method)

    @wraps(method)
    def wrapper(*args, **kwargs):
        bound = sig.bind_partial(*args, **kwargs)
        bound.apply_defaults()

        params = dict(bound.arguments)
        class_handle = None
        if args:
            first_arg = next(iter(sig.parameters.keys()))
            if first_arg in ("self", "cls"):
                class_handle = args[0]
                params.pop(first_arg, None)

        if hasattr(class_handle, "_check_args"):
            class_handle._check_args(params)  # pylint: disable=protected-access
        else:
            raise AttributeError(f"{class_handle.__class__.__name__} should implement _check_args method.")

        return method(*args, **kwargs)

    return wrapper


def check_type(value, name, valid_type, invalid_type=(), allow_none=False):
    """Check if the value is of the valid type."""
    if allow_none and value is None:
        return
    if not isinstance(value, valid_type) or isinstance(value, invalid_type):
        raise TypeError(f"{name} must be {valid_type}, but got: {type(value).__name__}.")


def check_positive(value, name, allow_none=False):
    """Check if the value is positive."""
    if allow_none and value is None:
        return
    try:
        if value <= 0:
            raise ValueError(f"{name} must be positive, but got: {value}.")
    except TypeError as exc:
        raise TypeError(f"{name} must be a number, but got: {type(value).__name__}.") from exc


def check_non_negative(value, name, allow_none=False):
    """Check if the value is non-negative."""
    if allow_none and value is None:
        return
    try:
        if value < 0:
            raise ValueError(f"{name} must be non-negative, but got: {value}.")
    except TypeError as exc:
        raise TypeError(f"{name} must be a number, but got: {type(value).__name__}.") from exc


def check_exclusive_args(condition, arg_name, description):
    """Check if the arguments are exclusive."""
    if condition:
        raise ValueError(f"{arg_name} cannot be specified {description}.")
