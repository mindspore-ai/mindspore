# Copyright 2023 Huawei Technologies Co., Ltd
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
"""test decorator."""

import mindspore.nn as nn
import inspect
from mindspore.rewrite import SymbolTree
from functools import wraps

try:
    from mindspore._checkparam import Validator
except ImportError:
    import mindspore._checkparam as Validator


def my_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        func(*args, **kwargs)

    return wrapper


def _args_type_validator_check(*type_args, **type_kwargs):
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
                    bound_types[name](value, name)
            return func(*args, **kwargs)

        return wrapper

    return type_check


def register_denied_func_decorators(fn):
    """user deny certain decorators"""
    from mindspore.rewrite.parsers.class_def_parser import ClassDefParser
    name = "denied_function_decorator_list"
    setattr(ClassDefParser, name, fn)


class MyNet(nn.Cell):
    @my_decorator
    @_args_type_validator_check(in_channels=Validator.check_positive_int)
    def __init__(self, in_channels):
        super(MyNet, self).__init__()
        self.conv = nn.Conv2d(16, 16, 3)
        self.dense = nn.Dense(in_channels=in_channels, out_channels=32, weight_init="ones")
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.relu3 = nn.ReLU()

    def construct(self, x):
        x = self.conv(x)
        x = self.dense(x)
        x = self.relu1(x)
        x = self.relu2(x)
        x = self.relu3(x)
        return x


def test_decorator():
    """
    Feature: parse decorators
    Description: parse decorators of function which are allowed according to users.
    Expectation: Success.
    """
    # the decorator "_args_type_validator_check" is denied
    register_denied_func_decorators(["_args_type_validator_check"])
    net = MyNet(32)
    stree = SymbolTree.create(net)
    codes = stree.get_code()

    # @my_decorator is allowed
    assert codes.count("@my_decorator") == 1

    # @_args_type_validator_check is denied
    assert codes.count("@_args_type_validator_check") == 0
