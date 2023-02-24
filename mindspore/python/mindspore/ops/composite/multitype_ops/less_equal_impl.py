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
# ============================================================================
"""Implementation for internal polymorphism `less_equal` operations."""
from __future__ import absolute_import
from __future__ import division

from mindspore.ops.composite import base
from mindspore.ops import functional as F
from mindspore.ops.operations import _inner_ops as inner
from mindspore.ops.operations import _sequence_ops as _seq

# less_equal is a metagraph object which will determine if two objects are less_equal according to input type
# using ".register" decorator
less_equal = base.MultitypeFuncGraph("less_equal", True)


@less_equal.register("Number", "Number")
def _less_equal_scala(x, y):
    """
    Determine whether x is less than or equal to y.

    Args:
       x(Number): Number.
       y(Number): Number.

    Returns:
       bool, if x <= y return true, x > y return false.
   """
    return F.scalar_le(x, y)


@less_equal.register("String", "String")
def _less_equal_string(x, y):
    """
    Determine whether x is less than or equal to y.

    Args:
       x(String): String.
       y(String): String.

    Returns:
       bool, if x <= y return true, x > y return false.
   """
    return inner.string_le(x, y)


@less_equal.register("Tensor", "Number")
@less_equal.register("Number", "Tensor")
@less_equal.register("Tensor", "Tensor")
def _less_equal_tensor(x, y):
    """
    Determine  whether x is less equal than y elementwise.

    Args:
       x(Tensor): Tensor.
       y(Tensor): Tensor.

    Returns:
       Tensor, return value by operator P.LessEqual.
   """
    return F.tensor_le(x, y)


@less_equal.register("Tuple", "Tuple")
def _less_equal_tuple(x, y):
    """
    Determine whether x is less than or equal to y.

    Args:
       x(Tuple): Tuple.
       y(Tuple): Tuple.

    Returns:
       bool, if x <= y return true in python logic, x > y return false.
   """
    return _seq.tuple_le()(x, y)


@less_equal.register("List", "List")
def _less_equal_list(x, y):
    """
    Determine whether x is less than or equal to y.

    Args:
       x(List): List.
       y(List): List.

    Returns:
       bool, if x <= y return true in python logic, x > y return false.
   """
    return _seq.list_le()(x, y)
