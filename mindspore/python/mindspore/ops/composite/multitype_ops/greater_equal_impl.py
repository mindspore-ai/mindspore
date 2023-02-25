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

"""Implementation for internal polymorphism `greater_equal` operations."""
from mindspore.ops.composite import base
from mindspore.ops import functional as F
from mindspore.ops.operations import _inner_ops as inner
from ...operations._sequence_ops import tuple_greater_equal, list_greater_equal

# greater_equal is a metagraph object which will determine if two objects are greater_equal according to input type
# using ".register" decorator
greater_equal = base.MultitypeFuncGraph("greater_equal", True)


@greater_equal.register("Number", "Number")
def _greater_equal_scala(x, y):
    """
    Determine whether x is greater than or equal to y.

    Args:
       x(Number): Number.
       y(Number): Number.

    Returns:
       bool, if x >= y return true, x < y return false.
   """
    return F.scalar_ge(x, y)


@greater_equal.register("String", "String")
def _greater_equal_string(x, y):
    """
    Determine whether x is greater than or equal to y.

    Args:
       x(String): String.
       y(String): String.

    Returns:
       bool, if x >= y return true, x < y return false.
   """
    return inner.string_ge(x, y)


@greater_equal.register("Tensor", "Number")
@greater_equal.register("Number", "Tensor")
@greater_equal.register("Tensor", "Tensor")
def _greater_equal_tensor(x, y):
    """
    Determine whether x is greater than or equal to y elementwise.

    Args:
       x(Tensor): Tensor.
       y(Tensor): Tensor.

    Returns:
       Tensor, return value by operator P.GreaterEqual.
   """
    return F.tensor_ge(x, y)


@greater_equal.register("Tuple", "Tuple")
def _greater_equal_tuple(x, y):
    """
    Determine whether x is greater than or equal to y.

    Args:
       x(Tuple): Tuple.
       y(Tuple): Tuple.

    Returns:
       bool, if x >= y return true in python logic, x < y return false.
   """
    return tuple_greater_equal()(x, y)


@greater_equal.register("List", "List")
def _greater_equal_list(x, y):
    """
    Determine whether x is greater than or equal to y.

    Args:
       x(List): List.
       y(List): List.

    Returns:
       bool, if x >= y return true in python logic, x < y return false.
   """
    return list_greater_equal()(x, y)
