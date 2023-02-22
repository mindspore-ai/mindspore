# Copyright 2020-2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Ungreater required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

"""Implementation for internal polymorphism `greater` operations."""

from mindspore.ops.composite import base
from mindspore.ops import functional as F
from mindspore.ops.operations import _inner_ops as inner
from ...operations._sequence_ops import tuple_greater_than, list_greater_than

# greater is a metafuncgraph object which will determine if two objects are greater according to input type
# using ".register" decorator
greater = base.MultitypeFuncGraph("greater", True)


@greater.register("Number", "Number")
def _greater_scalar(x, y):
    """
    Determine whether x is greater than y.

    Args:
       x(Number): Number.
       y(Number): Number.

    Returns:
       bool, if x > y return true, x <= y return false.
   """
    return F.scalar_gt(x, y)


@greater.register("String", "String")
def _greater_string(x, y):
    """
    Determine whether x is greater than y.

    Args:
       x(String): String.
       y(String): String.

    Returns:
       bool, if x > y return true, x <= y return false.
   """
    return inner.string_gt(x, y)


@greater.register("Tensor", "Number")
@greater.register("Number", "Tensor")
@greater.register("Tensor", "Tensor")
def _greater_tensor(x, y):
    """
    Determine whether x is greater than y elementwise.

    Args:
       x(Tensor): Tensor.
       y(Tensor): Tensor.

    Returns:
       tensor, return operation of x and y by P.Greater.
   """
    return F.tensor_gt(x, y)


@greater.register("Tuple", "Tuple")
def _greater_than_tuple(x, y):
    """
    Determine whether x is greater than y.

    Args:
       x(Tuple): Tuple.
       y(Tuple): Tuple.

    Returns:
       bool, if x > y return true in python logic, x <= y return false.
   """
    return tuple_greater_than()(x, y)


@greater.register("List", "List")
def _greater_than_list(x, y):
    """
    Determine whether x is greater than y.

    Args:
       x(List): List.
       y(List): List.

    Returns:
       bool, if x > y return true in python logic, x <= y return false.
   """
    return list_greater_than()(x, y)
