# Copyright 2020 Huawei Technologies Co., Ltd
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

"""Implementation for internal polymorphism `not equal` operations."""

from . import _constexpr_utils as const_utils
from ...composite import base
from ... import functional as F


not_equal = base.MultitypeFuncGraph("not_equal", True)
"""
not_equal is a metafuncgraph object which will determine if two objects are not_equal according to input type
using ".register" decorator
"""


@not_equal.register("Number", "Number")
def _not_equal_scalar(x, y):
    """
    Determine if two numbers is not equal.

    Args:
       x (Number): x
       y (Number): y

    Returns:
       bool, if x != y return true, x == y return false.
   """
    return not F.scalar_eq(x, y)


@not_equal.register("mstype", "mstype")
def _not_equal_mstype(x, y):
    """
    Determine if two mindspore types are not equal.

    Args:
       x (mstype): first input mindspore type.
       y (mstype): second input mindspore type.

    Returns:
       bool, if x != y return true, x == y return false.
   """
    return not const_utils.mstype_eq(x, y)


@not_equal.register("String", "String")
def _not_equal_string(x, y):
    """
    Determine if two strings are not equal.

    Args:
       x: str
       y: str

    Returns:
       bool, if x != y return true, x == y return false.
   """
    return not F.string_eq(x, y)


@not_equal.register("String", "None")
def _string_not_equal_none(x, y):
    """
    Determine if string not equals none.

    Args:
       x: str.
       y: None.

    Returns:
       bool, return True.
   """
    return True


@not_equal.register("None", "String")
def _none_not_equal_string(x, y):
    """
    Determine if string not equals none.

    Args:
       x: None.
       y: str.

    Returns:
       bool, return True.
   """
    return True


@not_equal.register("None", "None")
def _none_not_equal_none(x, y):
    """
    Determine if none not equals none.

    Args:
       x: None.
       y: None.

    Returns:
       bool, return False.
   """
    return False


@not_equal.register("Number", "None")
def _scalar_not_equal_none(x, y):
    """
    Determine if number not equals none.

    Args:
       x: Number.
       y: None.

    Returns:
       bool, return True.
   """
    return True


@not_equal.register("None", "Number")
def _none_not_equal_scalar(x, y):
    """
    Determine if number not_equals none.

    Args:
       x: None.
       y: Number.

    Returns:
       bool, return True.
   """
    return True


@not_equal.register("Tuple", "Tuple")
def _not_euqal_tuple(x, y):
    """
    Determine if two tuples are not equal by element.

    Args:
       x (tuple): x
       y (tuple): y

    Returns:
       bool, if x and y are not equal by element return true, else return false.
   """
    return not F.tuple_equal(x, y)


@not_equal.register("List", "List")
def _not_euqal_list(x, y):
    """
    Determine if two lists are not equal by element.

    Args:
       x (list): x
       y (list): y

    Returns:
       bool, if x and y are not equal by element return true, else return false.
   """
    return not F.list_equal(x, y)


@not_equal.register("Tuple", "None")
def _tuple_not_euqal_none(x, y):
    """
    Determine if tuple element not equals none element.

    Args:
       x: Tuple.
       y: None.

    Returns:
       bool, return True.
   """
    return True


@not_equal.register("None", "Tuple")
def _none_not_equal_tuple(x, y):
    """
    Determine if tuple element not equals none element.

    Args:
       x: None.
       y: Tuple.

    Returns:
       bool, return True.
   """
    return True


@not_equal.register("Tensor", "Number")
@not_equal.register("Number", "Tensor")
@not_equal.register("Tensor", "Tensor")
def _tensor_not_equal_tensor(x, y):
    """
    Determine if two tensors are not_equal.

    Args:
       x : Tensor.
       y : Tensor.

    Returns:
       bool, if x == y return true, x != y return false.
   """
    return F.not_equal(x, y)


@not_equal.register("Tensor", "None")
def _tensor_not_equal_none(x, y):
    """
    Determine if tensor not_equal none.

    Args:
       x : Tensor.
       y : None.

    Returns:
       bool, return True.
   """
    return True


@not_equal.register("None", "Tensor")
def _none_not_equal_tensor(x, y):
    """
    Determine if tensor not equal none.

    Args:
       x : None.
       y : Tensor.

    Returns:
       bool, return True.
   """
    return True


@not_equal.register("List", "None")
def _list_not_equal_none(x, y):
    """
    Determine if list not equal none.

    Args:
       x (list): The first input which is a list.
       y (none): The second input which is none.

    Returns:
       bool, return true.
   """
    return True


@not_equal.register("None", "List")
def _none_not_equal_list(x, y):
    """
    Determine if none not equal list.

    Args:
       x (none): The first input which is none.
       y (list): The second input which is a list.

    Returns:
       bool, return true.
   """
    return True
