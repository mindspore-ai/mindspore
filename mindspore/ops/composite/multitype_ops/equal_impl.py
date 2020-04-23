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

"""equal_impl"""

from ...composite import base
from ... import functional as F


equal = base.MultitypeFuncGraph("equal")
"""
equal is a metafuncgraph object which will determine if two objects are equal according to input type
using ".register" decorator
"""


@equal.register("Number", "Number")
def _equal_scalar(x, y):
    """
    Determine if two numbers are equal.

    Args:
       x (Number): x
       y (NUmber): y

    Returns:
       bool, if x == y return true, x != y return false.
   """
    return F.scalar_eq(x, y)


@equal.register("String", "String")
def _equal_string(x, y):
    """
    Determine if two strings are equal.

    Args:
       x: str
       y: str

    Returns:
       bool, if x == y return true, x != y return false.
   """
    return F.string_eq(x, y)


@equal.register("String", "None")
def _string_equal_none(x, y):
    """
    Determine if string equals none.

    Args:
       x: str.
       y: None.

    Returns:
       bool, return false.
   """
    return False


@equal.register("None", "String")
def _none_equal_string(x, y):
    """
    Determine if string equals none.

    Args:
       x: None.
       y: str.

    Returns:
       bool, return false.
   """
    return False


@equal.register("None", "None")
def _none_equal_none(x, y):
    """
    Determine if none equals none.

    Args:
       x: None.
       y: None.

    Returns:
       bool, return true.
   """
    return True


@equal.register("Number", "None")
def _scalar_equal_none(x, y):
    """
    Determine if number equals none.

    Args:
       x: Number.
       y: None.

    Returns:
       bool, return false.
   """
    return False


@equal.register("None", "Number")
def _none_equal_scalar(x, y):
    """
    Determine if number equals none.

    Args:
       x: None.
       y: NUmber.

    Returns:
       bool, return false.
   """
    return False


@equal.register("Tuple", "Tuple")
def _euqal_tuple(x, y):
    """
    Determine if two tuples are equal by element.

    Args:
       x (tuple): x
       y (tuple): y

    Returns:
       bool, if x and y are equal by element return true, else return false.
   """
    return F.tuple_equal(x, y)


@equal.register("List", "List")
def _euqal_list(x, y):
    """
    Determine if two lists are equal by element.

    Args:
       x (list): x
       y (list): y

    Returns:
       bool, if x and y are equal by element return true, else return false.
   """
    return F.list_equal(x, y)


@equal.register("Tuple", "None")
def _tuple_euqal_none(x, y):
    """
    Determine if tuple element equals none element.

    Args:
       x: Tuple.
       y: None.

    Returns:
       bool, return false.
   """
    return False


@equal.register("None", "Tuple")
def _none_equal_tuple(x, y):
    """
    Determine if tuple element equals none element.

    Args:
       x: None.
       y: Tuple.

    Returns:
       bool, return false.
   """
    return False


@equal.register("Tensor", "Number")
@equal.register("Number", "Tensor")
@equal.register("Tensor", "Tensor")
def _tensor_equal_tensor(x, y):
    """
    Determine if two tensors are equal.

    Args:
       x : Tensor.
       y : Tensor.

    Returns:
       bool, if x == y return true, x != y return false.
   """
    return F.equal(x, y)


@equal.register("Tensor", "None")
def _tensor_equal_none(x, y):
    """
    Determine if tensor equal none.

    Args:
       x : Tensor.
       y : None.

    Returns:
       bool, return false.
   """
    return False


@equal.register("None", "Tensor")
def _none_equal_tensor(x, y):
    """
    Determine if tensor equal none.

    Args:
       x : None.
       y : Tensor.

    Returns:
       bool, return false.
   """
    return False


@equal.register("List", "None")
def _list_equal_none(x, y):
    """
    Determine if list equal none.

    Args:
       x (list): The first input which is a list.
       y (none): The second input which is none.

    Returns:
       bool, return false.
   """
    return False


@equal.register("None", "List")
def _none_equal_list(x, y):
    """
    Determine if none equal list.

    Args:
       x (none): The first input which is none.
       y (list): The second input which is a list.

    Returns:
       bool, return false.
   """
    return False
