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

"""Implementation for internal polymorphism `not in` operations."""

from . import _constexpr_utils as const_utils
from . import _compile_utils as compile_utils
from ... import functional as F
from ...composite import base

not_in_ = base.MultitypeFuncGraph("not_in", True)
"""
"not_in_" is a multi type func graph object which will determine if a not in b.
using ".register" decorator
"""


@not_in_.register("Number", "Tuple")
def _number_not_in_tuple(x, y):
    """
    Determine if a number not in tuple.

    Args:
       x (Number): x
       y (tuple): y

    Returns:
       bool, if x not in y return true, x in y return false.
   """
    return not const_utils.scalar_in_sequence(x, y)


@not_in_.register("Number", "List")
def _number_not_in_list(x, y):
    """
    Determine if a number not in list.

    Args:
       x (Number): x
       y (list): y

    Returns:
       bool, if x not in y return true, x in y return false.
   """
    return not const_utils.scalar_in_sequence(x, y)


@not_in_.register("String", "Tuple")
def _string_not_in_tuple(x, y):
    """
    Determine if a str not in a tuple.

    Args:
       x (str): x
       y (tuple): y

    Returns:
       bool, if x not in y return true, x in y return false.
   """
    return not const_utils.scalar_in_sequence(x, y)


@not_in_.register("String", "List")
def _string_not_in_list(x, y):
    """
    Determine if a str not in a list.

    Args:
       x (str): x
       y (list): y

    Returns:
       bool, if x not in y return true, x in y return false.
   """
    return not const_utils.scalar_in_sequence(x, y)


@not_in_.register("String", "Dictionary")
def _str_not_in_dict(x, y):
    """
    Determine if a str not in dict.

    Args:
       x: str
       y: dict

    Returns:
       bool, if x not in y return true, x in y return false.
   """
    return F.not_in_dict(x, y)


@not_in_.register("Tensor", "List")
def _tensor_not_in_list(x, y):
    """
    Determine if a tensor not in a list.

    Args:
       x: Tensor
       y: List

    Returns:
       bool, if x not in y return true, x in y return false.
   """
    return not compile_utils.tensor_in_sequence(x, y)


@not_in_.register("Tensor", "Tuple")
def _tensor_not_in_tuple(x, y):
    """
    Determine if a tensor not in a tuple.

    Args:
       x: Tensor
       y: Tuple

    Returns:
       bool, if x not in y return true, x in y return false.
   """
    return not compile_utils.tensor_in_sequence(x, y)
