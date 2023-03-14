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

"""Implementation for internal polymorphism `in` operations."""
from __future__ import absolute_import
from mindspore.ops.operations import _inner_ops as inner
from mindspore.ops.composite.multitype_ops import _constexpr_utils as const_utils
from mindspore.ops.composite.multitype_ops import _compile_utils as compile_utils
from mindspore.ops import functional as F
from mindspore.ops.composite import base
from ...operations._sequence_ops import InSequence

in_ = base.MultitypeFuncGraph("in", True)
"""
"in_" is a multi type func graph  object which will determine if a in b
using ".register" decorator
"""


@in_.register("Number", "Tuple")
def _number_in_tuple(x, y):
    """
    Determine if a number in tuple.

    Args:
       x (Number): x
       y (tuple): y

    Returns:
       bool, if x in y return true, x not in y return false.
   """
    if F.is_sequence_shape_unknown(y):
        return InSequence()(x, y)
    return const_utils.scalar_in_sequence(x, y)


@in_.register("Number", "List")
def _number_in_list(x, y):
    """
    Determine if a number in list.

    Args:
       x (Number): x
       y (list): y

    Returns:
       bool, if x in y return true, x not in y return false.
   """
    if F.is_sequence_shape_unknown(y):
        return InSequence()(x, y)
    return const_utils.scalar_in_sequence(x, y)


@in_.register("String", "String")
def _string_in_string(x, y):
    """
    Determine if a str in another str.

    Args:
       x (str): x
       y (str): y

    Returns:
       bool, if x in y return true, x not in y return false.
   """
    return inner.string_in(x, y)


@in_.register("String", "Tuple")
def _string_in_tuple(x, y):
    """
    Determine if a str in a tuple.

    Args:
       x (str): x
       y (tuple): y

    Returns:
       bool, if x in y return true, x not in y return false.
   """
    return const_utils.scalar_in_sequence(x, y)


@in_.register("String", "List")
def _string_in_list(x, y):
    """
    Determine if a str in a list.

    Args:
       x (str): x
       y (list): y

    Returns:
       bool, if x in y return true, x not in y return false.
   """
    return const_utils.scalar_in_sequence(x, y)


@in_.register("Tensor", "Dictionary")
@in_.register("Tuple", "Dictionary")
@in_.register("Number", "Dictionary")
@in_.register("String", "Dictionary")
def _str_in_dict(x, y):
    """
    Determine if an element is in dict.

    Args:
       x: Tensor, Tuple, Number, String
       y: dict

    Returns:
       bool, if x in y return true, x not in y return false.
   """
    return F.in_dict(x, y)


@in_.register("Tensor", "List")
def _tensor_in_list(x, y):
    """
    Determine if a tensor in a list.

    Args:
       x: Tensor
       y: List

    Returns:
       bool, if x in y return true, x not in y return false.
   """
    if F.is_sequence_shape_unknown(y):
        return InSequence()(x, y)
    return compile_utils.tensor_in_sequence(x, y)


@in_.register("Tensor", "Tuple")
def _tensor_in_tuple(x, y):
    """
    Determine if a tensor in a tuple.

    Args:
       x: Tensor
       y: Tuple

    Returns:
       bool, if x in y return true, x not in y return false.
   """
    if F.is_sequence_shape_unknown(y):
        return InSequence()(x, y)
    return compile_utils.tensor_in_sequence(x, y)


@in_.register("mstype", "List")
def _mstype_in_list(x, y):
    """
    Determine if a mindspore type is in a list.

    Args:
       x: mstype
       y: List

    Returns:
       bool, if x in y return true, x not in y return false.
   """
    return const_utils.check_in_sequence(x, y)


@in_.register("mstype", "Tuple")
def _mstype_in_tuple(x, y):
    """
    Determine if a mindspore type is in a tuple.

    Args:
       x: mstype
       y: Tuple

    Returns:
       bool, if x in y return true, x not in y return false.
   """
    return const_utils.check_in_sequence(x, y)
