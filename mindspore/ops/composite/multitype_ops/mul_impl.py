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

"""Implementation for internal polymorphism `mul` operations."""

from . import _constexpr_utils as const_utils
from . import _compile_utils as utils
from ...composite import base
from ... import functional as F


mul = base.MultitypeFuncGraph("mul", True)
"""
`mul` is a metafuncgraph object which will multiply two objects according to input type
using ".register" decorator.
"""


@mul.register("Number", "Number")
def _mul_scalar(x, y):
    """
    Returns x * y where x and y are all scalars.

    Outputs:
       Number, equal to x * y, the type is same as x.
   """
    return F.scalar_mul(x, y)


@mul.register("Tensor", "Tensor")
def _mul_tensor(x, y):
    """
    Returns x * y by element-wise where x and y are all tensors.

    Outputs:
        Tensor, has the same dtype as x.
    """
    return F.tensor_mul(x, y)


@mul.register("Number", "Tensor")
def _scalar_mul_tensor(x, y):
    """
    Returns x * y where x is a scalar and y is a tensor. x and y have same dtype.

    Outputs:
       Tensor, has the same dtype as x.
    """
    return F.tensor_mul(x, y)


@mul.register("Tensor", "Number")
def _tensor_mul_scalar(x, y):
    """
    Returns x * y where x is a tensor and y is a scalar. x and y have same dtype.

    Outputs:
        Tensor, has the same dtype as x.
    """
    return F.tensor_mul(x, y)


@mul.register("List", "Number")
def _list_mul_scalar(x, y):
    """
    Returns x * y where x is a list and y is a number. y must be integer.

    Outputs:
        List.
    """
    return const_utils.sequence_mul_int(x, y)


@mul.register("Number", "List")
def _scalar_mul_list(x, y):
    """
    Returns x * y where x is a number and y is a list. x must be integer.

    Outputs:
        List.
    """
    return const_utils.sequence_mul_int(y, x)


@mul.register("Tuple", "Number")
def _tuple_mul_scalar(x, y):
    """
    Returns x * y where x is a tuple and y is a number. y must be integer.

    Outputs:
        Tuple.
    """
    return const_utils.sequence_mul_int(x, y)


@mul.register("Number", "Tuple")
def _scalar_mul_tuple(x, y):
    """
    Returns x * y where x is a number and y is a tuple. x must be integer.

    Outputs:
        Tuple.
    """
    return const_utils.sequence_mul_int(y, x)


@mul.register("Tensor", "Tuple")
def _tensor_mul_tuple(x, y):
    """
    Returns x * y where x is a tensor and y is a tuple.

    Args:
        x (Tensor): x
        y (Tuple): The dtype is same as x.

    Returns:
        Tensor, has the same dtype as x.
    """
    y = utils.sequence_to_tensor(y, x.dtype)
    return F.tensor_mul(x, y)


@mul.register("Tuple", "Tensor")
def _tuple_mul_tensor(x, y):
    """
    Returns x * y where x is a tuple and y is a tensor.

    Args:
        x (Tuple): x
        y (Tensor): The dtype is same as x.

    Returns:
        Tensor, has the same dtype as x.
    """
    x = utils.sequence_to_tensor(x, y.dtype)
    return F.tensor_mul(x, y)


@mul.register("Tensor", "List")
def _tensor_mul_list(x, y):
    """
    Returns x * y where x is a tensor and y is a list.

    Args:
        x (Tensor): x
        y (List): The dtype is same as x.

    Returns:
        Tensor, has the same dtype as x.
    """
    y = utils.sequence_to_tensor(y, x.dtype)
    return F.tensor_mul(x, y)


@mul.register("List", "Tensor")
def _list_mul_tensor(x, y):
    """
    Returns x * y where x is a list and y is a tensor.

    Args:
        x (List): x
        y (Tensor): The dtype is same as x.

    Returns:
        Tensor, has the same dtype as x.
    """
    x = utils.sequence_to_tensor(x, y.dtype)
    return F.tensor_mul(x, y)
